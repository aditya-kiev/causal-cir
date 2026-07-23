"""HSIC (Hilbert-Schmidt Independence Criterion) estimators.

Implements:
  1. Biased HSIC V-statistic: HSIC_b = 1/n^2 tr(K H L H)
  2. Pairwise-coordinate HSIC: sum_{j != k} HSIC(R[:,j], R[:,k])
     — the correct CIR penalty from Section 5.2 of the paper.
  3. RFF (Random Fourier Features) approximation for O(N) scaling.

References:
  - Gretton et al. (2005). "Measuring Statistical Dependence with HSIC"
  - Rahimi & Recht (2007). "Random Features for Large-Scale Kernel Machines"
  - Zhang et al. (2018). "Large-Scale Kernel Methods for Independence Testing"
"""

import torch
import numpy as np
from typing import Optional


def _centering_matrix(n: int, device: torch.device) -> torch.Tensor:
    """H = I - (1/n) 11^T."""
    return torch.eye(n, device=device) - (1.0 / n) * torch.ones(n, n, device=device)


def biased_hsic_vstat(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Biased HSIC V-statistic: 1/n^2 tr(K H L H)."""
    n = K.shape[0]
    H = _centering_matrix(n, K.device)
    return torch.trace(K @ H @ L @ H) / (n ** 2)


def _rbf_kernel_1d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """RBF kernel matrix for a 1D column vector x of shape (n,)."""
    diff = x.unsqueeze(1) - x.unsqueeze(0)
    K = torch.exp(-(diff ** 2) / (2.0 * sigma ** 2))
    return K


def _median_heuristic_1d(X: torch.Tensor) -> float:
    """Median heuristic from pairwise distances of all points.
    X: (n, d) tensor."""
    dists = torch.cdist(X, X, p=2)
    n = dists.shape[0]
    triu = dists[~torch.eye(n, dtype=torch.bool, device=dists.device)]
    med = triu.median().item()
    return max(med, 1.0)


# ---------------------------------------------------------------------------
# Per-coordinate RFF features
# ---------------------------------------------------------------------------

class _RFFEncoder:
    """Lazy-init RFF encoder that maps each coordinate column to D features."""

    def __init__(self, sigma: float, n_features: int, seed: int = 42):
        self.sigma = sigma
        self.n_features = n_features
        self.seed = seed
        self._W = None   # will be (1, D) per coordinate
        self._b = None   # (D,)
        self._fitted = False

    def _ensure_fitted(self, device: torch.device):
        if not self._fitted:
            gen = torch.Generator(device=device).manual_seed(self.seed)
            self._W = torch.randn(1, self.n_features, device=device, generator=gen) / self.sigma
            self._b = torch.rand(self.n_features, device=device, generator=gen) * 2 * np.pi
            self._fitted = True

    def embed(self, col: torch.Tensor) -> torch.Tensor:
        """Embed a single column vector (n,) -> (n, D)."""
        self._ensure_fitted(col.device)
        # col: (n,) -> (n, 1)
        x = col.unsqueeze(1)  # (n, 1)
        phi = np.sqrt(2.0 / self.n_features) * torch.cos(x @ self._W + self._b)
        return phi


# ---------------------------------------------------------------------------
# Pairwise-coordinate HSIC (the CIR penalty)
# ---------------------------------------------------------------------------

def pairwise_hsic_exact(R: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
    """Sum_{j != k} HSIC_b(R[:,j], R[:,k]) via exact biased V-statistic.

    Args:
        R: Residual matrix, shape (n, p) where n = 2*N, p = feature dim.
        sigma: RBF bandwidth. None = median heuristic over all entries.

    Returns:
        Scalar = sum_{j != k} HSIC(R[:,j], R[:,k]).
    """
    n, p = R.shape
    device = R.device

    if sigma is None:
        sigma = _median_heuristic_1d(R)

    # Precompute kernel matrices for all coordinates: (p, n, n)
    K_all = torch.zeros(p, n, n, device=device)
    for j in range(p):
        col = R[:, j]
        K_all[j] = _rbf_kernel_1d(col, sigma)

    H = _centering_matrix(n, device)
    K_centered = K_all @ H  # (p, n, n): each slice is K_j @ H

    # T[j,k] = tr(K_centered[j] @ K_centered[k])
    # tr(A @ B) = sum_{i,j} A_{ij} B_{ji} = einsum('jab,kba->jk', K_centered, K_centered)
    T = torch.einsum('jab,kba->jk', K_centered, K_centered)  # (p, p)

    # HSIC[j,k] = T[j,k] / n^2
    # sum_{j!=k} HSIC[j,k] = (sum_{j,k} T[j,k] - sum_j T[j,j]) / n^2
    total = T.sum() - T.trace()
    return total / (n ** 2)


# ---------------------------------------------------------------------------
# Normalized by 1/n² to match the exact estimator (see tests/test_hsic.py
# RFF-vs-exact convergence test). If you're cross-checking against the
# paper's Appendix A.3, note the paper text has an error here (states 1/n)
# that needs a separate fix in the paper source, not here.
# ---------------------------------------------------------------------------

def pairwise_hsic_rff(R: torch.Tensor,
                      sigma: Optional[float] = None,
                      n_features: int = 128) -> torch.Tensor:
    """Sum_{j != k} HSIC(R[:,j], R[:,k]) via RFF approximation.

    Uses the trace-trick to avoid the expensive (p*D, p*D) Gram matrix:
      Let Psi_j be (n, D) centered RFF features for coordinate j.
      G_j = Psi_j @ Psi_j^T of shape (n, n).
      Then:
        ||sum_j G_j||_F^2 = ||Psi @ Psi^T||_F^2 = ||Psi^T @ Psi||_F^2
        sum_j ||G_j||_F^2 = sum_j ||Psi_j^T @ Psi_j||_F^2  (diagonal blocks)
      Result = (||sum G_j||_F^2 - sum ||G_j||_F^2) / n^2

    This reduces the dominant cost from O(p²·D²) to O(p·n²), a large
    speedup when p >> n (typical setting: n=8-512, p=128).

    Args:
        R: Residual matrix, shape (n, p).
        sigma: RBF bandwidth. None = median heuristic over all entries.
        n_features: Number of random Fourier features per coordinate.

    Returns:
        Scalar = sum_{j != k} HSIC_RFF(R[:,j], R[:,k]).
    """
    n, p = R.shape
    device = R.device

    if sigma is None:
        sigma = _median_heuristic_1d(R)

    rff = _RFFEncoder(sigma=sigma, n_features=n_features)
    rff._ensure_fitted(device)
    H = _centering_matrix(n, device)

    # Batched RFF embedding: (p, n) -> (p, n, D) — one matmul, no Python loop
    scale = np.sqrt(2.0 / n_features)
    # R.T: (p, n); unsqueeze(-1): (p, n, 1); rff._W: (1, D)
    phi_all = scale * torch.cos(R.T.unsqueeze(-1) @ rff._W + rff._b)  # (p, n, D)

    # Centering: H @ phi_j for all j — one einsum, no Python loop
    psi_all = torch.einsum('ij,kjd->kid', H, phi_all)  # (p, n, D)

    # G_j = Psi_j @ Psi_j^T for all j — one batched matmul, no Python loop
    G_all = torch.bmm(psi_all, psi_all.transpose(1, 2))  # (p, n, n)

    # Full Frobenius norm: ||sum_j G_j||_F^2 = ||Psi @ Psi^T||_F^2
    G = G_all.sum(dim=0)  # (n, n)
    frob_sq_full = (G @ G).trace()

    # Diagonal block Frobenius norms: sum_j ||G_j||_F^2
    diag_frob_sq = G_all.pow(2).sum()

    return (frob_sq_full - diag_frob_sq) / (n ** 2)


# ---------------------------------------------------------------------------
# Cross-HSIC: sum_{j,k} HSIC(X[:,j], Y[:,k]) for two DIFFERENT matrices
# Needed by HSICBaseline (HSIC(Z, A)) and simclr_marginal (HSIC(z1, z1)).
# ---------------------------------------------------------------------------

def cross_hsic_exact(X: torch.Tensor, Y: torch.Tensor,
                     sigma: Optional[float] = None) -> torch.Tensor:
    """Exact sum_{j,k} HSIC(X[:,j], Y[:,k]) via biased V-statistic.

    Args:
        X: (n, d1) tensor.
        Y: (n, d2) tensor.
        sigma: RBF bandwidth. None = median heuristic.

    Returns:
        Scalar = sum_{j,k} HSIC(X_j, Y_k).
    """
    n = X.shape[0]
    device = X.device
    d1, d2 = X.shape[1], Y.shape[1]

    if sigma is None:
        Z = torch.cat([X, Y], dim=1)
        sigma = _median_heuristic_1d(Z)

    # Kernel matrices per column: (d1, n, n) and (d2, n, n)
    K_all = torch.zeros(d1, n, n, device=device)
    for j in range(d1):
        K_all[j] = _rbf_kernel_1d(X[:, j], sigma)

    L_all = torch.zeros(d2, n, n, device=device)
    for k in range(d2):
        L_all[k] = _rbf_kernel_1d(Y[:, k], sigma)

    H = _centering_matrix(n, device)
    K_centered = K_all @ H  # (d1, n, n)
    L_centered = L_all @ H  # (d2, n, n)

    # T[j,k] = tr(K_j H K_k H) = einsum('jab,kba->jk', K_centered, L_centered)
    T = torch.einsum('jab,kba->jk', K_centered, L_centered)  # (d1, d2)
    return T.sum() / (n ** 2)


def cross_hsic_rff(X: torch.Tensor, Y: torch.Tensor,
                   sigma: Optional[float] = None,
                   n_features: int = 128) -> torch.Tensor:
    """RFF-approximated sum_{j,k} HSIC(X[:,j], Y[:,k]).

    Uses the trace trick: ||Psi_x^T @ Psi_y||_F^2 = tr((Psi_x @ Psi_x^T) @ (Psi_y @ Psi_y^T)),
    avoiding the expensive (d1*D, d2*D) Gram matrix.
    """
    n = X.shape[0]
    device = X.device
    d1, d2 = X.shape[1], Y.shape[1]

    if sigma is None:
        Z = torch.cat([X, Y], dim=1)
        sigma = _median_heuristic_1d(Z)

    rff = _RFFEncoder(sigma=sigma, n_features=n_features)
    rff._ensure_fitted(device)
    H = _centering_matrix(n, device)
    scale = np.sqrt(2.0 / n_features)

    # Batched embedding for X — one matmul, no Python loop
    phi_x = scale * torch.cos(X.T.unsqueeze(-1) @ rff._W + rff._b)  # (d1, n, D)
    psi_x = torch.einsum('ij,kjd->kid', H, phi_x)  # (d1, n, D)
    G_x = torch.bmm(psi_x, psi_x.transpose(1, 2)).sum(dim=0)  # (n, n)

    # Batched embedding for Y — one matmul, no Python loop
    phi_y = scale * torch.cos(Y.T.unsqueeze(-1) @ rff._W + rff._b)  # (d2, n, D)
    psi_y = torch.einsum('ij,kjd->kid', H, phi_y)  # (d2, n, D)
    G_y = torch.bmm(psi_y, psi_y.transpose(1, 2)).sum(dim=0)  # (n, n)

    # tr(G_x @ G_y) = ||Psi_x^T @ Psi_y||_F^2
    return (G_x @ G_y).trace() / (n ** 2)


class CrossHSICExact(torch.nn.Module):
    """Exact sum_{j,k} HSIC(X[:,j], Y[:,k])."""

    def __init__(self, sigma: Optional[float] = None):
        super().__init__()
        self.sigma = sigma

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return cross_hsic_exact(X, Y, sigma=self.sigma)


class CrossHSICRFF(torch.nn.Module):
    """RFF-approximated sum_{j,k} HSIC(X[:,j], Y[:,k])."""

    def __init__(self, sigma: Optional[float] = None, n_features: int = 128):
        super().__init__()
        self.sigma = sigma
        self.n_features = n_features

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return cross_hsic_rff(X, Y, sigma=self.sigma, n_features=self.n_features)


# ---------------------------------------------------------------------------
# Convenience wrappers for the CIR loss module
# ---------------------------------------------------------------------------

class PairwiseHSICExact(torch.nn.Module):
    """Exact sum_{j != k} HSIC(R[:,j], R[:,k]) via biased V-statistic."""

    def __init__(self, sigma: Optional[float] = None):
        super().__init__()
        self.sigma = sigma

    def forward(self, R: torch.Tensor) -> torch.Tensor:
        return pairwise_hsic_exact(R, sigma=self.sigma)


class PairwiseHSICRFF(torch.nn.Module):
    """RFF-approximated sum_{j != k} HSIC(R[:,j], R[:,k])."""

    def __init__(self, sigma: Optional[float] = None, n_features: int = 128):
        super().__init__()
        self.sigma = sigma
        self.n_features = n_features

    def forward(self, R: torch.Tensor) -> torch.Tensor:
        return pairwise_hsic_rff(R, sigma=self.sigma, n_features=self.n_features)
