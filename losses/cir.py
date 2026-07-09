"""CIR: Conditional Independence Regularization.

This implements the CORRECT definition from Section 5.2 of the paper:

  R_CIR(R) = sum_{j != k} HSIC(R[:,j], R[:,k])

where R is the (2N, p) matrix of stacked residuals from paired views.
This is a SUM OVER PAIRS OF COORDINATE-COLUMNS, NOT a single row-wise HSIC.

Algorithm per batch:
  1. Two augmented views X1, X2 -> encoder -> projection -> h1, h2.
  2. Per-pair mean: h_bar = (h1 + h2) / 2.
  3. Residuals: r1 = h1 - h_bar, r2 = h2 - h_bar.
  4. Stack: R = [r1; r2] of shape (2N, p).
  5. R_CIR = sum_{j != k} HSIC(R[:,j], R[:,k]).

Total loss = L_NT-Xent + lambda * R_CIR.
"""

import torch
from typing import Optional
from .hsic import PairwiseHSICExact, PairwiseHSICRFF


class CIRLoss(torch.nn.Module):
    """CIR regularizer: sum_{j != k} HSIC on coordinate-columns of stacked residuals.

    Args:
        hsic_mode: 'exact' (biased V-stat) or 'rff' (Random Fourier Features).
        hsic_sigma: RBF kernel bandwidth. None = median heuristic.
        rff_dim: Number of random features (only used when hsic_mode='rff').
    """

    def __init__(
        self,
        hsic_mode: str = "rff",
        hsic_sigma: Optional[float] = None,
        rff_dim: int = 128,
    ):
        super().__init__()
        self.hsic_mode = hsic_mode
        if hsic_mode == "exact":
            self._hsic_fn = PairwiseHSICExact(sigma=hsic_sigma)
        elif hsic_mode == "rff":
            self._hsic_fn = PairwiseHSICRFF(
                sigma=hsic_sigma if hsic_sigma is not None else None,
                n_features=rff_dim,
            )
        else:
            raise ValueError(f"Unknown hsic_mode '{hsic_mode}'. Choose 'exact' or 'rff'.")

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        """Compute R_CIR regularizer.

        Args:
            h1: Projected embeddings from first view, (N, p).
            h2: Projected embeddings from second view, (N, p).

        Returns:
            Scalar CIR penalty.
        """
        h_bar = (h1 + h2) / 2.0
        r1 = h1 - h_bar
        r2 = h2 - h_bar
        R = torch.cat([r1, r2], dim=0)  # (2N, p)
        return self._hsic_fn(R)
