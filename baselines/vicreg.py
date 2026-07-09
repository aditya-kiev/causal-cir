"""VICReg (Bardes et al., 2022).

Loss = variance_term + invariance_term + covariance_term
  - Variance: hinge loss on std of each dimension (per view)
  - Invariance: MSE between views
  - Covariance: sum of squared off-diagonal elements of covariance (per view)

VICReg does not require negative pairs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VICReg(nn.Module):
    """VICReg loss.

    Args:
        sim_coeff: Weight of invariance term.
        std_coeff: Weight of variance term.
        cov_coeff: Weight of covariance term.
    """

    def __init__(self, sim_coeff: float = 25.0, std_coeff: float = 25.0, cov_coeff: float = 1.0):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def _compute_variance(self, z: torch.Tensor) -> torch.Tensor:
        """Variance regularization: hinge loss on std."""
        std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
        return F.relu(1.0 - std).mean()

    def _compute_covariance(self, z: torch.Tensor) -> torch.Tensor:
        """Covariance regularization: sum of squared off-diagonal cov elements."""
        batch_size, d = z.shape
        z_centered = z - z.mean(dim=0)
        cov = (z_centered.T @ z_centered) / batch_size  # (d, d)
        off_diag = cov.flatten()[~torch.eye(d, dtype=torch.bool, device=z.device)].pow(2).sum()
        return off_diag / d

    def forward(self, z1, z2, h1=None, h2=None):
        # Invariance: MSE between the two views
        inv_loss = F.mse_loss(z1, z2)

        # Variance (per view)
        var_loss = (self._compute_variance(z1) + self._compute_variance(z2)) / 2.0

        # Covariance (per view)
        cov_loss = (self._compute_covariance(z1) + self._compute_covariance(z2)) / 2.0

        loss = self.sim_coeff * inv_loss + self.std_coeff * var_loss + self.cov_coeff * cov_loss
        return loss, {
            "vicreg_inv": inv_loss.item(),
            "vicreg_var": var_loss.item(),
            "vicreg_cov": cov_loss.item(),
            "total_loss": loss.item(),
        }
