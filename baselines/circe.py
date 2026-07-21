"""CIRCE: Conditional Independence Regression-based Contrastive Estimation.

Reference (VERIFY this citation — original draft may have been inaccurate):
  - Pogodin et al., "Efficient Conditionally Invariant Representation Learning",
    ICLR 2023.

CIRCE estimates the conditional independence regularization by first learning
a conditional mean embedding, then measuring the HSIC of the residuals
(analogous to CIR but with a learned conditional mean instead of per-pair avg).

Implementation:
  1. Learn a conditional mean network f(z_i | aug) that predicts one view's
     embedding from the other view's embedding.
  2. Residuals: r_i = h_i - f(z_i|aug).
  3. Apply pairwise-coordinate HSIC (same as CIR) on stacked residuals.
  4. Unlike the CIR loss which uses a fixed per-pair mean, the conditional
     mean is trained jointly via an additional MSE term.

This is the method closest to CIR and must be compared head-to-head.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from losses.hsic import PairwiseHSICExact


class _ConditionalMeanMLP(nn.Module):
    """Small MLP that estimates E[z1 | z2] or E[z2 | z1]."""

    def __init__(self, dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)


class CIRCE(nn.Module):
    """CIRCE regularizer.

    Trains a conditional mean MLP jointly (via MSE) and penalizes
    residual dependence via pairwise-coordinate HSIC (same as CIR core).

    Args:
        hsic_lambda: Weight of HSIC regularization on residuals.
        hsic_sigma: RBF bandwidth for HSIC.
    """

    def __init__(self, dim: int = 128, hsic_lambda: float = 0.05, hsic_sigma: float = None):
        super().__init__()
        self.hsic_lambda = hsic_lambda
        self.hsic = PairwiseHSICExact(sigma=hsic_sigma)
        self.cond_mean = _ConditionalMeanMLP(dim)

    def forward(self, z1, z2, h1=None, h2=None):
        """CIRCE regularization.

        Returns (loss, metrics_dict). The contrastive loss must be added externally.
        """

        z1_pred = self.cond_mean(z2)
        z2_pred = self.cond_mean(z1)

        # Residuals
        r1 = z1 - z1_pred
        r2 = z2 - z2_pred
        R = torch.cat([r1, r2], dim=0)

        # MSE trains the conditional mean
        mse_loss = F.mse_loss(z1_pred, z1) + F.mse_loss(z2_pred, z2)

        # Pairwise-coordinate HSIC on residuals (same as CIR)
        hsic_val = self.hsic(R)

        total = mse_loss + self.hsic_lambda * hsic_val

        return total, {
            "circe_mse": mse_loss.item(),
            "circe_hsic": hsic_val.item(),
            "circe_loss": total.item(),
        }
