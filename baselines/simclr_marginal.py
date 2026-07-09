"""SimCLR + marginal HSIC regularization.

Instead of HSIC on residuals (CIR), apply HSIC separately to each view's
projected embeddings. This penalizes dependence among coordinates of a
single view's representation, which is a simpler alternative.

Reference: This is a baseline variant that separates the "on residuals" part
from the "HSIC regularization" part of CIR.
"""

import torch.nn as nn
from losses.ntxent import NTXentLoss
from losses.hsic import PairwiseHSICExact


class SimCLRMarginalHSIC(nn.Module):
    """SimCLR + HSIC on each marginal view (not on residuals)."""

    def __init__(self, temperature: float = 0.1, hsic_lambda: float = 0.05, sigma: float = None):
        super().__init__()
        self.contrastive = NTXentLoss(temperature=temperature)
        self.hsic_fn = PairwiseHSICExact(sigma=sigma)
        self.hsic_lambda = hsic_lambda

    def forward(self, z1, z2, h1=None, h2=None):
        c_loss = self.contrastive(z1, z2)
        # Pairwise HSIC on each view's columns (sum_{j!=k} HSIC)
        hsic_loss = (self.hsic_fn(z1) + self.hsic_fn(z2)) / 2.0
        total = c_loss + self.hsic_lambda * hsic_loss
        return total, {
            "contrastive_loss": c_loss.item(),
            "hsic_marginal_loss": hsic_loss.item(),
            "total_loss": total.item(),
        }
