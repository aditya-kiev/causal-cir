"""HSIC Baseline: Dependence regularization via cross-HSIC.

This baseline penalizes dependence between the representation Z and the
augmentation index A using cross-HSIC.

NOTE: Previously named "PIDReg". The original publication reference
could not be verified (Tsai et al., 2021 does not correspond to any
identifiable paper). Renamed to HSICBaseline to avoid incorrect attribution.

Implementation:
  1. Concatenate both views: Z = [z1; z2] of shape (2N, p).
  2. Create augmentation labels: A = one-hot(view_idx) of shape (2N, 2).
  3. Compute cross-HSIC: sum_{j,k} HSIC(Z[:,j], A[:,k]).
  4. Total loss = L_contrastive + lambda * HSIC(Z, A).
"""

import torch
import torch.nn as nn
from losses.hsic import CrossHSICExact


class HSICBaseline(nn.Module):
    """HSIC-based dependence regularizer.

    Args:
        hsic_lambda: Weight for the regularizer.
        hsic_sigma: RBF bandwidth.
    """

    def __init__(self, hsic_lambda: float = 0.05, hsic_sigma: float = None):
        super().__init__()
        self.hsic_lambda = hsic_lambda
        self.hsic = CrossHSICExact(sigma=hsic_sigma)

    def forward(self, z1, z2, h1=None, h2=None):
        N = z1.shape[0]
        Z = torch.cat([z1, z2], dim=0)  # (2N, p)
        A = torch.zeros(2 * N, 2, device=z1.device)
        A[:N, 0] = 1.0
        A[N:, 1] = 1.0

        hsic_val = self.hsic(Z, A)
        loss = self.hsic_lambda * hsic_val

        return loss, {
            "hsic_baseline": hsic_val.item(),
            "hsic_baseline_loss": loss.item(),
        }
