"""Barlow Twins (Zbontar et al., 2021).

Loss = sum_i (1 - C_ii)^2 + lambda * sum_i sum_{j != i} C_ij^2
where C is the cross-correlation matrix between the two views' embeddings.

This encourages the cross-correlation matrix to be the identity.
"""

import torch
import torch.nn as nn


class BarlowTwins(nn.Module):
    """Barlow Twins loss.

    Args:
        lambd: Off-diagonal penalty weight.
    """

    def __init__(self, lambd: float = 5e-3):
        super().__init__()
        self.lambd = lambd

    def forward(self, z1, z2, h1=None, h2=None):
        batch_size = z1.shape[0]
        d = z1.shape[1]

        # Normalize along batch dimension
        z1_norm = (z1 - z1.mean(dim=0)) / z1.std(dim=0, unbiased=False)
        z2_norm = (z2 - z2.mean(dim=0)) / z2.std(dim=0, unbiased=False)

        # Cross-correlation matrix
        C = (z1_norm.T @ z2_norm) / batch_size  # (d, d)

        # Loss
        on_diag = (C.diag() - 1).pow(2).sum()
        off_diag = C.flatten()[~torch.eye(d, dtype=torch.bool, device=C.device)].pow(2).sum()

        loss = on_diag + self.lambd * off_diag
        return loss, {
            "bt_on_diag": on_diag.item(),
            "bt_off_diag": off_diag.item(),
            "total_loss": loss.item(),
        }
