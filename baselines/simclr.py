"""Vanilla SimCLR baseline.

Total loss = L_NT-Xent (no additional regularization).
"""

import torch.nn as nn
from losses.ntxent import NTXentLoss


class SimCLR(nn.Module):
    """SimCLR with NT-Xent loss only."""

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.contrastive = NTXentLoss(temperature=temperature)

    def forward(self, z1, z2, h1=None, h2=None):
        """Returns (loss, metrics_dict)."""
        loss = self.contrastive(z1, z2)
        return loss, {"contrastive_loss": loss.item()}
