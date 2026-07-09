"""NT-Xent loss (Normalized Temperature-scaled Cross Entropy).

Implementation follows SimCLR (Chen et al., 2020).
Given a batch of 2N projected embeddings (N pairs of augmented views),
compute the contrastive loss:
  L_i,j = -log( exp(sim(z_i,z_j)/tau) / sum_{k≠i} exp(sim(z_i,z_k)/tau) )
where sim(a,b) = a^T b / (||a|| ||b||) (cosine similarity).

Paper setting: tau = 0.1.
"""

import torch
import torch.nn.functional as F


class NTXentLoss(torch.nn.Module):
    """NT-Xent loss.

    Args:
        temperature: Scale factor tau for logits (paper default: 0.1).
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute NT-Xent loss.

        Args:
            z1: First augmented views, shape (N, D).
            z2: Second augmented views, shape (N, D).

        Returns:
            Scalar loss.
        """
        N = z1.shape[0]
        # Concatenate: (2N, D)
        z = torch.cat([z1, z2], dim=0)
        # Normalize
        z = F.normalize(z, dim=1)
        # Cosine similarity matrix (2N, 2N)
        sim = z @ z.T / self.temperature

        # Mask out self-pairs
        mask = torch.eye(2 * N, device=z.device, dtype=torch.bool)
        sim = sim[~mask].view(2 * N, -1)

        # Positive pairs: (i, i+N) and (i+N, i)
        targets = torch.zeros(2 * N, device=z.device, dtype=torch.long)
        targets[:N] = torch.arange(N, device=z.device) + N
        targets[N:] = torch.arange(N, device=z.device)

        loss = F.cross_entropy(sim, targets)
        return loss
