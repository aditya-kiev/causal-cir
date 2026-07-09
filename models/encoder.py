"""Encoder + projection head, following SimCLR (Chen et al., 2020).

Architecture:
  Backbone (e.g. ResNet-18) -> h (representation)
  MLP projection head (2-3 layers) -> z (projected embedding for contrastive loss)

CIR is applied to h (encoder output) or z (projection head output), configurable.
"""

import torch.nn as nn
from .backbones import get_backbone


class ProjectionHead(nn.Module):
    """MLP projection head. Default: hidden 2048 -> out 128 (paper: p=128)."""

    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    """SimCLR-style encoder: backbone + projection head.

    Args:
        backbone_name: 'resnet18', 'resnet50', or 'tiny_vit'.
        proj_hidden_dim: Hidden dim of projection MLP.
        proj_out_dim: Output dim of projection MLP.
        in_channels: Input image channels.
        img_size: Input image size (for ViT).
        pretrained: Whether to use ImageNet-pretrained backbone.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        proj_hidden_dim: int = 2048,
        proj_out_dim: int = 128,
        in_channels: int = 3,
        img_size: int = 32,
        pretrained: bool = False,
    ):
        super().__init__()
        self.backbone, self._rep_dim = get_backbone(
            backbone_name,
            in_channels=in_channels,
            img_size=img_size,
            pretrained=pretrained,
        )
        self.projection = ProjectionHead(self._rep_dim, proj_hidden_dim, proj_out_dim)

    def forward(self, x):
        """Returns (h, z) where h is backbone output, z is projected embedding."""
        h = self.backbone(x)
        z = self.projection(h)
        return h, z

    @property
    def rep_dim(self):
        """Dimension of h (encoder output)."""
        return self._rep_dim
