"""Backbone factory: ResNet-18, ResNet-50, tiny ViT.

Each backbone outputs a 1D feature vector (before the classification head).
"""

import torch.nn as nn
from torchvision.models import (
    resnet18, resnet50,
    ResNet18_Weights, ResNet50_Weights,
)


class _TinyViT(nn.Module):
    """Minimal convnet placeholder for small-scale experiments (e.g. Colored MNIST).

    TODO: replace with a proper lightweight ViT (e.g. timm ViT-tiny).
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self._net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self._out_dim = 128

    def forward(self, x):
        return self._net(x)

    @property
    def out_dim(self):
        return self._out_dim


def get_backbone(
    name: str,
    in_channels: int = 3,
    img_size: int = 32,
    pretrained: bool = False,
) -> tuple:
    """Factory: returns (backbone, out_dim)."""
    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, dim
    elif name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, dim
    elif name == "tiny_vit":
        vit = _TinyViT(in_channels=in_channels)
        return vit, vit.out_dim
    else:
        raise ValueError(f"Unknown backbone '{name}'")
