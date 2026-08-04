"""Colored MNIST with configurable spurious correlation strength.

Protocol (from Bahng et al., 2020; modified for contrastive learning):
  1. Load MNIST (digits 0-9) as PIL images (grayscale).
  2. Convert to 3-channel RGB: repeat the grayscale channel.
  3. Assign each digit a "default" RGB tint with probability p_corr.
     - With probability p_corr: use the class-default tint.
     - With probability (1-p_corr): use a random tint.
  4. Return the tinted image as a (3, 32, 32) float32 tensor in [0, 1].
  5. The *contrastive pipeline* (external) adds: RandomResizedCrop(32),
     RandomHorizontalFlip, ColorJitter, and ImageNet normalization.

This matches the spurious-correlation setup used in the CIR paper.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision import transforms as T


class ColoredMNIST(Dataset):
    """Colored MNIST with controllable spurious correlation.

    Returns (img_tensor, label, spurious_label) where:
      - img_tensor: (3, 32, 32) float32 tensor in [0,1], tinted
      - label: digit class (0-9)
      - spurious_label: the tint index (0-9)

    Args:
        root: Path to store MNIST data.
        split: 'train' (60000) or 'test' (10000).
        p_corr: Probability that a sample gets its class-default tint.
        seed: RNG seed for reproducibility.
        download: Whether to download MNIST if not found.
    """

    NUM_CLASSES = 10

    def __init__(
        self,
        root: str = "./data",
        split: str = "train",
        p_corr: float = 0.9,
        seed: int = 42,
        download: bool = True,
    ):
        self.p_corr = p_corr
        train_flag = split == "train"
        self._mnist = MNIST(root=root, train=train_flag, download=download)
        self._rng = np.random.RandomState(seed)
        self._base_transform = T.Compose([
            T.Resize(32),
            T.ToTensor(),  # converts PIL [0,255] -> tensor [0,1]
        ])
        # Precompute one RGB tint per digit
        self._tints = self._precompute_tints()
        # Cache of decoded+resized (but NOT tinted) grayscale tensors, built
        # lazily on first access. None of the decode/resize/ToTensor work is
        # random or epoch-dependent, so it only ever needs to happen once.
        # Shape (N, 1, 32, 32) float32 in [0, 1]; ~60k images for train split.
        self._gray_cache = None

    def _precompute_tints(self):
        """Generate a distinct RGB tint vector for digits 0-9."""
        tints = []
        for digit in range(10):
            hue = digit / 10.0
            # Simple hue-to-RGB mapping (no saturation/value variation)
            h = hue * 6.0
            i = int(h)
            f = h - i
            q = 1.0 - f
            t = 1.0 - (1.0 - f)
            if i % 6 == 0:
                r, g, b = 1.0, t, 0.0
            elif i % 6 == 1:
                r, g, b = q, 1.0, 0.0
            elif i % 6 == 2:
                r, g, b = 0.0, 1.0, t
            elif i % 6 == 3:
                r, g, b = 0.0, q, 1.0
            elif i % 6 == 4:
                r, g, b = t, 0.0, 1.0
            else:
                r, g, b = 1.0, 0.0, q
            tints.append([r, g, b])
        return torch.tensor(tints, dtype=torch.float32)  # (10, 3)

    def _tint_image(self, img_tensor, tint_rgb):
        """Tint a (3, 32, 32) tensor by multiplying channels by [r, g, b]."""
        return img_tensor * tint_rgb.view(3, 1, 1)

    def _build_gray_cache(self):
        """Decode + resize all MNIST images once into a single in-memory tensor.

        Returns:
            torch.Tensor of shape (N, 1, 32, 32), float32 in [0, 1].
        """
        n = len(self._mnist)
        cache = torch.empty((n, 1, 32, 32), dtype=torch.float32)
        for idx in range(n):
            img_pil, _ = self._mnist[idx]
            cache[idx] = self._base_transform(img_pil)
        return cache

    def __len__(self):
        return len(self._mnist)

    def __getitem__(self, idx):
        if self._gray_cache is None:
            self._gray_cache = self._build_gray_cache()
        label = self._mnist.targets[idx].item()
        # Resize + [0,1] tensor (1, 32, 32) — pulled from cache instead of re-decoding
        img_tensor = self._gray_cache[idx]  # (1, 32, 32)
        # Repeat to 3 channels
        img_rgb = img_tensor.repeat(3, 1, 1)  # (3, 32, 32)

        # Determine tint
        if self._rng.random() < self.p_corr:
            spurious_label = label  # class-default tint
        else:
            spurious_label = self._rng.randint(0, 10)

        tint_color = self._tints[spurious_label]
        tinted = self._tint_image(img_rgb, tint_color)

        return tinted, label, spurious_label
