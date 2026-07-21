"""Unit tests for the NT-Xent loss.

Tests:
  1. Random inputs: no crash, finite scalar loss.
  2. Identical views (z1 == z2): loss near theoretical minimum.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from losses.ntxent import NTXentLoss


def test_ntxent_random_inputs():
    """Random z1, z2 should produce a finite scalar with no IndexError."""
    N, D = 4, 8
    z1 = torch.randn(N, D)
    z2 = torch.randn(N, D)

    loss_fn = NTXentLoss(temperature=0.1)
    loss = loss_fn(z1, z2)

    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    assert loss.ndim == 0, f"Loss is not scalar: shape {loss.shape}"
    print(f"  Random inputs: loss = {loss.item():.6f}  PASSED\n")


def test_ntxent_identical_views():
    """z1 == z2 exactly should give loss close to 0 (perfect matching)."""
    N, D = 4, 8
    z1 = torch.randn(N, D)
    z2 = z1.clone()  # identical views

    loss_fn = NTXentLoss(temperature=0.1)
    loss = loss_fn(z1, z2)

    # For identical vectors, cosine similarity of positive pairs = 1.0,
    # so the loss should be well below 1.0 for D=8, tau=0.1.
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    assert loss.item() < 1.0, (
        f"Loss too high for identical views: {loss.item():.6f} >= 1.0"
    )
    print(f"  Identical views: loss = {loss.item():.6f}  PASSED\n")


if __name__ == "__main__":
    test_ntxent_random_inputs()
    test_ntxent_identical_views()
    print("All NT-Xent tests passed!")
