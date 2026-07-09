"""Unit tests for the RFF approximation of HSIC.

Tests:
  1. RFF converges to exact HSIC as D increases.
  2. RFF correctly distinguishes independent vs. dependent coordinates.
  3. RFF is symmetric in its inputs.
"""

import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from losses.hsic import pairwise_hsic_exact, pairwise_hsic_rff


def test_rff_convergence():
    """As D grows, RFF HSIC should approach exact HSIC."""
    n, p = 64, 3
    gen = torch.Generator().manual_seed(42)
    # Correlated data
    z = torch.randn(n, 1, generator=gen)
    eps = 0.3 * torch.randn(n, p, generator=gen)
    R = z + eps

    sigma = 1.0
    exact_val = pairwise_hsic_exact(R, sigma=sigma).item()

    results = {}
    for D in [10, 50, 100, 200, 500, 1000]:
        rff_val = pairwise_hsic_rff(R, sigma=sigma, n_features=D).item()
        results[D] = rff_val
        rel_err = abs(rff_val - exact_val) / max(abs(exact_val), 1e-8)
        print(f"  D={D:4d}  RFF={rff_val:.6f}  Exact={exact_val:.6f}  RelErr={rel_err:.4f}")

    # Check that with large D, RFF is within 20% of exact
    final_err = abs(results[1000] - exact_val) / max(abs(exact_val), 1e-8)
    assert final_err < 0.2, (
        f"RFF at D=1000 not close to exact: rel_err={final_err:.4f}"
    )
    print("  PASSED: RFF converges to exact HSIC with increasing D\n")


def test_rff_detects_dependence():
    """RFF should report larger values for dependent vs independent columns."""
    n, p = 256, 4
    sigma = 1.0

    # Independent
    R_indep = torch.randn(n, p)
    # Dependent
    z = torch.randn(n, 1)
    R_dep = z + 0.1 * torch.randn(n, p)

    for D in [50, 200]:
        v_indep = pairwise_hsic_rff(R_indep, sigma=sigma, n_features=D).item()
        v_dep = pairwise_hsic_rff(R_dep, sigma=sigma, n_features=D).item()
        print(f"  D={D}: HSIC(indep)={v_indep:.6f}  HSIC(dep)={v_dep:.6f}")
        assert v_dep > v_indep, f"RFF failed to detect dependence at D={D}"
    print("  PASSED: RFF detects dependence\n")


def test_rff_symmetry():
    """RFF HSIC should be the same for (R,R) regardless of column order."""
    n, p = 64, 4
    sigma = 1.0
    R = torch.randn(n, p)

    val1 = pairwise_hsic_rff(R, sigma=sigma, n_features=100)
    val2 = pairwise_hsic_rff(R[:, [2, 0, 3, 1]], sigma=sigma, n_features=100)

    diff = abs(val1.item() - val2.item())
    print(f"  HSIC(original)={val1.item():.6f}  HSIC(shuffled)={val2.item():.6f}  diff={diff:.8f}")
    assert diff < 1e-6, f"RFF not symmetric under column permutation: diff={diff:.8f}"
    print("  PASSED: RFF is symmetric\n")


if __name__ == "__main__":
    test_rff_convergence()
    test_rff_detects_dependence()
    test_rff_symmetry()
    print("All RFF tests passed!")
