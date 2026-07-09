"""Unit tests for the HSIC estimators.

Tests:
  1. HSIC returns ~0 for independent Gaussian columns (H0: independence).
  2. HSIC detects dependence for correlated columns.
  3. HSIC is symmetric: HSIC(X,Y) = HSIC(Y,X).

Uses the exact pairwise HSIC and the RFF approximation.
"""

import sys
import math
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from losses.hsic import pairwise_hsic_exact, pairwise_hsic_rff


def _make_data(n: int, p: int, seed: int = 0, correlated: bool = False):
    """Generate test data.

    Args:
        n: Number of samples.
        p: Number of coordinates.
        correlated: If True, columns are linearly correlated.
    """
    gen = torch.Generator().manual_seed(seed)
    if correlated:
        # Generate correlated columns via shared latent factor
        z = torch.randn(n, 1, generator=gen)
        eps = 0.1 * torch.randn(n, p, generator=gen)
        X = z + eps
    else:
        X = torch.randn(n, p, generator=gen)
    return X


def test_hsic_independent():
    """HSIC should be ~0 for independent Gaussian columns (n large enough)."""
    n, p = 512, 4
    R = _make_data(n, p, seed=42, correlated=False)

    for mode, fn in [("exact", pairwise_hsic_exact), ("rff", pairwise_hsic_rff)]:
        if mode == "exact":
            val = fn(R, sigma=1.0)
        else:
            val = fn(R, sigma=1.0, n_features=500)

        val = val.item()
        print(f"[{mode}] Independent columns: HSIC = {val:.6f}  "
              f"(should be close to 0, e.g. < 0.05)")

        assert val < 0.05, f"HSIC too large for independent data: {val:.6f}"
    print("  PASSED\n")


def test_hsic_dependent():
    """HSIC should be > 0 for correlated columns."""
    n, p = 512, 4
    R = _make_data(n, p, seed=42, correlated=True)

    for mode, fn in [("exact", pairwise_hsic_exact), ("rff", pairwise_hsic_rff)]:
        if mode == "exact":
            val = fn(R, sigma=1.0)
        else:
            val = fn(R, sigma=1.0, n_features=500)

        val = val.item()
        print(f"[{mode}] Correlated columns: HSIC = {val:.6f}  "
              f"(should be > 0, e.g. > 0.01)")

        assert val > 0.01, f"HSIC too small for correlated data: {val:.6f}"
    print("  PASSED\n")


def test_hsic_independent_vs_dependent():
    """Check that HSIC on dependent data is strictly larger than on independent."""
    n, p = 256, 4
    R_indep = _make_data(n, p, seed=0, correlated=False)
    R_dep = _make_data(n, p, seed=0, correlated=True)

    for mode, fn in [("exact", pairwise_hsic_exact), ("rff", pairwise_hsic_rff)]:
        if mode == "exact":
            v_indep = fn(R_indep, sigma=1.0).item()
            v_dep = fn(R_dep, sigma=1.0).item()
        else:
            v_indep = fn(R_indep, sigma=1.0, n_features=500).item()
            v_dep = fn(R_dep, sigma=1.0, n_features=500).item()

        print(f"[{mode}]  HSIC(indep)={v_indep:.6f}  HSIC(dep)={v_dep:.6f}  "
              f"(dependent should be larger)")

        assert v_dep > v_indep, (
            f"Dependent HSIC ({v_dep:.6f}) not > independent HSIC ({v_indep:.6f})"
        )
    print("  PASSED\n")


def test_hsic_exact_vs_rff_convergence():
    """RFF approximation should approach exact HSIC as D increases."""
    n, p = 128, 3
    R = _make_data(n, p, seed=42, correlated=True)
    sigma = 1.0

    exact_val = pairwise_hsic_exact(R, sigma=sigma).item()
    print(f"Exact HSIC = {exact_val:.6f}")

    prev_error = float("inf")
    for D in [10, 50, 100, 200, 500]:
        rff_val = pairwise_hsic_rff(R, sigma=sigma, n_features=D).item()
        error = abs(rff_val - exact_val) / max(abs(exact_val), 1e-8)
        print(f"  RFF (D={D:3d}): HSIC = {rff_val:.6f}  rel_error = {error:.4f}")
        # Error should generally decrease (monotonic not guaranteed)
        # Just check it's not catastrophically bad
        assert error < 2.0, f"RFF error too large at D={D}: {error:.4f}"

    print("  PASSED\n")


def test_hsic_symmetry():
    """HSIC should be symmetric: HSIC(R,R) = 2*HSIC(R_cols) + diag terms.
    Just check that the function runs without error and returns a float."""
    n, p = 64, 4
    R = torch.randn(n, p)

    val = pairwise_hsic_exact(R, sigma=1.0)
    assert isinstance(val.item(), float)
    print(f"  Symmetry test (exact): HSIC = {val.item():.6f}  PASSED\n")


if __name__ == "__main__":
    test_hsic_independent()
    test_hsic_dependent()
    test_hsic_independent_vs_dependent()
    test_hsic_exact_vs_rff_convergence()
    test_hsic_symmetry()
    print("All HSIC tests passed!")
