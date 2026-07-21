"""Paired significance testing: SimCLR vs SimCLR+CIR.

Usage:
    # 3 seeds (default)
    python experiments/significance_test.py --output_dir ./outputs

    # 5 seeds
    python experiments/significance_test.py --output_dir ./outputs --seeds 0 1 2 3 4
"""

import json
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel


def load_metric(output_dir: str, method: str, seed: int, key: str, default=None):
    """Load the last value of a metric from a run's metrics.json."""
    path = Path(output_dir) / f"{method}_seed={seed}" / "metrics.json"
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"Missing metrics file: {path}")
    with open(path) as f:
        metrics = json.load(f)
    vals = metrics.get(key, [])
    if len(vals) == 0:
        if default is not None:
            return default
        raise ValueError(f"Key '{key}' is empty in {path}")
    return vals[-1]


def run_significance_test(output_dir: str, seeds: list[int], alpha: float = 0.05):
    """Run paired t-tests for OOD Acc and ITG drop (SimCLR vs SimCLR+CIR)."""
    methods = {
        "SimCLR": "simclr",
        "SimCLR+CIR": "cir",
    }
    metrics = {"OOD Acc": "acc_uncorr", "ITG Drop (%)": "itg_drop"}

    results = {}
    for metric_name, metric_key in metrics.items():
        pairs = {label: [] for label in methods}
        for label, method in methods.items():
            for seed in seeds:
                try:
                    val = load_metric(output_dir, method, seed, metric_key)
                    pairs[label].append(val)
                except (FileNotFoundError, ValueError):
                    print(f"  Warning: {method} seed={seed} missing {metric_key} — skipping")
                    continue

        a, b = list(pairs.values())
        if len(a) != len(b) or len(a) < 2:
            print(f"  Skipping {metric_name}: insufficient paired data")
            continue

        mean_a, std_a = float(np.mean(a)), float(np.std(a, ddof=1))
        mean_b, std_b = float(np.mean(b)), float(np.std(b, ddof=1))
        t_stat, p_val = ttest_rel(a, b)
        sig = "significant" if p_val < alpha else "not significant"

        results[metric_name] = {
            "simclr_mean_std": f"{mean_a:.4f} +/- {std_a:.4f}",
            "cir_mean_std": f"{mean_b:.4f} +/- {std_b:.4f}",
            "t_statistic": float(f"{t_stat:.4f}"),
            "p_value": float(f"{p_val:.4f}"),
            "significant": sig,
        }

    # Also compute Cohen's d if both metrics available
    if "OOD Acc" in results and "ITG Drop (%)" in results:
        results["_note"] = (
            "Paired t-test (ttest_rel) used because same seeds pair runs. "
            f"alpha = {alpha}, n_seeds = {len(seeds)}"
        )

    return results


def print_results(results: dict):
    """Pretty-print the test results."""
    print("\n" + "=" * 70)
    print("Significance Test: SimCLR vs SimCLR+CIR")
    print("=" * 70)
    for metric, res in results.items():
        if metric.startswith("_"):
            continue
        print(f"\n{metric}:")
        print(f"  SimCLR:      {res['simclr_mean_std']}")
        print(f"  SimCLR+CIR:  {res['cir_mean_std']}")
        print(f"  t({res.get('_df', '?')})  = {res['t_statistic']}, ", end="")
        print(f"p = {res['p_value']}  [{res['significant']}]")
    if "_note" in results:
        print(f"\nNote: {results['_note']}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    results = run_significance_test(args.output_dir, args.seeds, args.alpha)
    print_results(results)

    out_path = Path(args.output_dir) / "significance_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")
