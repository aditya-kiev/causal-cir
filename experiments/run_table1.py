"""Multi-seed runner for Table 1.

Loops over METHODS, SEEDS, and dataset configs, then writes aggregated
mean/std results to a CSV (same format as train_ablation.py's output).

Usage:
    python experiments/run_table1.py
    python experiments/run_table1.py --dataset colored_mnist

Command-line overrides make it easy to target specific subsets.
"""

import sys
import json
import csv
from pathlib import Path
from dataclasses import asdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.configs import TrainConfig, get_config
from experiments.train import main as train_main

# Table 1 methods — confirm these match build_model_and_loss() in train.py
METHODS = ["simclr", "simclr_marginal", "barlow_twins", "vicreg", "cir"]

SEEDS = [0, 1, 2]

# TODO: Table 1 is currently ambiguous on which dataset(s) it covers.
# It may be Colored MNIST only, Waterbirds only, or both.  Uncomment
# the relevant line below or pass --dataset on the command line once
# this is resolved (ask Aditya or check paper_fixes.md item 6).
_DATASET_CONFIGS = {
    "colored_mnist": "config_cir_cmnist",
    "waterbirds": "config_cir_waterbirds",
}


def _aggregate_seed_metrics(metrics_list: list, keys: list) -> dict:
    """Compute mean/std across seeds for each metric key.

    Args:
        metrics_list: List of dicts, one per seed.
        keys: Metric names to aggregate.

    Returns:
        dict of {key: {"mean": ..., "std": ...}}.
    """
    result = {}
    for key in keys:
        vals = [m.get(key) for m in metrics_list if m.get(key) is not None]
        if vals:
            result[f"{key}_mean"] = float(np.mean(vals))
            result[f"{key}_std"] = float(np.std(vals) if len(vals) > 1 else 0.0)
        else:
            result[f"{key}_mean"] = None
            result[f"{key}_std"] = None
    return result


def run_table1(
    dataset: str = "colored_mnist",
    config_suffix: str = None,
    output_dir: str = "./outputs/table1",
):
    """Run Table 1 comparison across methods and seeds.

    Args:
        dataset: Dataset name ("colored_mnist" or "waterbirds").
        config_suffix: Optional suffix for config names (e.g. "_scm_a").
        output_dir: Root output directory.
    """
    base_config_name = _DATASET_CONFIGS[dataset]
    results = []

    for method in METHODS:
        seed_metrics_list = []

        for seed in SEEDS:
            cfg = TrainConfig(**asdict(get_config(base_config_name)))
            cfg.method = method
            cfg.seed = seed
            cfg.num_seeds = 1
            cfg.run_name = f"{method}_{dataset}_seed={seed}"
            cfg.output_dir = output_dir

            print(f"\n{'='*60}")
            print(f"Method: {method}, seed={seed}, dataset={dataset}")
            print(f"{'='*60}")

            try:
                train_main(cfg)
            except Exception as e:
                print(f"Run failed: {e}")
                import traceback
                traceback.print_exc()
                continue

            metrics_path = Path(output_dir) / cfg.run_name / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics_list = [json.load(f)]
            else:
                metrics_list = []

            for m in metrics_list:
                seed_metrics_list.append(m)

        # Aggregate across seeds
        all_keys = ["train_loss", "eval_loss", "itg_drop", "acs",
                     "acc_corr", "acc_uncorr"]
        row = {"method": method, "dataset": dataset,
               "config": base_config_name, "n_seeds": len(SEEDS)}
        row.update(_aggregate_seed_metrics(seed_metrics_list, all_keys))
        results.append(row)

        # Write incremental output
        _write_csv(results, output_dir)

    return results


def _write_csv(results: list, output_dir: str):
    out_path = Path(output_dir) / "table1_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
    print(f"Results written to {out_path}")


def _print_table(results: list):
    print("\n")
    print("=" * 80)
    header = f"{'Method':<20} {'Train Loss':>12} {'ITG Drop %':>12} {'ACS':>12}"
    print(header)
    print("-" * 80)
    for row in results:
        tl = f"{row['train_loss_mean']:.4f}" if row.get('train_loss_mean') is not None else "N/A"
        itg = f"{row['itg_drop_mean']:.2f}" if row.get('itg_drop_mean') is not None else "N/A"
        acs = f"{row['acs_mean']:.4f}" if row.get('acs_mean') is not None else "N/A"
        print(f"{row['method']:<20} {tl:>12} {itg:>12} {acs:>12}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="colored_mnist",
                        choices=list(_DATASET_CONFIGS.keys()))
    parser.add_argument("--output_dir", type=str, default="./outputs/table1")
    args = parser.parse_args()

    results = run_table1(
        dataset=args.dataset,
        output_dir=args.output_dir,
    )
    _print_table(results)
