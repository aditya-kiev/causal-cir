"""Multi-seed runner for Table 1.

Runs the five Table 1 benchmarks (3 synthetic SCMs + Colored MNIST +
Waterbirds) across methods and seeds, then writes three CSVs:

- results_by_dataset.csv            one row per (method, dataset, seed): raw
                                    last-epoch per-seed numbers, nothing lost
- results_by_dataset_summary.csv    one row per (method, dataset): mean/std
                                    over seeds (appendix per-dataset breakdown)
- results_table1_avg.csv            one row per method: unweighted mean over
                                    per-dataset means (main-paper Table 1)

Table 1 average formula (results_table1_avg.csv)
------------------------------------------------
For each metric m and each method:

  1. Per-dataset statistics (within each dataset d):
        mu_{m,d}  = mean over seeds of m on dataset d
        sig_{m,d} = population std (ddof=0) over seeds of m on dataset d

  2. Across-dataset aggregation (Table 1 columns):
        {m}_mean                = unweighted mean over datasets d of mu_{m,d}
        {m}_std_across_datasets = population std (ddof=0) over {mu_{m,d}}
        {m}_std_within_datasets = unweighted mean over d of {sig_{m,d}}
        {m}_n_datasets          = number of datasets d reporting m

Raw per-seed numbers are NEVER averaged directly across datasets; each
benchmark contributes exactly one per-dataset mean, so no dataset with more
seeds/samples can implicitly dominate.

Metric availability by dataset (diagnostics in train.py):
  acs                  -> SCM benchmarks only (3 datasets)
  itg_drop / acc_corr / acc_uncorr -> Colored MNIST + Waterbirds (2 datasets)
  train_loss / eval_loss           -> all 5 datasets

For the paper: main Table 1 reports {m}_mean +/- {m}_std_across_datasets
(mean +/- std across benchmarks; n=3 for ACS, n=2 for ITG/ID Acc/OOD Acc).
{std_within_datasets} is kept for diagnostics only; the per-dataset appendix
table carries the seed-level std.

Usage:
    python experiments/run_table1.py                                # all 5 benchmarks
    python experiments/run_table1.py --dataset colored_mnist        # single benchmark
    python experiments/run_table1.py --dataset scm_independent scm_causal_chain
"""

import sys
import json
import csv
from pathlib import Path
from dataclasses import asdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.configs import TrainConfig, get_config
from experiments.train import main as train_main, find_latest_ckpt, _collect_repro_meta

# Table 1 methods — confirm these match build_model_and_loss() in train.py
METHODS = ["simclr", "simclr_marginal", "barlow_twins", "vicreg", "cir"]

SEEDS = [0, 1, 2]

# The five Table 1 benchmarks: label -> base config (method is overridden below).
_DATASET_CONFIGS = {
    "scm_independent": "config_cir_scm_a",
    "scm_causal_chain": "config_cir_scm_b",
    "scm_confounded": "config_cir_scm_c",
    "colored_mnist": "config_cir_cmnist",
    "waterbirds": "config_cir_waterbirds",
}

# Headline metrics reported in the main-paper Table 1 (order = CSV column order).
TABLE1_METRICS = ["acs", "itg_drop", "acc_corr", "acc_uncorr"]

# All metrics aggregated (headline + losses). Column order for the avg CSV.
ALL_METRICS = ["acs", "itg_drop", "acc_corr", "acc_uncorr",
               "train_loss", "eval_loss"]


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


def _aggregate_across_datasets(dataset_summaries: list, keys: list) -> dict:
    """Unweighted average of per-dataset means for each metric key.

    Args:
        dataset_summaries: List of per-dataset aggregation dicts (as returned
            by _aggregate_seed_metrics), one per dataset that ran successfully.
        keys: Metric names to aggregate.

    Returns:
        dict with, per key: "{key}_mean", "{key}_std_across_datasets",
        "{key}_std_within_datasets", "{key}_n_datasets".
    """
    result = {}
    for key in keys:
        means = [s[f"{key}_mean"] for s in dataset_summaries
                 if s.get(f"{key}_mean") is not None]
        stds = [s[f"{key}_std"] for s in dataset_summaries
                if s.get(f"{key}_mean") is not None]
        if means:
            result[f"{key}_mean"] = float(np.mean(means))
            result[f"{key}_std_across_datasets"] = (
                float(np.std(means)) if len(means) > 1 else 0.0
            )
            result[f"{key}_std_within_datasets"] = float(np.mean(stds))
            result[f"{key}_n_datasets"] = len(means)
        else:
            result[f"{key}_mean"] = None
            result[f"{key}_std_across_datasets"] = None
            result[f"{key}_std_within_datasets"] = None
            result[f"{key}_n_datasets"] = 0
    return result


def _scalarize(metrics: dict) -> dict:
    """Flatten metrics.json to last-epoch scalars (skip meta/nested dicts)."""
    flat = {}
    for k, v in metrics.items():
        if k == "meta" or isinstance(v, dict):
            continue
        if isinstance(v, list):
            flat[k] = v[-1] if v else None
        else:
            flat[k] = v
    return flat


def run_table1(
    datasets: list = None,
    output_dir: str = "./outputs/table1",
    ckpt_root: str = None,
):
    """Run Table 1 comparison across methods, seeds, and benchmarks.

    Args:
        datasets: Subset of benchmark labels to run. None = all five.
        output_dir: Root output directory.
        ckpt_root: Checkpoint root dir (Drive on Colab). Defaults to cfg.ckpt_root.
    """
    if datasets is None:
        datasets = list(_DATASET_CONFIGS.keys())
    datasets = [d for d in datasets if d in _DATASET_CONFIGS]

    per_seed = []          # raw rows: (method, dataset, seed)
    per_dataset_rows = []  # appendix rows: (method, dataset)
    avg_rows = []          # Table 1 rows: (method)
    cfg = None

    for method in METHODS:
        for dataset in datasets:
            base_config_name = _DATASET_CONFIGS[dataset]
            seed_metrics_list = []

            for seed in SEEDS:
                cfg = TrainConfig(**asdict(get_config(base_config_name)))
                cfg.method = method
                cfg.seed = seed
                cfg.num_seeds = 1
                cfg.run_name = f"{method}_{dataset}_seed={seed}"
                cfg.output_dir = output_dir
                if ckpt_root is not None:
                    cfg.ckpt_root = ckpt_root

                print(f"\n{'='*60}")
                print(f"Method: {method}, dataset={dataset}, seed={seed}")
                print(f"{'='*60}")

                # Auto-resume: if a checkpoint exists for (method, dataset, seed),
                # resume from it instead of restarting. train_main() also skips the
                # run entirely if the checkpoint already reached cfg.epochs.
                if find_latest_ckpt(cfg, Path(output_dir) / cfg.run_name):
                    cfg.resume = True
                    print(f"Found existing checkpoint for {cfg.run_name} -> resuming")

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
                        m = _scalarize(json.load(f))
                else:
                    m = {}

                per_seed.append({"method": method, "dataset": dataset,
                                 "seed": seed, **m})
                seed_metrics_list.append(m)

            # Per-dataset (appendix) aggregation: mean/std over seeds.
            row = {"method": method, "dataset": dataset,
                   "config": base_config_name, "n_seeds": len(SEEDS)}
            row.update(_aggregate_seed_metrics(seed_metrics_list, ALL_METRICS))
            per_dataset_rows.append(row)

        # Table 1 (main-paper) aggregation: unweighted mean over per-dataset means.
        dataset_summaries = [
            r for r in per_dataset_rows
            if r["method"] == method
        ]
        avg_row = {"method": method, "n_seeds": len(SEEDS)}
        avg_row.update(_aggregate_across_datasets(dataset_summaries, ALL_METRICS))
        avg_rows.append(avg_row)

        # Incremental CSV writes so partial results survive long runs.
        _write_per_seed(per_seed, output_dir)
        _write_dataset_summary(per_dataset_rows, output_dir)
        _write_table1_avg(avg_rows, output_dir)

    # Consolidated reproducibility output
    consolidated = {
        "meta": _collect_repro_meta(cfg),
        "seeds": SEEDS,
        "datasets": datasets,
        "table1_avg": avg_rows,
        "by_dataset": per_dataset_rows,
        "per_seed": per_seed,
    }
    json_path = Path(output_dir) / "table1_results.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(consolidated, f, indent=2, default=str)
    print(f"Consolidated results written to {json_path}")

    _print_formula_summary()
    _print_table1(avg_rows)

    return avg_rows


def _write_per_seed(rows: list, output_dir: str):
    out_path = Path(output_dir) / "results_by_dataset.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "dataset", "seed"] + ALL_METRICS
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _write_dataset_summary(rows: list, output_dir: str):
    out_path = Path(output_dir) / "results_by_dataset_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (["method", "dataset", "config", "n_seeds"]
                  + [f"{m}_mean" for m in ALL_METRICS]
                  + [f"{m}_std" for m in ALL_METRICS])
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _write_table1_avg(rows: list, output_dir: str):
    out_path = Path(output_dir) / "results_table1_avg.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (["method", "n_seeds"]
                  + [f"{m}_mean" for m in ALL_METRICS]
                  + [f"{m}_std_across_datasets" for m in ALL_METRICS]
                  + [f"{m}_std_within_datasets" for m in ALL_METRICS]
                  + [f"{m}_n_datasets" for m in ALL_METRICS])
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _print_formula_summary():
    print("\n" + "=" * 80)
    print("Table 1 average formula (results_table1_avg.csv)")
    print("=" * 80)
    print("For each metric m and each method:")
    print("  Per-dataset stats (within each dataset d):")
    print("    mu_{m,d}  = mean over seeds of m on dataset d")
    print("    sig_{m,d} = population std (ddof=0) over seeds")
    print("  Table 1 columns:")
    print("    {m}_mean                = unweighted mean over datasets d of mu_{m,d}")
    print("    {m}_std_across_datasets = population std (ddof=0) over mu_{m,d}")
    print("    {m}_std_within_datasets = unweighted mean over d of sig_{m,d}")
    print("    {m}_n_datasets          = number of datasets d reporting m")
    print("Raw per-seed numbers are never averaged directly across datasets;")
    print("each benchmark contributes exactly one per-dataset mean.")
    print("Metric availability: acs -> SCMs (3); itg_drop/acc_corr/acc_uncorr ->")
    print("CMNIST+Waterbirds (2); train_loss/eval_loss -> all 5.")
    print("Main-paper Table 1 reports {m}_mean +/- {m}_std_across_datasets")
    print("(n=3 for ACS, n=2 for ITG/ID Acc/OOD Acc).")
    print("=" * 80)


def _print_table1(avg_rows: list):
    print("\nTable 1 (mean +/- std across benchmarks):")
    header = (f"{'Method':<18} {'ACS':>14} {'ITG Drop %':>14} "
              f"{'ID Acc':>12} {'OOD Acc':>12}")
    print(header)
    print("-" * len(header))
    for row in avg_rows:
        cells = []
        for m in ["acs", "itg_drop", "acc_corr", "acc_uncorr"]:
            mean = row.get(f"{m}_mean")
            std = row.get(f"{m}_std_across_datasets")
            if mean is None:
                cells.append("N/A".rjust(12))
            else:
                cells.append(f"{mean:.4f}±{std:.4f}".rjust(14))
        print(f"{row['method']:<18} " + " ".join(cells))
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Multi-seed runner for Table 1 (5 benchmarks).")
    parser.add_argument("--dataset", type=str, nargs="*", default=None,
                        choices=list(_DATASET_CONFIGS.keys()),
                        help="Benchmark labels to run. Default: all five.")
    parser.add_argument("--output_dir", type=str, default="./outputs/table1")
    parser.add_argument("--ckpt-root", type=str, default=None,
                        help="Checkpoint root dir (Drive on Colab)")
    args = parser.parse_args()

    avg_rows = run_table1(
        datasets=args.dataset,
        output_dir=args.output_dir,
        ckpt_root=args.ckpt_root,
    )
