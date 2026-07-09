"""Ablation harness.

Sweeps over hyperparameters and reports mean/std over >=3 seeds.
Results are written to a single `results.csv`.

Usage:
    python experiments/train_ablation.py --sweep hsic_lambda --values 0.0,0.01,0.1,1.0
    python experiments/train_ablation.py --sweep hsic_mode --values exact,rff
    python experiments/train_ablation.py --sweep rff_dim --values 10,50,100,500
    python experiments/train_ablation.py --sweep batch_size --values 64,128,256
    python experiments/train_ablation.py --sweep cir_on --values encoder,projection
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


def run_sweep(
    base_config_name: str,
    sweep_param: str,
    sweep_values: list,
    output_dir: str = "./outputs/ablations",
    num_seeds: int = 3,
):
    """Run ablation sweep.

    Args:
        base_config_name: Name of base config from CONFIG_REGISTRY.
        sweep_param: Config attribute to sweep.
        sweep_values: List of values for sweep_param.
        output_dir: Root output directory.
        num_seeds: Number of seeds to average over.
    """
    base_cfg = get_config(base_config_name)
    results = []

    for val in sweep_values:
        seed_metrics = {"train_loss": [], "test_acc_corr": [], "test_acc_uncorr": [],
                        "itg_drop": [], "acs": []}

        for seed in range(num_seeds):
            # Build config for this run
            cfg = TrainConfig(**{k: v for k, v in asdict(base_cfg).items()})
            setattr(cfg, sweep_param, val)
            cfg.seed = seed
            cfg.num_seeds = 1  # we handle multi-seed at the sweep level
            cfg.run_name = f"{sweep_param}={val}_seed={seed}"
            cfg.output_dir = output_dir

            print(f"\n{'='*60}")
            print(f"Sweep: {sweep_param}={val}, seed={seed}")
            print(f"{'='*60}")

            # Run training
            try:
                train_main(cfg)
            except Exception as e:
                print(f"Run failed: {e}")
                continue

            # Load metrics
            metrics_path = Path(output_dir) / cfg.run_name / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
                seed_metrics["train_loss"].append(metrics["train_loss"][-1] if metrics["train_loss"] else None)
                if "itg_drop" in metrics and metrics["itg_drop"]:
                    seed_metrics["itg_drop"].append(metrics["itg_drop"][-1])
                if "acs" in metrics and metrics["acs"]:
                    seed_metrics["acs"].append(metrics["acs"][-1])

        # Aggregate across seeds
        row = {
            "sweep_param": sweep_param,
            "sweep_value": str(val),
            "n_seeds": num_seeds,
            "train_loss_mean": np.mean([v for v in seed_metrics["train_loss"] if v is not None]) if any(v is not None for v in seed_metrics["train_loss"]) else None,
            "train_loss_std": np.std([v for v in seed_metrics["train_loss"] if v is not None]) if any(v is not None for v in seed_metrics["train_loss"]) else None,
            "itg_drop_mean": np.mean(seed_metrics["itg_drop"]) if seed_metrics["itg_drop"] else None,
            "itg_drop_std": np.std(seed_metrics["itg_drop"]) if seed_metrics["itg_drop"] else None,
            "acs_mean": np.mean(seed_metrics["acs"]) if seed_metrics["acs"] else None,
            "acs_std": np.std(seed_metrics["acs"]) if seed_metrics["acs"] else None,
        }
        results.append(row)

        # Write incremental results
        _write_csv(results, output_dir, sweep_param)

    return results


def _write_csv(results: list, output_dir: str, sweep_param: str):
    out_path = Path(output_dir) / "results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_cir_cmnist")
    parser.add_argument("--sweep", type=str, required=True,
                        help="Config parameter to sweep")
    parser.add_argument("--values", type=str, required=True,
                        help="Comma-separated list of values")
    parser.add_argument("--output_dir", type=str, default="./outputs/ablations")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    values = [v.strip() for v in args.values.split(",")]
    # Try to convert to appropriate types
    typed_values = []
    for v in values:
        try:
            typed_values.append(int(v))
        except ValueError:
            try:
                typed_values.append(float(v))
            except ValueError:
                typed_values.append(v)

    run_sweep(args.config, args.sweep, typed_values, args.output_dir, args.seeds)
