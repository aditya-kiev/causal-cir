"""Head-to-head comparison: CIR vs CIRCE vs HSICBaseline.

Runs all three methods on the same data, same backbone, same budget.
Outputs a unified results table to disk and stdout.

Usage:
    python experiments/compare_circe_pidreg.py
    python experiments/compare_circe_pidreg.py --dataset colored_mnist --epochs 100

Reference verification:
  - CIRCE: Pogodin et al., "Efficient Conditionally Invariant Representation
    Learning", ICLR 2023.  (VERIFY this citation — original draft may have
    been inaccurate.)
  - HSICBaseline: previously named PIDReg. The original publication
    reference (Tsai et al., 2021) could not be verified — renamed to
    avoid incorrect attribution.
"""

import sys
import json
import csv
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.configs import TrainConfig
from experiments.train import main as train_main


METHODS = ["cir", "circe", "hsic_baseline"]
SEEDS = [0, 1, 2]


def run_comparison(
    dataset: str = "colored_mnist",
    backbone: str = "resnet18",
    epochs: int = 200,
    batch_size: int = 256,
    hsic_lambda: float = 0.05,
    output_dir: str = "./outputs/comparison",
):
    """Run all three methods across seeds and aggregate results."""
    results = {}

    for method in METHODS:
        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print(f"{'='*60}")

        seed_metrics = {"train_loss": [], "itg_drop": [], "acc_corr": [], "acc_uncorr": []}

        for seed in SEEDS:
            cfg = TrainConfig(
                dataset=dataset,
                backbone=backbone,
                method=method,
                hsic_lambda=hsic_lambda,
                temperature=0.1,
                epochs=epochs,
                batch_size=batch_size,
                lr=3e-4,
                seed=seed,
                run_name=f"{method}_seed={seed}",
                output_dir=output_dir,
                eval_every=epochs,  # only eval at end
                save_every=9999,    # no intermediate checkpoints
            )
            # Override for specific methods
            if method in ("circe", "hsic_baseline"):
                cfg.hsic_mode = "rff"
                cfg.rff_dim = 128

            try:
                train_main(cfg)
            except Exception as e:
                print(f"Run failed: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Load metrics
            metrics_path = Path(output_dir) / cfg.run_name / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
                if metrics.get("train_loss"):
                    seed_metrics["train_loss"].append(metrics["train_loss"][-1])
                if metrics.get("itg_drop"):
                    seed_metrics["itg_drop"].extend(metrics["itg_drop"])
                # TODO: load acc_corr / acc_uncorr from diagnostics output
                # These need to be saved explicitly in train.py

        # Aggregate
        results[method] = {}
        for key, vals in seed_metrics.items():
            if vals:
                results[method][f"{key}_mean"] = float(np.mean(vals))
                results[method][f"{key}_std"] = float(np.std(vals) if len(vals) > 1 else 0.0)
            else:
                results[method][f"{key}_mean"] = None
                results[method][f"{key}_std"] = None

    # Print table
    _print_table(results)
    _save_results(results, output_dir)

    return results


def _print_table(results: dict):
    print("\n")
    print("=" * 70)
    print(f"{'Method':<20} {'Train Loss':>12} {'ITG Drop %':>12} {'Acc Corr':>12} {'Acc Uncorr':>12}")
    print("-" * 70)
    for method, metrics in results.items():
        tl = f"{metrics.get('train_loss_mean', 'N/A'):.4f}" if metrics.get('train_loss_mean') is not None else "N/A"
        itg = f"{metrics.get('itg_drop_mean', 'N/A'):.2f}" if metrics.get('itg_drop_mean') is not None else "N/A"
        ac = f"{metrics.get('acc_corr_mean', 'N/A'):.3f}" if metrics.get('acc_corr_mean') is not None else "N/A"
        au = f"{metrics.get('acc_uncorr_mean', 'N/A'):.3f}" if metrics.get('acc_uncorr_mean') is not None else "N/A"
        print(f"{method:<20} {tl:>12} {itg:>12} {ac:>12} {au:>12}")
    print("=" * 70)
    print("\nNOTE: All values are mean ± std over seeds", SEEDS)


def _save_results(results: dict, output_dir: str):
    out_path = Path(output_dir) / "comparison_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for method, metrics in results.items():
        row = {"method": method}
        row.update(metrics)
        rows.append(row)

    with open(out_path, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="colored_mnist")
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hsic_lambda", type=float, default=0.05)
    parser.add_argument("--output_dir", type=str, default="./outputs/comparison")
    args = parser.parse_args()

    run_comparison(
        dataset=args.dataset,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hsic_lambda=args.hsic_lambda,
        output_dir=args.output_dir,
    )
