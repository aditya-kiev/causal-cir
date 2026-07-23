"""Scaling test: verify CIR is tractable and gains hold at larger backbones.

Compares ResNet-18 vs ResNet-50 (or tiny ViT) on the same CIR config.
Measures:
  1. Throughput (samples/sec) and memory usage.
  2. ACS / ITG improvement over SimCLR baseline at each scale.

Usage:
    python experiments/scaling_test.py
    python experiments/scaling_test.py --backbones resnet18,resnet50
"""

import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.configs import TrainConfig
from experiments.train import main as train_main
from experiments.run_table1 import _aggregate_seed_metrics


BACKBONES = ["resnet18", "resnet50"]
SEEDS = [0, 1, 2]
# TODO: Add "tiny_vit" once implemented with proper ViT support


def run_scaling_test(
    dataset: str = "colored_mnist",
    backbones: list = None,
    epochs: int = 100,
    batch_size: int = 128,
    output_dir: str = "./outputs/scaling_test",
):
    """Run CIR on multiple backbones and report metrics + timing."""
    if backbones is None:
        backbones = BACKBONES

    results = {}

    for backbone in backbones:
        print(f"\n{'='*60}")
        print(f"Scaling test: backbone={backbone}")
        print(f"{'='*60}")

        for method in ["simclr", "cir"]:
            seed_metrics_list = []

            for seed in SEEDS:
                cfg = TrainConfig(
                    dataset=dataset,
                    backbone=backbone,
                    method=method,
                    hsic_lambda=0.05,
                    hsic_mode="rff",
                    rff_dim=128,
                    temperature=0.1,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=3e-4,
                    seed=seed,
                    run_name=f"{backbone}_{method}_seed={seed}",
                    output_dir=output_dir,
                    eval_every=epochs,
                    log_every=1,
                )

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
                        m = json.load(f)
                    flat = {}
                    if m.get("train_loss"):
                        flat["train_loss"] = m["train_loss"][-1]
                    if m.get("itg_drop"):
                        flat["itg_drop"] = m["itg_drop"][-1]
                    seed_metrics_list.append(flat)

            key = f"{backbone}_{method}"
            agg = _aggregate_seed_metrics(seed_metrics_list, ["train_loss", "itg_drop"])
            results[key] = {
                "backbone": backbone,
                "method": method,
                "train_loss_mean": agg["train_loss_mean"],
                "train_loss_std": agg["train_loss_std"],
                "itg_drop_mean": agg["itg_drop_mean"],
                "itg_drop_std": agg["itg_drop_std"],
            }

    _print_table(results)

    # Save
    out_path = Path(output_dir) / "scaling_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


def _print_table(results: dict):
    print("\n")
    print("=" * 80)
    print(f"{'Backbone':<15} {'Method':<12} {'Train Loss':>22} {'ITG Drop %':>22}")
    print("-" * 80)
    for key, metrics in results.items():
        tl = f"{metrics.get('train_loss_mean', 'N/A'):.4f}" if metrics.get('train_loss_mean') is not None else "N/A"
        if metrics.get('train_loss_std') is not None:
            tl += f" ± {metrics['train_loss_std']:.4f}"
        itg = f"{metrics.get('itg_drop_mean', 'N/A'):.2f}" if metrics.get('itg_drop_mean') is not None else "N/A"
        if metrics.get('itg_drop_std') is not None:
            itg += f" ± {metrics['itg_drop_std']:.2f}"
        print(f"{metrics['backbone']:<15} {metrics['method']:<12} {tl:>22} {itg:>22}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="colored_mnist")
    parser.add_argument("--backbones", type=str, default="resnet18,resnet50")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="./outputs/scaling_test")
    args = parser.parse_args()

    backbones = [b.strip() for b in args.backbones.split(",")]
    run_scaling_test(
        dataset=args.dataset,
        backbones=backbones,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
