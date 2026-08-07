"""Launch the full Table 1 matrix on Google Colab (T4 GPU).

This script bundles everything needed to reproduce Table 1 on a Colab T4:
dependency bootstrap, data handling, and the run itself, with checkpoint
resume so a dropped Colab session picks up where it left off in Drive.

Usage (in a Colab cell, after cloning the repo):
    %cd causal-cir
    !python experiments/colab_table1.py

It runs all 5 methods x 5 benchmarks x 3 seeds (75 runs) via run_table1.
Checkpoints write to cfg.ckpt_root (Drive) so you can re-attach after the
~2h session budget; re-running this same script resumes instead of restarts.

Outputs land in ./outputs/table1:
    results_by_dataset.csv           75 rows (method, dataset, seed, metrics)
    results_by_dataset_summary.csv   25 rows (method, dataset, mean/std)
    results_table1_avg.csv           5 rows (method, mean +/- std across ds)
    table1_results.json              consolidated structure
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def pip(*pkgs: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs],
                   check=True)


def main() -> None:
    # 1. Dependencies (Colab fresh-session friendly).
    pip("torch", "torchvision", "wilds", "scipy", "einops", "pandas",
        "matplotlib", "tqdm")

    # 2. Path setup: run scripts import the repo root.
    sys.path.insert(0, str(REPO_ROOT))

    # 3. WILDS dataset is large; build_dataloaders downloads to ./data on request.
    from experiments.configs import TrainConfig
    print("Prompt: attach a GPU runtime. T4 = ~24h wall-clock for the")
    print("full 75-run matrix as configured (resnet18 @ 128 training).")

    # 4. Launch the full matrix. --ckpt-root keeps checkpoints on Drive via
    #    the config default so a dropped session resumes.
    from experiments.run_table1 import run_table1
    out = REPO_ROOT / "outputs" / "table1"
    run_table1(datasets=None, output_dir=str(out), ckpt_root=None)

    print(f"\nTable 1 complete. CSVs in {out}")


if __name__ == "__main__":
    main()