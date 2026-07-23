# Conditional Independence Regularization (CIR) for Contrastive Representation Learning

Codebase for the paper: *Conditional Independence Regularization (CIR) for Contrastive Representation Learning*.

## Reproducibility Protocol

### Environment
```bash
conda create -n cir python=3.10
conda activate cir
pip install -r requirements.txt
python setup.py develop
```

### Dataset Setup

| Dataset | Source | Preprocessing |
|---|---|---|
| **Colored MNIST** | `torchvision.datasets.MNIST` | Grayscale → 3-channel repeat; apply color bias per class via `ColoredMNIST` wrapper. Default: label 0→red (p=0.9), label 1→green (p=0.9), rest flipped. |
| **Waterbirds** | [WILDS](https://wilds.stanford.edu/) | Use `wilds.get_dataset('waterbirds')` with standard `confounder` and `target_name` splits. Train/val/test from official GroupDataset. |

- Colored MNIST does not require any download beyond torchvision MNIST.
- Waterbirds requires: `pip install wilds`. The dataset will auto-download on first load.

### Running Experiments

```bash
# CIR on Colored MNIST
python experiments/train.py --config config_cir_cmnist

# Baseline comparison (CIR vs CIRCE vs HSICBaseline)
python experiments/compare_circe_pidreg.py --config config_compare

# Ablation sweep
python experiments/train_ablation.py --sweep hsic_lambda

# Scaling test
python experiments/scaling_test.py --backbone resnet50
```

All results are logged to `outputs/{run_name}/`. Each run creates:
- `checkpoints/` — best and latest model weights
- `metrics.json` — per-epoch losses and diagnostics
- `config.json` — frozen hyperparameters
- `results.csv` (ablation only) — aggregated results across seeds

### Diagnostics (run after training)

```bash
# Axis-Wise Causal Sensitivity
python diagnostics/acs.py --checkpoint path/to/checkpoint.pt --scm confounded

# Interventional Transfer Gap
python diagnostics/itg.py --checkpoint path/to/checkpoint.pt --dataset waterbirds
```

### Fixed Seeds

Every training script seeds at three levels:
1. Python `random.seed(seed)` and `np.random.seed(seed)`
2. `torch.manual_seed(seed)` + `torch.cuda.manual_seed_all(seed)`
3. `torch.use_deterministic_algorithms(True)` (GPU only, may require `torch.backends.cudnn.deterministic=True`)

See `experiments/train.py` → `set_seed(seed)`.

### Reporting Results

- Every table entry is the mean ± std over **3 seeds** (seeds 0, 1, 2).
- No numbers in the paper originate from a single seed without explicit notation.
- Baseline numbers are re-run in our environment using the same backbone and budget.

---

## Repository Structure

```
causal-cir/
├── data/               # Dataset classes and synthetic SCM generators
│   ├── synthetic_scm.py   # 3 SCM benchmarks with intervention API
│   ├── colored_mnist.py   # ColoredMNIST with configurable bias
│   └── waterbirds.py      # Waterbirds wrapper
├── models/             # Encoder, projection head, backbone factory
│   ├── encoder.py         # SimCLR-style encoder (ResNet + projection)
│   └── backbones.py       # ResNet-18, ResNet-50, tiny ViT
├── losses/             # Loss functions
│   ├── ntxent.py          # NT-Xent (normalized temperature-scaled)
│   ├── hsic.py            # HSIC exact (biased V-stat) + RFF approximation
│   └── cir.py             # CIR regularizer wrapper
├── diagnostics/        # Evaluation protocols
│   ├── acs.py             # Axis-Wise Causal Sensitivity
│   └── itg.py             # Interventional Transfer Gap
├── baselines/          # Re-implementations
│   ├── simclr.py          # Vanilla SimCLR (baseline)
│   ├── simclr_marginal.py # SimCLR + marginal HSIC on each view
│   ├── barlow_twins.py    # Barlow Twins
│   ├── vicreg.py          # VICReg
│   ├── circe.py           # CIRCE (Pogodin et al., 2023, ICLR Oral)
│   └── hsic_baseline.py   # HSIC baseline (previously "PIDReg"; citation unverifiable)
├── experiments/        # Training scripts
│   ├── configs.py         # Centralized hyperparameter configs
│   ├── train.py           # Main training loop
│   ├── train_ablation.py  # Ablation harness
│   ├── compare_circe_pidreg.py  # Head-to-head comparison
│   └── scaling_test.py    # Scaling to larger backbones
├── notebooks/          # Analysis notebooks
├── tests/              # Unit tests
│   ├── test_hsic.py       # HSIC estimator correctness
│   └── test_rff.py        # RFF convergence
└── requirements.txt
```
