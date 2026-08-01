"""Centralized hyperparameter configurations.

All configs are frozen before each run and saved to the output directory.
Defaults match the paper's experimental setup.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List


@dataclass
class TrainConfig:
    # Data
    dataset: str = "colored_mnist"          # "colored_mnist" | "waterbirds" | "scm"
    data_root: str = "./data"
    scm_name: Optional[str] = None          # "independent" | "causal_chain" | "confounded"
    scm_causal_parents: List[int] = field(default_factory=lambda: [0, 1])
    scm_nuisance_dims: int = 4
    colored_mnist_p_corr: float = 0.9       # spurious correlation strength

    # Model
    backbone: str = "resnet18"              # "resnet18" | "resnet50" | "tiny_vit"
    pretrained: bool = False                # ImageNet pretrained (True for Waterbirds)
    img_size: int = 32                      # input size (32 for ColoredMNIST, 128 for Waterbirds; ViT-only)
    proj_hidden_dim: int = 2048
    proj_out_dim: int = 128                 # paper: output dim p = 128
    cir_on: str = "projection"              # paper: applied to projection head

    # Loss — paper defaults
    method: str = "cir"                     # "simclr" | "simclr_marginal" | "barlow_twins" |
                                            # "vicreg" | "cir" | "circe"
    temperature: float = 0.1                # paper: tau = 0.1
    hsic_mode: str = "rff"                  # "exact" | "rff" — RFF is practical for O(N)
    hsic_sigma: Optional[float] = None      # None = median heuristic
    hsic_lambda: float = 0.05               # paper: lambda = 0.05 (from sweep)
    rff_dim: int = 128                      # paper: RFF dimension D = 128
    bt_lambd: float = 5e-3                  # Barlow Twins off-diag weight
    vicreg_sim_coeff: float = 25.0
    vicreg_std_coeff: float = 25.0
    vicreg_cov_coeff: float = 1.0

    # Training
    epochs: int = 200
    batch_size: int = 256
    lr: float = 3e-4                        # synthetic SCMs: 1e-3; real datasets: 1e-4 (override)
    weight_decay: float = 1e-6
    lr_schedule: str = "cosine"
    lr_warmup_epochs: int = 10
    seed: int = 0
    num_seeds: int = 3                      # all results are mean ± std over seeds 0,1,2

    # Logging / Checkpointing
    log_every: int = 10
    save_every: int = 10
    output_dir: str = "./outputs"
    run_name: Optional[str] = None
    resume: bool = False
    # Colab/Drive checkpointing. Checkpoints are written under ckpt_root/{run_name}
    # (mounted Drive path by default). On non-POSIX hosts (e.g. Windows dev boxes)
    # ckpt_root is ignored and {output_dir}/{run_name}/checkpoints is used instead.
    ckpt_root: str = "/content/drive/MyDrive/causal-cir-checkpoints"
    ckpt_every: int = 10
    use_wandb: bool = False
    wandb_project: str = "causal-cir"

    # Mixed precision (only active on CUDA)
    amp: bool = True

    # Waterbirds compute reduction. 128x128 is a documented compute-constraint
    # deviation from the paper's original 256x256 design; applied identically
    # to all methods/baselines so cross-method comparison stays fair. Backbone
    # is resnet18 per the paper (Appendix B.2) for all Waterbirds methods.
    resolution: int = 128

    # Plateau early stopping (opt-in): stop if validation loss has not improved
    # by more than plateau_min_improve (relative) for plateau_patience epochs.
    plateau_early_stop: bool = False
    plateau_patience: int = 20
    plateau_min_improve: float = 0.005       # 0.5%

    # Hardware
    device: str = "cuda"
    num_workers: int = 4

    # Diagnostics
    eval_every: int = 5
    acs_n_samples: int = 512
    acs_delta: float = 1.0
    itg_C: float = 1.0


CONFIG_REGISTRY = {
    "config_cir_cmnist": TrainConfig(
        dataset="colored_mnist",
        backbone="resnet18",
        method="cir",
        hsic_lambda=0.05,
        hsic_mode="rff",
        rff_dim=128,
        temperature=0.1,
        epochs=200,
        batch_size=256,
        lr=3e-4,
    ),
    "config_simclr_cmnist": TrainConfig(
        dataset="colored_mnist",
        backbone="resnet18",
        method="simclr",
        temperature=0.1,
        epochs=200,
        batch_size=256,
        lr=3e-4,
    ),
    "config_barlow_cmnist": TrainConfig(
        dataset="colored_mnist",
        backbone="resnet18",
        method="barlow_twins",
        epochs=200,
        batch_size=256,
        lr=3e-4,
    ),
    "config_vicreg_cmnist": TrainConfig(
        dataset="colored_mnist",
        backbone="resnet18",
        method="vicreg",
        epochs=200,
        batch_size=256,
        lr=3e-4,
    ),
    "config_circe_cmnist": TrainConfig(
        dataset="colored_mnist",
        backbone="resnet18",
        method="circe",
        hsic_lambda=0.05,
        epochs=200,
        batch_size=256,
        lr=3e-4,
    ),
    "config_simclr_marginal_cmnist": TrainConfig(
        dataset="colored_mnist",
        backbone="resnet18",
        method="simclr_marginal",
        hsic_lambda=0.05,
        epochs=200,
        batch_size=256,
        lr=3e-4,
    ),
    # SCM configs (Datasets A/B/C: Independent / Causal Chain / Confounded)
    "config_cir_scm_a": TrainConfig(
        dataset="scm",
        scm_name="independent",
        backbone="resnet18",
        method="cir",
        hsic_lambda=0.05,
        hsic_mode="rff",
        rff_dim=128,
        temperature=0.1,
        epochs=200,
        batch_size=256,
        lr=3e-4,
    ),
    "config_cir_scm_b": TrainConfig(
        dataset="scm",
        scm_name="causal_chain",
        backbone="resnet18",
        method="cir",
        hsic_lambda=0.05,
        hsic_mode="rff",
        rff_dim=128,
        temperature=0.1,
        epochs=200,
        batch_size=256,
        lr=3e-4,
    ),
    "config_cir_scm_c": TrainConfig(
        dataset="scm",
        scm_name="confounded",
        backbone="resnet18",
        method="cir",
        hsic_lambda=0.05,
        hsic_mode="rff",
        rff_dim=128,
        temperature=0.1,
        epochs=200,
        batch_size=256,
        lr=3e-4,
    ),
    # Waterbirds configs below: resnet18 @ 128px — matches paper Appendix B.2,
    # reduced resolution documented as a compute-constraint deviation. This is
    # the main Table 1 protocol; resnet50 appears only in scaling_test.py.
    "config_cir_waterbirds": TrainConfig(
        dataset="waterbirds",
        backbone="resnet18",
        pretrained=True,
        img_size=128,
        method="cir",
        hsic_lambda=0.05,
        hsic_mode="rff",
        rff_dim=128,
        temperature=0.1,
        epochs=100,
        batch_size=128,
        lr=1e-4,
    ),
    "config_simclr_waterbirds": TrainConfig(
        dataset="waterbirds",
        backbone="resnet18",
        pretrained=True,
        img_size=128,
        method="simclr",
        temperature=0.1,
        epochs=100,
        batch_size=128,
        lr=1e-4,
    ),
    "config_barlow_waterbirds": TrainConfig(
        dataset="waterbirds",
        backbone="resnet18",
        pretrained=True,
        img_size=128,
        method="barlow_twins",
        epochs=100,
        batch_size=128,
        lr=1e-4,
    ),
    "config_vicreg_waterbirds": TrainConfig(
        dataset="waterbirds",
        backbone="resnet18",
        pretrained=True,
        img_size=128,
        method="vicreg",
        epochs=100,
        batch_size=128,
        lr=1e-4,
    ),
    "config_simclr_marginal_waterbirds": TrainConfig(
        dataset="waterbirds",
        backbone="resnet18",
        pretrained=True,
        img_size=128,
        method="simclr_marginal",
        hsic_lambda=0.05,
        epochs=100,
        batch_size=128,
        lr=1e-4,
    ),
}


def get_config(name: str) -> TrainConfig:
    if name in CONFIG_REGISTRY:
        return CONFIG_REGISTRY[name]
    raise ValueError(f"Unknown config '{name}'")
