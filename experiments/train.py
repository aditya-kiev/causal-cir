"""Main training loop.

Usage:
    python experiments/train.py --config config_cir_cmnist
    python experiments/train.py --config config_simclr_cmnist
    python experiments/train.py --config config_cir_cmnist --hsic_lambda 0.5 --epochs 100
"""

import sys
import json
import random
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.configs import TrainConfig, get_config
from data.synthetic_scm import get_scm
from data.colored_mnist import ColoredMNIST
from data.waterbirds import WaterbirdsWrapper
from models.encoder import Encoder
from losses.ntxent import NTXentLoss
from losses.cir import CIRLoss
from baselines.simclr import SimCLR
from baselines.simclr_marginal import SimCLRMarginalHSIC
from baselines.barlow_twins import BarlowTwins
from baselines.vicreg import VICReg
from baselines.circe import CIRCE
from baselines.hsic_baseline import HSICBaseline
from diagnostics.acs import average_causal_sensitivity
from diagnostics.itg import invariance_to_spurious_correlation


def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Contrastive dataset wrapper: applies augmentation twice per sample
# ---------------------------------------------------------------------------

class ContrastiveDataset(Dataset):
    """Wraps a base dataset to produce two augmented views per sample.

    The base dataset should return (img, label, spurious_label) where img
    is a PIL image or preprocessed tensor. The transform is applied
    independently to generate two views.
    """

    def __init__(self, base_dataset, transform):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label, spurious = self.base[idx]
        x1 = self.transform(img)
        x2 = self.transform(img)
        return x1, x2, label, spurious


def build_contrastive_transform(dataset_name: str):
    """Build augmentation pipeline for contrastive learning.

    For Colored MNIST: the dataset returns (3,32,32) tinted tensors.
    We still apply color-jitter and crop augmentations.

    For Waterbirds: ImageNet-style crop+flip.

    Returns a transform or None if the dataset needs raw PIL.
    """
    if dataset_name == "colored_mnist":
        # ColoredMNIST already returns tensors — we add on-the-fly augmentation
        return T.Compose([
            T.RandomResizedCrop(32, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif dataset_name == "waterbirds":
        aug = T.Compose([
            T.RandomResizedCrop(224, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return aug
    return None


class SCMDataset(Dataset):
    """Synthetic SCM dataset: generates random latents and renders to images on the fly.

    Returns (aug1, aug2, label, spurious) where label/spurious are dummy values
    since SCM data has no natural class labels.
    """

    def __init__(self, scm, size=5000):
        self.scm = scm
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        with torch.no_grad():
            z = self.scm.sample_prior(1)
            img = self.scm.render(z).squeeze(0)  # (3, H, W)
        return img, img, 0, 0


def build_dataloaders(cfg: TrainConfig):
    """Build train and eval dataloaders.

    For contrastive training, the dataset returns (x1, x2, label, spurious_label).
    For evaluation (ITG probe), we load each sample once with eval transforms.
    Returns: (scm_or_None, train_loader, probe_train_loader, corr_loader, uncorr_loader)
    """
    train_kwargs = {
        "batch_size": cfg.batch_size,
        "shuffle": True,
        "num_workers": cfg.num_workers,
        "pin_memory": True,
        "drop_last": True,
    }
    eval_kwargs = {**train_kwargs, "shuffle": False, "drop_last": False}

    if cfg.dataset == "colored_mnist":
        # Base datasets return tinted tensors (no augmentation)
        train_base = ColoredMNIST(
            root=cfg.data_root, split="train",
            p_corr=cfg.colored_mnist_p_corr, seed=cfg.seed,
        )
        # For ITG probe training — use base dataset (single views, no contrastive aug)
        probe_train_loader = DataLoader(train_base, **eval_kwargs)

        # For contrastive training — wrap with paired augmentations
        train_transform = build_contrastive_transform("colored_mnist")
        train_dataset = ContrastiveDataset(train_base, transform=train_transform)
        train_loader = DataLoader(train_dataset, **train_kwargs)

        # Eval datasets
        corr_base = ColoredMNIST(
            root=cfg.data_root, split="test",
            p_corr=cfg.colored_mnist_p_corr, seed=cfg.seed + 1,
        )
        uncorr_base = ColoredMNIST(
            root=cfg.data_root, split="test",
            p_corr=1.0 - cfg.colored_mnist_p_corr, seed=cfg.seed + 2,
        )
        corr_loader = DataLoader(corr_base, **eval_kwargs)
        uncorr_loader = DataLoader(uncorr_base, **eval_kwargs)

    elif cfg.dataset == "waterbirds":
        train_base = WaterbirdsWrapper(root=cfg.data_root, split="train", augment=None)
        val_dataset = WaterbirdsWrapper(root=cfg.data_root, split="val", augment=False)
        test_dataset = WaterbirdsWrapper(root=cfg.data_root, split="test", augment=False)
        train_transform = build_contrastive_transform("waterbirds")
        train_dataset = ContrastiveDataset(train_base, transform=train_transform)
        train_loader = DataLoader(train_dataset, **train_kwargs)
        probe_train_loader = DataLoader(
            WaterbirdsWrapper(root=cfg.data_root, split="train", augment=False),
            **eval_kwargs,
        )
        corr_loader = DataLoader(val_dataset, **eval_kwargs)
        uncorr_loader = DataLoader(test_dataset, **eval_kwargs)

    elif cfg.dataset == "scm":
        scm = get_scm(cfg.scm_name, causal_parents=cfg.scm_causal_parents,
                      nuisance_dims=cfg.scm_nuisance_dims)
        scm_kwargs = {**train_kwargs, "num_workers": 0, "pin_memory": False}
        scm_eval_kwargs = {**scm_kwargs, "shuffle": False, "drop_last": False}
        train_dataset = SCMDataset(scm, size=cfg.batch_size * 3)
        train_loader = DataLoader(train_dataset, **scm_kwargs)
        probe_train_loader = DataLoader(SCMDataset(scm, size=cfg.batch_size * 2), **scm_eval_kwargs)
        corr_loader = DataLoader(SCMDataset(scm, size=cfg.batch_size * 2), **scm_eval_kwargs)
        uncorr_loader = DataLoader(SCMDataset(scm, size=cfg.batch_size * 2), **scm_eval_kwargs)
        return scm, train_loader, probe_train_loader, corr_loader, uncorr_loader

    else:
        raise ValueError(f"Unknown dataset '{cfg.dataset}'")

    return None, train_loader, probe_train_loader, corr_loader, uncorr_loader


class _ContrastiveWithRegularizer(torch.nn.Module):
    """Wraps an NTXentLoss contrastive term with a regularizer.

    The regularizer is expected to return (loss, metrics_dict).
    Total = contrastive_loss + regularizer_loss.
    """

    def __init__(self, contrastive, regularizer):
        super().__init__()
        self.contrastive = contrastive
        self.regularizer = regularizer

    def forward(self, z1, z2, h1=None, h2=None):
        c_loss = self.contrastive(z1, z2)
        r_loss, r_metrics = self.regularizer(z1, z2, h1, h2)
        total = c_loss + r_loss
        metrics = {"contrastive_loss": c_loss.item(), "total_loss": total.item()}
        for k, v in r_metrics.items():
            if k != "total_loss":
                metrics[k] = v
        return total, metrics


def build_model_and_loss(cfg: TrainConfig, device: torch.device):
    """Build encoder and loss module."""
    encoder = Encoder(
        backbone_name=cfg.backbone,
        proj_hidden_dim=cfg.proj_hidden_dim,
        proj_out_dim=cfg.proj_out_dim,
        img_size=cfg.img_size,
        pretrained=cfg.pretrained,
    ).to(device)

    if cfg.method == "simclr":
        loss_fn = SimCLR(temperature=cfg.temperature)
    elif cfg.method == "simclr_marginal":
        loss_fn = SimCLRMarginalHSIC(
            temperature=cfg.temperature,
            hsic_lambda=cfg.hsic_lambda,
            sigma=cfg.hsic_sigma,
        )
    elif cfg.method == "barlow_twins":
        loss_fn = BarlowTwins(lambd=cfg.bt_lambd)
    elif cfg.method == "vicreg":
        loss_fn = VICReg(
            sim_coeff=cfg.vicreg_sim_coeff,
            std_coeff=cfg.vicreg_std_coeff,
            cov_coeff=cfg.vicreg_cov_coeff,
        )
    elif cfg.method == "cir":
        contrastive = NTXentLoss(temperature=cfg.temperature)
        regularizer = CIRLoss(
            hsic_mode=cfg.hsic_mode,
            hsic_sigma=cfg.hsic_sigma,
            rff_dim=cfg.rff_dim,
        )
        def loss_fn(z1, z2, h1, h2):
            c_loss = contrastive(z1, z2)
            if cfg.cir_on == "encoder":
                r_loss = regularizer(h1, h2)
            else:
                r_loss = regularizer(z1, z2)
            total = c_loss + cfg.hsic_lambda * r_loss
            return total, {
                "contrastive_loss": c_loss.item(),
                "cir_loss": r_loss.item(),
                "total_loss": total.item(),
            }
    elif cfg.method == "circe":
        loss_fn = _ContrastiveWithRegularizer(
            NTXentLoss(temperature=cfg.temperature),
            CIRCE(dim=cfg.proj_out_dim, hsic_lambda=cfg.hsic_lambda, hsic_sigma=cfg.hsic_sigma),
        )
    elif cfg.method == "hsic_baseline":
        loss_fn = _ContrastiveWithRegularizer(
            NTXentLoss(temperature=cfg.temperature),
            HSICBaseline(hsic_lambda=cfg.hsic_lambda, hsic_sigma=cfg.hsic_sigma),
        )
    else:
        raise ValueError(f"Unknown method '{cfg.method}'")

    return encoder, loss_fn


def train_epoch(encoder, loss_fn, loader, optimizer, scheduler, cfg, device, epoch):
    encoder.train()
    total_losses = []

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for batch in pbar:
        if len(batch) == 4:
            x1, x2, y, spurious = batch
        else:
            x, y, spurious = batch
            x1 = x2 = x

        x1, x2 = x1.to(device), x2.to(device)
        optimizer.zero_grad()

        h1, z1 = encoder(x1)
        h2, z2 = encoder(x2)

        loss, metrics = loss_fn(z1, z2, h1, h2)
        loss.backward()
        optimizer.step()

        total_losses.append(metrics.get("total_loss", loss.item()))
        if scheduler is not None:
            scheduler.step()

        pbar.set_postfix({"loss": f"{total_losses[-1]:.4f}"})

    return float(np.mean(total_losses))


@torch.no_grad()
def evaluate(encoder, loss_fn, loader, device):
    encoder.eval()
    losses = []
    for batch in loader:
        if len(batch) == 4:
            x1, x2, y, spurious = batch
        else:
            x, y, spurious = batch
            x1 = x2 = x
        x1, x2 = x1.to(device), x2.to(device)
        h1, z1 = encoder(x1)
        h2, z2 = encoder(x2)
        loss, metrics = loss_fn(z1, z2, h1, h2)
        losses.append(metrics.get("total_loss", loss.item()))
    return float(np.mean(losses))


def main(cfg: TrainConfig):
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if cfg.run_name is None:
        cfg.run_name = f"{cfg.method}_{cfg.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    output_dir = Path(cfg.output_dir) / cfg.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)

    # Data
    scm, train_loader, probe_train_loader, corr_loader, uncorr_loader = build_dataloaders(cfg)
    if cfg.dataset == "scm":
        print(f"SCM mode: using '{cfg.scm_name}' with causal parents {cfg.scm_causal_parents}")

    encoder, loss_fn = build_model_and_loss(cfg, device)

    # NOTE: Paper Appendix B.2 states plain Adam, but the code currently uses
    # AdamW with weight_decay=1e-6. This discrepancy needs a decision from
    # Aditya — either the paper text changes to say AdamW, or the code changes
    # to plain Adam (removing weight_decay) — before real experiments are run
    # under this config, since it will affect reproducing numbers either way.
    params = list(encoder.parameters())
    if isinstance(loss_fn, torch.nn.Module):
        params += list(loss_fn.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    scheduler = None
    if cfg.lr_schedule == "cosine" and train_loader is not None:
        total_steps = cfg.epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps
        )
    elif cfg.lr_schedule == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)

    best_loss = float("inf")
    history = {"train_loss": [], "eval_loss": [], "acs": [], "itg_drop": []}

    start_epoch = 1
    ckpt_dir = output_dir / "checkpoints"
    latest_ckpt = ckpt_dir / "latest.pt"
    if cfg.resume and latest_ckpt.exists():
        print(f"Resuming from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        history = ckpt["history"]
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"Resumed at epoch {start_epoch} (best_loss={best_loss:.4f})")

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_loss = train_epoch(
            encoder, loss_fn, train_loader, optimizer, scheduler, cfg, device, epoch
        )
        history["train_loss"].append(train_loss)

        if epoch % cfg.eval_every == 0 and corr_loader is not None:
            eval_loss = evaluate(encoder, loss_fn, corr_loader, device)
            history["eval_loss"].append(eval_loss)
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, eval_loss={eval_loss:.4f}")

            # Diagnostics
            if cfg.dataset == "scm":
                acs_result = average_causal_sensitivity(
                    encoder, scm, n_samples=cfg.acs_n_samples,
                    delta=cfg.acs_delta, device=device,
                )
                history["acs"].append(acs_result["concentration"])
                print(f"  ACS concentration: {acs_result['concentration']:.4f}")

            if cfg.dataset in ("colored_mnist", "waterbirds"):
                itg_result = invariance_to_spurious_correlation(
                    encoder, probe_train_loader, corr_loader, uncorr_loader, device,
                    C=cfg.itg_C,
                )
                history["itg_drop"].append(itg_result["accuracy_drop"])
                history.setdefault("acc_corr", []).append(itg_result["acc_corr"])
                history.setdefault("acc_uncorr", []).append(itg_result["acc_uncorr"])
                print(f"  ITG accuracy drop: {itg_result['accuracy_drop']:.2f}% "
                      f"(corr={itg_result['acc_corr']:.3f}, uncorr={itg_result['acc_uncorr']:.3f})")

            # Save best model
            if eval_loss < best_loss:
                best_loss = eval_loss
                torch.save(encoder.state_dict(), output_dir / "best.pt")

        if epoch % cfg.save_every == 0:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_dict = {
                "epoch": epoch,
                "encoder_state_dict": encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "best_loss": best_loss,
                "config": asdict(cfg),
            }
            if scheduler is not None:
                ckpt_dict["scheduler_state_dict"] = scheduler.state_dict()
            torch.save(ckpt_dict, ckpt_dir / f"epoch_{epoch}.pt")
            torch.save(ckpt_dict, latest_ckpt)

        with open(output_dir / "metrics.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"Training complete. Results in {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None,
                        help="Explicit run name (default: {method}_{dataset}_{timestamp})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from outputs/{run_name}/checkpoints/latest.pt")
    args, overrides = parser.parse_known_args()

    cfg = get_config(args.config)

    override_iter = iter(overrides)
    for k, v in zip(override_iter, override_iter):
        if k.startswith("--"):
            key = k.lstrip("--")
            if hasattr(cfg, key):
                current = getattr(cfg, key)
                if current is None:
                    setattr(cfg, key, v)
                else:
                    setattr(cfg, key, type(current)(v))

    if args.run_name is not None:
        cfg.run_name = args.run_name
    if args.resume:
        cfg.resume = True

    main(cfg)
