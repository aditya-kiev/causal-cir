"""Interventional Transfer Gap (ITG).

Protocol:
  1. Train a linear probe on top of frozen representations using the
     training set where spurious correlation holds (e.g., Colored MNIST
     with p_corr=0.9 or Waterbirds training split).
  2. Evaluate the probe on a test set where the correlation is flipped
     or removed (e.g., p_corr=0.1 for Colored MNIST, or Waterbirds val split).
  3. Report: % accuracy drop = (acc_corr - acc_anti) / acc_corr * 100.

A smaller drop indicates better invariance.

Reference:
  - Arjovsky et al. (2019). "Invariant Risk Minimization."
  - Sagawa et al. (2020). "Distributionally Robust Neural Networks."
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset
from typing import Dict


def extract_representations(
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple:
    """Extract frozen representations from encoder.

    Returns:
        h_all: (N, D) numpy array.
        y_all: (N,) numpy array of labels.
    """
    encoder.eval()
    encoder.to(device)

    h_list, y_list = [], []
    with torch.no_grad():
        for x, y, *_ in loader:
            x = x.to(device)
            h, _ = encoder(x)
            h_list.append(h.cpu().numpy())
            y_list.append(y.numpy())

    return np.concatenate(h_list, axis=0), np.concatenate(y_list, axis=0)


def _train_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
) -> LogisticRegression:
    """Train a multiclass logistic regression probe."""
    clf = LogisticRegression(C=C, max_iter=max_iter, multi_class="multinomial")
    clf.fit(X, y)
    return clf


def invariance_to_spurious_correlation(
    encoder: nn.Module,
    train_loader: DataLoader,
    test_loader_corr: DataLoader,
    test_loader_uncorr: DataLoader,
    device: torch.device = torch.device("cpu"),
    C: float = 1.0,
) -> Dict:
    """Compute ITG metric.

    Args:
        encoder: Trained encoder (frozen).
        train_loader: DataLoader for training (correlated environment).
        test_loader_corr: DataLoader for correlated test set.
        test_loader_uncorr: DataLoader for uncorrelated/anti-correlated test set.
        device: Torch device.
        C: Logistic regression regularization strength.

    Returns:
        dict with keys:
          - 'acc_corr': Accuracy on correlated test set.
          - 'acc_uncorr': Accuracy on uncorrelated test set.
          - 'accuracy_drop': (acc_corr - acc_uncorr) / acc_corr * 100.
          - 'probe_coef': Learned linear probe weights.
    """
    print("Extracting training representations...")
    X_train, y_train = extract_representations(encoder, train_loader, device)
    print(f"Training probe on {X_train.shape[0]} samples, dim={X_train.shape[1]}...")
    probe = _train_linear_probe(X_train, y_train, C=C)

    # Evaluate on correlated test set
    X_corr, y_corr = extract_representations(encoder, test_loader_corr, device)
    acc_corr = probe.score(X_corr, y_corr)

    # Evaluate on uncorrelated test set
    X_uncorr, y_uncorr = extract_representations(encoder, test_loader_uncorr, device)
    acc_uncorr = probe.score(X_uncorr, y_uncorr)

    accuracy_drop = (acc_corr - acc_uncorr) / max(acc_corr, 1e-8) * 100.0

    return {
        "acc_corr": float(acc_corr),
        "acc_uncorr": float(acc_uncorr),
        "accuracy_drop": float(accuracy_drop),
        "probe_coef": probe.coef_,
    }
