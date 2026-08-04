"""Unit tests for ColoredMNIST caching correctness.

The critical guarantee: caching the PIL-decode+Resize+ToTensor step must NOT
change the spurious-tint assignment for any sample given the same seed. This
file pins the tint assignments (and full tinted-image bytes) for the first 20
indices against a reference captured from the ORIGINAL pre-cache code with
seed=42, p_corr=0.9. If this test fails, experimental data (which image gets
which spurious color) has changed and the cache is NOT safe to use.

Note: each __getitem__ consumes the dataset RNG (one random() + a conditional
randint), so the first-N-calls sequence is only reproducible from a fresh
dataset. Tests below therefore use a single pass over indices on a fresh
dataset per byte-identity assertion.
"""

import sys
import hashlib
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.colored_mnist import ColoredMNIST

# Reference captured from pre-cache ColoredMNIST, seed=42, p_corr=0.9, train split.
REF_LABELS = [5, 0, 4, 1, 9, 2, 1, 3, 1, 4, 3, 5, 3, 6, 1, 7, 2, 8, 6, 9]
REF_SPURIOUS = [5, 7, 4, 1, 9, 2, 1, 3, 1, 4, 5, 5, 0, 6, 1, 7, 2, 8, 6, 9]
# sha256 of the concatenated bytes of the 20 tinted (3,32,32) tensors, pre-cache.
REF_IMG_SHA256 = "8be567ede917ebb064bea20bac89620a724eb6911eacb8a2ecec57d075afe9b1"

N_SAMPLES = 20

DATA_ROOT = str(Path(__file__).resolve().parent.parent / "data")


def _combined_sha256(tensors):
    buf = b""
    for t in tensors:
        buf += t.detach().cpu().numpy().tobytes()
    return hashlib.sha256(buf).hexdigest()


def _fresh_dataset(seed=42):
    return ColoredMNIST(root=DATA_ROOT, split="train", p_corr=0.9,
                        seed=seed, download=False)


def test_tint_and_images_byte_identical():
    """Single pass over the first 20 indices on a fresh dataset.

    Asserts both the spurious-tint assignments AND the full tinted-image bytes
    match the pre-cache reference exactly. Also validates the cache structure
    and that the cache matches the referenced transform on first access.
    """
    ds = _fresh_dataset()
    labels = []
    spurious = []
    tensors = []
    for idx in range(N_SAMPLES):
        img, label, spurious_label = ds[idx]
        labels.append(int(label))
        spurious.append(int(spurious_label))
        tensors.append(img)

    assert labels == REF_LABELS, f"labels changed: {labels}"
    assert spurious == REF_SPURIOUS, f"spurious assignments changed: {spurious}"
    assert _combined_sha256(tensors) == REF_IMG_SHA256, "tinted image bytes changed"

    # Cache integrity
    assert ds._gray_cache is not None
    assert tuple(ds._gray_cache.shape) == (len(ds), 1, 32, 32)
    assert ds._gray_cache.dtype == torch.float32
    img_pil, _ = ds._mnist[0]
    assert torch.allclose(ds._gray_cache[0], ds._base_transform(img_pil))


def test_rng_determinism_same_seed():
    """Two fresh instances with the same seed produce identical tint assignments."""
    ds1 = _fresh_dataset(seed=7)
    ds2 = _fresh_dataset(seed=7)
    sp1 = [ds1[i][2] for i in range(50)]
    sp2 = [ds2[i][2] for i in range(50)]
    assert sp1 == sp2
