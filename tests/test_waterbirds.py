"""Unit tests for WaterbirdsWrapper caching correctness.

The caching split:
  - augment=False (eval): the fully decoded+resized+normalized tensor is cached
    per index. That path is fully deterministic, so cached output must be
    byte-identical to the original per-access decode+transform.
  - augment=True / None (train): only the decoded PIL image is cached
    (pre-augmentation). RandomResizedCrop/RandomHorizontalFlip still run fresh
    per access; the cached PIL must be byte-identical to a fresh Image.open().

WILDS Waterbirds requires a ~1.3GB download, so these tests inject a
synthetic WILDS-compatible dataset that mimics WILDS' behavior of opening the
image file fresh on every __getitem__ (WaterbirdsDataset.get_input -> Image.open).
The wrapper code path exercised is identical; only the data source differs.
"""

import sys
import numpy as np
from pathlib import Path

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.waterbirds import (
    WaterbirdsWrapper,
    _waterbird_transform_eval,
    _waterbird_transform_train,
)

N = 10
IMG_SIZE = 64


class FakeWildsWaterbirds:
    """Mimics the subset of the WILDS Waterbirds interface the wrapper uses."""

    n_classes = 2

    def __init__(self, n, img_dir, seed=42):
        self.n = n
        rng = np.random.RandomState(seed)
        self.img_dir = Path(img_dir)
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.files = []
        for i in range(n):
            arr = rng.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            path = self.img_dir / f"img_{i:05d}.png"
            Image.fromarray(arr).save(path)
            self.files.append(path)
        self.y_array = torch.from_numpy(rng.randint(0, 2, (n,)))
        self.metadata_array = torch.from_numpy(rng.randint(0, 2, (n, 2)))
        self._split_indices = list(range(n))

    def get_split_indices(self, name):
        return self._split_indices

    def __getitem__(self, idx):
        # Mirrors WaterbirdsDataset.get_input: fresh file open per access.
        x = Image.open(self.files[idx]).convert("RGB")
        y = self.y_array[idx]
        metadata = self.metadata_array[idx]
        return x, y, metadata


@pytest.fixture
def fake_dataset(tmp_path):
    return FakeWildsWaterbirds(N, tmp_path / "imgs")


def _fresh_image(ds, real_idx):
    return Image.open(ds.files[real_idx]).convert("RGB")


def test_eval_cache_byte_identical(fake_dataset):
    """Eval output must equal the original fresh decode+transform path exactly."""
    wrapper = WaterbirdsWrapper(root="", split="val", augment=False,
                                resolution=32, dataset=fake_dataset)
    for idx in range(N):
        real_idx = wrapper._indices[idx]
        x_fresh, y_fresh, meta_fresh = fake_dataset[real_idx]
        expected = _waterbird_transform_eval(32)(x_fresh)
        got, y, spurious = wrapper[idx]
        assert torch.equal(got, expected), f"eval tensor differs at idx {idx}"
        assert torch.equal(y, y_fresh)
        assert int(spurious) == int(meta_fresh[0].item())


def test_eval_cache_stable_across_accesses(fake_dataset):
    """Repeated access to the same eval index returns the identical tensor."""
    wrapper = WaterbirdsWrapper(root="", split="val", augment=False,
                                resolution=32, dataset=fake_dataset)
    first = [wrapper[i][0] for i in range(N)]
    second = [wrapper[i][0] for i in range(N)]
    for i in range(N):
        assert torch.equal(first[i], second[i])


def test_train_pil_cache_matches_fresh_open(fake_dataset):
    """Cached pre-augmentation PIL must be byte-identical to fresh Image.open."""
    wrapper = WaterbirdsWrapper(root="", split="train", augment=True,
                                resolution=32, dataset=fake_dataset)
    _ = wrapper[0]  # trigger lazy PIL cache build
    for idx in range(N):
        real_idx = wrapper._indices[idx]
        fresh = _fresh_image(fake_dataset, real_idx)
        cached = wrapper._pil_cache[idx][0]
        assert cached.tobytes() == fresh.tobytes(), f"PIL differs at idx {idx}"


def test_train_augmentation_still_random(fake_dataset):
    """augment=True keeps RandomResizedCrop/Flip randomness (not cached)."""
    wrapper = WaterbirdsWrapper(root="", split="train", augment=True,
                                resolution=32, dataset=fake_dataset)
    x1, _, _ = wrapper[0]
    x2, _, _ = wrapper[0]
    assert not torch.equal(x1, x2), "augment=True output was cached (lost randomness)"
    assert x1.shape == (3, 32, 32)


def test_train_raw_pil_matches_fresh_open(fake_dataset):
    """augment=None returns the cached raw PIL, byte-identical to fresh open."""
    wrapper = WaterbirdsWrapper(root="", split="train", augment=None,
                                resolution=32, dataset=fake_dataset)
    _ = wrapper[0]
    for idx in range(N):
        real_idx = wrapper._indices[idx]
        fresh = _fresh_image(fake_dataset, real_idx)
        got = wrapper[idx][0]
        assert got.tobytes() == fresh.tobytes(), f"raw PIL differs at idx {idx}"
