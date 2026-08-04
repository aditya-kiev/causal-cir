"""Waterbirds dataset wrapper for the CIR evaluation.

Uses the WILDS Waterbirds dataset (Sagawa et al., 2020):
  - Task: classify bird vs. waterbird (target).
  - Spurious correlation: landbird co-occurs with land background (confounder).
  - Standard splits: train (mostly correlated), val (balanced), test (balanced).

We follow the exact preprocessing from Sagawa et al.:
  - Input size: 224x224
  - Normalization: ImageNet stats
  - Augmentation: RandomResizedCrop, RandomHorizontalFlip for training.

Reference: https://wilds.stanford.edu/datasets/#waterbirds
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

try:
    from wilds import get_dataset
    from wilds.common.data_loaders import get_train_loader, get_eval_loader
except ImportError:
    raise ImportError(
        "Waterbirds requires the WILDS package. Install with: pip install wilds"
    )


def _waterbird_transform_train(resolution: int):
    return T.Compose([
        T.RandomResizedCrop(resolution, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _waterbird_transform_eval(resolution: int):
    return T.Compose([
        T.Resize(int(resolution * 256 / 224)),
        T.CenterCrop(resolution),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class WaterbirdsWrapper(Dataset):
    """Wraps WILDS Waterbirds into a standard PyTorch Dataset.

    Args:
        root: Path to store WILDS data.
        split: 'train', 'val', or 'test'.
        augment: Whether to apply training augmentations.
        resolution: Input resolution in px (default 224; use 128 for lower compute).
        dataset: Optional pre-built WILDS dataset (used by tests to inject a
            WILDS-compatible fake without downloading the real data).
    """

    def __init__(
        self,
        root: str = "./data/waterbirds",
        split: str = "train",
        augment: bool = True,
        resolution: int = 224,
        dataset=None,
    ):
        self._dataset = dataset if dataset is not None else get_dataset(
            dataset="waterbirds", root_dir=root, download=True)
        self._split = split
        self.resolution = resolution
        self._augment = augment

        split_map = {"train": "train", "val": "val", "test": "test"}
        self._split_name = split_map[split]
        self._indices = self._dataset.get_split_indices(self._split_name)

        if augment:
            self.transform = _waterbird_transform_train(resolution)
        elif augment is False:
            self.transform = _waterbird_transform_eval(resolution)
        else:
            self.transform = None  # return raw PIL image

        # Caches, built lazily on first access:
        #  - augment=False (eval): fully decoded+resized+normalized tensor per
        #    index — fully deterministic, so it only ever needs computing once.
        #  - augment=True / None (train): decoded PIL image per index, BEFORE
        #    augmentation. RandomResizedCrop/RandomHorizontalFlip are still
        #    applied fresh on every access so augmentation randomness is kept.
        self._eval_cache = None   # list of tensors (augment=False)
        self._pil_cache = None    # list of PIL images (augment True/None)

    def __len__(self):
        return len(self._indices)

    def _build_eval_cache(self):
        n = len(self._indices)
        cache = [None] * n
        for i, real_idx in enumerate(self._indices):
            x, y, metadata = self._dataset[real_idx]
            cache[i] = (self.transform(x), y, metadata)
        return cache

    def _build_pil_cache(self):
        n = len(self._indices)
        cache = [None] * n
        for i, real_idx in enumerate(self._indices):
            x, y, metadata = self._dataset[real_idx]
            cache[i] = (x, y, metadata)
        return cache

    def __getitem__(self, idx):
        if self._augment is False:
            # Eval path (augment=False): cache the fully-transformed tensor.
            if self._eval_cache is None:
                self._eval_cache = self._build_eval_cache()
            x, y, metadata = self._eval_cache[idx]
        else:
            # Train path (augment True/None): cache decoded PIL, augment fresh.
            if self._pil_cache is None:
                self._pil_cache = self._build_pil_cache()
            x, y, metadata = self._pil_cache[idx]
            if self.transform:
                x = self.transform(x)

        spurious_label = metadata[0].item()  # background type
        return x, y, spurious_label

    @property
    def n_classes(self):
        return self._dataset.n_classes

    @property
    def n_spurious(self):
        return 2  # land / water background
