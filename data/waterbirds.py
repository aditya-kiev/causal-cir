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


_WATERBIRD_TRANSFORM_TRAIN = T.Compose([
    T.RandomResizedCrop(224, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_WATERBIRD_TRANSFORM_EVAL = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class WaterbirdsWrapper(Dataset):
    """Wraps WILDS Waterbirds into a standard PyTorch Dataset.

    Args:
        root: Path to store WILDS data.
        split: 'train', 'val', or 'test'.
        augment: Whether to apply training augmentations.
    """

    def __init__(
        self,
        root: str = "./data/waterbirds",
        split: str = "train",
        augment: bool = True,
    ):
        self._dataset = get_dataset(dataset="waterbirds", root_dir=root, download=True)
        self._split = split

        split_map = {"train": "train", "val": "val", "test": "test"}
        self._split_name = split_map[split]
        self._indices = self._dataset.get_split_indices(self._split_name)

        self.transform = _WATERBIRD_TRANSFORM_TRAIN if augment else _WATERBIRD_TRANSFORM_EVAL

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        real_idx = self._indices[idx]
        x, y, metadata = self._dataset[real_idx]
        # metadata: [background, ...] where background = 0 (land) or 1 (water)
        if self.transform:
            x = self.transform(x)
        # Return: (image, label, spurious_label)
        spurious_label = metadata[0].item()  # background type
        return x, y, spurious_label

    @property
    def n_classes(self):
        return self._dataset.n_classes

    @property
    def n_spurious(self):
        return 2  # land / water background
