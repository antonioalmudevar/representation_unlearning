# src/datasets/svhn.py
import numpy as np
from torchvision import datasets
from .common import get_transforms, SubsetDataset
from .splits import make_retain_forget_splits

def get_svhn(
    root: str,
    split: str = "train",
    forget_split_type: str = "class_forget",
    normalize: bool = True,
    size: int = 32,
    mean=None,
    std=None,
    **kwargs
):
    """
    SVHN loader with normalization. SVHN has 'train' and 'test' split names in torchvision.
    We also provide 'val', 'retain', 'forget' on top of the 'train' split.
    """
    augment = split in ["train", "retain"]
    dataset_key = "svhn"

    tf_train = get_transforms(
        dataset_key=dataset_key, train=True, augment=True, size=size, normalize=normalize, mean=mean, std=std
    )
    tf_eval = get_transforms(
        dataset_key=dataset_key, train=False, augment=False, size=size, normalize=normalize, mean=mean, std=std
    )

    if split == "test":
        base = datasets.SVHN(root=root, split="test", download=True, transform=tf_eval)
        return base

    # Base train set
    base = datasets.SVHN(root=root, split="train", download=True, transform=tf_train if augment else tf_eval)

    if split == "train":
        return base

    if split in ["retain", "forget"]:
        retain_ds, forget_ds = make_retain_forget_splits(base, {"type": forget_split_type, **kwargs})
        return retain_ds if split == "retain" else forget_ds

    if split == "val":
        rng = np.random.default_rng(42)
        idx = np.arange(len(base))
        rng.shuffle(idx)
        val_size = int(0.1 * len(idx))
        val_idx = idx[:val_size]
        base_val = datasets.SVHN(root=root, split="train", download=False, transform=tf_eval)
        return SubsetDataset(base_val, val_idx)

    return base
