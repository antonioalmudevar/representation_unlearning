# src/datasets/cifar.py
import os
import numpy as np
from torchvision import datasets
from .common import get_transforms, SubsetDataset
from .splits import make_retain_forget_splits

def get_cifar(
    name: str,
    root: str,
    split: str = "train",
    forget_split_type: str = "class_forget",  # <-- renamed (was 'type') to avoid shadowing built-in
    normalize: bool = True,
    size: int = 32,
    mean=None,
    std=None,
    **kwargs
):
    """
    CIFAR loader with normalization. Supports splits:
      - 'train', 'val' (10% of train), 'retain', 'forget', 'test'
    """
    name = name.lower()
    if name not in ["cifar10", "cifar100"]:
        raise ValueError(f"Unknown CIFAR dataset: {name}")

    is_train_like = split in ["train", "retain", "forget", "val"]
    augment = split in ["train", "retain"]
    dataset_key = name  # for default stats

    # Build transforms
    tf_train = get_transforms(
        dataset_key=dataset_key, train=True, augment=True, size=size, normalize=normalize, mean=mean, std=std
    )
    tf_eval = get_transforms(
        dataset_key=dataset_key, train=False, augment=False, size=size, normalize=normalize, mean=mean, std=std
    )

    # Base datasets
    if split == "test":
        if name == "cifar10":
            base = datasets.CIFAR10(root=root, train=False, download=True, transform=tf_eval)
        else:
            base = datasets.CIFAR100(root=root, train=False, download=True, transform=tf_eval)
        return base

    # Train base (always constructed once)
    if name == "cifar10":
        base = datasets.CIFAR10(root=root, train=True, download=True, transform=tf_train if augment else tf_eval)
    else:
        base = datasets.CIFAR100(root=root, train=True, download=True, transform=tf_train if augment else tf_eval)

    if split == "train":
        return base

    if split in ["retain", "forget"]:
        retain_ds, forget_ds = make_retain_forget_splits(base, {"type": forget_split_type, **kwargs})
        return retain_ds if split == "retain" else forget_ds

    if split == "val":
        # Make validation indices over the TRAIN set, but apply EVAL transforms
        rng = np.random.default_rng(42)
        idx = np.arange(len(base))
        rng.shuffle(idx)
        val_size = int(0.1 * len(idx))
        val_idx = idx[:val_size]

        # Reconstruct a TRAIN dataset with eval transforms to wrap the same underlying files
        if name == "cifar10":
            base_val = datasets.CIFAR10(root=base.root, train=True, download=False, transform=tf_eval)
        else:
            base_val = datasets.CIFAR100(root=base.root, train=True, download=False, transform=tf_eval)
        return SubsetDataset(base_val, val_idx)

    raise ValueError(f"Unknown split: {split}")
