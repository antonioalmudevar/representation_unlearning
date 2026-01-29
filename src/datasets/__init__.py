# src/datasets/__init__.py
from typing import Dict, Any
from torch.utils.data import DataLoader

from .cifar import get_cifar
from .svhn import get_svhn
from .tiny_imagenet import get_tinyimagenet
from .toy import get_toy_dataset
from .splits import make_retain_forget_splits

# -----------------------------------------------------------
# Number of classes per dataset (for N_CLASSES.get usage)
# -----------------------------------------------------------
N_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
    "svhn": 10,
    "tiny_imagenet": 200,
    "toy": 4,  # Default, can be overridden in config
}

# -----------------------------------------------------------
# Dataset factory
# -----------------------------------------------------------
def get_dataset(name: str, **kwargs):
    name = name.lower()
    if name in ["cifar10", "cifar100"]:
        return get_cifar(name, **kwargs)
    elif name == "svhn":
        return get_svhn(**kwargs)
    elif name == "tiny_imagenet":
        return get_tinyimagenet(**kwargs)
    elif name == "toy":
        return get_toy_dataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")

# -----------------------------------------------------------
# Loader factory
# -----------------------------------------------------------
def get_loader(
    cfg: Dict[str, Any],
    split: str = "train",
    batch_size: int = None,
    shuffle: bool = None,
    num_workers: int = None,
):
    """
    Returns a DataLoader for a given split (train/retain/forget/val/test).
    The config must include:
      - name, data_root, batch_size, num_workers
      - split_protocol (dict) specifying how to build retain/forget
          * type: "class_forget" | "random_forget" | "identity_forget" | "attr_forget"
          * additional keys depending on dataset:
              - CIFAR/SVHN/TinyIN: forget_classes or forget_ratio
              - CelebA_attr: attr_index, attr_value
              - CelebA_id: identity_ids
    """
    name = cfg["name"].lower()
    batch_size = batch_size or cfg.get("batch_size", 128)
    num_workers = num_workers or cfg.get("num_workers", 4)
    shuffle = shuffle if shuffle is not None else (split in ["train", "retain", "forget"])

    # Build dataset kwargs based on dataset type
    if name == "toy":
        # Toy dataset doesn't use data_root
        ds_kwargs = dict(split=split)
        # Pass toy-specific parameters
        for key in ["n_samples_per_class", "input_dim", "n_classes", "noise_std", "seed"]:
            if key in cfg:
                ds_kwargs[key] = cfg[key]
    else:
        # Standard datasets use data_root
        ds_kwargs = dict(root=cfg["data_root"], split=split)
    
    # Add split protocol for all datasets
    ds_kwargs.update(cfg.get("split_protocol", {}))

    dataset = get_dataset(name, **ds_kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)
