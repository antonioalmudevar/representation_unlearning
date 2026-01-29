"""
Defines how to split datasets into retain / forget / val / test sets.
Supports:
  - class-based forgetting (forget specific class IDs)
  - random percentage forgetting
"""
import numpy as np
from typing import Dict, Tuple, Any, Sequence
from .common import SubsetDataset

def _extract_targets(dataset) -> np.ndarray:
    """
    Robustly get targets from many torchvision datasets:
      - CIFAR*: .targets (list[int])
      - SVHN: .labels (ndarray[int])
      - ImageFolder: .samples -> list[(path, class_idx)]
    """
    # Common attribute names
    for name in ("targets", "labels", "y", "ys"):
        if hasattr(dataset, name):
            arr = getattr(dataset, name)
            return np.array(arr)

    # ImageFolder and similar provide .samples
    if hasattr(dataset, "samples"):
        return np.array([s[1] for s in dataset.samples])

    raise AttributeError(
        "Could not find targets/labels in dataset. "
        "Expected one of attributes {targets, labels, y, ys} or samples."
    )

def make_retain_forget_splits(dataset, protocol: Dict[str, Any]) -> Tuple[Any, Any]:
    """
    Returns (retain_dataset, forget_dataset).
    protocol may include:
      - type: "class_forget" or "random_forget"
      - forget_classes: Sequence[int]
      - forget_ratio: float in [0,1]
      - seed: int
      - balance_forget: bool (default False) - if True, balance forget set by downsampling

    Notes:
      - For SVHN, '0' is encoded as label 10 in torchvision.
        If you want to forget digit 0, include 10 in forget_classes.
    """
    rng = np.random.default_rng(protocol.get("seed", 42))
    targets = _extract_targets(dataset)

    forget_type = protocol.get("type", "class_forget")
    if forget_type == "class_forget":
        forget_classes: Sequence[int] = protocol.get("forget_classes", [0])
        forget_mask = np.isin(targets, forget_classes)
    elif forget_type == "random_forget":
        forget_mask = np.zeros_like(targets, dtype=bool)
        forget_ratio = float(protocol.get("forget_ratio", 0.1))
        n_forget = int(len(targets) * forget_ratio)
        forget_idx = rng.choice(len(targets), size=n_forget, replace=False)
        forget_mask[forget_idx] = True
    else:
        raise ValueError(f"Unknown forget type {forget_type}")

    retain_mask = ~forget_mask
    retain_idx = np.where(retain_mask)[0]
    forget_idx = np.where(forget_mask)[0]
    
    # Balance forget set if requested
    if protocol.get("balance_forget", False):
        forget_targets = targets[forget_idx]
        unique_classes, class_counts = np.unique(forget_targets, return_counts=True)
        
        if len(unique_classes) > 0:
            # Find minimum count across all classes in forget set
            min_count = class_counts.min()
            
            # Sample min_count samples from each class
            balanced_forget_idx = []
            for cls in unique_classes:
                cls_mask = forget_targets == cls
                cls_indices = forget_idx[cls_mask]
                sampled = rng.choice(cls_indices, size=min_count, replace=False)
                balanced_forget_idx.extend(sampled)
            
            forget_idx = np.array(balanced_forget_idx)
    
    retain_ds = SubsetDataset(dataset, retain_idx)
    forget_ds = SubsetDataset(dataset, forget_idx)
    return retain_ds, forget_ds
