# src/datasets/toy.py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional


class ToyDataset(Dataset):
    """
    Synthetic toy dataset with 4 classes and 10-dimensional features.
    Each class is sampled from a Gaussian distribution with different means.
    """
    def __init__(
        self,
        n_samples_per_class: int = 250,
        input_dim: int = 10,
        n_classes: int = 4,
        seed: int = 42,
        noise_std: float = 1.0,
    ):
        super().__init__()
        self.n_samples_per_class = n_samples_per_class
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.noise_std = noise_std
        
        # Generate data
        rng = np.random.RandomState(seed)
        
        data = []
        labels = []
        
        # Create class-specific means in a circle pattern
        for class_idx in range(n_classes):
            angle = 2 * np.pi * class_idx / n_classes
            # Class means positioned in a high-dimensional space
            class_mean = np.zeros(input_dim)
            class_mean[0] = 5.0 * np.cos(angle)
            class_mean[1] = 5.0 * np.sin(angle)
            # Add some variation in other dimensions
            class_mean[2:] = rng.randn(input_dim - 2) * 0.5
            
            # Sample from Gaussian
            class_data = rng.randn(n_samples_per_class, input_dim) * noise_std + class_mean
            data.append(class_data)
            labels.extend([class_idx] * n_samples_per_class)
        
        self.data = torch.FloatTensor(np.vstack(data))
        self.targets = labels
        self.classes = list(range(n_classes))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def get_toy_dataset(
    n_samples_per_class: int = 250,
    input_dim: int = 10,
    n_classes: int = 4,
    seed: int = 42,
    noise_std: float = 1.0,
    split: str = "train",
    **kwargs  # Absorb split_protocol and other args
):
    """
    Get toy dataset with train/test/retain/forget splits.
    
    Args:
        n_samples_per_class: Number of samples per class
        input_dim: Dimensionality of input features
        n_classes: Number of classes
        seed: Random seed
        noise_std: Standard deviation of noise
        split: "train", "test", "retain", "forget", or "val"
        **kwargs: Additional arguments (e.g., split protocol params)
    
    Returns:
        ToyDataset instance or subset for retain/forget splits
    """
    from .common import SubsetDataset
    from .splits import make_retain_forget_splits
    
    # Use different seeds for train/test
    if split in ["train", "retain", "forget"]:
        dataset_seed = seed
        base_split = "train"
    else:
        dataset_seed = seed + 1000
        base_split = "test"
    
    # Create base dataset
    base_dataset = ToyDataset(
        n_samples_per_class=n_samples_per_class,
        input_dim=input_dim,
        n_classes=n_classes,
        seed=dataset_seed,
        noise_std=noise_std,
    )
    
    # Handle retain/forget splits
    if split in ["retain", "forget"]:
        # Build protocol dict from kwargs
        protocol = {k: v for k, v in kwargs.items() if k not in ['split']}
        
        retain_ds, forget_ds = make_retain_forget_splits(base_dataset, protocol)
        
        if split == "retain":
            return retain_ds
        else:
            return forget_ds
    
    # For val split, return a small subset of test
    if split == "val":
        # Use a fixed subset of test as validation
        n_val = min(100, len(base_dataset) // 4)
        val_indices = list(range(n_val))
        return SubsetDataset(base_dataset, val_indices)
    
    return base_dataset
