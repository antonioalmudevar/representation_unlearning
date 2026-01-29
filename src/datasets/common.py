# src/datasets/common.py
import torch
# src/datasets/common.py
from typing import Union, Tuple, Sequence, Optional
from torchvision import transforms

# Estadísticas por defecto
DATASET_STATS = {
    "cifar10":   ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    "cifar100":  ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    "svhn":      ([0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]),
    "imagenet":  ([0.485, 0.456, 0.406],     [0.229, 0.224, 0.225]),
    "celeba":    ([0.5063, 0.4258, 0.3832], [0.3100, 0.2900, 0.2880]),
}


NormSpec = Union[
    None,
    bool,
    str,
    Tuple[Sequence[float], Sequence[float]],
]

def _norm_transform(
    dataset_key: str,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> transforms.Normalize:
    if mean is None or std is None:
        if dataset_key not in DATASET_STATS:
            raise ValueError(f"No default normalization for dataset '{dataset_key}'. "
                             "Provide mean/std manually.")
        mean, std = DATASET_STATS[dataset_key]
    return transforms.Normalize(mean=mean, std=std)

def get_transforms(
    *,
    dataset_key: str,          # "cifar10" | "cifar100" | "svhn" | "imagenet"
    train: bool = True,
    augment: bool = True,
    size: int = 32,
    normalize: bool = True,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
):
    """
    Default transforms with optional normalization.
    - dataset_key selects default mean/std if not provided.
    - size controls RandomCrop padding size; eval pipeline keeps tensor shape.
    """
    aug = []
    if train and augment:
        aug = [
            transforms.RandomCrop(size, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    core = [transforms.ToTensor()]
    if normalize:
        core.append(_norm_transform(dataset_key, mean, std))
    return transforms.Compose([*aug, *core])


class SubsetDataset(torch.utils.data.Dataset):
    """Subset wrapper to keep consistent .targets and .classes fields."""
    def __init__(self, base_dataset, indices):
        self.base = base_dataset
        self.indices = list(indices)
        if hasattr(base_dataset, "classes"):
            self.classes = base_dataset.classes
        if hasattr(base_dataset, "targets"):
            targets = base_dataset.targets
            self.targets = [targets[i] for i in indices]
        else:
            self.targets = None

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.base[real_idx]

    def __len__(self):
        return len(self.indices)
