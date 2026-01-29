# src/datasets/tiny_imagenet.py

import os
import csv
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets

from .common import get_transforms, SubsetDataset
from .splits import make_retain_forget_splits


# ---------------------------------------------------------
# Custom dataset for Tiny-ImageNet validation partition
# ---------------------------------------------------------
class TinyImageNetValDataset(Dataset):
    """
    Tiny-ImageNet validation dataset.
    The official val/ folder contains:
        val/images/*.JPEG
        val/val_annotations.txt
    Images are NOT in class subfolders, so ImageFolder cannot be used.
    This dataset parses val_annotations.txt and maps each image to its label
    using the class_to_idx mapping obtained from the training split.
    """

    def __init__(self, root, transform, class_to_idx):
        """
        Args:
            root: path to val/ folder
            transform: evaluation transform
            class_to_idx: mapping from class name (wnid) to integer label
        """
        self.root = root
        self.img_dir = os.path.join(root, "images")
        self.ann_path = os.path.join(root, "val_annotations.txt")
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.samples = []

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"val/images folder not found at: {self.img_dir}")

        if not os.path.isfile(self.ann_path):
            raise FileNotFoundError(f"val_annotations.txt not found at: {self.ann_path}")

        # Parse annotations
        with open(self.ann_path, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 2:
                    continue

                fname, class_id = row[0], row[1]
                path = os.path.join(self.img_dir, fname)

                # Some datasets use inconsistent JPEG extensions
                if not os.path.isfile(path):
                    alt1 = os.path.splitext(path)[0] + ".JPEG"
                    alt2 = os.path.splitext(path)[0] + ".jpg"
                    if os.path.isfile(alt1):
                        path = alt1
                    elif os.path.isfile(alt2):
                        path = alt2
                    else:
                        continue  # skip missing file

                if class_id not in class_to_idx:
                    # Very unlikely, but skip if label not found
                    continue

                target = class_to_idx[class_id]
                self.samples.append((path, target))

        if len(self.samples) == 0:
            raise RuntimeError(
                "Validation dataset is empty. Check val/images/ and val_annotations.txt."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


# ---------------------------------------------------------
# Tiny-ImageNet main loader
# ---------------------------------------------------------
def get_tinyimagenet(
    root,
    split="train",
    forget_split_type="class_forget",
    normalize=True,
    size=64,
    mean=None,
    std=None,
    **kwargs
):
    """
    Main Tiny-ImageNet loader.
    Expected folder structure:
        root/train/<class>/*.JPEG
        root/val/images/*.JPEG + val_annotations.txt

    Notes:
        - Validation split is the official val/ folder.
        - val/ is NOT folderized. We always parse val_annotations.txt.
        - train/ uses ImageFolder normally.
    """

    # Transforms
    tf_train = get_transforms(
        dataset_key="imagenet", train=True, augment=True, size=size,
        normalize=normalize, mean=mean, std=std
    )
    tf_eval = get_transforms(
        dataset_key="imagenet", train=False, augment=False, size=size,
        normalize=normalize, mean=mean, std=std
    )

    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")

    # ---------------------------------------------------------
    # Validation and test splits
    # ---------------------------------------------------------
    if split in ["val", "test"]:
        # Obtain class_to_idx mapping from ImageFolder(train)
        train_folder = datasets.ImageFolder(train_dir)
        class_to_idx = train_folder.class_to_idx

        # Always use our custom val dataset (val is not folderized)
        base_val = TinyImageNetValDataset(val_dir, transform=tf_eval, class_to_idx=class_to_idx)

        # For Tiny-ImageNet we treat val as test
        return base_val

    # ---------------------------------------------------------
    # Train / Retain / Forget
    # ---------------------------------------------------------
    base_train = datasets.ImageFolder(
        train_dir,
        transform=tf_train if split in ["train", "retain"] else tf_eval
    )

    if split == "train":
        return base_train

    if split in ["retain", "forget"]:
        retain_ds, forget_ds = make_retain_forget_splits(
            base_train,
            {"type": forget_split_type, **kwargs}
        )
        return retain_ds if split == "retain" else forget_ds

    # Fallback
    return base_train
