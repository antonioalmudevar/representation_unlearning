# src/methods/natmu/method.py
from typing import Any, Dict, List
import copy, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from .base import IUnlearningMethod
from ..helpers.registry import register
from ..helpers.train_utils import evaluate_acc


def _generate_masks_cifar(delta: float) -> np.ndarray:
    """
    Generate the 4 gradual MixUp masks for CIFAR (32x32x3),
    matching generate_mask_cifar in the original code.
    Each mask has shape (3, 32, 32), values in [0, 1].
    delta scales the forget contribution: higher delta = more forget info retained.
    """
    H, W, C = 32, 32, 3
    masks = []
    for direction in range(4):
        m = np.zeros((H, W), dtype=np.float32)
        for i in range(H):
            for j in range(W):
                if direction == 0:   # left-to-right gradient
                    m[i, j] = j / (W - 1)
                elif direction == 1: # right-to-left gradient
                    m[i, j] = 1.0 - j / (W - 1)
                elif direction == 2: # top-to-bottom gradient
                    m[i, j] = i / (H - 1)
                else:                # bottom-to-top gradient
                    m[i, j] = 1.0 - i / (H - 1)
        m = np.clip(delta + m, 0.0, 1.0)
        masks.append(np.stack([m] * C, axis=0))  # (3, H, W)
    return np.stack(masks, axis=0)  # (4, 3, H, W)


def _generate_masks_tinyimagenet(delta: float) -> np.ndarray:
    """
    Generate the 4 gradual MixUp masks for Tiny-ImageNet (64x64x3).
    """
    H, W, C = 64, 64, 3
    masks = []
    for direction in range(4):
        m = np.zeros((H, W), dtype=np.float32)
        for i in range(H):
            for j in range(W):
                if direction == 0:
                    m[i, j] = j / (W - 1)
                elif direction == 1:
                    m[i, j] = 1.0 - j / (W - 1)
                elif direction == 2:
                    m[i, j] = i / (H - 1)
                else:
                    m[i, j] = 1.0 - i / (H - 1)
        m = np.clip(delta + m, 0.0, 1.0)
        masks.append(np.stack([m] * C, axis=0))
    return np.stack(masks, axis=0)  # (4, 3, H, W)


@register("natmu")
class NatMU(IUnlearningMethod):
    """
    Natural Machine Unlearning (NatMU).
    "Towards Natural Machine Unlearning" (He et al., 2024)
    arXiv:2405.15495

    Algorithm:
      For each forget sample x_f:
        1. Compute top-n predicted classes (excluding true class y_f).
        2. For each of the n classes, select one retain sample with that class.
        3. Blend: T_m(x_f, x_r) = x_f * m + x_r * (1 - m)
           using n=4 gradual MixUp masks scaled by delta.
        4. Assign the retain sample's label y_r to the hybrid sample.
      Fine-tune on: retain_set ∪ {all hybrid samples}

    Config keys (under method:):
        epochs          (int,   default 5)
        lr              (float, default 0.1)
        momentum        (float, default 0.9)
        weight_decay    (float, default 0.0)
        lr_scheduler    (str,   default "cosine")  "cosine" or "step"
        delta           (float, default 1.0)  scaling factor for masks
        pattern_length  (int,   default 4)    number of hybrid samples per forget sample (n)
        dataset         (str,   default "cifar")  "cifar" or "tinyimagenet" for mask size
    """

    def setup(self, model, *, retain_loader, forget_loader, val_loader=None,
              cfg: Dict[str, Any], device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self._model = copy.deepcopy(model).to(self.device)
        self.retain_loader = retain_loader
        self.forget_loader = forget_loader
        self.val_loader = val_loader

        hp = cfg.get("method", {})
        self.epochs         = int(hp.get("epochs", 5))
        self.lr             = float(hp.get("lr", 0.1))
        self.momentum       = float(hp.get("momentum", 0.9))
        self.weight_decay   = float(hp.get("weight_decay", 0.0))
        self.lr_scheduler   = str(hp.get("lr_scheduler", "cosine"))
        self.delta          = float(hp.get("delta", 1.0))
        self.pattern_length = int(hp.get("pattern_length", 4))
        dataset_name        = str(cfg.get("dataset", {}).get("name", "cifar10"))
        self.dataset        = "tinyimagenet" if "tiny" in dataset_name else "cifar"
        self.num_classes    = cfg["model"]["num_classes"]

        self._report = {}

    # ------------------------------------------------------------------
    def run(self) -> None:
        start = time.time()

        # ---- Step 1: generate masks ----------------------------------
        if self.dataset == "tinyimagenet":
            masks = _generate_masks_tinyimagenet(self.delta)  # (4, 3, 64, 64)
        else:
            masks = _generate_masks_cifar(self.delta)          # (4, 3, 32, 32)
        masks_forget = masks                                   # weight for forget part
        masks_retain = 1.0 - masks                            # weight for retain part

        # ---- Step 2: select top-k classes for each forget sample -----
        # ---- Step 2b: collect datasets (needed to find absent classes) ----
        print("[NatMU] Collecting datasets...")
        forget_data, forget_labels = self._collect_dataset(self.forget_loader)
        retain_data, retain_labels = self._collect_dataset(self.retain_loader)

        print("[NatMU] Selecting patch classes for forget samples...")
        # Exclude classes absent from retain set (e.g. forgotten classes in class unlearning)
        # to avoid an infinite loop in _get_retain_indices.
        retain_label_set = set(retain_labels.tolist())
        exclude = [c for c in range(self.num_classes) if c not in retain_label_set]
        if exclude:
            print(f"[NatMU] Excluding {len(exclude)} classes absent from retain set: {exclude}")
        random_labels = self._select_patch_classes(masks.shape[0], exclude_classes=exclude)

        # ---- Step 4: find retain indices matching random_labels ------
        retain_idx = self._get_retain_indices(retain_labels, random_labels)

        # ---- Step 5: build hybrid (patch) dataset --------------------
        print("[NatMU] Building hybrid dataset...")
        # forget_data repeated pattern_length times: (n_forget * pattern_length, C, H, W)
        forget_data_rep = np.repeat(forget_data, self.pattern_length, axis=0)
        masks_f = np.tile(masks_forget, (len(forget_data), 1, 1, 1))  # (n_forget*4, C, H, W)
        masks_r = np.tile(masks_retain, (len(forget_data), 1, 1, 1))

        retain_data_part = retain_data[retain_idx].astype(np.float32)

        # Convert forget_data to float for blending
        patch_data = (retain_data_part * masks_r +
                      forget_data_rep.astype(np.float32) * masks_f)
        patch_data = np.clip(patch_data, 0.0, 1.0)  # already normalized

        patch_labels = retain_labels[retain_idx]

        print(f"[NatMU] Hybrid samples: {len(patch_data)} | "
              f"Label match (should be 0): {(patch_labels == np.repeat(forget_labels, self.pattern_length)).sum()}")

        # ---- Step 6: build combined fine-tuning loader ---------------
        combined_loader = self._build_loader(
            retain_data, retain_labels, patch_data, patch_labels)

        # ---- Step 7: fine-tune ---------------------------------------
        optimizer = optim.SGD(self._model.parameters(), lr=self.lr,
                              momentum=self.momentum, weight_decay=self.weight_decay)

        if self.lr_scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs * len(combined_loader))
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

        criterion = nn.CrossEntropyLoss()
        epoch_losses = []

        print(f"[NatMU] Fine-tuning for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            self._model.train()
            total_loss, n_batches = 0.0, 0

            for x, y in combined_loader:
                x, y = x.to(self.device), y.to(self.device)
                loss = criterion(self._model(x), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if self.lr_scheduler == "cosine":
                    scheduler.step()
                total_loss += loss.item()
                n_batches += 1

            if self.lr_scheduler != "cosine":
                scheduler.step()

            avg_loss = total_loss / max(n_batches, 1)
            epoch_losses.append(avg_loss)

            self._model.eval()
            with torch.no_grad():
                acc_r = evaluate_acc(self._model, self.retain_loader, self.device)
                acc_f = evaluate_acc(self._model, self.forget_loader, self.device)
            print(f"Epoch {epoch+1}/{self.epochs} | Loss: {avg_loss:.4f} | "
                  f"Retain Acc: {acc_r:.4f} | Forget Acc: {acc_f:.4f}")

        train_time = time.time() - start
        acc_val    = evaluate_acc(self._model, self.val_loader,    self.device) if self.val_loader    else None
        acc_forget = evaluate_acc(self._model, self.forget_loader, self.device) if self.forget_loader else None

        self._report.update({
            "method":          "natmu",
            "epochs":          self.epochs,
            "lr":              self.lr,
            "delta":           self.delta,
            "pattern_length":  self.pattern_length,
            "train_time_sec":  train_time,
            "train_loss_last": epoch_losses[-1] if epoch_losses else None,
            "val_acc":         acc_val,
            "forget_acc":      acc_forget,
        })

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _select_patch_classes(self, pattern_length: int,
                               exclude_classes: list = None) -> np.ndarray:
        """
        For each forget sample, select top-n predicted classes (excl. true class
        and any classes in exclude_classes, e.g. forget classes absent from retain).
        Returns flat array of shape (n_forget * pattern_length,).
        Matches select_patch_class in the original code.
        """
        self._model.eval()
        top_k = []
        for x, y in self.forget_loader:
            x, y = x.to(self.device), y.to(self.device)
            outputs = self._model(x)
            mask = torch.ones_like(outputs)
            # Exclude true class
            mask.scatter_(1, y.unsqueeze(1), 0)
            # Exclude classes absent from retain set (e.g. forgotten classes)
            if exclude_classes:
                for c in exclude_classes:
                    mask[:, c] = 0.0
            top_k_batch = torch.topk(
                torch.softmax(outputs, dim=1) * mask,
                k=pattern_length, dim=1)[1]
            top_k.append(top_k_batch)
        top_k = torch.cat(top_k, dim=0)  # (n_forget, pattern_length)
        return top_k.reshape(-1).cpu().numpy()

    # ------------------------------------------------------------------
    @staticmethod
    def _collect_dataset(loader: DataLoader):
        """Collect all (data, labels) from a loader as numpy arrays."""
        xs, ys = [], []
        for x, y in loader:
            xs.append(x.numpy() if isinstance(x, torch.Tensor) else x)
            ys.append(y.numpy() if isinstance(y, torch.Tensor) else np.array(y))
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

    # ------------------------------------------------------------------
    @staticmethod
    def _get_retain_indices(retain_labels: np.ndarray,
                            random_labels: np.ndarray) -> np.ndarray:
        """
        For each label in random_labels, find a retain sample with that label.
        Uses direct per-class sampling instead of the original while loop,
        which can be very slow for large datasets or many classes.
        """
        # Build a mapping from class -> list of retain indices
        class_to_indices = {}
        for idx, lbl in enumerate(retain_labels):
            class_to_indices.setdefault(int(lbl), []).append(idx)

        result = np.empty(len(random_labels), dtype=np.int64)
        for i, lbl in enumerate(random_labels):
            candidates = class_to_indices[int(lbl)]
            result[i] = candidates[np.random.randint(len(candidates))]
        return result

    # ------------------------------------------------------------------
    def _build_loader(self, retain_data: np.ndarray, retain_labels: np.ndarray,
                      patch_data: np.ndarray, patch_labels: np.ndarray) -> DataLoader:
        """
        Combine retain set + hybrid patch set into a single shuffled DataLoader.
        Both are already normalized tensors.
        """
        retain_t = torch.from_numpy(retain_data).float()
        retain_y = torch.from_numpy(retain_labels).long()
        patch_t  = torch.from_numpy(patch_data).float()
        patch_y  = torch.from_numpy(patch_labels).long()

        retain_ds = TensorDataset(retain_t, retain_y)
        patch_ds  = TensorDataset(patch_t,  patch_y)
        combined  = ConcatDataset([retain_ds, patch_ds])

        return DataLoader(
            combined,
            batch_size=self.retain_loader.batch_size,
            shuffle=True,
            num_workers=0,
        )

    # ------------------------------------------------------------------
    def get_model(self):
        return self._model

    def report(self) -> Dict[str, Any]:
        return self._report