# Python 3.8 compatible. No __main__ block.
from typing import Any, Dict, List, Tuple, Optional
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset


def _extract_xy(batch):
    # Support (x, y) or (x, _, y)
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            x, y = batch
        elif len(batch) == 3:
            x, _, y = batch
        else:
            raise ValueError("Unsupported batch format.")
    else:
        raise ValueError("Unsupported batch type.")
    return x, y


def _first_sample_shape(dataset) -> Tuple[int, int, int]:
    # Try to get the first sample
    sample = dataset[0]
    
    # Handle different tuple formats: (x, y) or (x, _, y)
    if isinstance(sample, (list, tuple)):
        x0 = sample[0]
    else:
        raise ValueError("Expected dataset to return tuple/list, got: {}".format(type(sample)))
    
    if not torch.is_tensor(x0):
        x0 = torch.as_tensor(x0)
    
    if x0.ndim == 3:  # C,H,W
        return int(x0.shape[0]), int(x0.shape[1]), int(x0.shape[2])
    elif x0.ndim == 1:
        # Might be a flattened image or feature vector - try to infer from common sizes
        total = x0.shape[0]
        # Common image sizes: 28x28=784 (MNIST), 32x32x3=3072 (CIFAR)
        if total == 784:  # MNIST
            return 1, 28, 28
        elif total == 3072:  # CIFAR-10/100
            return 3, 32, 32
        else:
            raise ValueError("Cannot infer image shape from flattened vector of size {}".format(total))
    raise ValueError("Expected CHW image samples, got shape: {}".format(tuple(x0.shape)))


def _infer_num_classes(dataset, max_scan: int = 4096) -> int:
    labels = []
    n = min(len(dataset), max_scan)
    for i in range(n):
        item = dataset[i]
        y = item[1] if len(item) == 2 else item[2]
        if torch.is_tensor(y): y = int(y.item())
        labels.append(int(y))
    return int(max(labels) + 1) if labels else 10


def _infer_forget_classes_from_cfg(cfg: Dict[str, Any]) -> List[int]:
    proto = (cfg.get("dataset") or {}).get("split_protocol") or {}
    if proto.get("type") == "class_forget":
        f = proto.get("forget_classes", [])
        return [int(v) for v in f]
    # optional override
    m = cfg.get("method", {})
    if "forget_class" in m:
        return [int(m["forget_class"])]
    if "forget_classes" in m:
        return [int(v) for v in m["forget_classes"]]
    return []


def _classwise_indices(dataset, num_classes: int) -> List[List[int]]:
    idxs = [[] for _ in range(num_classes)]
    for i in range(len(dataset)):
        item = dataset[i]
        y = item[1] if len(item) == 2 else item[2]
        if torch.is_tensor(y): y = int(y.item())
        if 0 <= y < num_classes:
            idxs[y].append(i)
    return idxs


class _NoiseAndRetainDataset(Dataset):
    """
    Dataset mixing:
      - optimized noise images labeled as their target forget class(es)
      - a (small) pool of retained samples (image + label)
    """
    def __init__(self, noise: torch.Tensor, noise_labels: torch.Tensor,
                 retain_samples: List[Tuple[torch.Tensor, int]]):
        super().__init__()
        self.noise = noise.detach().cpu()           # [N, C, H, W]
        self.noise_y = noise_labels.detach().cpu()  # [N]
        self.retain = retain_samples                # list[(x_cpu, y_int)]

    def __len__(self):
        return self.noise.shape[0] + len(self.retain)

    def __getitem__(self, idx):
        if idx < self.noise.shape[0]:
            return self.noise[idx], int(self.noise_y[idx].item())
        j = idx - self.noise.shape[0]
        x, y = self.retain[j]
        return x, y


def _optimize_noise_for_class(
    model: nn.Module,
    device: torch.device,
    *,
    noise_batch: int,
    channels: int,
    height: int,
    width: int,
    target_class: int,
    iters: int,
    lr: float,
    l2_lambda: float,
    clamp: float,
) -> torch.Tensor:
    """
    Learn error-maximizing noise N for a target class c_f by minimizing:
      J(N) = -(CE(f(N), c_f)) + λ ||N||_2^2
    (Maximizes classification loss for class c_f; λ regularizes the noise magnitude.)
    """
    model.eval()
    N = torch.randn(noise_batch, channels, height, width, device=device, requires_grad=True)
    opt = torch.optim.Adam([N], lr=lr)

    target = torch.full((noise_batch,), int(target_class), device=device, dtype=torch.long)
    for _ in range(max(1, int(iters))):
        opt.zero_grad(set_to_none=True)
        logits = model(N)
        ce = F.cross_entropy(logits, target)  # CE(f(N), c_f)
        reg = (N.pow(2).mean()) if l2_lambda > 0 else 0.0
        # Minimize -(CE) + λ||N||^2  <=>  maximize CE while penalizing ||N||
        loss = -ce + (l2_lambda * reg if isinstance(reg, torch.Tensor) else l2_lambda * reg)
        loss.backward()
        opt.step()
        if clamp is not None and clamp > 0:
            N.data.clamp_(-clamp, clamp)
    return N.detach()


def _fit_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    lr: float,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    max_norm: float = 0.0,
    head_last_layer_lr: Optional[float] = None,
) -> None:
    model.train()

    params = model.parameters()
    if head_last_layer_lr is not None:
        # Optional: use a higher LR for the classifier head (as in paper's large-scale setting)
        groups = []
        head = getattr(model, "fc", None) or getattr(model, "classifier", None)
        if head is not None and hasattr(head, "parameters"):
            head_params = list(head.parameters())
            body_params = [p for p in model.parameters() if all(p is not q for q in head_params)]
            groups = [
                {"params": body_params, "lr": lr},
                {"params": head_params, "lr": head_last_layer_lr},
            ]
            opt = torch.optim.SGD(groups, momentum=momentum, weight_decay=weight_decay)
        else:
            opt = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        opt = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    for x, y in loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if max_norm and max_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        opt.step()


class UNSIR:
    """
    UNSIR (Unlearning with Single-pass Impair and Repair)
      1) Learn error-maximizing noise per forgotten class (with L2 penalty).
      2) Impair: 1 epoch on retain_sub + noise (high LR).
      3) Repair: 1 epoch on retain_sub (lower LR).
    Paper shows 1+1 epochs and high LR are effective; class-forget and multi-class supported.
    """

    def setup(
        self,
        model: nn.Module,
        *,
        retain_loader,
        forget_loader,
        val_loader,
        cfg: Dict[str, Any],
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.retain_loader = retain_loader
        self.forget_loader = forget_loader
        self.val_loader = val_loader
        self.cfg = cfg

        m = dict(cfg.get("method", {}))
        # ---- noise learning hyperparams (Eq. 1)
        self.noise_batch_size = int(m.get("noise_batch_size", 32))
        self.noise_iters = int(m.get("noise_iters", 25))
        self.noise_lr = float(m.get("noise_lr", 1e-2))
        self.noise_l2_lambda = float(m.get("noise_l2_lambda", 0.1))  # λ in Eq. (1)
        self.noise_clamp = float(m.get("noise_clamp", 2.5))
        self.noise_copies = int(m.get("noise_copies", 20))  # copies per class to build a small noise set

        # ---- impair / repair
        self.impair_epochs = int(m.get("impair_epochs", 1))
        self.repair_epochs = int(m.get("repair_epochs", 1))
        self.impair_lr = float(m.get("impair_lr", 2e-2))   # paper uses high LR for impair on CIFAR-10
        self.repair_lr = float(m.get("repair_lr", 1e-2))
        self.weight_decay = float(m.get("weight_decay", 0.0))
        self.momentum = float(m.get("momentum", 0.9))
        self.max_norm = float(m.get("max_norm", 0.0))
        # optional different LR for classifier head (used in large-scale settings in paper)
        self.head_last_layer_lr_impair = m.get("head_last_layer_lr_impair", None)
        self.head_last_layer_lr_repair = m.get("head_last_layer_lr_repair", None)

        # retain subset size per class (paper uses small Dr_sub)
        self.samples_per_retain_class = int(m.get("samples_per_retain_class", 1000))

        # infer classes & shapes
        base_ds = self.retain_loader.dataset
        self.num_classes = int(m.get("num_classes", _infer_num_classes(base_ds)))
        self.forget_classes = _infer_forget_classes_from_cfg(cfg)
        if not self.forget_classes:
            raise ValueError("UNSIR: could not infer forget classes (set dataset.split_protocol or method.forget_class(es)).")

        # Try to infer image shape - if it fails, UNSIR cannot be used with this dataset
        try:
            c, h, w = _first_sample_shape(base_ds)
            self.img_shape = (c, h, w)
        except ValueError as e:
            # UNSIR requires image data for noise generation
            dataset_name = cfg.get("dataset", {}).get("name", "unknown")
            raise ValueError(
                f"UNSIR requires image datasets (CHW format) but got dataset '{dataset_name}'. "
                f"Original error: {e}. "
                f"UNSIR generates error-maximizing noise images and is not applicable to non-image datasets."
            )

        # logs
        self.logs: Dict[str, Any] = {}

    def run(self) -> None:
        device = self.device
        model = self.model

        # Build a class-wise pool from retain + forget train datasets (to sample small Dr_sub)
        combo_ds = ConcatDataset((self.retain_loader.dataset, self.forget_loader.dataset))
        class_idxs = _classwise_indices(combo_ds, self.num_classes)

        # Collect retained samples excluding forget classes
        retain_samples: List[Tuple[torch.Tensor, int]] = []
        per_cls = max(1, self.samples_per_retain_class)
        for cls in range(self.num_classes):
            if cls in self.forget_classes:
                continue
            idxs = class_idxs[cls]
            if not idxs:
                continue
            take = random.sample(idxs, min(len(idxs), per_cls))
            for i in take:
                x, y = combo_ds[i]
                if not torch.is_tensor(x):
                    x = torch.as_tensor(x)
                y_int = int(y if not torch.is_tensor(y) else y.item())
                retain_samples.append((x.cpu(), y_int))

        # 1) Learn error-maximizing noise per forgotten class
        c, h, w = self.img_shape
        per_class_batch = math.ceil(self.noise_batch_size / max(1, len(self.forget_classes)))
        noises, labels = [], []
        for cls in self.forget_classes:
            n = _optimize_noise_for_class(
                model, device,
                noise_batch=per_class_batch, channels=c, height=h, width=w,
                target_class=int(cls),
                iters=self.noise_iters, lr=self.noise_lr,
                l2_lambda=self.noise_l2_lambda, clamp=self.noise_clamp,
            )
            # replicate to enlarge the noise set
            copies = max(1, self.noise_copies)
            noises.append(n.repeat(copies, 1, 1, 1))
            labels.append(torch.full((n.size(0) * copies,), int(cls), dtype=torch.long, device=device))

        noise_tensor = torch.cat(noises, dim=0) if len(noises) else torch.empty(0, c, h, w, device=device)
        noise_labels = torch.cat(labels, dim=0) if len(labels) else torch.empty(0, dtype=torch.long, device=device)

        # Build impairment loader mixing noise + small retain pool
        impair_ds = _NoiseAndRetainDataset(noise_tensor, noise_labels, retain_samples)
        impair_loader = DataLoader(impair_ds, batch_size=self.noise_batch_size, shuffle=True)

        # 2) Impair (high LR, 1 epoch)
        for _ in range(max(1, self.impair_epochs)):
            _fit_one_epoch(
                model, impair_loader, device,
                lr=self.impair_lr, weight_decay=self.weight_decay, momentum=self.momentum,
                max_norm=self.max_norm, head_last_layer_lr=self.head_last_layer_lr_impair,
            )

        # 3) Repair (retain only, lower LR, 1 epoch)
        if retain_samples:
            rx = torch.stack([p[0] for p in retain_samples], dim=0)
            ry = torch.tensor([p[1] for p in retain_samples], dtype=torch.long)
            repair_loader = DataLoader(TensorDataset(rx, ry), batch_size=128, shuffle=True)
            for _ in range(max(1, self.repair_epochs)):
                _fit_one_epoch(
                    model, repair_loader, device,
                    lr=self.repair_lr, weight_decay=self.weight_decay, momentum=self.momentum,
                    max_norm=self.max_norm, head_last_layer_lr=self.head_last_layer_lr_repair,
                )

        # logs
        self.logs = {
            "forget_classes": list(map(int, self.forget_classes)),
            "noise_batch_size": self.noise_batch_size,
            "noise_iters": self.noise_iters,
            "noise_lr": self.noise_lr,
            "noise_l2_lambda": self.noise_l2_lambda,
            "noise_copies": self.noise_copies,
            "impair_epochs": self.impair_epochs,
            "repair_epochs": self.repair_epochs,
            "impair_lr": self.impair_lr,
            "repair_lr": self.repair_lr,
            "samples_per_retain_class": self.samples_per_retain_class,
            "num_retain_samples": len(retain_samples),
        }

    def get_model(self) -> nn.Module:
        return self.model

    def report(self) -> Dict[str, Any]:
        return {"hparams": {
                    "forget_classes": self.logs.get("forget_classes", []),
                    "noise_batch_size": self.noise_batch_size,
                    "noise_iters": self.noise_iters,
                    "noise_lr": self.noise_lr,
                    "noise_l2_lambda": self.noise_l2_lambda,
                    "noise_copies": self.noise_copies,
                    "impair_epochs": self.impair_epochs,
                    "repair_epochs": self.repair_epochs,
                    "impair_lr": self.impair_lr,
                    "repair_lr": self.repair_lr,
                    "samples_per_retain_class": self.samples_per_retain_class,
                },
                "logs": self.logs}
