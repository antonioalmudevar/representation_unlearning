# src/helpers/train_utils.py
from typing import Iterable, Dict, Any, List, Tuple
import torch
from torch import nn, optim
from torch.optim import Optimizer

@torch.no_grad()
def evaluate_acc(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)

@torch.no_grad()
def evaluate_acc_by_class(model: nn.Module, loader, device: torch.device, 
                          retain_classes: List[int] = None, 
                          forget_classes: List[int] = None) -> Dict[str, float]:
    """
    Evaluate accuracy separately for retain and forget classes.
    
    Args:
        model: The model to evaluate
        loader: DataLoader containing test samples
        device: Device to run evaluation on
        retain_classes: List of class indices that are retained
        forget_classes: List of class indices that are forgotten
        
    Returns:
        Dictionary with 'test_acc', 'test_acc_retain', and 'test_acc_forget'
    """
    model.eval()
    
    # Overall accuracy
    correct_total, total = 0, 0
    
    # Retain classes accuracy
    correct_retain, total_retain = 0, 0
    
    # Forget classes accuracy
    correct_forget, total_forget = 0, 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        
        # Overall accuracy
        correct_total += (pred == y).sum().item()
        total += y.numel()
        
        # Separate by class type
        if retain_classes is not None:
            retain_mask = torch.zeros_like(y, dtype=torch.bool)
            for cls in retain_classes:
                retain_mask |= (y == cls)
            if retain_mask.any():
                correct_retain += (pred[retain_mask] == y[retain_mask]).sum().item()
                total_retain += retain_mask.sum().item()
        
        if forget_classes is not None:
            forget_mask = torch.zeros_like(y, dtype=torch.bool)
            for cls in forget_classes:
                forget_mask |= (y == cls)
            if forget_mask.any():
                correct_forget += (pred[forget_mask] == y[forget_mask]).sum().item()
                total_forget += forget_mask.sum().item()
    
    result = {"test_acc": correct_total / max(total, 1)}
    
    if retain_classes is not None and total_retain > 0:
        result["test_acc_retain"] = correct_retain / total_retain
    
    if forget_classes is not None and total_forget > 0:
        result["test_acc_forget"] = correct_forget / total_forget
    
    return result

def fit_simple(
    model: nn.Module,
    loader,
    *,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    epochs: int = 1,
    max_norm: float = 0.0,
    trainable_params: List[nn.Parameter] = None,
    desc: str = "train"
) -> Dict[str, Any]:
    model.train()
    history = {"loss": []}
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            if max_norm and max_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params or model.parameters(), max_norm)
            optimizer.step()
            history["loss"].append(float(loss.detach().cpu()))
    return history


def build_optimizer(
    model: nn.Module,
    *,
    name: str = "adamw",
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    cfg: Dict[str, Any] = None,
) -> Optimizer:
    """
    Construct a torch optimizer from YAML config.
    Supported names: adamw, adam, sgd, rmsprop, adagrad
    Extra kwargs are read from `cfg` (e.g., betas, eps, momentum, nesterov, etc.).
    """
    cfg = cfg or {}
    name = str(name).lower()

    # common optional kwargs
    eps = float(cfg.get("eps", 1e-8))
    amsgrad = bool(cfg.get("amsgrad", False))
    momentum = float(cfg.get("momentum", 0.9))
    nesterov = bool(cfg.get("nesterov", False))
    betas = cfg.get("betas", [0.9, 0.999])
    if isinstance(betas, (list, tuple)) and len(betas) == 2:
        betas = (float(betas[0]), float(betas[1]))

    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps, amsgrad=amsgrad)
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps, amsgrad=amsgrad)
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
    if name == "rmsprop":
        alpha = float(cfg.get("alpha", 0.99))
        centered = bool(cfg.get("centered", False))
        return optim.RMSprop(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            momentum=momentum, centered=centered, eps=eps, alpha=alpha
        )
    if name == "adagrad":
        lr_decay = float(cfg.get("lr_decay", 0.0))
        return optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay, lr_decay=lr_decay, eps=eps)

    raise ValueError(f"Unknown optimizer '{name}'. Supported: adamw, adam, sgd, rmsprop, adagrad.")


class WarmupCosineScheduler:
    """
    Linear warmup for warmup_epochs, then CosineAnnealingLR.
    Call .step() once per epoch.
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]

        # Cosine part (starts after warmup)
        self.cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs
        )

        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # linear warmup: lr = base_lr * (epoch / warmup_epochs)
            warmup_factor = self.current_epoch / float(self.warmup_epochs)
            for lr, g in zip(self.base_lrs, self.optimizer.param_groups):
                g['lr'] = lr * warmup_factor
        else:
            # step cosine (but offset)
            self.cosine.step()



def build_scheduler(
    optimizer: Optimizer,
    *,
    name: str,
    epochs: int,
    steps_per_epoch: int = None,
    max_lr: float = None,
    cfg: Dict[str, Any] = None,
):
    """
    Construct a LR scheduler from YAML config.
    `steps_per_epoch` is required only for OneCycle.
    """
    cfg = cfg or {}
    name = str(name).lower()

    cfg = cfg or {}
    name = str(name).lower()

    if name == "none":
        return None

    # --- NEW: warmup for cosine ---
    warmup_epochs = int(cfg.get("warmup_epochs", 0))
    if name == "cosine":
        if warmup_epochs > 0:
            return WarmupCosineScheduler(
                optimizer,
                warmup_epochs=warmup_epochs,
                total_epochs=epochs
            )
        else:
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs
            )

    if name == "onecycle":
        if steps_per_epoch is None:
            raise ValueError("OneCycleLR requires steps_per_epoch.")
        pct_start = float(cfg.get("pct_start", cfg.get("onecycle_pct_start", 0.3)))
        div_factor = float(cfg.get("div_factor", cfg.get("onecycle_div_factor", 25.0)))
        final_div_factor = float(cfg.get("final_div_factor", cfg.get("onecycle_final_div_factor", 1e4)))

        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor
        )

    raise ValueError(f"Unknown scheduler '{name}'.")