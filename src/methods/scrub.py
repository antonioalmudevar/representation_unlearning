from copy import deepcopy
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _unpack_batch(batch, device):
    # Supports (x, y) or (x, ..., y)
    if isinstance(batch, (list, tuple)):
        x, y = batch[0], batch[-1]
    else:
        x, y = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def _kl_teacher_student(s_logits: torch.Tensor, t_logits: torch.Tensor) -> torch.Tensor:
    """KL divergence from teacher to student using softmax probabilities."""
    log_p_s = F.log_softmax(s_logits, dim=1)
    p_t = F.softmax(t_logits, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean")


def _build_optimizer(params, name: str, lr: float, weight_decay: float, momentum: float = 0.9):
    name = (name or "adam").lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=True)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)


@torch.no_grad()
def _acc(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute accuracy on a dataset."""
    model.eval()
    correct = total = 0
    for batch in loader:
        x, y = _unpack_batch(batch, device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


class SCRUB:
    """
    SCRUB: SCalable Remembering and Unlearning unBound
    
    Teacher-student unlearning with alternating max-min optimization.
    Optional rewind (SCRUB+R) for privacy applications.
    
    Based on: "Towards Unbounded Machine Unlearning" (Kurmanji et al., NeurIPS 2023)
    
    Public API:
      - setup(model, retain_loader, forget_loader, val_loader, cfg, device)
      - run()
      - get_model()
      - report()
    """

    def setup(
        self,
        model: nn.Module,
        *,
        retain_loader: DataLoader,
        forget_loader: DataLoader,
        val_loader: Optional[DataLoader],
        cfg: Dict[str, Any],
        device: str = "cuda",
    ) -> None:
        """
        Initialize SCRUB method.
        
        Args:
            model: Neural network to unlearn from
            retain_loader: DataLoader for retain set
            forget_loader: DataLoader for forget set
            val_loader: DataLoader for validation (used for SCRUB+R)
            cfg: Full config dict containing method parameters
            device: Device string ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.retain_loader = retain_loader
        self.forget_loader = forget_loader
        self.val_loader = val_loader
        self.cfg = cfg

        m = dict(cfg.get("method", {}))
        
        # Optimization parameters
        self.optimizer_name = str(m.get("optimizer", "adam")).lower()
        self.lr = float(m.get("lr", 5e-4))
        self.weight_decay = float(m.get("weight_decay", 5e-4))
        self.momentum = float(m.get("momentum", 0.9))
        self.lr_decay_after = m.get("lr_decay_after", None)

        # Alternating training parameters
        self.max_steps = int(m.get("max_steps", 3))
        self.min_steps = int(m.get("min_steps", 3))
        self.final_min_steps = int(m.get("final_min_steps", 0))
        self.alpha = float(m.get("alpha", 1.0))   # KL weight on retain
        self.gamma = float(m.get("gamma", 1.0))   # CE weight on retain
        self.clip_grad_norm = m.get("clip_grad_norm", None)
        self.rewind = bool(m.get("rewind", False))  # SCRUB+R

        # Teacher = frozen copy of initial model
        print("[SCRUB] Creating teacher (frozen copy of model)")
        self.teacher = deepcopy(self.model).eval().to(self.device)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # Optimizer
        self.optimizer = _build_optimizer(
            self.model.parameters(), 
            self.optimizer_name,
            self.lr, 
            self.weight_decay, 
            self.momentum
        )
        
        # Checkpointing for rewind
        self._checkpoints: List[Dict[str, torch.Tensor]] = []
        self._forget_errs: List[float] = []
        self._logs: Dict[str, Any] = {
            "max_loss": [], 
            "min_loss": [], 
            "lr": [],
            "forget_err": []
        }
        
        print(f"[SCRUB] Initialized with max_steps={self.max_steps}, min_steps={self.min_steps}")
        print(f"[SCRUB] alpha={self.alpha}, gamma={self.gamma}, rewind={self.rewind}")

    def _maybe_decay_lr(self, epoch_idx: int):
        """Apply learning rate decay if configured."""
        if self.lr_decay_after is not None and epoch_idx == int(self.lr_decay_after):
            for pg in self.optimizer.param_groups:
                pg["lr"] = pg["lr"] * 0.1
            print(f"[SCRUB] LR decayed to {pg['lr']:.2e} at epoch {epoch_idx}")

    def _epoch_max(self) -> float:
        """
        MAX step: Push student away from teacher on forget set.
        Maximizes KL divergence (negative loss to ascend).
        """
        self.model.train()
        self.teacher.eval()
        total, n = 0.0, 0
        for batch in self.forget_loader:
            x, _ = _unpack_batch(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            s_logits = self.model(x)
            with torch.no_grad():
                t_logits = self.teacher(x)
            loss = -_kl_teacher_student(s_logits, t_logits)  # negative = ascend KL
            loss.backward()
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.clip_grad_norm))
            self.optimizer.step()
            total += float(loss.detach().cpu())
            n += 1
        return total / max(n, 1)

    def _epoch_min(self) -> float:
        """
        MIN step: Keep student close to teacher on retain set.
        Minimizes: alpha * KL + gamma * CE
        """
        self.model.train()
        self.teacher.eval()
        total, n = 0.0, 0
        for batch in self.retain_loader:
            x, y = _unpack_batch(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            s_logits = self.model(x)
            with torch.no_grad():
                t_logits = self.teacher(x)
            kl = _kl_teacher_student(s_logits, t_logits)
            ce = F.cross_entropy(s_logits, y)
            loss = self.alpha * kl + self.gamma * ce
            loss.backward()
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.clip_grad_norm))
            self.optimizer.step()
            total += float(loss.detach().cpu())
            n += 1
        return total / max(n, 1)

    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint and compute forget error for potential rewind."""
        with torch.no_grad():
            f_err = 1.0 - _acc(self.model, self.forget_loader, self.device)
        self._forget_errs.append(f_err)
        self._checkpoints.append({
            k: v.detach().cpu().clone() 
            for k, v in self.model.state_dict().items()
        })
        self._logs["forget_err"].append(f_err)
        return f_err

    def run(self) -> None:
        """
        Execute SCRUB unlearning with alternating max-min optimization.
        
        Training schedule:
        1. Alternate: MAX epoch, then MIN epoch (repeat max_steps times)
        2. Additional MIN-only epochs (min_steps - max_steps)
        3. Optional final MIN-only tail (final_min_steps)
        4. If rewind=True: select best checkpoint based on validation error
        """
        epoch = 0
        print(f"\n[SCRUB] Starting alternating optimization")
        print(f"[SCRUB] Phase 1: {self.max_steps} alternating MAX-MIN cycles")
        
        # Phase 1: Alternating MAX and MIN
        for i in range(self.max_steps):
            # MAX epoch
            self._maybe_decay_lr(epoch)
            max_loss = self._epoch_max()
            self._logs["max_loss"].append(max_loss)
            self._logs["min_loss"].append(None)
            self._logs["lr"].append(self.optimizer.param_groups[0]["lr"])
            f_err = self._save_checkpoint(epoch)
            print(f"[SCRUB] Epoch {epoch:2d} (MAX): loss={max_loss:+.4f}, forget_err={f_err:.4f}")
            epoch += 1
            
            # MIN epoch
            self._maybe_decay_lr(epoch)
            min_loss = self._epoch_min()
            self._logs["max_loss"].append(None)
            self._logs["min_loss"].append(min_loss)
            self._logs["lr"].append(self.optimizer.param_groups[0]["lr"])
            f_err = self._save_checkpoint(epoch)
            print(f"[SCRUB] Epoch {epoch:2d} (MIN): loss={min_loss:+.4f}, forget_err={f_err:.4f}")
            epoch += 1

        # Phase 2: Additional MIN-only epochs
        remaining_min = self.min_steps - self.max_steps
        if remaining_min > 0:
            print(f"\n[SCRUB] Phase 2: {remaining_min} additional MIN-only epochs")
            for i in range(remaining_min):
                self._maybe_decay_lr(epoch)
                min_loss = self._epoch_min()
                self._logs["max_loss"].append(None)
                self._logs["min_loss"].append(min_loss)
                self._logs["lr"].append(self.optimizer.param_groups[0]["lr"])
                f_err = self._save_checkpoint(epoch)
                print(f"[SCRUB] Epoch {epoch:2d} (MIN): loss={min_loss:+.4f}, forget_err={f_err:.4f}")
                epoch += 1

        # Phase 3: Optional final MIN-only tail
        if self.final_min_steps > 0:
            print(f"\n[SCRUB] Phase 3: {self.final_min_steps} final MIN-only epochs")
            for t in range(self.final_min_steps):
                self._maybe_decay_lr(epoch)
                min_loss = self._epoch_min()
                self._logs["max_loss"].append(None)
                self._logs["min_loss"].append(min_loss)
                self._logs["lr"].append(self.optimizer.param_groups[0]["lr"])
                f_err = self._save_checkpoint(epoch)
                print(f"[SCRUB] Epoch {epoch:2d} (MIN): loss={min_loss:+.4f}, forget_err={f_err:.4f}")
                epoch += 1

        print(f"\n[SCRUB] Training complete: {epoch} total epochs")
        print(f"[SCRUB] Final forget error: {self._forget_errs[-1]:.4f}")

        # SCRUB+R: Rewind to best checkpoint
        if self.rewind and self.val_loader is not None and len(self._checkpoints) > 0:
            print("\n[SCRUB+R] Selecting best checkpoint via rewind...")
            
            # Compute reference error on validation set (forget-valid distribution)
            ref_err = None
            try:
                ref_err = 1.0 - _acc(self.model, self.val_loader, self.device)
                print(f"[SCRUB+R] Reference error (validation): {ref_err:.4f}")
            except Exception as e:
                print(f"[SCRUB+R] Could not compute validation error: {e}")
                ref_err = self._forget_errs[-1]
                print(f"[SCRUB+R] Using final forget error as reference: {ref_err:.4f}")
            
            # Find checkpoint with forget error closest to reference
            best_idx, best_gap = None, float("inf")
            for i, f_err in enumerate(self._forget_errs):
                gap = abs(f_err - ref_err)
                if gap < best_gap:
                    best_gap, best_idx = gap, i
            
            if best_idx is not None:
                print(f"[SCRUB+R] Best checkpoint: epoch {best_idx}, forget_err={self._forget_errs[best_idx]:.4f}")
                print(f"[SCRUB+R] Gap from reference: {best_gap:.4f}")
                self.model.load_state_dict(self._checkpoints[best_idx], strict=True)
                print("[SCRUB+R] Model rewound to best checkpoint")
            else:
                print("[SCRUB+R] No valid checkpoint found, keeping final model")

    def get_model(self) -> nn.Module:
        """Return the unlearned model."""
        return self.model

    def report(self) -> Dict[str, Any]:
        """
        Return training report with hyperparameters and logs.
        
        Returns:
            Dict containing:
                - hparams: All hyperparameters used
                - logs: Training losses, LR schedule, forget errors
        """
        return {
            "hparams": {
                "optimizer": self.optimizer_name,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "momentum": self.momentum,
                "max_steps": self.max_steps,
                "min_steps": self.min_steps,
                "final_min_steps": self.final_min_steps,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "lr_decay_after": self.lr_decay_after,
                "clip_grad_norm": self.clip_grad_norm,
                "rewind": self.rewind,
            },
            "logs": self._logs,
            "final_forget_error": self._forget_errs[-1] if self._forget_errs else None,
        }