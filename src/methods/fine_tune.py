from typing import Any, Dict
import time, torch
from torch import nn, optim

from .base import IUnlearningMethod
from ..helpers.registry import register
from ..helpers.train_utils import fit_simple, evaluate_acc

@register("fine_tune")
class FineTune(IUnlearningMethod):
    """
    Fine-tuning completo sobre retained.
    Útil como baseline fuerte y barato frente a retrain-from-scratch.
    """
    def setup(self, model, *, retain_loader, forget_loader, val_loader=None, cfg: Dict[str, Any], device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._model = model.to(self.device)
        self.retain_loader = retain_loader
        self.val_loader = val_loader

        hp = cfg.get("method", {})
        self.epochs = int(hp.get("epochs", 10))
        self.lr = float(hp.get("lr", 5e-4))
        self.weight_decay = float(hp.get("weight_decay", 1e-4))
        self.max_norm = float(hp.get("max_norm", 0.0))

    def run(self) -> None:
        start = time.time()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        hist = fit_simple(
            self._model, self.retain_loader, device=self.device, loss_fn=loss_fn,
            optimizer=optimizer, epochs=self.epochs, max_norm=self.max_norm, desc="fine_tune"
        )
        train_time = time.time() - start
        acc_val = evaluate_acc(self._model, self.val_loader, self.device) if self.val_loader else None
        self._report.update({
            "method": "fine_tune",
            "epochs": self.epochs,
            "lr": self.lr,
            "train_time_sec": train_time,
            "train_loss_last": hist["loss"][-1] if hist["loss"] else None,
            "val_acc": acc_val,
        })
