# src/methods/langevin_unlearning/method.py
from typing import Any, Dict
import copy, time
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

from .base import IUnlearningMethod
from ..helpers.registry import register
from ..helpers.train_utils import evaluate_acc


@register("langevin_unlearning")
class LangevinUnlearning(IUnlearningMethod):
    """
    Langevin Unlearning (Chien et al., NeurIPS 2024).
    "Langevin Unlearning: A New Perspective of Noisy Gradient Descent
    for Machine Unlearning"  arXiv:2401.10371

    Unlearning algorithm: fine-tune on the retain set (D') using
    Projected Noisy Gradient Descent (PNGD):

        y_{k+1} = Π_CR[ y_k - η∇f_{D'}(y_k) + sqrt(2ησ²) * W_k ]

    where W_k ~ N(0, I) is i.i.d. Gaussian noise and Π_CR is an
    orthogonal projection onto the L2 ball of radius R. Per-sample
    gradient clipping (clip norm M) is applied before the update,
    matching the DP-SGD interpretation in the paper.

    Config keys (under method:):
        epochs          (int,   default 10)   unlearning epochs
        lr              (float, default 0.01) step size η
        momentum        (float, default 0.0)  SGD momentum (0 = pure GD)
        weight_decay    (float, default 0.0)
        sigma           (float, default 1.0)  noise std σ for PNGD noise
                                              injected as sqrt(2ησ²)*N(0,I)
        max_grad_norm   (float, default 1.0)  per-sample gradient clip norm M
        projection_r    (float, default 0.0)  L2 ball radius R for projection
                                              (0 = disabled, as in practice
                                              for non-convex deep nets)
    """

    def setup(self, model, *, retain_loader, forget_loader, val_loader=None,
              cfg: Dict[str, Any], device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self._model = copy.deepcopy(model).to(self.device)
        self.retain_loader = retain_loader
        self.forget_loader = forget_loader
        self.val_loader = val_loader

        hp = cfg.get("method", {})
        self.epochs        = int(hp.get("epochs", 10))
        self.lr            = float(hp.get("lr", 0.01))
        self.momentum      = float(hp.get("momentum", 0.0))
        self.weight_decay  = float(hp.get("weight_decay", 0.0))
        self.sigma         = float(hp.get("sigma", 1.0))
        self.max_grad_norm = float(hp.get("max_grad_norm", 1.0))
        self.projection_r  = float(hp.get("projection_r", 0.0))

        self._report = {}

    # ------------------------------------------------------------------
    def run(self) -> None:
        start = time.time()

        optimizer = optim.SGD(
            self._model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        epoch_losses = []

        # Noise scale: sqrt(2 * η * σ²) per the PNGD update rule
        noise_scale = (2.0 * self.lr * self.sigma ** 2) ** 0.5

        print(f"[LangevinUnlearning] Fine-tuning on retain set for {self.epochs} epochs "
              f"(σ={self.sigma}, clip={self.max_grad_norm}, noise_scale={noise_scale:.4f})...")

        for epoch in range(self.epochs):
            self._model.train()
            total_loss, n_batches = 0.0, 0

            for x, y in self.retain_loader:
                x, y = x.to(self.device), y.to(self.device)

                # ---- Forward + per-sample gradient clipping ----------
                # Accumulate gradients with clipping (mimics DP-SGD / PNGD)
                optimizer.zero_grad()
                loss = criterion(self._model(x), y)
                loss.backward()

                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        self._model.parameters(), self.max_grad_norm)

                optimizer.step()

                # ---- Inject Langevin noise (the W_k term in PNGD) ---
                with torch.no_grad():
                    for p in self._model.parameters():
                        if p.requires_grad:
                            noise = torch.randn_like(p) * noise_scale
                            p.add_(noise)

                # ---- Optional projection onto L2 ball ----------------
                if self.projection_r > 0:
                    self._project_parameters(self.projection_r)

                total_loss += loss.item()
                n_batches += 1

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
            "method":          "langevin_unlearning",
            "epochs":          self.epochs,
            "lr":              self.lr,
            "sigma":           self.sigma,
            "max_grad_norm":   self.max_grad_norm,
            "projection_r":    self.projection_r,
            "noise_scale":     noise_scale,
            "train_time_sec":  train_time,
            "train_loss_last": epoch_losses[-1] if epoch_losses else None,
            "val_acc":         acc_val,
            "forget_acc":      acc_forget,
        })

    # ------------------------------------------------------------------
    def _project_parameters(self, radius: float) -> None:
        """Project all parameters onto the L2 ball of given radius."""
        with torch.no_grad():
            all_params = torch.cat([p.flatten() for p in self._model.parameters()])
            norm = all_params.norm()
            if norm > radius:
                scale = radius / norm
                for p in self._model.parameters():
                    p.mul_(scale)

    # ------------------------------------------------------------------
    def get_model(self):
        return self._model

    def report(self) -> Dict[str, Any]:
        return self._report