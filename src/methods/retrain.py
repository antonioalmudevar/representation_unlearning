# src/methods/retrain/method.py
from typing import Any, Dict
import copy, time, torch
from torch import nn, optim

from .base import IUnlearningMethod
from ..helpers.registry import register
from ..helpers.train_utils import build_optimizer, build_scheduler, evaluate_acc
from ..models import get_model  # asumiendo tu helper get_model

@register("retrain")
class Retrain(IUnlearningMethod):
    def setup(self, model, *, retain_loader, forget_loader, val_loader=None, cfg: Dict[str, Any], device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # Re-construir modelo desde cero (mismo cfg del modelo)
        self._model = get_model(**cfg["model"]).to(self.device)
        self.retain_loader = retain_loader
        self.forget_loader = forget_loader
        self.val_loader = val_loader
        
        hp = cfg.get("method", {})
        self.epochs = int(hp.get("epochs", 20))
        self.lr = float(hp.get("lr", 1e-3))
        self.weight_decay = float(hp.get("weight_decay", 1e-4))
        self.max_norm = float(hp.get("max_norm", 0.0))
        
        # Optimizer configuration
        self.optimizer_name = hp.get("optimizer", "adamw")
        self.optimizer_cfg = hp.get("optimizer_cfg", {})
        
        # Scheduler configuration
        self.scheduler_name = hp.get("scheduler", "none")
        self.scheduler_cfg = hp.get("scheduler_cfg", {})
        
        self._report = {}

    def run(self) -> None:
        start = time.time()
        loss_fn = nn.CrossEntropyLoss()
        
        # Build optimizer using utility function
        optimizer = build_optimizer(
            self._model,
            name=self.optimizer_name,
            lr=self.lr,
            weight_decay=self.weight_decay,
            cfg=self.optimizer_cfg
        )
        
        # Build scheduler if specified
        scheduler = None
        if self.scheduler_name and self.scheduler_name.lower() != "none":
            scheduler = build_scheduler(
                optimizer,
                name=self.scheduler_name,
                epochs=self.epochs,
                steps_per_epoch=len(self.retain_loader),
                max_lr=self.lr,
                cfg=self.scheduler_cfg
            )
        
        # Training loop with scheduler support
        self._model.train()
        history = {"loss": [], "epoch_loss": []}
        
        print(f"[Retrain] Starting training for {self.epochs} epochs with {self.optimizer_name} optimizer and {self.scheduler_name} scheduler")
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for x, y in self.retain_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self._model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                
                if self.max_norm and self.max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.max_norm)
                
                optimizer.step()
                
                # For OneCycleLR, step per batch
                if scheduler is not None and self.scheduler_name.lower() == "onecycle":
                    scheduler.step()
                
                batch_loss = float(loss.detach().cpu())
                history["loss"].append(batch_loss)
                epoch_loss += batch_loss
                n_batches += 1
            
            avg_epoch_loss = epoch_loss / max(n_batches, 1)
            history["epoch_loss"].append(avg_epoch_loss)
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log every 10 epochs or at the end
            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == self.epochs - 1:
                # Evaluate accuracy on retain and forget sets
                acc_retain = evaluate_acc(self._model, self.retain_loader, self.device)
                acc_forget = evaluate_acc(self._model, self.forget_loader, self.device)
                print(f"[Retrain] Epoch {epoch + 1}/{self.epochs} - Loss: {avg_epoch_loss:.4f}, LR: {current_lr:.6f}, Retain Acc: {acc_retain:.4f}, Forget Acc: {acc_forget:.4f}")
                self._model.train()  # Set back to training mode after evaluation
            
            # For epoch-based schedulers (cosine, step, etc.)
            if scheduler is not None and self.scheduler_name.lower() != "onecycle":
                scheduler.step()
        
        train_time = time.time() - start
        acc_val = evaluate_acc(self._model, self.val_loader, self.device) if self.val_loader else None
        
        print(f"[Retrain] Training completed in {train_time:.2f}s")
        if acc_val is not None:
            print(f"[Retrain] Validation accuracy: {acc_val:.4f}")
        
        self._report.update({
            "method": "retrain",
            "epochs": self.epochs,
            "lr": self.lr,
            "optimizer": self.optimizer_name,
            "scheduler": self.scheduler_name,
            "train_time_sec": train_time,
            "train_loss_last": history["loss"][-1] if history["loss"] else None,
            "train_loss_final_epoch": history["epoch_loss"][-1] if history["epoch_loss"] else None,
            "val_acc": acc_val,
        })
