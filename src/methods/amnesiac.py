# src/methods/amnesiac_unlearning/method.py
from typing import Any, Dict, List
import copy, time, torch
from torch import nn, optim

from .base import IUnlearningMethod
from ..helpers.registry import register
from ..helpers.train_utils import fit_simple, evaluate_acc

@register("amnesiac_unlearning")
class AmnesiacUnlearning(IUnlearningMethod):
    """
    Amnesiac Unlearning method from "Amnesiac Machine Learning" (Graves et al., 2020).
    
    This method selectively undoes the learning steps that involved the forget data.
    It requires storing parameter updates during the initial training phase, then
    subtracts the updates from batches containing forget data.
    
    Formula: θ_M' = θ_M - Σ Δθ_sb (where sb are batches containing forget data)
    
    Note: This implementation assumes the parameter updates were stored during
    the original training. For practical use, the model should be trained with
    a custom training loop that stores these updates.
    """
    
    def setup(self, model, *, retain_loader, forget_loader, val_loader=None, 
              cfg: Dict[str, Any], device: str = "cuda", 
              stored_updates: Dict = None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._model = model.to(self.device)
        self.retain_loader = retain_loader
        self.forget_loader = forget_loader
        self.val_loader = val_loader
        
        hp = cfg.get("method", {})
        self.finetune_epochs = int(hp.get("finetune_epochs", 1))
        self.lr = float(hp.get("lr", 1e-4))
        self.weight_decay = float(hp.get("weight_decay", 1e-4))
        self.max_norm = float(hp.get("max_norm", 0.0))
        
        # Stored updates from training: {batch_idx: parameter_updates}
        self.stored_updates = stored_updates or {}
        
        self._report = {}

    def run(self) -> None:
        start = time.time()
        
        if self.stored_updates:
            # Apply amnesiac unlearning: subtract stored parameter updates
            self._undo_parameter_updates()
            undo_time = time.time() - start
        else:
            # If no stored updates, this degrades to simple fine-tuning
            undo_time = 0.0
            print("Warning: No stored parameter updates provided. "
                  "Amnesiac unlearning requires parameter updates from training. "
                  "Falling back to fine-tuning only.")
        
        # Optional fine-tuning on retain data to recover performance
        if self.finetune_epochs > 0:
            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(self._model.parameters(), lr=self.lr, 
                                    weight_decay=self.weight_decay)
            
            hist = fit_simple(
                self._model, self.retain_loader, device=self.device, 
                loss_fn=loss_fn, optimizer=optimizer, 
                epochs=self.finetune_epochs, max_norm=self.max_norm
            )
        else:
            hist = {"loss": []}
        
        train_time = time.time() - start
        
        # Evaluate
        acc_val = evaluate_acc(self._model, self.val_loader, self.device) if self.val_loader else None
        acc_forget = evaluate_acc(self._model, self.forget_loader, self.device) if self.forget_loader else None
        
        self._report.update({
            "method": "amnesiac_unlearning",
            "finetune_epochs": self.finetune_epochs,
            "lr": self.lr,
            "undo_time_sec": undo_time,
            "train_time_sec": train_time,
            "num_updates_undone": len(self.stored_updates),
            "train_loss_last": hist["loss"][-1] if hist["loss"] else None,
            "val_acc": acc_val,
            "forget_acc": acc_forget,
        })

    def _undo_parameter_updates(self):
        """
        Undo parameter updates from batches containing forget data.
        Subtracts the stored parameter updates from the current model parameters.
        """
        with torch.no_grad():
            for batch_idx, param_updates in self.stored_updates.items():
                for name, param in self._model.named_parameters():
                    if name in param_updates:
                        # Subtract the update: θ_new = θ_current - Δθ
                        param.data -= param_updates[name].to(self.device)

    def get_model(self):
        return self._model

    def report(self) -> Dict[str, Any]:
        return self._report


# Helper function for training with parameter tracking
def train_with_parameter_tracking(model, train_loader, forget_indices: set, 
                                   device, epochs=10, lr=1e-3, weight_decay=1e-4):
    """
    Training loop that tracks parameter updates for batches containing forget data.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader with dataset that has sample indices
        forget_indices: Set of sample indices to track (forget data)
        device: Device to train on
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay
        
    Returns:
        model: Trained model
        stored_updates: Dictionary mapping batch_idx to parameter updates
    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    stored_updates = {}
    global_batch_idx = 0
    
    model.train()
    for epoch in range(epochs):
        for batch_idx, (x, y, indices) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            # Check if this batch contains any forget data
            batch_has_forget = any(idx.item() in forget_indices for idx in indices)
            
            # Store parameters before update if batch contains forget data
            if batch_has_forget:
                params_before = {
                    name: param.data.clone().cpu() 
                    for name, param in model.named_parameters()
                }
            
            # Forward and backward pass
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            
            # Store parameter updates for forget batches
            if batch_has_forget:
                param_updates = {}
                for name, param in model.named_parameters():
                    # Δθ = θ_after - θ_before
                    delta = param.data.cpu() - params_before[name]
                    param_updates[name] = delta
                stored_updates[global_batch_idx] = param_updates
            
            global_batch_idx += 1
    
    return model, stored_updates