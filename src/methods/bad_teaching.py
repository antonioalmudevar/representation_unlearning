# src/methods/bad_teaching/method.py
from typing import Any, Dict
import copy, time, torch
from torch import nn, optim
import torch.nn.functional as F

from .base import IUnlearningMethod
from ..helpers.registry import register
from ..helpers.train_utils import evaluate_acc

@register("bad_teaching")
class BadTeaching(IUnlearningMethod):
    """
    Bad Teaching Unlearning from "Can Bad Teaching Induce Forgetting?" (Chundawat et al., 2023).
    
    Uses competent (original model) and incompetent (random) teachers in a student-teacher
    framework with KL-divergence loss:
    L = (1 - lu) * KL(Ts || S) + lu * KL(Td || S)
    where lu=0 for retain, lu=1 for forget
    """
    
    def setup(self, model, *, retain_loader, forget_loader, val_loader=None, 
              cfg: Dict[str, Any], device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Student: copy of original model (not the original itself!)
        self._model = copy.deepcopy(model).to(self.device)
        
        # Competent teacher: frozen copy of original model
        self.teacher_competent = copy.deepcopy(model).to(self.device)
        self.teacher_competent.eval()
        for p in self.teacher_competent.parameters():
            p.requires_grad = False
        
        self.retain_loader = retain_loader
        self.forget_loader = forget_loader
        self.val_loader = val_loader
        
        hp = cfg.get("method", {})
        self.epochs = int(hp.get("epochs", 1))
        self.lr = float(hp.get("lr", 1e-4))
        self.weight_decay = float(hp.get("weight_decay", 5e-4))
        self.momentum = float(hp.get("momentum", 0.9))
        self.max_norm = float(hp.get("max_norm", 0.0))
        self.temperature = float(hp.get("temperature", 1.0))
        self.retain_fraction = float(hp.get("retain_fraction", 0.3))
        self.num_classes = cfg["model"]["num_classes"]
        self.incompetent_type = str(hp.get("incompetent_type", "random_init"))
        
        # Incompetent teacher: random initialization
        if self.incompetent_type == "random_init":
            self.teacher_incompetent = copy.deepcopy(model).to(self.device)
            self._reset_weights(self.teacher_incompetent)
            self.teacher_incompetent.eval()
            for p in self.teacher_incompetent.parameters():
                p.requires_grad = False
        else:  # random_predictor
            self.teacher_incompetent = self._RandomTeacher(self.num_classes).to(self.device)
        
        self._report = {}

    def run(self) -> None:
        start = time.time()
        
        # Create combined dataset with unlearning labels
        combined_loader = self._create_combined_loader()
        
        # Optimizer
        optimizer = optim.SGD(self._model.parameters(), lr=self.lr, 
                             momentum=self.momentum, weight_decay=self.weight_decay)
        
        # Training loop
        self._model.train()
        epoch_losses = []
        
        print(f"[BadTeaching] Starting training for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            total_loss = 0.0
            n_batches = 0
            
            for x, y, lu in combined_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                lu = lu.to(self.device).float()
                
                # Forward pass
                logits_student = self._model(x)
                
                with torch.no_grad():
                    logits_competent = self.teacher_competent(x)
                    logits_incompetent = self.teacher_incompetent(x)
                
                # Compute KL divergences with temperature
                T = self.temperature
                
                # KL(Ts || S) for retain samples
                log_p_student = F.log_softmax(logits_student / T, dim=1)
                p_competent = F.softmax(logits_competent / T, dim=1)
                p_incompetent = F.softmax(logits_incompetent / T, dim=1)
                
                # Per-sample KL divergences
                kl_competent = (p_competent * (p_competent.clamp(min=1e-12).log() - log_p_student)).sum(dim=1) * (T * T)
                kl_incompetent = (p_incompetent * (p_incompetent.clamp(min=1e-12).log() - log_p_student)).sum(dim=1) * (T * T)
                
                # Combined loss: (1-lu)*KL(Ts||S) + lu*KL(Td||S)
                loss_per_sample = (1.0 - lu) * kl_competent + lu * kl_incompetent
                loss = loss_per_sample.mean()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                if self.max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.max_norm)
                
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / max(n_batches, 1)
            epoch_losses.append(avg_loss)
            
            # Log progress with retain and forget accuracies
            self._model.eval()
            with torch.no_grad():
                acc_retain = evaluate_acc(self._model, self.retain_loader, self.device)
                acc_forget = evaluate_acc(self._model, self.forget_loader, self.device)
            self._model.train()
            
            print(f"Epoch {epoch+1}/{self.epochs} | Loss: {avg_loss:.4f} | Retain Acc: {acc_retain:.4f} | Forget Acc: {acc_forget:.4f}")
        
        train_time = time.time() - start
        
        # Evaluate
        acc_val = evaluate_acc(self._model, self.val_loader, self.device) if self.val_loader else None
        acc_forget = evaluate_acc(self._model, self.forget_loader, self.device) if self.forget_loader else None
        
        self._report.update({
            "method": "bad_teaching",
            "epochs": self.epochs,
            "lr": self.lr,
            "retain_fraction": self.retain_fraction,
            "temperature": self.temperature,
            "train_time_sec": train_time,
            "train_loss_last": epoch_losses[-1] if epoch_losses else None,
            "val_acc": acc_val,
            "forget_acc": acc_forget,
        })

    def _create_combined_loader(self):
        """Create DataLoader with retain subset + forget data, each with unlearning label"""
        from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Subset
        import numpy as np
        
        # Get retain subset
        retain_ds = self.retain_loader.dataset
        n_retain = len(retain_ds)
        n_subset = max(1, int(n_retain * self.retain_fraction))
        indices = np.random.choice(n_retain, n_subset, replace=False)
        retain_subset = Subset(retain_ds, indices)
        
        # Collect retain data with lu=0
        retain_data, retain_labels = [], []
        for x, y in DataLoader(retain_subset, batch_size=256, shuffle=False):
            retain_data.append(x)
            retain_labels.append(y)
        retain_data = torch.cat(retain_data, dim=0)
        retain_labels = torch.cat(retain_labels, dim=0)
        retain_lu = torch.zeros(len(retain_data), dtype=torch.long)
        
        # Collect forget data with lu=1
        forget_data, forget_labels = [], []
        for x, y in self.forget_loader:
            forget_data.append(x)
            forget_labels.append(y)
        forget_data = torch.cat(forget_data, dim=0)
        forget_labels = torch.cat(forget_labels, dim=0)
        forget_lu = torch.ones(len(forget_data), dtype=torch.long)
        
        # Combine
        all_data = torch.cat([retain_data, forget_data], dim=0)
        all_labels = torch.cat([retain_labels, forget_labels], dim=0)
        all_lu = torch.cat([retain_lu, forget_lu], dim=0)
        
        dataset = TensorDataset(all_data, all_labels, all_lu)
        return DataLoader(dataset, batch_size=self.retain_loader.batch_size, 
                         shuffle=True, num_workers=0)

    @staticmethod
    def _reset_weights(model):
        """Reset model weights to random initialization"""
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    class _RandomTeacher(nn.Module):
        """Random predictor that outputs uniform + noise"""
        def __init__(self, num_classes, noise_std=0.01):
            super().__init__()
            self.num_classes = num_classes
            self.noise_std = noise_std
        
        @torch.no_grad()
        def forward(self, x):
            b = x.shape[0]
            logits = torch.zeros(b, self.num_classes, device=x.device)
            if self.noise_std > 0:
                logits += torch.randn_like(logits) * self.noise_std
            return logits

    def get_model(self):
        return self._model

    def report(self) -> Dict[str, Any]:
        return self._report