# src/methods/error_minmax_noise/method.py
from typing import Any, Dict, List
import time
import torch
from torch import nn, optim

from .base import IUnlearningMethod
from ..helpers.registry import register
from ..helpers.train_utils import evaluate_acc


@register("error_minmax_noise")
class ErrorMinMaxNoise(IUnlearningMethod):
    """
    Error Minimization-Maximization Noise method for Zero-Shot Machine Unlearning.
    
    From "Zero-Shot Machine Unlearning" (Chundawat et al., 2023).
    
    This method generates noise matrices for both forget and retain classes:
    - Error-maximizing noise for forget classes: acts as anti-samples to induce forgetting
    - Error-minimizing noise for retain classes: acts as proxy for retain data
    
    The model is then fine-tuned on these noise samples to achieve unlearning
    without access to the original training data.
    """
    
    def setup(self, model, *, retain_loader, forget_loader, val_loader=None,
              cfg: Dict[str, Any], device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._model = model.to(self.device)
        self.retain_loader = retain_loader
        self.forget_loader = forget_loader
        self.val_loader = val_loader
        
        hp = cfg.get("method", {})
        
        # Noise generation parameters
        self.noise_batch_size = int(hp.get("noise_batch_size", 256))
        self.noise_steps = int(hp.get("noise_steps", 400))
        self.noise_lr = float(hp.get("noise_lr", 0.1))
        self.noise_lr_decay = float(hp.get("noise_lr_decay", 0.5))
        self.noise_patience = int(hp.get("noise_patience", 50))
        self.lambda_reg = float(hp.get("lambda_reg", 0.01))
        
        # Model fine-tuning parameters
        self.impair_steps = int(hp.get("impair_steps", 2))
        self.impair_lr = float(hp.get("impair_lr", 0.01))
        self.weight_decay = float(hp.get("weight_decay", 1e-4))
        self.max_norm = float(hp.get("max_norm", 0.0))
        
        # Extract class information from config
        self.num_classes = cfg.get("model", {}).get("num_classes", 10)
        split_protocol = cfg.get("dataset", {}).get("split_protocol", {})
        self.forget_classes = list(split_protocol.get("forget_classes", []))
        self.retain_classes = [c for c in range(self.num_classes) if c not in self.forget_classes]
        
        # Infer input shape from data loaders
        self.input_shape = self._infer_input_shape()
        
        self._report = {}

    def _infer_input_shape(self) -> tuple:
        """Infer input shape from the data loaders."""
        loader = self.retain_loader or self.forget_loader or self.val_loader
        if loader is None:
            raise ValueError("At least one data loader must be provided to infer input shape")
        
        for batch in loader:
            x = batch[0]
            # x shape is [B, C, H, W], we want [C, H, W]
            return tuple(x.shape[1:])
        
        raise ValueError("Data loader is empty, cannot infer input shape")

    def run(self) -> None:
        start = time.time()
        
        # Step 1: Generate error-maximizing noise for forget classes
        forget_noises = self._generate_noise_for_classes(
            classes=self.forget_classes,
            maximize=True
        )
        
        # Step 2: Generate error-minimizing noise for retain classes
        retain_noises = self._generate_noise_for_classes(
            classes=self.retain_classes,
            maximize=False
        )
        
        noise_gen_time = time.time() - start
        
        # Step 3: Combine noises and create dataset
        all_noises = []
        all_labels = []
        all_is_forget = []
        
        for class_idx, noise in forget_noises.items():
            all_noises.append(noise)
            all_labels.append(torch.full((noise.size(0),), class_idx, dtype=torch.long))
            all_is_forget.append(torch.ones(noise.size(0), dtype=torch.bool))
        
        for class_idx, noise in retain_noises.items():
            all_noises.append(noise)
            all_labels.append(torch.full((noise.size(0),), class_idx, dtype=torch.long))
            all_is_forget.append(torch.zeros(noise.size(0), dtype=torch.bool))
        
        if all_noises:
            combined_noises = torch.cat(all_noises, dim=0)
            combined_labels = torch.cat(all_labels, dim=0)
            combined_is_forget = torch.cat(all_is_forget, dim=0)
            
            # Step 4: Fine-tune model with impair step
            hist = self._impair_model(combined_noises, combined_labels, combined_is_forget)
        else:
            hist = {"loss": []}
        
        train_time = time.time() - start
        
        # Evaluate
        acc_val = evaluate_acc(self._model, self.val_loader, self.device) if self.val_loader else None
        acc_forget = evaluate_acc(self._model, self.forget_loader, self.device) if self.forget_loader else None
        acc_retain = evaluate_acc(self._model, self.retain_loader, self.device) if self.retain_loader else None
        
        self._report.update({
            "method": "error_minmax_noise",
            "noise_steps": self.noise_steps,
            "impair_steps": self.impair_steps,
            "noise_gen_time_sec": noise_gen_time,
            "train_time_sec": train_time,
            "train_loss_last": hist["loss"][-1] if hist["loss"] else None,
            "val_acc": acc_val,
            "forget_acc": acc_forget,
            "retain_acc": acc_retain,
        })

    def _generate_noise_for_classes(self, classes: List[int], 
                                     maximize: bool) -> Dict[int, torch.Tensor]:
        """
        Generate noise matrices for a set of classes.
        
        Args:
            classes: List of class indices
            maximize: If True, maximize loss (forget classes); else minimize (retain classes)
        """
        noises = {}
        loss_fn = nn.CrossEntropyLoss()
        
        self._model.eval()
        
        for class_idx in classes:
            # Initialize noise from standard normal distribution
            noise = torch.randn(
                self.noise_batch_size, 
                *self.input_shape, 
                device=self.device,
                requires_grad=True
            )
            
            target = torch.full(
                (self.noise_batch_size,), 
                class_idx, 
                dtype=torch.long, 
                device=self.device
            )
            
            optimizer = optim.SGD([noise], lr=self.noise_lr)
            best_loss = float('-inf') if maximize else float('inf')
            patience_counter = 0
            current_lr = self.noise_lr
            
            for step in range(self.noise_steps):
                optimizer.zero_grad()
                
                outputs = self._model(noise)
                cls_loss = loss_fn(outputs, target)
                reg_loss = self.lambda_reg * torch.norm(noise, p=2)
                
                if maximize:
                    # Maximize classification loss for forget classes
                    total_loss = -cls_loss + reg_loss
                else:
                    # Minimize classification loss for retain classes
                    total_loss = cls_loss + reg_loss
                
                total_loss.backward()
                optimizer.step()
                
                # Manual learning rate decay on plateau
                improved = (cls_loss.item() > best_loss) if maximize else (cls_loss.item() < best_loss)
                if improved:
                    best_loss = cls_loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.noise_patience:
                        current_lr *= self.noise_lr_decay
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = current_lr
                        patience_counter = 0
            
            noises[class_idx] = noise.detach().cpu()
        
        return noises

    def _impair_model(self, noises: torch.Tensor, labels: torch.Tensor, 
                      is_forget: torch.Tensor) -> Dict[str, List[float]]:
        """
        Fine-tune the model on noise samples to achieve unlearning.
        
        For forget class noise: maximize loss (make model predict wrongly)
        For retain class noise: minimize loss (make model predict correctly)
        """
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        optimizer = optim.SGD(
            self._model.parameters(),
            lr=self.impair_lr,
            weight_decay=self.weight_decay,
            momentum=0.9
        )
        
        num_samples = noises.size(0)
        
        history = {"loss": []}
        self._model.train()
        
        for epoch in range(self.impair_steps):
            # Shuffle data each epoch
            perm = torch.randperm(num_samples)
            noises_shuffled = noises[perm]
            labels_shuffled = labels[perm]
            is_forget_shuffled = is_forget[perm]
            
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, num_samples, self.noise_batch_size):
                batch_noise = noises_shuffled[i:i+self.noise_batch_size].to(self.device)
                batch_labels = labels_shuffled[i:i+self.noise_batch_size].to(self.device)
                batch_is_forget = is_forget_shuffled[i:i+self.noise_batch_size].to(self.device)
                
                optimizer.zero_grad()
                outputs = self._model(batch_noise)
                
                # Compute per-sample loss
                per_sample_loss = loss_fn(outputs, batch_labels)
                
                # For forget samples: negate the loss (maximize it)
                # For retain samples: keep the loss as is (minimize it)
                signs = torch.where(batch_is_forget, -torch.ones_like(per_sample_loss), 
                                   torch.ones_like(per_sample_loss))
                weighted_loss = (signs * per_sample_loss).mean()
                
                weighted_loss.backward()
                
                if self.max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.max_norm)
                
                optimizer.step()
                
                epoch_loss += weighted_loss.item()
                num_batches += 1
            
            history["loss"].append(epoch_loss / max(num_batches, 1))
        
        return history

    def get_model(self):
        return self._model

    def report(self) -> Dict[str, Any]:
        return self._report