# src/methods/boundary_shrink/method.py
from typing import Any, Dict, List, Tuple
import copy, time, torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .base import IUnlearningMethod
from ..helpers.registry import register
from ..helpers.train_utils import evaluate_acc

@register("boundary_shrink")
class BoundaryShrink(IUnlearningMethod):
    """
    Boundary Shrink from "Boundary Unlearning" (Chen et al., 2023).
    
    Uses FGSM to find nearest-but-incorrect (neighbor) labels, then fine-tunes
    on the ORIGINAL forget samples with these neighbor labels to shrink the
    decision boundary of the forget class.
    
    Key equations from paper:
    - Eq. 2: x' = x + ε·sign(∇_x L(x, y; w0))  [cross sample generation]
    - Eq. 3: y_nbi ← softmax(f_w0(x'))         [neighbor label prediction]
    - Eq. 4: w' = argmin Σ L(x, y_nbi, w0)     [fine-tune on ORIGINAL x]
    """
    
    def setup(self, model, *, retain_loader, forget_loader, val_loader=None, 
              cfg: Dict[str, Any], device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Teacher: frozen copy for generating neighbor labels
        self.teacher = copy.deepcopy(model).to(self.device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        
        # Student: will be fine-tuned
        self._model = copy.deepcopy(model).to(self.device)
        
        self.retain_loader = retain_loader
        self.forget_loader = forget_loader
        self.val_loader = val_loader
        
        hp = cfg.get("method", {})
        self.eps = float(hp.get("eps", 0.25))
        self.clamp_min = float(hp.get("clamp_min", 0.0))
        self.clamp_max = float(hp.get("clamp_max", 1.0))
        self.ft_epochs = int(hp.get("ft_epochs", 10))
        self.ft_lr = float(hp.get("ft_lr", 1e-5))
        self.ft_momentum = float(hp.get("ft_momentum", 0.9))
        self.ft_weight_decay = float(hp.get("ft_weight_decay", 0.0))
        self.batch_size = int(hp.get("batch_size", 128))
        self.max_norm = float(hp.get("max_norm", 0.0))
        
        self._report = {}

    def run(self) -> None:
        start = time.time()
        
        # Step 1: Generate cross samples and find neighbor labels
        x_forget, y_nbi = self._find_neighbor_labels()
        
        neighbor_time = time.time() - start
        
        # Step 2: Fine-tune on ORIGINAL forget samples with neighbor labels
        remap_loader = self._create_remap_loader(x_forget, y_nbi)
        
        optimizer = optim.SGD(
            self._model.parameters(),
            lr=self.ft_lr,
            momentum=self.ft_momentum,
            weight_decay=self.ft_weight_decay
        )
        
        self._model.train()
        loss_fn = nn.CrossEntropyLoss()
        
        for epoch in range(self.ft_epochs):
            for x, y_nb in remap_loader:
                x = x.to(self.device)
                y_nb = y_nb.to(self.device)
                
                optimizer.zero_grad()
                logits = self._model(x)
                loss = loss_fn(logits, y_nb)
                loss.backward()
                
                if self.max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.max_norm)
                
                optimizer.step()
        
        train_time = time.time() - start
        
        # Evaluate
        acc_val = evaluate_acc(self._model, self.val_loader, self.device) if self.val_loader else None
        acc_forget = evaluate_acc(self._model, self.forget_loader, self.device) if self.forget_loader else None
        
        self._report.update({
            "method": "boundary_shrink",
            "eps": self.eps,
            "ft_epochs": self.ft_epochs,
            "ft_lr": self.ft_lr,
            "neighbor_search_time_sec": neighbor_time,
            "total_time_sec": train_time,
            "val_acc": acc_val,
            "forget_acc": acc_forget,
        })

    def _find_neighbor_labels(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate cross samples via FGSM and predict neighbor labels.
        Returns: (original_x_forget, y_nbi)
        """
        xs_original = []
        ys_nbi = []
        
        for batch in self.forget_loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[-1]
            else:
                x, y = batch
            
            x = x.to(self.device).float()
            y = y.to(self.device)
            
            # Generate cross samples using FGSM (Eq. 2)
            x_cross = self._fgsm_cross_sample(x, y)
            
            # Predict neighbor labels on cross samples (Eq. 3)
            with torch.no_grad():
                logits_cross = self.teacher(x_cross)
                y_nb = self._get_neighbor_label(logits_cross, y)
            
            xs_original.append(x.detach().cpu())
            ys_nbi.append(y_nb.cpu())
        
        x_all = torch.cat(xs_original, dim=0)
        y_all = torch.cat(ys_nbi, dim=0)
        
        return x_all, y_all

    def _fgsm_cross_sample(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Generate cross sample: x' = x + ε·sign(∇_x L(x, y; w0))
        """
        x = x.clone().detach().requires_grad_(True)
        
        # Forward pass
        logits = self.teacher(x)
        loss = F.cross_entropy(logits, y)
        
        # Backward to get gradients w.r.t. input
        self.teacher.zero_grad()
        loss.backward()
        
        # FGSM step
        x_cross = x + self.eps * x.grad.sign()
        x_cross = x_cross.clamp(min=self.clamp_min, max=self.clamp_max).detach()
        
        return x_cross

    def _get_neighbor_label(self, logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Get nearest-but-incorrect label.
        If top-1 prediction equals true label, use top-2.
        Otherwise, use top-1.
        """
        # Get top-2 predictions
        top2_indices = torch.topk(logits, k=2, dim=1).indices
        top1 = top2_indices[:, 0]
        top2 = top2_indices[:, 1]
        
        # If top-1 == true label, use top-2 as neighbor
        # Otherwise, use top-1
        same_as_true = (top1 == y_true)
        y_nbi = torch.where(same_as_true, top2, top1)
        
        return y_nbi

    def _create_remap_loader(self, x: torch.Tensor, y_nbi: torch.Tensor) -> DataLoader:
        """Create DataLoader with original forget samples and neighbor labels"""
        dataset = TensorDataset(x, y_nbi)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

    def get_model(self):
        return self._model

    def report(self) -> Dict[str, Any]:
        return self._report