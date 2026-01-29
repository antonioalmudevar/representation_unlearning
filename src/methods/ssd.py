# src/methods/ssd.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Union
from copy import deepcopy

def _unpack_batch(batch, device):
    if isinstance(batch, (list, tuple)):
        x, y = batch[0], batch[-1]
    else:
        x, y = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

class SSD:
    """
    Selective Synaptic Dampening (SSD)
    
    A zero-shot unlearning method that modifies weights based on the ratio 
    of Fisher Information between the forget set and the full training set.
    
    Paper: "Selective Synaptic Dampening" (Foster et al., 2024)
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
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.retain_loader = retain_loader
        self.forget_loader = forget_loader
        self.val_loader = val_loader
        self.cfg = cfg

        # Extract hyperparameters
        m = dict(cfg.get("method", {}))
        
        # Lambda in paper: Determines how aggressively to dampen
        self.dampening_constant = float(m.get("dampening_constant", 1.0)) 
        
        # Alpha in paper: Selection threshold
        self.selection_weighting = float(m.get("selection_weighting", 10.0)) 
        
        # Lower bound for scaling factor (usually 1.0 to prevent weight growth)
        self.lower_bound = float(m.get("lower_bound", 1.0)) 
        
        # Exponent for the dampening function
        self.exponent = float(m.get("exponent", 1.0))

        # Computation control
        # If true, calculates baseline importance on Retain + Forget. 
        # If false, only uses Retain (faster, but technically deviation from paper).
        self.use_full_dataset_for_importance = m.get("use_full_dataset", True)

        self._logs = {}

    def _calc_importance(self, loaders: List[DataLoader]) -> Dict[str, torch.Tensor]:
        """
        Calculate the diagonal Fisher Information (importance) for parameters.
        Iterates over a list of dataloaders.
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Initialize importance dictionary with zeros
        importances = {
            n: torch.zeros_like(p, device=self.device)
            for n, p in self.model.named_parameters() 
            if p.requires_grad
        }
        
        total_samples = 0
        
        for loader in loaders:
            for batch in loader:
                x, y = _unpack_batch(batch, self.device)
                
                self.model.zero_grad()
                out = self.model(x)
                
                # SSD uses CrossEntropy to estimate Fisher Information
                loss = F.cross_entropy(out, y)
                loss.backward()

                # Accumulate squared gradients (Fisher approximation)
                for name, p in self.model.named_parameters():
                    if p.grad is not None:
                        # Normalize by batch size later, so just add raw squared grads here?
                        # The original implementation divides by len(dataloader) at the end.
                        # It is more stable to average per batch, then average over batches.
                        # Here we follow the logic: sum(grad^2) / N_total
                        importances[name] += p.grad.detach().pow(2)
                
                total_samples += 1 # We use number of batches as normalization factor roughly
        
        # Normalize
        if total_samples > 0:
            for n in importances:
                importances[n] /= float(total_samples)
                
        return importances

    def _modify_weights(
        self,
        original_importance: Dict[str, torch.Tensor],
        forget_importance: Dict[str, torch.Tensor],
    ) -> None:
        """
        Perturb weights based on the SSD equations.
        """
        with torch.no_grad():
            modified_count = 0
            total_params = 0
            
            for (n, p) in self.model.named_parameters():
                if n not in original_importance or n not in forget_importance:
                    continue
                
                oimp = original_importance[n]
                fimp = forget_importance[n]
                
                # Equation: Synapse Selection
                # Identify parameters where Forget Importance is significantly higher than Original Importance
                # fimp > alpha * oimp
                oimp_norm = oimp.mul(self.selection_weighting)
                locations = torch.where(fimp > oimp_norm)
                
                # Equation: Synapse Dampening
                # weight = (oimp * lambda / fimp) ^ exponent
                term1 = oimp.mul(self.dampening_constant)
                term2 = fimp + 1e-8 # stability
                weight = (term1.div(term2)).pow(self.exponent)
                
                update = weight[locations]
                
                # Bound by lower_bound (default 1.0) to ensure we generally shrink weights, not grow them
                # (Logic from original code: if update > lower_bound, clamp it)
                # Note: If lower_bound=1, we are capping the multiplier at 1.
                min_locs = torch.where(update > self.lower_bound)
                update[min_locs] = self.lower_bound
                
                # Apply update
                p[locations] = p[locations].mul(update)
                
                modified_count += locations[0].numel()
                total_params += p.numel()
            
            print(f"[SSD] Modified {modified_count}/{total_params} parameters ({modified_count/total_params:.2%})")
            self._logs["modified_params_ratio"] = modified_count / max(total_params, 1)

    def run(self) -> None:
        print("[SSD] Computing importances...")
        
        # 1. Calculate Forget Importances
        forget_imps = self._calc_importance([self.forget_loader])
        
        # 2. Calculate Original (Full) Importances
        # To simulate D_train, we use both retain and forget loaders
        if self.use_full_dataset_for_importance:
            full_loaders = [self.retain_loader, self.forget_loader]
        else:
            full_loaders = [self.retain_loader]
            
        original_imps = self._calc_importance(full_loaders)
        
        # 3. Apply Dampening
        print(f"[SSD] Dampening parameters (alpha={self.selection_weighting}, lambda={self.dampening_constant})...")
        self._modify_weights(original_imps, forget_imps)

    def get_model(self) -> nn.Module:
        return self.model

    def report(self) -> Dict[str, Any]:
        return {
            "hparams": {
                "dampening_constant": self.dampening_constant,
                "selection_weighting": self.selection_weighting,
                "exponent": self.exponent,
                "lower_bound": self.lower_bound,
            },
            "logs": self._logs
        }

    # Optional: Allow main script to grab the custom name
    def get_method_name(self):
        return f"ssd_alpha{self.selection_weighting}_lambda{self.dampening_constant}"