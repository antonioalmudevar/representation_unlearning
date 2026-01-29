# src/methods/representation_unlearning/method.py
from typing import Any, Dict, List
import copy, time, torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from .base import IUnlearningMethod
from ..helpers.registry import register
from ..helpers.train_utils import evaluate_acc

class UnlearningTransformation(nn.Module):
    """
    The transformation module f_phi described in Section 3.2.
    Maps z -> z' to filter information while preserving semantics.
    
    Supports:
    - 'linear': Single affine transformation (z' = Wz + b)
    - 'mlp': Variable depth and width defined by hidden_dims list
    """
    def __init__(self, input_dim: int, hidden_dims: List[int] = [512], arch_type: str = "mlp"):
        super().__init__()
        self.arch_type = arch_type.lower()
        
        if self.arch_type == "linear":
            # Simple linear map: W * z + b
            self.net = nn.Linear(input_dim, input_dim)
            
            # Initialization to Identity (since no residual connection)
            nn.init.eye_(self.net.weight)
            nn.init.zeros_(self.net.bias)
            
        elif self.arch_type == "mlp":
            layers = []
            
            # If empty list passed, fallback to linear
            if not hidden_dims:
                layers.append(nn.Linear(input_dim, input_dim))
            else:
                # 1. Input -> First Hidden
                layers.append(nn.Linear(input_dim, hidden_dims[0]))
                layers.append(nn.ReLU())
                
                # 2. Hidden -> Hidden
                for i in range(len(hidden_dims) - 1):
                    layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                    layers.append(nn.ReLU())
                
                # 3. Last Hidden -> Output
                layers.append(nn.Linear(hidden_dims[-1], input_dim))
            
            self.net = nn.Sequential(*layers)
            
        else:
            raise ValueError(f"Unknown arch_type: {arch_type}. Use 'linear' or 'mlp'.")

    def forward(self, z):
        # Direct transformation (No residual connection)
        return self.net(z)

class RewiredModel(nn.Module):
    """
    Wraps the original model and the learned transformation.
    Intercepts features before the classifier, applies f_phi, and passes to classifier.
    """
    def __init__(self, original_model, transformation, layer_name=None):
        super().__init__()
        # We deepcopy to avoid modifying the original model instance passed in setup
        self.backbone = copy.deepcopy(original_model)
        self.transformation = transformation
        
        # If layer_name is provided (found during setup), use it directly
        if layer_name and hasattr(self.backbone, layer_name):
            self.classifier_layer = getattr(self.backbone, layer_name)
            setattr(self.backbone, layer_name, nn.Identity())
        else:
            # Fallback logic for robustness
            found = False
            for name in ['fc', 'classifier', 'linear', 'head']:
                if hasattr(self.backbone, name):
                    self.classifier_layer = getattr(self.backbone, name)
                    setattr(self.backbone, name, nn.Identity())
                    found = True
                    break
            
            if not found:
                 raise AttributeError("Could not identify classifier layer (fc/classifier/linear/head).")

    def forward(self, x, return_repr=False):
        z = self.backbone(x)
        z_prime = self.transformation(z)
        logits = self.classifier_layer(z_prime)
        
        if return_repr:
            return logits, z_prime
        return logits
    
    def get_representation(self, x):
        """Extract transformed representation without classification."""
        z = self.backbone(x)
        z_prime = self.transformation(z)
        return z_prime

@register("representation_unlearning")
class RepresentationUnlearning(IUnlearningMethod):
    """
    Representation Unlearning: Forgetting through Information Compression.
    Implements variational surrogates for both standard and zero-shot settings.
    """

    def setup(self, model, *, retain_loader, forget_loader, val_loader=None, 
              cfg: Dict[str, Any], device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Store original model
        self.original_model = copy.deepcopy(model).to(self.device)
        self.original_model.eval()
        for p in self.original_model.parameters():
            p.requires_grad = False
            
        # Robustly identify the feature layer
        self.target_layer_name, self.target_layer = self._resolve_layer(self.original_model)
        
        # Store data loaders
        self.retain_loader = retain_loader
        self.forget_loader = forget_loader
        self.val_loader = val_loader
        
        # Determine feature dimension
        self.feature_dim = self._get_feature_dim(self.original_model, self.target_layer)
        print(f"[RepresentationUnlearning] Features: {self.feature_dim} dims at layer '{self.target_layer_name}'")
        
        # --- Hyperparameters ---
        hp = cfg.get("method", {})
        self.epochs = int(hp.get("epochs", 5))
        self.lr = float(hp.get("lr", 1e-3))
        self.beta = float(hp.get("beta", 1.0))
        self.noise_scale = float(hp.get("noise_scale", 0.0))
        self.n_samples = int(hp.get("n_samples", 5))
        
        # --- Architecture Configuration ---
        self.arch_type = str(hp.get("arch_type", "mlp")) 
        
        # Parse hidden_dims: Support both list [512, 256] and legacy int 512
        raw_dims = hp.get("hidden_dims", [512])
        if isinstance(raw_dims, int):
            self.hidden_dims = [raw_dims]
        else:
            self.hidden_dims = list(raw_dims)
        
        # Zero-shot configuration
        self.zero_shot = bool(hp.get("zero_shot", False))
        self.num_classes = cfg["model"]["num_classes"]
        
        # Compute Class Counts (Nc)
        self.class_counts = self._get_class_counts(retain_loader, forget_loader, self.num_classes)
        print(f"[RepresentationUnlearning] Class counts (Nc): {self.class_counts.cpu().tolist()}")

        if self.zero_shot:
            print("[RepresentationUnlearning] Zero-shot mode enabled. Using Neural Collapse proxies.")
            self.class_prototypes = self._extract_prototypes(self.original_model).to(self.device)
            proto_cfg = cfg["dataset"].get("split_protocol", {})
            self.forget_classes = proto_cfg.get("forget_classes", []) if proto_cfg.get("type") == "class_forget" else []
            
        self.transformation = UnlearningTransformation(
            input_dim=self.feature_dim, 
            hidden_dims=self.hidden_dims, 
            arch_type=self.arch_type
        ).to(self.device)
        
        # --- Logging Container (Losses Only) ---
        self._history = {
            "loss_total": [],
            "loss_retention": [],
            "loss_forget": []
        }
        self._report = {}

    def _resolve_layer(self, model):
        for name in ['fc', 'classifier', 'linear', 'head']:
            if hasattr(model, name):
                return name, getattr(model, name)
        raise AttributeError("Could not find a standard classification layer (fc/classifier/linear/head).")

    def _get_feature_dim(self, model, target_layer):
        # Get a sample batch to determine input shape
        sample_batch = next(iter(self.retain_loader))[0]
        dummy = sample_batch[:1].to(self.device)
        
        feats = []
        def hook(module, input, output):
            feats.append(input[0])
        handle = target_layer.register_forward_hook(hook)
        with torch.no_grad():
            model(dummy)
        handle.remove()
        return feats[0].shape[1]

    def _extract_features(self, x):
        features = []
        def hook(module, input, output):
            features.append(input[0])
        handle = self.target_layer.register_forward_hook(hook)
        self.original_model(x)
        handle.remove()
        return features[0]

    def _extract_prototypes(self, model):
        _, layer = self._resolve_layer(model)
        return layer.weight.data.clone().detach()
        
    def _get_class_counts(self, retain_loader, forget_loader, num_classes):
        counts = torch.zeros(num_classes, device=self.device)
        def count_loader(loader):
            for _, y in loader:
                y = y.to(self.device)
                c = torch.bincount(y, minlength=num_classes).float()
                counts.add_(c)
        count_loader(retain_loader)
        count_loader(forget_loader)
        return counts

    def run(self) -> None:
        if self.zero_shot:
            self._run_zero_shot()
        else:
            self._run_standard()

    def _run_standard(self):
        """Original data-driven implementation (Algorithm 1)"""
        start = time.time()
        optimizer = optim.Adam(self.transformation.parameters(), lr=self.lr)
        
        ref_dataset = ConcatDataset([self.retain_loader.dataset, self.forget_loader.dataset])
        ref_loader = DataLoader(ref_dataset, batch_size=self.forget_loader.batch_size, shuffle=True)
        
        self.transformation.train()
        
        iter_retain = iter(self.retain_loader)
        iter_ref = iter(ref_loader)
        
        for epoch in range(self.epochs):
            total_loss = 0.0
            total_r = 0.0
            total_f = 0.0
            n_batches = 0
            
            for x_f, _ in self.forget_loader:
                x_f = x_f.to(self.device)
                
                # Get Retain batch
                try:
                    x_r, _ = next(iter_retain)
                except StopIteration:
                    iter_retain = iter(self.retain_loader)
                    x_r, _ = next(iter_retain)
                x_r = x_r.to(self.device)
                
                # Get Reference batch
                try:
                    x_ref, _ = next(iter_ref)
                except StopIteration:
                    iter_ref = iter(ref_loader)
                    x_ref, _ = next(iter_ref)
                x_ref = x_ref.to(self.device)

                with torch.no_grad():
                    z_r = self._extract_features(x_r)
                    z_f = self._extract_features(x_f)
                    z_ref = self._extract_features(x_ref)
                
                # Monte Carlo Sampling
                B_r, B_f, B_ref = z_r.size(0), z_f.size(0), z_ref.size(0)
                M, Dim = self.n_samples, self.feature_dim
                
                noise_r = torch.randn(B_r, M, Dim, device=self.device) * self.noise_scale
                noise_f = torch.randn(B_f, M, Dim, device=self.device) * self.noise_scale
                
                z_r_noisy = z_r.unsqueeze(1) + noise_r
                z_f_noisy = z_f.unsqueeze(1) + noise_f
                
                z_r_prime = self.transformation(z_r_noisy)
                z_f_prime = self.transformation(z_f_noisy)
                
                # Objectives
                loss_r = F.mse_loss(z_r_prime, z_r.unsqueeze(1))
                
                z_f_flat = z_f_prime.view(-1, Dim) 
                diff = z_f_flat.unsqueeze(1) - z_ref.unsqueeze(0)
                dist_matrix = torch.sum(diff ** 2, dim=2)
                loss_f = dist_matrix.mean()
                
                loss = loss_r + self.beta * loss_f
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_r += loss_r.item()
                total_f += loss_f.item()
                n_batches += 1
            
            avg_loss = total_loss / max(n_batches, 1)
            avg_r = total_r / max(n_batches, 1)
            avg_f = total_f / max(n_batches, 1)
            print(f"Epoch {epoch+1}/{self.epochs} | Loss: {avg_loss:.4f} (R: {avg_r:.4f}, F: {avg_f:.4f})")
            
            self._history["loss_total"].append(avg_loss)
            self._history["loss_retention"].append(avg_r)
            self._history["loss_forget"].append(avg_f)
            
        self._finalize(start, mode="standard")

    def _run_zero_shot(self):
        """Zero-Shot Implementation (Algorithm 2) with Weights"""
        start = time.time()
        optimizer = optim.Adam(self.transformation.parameters(), lr=self.lr)
        self.transformation.train()
        
        # Prepare Retain Weights (Eq 7)
        forget_set = set(self.forget_classes)
        retain_indices_list = [i for i in range(self.num_classes) if i not in forget_set]
        
        if len(retain_indices_list) == 0:
            retain_indices_list = [i for i in range(self.num_classes)]

        retain_indices = torch.tensor(retain_indices_list, device=self.device, dtype=torch.long)
        retain_prototypes = self.class_prototypes[retain_indices] 
        
        retain_counts = self.class_counts[retain_indices]
        retain_probs = retain_counts / retain_counts.sum()
        retain_weights = retain_probs.unsqueeze(1) 

        # Prepare Global Class Weights (Eq 17)
        all_prototypes = self.class_prototypes 
        total_samples = self.class_counts.sum()
        class_weights = self.class_counts / total_samples
        class_weights_expanded = class_weights.view(1, 1, -1)
        
        for epoch in range(self.epochs):
            total_loss = 0.0
            total_r = 0.0
            total_f = 0.0
            n_batches = 0
            
            for x_f, _ in self.forget_loader:
                x_f = x_f.to(self.device)
                
                with torch.no_grad():
                    z_f = self._extract_features(x_f)

                B_f = z_f.size(0)
                M, Dim = self.n_samples, self.feature_dim
                
                # L_r_zs: Weighted Retention (Eq 7)
                n_retain = retain_prototypes.size(0)
                proto_noise = torch.randn(n_retain, M, Dim, device=self.device) * self.noise_scale
                z_synthetic = retain_prototypes.unsqueeze(1) + proto_noise
                
                z_synthetic_prime = self.transformation(z_synthetic)
                target_r = retain_prototypes.unsqueeze(1).expand(n_retain, M, Dim)
                
                diff_sq = torch.sum((z_synthetic_prime - target_r) ** 2, dim=2)
                expected_sq_dist = diff_sq.mean(dim=1)
                loss_r_zs = torch.sum(expected_sq_dist * retain_probs)
                
                # L_f_zs: Weighted Forgetting (Eq 17)
                noise_f = torch.randn(B_f, M, Dim, device=self.device) * self.noise_scale
                z_f_noisy = z_f.unsqueeze(1) + noise_f 
                z_f_prime = self.transformation(z_f_noisy)
                
                z_expanded = z_f_prime.unsqueeze(2)
                w_expanded = all_prototypes.unsqueeze(0).unsqueeze(0)
                
                dist_sq = torch.sum((z_expanded - w_expanded) ** 2, dim=3)
                weighted_dist = dist_sq * class_weights_expanded
                loss_f_zs = weighted_dist.sum(dim=2).mean()
                
                loss = loss_r_zs + self.beta * loss_f_zs
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_r += loss_r_zs.item()
                total_f += loss_f_zs.item()
                n_batches += 1
            
            avg_loss = total_loss / max(n_batches, 1)
            avg_r = total_r / max(n_batches, 1)
            avg_f = total_f / max(n_batches, 1)
            
            print(f"Epoch {epoch+1}/{self.epochs} [ZS] | Loss: {avg_loss:.4f} (R: {avg_r:.4f}, F: {avg_f:.4f})")
            
            self._history["loss_total"].append(avg_loss)
            self._history["loss_retention"].append(avg_r)
            self._history["loss_forget"].append(avg_f)

        self._finalize(start, mode="zero_shot")

    def _finalize(self, start_time, mode="standard"):
        train_time = time.time() - start_time
        self._final_model = RewiredModel(self.original_model, self.transformation, self.target_layer_name)
        
        acc_val = evaluate_acc(self._final_model, self.val_loader, self.device) if self.val_loader else None
        acc_forget = evaluate_acc(self._final_model, self.forget_loader, self.device) if self.forget_loader else None

        self._report.update({
            "method": "representation_unlearning",
            "mode": mode,
            "epochs": self.epochs,
            "train_time_sec": train_time,
            "val_acc": acc_val,
            "forget_acc": acc_forget,
            "history": self._history
        })

    def get_model(self):
        return self._final_model

    def report(self) -> Dict[str, Any]:
        return self._report

    def get_method_name(self) -> str:
        """
        Generate a method name that includes hyperparameters for distinguishing experiments.
        Format: representation_unlearning_{zs_}_{arch}_{hidden_dims}_beta{beta}
        Examples:
        - representation_unlearning_linear_beta1.0
        - representation_unlearning_mlp_512_beta0.5
        - representation_unlearning_zs_mlp_512_256_beta2.0
        """
        # Start with base name
        name_parts = ["representation_unlearning"]
        
        # Add zero-shot flag if enabled
        if self.zero_shot:
            name_parts.append("zs")
        
        # Add arch type
        name_parts.append(self.arch_type)
        
        # Add hidden dims (only for MLP, skip for linear)
        if self.arch_type == "mlp" and self.hidden_dims:
            dims_str = "_".join(str(d) for d in self.hidden_dims)
            name_parts.append(dims_str)
        
        # Add beta value
        beta_str = f"beta{self.beta}"
        name_parts.append(beta_str)
        
        return "_".join(name_parts)