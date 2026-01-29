# src/methods/gkt.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from copy import deepcopy
from typing import Dict, Any, Optional, List
import numpy as np

# =============================================================================
# Losses (Matched to your snippet)
# =============================================================================

def attention(x):
    """Normalized L2-map of activations."""
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

def attention_diff(x, y):
    """MSE between attention maps."""
    return (attention(x) - attention(y)).pow(2).mean()

def divergence(student_logits, teacher_logits, KL_temperature):
    """Forward KL Divergence."""
    return F.kl_div(
        F.log_softmax(student_logits / KL_temperature, dim=1), 
        F.softmax(teacher_logits / KL_temperature, dim=1),
        reduction="batchmean"
    )

def kt_loss_generator(student_logits, teacher_logits, temperature):
    # Generator wants to maximize divergence (or minimize negative divergence)?
    # In GKT paper/code: Generator minimizes the KT loss (Cooperative) or maximizes (Adversarial)?
    # The snippet has: total_loss = -divergence_loss.
    # This means Generator MAXIMIZES divergence (Adversarial).
    div = divergence(student_logits, teacher_logits, temperature)
    return -div

def kt_loss_student(s_logits, s_acts, t_logits, t_acts, temperature, beta):
    div = divergence(s_logits, t_logits, temperature)
    at_loss = 0
    if beta > 0:
        # s_acts and t_acts are lists of features
        for sa, ta in zip(s_acts, t_acts):
            at_loss += attention_diff(sa, ta)
    return div + (beta * at_loss)

# =============================================================================
# Helper Modules
# =============================================================================

def _unpack_batch(batch, device):
    if isinstance(batch, (list, tuple)):
        x, y = batch[0], batch[-1]
    else:
        x, y = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size
    def forward(self, tensor):
        return tensor.view(self.size)

class Generator(nn.Module):
    """Standard DCGAN-like generator for 32x32."""
    def __init__(self, z_dim, out_size=32, num_channels=3):
        super(Generator, self).__init__()
        self.inter_dim = z_dim // 2
        initial_size = out_size // 4 
        initial_dim = self.inter_dim * initial_size * initial_size

        self.layers = nn.Sequential(
            nn.Linear(z_dim, initial_dim),
            nn.ReLU(inplace=True),
            View((-1, self.inter_dim, initial_size, initial_size)),
            nn.BatchNorm2d(self.inter_dim),
            nn.ConvTranspose2d(self.inter_dim, self.inter_dim, 4, 2, 1),
            nn.BatchNorm2d(self.inter_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(self.inter_dim, self.inter_dim // 2, 4, 2, 1),
            nn.BatchNorm2d(self.inter_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.inter_dim // 2, num_channels, 3, 1, 1),
            nn.BatchNorm2d(num_channels)
            # No Tanh/Sigmoid at end, allowing BN to scale output naturally
        )

    def forward(self, z):
        return self.layers(z)

class ModelWithActivations(nn.Module):
    """Wraps model to return (logits, [features])."""
    def __init__(self, model, layer_name=None):
        super().__init__()
        self.model = model
        self.activations = {}
        self.hook_handle = None
        
        # Auto-detect layer
        if layer_name is None:
            for name, module in self.model.named_modules():
                # ResNet target
                if "layer4" in name and isinstance(module, nn.Conv2d):
                    layer_name = name
                # VGG/AlexNet target
                elif "features" in name and isinstance(module, nn.Sequential):
                    layer_name = name
            if layer_name is None:
                # Fallback
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Conv2d):
                        layer_name = name
                        
        self.layer_name = layer_name
        if self.layer_name:
            for name, module in self.model.named_modules():
                if name == self.layer_name:
                    self.hook_handle = module.register_forward_hook(self._hook)
                    break

    def _hook(self, module, input, output):
        self.activations['feat'] = output

    def forward(self, x):
        logits = self.model(x)
        feat = self.activations.get('feat', None)
        # Return format matched to expected unpacking: logits, [list_of_acts]
        return logits, [feat] if feat is not None else []
    
    def __del__(self):
        if self.hook_handle: self.hook_handle.remove()

# =============================================================================
# GKT Method
# =============================================================================

class GKT:
    def setup(self, model, *, retain_loader, forget_loader, val_loader, cfg, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.retain_loader = retain_loader
        self.forget_loader = forget_loader
        self.cfg = cfg
        
        # Check if this is an image dataset
        dataset_name = cfg.get("dataset", {}).get("name", "unknown")
        if dataset_name.lower() in ["toy"]:
            raise ValueError(
                f"GKT requires image datasets (32x32x3) but got dataset '{dataset_name}'. "
                f"GKT uses a DCGAN-style generator to create synthetic images for knowledge transfer, "
                f"which is not applicable to non-image datasets."
            )
        
        # Verify the model expects image input by checking first batch
        try:
            sample_batch = next(iter(retain_loader))
            sample_x = sample_batch[0] if isinstance(sample_batch, (list, tuple)) else sample_batch
            if sample_x.dim() != 4:  # Should be [B, C, H, W]
                raise ValueError(
                    f"GKT expects 4D image tensors [B,C,H,W] but got shape {sample_x.shape}. "
                    f"This dataset appears to use feature vectors instead of images."
                )
        except StopIteration:
            pass  # Empty loader, will fail later anyway
        
        m = dict(cfg.get("method", {}))
        
        # --- Hyperparams from your snippet ---
        self.z_dim = int(m.get("z_dim", 128))
        self.batch_size = int(m.get("batch_size", 256))
        self.n_pseudo_batches = int(m.get("n_pseudo_batches", 4000))
        self.n_generator_iter = int(m.get("n_generator_iter", 1))
        self.n_student_iter = int(m.get("n_student_iter", 10))
        self.lr = float(m.get("lr", 0.001))
        self.kl_temperature = float(m.get("kl_temperature", 1.0))
        self.at_beta = float(m.get("at_beta", 250.0))
        self.threshold = float(m.get("threshold", 0.01))
        
        # Identify Forget Class
        split_proto = cfg["dataset"].get("split_protocol", {})
        self.forget_class = split_proto.get("forget_classes", [0])[0]

        # --- Models ---
        self.teacher = ModelWithActivations(deepcopy(model)).eval().to(self.device)
        self.student = ModelWithActivations(deepcopy(model)).train().to(self.device)
        
        # Generator
        self.generator = Generator(self.z_dim, 32, 3).to(self.device)

        # --- Optimizers (Adam for both, per snippet) ---
        self.opt_gen = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        self.opt_stu = torch.optim.Adam(self.student.parameters(), lr=self.lr)

        # --- Schedulers (ReduceLROnPlateau, per snippet) ---
        # Note: patience=2, factor=0.5
        self.sched_gen = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt_gen, mode='min', factor=0.5, patience=500, verbose=True
        )
        self.sched_stu = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt_stu, mode='min', factor=0.5, patience=500, verbose=True
        )
        
        self._logs = {}

    @torch.no_grad()
    def _evaluate_metrics(self):
        self.student.eval()
        r_acc = self._acc(self.student.model, self.retain_loader)
        f_acc = self._acc(self.student.model, self.forget_loader)
        
        # Get loss on retain set for scheduler
        total_loss = 0.0
        batches = 0
        for x, y in self.retain_loader:
            x, y = x.to(self.device), y.to(self.device)
            out = self.student.model(x)
            loss = F.cross_entropy(out, y)
            total_loss += loss.item()
            batches += 1
        avg_loss = total_loss / max(batches, 1)
        
        self.student.train()
        return r_acc, f_acc, avg_loss

    def _acc(self, model, loader):
        correct, total = 0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
        return correct / max(total, 1)

    def run(self):
        print("[GKT] Starting unlearning (Inspired by original implementation)...")
        
        n_repeat_batch = self.n_generator_iter + self.n_student_iter
        idx_pseudo = 0
        n_cycles_done = 0  # equivalent to n_pseudo_batches in snippet
        
        # Generator checkpointing logic
        saved_gens = []
        zero_count = 0
        
        # Running losses for averaging
        run_g_loss = []
        run_s_loss = []

        # We loop until we complete the requested number of CYCLES
        # 1 Cycle = n_generator_iter + n_student_iter steps
        
        while n_cycles_done < self.n_pseudo_batches:
            
            # --- 1. Generate & Filter (Happens EVERY step) ---
            z = torch.randn((self.batch_size, self.z_dim)).to(self.device)
            x_pseudo = self.generator(z)
            
            # Filter
            with torch.no_grad():
                # Teacher check
                t_logits, _ = self.teacher(x_pseudo)
                probs = torch.softmax(t_logits, dim=1)
            
            mask = (probs[:, self.forget_class] <= self.threshold)
            x_filtered = x_pseudo[mask]
            
            # Collapse check
            if x_filtered.size(0) == 0:
                zero_count += 1
                if zero_count > 100:
                    print(f"[GKT] Generator collapse. Rewinding...")
                    if len(saved_gens) > 0:
                        self.generator.load_state_dict(saved_gens[-1])
                    zero_count = 0
                continue
            else:
                zero_count = 0

            # --- 2. Step Logic (Alternating based on idx) ---
            step_in_cycle = idx_pseudo % n_repeat_batch
            
            if step_in_cycle < self.n_generator_iter:
                # === GENERATOR STEP ===
                self.opt_gen.zero_grad()
                
                # Regenerate z? No, snippet uses same x_pseudo batch from generator.__next__()
                # But x_filtered is detached from graph if we masked it? 
                # Actually x_pseudo is a leaf from generator(z). Masking is a select operation. Gradients flow.
                
                s_logits, _ = self.student(x_filtered)
                t_logits, _ = self.teacher(x_filtered)
                
                loss_g = kt_loss_generator(s_logits, t_logits, self.kl_temperature)
                
                loss_g.backward()
                nn.utils.clip_grad_norm_(self.generator.parameters(), 5)
                self.opt_gen.step()
                
                run_g_loss.append(loss_g.item())
                
            else:
                # === STUDENT STEP ===
                self.opt_stu.zero_grad()
                
                s_logits, s_acts = self.student(x_filtered.detach()) # Detach for student training
                with torch.no_grad():
                    t_logits, t_acts = self.teacher(x_filtered.detach())
                
                loss_s = kt_loss_student(
                    s_logits, s_acts, t_logits, t_acts, 
                    self.kl_temperature, self.at_beta
                )
                
                loss_s.backward()
                nn.utils.clip_grad_norm_(self.student.parameters(), 5)
                self.opt_stu.step()
                
                run_s_loss.append(loss_s.item())

            # --- 3. End of Cycle Check ---
            if (idx_pseudo + 1) % n_repeat_batch == 0:
                
                # Periodic Reporting (every 50 cycles)
                if n_cycles_done % 50 == 0:
                    r_acc, f_acc, r_loss = self._evaluate_metrics()
                    
                    mean_g = np.mean(run_g_loss) if run_g_loss else 0
                    mean_s = np.mean(run_s_loss) if run_s_loss else 0
                    run_g_loss, run_s_loss = [], []
                    
                    print(f"[Cycle {n_cycles_done}] Retain: {r_acc:.2%} | Forget: {f_acc:.2%} | G_Loss: {mean_g:.4f} | S_Loss: {mean_s:.4f}")
                    
                    # Schedulers step based on loss
                    # Note: Snippet steps student scheduler on Retain Valid Loss
                    self.sched_stu.step(r_loss)
                    self.sched_gen.step(mean_g) # Generator steps on its own loss
                    
                    # Checkpointing
                    saved_gens.append(deepcopy(self.generator.state_dict()))
                    if len(saved_gens) > 5: saved_gens.pop(0)

                n_cycles_done += 1
            
            idx_pseudo += 1

    def get_model(self):
        return self.student.model
    
    def report(self):
        return {"hparams": self.cfg["method"], "logs": self._logs}