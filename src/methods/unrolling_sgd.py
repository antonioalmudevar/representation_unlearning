# Python 3.8. No __main__ block.
from typing import Dict, Any, Optional
from dataclasses import dataclass
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class _UnrollSGDCfg:
    eta: float = 0.1
    epochs: int = 200
    train_batch_size: int = 128
    include_bn: bool = True
    include_bias: bool = True
    only_head: bool = False
    scale_multiplier: float = 0.1
    dataset_size: Optional[int] = None
    w0_path: Optional[str] = None
    max_forget_batches: Optional[int] = None
    device: str = "cuda"
    eps: float = 1e-12


def _iter_named_params(model: nn.Module, include_bn: bool, include_bias: bool, only_head: bool = False):
    head = getattr(model, "fc", None) or getattr(model, "classifier", None)
    head_params = set([id(p) for p in (head.parameters() if head and hasattr(head, "parameters") else [])])
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if only_head and id(p) not in head_params:
            continue
        if not include_bias and name.endswith(".bias"):
            continue
        if not include_bn:
            toks = name.split(".")
            if any(tok.lower().startswith(("bn","batchnorm","norm","ln","layernorm","gn","groupnorm")) for tok in toks):
                continue
        yield name, p


def _device_of(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


class UnrollingSGD:
    """
    Single-gradient unlearning (Unrolling SGD) from Thudi et al., 2022:
      w_unlearned  ≈  w_t  +  (η * m / b) * Σ_{x in D_f} ∇_w L(w0; x)
    where:
      - η is the training LR,
      - m is #epochs,
      - b is the training batch size,
      - gradients are taken at the initial weights w0, on the forget set D_f.
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
        m = dict(cfg.get("method", {}))
        self.cfg = _UnrollSGDCfg(
            eta=float(m.get("eta", 0.1)),
            epochs=int(m.get("epochs", 200)),
            train_batch_size=int(m.get("train_batch_size", 128)),
            include_bn=bool(m.get("include_bn", True)),
            include_bias=bool(m.get("include_bias", True)),
            only_head=bool(m.get("only_head", False)),
            scale_multiplier=float(m.get("scale_multiplier", 0.1)),
            dataset_size=m.get("dataset_size", None),
            w0_path=m.get("w0_path", None),
            max_forget_batches=m.get("max_forget_batches", None),
            device=device,
            eps=float(m.get("eps", 1e-12)),
        )

        self.model = model.to(device if torch.cuda.is_available() else "cpu")
        self.retain_loader = retain_loader
        self.forget_loader = forget_loader
        self.val_loader = val_loader
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")
        
        # Store original model for w0
        self.original_model = deepcopy(model).to(self.device)
        self.original_model.eval()

        self.logs: Dict[str, Any] = {}

    def run(self) -> None:
        # 1) Load w0 (initial trained model)
        w0_model = deepcopy(self.original_model).to(self.device)
        
        # If w0_path provided, load from file
        if self.cfg.w0_path:
            print(f"[Unroll SGD] Loading w0 from {self.cfg.w0_path}")
            try:
                sd = torch.load(self.cfg.w0_path, map_location=self.device)
                # Handle different checkpoint formats
                if "model_state_dict" in sd:
                    sd = sd["model_state_dict"]
                elif "state_dict" in sd:
                    sd = sd["state_dict"]
                
                w0_model.load_state_dict(sd, strict=True)
                print(f"[Unroll SGD] Successfully loaded w0")
            except Exception as e:
                print(f"[Unroll SGD] WARNING: Could not load w0 from {self.cfg.w0_path}: {e}")
                print(f"[Unroll SGD] Using current model as w0 (may give poor results)")
        else:
            print(f"[Unroll SGD] No w0_path provided - using current model as w0")
            print(f"[Unroll SGD] WARNING: This may give poor results. Provide w0_path for best performance.")

        w0_model.train(False)

        # 2) Accumulate SUM of per-sample gradients on D_f at w0
        grad_sum: Dict[str, torch.Tensor] = {}
        for name, p in _iter_named_params(w0_model, self.cfg.include_bn, self.cfg.include_bias, self.cfg.only_head):
            grad_sum[name] = torch.zeros_like(p, device=self.device)

        total_forget_samples = 0
        print(f"[Unroll SGD] Computing gradients on forget set...")
        
        for b_idx, batch in enumerate(self.forget_loader):
            if self.cfg.max_forget_batches is not None and b_idx >= int(self.cfg.max_forget_batches):
                break
            
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[-1]
            else:
                x, y = batch
            
            x = x.to(self.device, non_blocking=True).float()
            y = y.to(self.device, non_blocking=True)

            w0_model.zero_grad(set_to_none=True)
            logits = w0_model(x)
            loss = F.cross_entropy(logits, y, reduction="sum")
            loss.backward()

            for name, p in _iter_named_params(w0_model, self.cfg.include_bn, self.cfg.include_bias, self.cfg.only_head):
                if p.grad is not None:
                    grad_sum[name].add_(p.grad.detach())

            total_forget_samples += y.numel()
            
            if (b_idx + 1) % 10 == 0:
                print(f"[Unroll SGD]   Processed {total_forget_samples} forget samples...")

        print(f"[Unroll SGD] Total forget samples: {total_forget_samples}")

        # 3) Compute scale: η * m / b
        if self.cfg.dataset_size:
            scale = self.cfg.eta * float(self.cfg.epochs) / float(max(self.cfg.dataset_size, 1))
        else:
            scale = self.cfg.eta * float(self.cfg.epochs) / float(max(self.cfg.train_batch_size, 1))

        scale *= float(self.cfg.scale_multiplier)
        
        print(f"[Unroll SGD] Scale factor: {scale:.6f}")
        print(f"[Unroll SGD]   eta={self.cfg.eta}, epochs={self.cfg.epochs}, batch_size={self.cfg.train_batch_size}")
        print(f"[Unroll SGD]   scale_multiplier={self.cfg.scale_multiplier}")

        # 4) Apply w ← w + scale * Σ∇L(w0; x_i)
        affected_elems = 0
        delta_norm2 = 0.0
        
        print(f"[Unroll SGD] Applying unlearning update...")
        with torch.no_grad():
            for name, p in _iter_named_params(self.model, self.cfg.include_bn, self.cfg.include_bias, self.cfg.only_head):
                g = grad_sum.get(name, None)
                if g is None:
                    continue
                update = scale * g.to(p.device, non_blocking=True)
                p.add_(update)
                affected_elems += p.numel()
                delta_norm2 += float((update.reshape(-1) @ update.reshape(-1)).item())

        print(f"[Unroll SGD] Updated {affected_elems:,} parameters")
        print(f"[Unroll SGD] Update L2 norm: {delta_norm2 ** 0.5:.6f}")

        self.logs.update({
            "eta": self.cfg.eta,
            "epochs": self.cfg.epochs,
            "train_batch_size": self.cfg.train_batch_size,
            "include_bn": self.cfg.include_bn,
            "include_bias": self.cfg.include_bias,
            "only_head": self.cfg.only_head,
            "w0_path": self.cfg.w0_path,
            "forget_samples": int(total_forget_samples),
            "scale_eta_m_over_b": float(scale),
            "affected_param_elements": int(affected_elems),
            "update_l2": float(delta_norm2 ** 0.5),
            "scale_raw": float(self.cfg.eta * float(self.cfg.epochs) / float(max(self.cfg.train_batch_size,1))),
            "scale_multiplier": self.cfg.scale_multiplier,
            "scale_used": float(scale),
        })

    def get_model(self) -> nn.Module:
        return self.model

    def report(self) -> Dict[str, Any]:
        return {
            "hparams": {
                "eta": self.cfg.eta,
                "epochs": self.cfg.epochs,
                "train_batch_size": self.cfg.train_batch_size,
                "include_bn": self.cfg.include_bn,
                "include_bias": self.cfg.include_bias,
                "only_head": self.cfg.only_head,
                "w0_path": self.cfg.w0_path,
                "scale_multiplier": self.cfg.scale_multiplier,
                "max_forget_batches": self.cfg.max_forget_batches,
            },
            "logs": self.logs,
        }