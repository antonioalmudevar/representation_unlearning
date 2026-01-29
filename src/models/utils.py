# src/models/utils.py
from typing import Optional, Iterable
import torch
from torch import nn

def load_checkpoint_(model: nn.Module, checkpoint: Optional[str] = None, strict: bool = True):
    if checkpoint and len(str(checkpoint).strip()) > 0:
        state = torch.load(checkpoint, map_location="cpu")
        # Soportar checkpoints envueltos (p.ej. DataParallel)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=strict)

def set_bn_opts_(module: nn.Module, eps: float = 1e-5, momentum: float = 0.1):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = eps
            m.momentum = momentum

def freeze_all_but_(model: nn.Module, params: Iterable[nn.Parameter]):
    params_set = set(params)
    for p in model.parameters():
        p.requires_grad = (p in params_set)
