# src/metrics/classification.py
from typing import Dict, List
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, precision_score,
    recall_score
)
import numpy as np

def _ece(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE) in [0, 1]."""
    probs = F.softmax(logits, dim=1)
    conf, preds = probs.max(1)
    bins = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    ece = torch.zeros(1, device=logits.device)
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.sum() == 0:
            continue
        acc_bin = (preds[mask] == labels[mask]).float().mean()
        conf_bin = conf[mask].mean()
        ece += (mask.float().mean() * torch.abs(acc_bin - conf_bin))
    return ece.item()

@torch.no_grad()
def classification_metrics(model, loader, device: str = "cuda", average: str = "macro") -> Dict[str, float]:
    """Compute accuracy, F1, precision, recall, ECE, confusion."""
    model.eval()
    y_true, y_pred, logits_all = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(1)
        y_true.append(y.cpu())
        y_pred.append(preds.cpu())
        logits_all.append(logits.cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    logits_all = torch.cat(logits_all)
    ece = _ece(logits_all, torch.tensor(y_true))
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=average)),
        "precision": float(precision_score(y_true, y_pred, average=average)),
        "recall": float(recall_score(y_true, y_pred, average=average)),
        "ece": float(ece),
        "confusion": confusion_matrix(y_true, y_pred).tolist(),
    }
