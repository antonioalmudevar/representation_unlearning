# src/metrics/privacy.py
import torch
import numpy as np
from typing import Dict
from sklearn.metrics import roc_auc_score, average_precision_score

@torch.no_grad()
def compute_mia_scores(pos_scores: np.ndarray, neg_scores: np.ndarray) -> Dict[str, float]:
    """
    Computes privacy metrics given scores for positive (member) and negative (non-member) classes.
    Args:
        pos_scores: Confidence scores for the 'Member' class (e.g. Retain Set)
        neg_scores: Confidence scores for the 'Non-Member' class (e.g. Forget/Test Set)
    """
    # 1. Setup Labels
    y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    y_score = np.concatenate([pos_scores, neg_scores])
    
    # 2. AUC & AP
    # AUC > 0.5 implies the 'Members' have higher confidence than 'Non-Members'
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    
    # 3. Balanced Accuracy (Robust to Class Imbalance)
    # We find the threshold that maximizes balanced accuracy
    thresholds = np.unique(y_score)
    # Optimization: only check a subset of thresholds if too many
    if len(thresholds) > 1000:
        thresholds = np.percentile(y_score, np.linspace(0, 100, 1000))
        
    best_acc = 0.5
    for thr in thresholds:
        tpr = (pos_scores >= thr).mean() # True Positive Rate
        tnr = (neg_scores < thr).mean()  # True Negative Rate
        balanced_acc = (tpr + tnr) / 2
        if balanced_acc > best_acc:
            best_acc = balanced_acc
            
    return {"auc": float(auc), "ap": float(ap), "balanced_acc": float(best_acc)}

@torch.no_grad()
def membership_inference_attack(
    model,
    retain_loader,
    forget_loader,
    test_loader,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Performs Membership Inference Attack using loss-based signal.
    
    This is the standard MIA used in machine unlearning papers:
    - Members (training data) have lower loss
    - Non-members (unseen data) have higher loss
    
    We use negative loss as the signal, so higher values = more likely a member.
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def get_signals(loader):
        signals = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            # Use negative loss so higher = more like training data (member)
            signals.append(-loss.detach().cpu())
            
        return torch.cat(signals).numpy()

    # Get signals (negative loss)
    sig_retain = get_signals(retain_loader)
    sig_forget = get_signals(forget_loader)
    sig_test   = get_signals(test_loader)

    # --- Attack: Forget vs. Test (The Gold Standard) ---
    # Does the Forget set look like unseen data? 
    # Ideally, AUC should be ~0.5 (indistinguishable from test set).
    # If AUC > 0.5, the model still remembers the forget set.
    metrics_ft = compute_mia_scores(pos_scores=sig_forget, neg_scores=sig_test)
    
    results = {
        "mia_auc": metrics_ft["auc"],
        "mia_ap": metrics_ft["ap"],
    }

    return results