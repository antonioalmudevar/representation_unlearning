# src/metrics/output_divergence.py
import torch
import torch.nn.functional as F
from typing import Dict, Optional
from pathlib import Path


@torch.no_grad()
def cross_entropy_divergence(
    model_retrain,
    model_unlearned,
    retain_loader,
    forget_loader,
    test_loader=None,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Computes cross-entropy, KL, and JS divergence between retrain baseline and unlearned models.
    
    This metric measures how close the unlearned model's predictions are to the retrain baseline
    (the 'gold standard' model trained only on the retain set). Lower divergence indicates better
    approximation of the ideal unlearned model.
    
    Args:
        model_retrain: The retrain baseline model (trained only on retain set)
        model_unlearned: The model after unlearning
        retain_loader: DataLoader for the retain set
        forget_loader: DataLoader for the forget set
        test_loader: Optional DataLoader for the test set
        device: Device to run computations on
        
    Returns:
        Dictionary with divergence metrics:
        - retain_ce_divergence_mean: Mean CE between models on retain set (lower is better)
        - retain_ce_divergence_std: Std CE between models on retain set
        - forget_ce_divergence_mean: Mean CE between models on forget set (lower is better)
        - forget_ce_divergence_std: Std CE between models on forget set
        - retain_kl_divergence_mean: Mean KL divergence on retain set (lower is better)
        - retain_kl_divergence_std: Std KL divergence on retain set
        - forget_kl_divergence_mean: Mean KL divergence on forget set (lower is better)
        - forget_kl_divergence_std: Std KL divergence on forget set
        - retain_js_divergence_mean: Mean JS divergence on retain set (lower is better)
        - retain_js_divergence_std: Std JS divergence on retain set
        - forget_js_divergence_mean: Mean JS divergence on forget set (lower is better)
        - forget_js_divergence_std: Std JS divergence on forget set
        - test_ce_divergence_mean: Mean CE on test set (if test_loader provided, lower is better)
        - test_ce_divergence_std: Std CE on test set (if test_loader provided)
        - test_kl_divergence_mean: Mean KL divergence on test set (if test_loader provided, lower is better)
        - test_kl_divergence_std: Std KL divergence on test set (if test_loader provided)
        - test_js_divergence_mean: Mean JS divergence on test set (if test_loader provided, lower is better)
        - test_js_divergence_std: Std JS divergence on test set (if test_loader provided)
    """
    model_retrain.eval()
    model_unlearned.eval()
    
    def compute_divergence(loader):
        """Compute cross-entropy, KL, and JS divergence for a given loader."""
        ce_divs = []
        kl_divs = []
        js_divs = []
        
        for x, _ in loader:
            x = x.to(device)
            
            # Get logits from both models
            logits_retrain = model_retrain(x)
            logits_unlearned = model_unlearned(x)
            
            # Convert to probabilities
            probs_retrain = F.softmax(logits_retrain, dim=1)
            probs_unlearned = F.softmax(logits_unlearned, dim=1)
            
            # Cross-Entropy: CE(P_retrain || P_unlearned)
            # Measures how well the unlearned model's distribution matches the retrain baseline
            ce = -(probs_retrain * torch.log(probs_unlearned + 1e-10)).sum(dim=1)
            ce_divs.append(ce.cpu())
            
            # KL Divergence: KL(P_retrain || P_unlearned)
            # Forward KL divergence from retrain to unlearned
            kl = (probs_retrain * (torch.log(probs_retrain + 1e-10) - torch.log(probs_unlearned + 1e-10))).sum(dim=1)
            kl_divs.append(kl.cpu())
            
            # JS Divergence: JS(P_retrain || P_unlearned)
            # Jensen-Shannon divergence - symmetric version of KL
            m = 0.5 * (probs_retrain + probs_unlearned)
            js = 0.5 * (probs_retrain * (torch.log(probs_retrain + 1e-10) - torch.log(m + 1e-10))).sum(dim=1) + \
                 0.5 * (probs_unlearned * (torch.log(probs_unlearned + 1e-10) - torch.log(m + 1e-10))).sum(dim=1)
            js_divs.append(js.cpu())
        
        ce_tensor = torch.cat(ce_divs)
        kl_tensor = torch.cat(kl_divs)
        js_tensor = torch.cat(js_divs)
        
        return {
            "mean_ce": float(ce_tensor.mean()),
            "std_ce": float(ce_tensor.std()),
            "mean_kl": float(kl_tensor.mean()),
            "std_kl": float(kl_tensor.std()),
            "mean_js": float(js_tensor.mean()),
            "std_js": float(js_tensor.std()),
        }
    
    # Compute for retain set
    retain_metrics = compute_divergence(retain_loader)
    
    # Compute for forget set
    forget_metrics = compute_divergence(forget_loader)
    
    # Combine results
    results = {
        "retain_ce_divergence_mean": retain_metrics["mean_ce"],
        "retain_ce_divergence_std": retain_metrics["std_ce"],
        "forget_ce_divergence_mean": forget_metrics["mean_ce"],
        "forget_ce_divergence_std": forget_metrics["std_ce"],
        "retain_kl_divergence_mean": retain_metrics["mean_kl"],
        "retain_kl_divergence_std": retain_metrics["std_kl"],
        "forget_kl_divergence_mean": forget_metrics["mean_kl"],
        "forget_kl_divergence_std": forget_metrics["std_kl"],
        "retain_js_divergence_mean": retain_metrics["mean_js"],
        "retain_js_divergence_std": retain_metrics["std_js"],
        "forget_js_divergence_mean": forget_metrics["mean_js"],
        "forget_js_divergence_std": forget_metrics["std_js"],
    }
    
    # Compute for test set if provided
    if test_loader is not None:
        test_metrics = compute_divergence(test_loader)
        results.update({
            "test_ce_divergence_mean": test_metrics["mean_ce"],
            "test_ce_divergence_std": test_metrics["std_ce"],
            "test_kl_divergence_mean": test_metrics["mean_kl"],
            "test_kl_divergence_std": test_metrics["std_kl"],
            "test_js_divergence_mean": test_metrics["mean_js"],
            "test_js_divergence_std": test_metrics["std_js"],
        })
    
    return results
