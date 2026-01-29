# src/metrics/reps.py
"""
Representation similarity metrics:
  - Linear CKA
  - PWCCA
  - RSA (Representational Similarity Analysis)
"""

from typing import Dict
import torch
import torch.nn.functional as F

def _center_gram(K):
    n = K.shape[0]
    unit = torch.ones((n, n), device=K.device)
    I = torch.eye(n, device=K.device)
    H = I - unit / n
    return torch.matmul(torch.matmul(H, K), H)

def _gram_linear(X):
    return X @ X.T

def linear_CKA(X, Y) -> float:
    """X, Y: [n, d] activations (centered automatically)."""
    K = _center_gram(_gram_linear(X))
    L = _center_gram(_gram_linear(Y))
    hsic = (K * L).sum()
    var1 = torch.sqrt((K * K).sum())
    var2 = torch.sqrt((L * L).sum())
    return (hsic / (var1 * var2 + 1e-12)).item()

def pwcca(X, Y) -> float:
    """
    PWCCA (projection-weighted CCA) as in Morcos et al. (2018).
    Simple approximation using torch.linalg.svd.
    """
    # Normalize and center
    Xc, Yc = X - X.mean(0, keepdim=True), Y - Y.mean(0, keepdim=True)
    Ux, Sx, _ = torch.linalg.svd(Xc, full_matrices=False)
    Uy, Sy, _ = torch.linalg.svd(Yc, full_matrices=False)
    C = Ux.T @ Uy
    corr = torch.linalg.svdvals(C)
    weights = Sx / Sx.sum()
    return float((weights * corr).sum().item())

def rsa(X, Y) -> float:
    """Representational Similarity Analysis (correlation of distance matrices)."""
    def pdist(v):
        d = torch.cdist(v, v, p=2)
        return d.flatten()
    Xd, Yd = pdist(X), pdist(Y)
    Xd, Yd = Xd - Xd.mean(), Yd - Yd.mean()
    corr = torch.sum(Xd * Yd) / (torch.norm(Xd) * torch.norm(Yd) + 1e-12)
    return corr.item()

def representation_metrics(Z1: torch.Tensor, Z2: torch.Tensor) -> Dict[str, float]:
    """Compute all representation metrics for two activation matrices [n, d]."""
    Z1, Z2 = Z1.detach(), Z2.detach()
    return {
        "cka": linear_CKA(Z1, Z2),
        "pwcca": pwcca(Z1, Z2),
        "rsa": rsa(Z1, Z2),
    }


@torch.no_grad()
def cka_between_models(
    model_original,
    model_forget,
    retain_loader,
    forget_loader,
    device: str = "cuda",
    max_samples: int = 500
) -> Dict[str, float]:
    """
    Compute CKA between representations of original and forget models.
    
    Args:
        model_original: Original model before unlearning
        model_forget: Model after unlearning
        retain_loader: DataLoader for retain set
        forget_loader: DataLoader for forget set
        device: Device to run on
        max_samples: Maximum number of samples to use (randomly selected)
    
    Returns:
        Dictionary with CKA scores for retain and forget sets
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model_original = model_original.to(device)
    model_forget = model_forget.to(device)
    model_original.eval()
    model_forget.eval()
    
    def extract_representations_paired(model_orig, model_fgt, loader, max_samples):
        """Extract representations from both models on the same inputs."""
        reprs_orig = []
        reprs_fgt = []
        total_samples = 0
        
        for x, _ in loader:
            x = x.to(device)
            
            # Get representation from original model
            if hasattr(model_orig, 'get_representation'):
                repr_orig = model_orig.get_representation(x)
            elif hasattr(model_orig, 'encoder'):
                repr_orig = model_orig.encoder(x)
            else:
                try:
                    _, repr_orig = model_orig(x, return_repr=True)
                except:
                    repr_orig = model_orig(x)
            
            # Get representation from forget model
            if hasattr(model_fgt, 'get_representation'):
                repr_fgt = model_fgt.get_representation(x)
            elif hasattr(model_fgt, 'encoder'):
                repr_fgt = model_fgt.encoder(x)
            else:
                try:
                    _, repr_fgt = model_fgt(x, return_repr=True)
                except:
                    repr_fgt = model_fgt(x)
            
            reprs_orig.append(repr_orig.cpu())
            reprs_fgt.append(repr_fgt.cpu())
            total_samples += repr_orig.shape[0]
            
            if total_samples >= max_samples:
                break
        
        reprs_orig = torch.cat(reprs_orig, dim=0)
        reprs_fgt = torch.cat(reprs_fgt, dim=0)
        
        # Randomly sample if we have more than max_samples
        if reprs_orig.shape[0] > max_samples:
            indices = torch.randperm(reprs_orig.shape[0])[:max_samples]
            reprs_orig = reprs_orig[indices]
            reprs_fgt = reprs_fgt[indices]
        
        return reprs_orig, reprs_fgt
    
    # Extract representations from both models on retain set (same inputs)
    retain_reprs_original, retain_reprs_forget = extract_representations_paired(
        model_original, model_forget, retain_loader, max_samples
    )
    
    # Extract representations from both models on forget set (same inputs)
    forget_reprs_original, forget_reprs_forget = extract_representations_paired(
        model_original, model_forget, forget_loader, max_samples
    )
    
    # Compute CKA
    cka_retain = linear_CKA(retain_reprs_original, retain_reprs_forget)
    cka_forget = linear_CKA(forget_reprs_original, forget_reprs_forget)
    
    return {
        "cka_retain": cka_retain,
        "cka_forget": cka_forget,
    }
