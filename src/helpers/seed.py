# src/helpers/seed.py
import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set all relevant random seeds to ensure reproducibility across:
      - Python's `random`
      - NumPy
      - PyTorch (CPU, CUDA, cuDNN)
    
    Args:
        seed (int): The seed to set.
        deterministic (bool): If True, enforce deterministic algorithms
            (slower but fully reproducible). If False, allow nondeterministic
            CuDNN algorithms for performance.
    """
    # ---- Python & NumPy ----
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # ---- PyTorch ----
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # optional: reproducibility flag for other libs
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    print(f"[set_seed] Using seed={seed} (deterministic={deterministic})")
