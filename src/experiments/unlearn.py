# src/experiments/unlearn.py
import json
import time
import argparse
from pathlib import Path

import torch

from src.helpers import read_config
from src.helpers.io import get_repo_root
from src.helpers.seed import set_seed
from src.helpers.train_utils import evaluate_acc, evaluate_acc_by_class
from src.methods import get_method
from src.datasets import get_loader
from src.models import get_model
from src.metrics import *

def _protocol_tag(proto):
    ptype = proto.get("type", "class_forget")
    if ptype == "class_forget":
        forget_classes = proto.get("forget_classes", [])
        n = len(forget_classes) if isinstance(forget_classes, (list, tuple)) else 1
        return "cls%d" % n
    if ptype == "random_forget":
        r = float(proto.get("forget_ratio", 0.0))
        return "rand%d" % int(r * 100)
    if ptype == "identity_forget":
        ids = proto.get("identity_ids", [])
        return "id%d" % len(ids)
    if ptype == "attr_forget":
        a = int(proto.get("attr_index", 0))
        v = int(proto.get("attr_value", 1))
        return "attr%d_%d" % (a, v)
    return ptype

def _default_base_ckpt(cfg, seed):
    ds_name = cfg["dataset"]["name"]
    model_name = cfg["model"]["name"]
    return get_repo_root() / "results" / "train" / ds_name / model_name / ("seed%d" % seed) / "models" / "model_base.pt"

def _default_retrain_ckpt(cfg, seed):
    """Find the retrain baseline checkpoint for the current dataset and protocol."""
    ds_name = cfg["dataset"]["name"]
    tag = _protocol_tag(cfg["dataset"].get("split_protocol", {}))
    return get_repo_root() / "results" / "unlearn" / ds_name / tag / "retrain" / ("seed%d" % seed) / "models" / "model_forget.pt"

def _maybe_inject_checkpoint(cfg, seed):
    cfg = dict(cfg)
    cfg["model"] = dict(cfg["model"])
    if not cfg["model"].get("checkpoint"):
        ckpt = _default_base_ckpt(cfg, seed)
        if ckpt.exists():
            cfg["model"]["checkpoint"] = str(ckpt)
            print("[unlearn] Using default base checkpoint:", ckpt)
        else:
            print("[unlearn] No checkpoint provided and default not found:", ckpt)
            print("[unlearn] Proceeding from random initialization.")
    return cfg

def _make_outdir(cfg, method_name, seed, out_dir):
    if out_dir:
        root = Path(out_dir)
        # Convert to absolute path
        if not root.is_absolute():
            root = get_repo_root() / root
        root = root.resolve()
        print(f"[unlearn] Using absolute output directory: {root}")
    else:
        dataset = cfg["dataset"]["name"]
        tag = _protocol_tag(cfg["dataset"].get("split_protocol", {}))
        root = get_repo_root() / "results" / "unlearn" / dataset / tag / method_name / ("seed%d" % seed)
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    return root

def _save_json(obj, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str,
                        help="Config file path (absolute or relative identifier like 'subfolder/file_name')")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (absolute or relative path)")
    args = parser.parse_args()
    
    # Convert config to absolute path if it exists as a file
    config_path = args.config
    if Path(config_path).exists():
        config_path = str(Path(config_path).resolve())
        print(f"[unlearn] Using absolute config path: {config_path}")

    # Single-file config
    base_cfg = read_config(config_path)

    # Validate presence of method block
    if "method" not in base_cfg or not isinstance(base_cfg["method"], dict) or "name" not in base_cfg["method"]:
        raise ValueError("[unlearn] Config must contain method.name (single-file setup).")

    # Only top-level 'deterministic' is used now
    deterministic = bool(base_cfg.get("deterministic", True))

    cfg = _maybe_inject_checkpoint(base_cfg, args.seed)
    set_seed(args.seed, deterministic=deterministic)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Data
    ds_cfg = cfg["dataset"]
    retain_loader = get_loader(ds_cfg, split="retain")
    forget_loader = get_loader(ds_cfg, split="forget")
    val_loader = get_loader(ds_cfg, split="val") if ds_cfg.get("has_val", True) else None
    test_loader = get_loader(ds_cfg, split="test")

    # Model
    model = get_model(**cfg["model"]).to(device)
    
    # Keep a copy of the original model for divergence computation
    import copy
    model_original = copy.deepcopy(model)
    model_original.eval()

    # Method
    method_name = cfg["method"]["name"]
    Method = get_method(method_name)
    method = Method()
    method.setup(
        model,
        retain_loader=retain_loader,
        forget_loader=forget_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=str(device),
    )

    # Run + evaluate
    t0 = time.time()
    method.run()
    t1 = time.time()
    model_forget = method.get_model()

    # Use custom method name if the method provides one (for hyperparameter-specific naming)
    if hasattr(method, 'get_method_name'):
        method_name = method.get_method_name()
    
    out_dir = _make_outdir(cfg, method_name, args.seed, args.out_dir)
    _save_json(cfg, out_dir / "cfg.json")
    
    # Handle report() safely
    if hasattr(method, 'report'):
        _save_json(method.report(), out_dir / "report.json")

    # --- Metrics Calculation ---
    cls_ret = classification_metrics(model_forget, retain_loader, str(device))
    cls_for = classification_metrics(model_forget, forget_loader, str(device))
    cls_tst = classification_metrics(model_forget, test_loader, str(device))
    
    # [UPDATED] Direct call to membership_inference_attack with 3 loaders
    mia = membership_inference_attack(
        model_forget, 
        retain_loader=retain_loader, 
        forget_loader=forget_loader, 
        test_loader=test_loader,
        device=str(device)
    )
    
    # Load retrain baseline for divergence computation
    retrain_ckpt = _default_retrain_ckpt(cfg, args.seed)
    divergence = {}
    
    if retrain_ckpt.exists():
        print(f"[unlearn] Loading retrain baseline from {retrain_ckpt}")
        try:
            # Create a fresh model instance to ensure architecture matches the checkpoint
            model_retrain = get_model(**cfg["model"]).to(device)
            model_retrain.load_state_dict(torch.load(retrain_ckpt, map_location=device))
            model_retrain.eval()
            
            # Output divergence between retrain baseline and unlearned models
            divergence = cross_entropy_divergence(
                model_retrain,
                model_forget,
                retain_loader=retain_loader,
                forget_loader=forget_loader,
                test_loader=test_loader,
                device=str(device)
            )
        except Exception as e:
            print(f"[unlearn] Error loading retrain model: {e}")
    else:
        print(f"[unlearn] Retrain checkpoint not found: {retrain_ckpt}")
    
    eff = efficiency_metrics(model_forget, start_time=t0, end_time=t1, device=str(device))

    # CKA between original and forget model representations
    cka_metrics = cka_between_models(
        model_original,
        model_forget,
        retain_loader=retain_loader,
        forget_loader=forget_loader,
        device=str(device)
    )
    
    # Visualization for toy experiments with 2D representation
    if cfg["model"].get("repr_dim") == 2 or cfg["dataset"]["name"] == "toy":
        print("[unlearn] Generating representation space visualizations...")
        try:
            # Individual plots
            plot_representation_space(
                model_original,
                retain_loader,
                forget_loader,
                test_loader,
                device=str(device),
                save_path=out_dir / "figures" / "representation_before.pdf",
                title="Representation Space - Before Unlearning",
            )
            
            # Plot after unlearning with its own axis limits
            plot_representation_space(
                model_forget,
                retain_loader,
                forget_loader,
                test_loader,
                device=str(device),
                save_path=out_dir / "figures" / "representation_after.pdf",
                title="Representation Space - After Unlearning",
            )
        except Exception as e:
            print(f"[unlearn] Error generating visualizations: {e}")

    try:
        import pandas as pd
        pd.DataFrame([cls_ret]).to_csv(out_dir / "metrics" / "retained.csv", index=False)
        pd.DataFrame([cls_for]).to_csv(out_dir / "metrics" / "forget.csv", index=False)
        pd.DataFrame([cls_tst]).to_csv(out_dir / "metrics" / "test.csv", index=False)
        pd.DataFrame([mia]).to_csv(out_dir / "metrics" / "privacy_mia.csv", index=False)
        if divergence:
            pd.DataFrame([divergence]).to_csv(out_dir / "metrics" / "output_divergence.csv", index=False)
        pd.DataFrame([eff]).to_csv(out_dir / "metrics" / "efficiency.csv", index=False)
        pd.DataFrame([cka_metrics]).to_csv(out_dir / "metrics" / "cka.csv", index=False)
    except Exception as e:
        print("[unlearn] pandas not available or error writing CSVs:", str(e))

    torch.save(model_forget.state_dict(), out_dir / "models" / "model_forget.pt")

    # Evaluate test accuracy
    # Check if this is a class_forget protocol to report split accuracies
    split_protocol = cfg.get("dataset", {}).get("split_protocol", {})
    protocol_type = split_protocol.get("type", "class_forget")
    
    if protocol_type == "class_forget":
        forget_classes = split_protocol.get("forget_classes", [])
        if isinstance(forget_classes, (list, tuple)) and len(forget_classes) > 0:
            # Get all classes from the dataset
            num_classes = cfg.get("model", {}).get("num_classes", 10)
            all_classes = set(range(num_classes))
            forget_class_set = set(forget_classes)
            retain_class_set = all_classes - forget_class_set
            
            # Evaluate with class split
            test_acc_dict = evaluate_acc_by_class(
                model_forget.to(device), 
                test_loader, 
                device,
                retain_classes=list(retain_class_set),
                forget_classes=list(forget_class_set)
            )
            summary = {
                "method": method_name, 
                "seed": args.seed,
                "test_acc": float(test_acc_dict["test_acc"])
            }
            if "test_acc_retain" in test_acc_dict:
                summary["test_acc_retain"] = float(test_acc_dict["test_acc_retain"])
            if "test_acc_forget" in test_acc_dict:
                summary["test_acc_forget"] = float(test_acc_dict["test_acc_forget"])
        else:
            # Fallback to regular evaluation
            test_acc_scalar = evaluate_acc(model_forget.to(device), test_loader, device)
            summary = {"method": method_name, "seed": args.seed, "test_acc": float(test_acc_scalar)}
    else:
        # For non-class_forget protocols, use regular evaluation
        test_acc_scalar = evaluate_acc(model_forget.to(device), test_loader, device)
        summary = {"method": method_name, "seed": args.seed, "test_acc": float(test_acc_scalar)}
    
    for k, v in cls_ret.items():
        if k != "confusion":
            summary["ret_" + k] = v
    for k, v in cls_for.items():
        if k != "confusion":
            summary["for_" + k] = v
            
    # Update summary with MIA and divergence metrics
    summary.update(mia)
    summary.update(divergence)
    summary.update(eff)
    summary.update(cka_metrics)
    
    _save_json(summary, out_dir / "summary.json")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()