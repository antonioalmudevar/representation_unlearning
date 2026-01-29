# src/experiments/train.py
import os
import json
import argparse
import time
from pathlib import Path
from typing import Dict, Any

import torch
from torch import nn, optim

from src.helpers import read_config
from src.helpers.seed import set_seed
from src.datasets import get_loader
from src.models import get_model
from src.helpers.train_utils import evaluate_acc, build_optimizer, build_scheduler

REPO_ROOT = Path(__file__).resolve().parents[2]

def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model

def _make_outdir(cfg: Dict[str, Any], seed: int, out_dir: str):
    if out_dir:
        root = out_dir
    else:
        # prefer dataset/model names from cfg
        dataset = cfg["dataset"]["name"]
        model_name = cfg["model"]["name"]
        root = REPO_ROOT / "results" / "train" / dataset / model_name / f"seed{seed}"
        root = str(root)
    Path(root).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(root, "metrics")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(root, "models")).mkdir(parents=True, exist_ok=True)
    return root

@torch.no_grad()
def _evaluate(model: nn.Module, loader, device: torch.device) -> Dict[str, float]:
    return {"acc": evaluate_acc(model, loader, device)}

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    device: torch.device,
    loss_fn: nn.Module,
    scaler: torch.cuda.amp.GradScaler = None,
    max_norm: float = 0.0,
):
    model.train()
    running_loss, n_samples, n_correct = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and scaler.is_enabled():
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            if max_norm and max_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            if max_norm and max_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        running_loss += float(loss.detach().cpu()) * y.size(0)
        n_samples += y.size(0)
        n_correct += (logits.argmax(1) == y).sum().item()

    return {
        "loss": running_loss / max(1, n_samples),
        "acc": n_correct / max(1, n_samples),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=None, help="override epochs in config")
    parser.add_argument("--parallel", action="store_true", help="wrap model in DataParallel")
    parser.add_argument("--amp", action="store_true", help="enable automatic mixed precision")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = read_config(args.config)

    # ----- Config blocks (NEW layout + backward compat) -----
    # Preferred:
    #   training: {epochs, max_norm, amp, deterministic}
    #   optimizer: {name, lr, weight_decay, ...}
    #   scheduler: {name, ...}
    # Back-compat: fall back to "hyperparams" if present
    train_cfg = cfg.get("training", cfg.get("hyperparams", {}))
    opt_cfg   = cfg.get("optimizer", {})
    sch_cfg   = cfg.get("scheduler", {})

    # training-loop knobs
    deterministic = bool(train_cfg.get("deterministic", True))
    set_seed(args.seed, deterministic=deterministic)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- Data
    ds_cfg = cfg["dataset"]
    train_loader = get_loader(ds_cfg, split="train")
    val_loader = get_loader(ds_cfg, split="val") if ds_cfg.get("has_val", True) else None
    test_loader = get_loader(ds_cfg, split="test")

    # --- Model
    model = get_model(**cfg["model"]).to(device)
    if args.parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # --- Epochs & AMP
    epochs = args.epochs if args.epochs is not None else int(train_cfg.get("epochs", 100))
    max_norm = float(train_cfg.get("max_norm", 0.0))
    # CLI --amp still works; config 'training.amp' also works (CLI wins if set)
    cfg_amp = bool(train_cfg.get("amp", False))
    amp_enabled = bool(args.amp or cfg_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # --- Optimizer (from optimizer block; fallback to old hyperparams)
    fallback_lr = float(train_cfg.get("lr", 1e-3))
    fallback_wd = float(train_cfg.get("weight_decay", 1e-4))
    opt_name = str(opt_cfg.get("name", "adamw")).lower()
    lr = float(opt_cfg.get("lr", fallback_lr))
    weight_decay = float(opt_cfg.get("weight_decay", fallback_wd))
    optimizer = build_optimizer(model, name=opt_name, lr=lr, weight_decay=weight_decay, cfg=opt_cfg)

    # --- Scheduler (separate block; fallback to old hyperparams.scheduler)
    scheduler_name = str(sch_cfg.get("name", train_cfg.get("scheduler", "cosine"))).lower()
    scheduler = build_scheduler(
        optimizer,
        name=scheduler_name,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        max_lr=lr,
        cfg=sch_cfg
    )

    loss_fn = nn.CrossEntropyLoss()

    # --- Output dirs
    out_dir = _make_outdir(cfg, args.seed, args.out_dir)
    with open(os.path.join(out_dir, "cfg.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # --- Train loop
    best_val_acc = -1.0
    history = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(
            model, train_loader, optimizer, device, loss_fn, scaler=scaler, max_norm=max_norm
        )

        # step scheduler
        if scheduler is not None:
            scheduler.step()

        # eval
        val_stats = _evaluate(model, val_loader, device) if val_loader else {}
        test_stats = _evaluate(model, test_loader, device)

        # save best
        val_acc = val_stats.get("acc", test_stats["acc"])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(_unwrap(model).state_dict(), os.path.join(out_dir, "models", "model_best.pt"))

        # save last every epoch
        torch.save(_unwrap(model).state_dict(), os.path.join(out_dir, "models", "model_last.pt"))

        # log row
        lr_now = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["acc"],
            "val_acc": val_stats.get("acc", None),
            "test_acc": test_stats["acc"],
        }
        history.append(row)
        print(f"[epoch {epoch:03d}] "
              f"loss={row['train_loss']:.4f} acc={row['train_acc']:.4f} "
              f"val_acc={row['val_acc'] if row['val_acc'] is not None else float('nan'):.4f} "
              f"test_acc={row['test_acc']:.4f} lr={lr_now:.3e}")

    elapsed = time.time() - t0

    # save history CSV and summary
    try:
        import pandas as pd
        pd.DataFrame(history).to_csv(os.path.join(out_dir, "metrics", "history.csv"), index=False)
    except Exception:
        with open(os.path.join(out_dir, "metrics", "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    # final base checkpoint (alias)
    torch.save(_unwrap(model).state_dict(), os.path.join(out_dir, "models", "model_base.pt"))

    summary = {
        "best_val_acc": best_val_acc,
        "final_test_acc": history[-1]["test_acc"] if history else None,
        "epochs": epochs,
        "elapsed_sec": elapsed,
        "amp": bool(amp_enabled),
        "parallel": bool(args.parallel),
        "scheduler": scheduler_name,
        "optimizer": opt_name,
        "lr": lr,
        "weight_decay": weight_decay,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
