"""
Simple training script for toy experiments.
Example usage:
    bin/train_toy --seed 0
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn, optim

from src.helpers import read_config
from src.helpers.io import get_repo_root
from src.helpers.seed import set_seed
from src.helpers.train_utils import evaluate_acc, build_optimizer, build_scheduler
from src.datasets import get_loader
from src.models import get_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="toy.yaml",
                        help="Config file in configs/ directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    # Read config
    cfg = read_config(args.config)
    
    deterministic = cfg.get("deterministic", True)
    set_seed(args.seed, deterministic=deterministic)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Setup output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ds_name = cfg["dataset"]["name"]
        model_name = cfg["model"]["name"]
        out_dir = get_repo_root() / "results" / "train" / ds_name / model_name / f"seed{args.seed}"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)
    
    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Data loaders
    train_loader = get_loader(cfg["dataset"], split="train")
    test_loader = get_loader(cfg["dataset"], split="test")
    
    # Model
    model = get_model(**cfg["model"]).to(device)
    
    # Training setup
    training_cfg = cfg.get("training", {})
    optimizer_cfg = cfg.get("optimizer", {})
    scheduler_cfg = cfg.get("scheduler", {})
    
    epochs = training_cfg.get("epochs", 100)
    max_norm = training_cfg.get("max_norm", 0.0)
    
    optimizer = build_optimizer(
        model,
        name=optimizer_cfg.get("name", "adam"),
        lr=optimizer_cfg.get("lr", 0.001),
        weight_decay=optimizer_cfg.get("weight_decay", 0.0),
        cfg=optimizer_cfg,
    )
    
    scheduler = None
    scheduler_name = scheduler_cfg.get("name", "none")
    if scheduler_name and scheduler_name.lower() != "none":
        scheduler = build_scheduler(
            optimizer,
            name=scheduler_name,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            max_lr=optimizer_cfg.get("lr", 0.001),
            cfg=scheduler_cfg,
        )
    
    loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    print(f"[Train] Starting training for {epochs} epochs")
    start_time = time.time()
    
    history = {"train_loss": [], "test_acc": []}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            optimizer.step()
            
            if scheduler and scheduler_name.lower() == "onecycle":
                scheduler.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        history["train_loss"].append(avg_loss)
        
        # Evaluate
        test_acc = evaluate_acc(model, test_loader, device)
        history["test_acc"].append(test_acc)
        
        if scheduler and scheduler_name.lower() != "onecycle":
            scheduler.step()
        
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            lr = optimizer.param_groups[0]['lr']
            print(f"[Train] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Test Acc: {test_acc:.4f}, LR: {lr:.6f}")
    
    train_time = time.time() - start_time
    
    # Save model
    torch.save(model.state_dict(), out_dir / "models" / "model_base.pt")
    
    # Save results
    final_acc = evaluate_acc(model, test_loader, device)
    results = {
        "seed": args.seed,
        "epochs": epochs,
        "train_time": train_time,
        "final_test_acc": final_acc,
        "final_train_loss": history["train_loss"][-1],
    }
    
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"[Train] Training completed in {train_time:.2f}s")
    print(f"[Train] Final test accuracy: {final_acc:.4f}")
    print(f"[Train] Results saved to {out_dir}")


if __name__ == "__main__":
    main()
