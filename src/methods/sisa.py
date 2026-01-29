# Python 3.8. No __main__ block.
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from copy import deepcopy
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.utils.data.dataset import Dataset


# ---------- config & small utils ----------

@dataclass
class _SISAConfig:
    shards: int = 10                 # S (paper): number of shards
    slices: int = 5                  # R (paper): number of slices per shard
    epochs_per_slice: int = 2        # e_i (we use constant e per slice)
    batch_size: Optional[int] = None
    num_workers: int = 4
    pin_memory: bool = True
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    aggregation: str = "logits"      # "logits" or "vote"  (paper discusses both label vote & vector avg)
    shuffle_seed: int = 42
    verbose: bool = True


def _get_xy_from_batch(batch):
    # Accept (x,y) or (x,*,y) style batches
    if isinstance(batch, (list, tuple)):
        x, y = batch[0], batch[-1]
    else:
        x, y = batch
    return x, y


def _device_of(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


class _EnsembleAvgLogits(nn.Module):
    """Wrap several models; forward returns avg logits; vote returns majority label."""
    def __init__(self, members: List[nn.Module], aggregation: str = "logits"):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.aggregation = aggregation

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Support return_repr for visualization
        return_repr = kwargs.get('return_repr', False)
        
        if return_repr:
            # For representation extraction, use the first member model
            # (all members should have similar representations after training on similar data)
            return self.members[0](x, **kwargs)
        
        if self.aggregation == "logits":
            out = None
            for m in self.members:
                logits = m(x, **kwargs)
                out = logits if out is None else out + logits
            return out / len(self.members)
        elif self.aggregation == "vote":
            # Majority over argmax
            votes = []
            for m in self.members:
                with torch.no_grad():
                    votes.append(m(x, **kwargs).argmax(dim=1, keepdim=True))
            votes = torch.cat(votes, dim=1)      # [B, num_models]
            # mode along dim=1
            # simple but effective: count frequencies
            B, K = votes.shape
            preds = []
            for i in range(B):
                vals, counts = votes[i].unique(return_counts=True)
                preds.append(vals[counts.argmax()])
            return F.one_hot(torch.stack(preds), num_classes=self.members[0](x[:1], **kwargs).shape[-1]).float()
        else:
            raise ValueError(f"Unknown aggregation {self.aggregation}")


# ---------- SISA core ----------

class SISA:
    """
    SISA: Sharded–Isolated–Sliced–Aggregated training & unlearning.
    - Shard: split the (retain ⊕ forget) training set into S disjoint shards.
    - Slice: within each shard, train incrementally across R slices; save checkpoints after each slice.
    - Unlearn: for shards containing forget samples, reload the last checkpoint *before* the affected slice and
      retrain onward on the data with forget samples removed.  (Fig. 2 & §IV-B.3).
    - Aggregate: average logits (or majority vote) across constituents at inference (§IV-B.4).
    """

    # ---------- pipeline hooks ----------
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
        self.cfg = _SISAConfig(
            shards=int(m.get("shards", 10)),
            slices=int(m.get("slices", 5)),
            epochs_per_slice=int(m.get("epochs_per_slice", 2)),
            batch_size=m.get("batch_size", None),
            num_workers=int(m.get("num_workers", 4)),
            pin_memory=bool(m.get("pin_memory", True)),
            lr=float(m.get("lr", 0.1)),
            momentum=float(m.get("momentum", 0.9)),
            weight_decay=float(m.get("weight_decay", 5e-4)),
            aggregation=str(m.get("aggregation", "logits")),
            shuffle_seed=int(m.get("shuffle_seed", 42)),
            verbose=bool(m.get("verbose", True)),
        )

        self.base_model = model  # template to clone
        self.retain_loader = retain_loader
        self.forget_loader = forget_loader
        self.val_loader = val_loader
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Will be filled in run()
        self.constituents: List[nn.Module] = []
        self.logs: Dict[str, Any] = {"per_shard": []}

        # Build a merged dataset view to shard on (consistent with SISA: each point belongs to exactly 1 shard)
        self.train_merged = ConcatDataset((retain_loader.dataset, forget_loader.dataset))

        # Track which global indices correspond to forget set (for quick membership)
        self._forget_global_idxs = self._global_indices_of_second_dataset(retain_loader.dataset, forget_loader.dataset)

    def run(self) -> None:
        cfg = self.cfg
        rng = random.Random(cfg.shuffle_seed)

        # 1) Create shard index lists (disjoint partition of all training indices).
        all_idx = list(range(len(self.train_merged)))
        rng.shuffle(all_idx)
        shards = self._chunk_disjoint(all_idx, cfg.shards)

        # 2) For each shard: create per-slice splits (in-order slices), train across slices saving checkpoints in RAM.
        for s_id, shard_idx in enumerate(shards):
            if cfg.verbose:
                print(f"\n[SISA] Training shard {s_id+1}/{cfg.shards}...")
            
            shard_forget_idx = [i for i in shard_idx if i in self._forget_global_idxs]
            per_shard_log: Dict[str, Any] = {
                "shard_id": s_id,
                "size": len(shard_idx),
                "forget_in_shard": len(shard_forget_idx),
            }

            # Slice split (uniform contiguous blocks after a deterministic shuffle for the shard)
            rng.shuffle(shard_idx)
            slices = self._chunk_disjoint(shard_idx, cfg.slices) if cfg.slices > 0 else [shard_idx]

            # Data for cumulative training per slice (Si = union_{j<=i} slice_j)
            cumulative_slices: List[List[int]] = []
            acc = []
            for sl in slices:
                acc.extend(sl)
                cumulative_slices.append(list(acc))

            # Build a model for this shard and train cumulatively, checkpoint after each slice
            model_s = self._fresh_member()
            checkpoints: List[Dict[str, torch.Tensor]] = []

            start_slice = 0
            for i, idxs in enumerate(cumulative_slices, start=1):
                if cfg.verbose:
                    print(f"  Training slice {i}/{len(cumulative_slices)} (cumulative size: {len(idxs)})...")
                self._train_one(model_s, Subset(self.train_merged, idxs),
                                epochs=cfg.epochs_per_slice, bs=cfg.batch_size)
                # save checkpoint after slice i
                checkpoints.append({k: v.detach().cpu() for k, v in model_s.state_dict().items()})

            # 3) Unlearning for this shard: if it has forget points, find earliest slice containing any forget idx.
            affected_slice = None
            if shard_forget_idx:
                forget_set = set(shard_forget_idx)
                for i, sl in enumerate(slices, start=1):
                    if any((g in forget_set) for g in sl):
                        affected_slice = i
                        break

            # If affected, reload last pre-affected checkpoint and retrain from that slice onward on data with forget removed.
            retrain_from = None
            if affected_slice is not None:
                if cfg.verbose:
                    print(f"  Shard {s_id+1} affected at slice {affected_slice}. Retraining from slice {affected_slice}...")
                
                retrain_from = affected_slice  # we must restart training from this slice index (1-based)
                # Restore to checkpoint of slice (affected_slice - 1). If 1 → cold start (initial member again).
                if affected_slice > 1:
                    model_s.load_state_dict(checkpoints[affected_slice - 2], strict=True)
                else:
                    model_s = self._fresh_member()  # reinit

                # Build *purged* cumulative sets from affected_slice..end
                # Start from the data up to (but not including) the affected slice, then add remaining slices
                purged_cumulatives: List[List[int]] = []
                
                # Initialize with data up to (but not including) affected_slice, with forget points removed
                if affected_slice > 1:
                    acc_purged = [g for g in cumulative_slices[affected_slice - 2] if g not in self._forget_global_idxs]
                else:
                    acc_purged = []
                
                # Now add each slice from affected_slice onwards (excluding forget points)
                for i in range(affected_slice - 1, len(slices)):
                    # Add this slice's data (excluding forget points)
                    slice_data_purged = [g for g in slices[i] if g not in self._forget_global_idxs]
                    acc_purged.extend(slice_data_purged)
                    purged_cumulatives.append(list(acc_purged))

                # Retrain from affected_slice to end on purged data
                for i, idxs in enumerate(purged_cumulatives, start=affected_slice):
                    if cfg.verbose:
                        print(f"  Retraining slice {i}/{len(slices)} (purged cumulative size: {len(idxs)})...")
                    self._train_one(model_s, Subset(self.train_merged, idxs),
                                    epochs=cfg.epochs_per_slice, bs=cfg.batch_size)

            # Keep the final shard model
            self.constituents.append(model_s)

            # logs
            per_shard_log.update({
                "affected_slice": affected_slice,
                "retrained_from_slice": retrain_from,
            })
            self.logs["per_shard"].append(per_shard_log)

        # Aggregation wrapper
        self.ensemble = _EnsembleAvgLogits(self.constituents, aggregation=cfg.aggregation)

    def get_model(self) -> nn.Module:
        return self.ensemble if hasattr(self, "ensemble") else self._fresh_member()

    def report(self) -> Dict[str, Any]:
        # Compact counts & a few ratios
        total = sum(s["size"] for s in self.logs["per_shard"])
        n_forget = sum(s["forget_in_shard"] for s in self.logs["per_shard"])
        n_affected = sum(1 for s in self.logs["per_shard"] if s["affected_slice"] is not None)
        return {
            "hparams": {
                "shards": self.cfg.shards,
                "slices": self.cfg.slices,
                "epochs_per_slice": self.cfg.epochs_per_slice,
                "lr": self.cfg.lr,
                "momentum": self.cfg.momentum,
                "weight_decay": self.cfg.weight_decay,
                "aggregation": self.cfg.aggregation,
                "batch_size": self.cfg.batch_size,
            },
            "logs": {
                "total_points": total,
                "forget_points": n_forget,
                "affected_shards": n_affected,
                "per_shard": self.logs["per_shard"],
            },
        }

    # ---------- helpers ----------

    def _fresh_member(self) -> nn.Module:
        m = deepcopy(self.base_model).to(self.device)
        m.train()
        return m

    def _train_one(self, model: nn.Module, dataset: Dataset, *, epochs: int, bs: Optional[int]) -> None:
        # Local loader
        base_loader = self.retain_loader  # reuse its worker/pin settings if possible
        batch_size = bs or getattr(base_loader, "batch_size", 128)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last=False,
        )
        opt = torch.optim.SGD(model.parameters(), lr=self.cfg.lr,
                              momentum=self.cfg.momentum, weight_decay=self.cfg.weight_decay)

        dev = _device_of(model)
        model.train()
        for epoch in range(max(epochs, 0)):
            total_loss = 0.0
            n_batches = 0
            
            for batch in loader:
                x, y = _get_xy_from_batch(batch)
                x = x.to(dev, non_blocking=True).float()
                y = y.to(dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                opt.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            # Log progress
            if self.cfg.verbose and n_batches > 0:
                avg_loss = total_loss / n_batches
                
                # Evaluate on retain and forget loaders
                model.eval()
                with torch.no_grad():
                    # Retain accuracy
                    correct_retain, total_retain = 0, 0
                    for batch in self.retain_loader:
                        x, y = _get_xy_from_batch(batch)
                        x = x.to(dev, non_blocking=True).float()
                        y = y.to(dev, non_blocking=True)
                        logits = model(x)
                        preds = logits.argmax(dim=1)
                        correct_retain += (preds == y).sum().item()
                        total_retain += y.size(0)
                    acc_retain = correct_retain / max(total_retain, 1)
                    
                    # Forget accuracy
                    correct_forget, total_forget = 0, 0
                    for batch in self.forget_loader:
                        x, y = _get_xy_from_batch(batch)
                        x = x.to(dev, non_blocking=True).float()
                        y = y.to(dev, non_blocking=True)
                        logits = model(x)
                        preds = logits.argmax(dim=1)
                        correct_forget += (preds == y).sum().item()
                        total_forget += y.size(0)
                    acc_forget = correct_forget / max(total_forget, 1)
                
                model.train()
                
                print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Retain Acc: {acc_retain:.4f} | Forget Acc: {acc_forget:.4f}")

    @staticmethod
    def _chunk_disjoint(idxs: List[int], parts: int) -> List[List[int]]:
        n = len(idxs)
        if parts <= 1:
            return [list(idxs)]
        size = int(math.ceil(n / float(parts)))
        return [idxs[i * size:(i + 1) * size] for i in range(parts) if i * size < n]

    @staticmethod
    def _global_indices_of_second_dataset(ds1: Dataset, ds2: Dataset) -> set:
        """Return the set of global indices (in Concat(ds1, ds2)) that belong to ds2."""
        base = len(ds1)
        return set(range(base, base + len(ds2)))