# src/methods/sisa/method.py
from typing import Any, Dict, Optional
import copy, time, os
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from .base import IUnlearningMethod
from ..helpers.registry import register
from ..helpers.train_utils import evaluate_acc


@register("sisa")
class SISA(IUnlearningMethod):
    """
    SISA Training (Sharded, Isolated, Sliced, Aggregated) from
    "Machine Unlearning" (Bourtoule et al., IEEE S&P 2021).

    Single-model pipeline adaptation
    ---------------------------------
    setup():
        - Splits the full dataset (retain + forget) into num_slices slices.
        - Forget samples are appended last so they always land in the final
          slice, making all earlier checkpoints clean by construction.
        - Starts from the pre-trained weights and continues training
          slice-by-slice, saving a checkpoint after each slice.

    run():
        - Finds the latest slice checkpoint containing no forget samples.
        - Reloads it and retrains only the remaining slices without forget data.
        - Returns the single retrained model directly.
    """

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def setup(
        self,
        model: nn.Module,
        *,
        retain_loader: DataLoader,
        forget_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        cfg: Dict[str, Any],
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.val_loader = val_loader
        self.cfg = cfg

        hp = cfg.get("method", {})
        self.num_slices       = int(hp.get("num_slices", 4))
        self.epochs_per_slice = int(hp.get("epochs_per_slice", 15))
        self.checkpoint_dir   = str(hp.get("checkpoint_dir", "/tmp/sisa_checkpoints"))
        self.batch_size       = retain_loader.batch_size
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # --- Optimizer (SGD, flat lr, no scheduler) ---
        self.optimizer_name = str(hp.get("optimizer", "sgd")).lower()
        self.lr             = float(hp.get("lr", 0.01))
        self.weight_decay   = float(hp.get("weight_decay", 5e-4))
        self.momentum       = float(hp.get("momentum", 0.9))
        self.betas          = tuple(hp.get("betas", [0.9, 0.999]))
        self.max_norm       = float(hp.get("max_norm", 0.0))

        # Store architecture and pre-trained weights.
        self._arch            = copy.deepcopy(model).cpu()
        self.pretrained_state = copy.deepcopy(model).cpu().state_dict()

        # ------------------------------------------------------------------
        # Dataset: retain first, forget appended at the end.
        # ------------------------------------------------------------------
        retain_size = len(retain_loader.dataset)
        forget_size = len(forget_loader.dataset)

        self.full_dataset = _ConcatDataset(
            retain_loader.dataset, forget_loader.dataset
        )
        self.forget_global_indices = set(
            range(retain_size, retain_size + forget_size)
        )
        self.all_indices = list(range(retain_size + forget_size))

        print(
            f"[SISA] {retain_size} retain + {forget_size} forget | "
            f"{self.num_slices} slices x {self.epochs_per_slice} epochs/slice | "
            f"optimizer={self.optimizer_name} lr={self.lr}"
        )

        self._model = self._train(
            init_state=self.pretrained_state,
            exclude_indices=set(),
        )
        self._report: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    def run(self) -> None:
        start = time.time()

        n_total    = len(self.all_indices)
        slice_size = max(1, n_total // self.num_slices)

        last_clean_slice = -1
        for sl in range(self.num_slices - 1, -1, -1):
            until = min((sl + 1) * slice_size, n_total)
            if set(self.all_indices[:until]).isdisjoint(self.forget_global_indices):
                last_clean_slice = sl
                break

        if last_clean_slice >= 0:
            ckpt_path = self._ckpt_path(last_clean_slice)
            if os.path.exists(ckpt_path):
                init_state  = torch.load(ckpt_path, map_location=self.device)
                start_slice = last_clean_slice + 1
                print(
                    f"[SISA] Resuming from slice {last_clean_slice} "
                    f"-> retraining slices {start_slice}..{self.num_slices - 1}"
                )
            else:
                init_state  = self.pretrained_state
                start_slice = 0
                print("[SISA] Clean checkpoint missing -> retraining from pre-trained weights.")
        else:
            init_state  = self.pretrained_state
            start_slice = 0
            print("[SISA] Forget data in every slice -> retraining from pre-trained weights.")

        self._model = self._train(
            init_state=init_state,
            exclude_indices=self.forget_global_indices,
            start_slice=start_slice,
        )

        train_time = time.time() - start
        acc_val = (
            evaluate_acc(self._model, self.val_loader, self.device)
            if self.val_loader else None
        )

        self._report.update({
            "method": "sisa",
            "num_slices": self.num_slices,
            "epochs_per_slice": self.epochs_per_slice,
            "last_clean_slice": last_clean_slice,
            "train_time_sec": train_time,
            "val_acc": acc_val,
        })

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _fresh_model(self, init_state: dict) -> nn.Module:
        m = copy.deepcopy(self._arch)
        m.load_state_dict(init_state)
        return m.to(self.device)

    def _make_optimizer(self, model: nn.Module) -> optim.Optimizer:
        if self.optimizer_name == "adamw":
            return optim.AdamW(
                model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=self.betas,
            )
        elif self.optimizer_name == "adam":
            return optim.Adam(
                model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=self.betas,
            )
        elif self.optimizer_name == "sgd":
            return optim.SGD(
                model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer '{self.optimizer_name}'.")

    def _train(
        self,
        init_state: dict,
        exclude_indices: set,
        start_slice: int = 0,
    ) -> nn.Module:
        """
        Continue training from init_state with sliced checkpointing.
        No lr scheduler — flat lr throughout, matching the working config.
        """
        model     = self._fresh_model(init_state)
        optimizer = self._make_optimizer(model)

        clean_indices = [i for i in self.all_indices if i not in exclude_indices]
        n_clean       = len(clean_indices)
        slice_size    = max(1, n_clean // self.num_slices)

        for sl in range(start_slice, self.num_slices):
            until  = min((sl + 1) * slice_size, n_clean)
            loader = DataLoader(
                Subset(self.full_dataset, clean_indices[:until]),
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
            )

            model.train()
            for _ in range(self.epochs_per_slice):
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    loss = F.cross_entropy(model(x), y)
                    optimizer.zero_grad()
                    loss.backward()
                    if self.max_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)
                    optimizer.step()

            ckpt_path = self._ckpt_path(sl)
            torch.save(model.state_dict(), ckpt_path)
            print(
                f"  Slice {sl + 1}/{self.num_slices} "
                f"({until}/{n_clean} samples) -> {ckpt_path}"
            )

        return model

    def _ckpt_path(self, slice_idx: int) -> str:
        return os.path.join(self.checkpoint_dir, f"slice_{slice_idx}.pt")

    # ------------------------------------------------------------------
    # IUnlearningMethod interface
    # ------------------------------------------------------------------
    def get_model(self) -> nn.Module:
        return self._model

    def report(self) -> Dict[str, Any]:
        return self._report


# ---------------------------------------------------------------------------
# Helper classes
# ---------------------------------------------------------------------------

class _ConcatDataset(Dataset):
    """
    Retain-first concatenation.
    Forget samples always occupy [retain_size, retain_size + forget_size),
    so they land in the last slice and all earlier checkpoints are clean.
    """
    def __init__(self, ds_retain, ds_forget):
        self.ds_retain    = ds_retain
        self.ds_forget    = ds_forget
        self._retain_size = len(ds_retain)

    def __len__(self):
        return self._retain_size + len(self.ds_forget)

    def __getitem__(self, idx):
        if idx < self._retain_size:
            return self.ds_retain[idx]
        return self.ds_forget[idx - self._retain_size]