# src/methods/amun/method.py
from typing import Any, Dict
import copy, time
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from .base import IUnlearningMethod
from ..helpers.registry import register
from ..helpers.train_utils import evaluate_acc


class _PGDL2:
    """
    PGD attack in L2 norm.
    Maximises cross-entropy loss w.r.t. the true label within an L2 ball.
    """

    def __init__(self, model: nn.Module, eps: float, alpha: float,
                 n_steps: int = 10, random_start: bool = True,
                 device: torch.device = None):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.n_steps = n_steps
        self.random_start = random_start
        self.device = device or torch.device("cpu")

    @torch.enable_grad()
    def perturb(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return adversarial examples. x expected in [0, 1]."""
        x_adv = x.clone().detach()

        if self.random_start:
            delta = torch.randn_like(x_adv)
            delta = self._project_l2(delta, self.eps)
            x_adv = (x_adv + delta).clamp(0.0, 1.0).detach()

        for _ in range(self.n_steps):
            x_adv.requires_grad_(True)
            loss = F.cross_entropy(self.model(x_adv), y)
            grad = torch.autograd.grad(loss, x_adv)[0]

            with torch.no_grad():
                grad_norm = grad.flatten(1).norm(dim=1).clamp(min=1e-12)
                grad_unit = grad / grad_norm.view(-1, *([1] * (grad.dim() - 1)))
                x_adv = x_adv + self.alpha * grad_unit
                delta = self._project_l2(x_adv - x, self.eps)
                x_adv = (x + delta).clamp(0.0, 1.0).detach()

        return x_adv

    @staticmethod
    def _project_l2(delta: torch.Tensor, eps: float) -> torch.Tensor:
        norm = delta.flatten(1).norm(dim=1).clamp(min=1e-12)
        factor = (norm / eps).clamp(min=1.0)
        return delta / factor.view(-1, *([1] * (delta.dim() - 1)))


@register("amun")
class AMUN(IUnlearningMethod):
    """
    Adversarial Machine UNlearning (AMUN).
    "Not All Wrong is Bad: Using Adversarial Examples for Unlearning"
    OpenReview: BkrIQPREkn

    Algorithm:
      1. Compute adversarial examples for the forget set via PGD-L2.
         Labels are the model's own prediction on x_adv: when PGD succeeds
         this is a wrong class, driving forgetting.
      2. Fine-tune on retain set (true labels) + adversarial forget set
         (model-predicted labels) with SGD + StepLR.

    Config keys (under method:):
        epochs          (int,   default 10)
        lr              (float, default 0.05)
        momentum        (float, default 0.9)
        weight_decay    (float, default 5e-4)
        lr_step_size    (int,   default 5)
        lr_gamma        (float, default 0.1)
        pgd_eps         (float, default 2.0)   L2 perturbation radius
        pgd_alpha       (float, default 0.2)   PGD step size
        pgd_steps       (int,   default 20)    PGD iterations
        pgd_random_start  (bool,  default True)
        use_retain        (bool,  default True)  mix retain data during fine-tuning
        use_shadow_model  (bool,  default False) zero-shot variant: generate adversarial
                                                 examples on a shadow model trained only
                                                 on the forget set, then fine-tune target
                                                 model without any retain data
        shadow_epochs     (int,   default 10)    epochs to train the shadow model
        shadow_lr         (float, default 0.1)   lr for shadow model training
    """

    def setup(self, model, *, retain_loader, forget_loader, val_loader=None,
              cfg: Dict[str, Any], device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self._model = copy.deepcopy(model).to(self.device)
        self.retain_loader = retain_loader
        self.forget_loader = forget_loader
        self.val_loader = val_loader

        hp = cfg.get("method", {})
        self.epochs       = int(hp.get("epochs", 10))
        self.lr           = float(hp.get("lr", 0.05))
        self.momentum     = float(hp.get("momentum", 0.9))
        self.weight_decay = float(hp.get("weight_decay", 5e-4))
        self.lr_step_size = int(hp.get("lr_step_size", 5))
        self.lr_gamma     = float(hp.get("lr_gamma", 0.1))
        self.pgd_eps      = float(hp.get("pgd_eps", 2.0))
        self.pgd_alpha    = float(hp.get("pgd_alpha", 0.2))
        self.pgd_steps    = int(hp.get("pgd_steps", 20))
        self.pgd_random   = bool(hp.get("pgd_random_start", True))
        self.use_retain        = bool(hp.get("use_retain", True))
        self.use_shadow_model  = bool(hp.get("use_shadow_model", False))
        self.shadow_epochs     = int(hp.get("shadow_epochs", 10))
        self.shadow_lr         = float(hp.get("shadow_lr", 0.1))
        # Store model config for building shadow model
        self._cfg = cfg

        self._report = {}

    # ------------------------------------------------------------------
    def run(self) -> None:
        start = time.time()

        # ---- Step 1: compute adversarial forget set ------------------
        if self.use_shadow_model:
            print("[AMUN] Training shadow model on forget set...")
            shadow = self._train_shadow_model()
            print("[AMUN] Computing transferred adversarial examples via shadow model...")
            adv_data, adv_labels = self._build_adv_forget_set(attack_model=shadow)
        else:
            print("[AMUN] Computing adversarial examples for the forget set...")
            adv_data, adv_labels = self._build_adv_forget_set()
        print(f"[AMUN] Adversarial forget set size: {len(adv_data)}")

        # ---- Step 2: build combined loader ---------------------------
        combined_loader = self._build_combined_loader(adv_data, adv_labels)

        # ---- Step 3: fine-tune ---------------------------------------
        optimizer = optim.SGD(self._model.parameters(), lr=self.lr,
                              momentum=self.momentum,
                              weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)
        criterion = torch.nn.CrossEntropyLoss()
        epoch_losses = []

        print(f"[AMUN] Fine-tuning for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            self._model.train()
            total_loss, n_batches = 0.0, 0

            for x, y in combined_loader:
                x, y = x.to(self.device), y.to(self.device)
                loss = criterion(self._model(x), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = total_loss / max(n_batches, 1)
            epoch_losses.append(avg_loss)

            self._model.eval()
            with torch.no_grad():
                acc_r = evaluate_acc(self._model, self.retain_loader, self.device)
                acc_f = evaluate_acc(self._model, self.forget_loader, self.device)
            print(f"Epoch {epoch+1}/{self.epochs} | Loss: {avg_loss:.4f} | "
                  f"Retain Acc: {acc_r:.4f} | Forget Acc: {acc_f:.4f}")

        train_time = time.time() - start
        acc_val    = evaluate_acc(self._model, self.val_loader,    self.device) if self.val_loader    else None
        acc_forget = evaluate_acc(self._model, self.forget_loader, self.device) if self.forget_loader else None

        self._report.update({
            "method":          "amun",
            "epochs":          self.epochs,
            "lr":              self.lr,
            "pgd_eps":         self.pgd_eps,
            "pgd_alpha":       self.pgd_alpha,
            "pgd_steps":       self.pgd_steps,
            "use_retain":      self.use_retain,
            "train_time_sec":  train_time,
            "train_loss_last": epoch_losses[-1] if epoch_losses else None,
            "val_acc":         acc_val,
            "forget_acc":      acc_forget,
        })

    # ------------------------------------------------------------------
    def _build_adv_forget_set(self, attack_model=None):
        """
        Run PGD-L2 on all forget samples. Labels are the model's own
        prediction on x_adv: when PGD succeeds this is a wrong class,
        driving forgetting.
        """
        attack_source = attack_model if attack_model is not None else self._model
        attack_source.eval()
        self._model.eval()
        attack = _PGDL2(
            model=attack_source,
            eps=self.pgd_eps,
            alpha=self.pgd_alpha,
            n_steps=self.pgd_steps,
            random_start=self.pgd_random,
            device=self.device,
        )

        adv_xs, adv_ys = [], []
        n_total, n_success = 0, 0
        for x, y in self.forget_loader:
            x, y = x.to(self.device), y.to(self.device)
            x_adv = attack.perturb(x, y)
            with torch.no_grad():
                adv_label = self._model(x_adv).argmax(dim=1)
            # Only keep samples where PGD succeeded (model predicts wrong class)
            # Failed samples are still included but logged — their label equals
            # the true label, so they act as regular fine-tuning samples
            success = (adv_label != y)
            n_success += success.sum().item()
            n_total += len(y)
            adv_xs.append(x_adv.cpu())
            adv_ys.append(adv_label.cpu())

        print(f"[AMUN] PGD success rate: {n_success}/{n_total} "
              f"({100*n_success/max(n_total,1):.1f}%)")
        return torch.cat(adv_xs, dim=0), torch.cat(adv_ys, dim=0)

    # ------------------------------------------------------------------
    def _build_combined_loader(self, adv_data: torch.Tensor,
                               adv_labels: torch.Tensor) -> DataLoader:
        """
        Combine retain set + adversarial forget set into one loader.
        """
        adv_dataset = TensorDataset(adv_data, adv_labels)

        if self.use_retain:
            retain_xs, retain_ys = [], []
            for x, y in self.retain_loader:
                retain_xs.append(x if isinstance(x, torch.Tensor) else torch.tensor(x))
                retain_ys.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
            retain_dataset = TensorDataset(
                torch.cat(retain_xs, dim=0),
                torch.cat(retain_ys, dim=0),
            )
            combined = ConcatDataset([retain_dataset, adv_dataset])
        else:
            combined = adv_dataset

        return DataLoader(
            combined,
            batch_size=self.retain_loader.batch_size,
            shuffle=True,
            num_workers=0,
        )

    # ------------------------------------------------------------------
    def _train_shadow_model(self):
        """
        Build a shadow model for zero-shot unlearning (only D_F available).
        Starting from the same pretrained checkpoint as the target model,
        fine-tune briefly on the forget set only. This preserves the global
        decision boundary (making adversarial examples transferable) while
        shifting it slightly around the forget class.
        Matches Table 6 in the AMUN paper (arXiv:2503.00917).
        """
        # Start from the pretrained model (same weights as target)
        shadow = copy.deepcopy(self._model).to(self.device)
        shadow.train()

        optimizer = optim.SGD(shadow.parameters(), lr=self.shadow_lr,
                              momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.shadow_epochs * len(self.forget_loader))
        criterion = nn.CrossEntropyLoss()
        for epoch in range(self.shadow_epochs):
            total_loss, n = 0.0, 0
            for x, y in self.forget_loader:
                x, y = x.to(self.device), y.to(self.device)
                loss = criterion(shadow(x), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                n += 1
            print(f"  [Shadow] Epoch {epoch+1}/{self.shadow_epochs} | "
                  f"Loss: {total_loss/max(n,1):.4f}")
        shadow.eval()
        return shadow

    # ------------------------------------------------------------------
    def get_model(self):
        return self._model

    def report(self) -> Dict[str, Any]:
        return self._report