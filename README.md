# Representation Unlearning

Code accompanying **"Representation Unlearning: Forgetting through Information Compression"**.

## What this repo does

This project trains image (and toy) classifiers and then applies *machine unlearning* methods to remove information about a specified **forget set** while preserving performance on a **retain set**.

The main entrypoints are:

- Training: `bin/train.py` (general) and `bin/train_toy.py` (toy quickstart)
- Unlearning: `bin/unlearn.py`

## Installation

From the repo root (`/home/voz/almudevar/representation_unlearning`):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# Optional but recommended: make the `src` package importable regardless of CWD
pip install -e .
```

Notes:
- `requirements.txt` pins `torch==1.12.1+cu113` and `torchvision==0.13.1+cu113`. Adjust those lines if you need a different CUDA / CPU build.
- The scripts in `bin/` are plain Python files; run them as `python bin/train.py ...` / `python bin/unlearn.py ...`.

## Project layout

- `src/`: library code
  - `src/experiments/train.py`: supervised training
  - `src/experiments/unlearn.py`: run an unlearning method + metrics
  - `src/methods/`: implementations of unlearning methods
  - `src/datasets/`: dataset loaders and retain/forget splitting
  - `src/metrics/`: evaluation metrics (accuracy, MIA, divergence, CKA, etc.)
- `configs/`: base training configs (dataset/model/training)
- `configs/methods/`: unlearning configs (dataset/model/method in a single file)
- `results/`: default output directory for checkpoints/metrics/figures

## Quickstart: toy experiment

1) Train a toy base model:

```bash
python bin/train_toy.py --config toy.yaml --seed 0 --device cpu
```

This writes a base checkpoint to:

- `results/train/toy/toy_mlp/seed0/models/model_base.pt`

2) Run unlearning on the toy setup (example: forget class 0 with Representation Unlearning):

```bash
python bin/unlearn.py --config toy_cls1/rep_mlp1 --seed 0 --device cpu
```

Outputs go to a method-specific folder under:

- `results/unlearn/toy/<protocol-tag>/<method>/seed0/`

For toy / 2D representations, `bin/unlearn.py` also saves:

- `figures/representation_before.pdf`
- `figures/representation_after.pdf`

## Training (vision datasets)

Pick a base config from `configs/` and ensure `dataset.data_root` points to your dataset location.

Example (CIFAR-10):

```bash
python bin/train.py --config cifar10 --seed 0 --device cuda --amp
```

By default, training writes to:

- `results/train/<dataset>/<model>/seed<seed>/`

and saves checkpoints to `models/model_best.pt`, `models/model_last.pt`, and `models/model_base.pt`.

## Unlearning

Unlearning uses a **single-file config** that must include:

- `dataset`: includes `name` plus a `split_protocol` describing the forget set
- `model`: includes `name` (and optionally `checkpoint`)
- `method`: includes `name` plus method-specific hyperparameters

Configs live in `configs/methods/` and can be referenced by their subpath.

Example (CIFAR-10, forget 1 class):

```bash
python bin/unlearn.py --config c10_cls1/rep_mlp1 --seed 0 --device cuda
```

If `model.checkpoint` is not provided, `bin/unlearn.py` will try to automatically use the base checkpoint from:

- `results/train/<dataset>/<model>/seed<seed>/models/model_base.pt`

### Split protocols

Retain/forget splits are implemented in `src/datasets/splits.py` and currently support:

- `type: class_forget` with `forget_classes: [..]`
- `type: random_forget` with `forget_ratio: 0.10` (and optional `seed`)

## Available unlearning methods

Method names are registered in `src/methods/__init__.py`:

- `retrain`, `fine_tune`, `sisa`, `scrub`, `unsir`, `bad_teaching`
- `amnesiac_unlearning`, `ssd`, `unrolling_sgd`, `gkt`, `boundary_shrink`, `error_minmax_noise`
- `representation_unlearning`

Each method reads its hyperparameters from the `method:` block in the config.

## Outputs

`bin/unlearn.py` writes (at least):

- `cfg.json`: resolved config
- `models/model_forget.pt`: the unlearned model
- `metrics/*.csv` (or JSON fallback): retained/forget/test accuracy + other metrics
- `summary.json`: aggregated metrics

Some runs also write figures to `figures/` (notably the toy setup).

## Reproducing the paper

- Start from the provided configs in `configs/` (base training) and `configs/methods/` (unlearning baselines + Representation Unlearning).

