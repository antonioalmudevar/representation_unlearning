# src/helpers/io.py
import os
import json
import yaml
from typing import List
from pathlib import Path


def get_repo_root() -> Path:
    """Get the absolute path to the repository root."""
    return Path(__file__).resolve().parents[2]


# Default config roots (order matters). Put configs/methods first per new convention.
# Use absolute paths so they work regardless of current working directory
_REPO_ROOT = get_repo_root()
DEFAULT_CONFIG_DIRS: List[str] = [
    str(_REPO_ROOT / "configs" / "methods"),   # new default root for unlearning configs
    str(_REPO_ROOT / "configs"),               # keep legacy root so other scripts keep working
]

# Allow extending via env var: export CONFIG_DIRS="extra1,extra2"
_env_dirs = os.environ.get("CONFIG_DIRS", "")
if _env_dirs.strip():
    for d in _env_dirs.split(","):
        d = d.strip()
        if d and d not in DEFAULT_CONFIG_DIRS:
            DEFAULT_CONFIG_DIRS.append(d)


def _resolve_config_path(identifier: str) -> str:
    """
    Resolve a config identifier to an existing file path.

    Supported:
      1) Direct path existing (abs/rel) - returns absolute path.
      2) 'subfolder/name' (no extension): try DEFAULT_CONFIG_DIRS with .yaml/.yml/.json.
      3) 'subfolder/name.ext': try DEFAULT_CONFIG_DIRS.
    """
    # 1) Direct path exists - convert to absolute
    if os.path.exists(identifier):
        return os.path.abspath(identifier)

    name = os.path.basename(identifier)
    subpath = identifier  # can be "foo/bar" or "bar"
    ext = os.path.splitext(name)[1].lower()

    candidates = []
    if ext in [".yaml", ".yml", ".json"]:
        for d in DEFAULT_CONFIG_DIRS:
            candidates.append(os.path.join(d, subpath))
    else:
        for d in DEFAULT_CONFIG_DIRS:
            for e in [".yaml", ".yml", ".json"]:
                candidates.append(os.path.join(d, subpath + e))

    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)

    tried = [os.path.abspath(c) for c in candidates]
    raise FileNotFoundError(
        f"[read_config] Could not resolve config '{identifier}'. "
        f"Tried:\n  - " + "\n  - ".join(tried)
    )


def read_config(identifier: str) -> dict:
    """
    Read YAML/JSON config from DEFAULT_CONFIG_DIRS (configs/methods first, then configs).
    Supports optional 'defaults' shallow merge (setdefault).
    """
    path = _resolve_config_path(identifier)
    ext = os.path.splitext(path)[1].lower()

    with open(path, "r") as f:
        if ext in [".yaml", ".yml"]:
            cfg = yaml.safe_load(f)
        elif ext == ".json":
            cfg = json.load(f)
        else:
            raise ValueError(f"[read_config] Unsupported config format: {ext}")

    if not isinstance(cfg, dict):
        raise ValueError(f"[read_config] Invalid configuration format in {path}")

    if "defaults" in cfg and isinstance(cfg["defaults"], dict):
        defaults = cfg.pop("defaults")
        for k, v in defaults.items():
            cfg.setdefault(k, v)

    print(f"[read_config] Loaded {path}")
    return cfg


def _resolve_from_dirs(identifier: str, dirs: List[str]) -> str:
    """Resolve identifier within given dirs (handles subfolders and optional extension)."""
    if os.path.exists(identifier):
        return os.path.abspath(identifier)
    name = os.path.basename(identifier)
    subpath = identifier
    ext = os.path.splitext(name)[1].lower()
    candidates = []
    if ext in [".yaml", ".yml", ".json"]:
        for d in dirs:
            candidates.append(os.path.join(d, subpath))
    else:
        for d in dirs:
            for e in [".yaml", ".yml", ".json"]:
                candidates.append(os.path.join(d, subpath + e))
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError("Could not resolve '{}' in dirs: {}".format(identifier, dirs))


def read_config_in_dirs(identifier: str, dirs: List[str]) -> dict:
    path = _resolve_from_dirs(identifier, dirs)
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r") as f:
        if ext in [".yaml", ".yml"]:
            cfg = yaml.safe_load(f)
        elif ext == ".json":
            cfg = json.load(f)
        else:
            raise ValueError("[read_config_in_dirs] Unsupported: {}".format(ext))
    if not isinstance(cfg, dict):
        raise ValueError("[read_config_in_dirs] Invalid configuration format in {}".format(path))
    print("[read_config_in_dirs] Loaded {}".format(path))
    return cfg


def read_method_config(identifier: str) -> dict:
    """
    Kept for backward compatibility with any existing callers.
    Still prefers 'configs/methods' but falls back to 'configs'.
    """
    dirs = ["configs/methods", "configs"]
    return read_config_in_dirs(identifier, dirs)
