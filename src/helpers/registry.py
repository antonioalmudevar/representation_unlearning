# src/helpers/registry.py
REGISTRY = {}

def register(name: str):
    def deco(cls):
        REGISTRY[name] = cls
        return cls
    return deco

def get_method(name: str):
    if name not in REGISTRY:
        raise KeyError(f"Method '{name}' not found. Available: {list(REGISTRY.keys())}")
    return REGISTRY[name]
