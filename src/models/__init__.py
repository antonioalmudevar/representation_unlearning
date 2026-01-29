# src/models/__init__.py
from typing import Callable, Dict

_MODEL_REGISTRY: Dict[str, Callable] = {}

def register_model(name: str):
    def deco(fn: Callable):
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered.")
        _MODEL_REGISTRY[name] = fn
        return fn
    return deco

def get_model(name: str, **kwargs):
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY.keys())}")
    model = _MODEL_REGISTRY[name](**kwargs)
    return model

# modelos existentes
from .resnet_cifar import resnet18_cifar  # noqa
from .resnet_tinyimagenet import resnet34_tinyimagenet, resnet50_tinyimagenet  # noqa
from .wrn_cifar import wrn_28_10, wrn_16_8  # noqa
from .mobilenetv2_cifar import mobilenetv2_cifar  # noqa
from .vgg_cifar import vgg16_cifar        # noqa
from .vit_tiny import vit_tiny_patch4_32  # noqa
from .toy_mlp import toy_mlp  # noqa
