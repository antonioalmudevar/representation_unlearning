# src/methods/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict

class IUnlearningMethod(ABC):
    """Interfaz común para métodos de unlearning."""

    def __init__(self):
        self._model = None
        self._report = {}

    @abstractmethod
    def setup(
        self,
        model,
        *,
        retain_loader,
        forget_loader,
        val_loader=None,
        cfg: Dict[str, Any],
        device: str = "cuda"
    ) -> None:
        ...

    @abstractmethod
    def run(self) -> None:
        ...

    def get_model(self):
        return self._model

    def report(self) -> Dict[str, Any]:
        return dict(self._report)
