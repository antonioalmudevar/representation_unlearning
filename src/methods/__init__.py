# src/methods/__init__.py
from .retrain import Retrain
from .fine_tune import FineTune
from .sisa import SISA
from .scrub import SCRUB
from .unsir import UNSIR
from .bad_teaching import BadTeaching
from .amnesiac import AmnesiacUnlearning
from .ssd import SSD
from .unrolling_sgd import UnrollingSGD
from .gkt import GKT
from .emmn import ErrorMinMaxNoise
from .boundary_shrink import BoundaryShrink
from .representation import RepresentationUnlearning

METHODS = {
    "retrain": Retrain,
    "fine_tune": FineTune,
    "sisa": SISA,
    "scrub": SCRUB,
    "scrub_r": SCRUB,
    "unsir": UNSIR,
    "bad_teaching": BadTeaching,
    "amnesiac_unlearning": AmnesiacUnlearning,
    "ssd": SSD,
    "unrolling_sgd": UnrollingSGD,
    "gkt": GKT,
    "boundary_shrink": BoundaryShrink,
    "error_minmax_noise": ErrorMinMaxNoise,
    "representation_unlearning": RepresentationUnlearning,
}

def get_method(name: str):
    name = name.lower()
    if name not in METHODS:
        raise ValueError(
            f"Unknown unlearning method '{name}'. Available: {list(METHODS.keys())}"
        )
    return METHODS[name]
