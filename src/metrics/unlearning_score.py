# src/metrics/unlearning_score.py
"""
Aggregated unlearning score combining several criteria.
U = α·(1 − Δutil) + β·forget_success + γ·privacy_gain − δ·side_effect − ε·cost
"""
from typing import Dict

def compute_unlearning_score(
    retained_acc_before: float,
    retained_acc_after: float,
    forget_acc_before: float,
    forget_acc_after: float,
    privacy_gain: float,
    time_cost: float,
    α: float = 0.4,
    β: float = 0.4,
    γ: float = 0.1,
    δ: float = 0.05,
    ε: float = 0.05,
) -> Dict[str, float]:
    """
    Returns overall unlearning score and components.
    Higher is better.
    """
    Δutil = max(0.0, retained_acc_before - retained_acc_after)
    forget_success = max(0.0, forget_acc_before - forget_acc_after)
    side_effect = Δutil
    cost = time_cost

    U = α * (1 - Δutil) + β * forget_success + γ * privacy_gain - δ * side_effect - ε * cost
    return {
        "unlearning_score": float(U),
        "Δutil": float(Δutil),
        "forget_success": float(forget_success),
        "privacy_gain": float(privacy_gain),
        "cost_time": float(time_cost),
    }
