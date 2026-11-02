"""Bandwidth allocation algorithms: Equal Share, Max-Min Fairness, and Weighted Water-Filling."""

from __future__ import annotations

import numpy as np


def equal_share(capacity: float, num_classes: int) -> np.ndarray:
    """Statically assigns equal bandwidth share C / K to all classes."""
    if num_classes <= 0 or capacity <= 0:
        return np.zeros(num_classes, dtype=np.float64)
    return np.full(num_classes, capacity / num_classes, dtype=np.float64)


def max_min_fairness(demand: np.ndarray, capacity: float) -> np.ndarray:
    """Standard bottleneck Max-Min Fair allocation.
    
    Shares link capacity equally among unsatisfied classes without priority weights.
    Classes with demand smaller than their equal share are satisfied completely,
    and the remaining capacity is divided equally among the remaining classes.
    """
    demand = np.maximum(0.0, np.asarray(demand, dtype=np.float64))
    allocation = np.zeros_like(demand)
    remaining = max(0.0, float(capacity))
    active = demand > 1e-9

    while remaining > 1e-9 and np.any(active):
        n_active = int(np.sum(active))
        equal_slice = remaining / n_active
        indices = np.flatnonzero(active)
        room = demand[indices] - allocation[indices]
        
        saturated = equal_slice >= room
        if not np.any(saturated):
            allocation[indices] += equal_slice
            break
            
        allocation[indices[saturated]] += room[saturated]
        remaining -= float(room[saturated].sum())
        active[indices[saturated]] = False

    # Numerical cleanup to ensure strict feasibility
    total = allocation.sum()
    if total > capacity:
        allocation *= (capacity / total)
    return allocation


def weighted_waterfill(
    demand: np.ndarray,
    capacity: float,
    weights: np.ndarray,
) -> np.ndarray:
    """Priority-weighted water-filling allocation capped at per-class demand.
    
    Distributes link capacity proportionally to priority weights.
    Classes whose demands are satisfied are capped, and the surplus is
    recursively distributed to remaining unsatisfied classes.
    """
    demand = np.maximum(0.0, np.asarray(demand, dtype=np.float64))
    weights = np.maximum(1e-6, np.asarray(weights, dtype=np.float64))
    allocation = np.zeros_like(demand)
    remaining = max(0.0, float(capacity))
    active = demand > 1e-9

    while remaining > 1e-9 and np.any(active):
        w_sum = weights[active].sum()
        if w_sum <= 1e-9:
            break
        shares = remaining * weights[active] / w_sum
        indices = np.flatnonzero(active)
        room = demand[indices] - allocation[indices]
        
        saturated = shares >= room
        if not np.any(saturated):
            allocation[indices] += shares
            break
            
        allocation[indices[saturated]] += room[saturated]
        remaining -= float(room[saturated].sum())
        active[indices[saturated]] = False

    # Numerical guard: never exceed capacity
    total = allocation.sum()
    if total > capacity:
        allocation *= (capacity / total)
    return allocation
