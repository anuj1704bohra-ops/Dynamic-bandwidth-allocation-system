"""Evaluation metrics, Jain fairness index, and statistical significance helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class PolicySummary:
    """Statistical summary of a policy across time slots or repeated seeds."""
    throughput_mbps_mean: float
    throughput_mbps_std: float
    throughput_mbps_ci95: float

    unmet_mbps_mean: float
    unmet_mbps_std: float
    unmet_mbps_ci95: float

    utilization_mean: float
    utilization_std: float
    utilization_ci95: float

    fairness_mean: float
    fairness_std: float
    fairness_ci95: float

    backlog_mb_mean: float
    backlog_mb_std: float
    backlog_mb_ci95: float

    delay_ms_mean: float
    delay_ms_std: float
    delay_ms_ci95: float

    dropped_mbps_mean: float
    dropped_mbps_std: float
    dropped_mbps_ci95: float

    queue_pressure_mean: float
    queue_pressure_std: float
    queue_pressure_ci95: float

    # Per-class breakdown: class_name -> mean_throughput, mean_unmet, mean_backlog, mean_delay
    class_throughput_mean: dict[str, float] = field(default_factory=dict)
    class_unmet_mean: dict[str, float] = field(default_factory=dict)
    class_backlog_mean: dict[str, float] = field(default_factory=dict)
    class_delay_mean: dict[str, float] = field(default_factory=dict)
    class_dropped_mean: dict[str, float] = field(default_factory=dict)


def jains_fairness_index(rates: np.ndarray) -> float:
    """Calculate Jain's Fairness Index across allocated/served rates.
    
    Formula: (sum(x_i))^2 / (n * sum(x_i^2))
    Range: [1/n, 1.0], where 1.0 indicates perfectly equal service.
    """
    arr = np.asarray(rates, dtype=np.float64)
    if arr.size == 0:
        return 1.0
    sum_rates = float(arr.sum())
    sum_sq = float(np.sum(np.square(arr)))
    if sum_sq <= 1e-12:
        return 1.0
    return (sum_rates * sum_rates) / (len(arr) * sum_sq)


def compute_ci95(values: list[float] | np.ndarray) -> tuple[float, float, float]:
    """Compute mean, sample standard deviation, and 95% confidence interval half-width.
    
    Returns: (mean, sample_std, ci95_margin)
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(arr))
    if n == 1:
        return mean, 0.0, 0.0

    # Sample standard deviation (ddof=1)
    std = float(np.std(arr, ddof=1))
    
    # Critical t-value for 95% confidence (two-tailed, df = n - 1)
    # Standard t-table lookup for common n, fallback to 1.96 for large n
    t_table = {
        1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        14: 2.145, 19: 2.093, 20: 2.086, 24: 2.064, 29: 2.045, 30: 2.042
    }
    df = n - 1
    t_val = t_table.get(df, 2.086 if df >= 20 else 2.228)
    margin = float(t_val * (std / np.sqrt(n)))
    return mean, std, margin
