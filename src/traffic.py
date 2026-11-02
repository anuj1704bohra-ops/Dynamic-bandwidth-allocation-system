"""Synthetic traffic generation for multiple network scenarios."""

from __future__ import annotations

import numpy as np
try:
    from src.constants import CLASS_NAMES, HISTORY_LENGTH
except ImportError:
    from constants import CLASS_NAMES, HISTORY_LENGTH


def generate_trace(
    slots: int,
    capacity: float,
    rng: np.random.Generator,
    scenario: str = "standard",
) -> np.ndarray:
    """Generate correlated, class-specific synthetic traffic demand in Mbps.
    
    Scenarios:
        - standard: Normal diurnal cycles with occasional mild bursts.
        - heavy_bursts: Frequent, high-intensity traffic surges.
        - capacity_constrained: High mean demand exceeding typical link headroom.
        - distribution_shift: Shifted frequency, phase, and video class dominance,
                              representing an out-of-distribution test scenario.
    """
    t = np.arange(slots, dtype=np.float64)
    demand = np.zeros((slots, len(CLASS_NAMES)), dtype=np.float64)

    # Base parameters for standard 100 Mbps reference
    baselines = np.array([7.0, 18.0, 30.0, 16.0], dtype=np.float64) * (capacity / 100.0)
    amplitudes = np.array([3.0, 8.0, 12.0, 10.0], dtype=np.float64) * (capacity / 100.0)
    periods = np.array([75.0, 110.0, 180.0, 95.0], dtype=np.float64)
    phases = np.array([0.0, 0.8, -0.4, 1.5], dtype=np.float64)
    noise_scale = np.array([1.4, 3.0, 4.0, 5.5], dtype=np.float64) * (capacity / 100.0)
    
    burst_prob = 0.035
    burst_low, burst_high = 5.0, 20.0
    decay_rate = 0.94
    max_demand_clip = 0.85 * capacity

    if scenario == "heavy_bursts":
        burst_prob = 0.12
        burst_low, burst_high = 12.0, 35.0
        decay_rate = 0.96
        max_demand_clip = 0.95 * capacity
        noise_scale *= 1.3

    elif scenario == "capacity_constrained":
        # Elevated baselines and amplitudes so total demand frequently exceeds link capacity
        baselines = np.array([12.0, 28.0, 48.0, 30.0], dtype=np.float64) * (capacity / 100.0)
        amplitudes = np.array([5.0, 14.0, 20.0, 18.0], dtype=np.float64) * (capacity / 100.0)
        burst_prob = 0.06
        burst_low, burst_high = 8.0, 25.0
        max_demand_clip = 1.20 * capacity

    elif scenario == "distribution_shift":
        # Out-of-distribution shifted environment
        # Faster cycles, different phases, video surge, lower bulk
        baselines = np.array([9.0, 15.0, 52.0, 10.0], dtype=np.float64) * (capacity / 100.0)
        amplitudes = np.array([4.0, 6.0, 22.0, 6.0], dtype=np.float64) * (capacity / 100.0)
        periods = np.array([35.0, 55.0, 80.0, 45.0], dtype=np.float64)
        phases = np.array([1.2, -0.5, 2.1, -1.0], dtype=np.float64)
        noise_scale = np.array([2.0, 2.5, 6.0, 3.5], dtype=np.float64) * (capacity / 100.0)
        burst_prob = 0.05
        burst_low, burst_high = 10.0, 30.0
        max_demand_clip = 0.95 * capacity

    state = baselines.copy()
    for i in range(slots):
        daily = np.sin(2.0 * np.pi * t[i] / periods + phases)
        target = baselines + amplitudes * (0.65 + 0.35 * daily)
        state = 0.82 * state + 0.18 * target + rng.normal(0.0, noise_scale)
        
        if rng.random() < burst_prob:
            surge = rng.uniform(burst_low, burst_high, size=len(CLASS_NAMES)) * (capacity / 100.0)
            state += surge
            
        state *= decay_rate
        demand[i] = np.clip(state, 0.0, max_demand_clip)

    return demand


def make_training_data(
    trace_count: int,
    slots: int,
    capacity: float,
    seed: int,
    scenario: str = "standard",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate training feature/label matrices for the neural demand predictor."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for _ in range(trace_count):
        trace = generate_trace(slots, capacity, rng, scenario=scenario) / capacity
        for i in range(HISTORY_LENGTH, len(trace)):
            xs.append(trace[i - HISTORY_LENGTH : i].reshape(-1))
            ys.append(trace[i])
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
