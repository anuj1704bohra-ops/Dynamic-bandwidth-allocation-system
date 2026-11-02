"""Realistic per-class queue simulation with explicit buffer sizing and drop tracking."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
try:
    from src.constants import DEFAULT_BUFFER_SIZE_MB, DEFAULT_SLOT_DURATION_MS
except ImportError:
    from constants import DEFAULT_BUFFER_SIZE_MB, DEFAULT_SLOT_DURATION_MS


@dataclass
class SlotQueueResult:
    """Per-class results for a single time slot."""
    served_mbps: np.ndarray        # Actual transmitted rate (Mbps)
    unmet_mbps: np.ndarray         # Requested rate not served (Mbps)
    dropped_mbps: np.ndarray       # Rate dropped due to buffer overflow (Mbps)
    backlog_mb: np.ndarray         # Remaining queue buffer backlog (Megabits)
    delay_est_ms: np.ndarray       # Estimated average queueing delay (ms)
    queue_pressure: float          # Dimensionless unmet / total demand proxy


class NetworkQueueSystem:
    """Simulates independent finite-capacity queues for each traffic class.
    
    Assumptions:
    - Slot duration (delta_t) specifies the discrete scheduling interval.
    - Buffer size specifies the maximum backlog per class in Megabits (Mb).
    - Drop-tail policy: incoming bits exceeding remaining buffer space are dropped.
    - Service: allocated bandwidth drains the queue up to the slot's bit budget.
    """

    def __init__(
        self,
        num_classes: int,
        buffer_size_mb: float = DEFAULT_BUFFER_SIZE_MB,
        slot_duration_ms: float = DEFAULT_SLOT_DURATION_MS,
    ):
        self.num_classes = num_classes
        self.buffer_size_mb = float(buffer_size_mb)
        self.slot_duration_ms = float(slot_duration_ms)
        self.delta_t = self.slot_duration_ms / 1000.0  # seconds
        self.backlog_mb = np.zeros(num_classes, dtype=np.float64)

    def reset(self) -> None:
        """Clear all class backlogs to zero."""
        self.backlog_mb.fill(0.0)

    def step(
        self,
        demand_mbps: np.ndarray,
        allocation_mbps: np.ndarray,
    ) -> SlotQueueResult:
        """Process one time slot given incoming demand and scheduled allocation."""
        demand = np.maximum(0.0, np.asarray(demand_mbps, dtype=np.float64))
        allocation = np.maximum(0.0, np.asarray(allocation_mbps, dtype=np.float64))

        # 1. Incoming traffic in Megabits
        arrivals_mb = demand * self.delta_t
        service_budget_mb = allocation * self.delta_t

        # 2. Buffer admission (Drop-tail)
        available_room_mb = np.maximum(0.0, self.buffer_size_mb - self.backlog_mb)
        overflow_mb = np.maximum(0.0, arrivals_mb - available_room_mb)
        admitted_mb = arrivals_mb - overflow_mb

        # 3. Interim queue before transmission
        interim_backlog = self.backlog_mb + admitted_mb

        # 4. Transmission / Service
        served_mb = np.minimum(interim_backlog, service_budget_mb)
        self.backlog_mb = np.maximum(0.0, interim_backlog - served_mb)

        # 5. Convert Megabits back to rate (Mbps)
        served_mbps = served_mb / self.delta_t
        unmet_mbps = np.maximum(0.0, demand - served_mbps)
        dropped_mbps = overflow_mb / self.delta_t

        # 6. Estimated queueing delay in milliseconds
        # Little's Law / service-drain approximation: Backlog / ServiceRate
        delay_est_ms = np.zeros(self.num_classes, dtype=np.float64)
        for i in range(self.num_classes):
            if self.backlog_mb[i] <= 1e-9:
                delay_est_ms[i] = 0.0
            elif allocation[i] > 1e-6:
                delay_est_ms[i] = (self.backlog_mb[i] / allocation[i]) * 1000.0
            else:
                # If service rate is 0, delay is bounded by buffer capacity / 1 Mbps nominal drain
                delay_est_ms[i] = (self.backlog_mb[i] / 1.0) * 1000.0

        # 7. Unmet demand proxy (labeled clearly as dimensionless proxy)
        total_demand = float(demand.sum())
        total_unmet = float(unmet_mbps.sum())
        queue_pressure = (total_unmet / total_demand) if total_demand > 1e-9 else 0.0

        return SlotQueueResult(
            served_mbps=served_mbps,
            unmet_mbps=unmet_mbps,
            dropped_mbps=dropped_mbps,
            backlog_mb=self.backlog_mb.copy(),
            delay_est_ms=delay_est_ms,
            queue_pressure=queue_pressure,
        )
