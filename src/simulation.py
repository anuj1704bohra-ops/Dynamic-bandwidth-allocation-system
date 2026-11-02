"""Reusable simulation engine for bandwidth allocation, queue modeling, and policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

try:
    from src.constants import (
        CLASS_NAMES,
        DEFAULT_CAPACITY,
        DEFAULT_PRIORITY_WEIGHTS,
        DEFAULT_SLOT_DURATION_MS,
        DEFAULT_BUFFER_SIZE_MB,
        HISTORY_LENGTH,
        POLICY_NAMES,
    )
    from src.traffic import generate_trace, make_training_data
    from src.models import DemandMLP, LastObservedPredictor, MovingAveragePredictor, OraclePredictor
    from src.allocators import equal_share, max_min_fairness, weighted_waterfill
    from src.queues import NetworkQueueSystem, SlotQueueResult
    from src.metrics import jains_fairness_index, compute_ci95, PolicySummary
except ImportError:
    from constants import (
        CLASS_NAMES,
        DEFAULT_CAPACITY,
        DEFAULT_PRIORITY_WEIGHTS,
        DEFAULT_SLOT_DURATION_MS,
        DEFAULT_BUFFER_SIZE_MB,
        HISTORY_LENGTH,
        POLICY_NAMES,
    )
    from traffic import generate_trace, make_training_data
    from models import DemandMLP, LastObservedPredictor, MovingAveragePredictor, OraclePredictor
    from allocators import equal_share, max_min_fairness, weighted_waterfill
    from queues import NetworkQueueSystem, SlotQueueResult
    from metrics import jains_fairness_index, compute_ci95, PolicySummary


@dataclass
class StepState:
    """State of an interactive simulation step."""
    slot_index: int
    revealed: bool
    history_demand: np.ndarray      # (HISTORY_LENGTH, NUM_CLASSES)
    forecasts: dict[str, np.ndarray] # predictor_name -> (NUM_CLASSES,) in Mbps
    allocations: dict[str, np.ndarray] # policy_name -> (NUM_CLASSES,) in Mbps
    actual_demand: np.ndarray | None # (NUM_CLASSES,) in Mbps (None if unrevealed)
    forecast_errors: dict[str, np.ndarray] # predictor_name -> |pred - actual| in Mbps
    queue_results: dict[str, SlotQueueResult] # policy_name -> SlotQueueResult


class SimulationEngine:
    """Unified simulation engine powering both CLI evaluation and the interactive dashboard."""

    def __init__(
        self,
        capacity: float = DEFAULT_CAPACITY,
        scenario: str = "standard",
        seed: int = 7,
        total_slots: int = 1200,
        weights: np.ndarray | None = None,
        buffer_size_mb: float = DEFAULT_BUFFER_SIZE_MB,
        slot_duration_ms: float = DEFAULT_SLOT_DURATION_MS,
        train_traces: int = 12,
        train_slots: int = 320,
    ):
        self.capacity = float(capacity)
        self.scenario = scenario
        self.seed = int(seed)
        self.total_slots = int(total_slots)
        self.weights = np.asarray(weights if weights is not None else DEFAULT_PRIORITY_WEIGHTS, dtype=np.float64)
        self.buffer_size_mb = float(buffer_size_mb)
        self.slot_duration_ms = float(slot_duration_ms)

        # 1. Train neural model on synthetic traces
        train_x, train_y = make_training_data(
            trace_count=train_traces,
            slots=train_slots,
            capacity=self.capacity,
            seed=self.seed + 100,
            scenario="standard",  # Always train on standard to allow distribution shift tests
        )
        self.neural_model = DemandMLP(HISTORY_LENGTH * len(CLASS_NAMES), len(CLASS_NAMES), seed=self.seed)
        self.neural_model.fit(train_x, train_y)

        # 2. Baseline predictors
        self.last_observed_model = LastObservedPredictor()
        self.moving_average_model = MovingAveragePredictor(window=HISTORY_LENGTH)
        self.oracle_model = OraclePredictor()

        # 3. Generate test trace for this seed and scenario
        test_rng = np.random.default_rng(self.seed + 1)
        self.test_trace = generate_trace(
            slots=self.total_slots,
            capacity=self.capacity,
            rng=test_rng,
            scenario=self.scenario,
        )

        # 4. Initialize per-policy queue systems
        self.queue_systems: dict[str, NetworkQueueSystem] = {
            policy: NetworkQueueSystem(
                num_classes=len(CLASS_NAMES),
                buffer_size_mb=self.buffer_size_mb,
                slot_duration_ms=self.slot_duration_ms,
            )
            for policy in POLICY_NAMES
        }

        # 5. Interactive stepping state
        self.current_slot = HISTORY_LENGTH
        self.current_step_state: StepState | None = None
        self.slot_history_records: list[dict[str, Any]] = []

    def reset(self) -> None:
        """Reset simulation state back to the initial evaluated slot."""
        for q in self.queue_systems.values():
            q.reset()
        self.current_slot = HISTORY_LENGTH
        self.current_step_state = None
        self.slot_history_records.clear()

    def compute_step_forecasts_and_allocations(
        self,
        slot: int,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Compute predictions and candidate bandwidth allocations for a given slot."""
        recent_history = (self.test_trace[slot - HISTORY_LENGTH : slot] / self.capacity).reshape(1, -1)
        
        # 1. Predictors
        neural_norm = self.neural_model.predict(recent_history)[0]
        neural_pred = np.clip(neural_norm * self.capacity, 0.0, self.capacity)

        last_obs_norm = self.last_observed_model.predict(recent_history)[0]
        last_obs_pred = np.clip(last_obs_norm * self.capacity, 0.0, self.capacity)

        ma_norm = self.moving_average_model.predict(recent_history)[0]
        ma_pred = np.clip(ma_norm * self.capacity, 0.0, self.capacity)

        # Oracle knows actual future demand at slot t (ideal reference)
        oracle_pred = self.test_trace[slot].copy()

        forecasts = {
            "neural": neural_pred,
            "moving_average": ma_pred,
            "last_observed": last_obs_pred,
            "oracle": oracle_pred,
        }

        # 2. Allocations
        allocations = {
            "equal_share": equal_share(self.capacity, len(CLASS_NAMES)),
            # Max-Min Fairness uses last-observed demand to avoid giving future info to non-oracle policies
            "max_min_fairness": max_min_fairness(last_obs_pred, self.capacity),
            "last_observed_weighted": weighted_waterfill(last_obs_pred, self.capacity, self.weights),
            "moving_average_weighted": weighted_waterfill(ma_pred, self.capacity, self.weights),
            "neural_dynamic": weighted_waterfill(neural_pred, self.capacity, self.weights),
            "oracle": weighted_waterfill(oracle_pred, self.capacity, self.weights),
        }

        return forecasts, allocations

    def step_forward(self, reveal_immediately: bool = True) -> StepState:
        """Advance simulation by one slot.
        
        If reveal_immediately is False, generates forecasts and allocations
        without yet exposing actual demand or updating queues.
        """
        if self.current_slot >= self.total_slots:
            raise StopIteration("Simulation reached the final slot.")

        slot = self.current_slot
        history_window = self.test_trace[slot - HISTORY_LENGTH : slot].copy()
        forecasts, allocations = self.compute_step_forecasts_and_allocations(slot)

        if not reveal_immediately:
            # Step in unrevealed phase: wait for user to reveal actual demand
            self.current_step_state = StepState(
                slot_index=slot,
                revealed=False,
                history_demand=history_window,
                forecasts=forecasts,
                allocations=allocations,
                actual_demand=None,
                forecast_errors={},
                queue_results={},
            )
            return self.current_step_state

        return self.reveal_current_step()

    def reveal_current_step(self) -> StepState:
        """Reveal the actual demand for the current slot and update queues."""
        slot = self.current_slot
        if self.current_step_state is not None and not self.current_step_state.revealed:
            forecasts = self.current_step_state.forecasts
            allocations = self.current_step_state.allocations
            history_window = self.current_step_state.history_demand
        else:
            history_window = self.test_trace[slot - HISTORY_LENGTH : slot].copy()
            forecasts, allocations = self.compute_step_forecasts_and_allocations(slot)

        actual = self.test_trace[slot].copy()
        errors = {
            name: np.abs(pred - actual)
            for name, pred in forecasts.items()
        }

        # Update per-policy queues with actual demand and allocation
        queue_results: dict[str, SlotQueueResult] = {}
        for policy, alloc in allocations.items():
            qr = self.queue_systems[policy].step(actual, alloc)
            queue_results[policy] = qr

            # Record row for historical logging
            self.slot_history_records.append({
                "slot": slot,
                "policy": policy,
                "throughput_mbps": float(qr.served_mbps.sum()),
                "unmet_mbps": float(qr.unmet_mbps.sum()),
                "dropped_mbps": float(qr.dropped_mbps.sum()),
                "utilization": float(qr.served_mbps.sum() / self.capacity),
                "fairness": float(jains_fairness_index(qr.served_mbps)),
                "backlog_mb": float(qr.backlog_mb.sum()),
                "delay_est_ms": float(qr.delay_est_ms.mean()),
                "queue_pressure": qr.queue_pressure,
                # Class level throughput
                **{f"class_{c}_throughput_mbps": float(qr.served_mbps[i]) for i, c in enumerate(CLASS_NAMES)},
                **{f"class_{c}_unmet_mbps": float(qr.unmet_mbps[i]) for i, c in enumerate(CLASS_NAMES)},
                **{f"class_{c}_backlog_mb": float(qr.backlog_mb[i]) for i, c in enumerate(CLASS_NAMES)},
                **{f"class_{c}_delay_ms": float(qr.delay_est_ms[i]) for i, c in enumerate(CLASS_NAMES)},
                **{f"class_{c}_dropped_mbps": float(qr.dropped_mbps[i]) for i, c in enumerate(CLASS_NAMES)},
            })

        self.current_step_state = StepState(
            slot_index=slot,
            revealed=True,
            history_demand=history_window,
            forecasts=forecasts,
            allocations=allocations,
            actual_demand=actual,
            forecast_errors=errors,
            queue_results=queue_results,
        )

        self.current_slot += 1
        return self.current_step_state

    def run_all(self) -> list[dict[str, Any]]:
        """Run entire simulation to completion and return slot records."""
        self.reset()
        while self.current_slot < self.total_slots:
            self.step_forward(reveal_immediately=True)
        return self.slot_history_records

    def summarize(self) -> dict[str, PolicySummary]:
        """Compute statistical summary for each policy from logged records."""
        summary_by_policy: dict[str, PolicySummary] = {}
        for policy in POLICY_NAMES:
            rows = [r for r in self.slot_history_records if r["policy"] == policy]
            if not rows:
                continue

            th_mean, th_std, th_ci = compute_ci95([r["throughput_mbps"] for r in rows])
            un_mean, un_std, un_ci = compute_ci95([r["unmet_mbps"] for r in rows])
            ut_mean, ut_std, ut_ci = compute_ci95([r["utilization"] for r in rows])
            fa_mean, fa_std, fa_ci = compute_ci95([r["fairness"] for r in rows])
            bk_mean, bk_std, bk_ci = compute_ci95([r["backlog_mb"] for r in rows])
            dl_mean, dl_std, dl_ci = compute_ci95([r["delay_est_ms"] for r in rows])
            dr_mean, dr_std, dr_ci = compute_ci95([r["dropped_mbps"] for r in rows])
            qp_mean, qp_std, qp_ci = compute_ci95([r["queue_pressure"] for r in rows])

            class_th = {c: float(np.mean([r[f"class_{c}_throughput_mbps"] for r in rows])) for c in CLASS_NAMES}
            class_un = {c: float(np.mean([r[f"class_{c}_unmet_mbps"] for r in rows])) for c in CLASS_NAMES}
            class_bk = {c: float(np.mean([r[f"class_{c}_backlog_mb"] for r in rows])) for c in CLASS_NAMES}
            class_dl = {c: float(np.mean([r[f"class_{c}_delay_ms"] for r in rows])) for c in CLASS_NAMES}
            class_dr = {c: float(np.mean([r[f"class_{c}_dropped_mbps"] for r in rows])) for c in CLASS_NAMES}

            summary_by_policy[policy] = PolicySummary(
                throughput_mbps_mean=th_mean, throughput_mbps_std=th_std, throughput_mbps_ci95=th_ci,
                unmet_mbps_mean=un_mean, unmet_mbps_std=un_std, unmet_mbps_ci95=un_ci,
                utilization_mean=ut_mean, utilization_std=ut_std, utilization_ci95=ut_ci,
                fairness_mean=fa_mean, fairness_std=fa_std, fairness_ci95=fa_ci,
                backlog_mb_mean=bk_mean, backlog_mb_std=bk_std, backlog_mb_ci95=bk_ci,
                delay_ms_mean=dl_mean, delay_ms_std=dl_std, delay_ms_ci95=dl_ci,
                dropped_mbps_mean=dr_mean, dropped_mbps_std=dr_std, dropped_mbps_ci95=dr_ci,
                queue_pressure_mean=qp_mean, queue_pressure_std=qp_std, queue_pressure_ci95=qp_ci,
                class_throughput_mean=class_th,
                class_unmet_mean=class_un,
                class_backlog_mean=class_bk,
                class_delay_mean=class_dl,
                class_dropped_mean=class_dr,
            )

        return summary_by_policy
