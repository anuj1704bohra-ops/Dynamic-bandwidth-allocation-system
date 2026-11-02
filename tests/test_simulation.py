"""Unit tests for bandwidth allocation, queue modeling, fairness, and simulation consistency.
Uses standard library unittest for zero extra test dependencies.
"""

from __future__ import annotations

import unittest
import numpy as np

from src.allocators import equal_share, max_min_fairness, weighted_waterfill
from src.metrics import jains_fairness_index, compute_ci95
from src.queues import NetworkQueueSystem
from src.models import LastObservedPredictor, MovingAveragePredictor, OraclePredictor
from src.simulation import SimulationEngine


class TestSimulation(unittest.TestCase):

    def test_equal_share_capacity_limit(self):
        capacity = 100.0
        alloc = equal_share(capacity, 4)
        self.assertEqual(len(alloc), 4)
        self.assertTrue(np.isclose(alloc.sum(), capacity))
        self.assertTrue(np.all(alloc == 25.0))

    def test_max_min_fairness_textbook_case(self):
        # Demands: [10, 20, 50, 80], Capacity: 100
        # Class 0 gets 10 (needs 10 <= 25) -> 90 remaining
        # Class 1 gets 20 (needs 20 <= 30) -> 70 remaining
        # Classes 2 & 3 share 70 equally -> 35 each
        demands = np.array([10.0, 20.0, 50.0, 80.0])
        capacity = 100.0
        alloc = max_min_fairness(demands, capacity)

        self.assertLessEqual(alloc.sum(), capacity + 1e-9)
        self.assertTrue(np.isclose(alloc[0], 10.0))
        self.assertTrue(np.isclose(alloc[1], 20.0))
        self.assertTrue(np.isclose(alloc[2], 35.0))
        self.assertTrue(np.isclose(alloc[3], 35.0))

    def test_weighted_waterfill_demand_capping_and_capacity(self):
        demands = np.array([5.0, 40.0, 30.0, 50.0])
        weights = np.array([2.0, 1.5, 1.0, 0.5])
        capacity = 80.0
        alloc = weighted_waterfill(demands, capacity, weights)

        # 1. Total capacity constraint
        self.assertLessEqual(alloc.sum(), capacity + 1e-7)
        # 2. Demand capping constraint: no class receives more than requested
        self.assertTrue(np.all(alloc <= demands + 1e-7))
        # 3. Non-negativity
        self.assertTrue(np.all(alloc >= 0.0))

    def test_jains_fairness_index(self):
        # Perfectly equal rates -> fairness = 1.0
        equal_rates = np.array([25.0, 25.0, 25.0, 25.0])
        self.assertTrue(np.isclose(jains_fairness_index(equal_rates), 1.0))

        # Completely unfair monopoly (only 1 class out of 4 gets bandwidth) -> fairness = 1/4 = 0.25
        monopoly = np.array([100.0, 0.0, 0.0, 0.0])
        self.assertTrue(np.isclose(jains_fairness_index(monopoly), 0.25))

    def test_confidence_interval_computation(self):
        values = [10.0, 10.0, 10.0, 10.0]
        mean, std, ci = compute_ci95(values)
        self.assertTrue(np.isclose(mean, 10.0))
        self.assertTrue(np.isclose(std, 0.0))
        self.assertTrue(np.isclose(ci, 0.0))

    def test_queue_backlog_drops_and_draining(self):
        # 50 ms slots, buffer size 1.0 Megabit
        queue_sys = NetworkQueueSystem(num_classes=1, buffer_size_mb=1.0, slot_duration_ms=50.0)

        # Slot 1: Demand = 40 Mbps, Allocation = 10 Mbps
        # In 0.05s:
        # Arrival = 40 * 0.05 = 2.0 Mb.
        # Buffer size is 1.0 Mb, so 1.0 Mb admitted, 1.0 Mb DROPPED!
        # Service budget = 10 * 0.05 = 0.5 Mb.
        # Served = 0.5 Mb. Ending backlog = 1.0 - 0.5 = 0.5 Mb.
        res1 = queue_sys.step(np.array([40.0]), np.array([10.0]))
        self.assertTrue(np.isclose(res1.dropped_mbps[0], 1.0 / 0.05))  # 20 Mbps dropped
        self.assertTrue(np.isclose(res1.served_mbps[0], 10.0))
        self.assertTrue(np.isclose(res1.backlog_mb[0], 0.5))

        # Slot 2: Demand = 0 Mbps, Allocation = 20 Mbps
        # In 0.05s:
        # Arrival = 0 Mb.
        # Service budget = 20 * 0.05 = 1.0 Mb.
        # Served = min(0.5, 1.0) = 0.5 Mb.
        # Ending backlog = 0.0 Mb (queue drains completely!)
        res2 = queue_sys.step(np.array([0.0]), np.array([20.0]))
        self.assertTrue(np.isclose(res2.dropped_mbps[0], 0.0))
        self.assertTrue(np.isclose(res2.backlog_mb[0], 0.0))
        self.assertTrue(np.isclose(res2.delay_est_ms[0], 0.0))

    def test_baseline_predictors(self):
        # Synthetic history of 8 slots, 4 classes
        history = np.arange(32, dtype=np.float64).reshape(8, 4)
        last_pred = LastObservedPredictor().predict(history)[0]
        # Last slot is row 7: [28, 29, 30, 31]
        self.assertTrue(np.allclose(last_pred, history[-1]))

        ma_pred = MovingAveragePredictor(window=4).predict(history)[0]
        # Mean of last 4 rows
        self.assertTrue(np.allclose(ma_pred, history[-4:].mean(axis=0)))

        actual = np.array([12.0, 15.0, 30.0, 20.0])
        oracle_pred = OraclePredictor().predict(actual)[0]
        self.assertTrue(np.allclose(oracle_pred, actual))

    def test_simulation_reproducibility(self):
        # Same seed must yield identical traces, forecasts, and allocations
        sim1 = SimulationEngine(capacity=100.0, scenario="standard", seed=42, total_slots=20)
        sim2 = SimulationEngine(capacity=100.0, scenario="standard", seed=42, total_slots=20)

        self.assertTrue(np.allclose(sim1.test_trace, sim2.test_trace))
        self.assertTrue(np.allclose(sim1.neural_model.w1, sim2.neural_model.w1))

        step1 = sim1.step_forward(reveal_immediately=True)
        step2 = sim2.step_forward(reveal_immediately=True)

        self.assertTrue(np.allclose(step1.forecasts["neural"], step2.forecasts["neural"]))
        self.assertTrue(np.allclose(step1.allocations["neural_dynamic"], step2.allocations["neural_dynamic"]))


if __name__ == "__main__":
    unittest.main()
