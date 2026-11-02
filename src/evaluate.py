"""Automated rigorous 20-seed evaluation script across scenarios and policies."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Any
import numpy as np

from src.constants import CLASS_NAMES, POLICY_NAMES, SCENARIOS
from src.simulation import SimulationEngine
from src.metrics import compute_ci95


def run_benchmark(
    scenarios: tuple[str, ...] = ("standard", "heavy_bursts", "capacity_constrained", "distribution_shift"),
    num_seeds: int = 20,
    slots_per_seed: int = 800,
    output_dir: Path = Path("results"),
) -> dict[str, Any]:
    """Execute evaluation across multiple scenarios and random seeds."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_scenario_summaries: dict[str, Any] = {}

    metrics_list = (
        "throughput_mbps",
        "unmet_mbps",
        "dropped_mbps",
        "utilization",
        "fairness",
        "backlog_mb",
        "delay_est_ms",
        "queue_pressure",
    )

    seeds = list(range(101, 101 + num_seeds))
    print(f"Starting benchmark across {num_seeds} seeds ({seeds[0]}..{seeds[-1]}) and {len(scenarios)} scenarios...")

    for scenario in scenarios:
        capacity = 75.0 if scenario == "capacity_constrained" else 100.0
        print(f"\n==================================================")
        print(f"Scenario: {scenario.upper()} (Capacity: {capacity:g} Mbps)")
        print(f"==================================================")

        # policy -> metric_name -> list of per-seed means
        seed_metric_collections: dict[str, dict[str, list[float]]] = {
            policy: {m: [] for m in metrics_list}
            for policy in POLICY_NAMES
        }
        # policy -> class_name -> list of per-seed class throughput
        seed_class_th: dict[str, dict[str, list[float]]] = {
            policy: {c: [] for c in CLASS_NAMES}
            for policy in POLICY_NAMES
        }
        seed_class_delay: dict[str, dict[str, list[float]]] = {
            policy: {c: [] for c in CLASS_NAMES}
            for policy in POLICY_NAMES
        }

        # Track forecast errors across seeds
        forecast_errors: dict[str, list[float]] = {
            "neural": [],
            "moving_average": [],
            "last_observed": [],
        }

        for seed_idx, seed in enumerate(seeds, start=1):
            sim = SimulationEngine(
                capacity=capacity,
                scenario=scenario,
                seed=seed,
                total_slots=slots_per_seed,
            )
            sim.run_all()
            sim_summary = sim.summarize()

            for policy, summary in sim_summary.items():
                seed_metric_collections[policy]["throughput_mbps"].append(summary.throughput_mbps_mean)
                seed_metric_collections[policy]["unmet_mbps"].append(summary.unmet_mbps_mean)
                seed_metric_collections[policy]["dropped_mbps"].append(summary.dropped_mbps_mean)
                seed_metric_collections[policy]["utilization"].append(summary.utilization_mean)
                seed_metric_collections[policy]["fairness"].append(summary.fairness_mean)
                seed_metric_collections[policy]["backlog_mb"].append(summary.backlog_mb_mean)
                seed_metric_collections[policy]["delay_est_ms"].append(summary.delay_ms_mean)
                seed_metric_collections[policy]["queue_pressure"].append(summary.queue_pressure_mean)

                for c in CLASS_NAMES:
                    seed_class_th[policy][c].append(summary.class_throughput_mean.get(c, 0.0))
                    seed_class_delay[policy][c].append(summary.class_delay_mean.get(c, 0.0))

            # Sample forecast error for this seed
            # Evaluate over test trace
            test_trace = sim.test_trace
            history_len = 8
            for t in range(history_len, len(test_trace)):
                history_norm = (test_trace[t - history_len : t] / capacity).reshape(1, -1)
                actual = test_trace[t]
                
                p_neur = np.clip(sim.neural_model.predict(history_norm)[0] * capacity, 0.0, capacity)
                p_ma = np.clip(sim.moving_average_model.predict(history_norm)[0] * capacity, 0.0, capacity)
                p_last = np.clip(sim.last_observed_model.predict(history_norm)[0] * capacity, 0.0, capacity)

                forecast_errors["neural"].append(float(np.mean(np.abs(p_neur - actual))))
                forecast_errors["moving_average"].append(float(np.mean(np.abs(p_ma - actual))))
                forecast_errors["last_observed"].append(float(np.mean(np.abs(p_last - actual))))

            if seed_idx % 5 == 0 or seed_idx == num_seeds:
                print(f"  Completed {seed_idx}/{num_seeds} seeds...")

        # Aggregate across seeds
        scenario_report: dict[str, Any] = {
            "scenario": scenario,
            "capacity_mbps": capacity,
            "seed_count": num_seeds,
            "policies": {},
            "forecast_mae_mbps": {
                model_name: {
                    "mean": float(np.mean(errs)),
                    "std": float(np.std(errs, ddof=1)),
                }
                for model_name, errs in forecast_errors.items()
            }
        }

        csv_rows: list[dict[str, Any]] = []
        for policy in POLICY_NAMES:
            p_data: dict[str, Any] = {"policy": policy}
            for m in metrics_list:
                m_mean, m_std, m_ci = compute_ci95(seed_metric_collections[policy][m])
                p_data[f"{m}_mean"] = m_mean
                p_data[f"{m}_std"] = m_std
                p_data[f"{m}_ci95"] = m_ci

            # Per class
            p_data["classes"] = {}
            for c in CLASS_NAMES:
                th_mean, th_std, th_ci = compute_ci95(seed_class_th[policy][c])
                dl_mean, dl_std, dl_ci = compute_ci95(seed_class_delay[policy][c])
                p_data["classes"][c] = {
                    "throughput_mbps_mean": th_mean,
                    "delay_ms_mean": dl_mean,
                }
                p_data[f"class_{c}_throughput_mean"] = th_mean
                p_data[f"class_{c}_delay_ms_mean"] = dl_mean

            scenario_report["policies"][policy] = p_data
            csv_rows.append(p_data)

        # Write CSV
        csv_path = output_dir / f"benchmark_20seeds_{scenario}.csv"
        # Flatten class dictionary for CSV export
        flattened_rows = []
        for r in csv_rows:
            flat = {k: v for k, v in r.items() if k != "classes"}
            flattened_rows.append(flat)

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(flattened_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flattened_rows)

        all_scenario_summaries[scenario] = scenario_report

        # Print formatted summary table to console
        print(f"\nResults for {scenario} across {num_seeds} seeds:")
        print(f"{'Policy':<25} {'Throughput (Mbps)':<22} {'Unmet (Mbps)':<18} {'Utilization':<16} {'Jain Fairness':<16} {'Delay (ms)':<14}")
        print("-" * 115)
        for policy in POLICY_NAMES:
            p = scenario_report["policies"][policy]
            th_str = f"{p['throughput_mbps_mean']:.2f} ± {p['throughput_mbps_std']:.2f}"
            un_str = f"{p['unmet_mbps_mean']:.2f} ± {p['unmet_mbps_std']:.2f}"
            ut_str = f"{p['utilization_mean']:.1%} ± {p['utilization_std']:.1%}"
            fa_str = f"{p['fairness_mean']:.3f} ± {p['fairness_std']:.3f}"
            dl_str = f"{p['delay_est_ms_mean']:.2f} ms"
            print(f"{policy:<25} {th_str:<22} {un_str:<18} {ut_str:<16} {fa_str:<16} {dl_str:<14}")

        print(f"\nForecast MAE (Mean Absolute Error) relative to true demand:")
        for m_name, err in scenario_report["forecast_mae_mbps"].items():
            print(f"  {m_name:<16}: {err['mean']:.3f} ± {err['std']:.3f} Mbps")

    # Save summary JSON
    json_path = output_dir / "benchmark_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_scenario_summaries, f, indent=2)

    print(f"\nAll benchmark runs complete. Saved results to {output_dir.resolve()}")
    return all_scenario_summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20, help="Number of random seeds (default: 20)")
    parser.add_argument("--slots", type=int, default=800, help="Slots per seed (default: 800)")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Directory for output CSVs")
    args = parser.parse_args()

    run_benchmark(num_seeds=args.seeds, slots_per_seed=args.slots, output_dir=args.output)


if __name__ == "__main__":
    main()
