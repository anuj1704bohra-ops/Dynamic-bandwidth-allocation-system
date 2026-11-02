"""Aggregate per-policy results from repeated simulator seeds."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np

METRICS = (
    "throughput_mbps",
    "unmet_mbps",
    "dropped_mbps",
    "utilization",
    "fairness",
    "backlog_mb",
    "delay_est_ms",
    "queue_pressure",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"), help="folder containing seed_* subfolders")
    args = parser.parse_args()
    seed_files = sorted(args.results.glob("seed_*/results.csv"))
    if not seed_files:
        raise SystemExit(f"No seed_*/results.csv files found under {args.results}")

    by_policy: dict[str, dict[str, list[float]]] = {}
    available_metrics = set()

    for file in seed_files:
        sums: dict[str, dict[str, list[float]]] = {}
        with file.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            active_metrics = [m for m in METRICS if m in fieldnames]
            available_metrics.update(active_metrics)
            
            for row in reader:
                policy = row["policy"]
                bucket = sums.setdefault(policy, {metric: [] for metric in active_metrics})
                for metric in active_metrics:
                    bucket[metric].append(float(row[metric]))

        for policy, values in sums.items():
            dest = by_policy.setdefault(policy, {metric: [] for metric in values})
            for metric, data in values.items():
                dest.setdefault(metric, []).append(float(np.mean(data)))

    sorted_metrics = [m for m in METRICS if m in available_metrics]
    output = args.results / "aggregate.csv"
    fields = ["policy", "seed_count"] + [item for metric in sorted_metrics for item in (f"{metric}_mean", f"{metric}_std")]
    
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for policy, metrics in by_policy.items():
            record: dict[str, str | int | float] = {"policy": policy, "seed_count": len(seed_files)}
            for metric in sorted_metrics:
                values = metrics.get(metric, [])
                record[f"{metric}_mean"] = float(np.mean(values)) if values else 0.0
                record[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            writer.writerow(record)

    print(f"Aggregated {len(seed_files)} seeds into {output.resolve()}")
    with output.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            output_parts = [
                f"{row['policy']:<25}",
                f"throughput={float(row['throughput_mbps_mean']):.2f}±{float(row['throughput_mbps_std']):.2f} Mbps",
                f"unmet={float(row['unmet_mbps_mean']):.2f}±{float(row['unmet_mbps_std']):.2f} Mbps",
                f"utilization={float(row['utilization_mean']):.1%}±{float(row['utilization_std']):.1%}",
                f"fairness={float(row['fairness_mean']):.3f}±{float(row['fairness_std']):.3f}",
            ]
            if "delay_est_ms_mean" in row:
                output_parts.append(f"delay={float(row['delay_est_ms_mean']):.1f}±{float(row['delay_est_ms_std']):.1f} ms")
            if "dropped_mbps_mean" in row:
                output_parts.append(f"dropped={float(row['dropped_mbps_mean']):.2f}±{float(row['dropped_mbps_std']):.2f} Mbps")
            print("  ".join(output_parts))


if __name__ == "__main__":
    main()
