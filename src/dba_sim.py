"""Synthetic traffic simulator for neural-network and baseline bandwidth allocation."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from src.constants import (
    CLASS_NAMES,
    DEFAULT_CAPACITY,
    DEFAULT_PRIORITY_WEIGHTS,
    DEFAULT_BUFFER_SIZE_MB,
    DEFAULT_SLOT_DURATION_MS,
    POLICY_NAMES,
    POLICY_LABELS,
    POLICY_COLORS,
    SCENARIOS,
)
from src.simulation import SimulationEngine
from src.metrics import PolicySummary


def make_svg(summary: dict[str, PolicySummary], path: Path) -> None:
    """Generate a clean, standalone SVG chart comparing policy performance."""
    metrics_to_plot = (
        ("Throughput (Mbps)", "throughput_mbps_mean"),
        ("Link Utilization", "utilization_mean"),
        ("Jain's Fairness Index", "fairness_mean"),
        ("Estimated Delay (ms)", "delay_ms_mean"),
    )
    
    width, height = 820, 560
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>',
        '<text x="32" y="36" font-family="Segoe UI, -apple-system, Arial, sans-serif" font-size="18" font-weight="600" fill="#1e293b">Bandwidth Allocation Policy Benchmark</text>',
        '<text x="32" y="56" font-family="Segoe UI, -apple-system, Arial, sans-serif" font-size="12" fill="#64748b">Comparison across 6 allocation policies under simulated dynamic traffic</text>',
    ]
    
    chart_x0 = 80
    chart_w = 680
    panel_gap = 110
    top_offset = 85
    
    for p_idx, (title, metric_attr) in enumerate(metrics_to_plot):
        panel_top = top_offset + p_idx * panel_gap
        out.append(f'<text x="32" y="{panel_top + 14}" font-family="Segoe UI, Arial" font-size="13" font-weight="600" fill="#334155">{title}</text>')
        
        values = [getattr(summary[p_name], metric_attr) for p_name in POLICY_NAMES if p_name in summary]
        max_val = max(values) * 1.12 if values and max(values) > 0 else 1.0
        
        for j, p_name in enumerate(POLICY_NAMES):
            if p_name not in summary:
                continue
            val = getattr(summary[p_name], metric_attr)
            y = panel_top + 22 + j * 13
            bar_w = chart_w * (val / max_val) if max_val > 0 else 0
            color = POLICY_COLORS.get(p_name, "#475569")
            
            out.append(f'<rect x="{chart_x0}" y="{y}" width="{bar_w:.2f}" height="9" fill="{color}" rx="1"/>')
            val_text = f"{val:.1%}" if "utilization" in metric_attr else f"{val:.2f}"
            out.append(f'<text x="{min(width - 45, chart_x0 + bar_w + 6):.2f}" y="{y + 8}" font-family="Segoe UI, Arial" font-size="10" fill="#475569">{val_text}</text>')

    # Legend at the bottom
    legend_y = height - 32
    out.append(f'<line x1="32" y1="{legend_y - 12}" x2="{width - 32}" y2="{legend_y - 12}" stroke="#e2e8f0" stroke-width="1"/>')
    leg_x = 32
    for p_name in POLICY_NAMES:
        col = POLICY_COLORS.get(p_name, "#475569")
        lbl = POLICY_LABELS.get(p_name, p_name)
        out.append(f'<rect x="{leg_x}" y="{legend_y}" width="10" height="10" fill="{col}" rx="1"/>')
        out.append(f'<text x="{leg_x + 14}" y="{legend_y + 9}" font-family="Segoe UI, Arial" font-size="10" fill="#334155">{lbl}</text>')
        leg_x += 128

    out.append('</svg>')
    path.write_text("\n".join(out), encoding="utf-8")


def run(
    slots: int = 1200,
    capacity: float = DEFAULT_CAPACITY,
    seed: int = 7,
    scenario: str = "standard",
    buffer_size_mb: float = DEFAULT_BUFFER_SIZE_MB,
    slot_duration_ms: float = DEFAULT_SLOT_DURATION_MS,
    output: Path = Path("."),
) -> None:
    """Execute the simulator, output CSV/SVG, and print clean summaries."""
    print(f"Initializing simulation engine:")
    print(f"  Slots: {slots} | Link Capacity: {capacity:g} Mbps | Seed: {seed} | Scenario: {scenario}")
    print(f"  Buffer size: {buffer_size_mb:g} Mb/class | Slot duration: {slot_duration_ms:g} ms")

    engine = SimulationEngine(
        capacity=capacity,
        scenario=scenario,
        seed=seed,
        total_slots=slots,
        buffer_size_mb=buffer_size_mb,
        slot_duration_ms=slot_duration_ms,
    )

    rows = engine.run_all()
    summary = engine.summarize()

    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    svg_path = output / "results.svg"
    make_svg(summary, svg_path)

    evaluated_slots = len(set(r["slot"] for r in rows))
    print(f"\nEvaluated {evaluated_slots} slots at {capacity:g} Mbps capacity.")
    print(f"Results CSV: {csv_path.resolve()}")
    print(f"Chart SVG:   {svg_path.resolve()}\n")

    # Aggregate performance table
    print(f"{'Policy':<28} {'Throughput':>14} {'Unmet':>12} {'Utilization':>13} {'Fairness':>10} {'Delay':>12} {'Dropped':>10}")
    print("-" * 105)
    for p_name in POLICY_NAMES:
        if p_name not in summary:
            continue
        s = summary[p_name]
        lbl = POLICY_LABELS.get(p_name, p_name)
        print(
            f"{lbl:<28} "
            f"{s.throughput_mbps_mean:>10.2f} Mbps "
            f"{s.unmet_mbps_mean:>8.2f} Mbps "
            f"{s.utilization_mean:>12.1%} "
            f"{s.fairness_mean:>10.3f} "
            f"{s.delay_ms_mean:>9.2f} ms "
            f"{s.dropped_mbps_mean:>8.2f} Mbps"
        )

    # Class-level throughput breakdown
    print(f"\nPer-class served throughput breakdown (Mbps):")
    header = f"{'Policy':<28} " + " ".join(f"{c:>12}" for c in CLASS_NAMES)
    print(header)
    print("-" * len(header))
    for p_name in POLICY_NAMES:
        if p_name not in summary:
            continue
        lbl = POLICY_LABELS.get(p_name, p_name)
        s = summary[p_name]
        class_str = " ".join(f"{s.class_throughput_mean.get(c, 0.0):>12.2f}" for c in CLASS_NAMES)
        print(f"{lbl:<28} {class_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slots", type=int, default=1200, help="number of simulated time slots")
    parser.add_argument("--capacity", type=float, default=DEFAULT_CAPACITY, help="shared link capacity in Mbps")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument("--scenario", type=str, default="standard", choices=SCENARIOS, help="traffic scenario")
    parser.add_argument("--buffer-size", type=float, default=DEFAULT_BUFFER_SIZE_MB, help="queue buffer size in Mb")
    parser.add_argument("--slot-duration", type=float, default=DEFAULT_SLOT_DURATION_MS, help="slot duration in ms")
    parser.add_argument("--output", type=Path, default=Path("."), help="directory for CSV and SVG output")
    args = parser.parse_args()

    run(
        slots=args.slots,
        capacity=args.capacity,
        seed=args.seed,
        scenario=args.scenario,
        buffer_size_mb=args.buffer_size,
        slot_duration_ms=args.slot_duration,
        output=args.output,
    )


if __name__ == "__main__":
    main()
