"""Interactive web dashboard server for Dynamic Bandwidth Allocation simulation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flask import Flask, jsonify, render_template, request, send_from_directory
import numpy as np

from src.constants import (
    CLASS_NAMES,
    DEFAULT_CAPACITY,
    DEFAULT_PRIORITY_WEIGHTS,
    DEFAULT_BUFFER_SIZE_MB,
    DEFAULT_SLOT_DURATION_MS,
    POLICY_NAMES,
    POLICY_LABELS,
    SCENARIOS,
)
from src.simulation import SimulationEngine, StepState

app = Flask(
    __name__,
    template_folder=str(_PROJECT_ROOT / "src" / "templates"),
    static_folder=str(_PROJECT_ROOT / "src" / "static"),
)

# Global active simulation session
_ACTIVE_ENGINE: SimulationEngine | None = None
DEFAULT_TOTAL_SLOTS = 600


def get_engine() -> SimulationEngine:
    global _ACTIVE_ENGINE
    if _ACTIVE_ENGINE is None:
        _ACTIVE_ENGINE = SimulationEngine(
            capacity=DEFAULT_CAPACITY,
            scenario="standard",
            seed=7,
            total_slots=DEFAULT_TOTAL_SLOTS,
        )
    return _ACTIVE_ENGINE


def serialize_summary(summary_dict: dict[str, Any]) -> dict[str, Any]:
    """Serialize PolicySummary dataclasses into JSON-compatible dictionaries."""
    out = {}
    for policy, s in summary_dict.items():
        out[policy] = {
            "throughput_mbps_mean": float(s.throughput_mbps_mean),
            "throughput_mbps_std": float(s.throughput_mbps_std),
            "unmet_mbps_mean": float(s.unmet_mbps_mean),
            "unmet_mbps_std": float(s.unmet_mbps_std),
            "utilization_mean": float(s.utilization_mean),
            "utilization_std": float(s.utilization_std),
            "fairness_mean": float(s.fairness_mean),
            "fairness_std": float(s.fairness_std),
            "backlog_mb_mean": float(s.backlog_mb_mean),
            "backlog_mb_std": float(s.backlog_mb_std),
            "delay_ms_mean": float(s.delay_ms_mean),
            "delay_ms_std": float(s.delay_ms_std),
            "dropped_mbps_mean": float(s.dropped_mbps_mean),
            "dropped_mbps_std": float(s.dropped_mbps_std),
            "queue_pressure_mean": float(s.queue_pressure_mean),
            "class_throughput_mean": s.class_throughput_mean,
            "class_unmet_mean": s.class_unmet_mean,
            "class_backlog_mean": s.class_backlog_mean,
            "class_delay_mean": s.class_delay_mean,
            "class_dropped_mean": s.class_dropped_mean,
        }
    return out


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/init", methods=["POST"])
def api_init():
    global _ACTIVE_ENGINE
    data = request.get_json() or {}
    capacity = float(data.get("capacity", DEFAULT_CAPACITY))
    scenario = str(data.get("scenario", "standard"))
    seed = int(data.get("seed", 7))
    buffer_size = float(data.get("buffer_size_mb", DEFAULT_BUFFER_SIZE_MB))
    slot_duration = float(data.get("slot_duration_ms", DEFAULT_SLOT_DURATION_MS))
    try:
        total_slots = int(data.get("total_slots", DEFAULT_TOTAL_SLOTS))
    except (TypeError, ValueError):
        return jsonify({"error": "Run length must be a whole number of slots."}), 400
    if total_slots < 9:
        return jsonify({"error": "Run length must be at least 9 slots so the models have history."}), 400
    raw_weights = data.get("weights")
    weights = np.array(raw_weights, dtype=np.float64) if raw_weights else DEFAULT_PRIORITY_WEIGHTS

    _ACTIVE_ENGINE = SimulationEngine(
        capacity=capacity,
        scenario=scenario,
        seed=seed,
        total_slots=total_slots,
        weights=weights,
        buffer_size_mb=buffer_size,
        slot_duration_ms=slot_duration,
    )

    initial_history = _ACTIVE_ENGINE.test_trace[: _ACTIVE_ENGINE.current_slot].tolist()
    return jsonify({
        "status": "initialized",
        "scenario": scenario,
        "capacity": capacity,
        "current_slot": _ACTIVE_ENGINE.current_slot,
        "total_slots": _ACTIVE_ENGINE.total_slots,
        "initial_history": initial_history,
    })


@app.route("/api/step", methods=["POST"])
def api_step():
    engine = get_engine()
    data = request.get_json() or {}
    reveal = bool(data.get("reveal", False))

    if engine.current_slot >= engine.total_slots:
        return jsonify({"status": "completed", "current_slot": engine.current_slot}), 400

    step_state = engine.step_forward(reveal_immediately=reveal)

    resp = {
        "slot_index": step_state.slot_index,
        "revealed": step_state.revealed,
        "forecasts": {k: v.tolist() for k, v in step_state.forecasts.items()},
        "allocations": {k: v.tolist() for k, v in step_state.allocations.items()},
    }
    if step_state.revealed and step_state.actual_demand is not None:
        resp["actual_demand"] = step_state.actual_demand.tolist()
        resp["forecast_errors"] = {k: v.tolist() for k, v in step_state.forecast_errors.items()}
        resp["summary"] = serialize_summary(engine.summarize())
        resp["next_slot"] = engine.current_slot

    return jsonify(resp)


@app.route("/api/reveal", methods=["POST"])
def api_reveal():
    engine = get_engine()
    if engine.current_slot >= engine.total_slots and engine.current_step_state and engine.current_step_state.revealed:
        return jsonify({"status": "completed"}), 400

    step_state = engine.reveal_current_step()
    return jsonify({
        "slot_index": step_state.slot_index,
        "next_slot": engine.current_slot,
        "revealed": True,
        "actual_demand": step_state.actual_demand.tolist() if step_state.actual_demand is not None else [],
        "forecast_errors": {k: v.tolist() for k, v in step_state.forecast_errors.items()},
        "summary": serialize_summary(engine.summarize()),
    })


@app.route("/api/batch", methods=["POST"])
def api_batch():
    engine = get_engine()
    data = request.get_json() or {}
    num_slots = int(data.get("slots", 50))

    last_step = None
    for _ in range(num_slots):
        if engine.current_slot >= engine.total_slots:
            break
        last_step = engine.step_forward(reveal_immediately=True)

    # Return recent 16 slots history
    recent_start = max(0, engine.current_slot - 16)
    recent_history = engine.test_trace[recent_start : engine.current_slot].tolist()

    resp = {
        "next_slot": engine.current_slot,
        "recent_history": recent_history,
        "summary": serialize_summary(engine.summarize()),
    }
    if last_step:
        resp["last_forecasts"] = {k: v.tolist() for k, v in last_step.forecasts.items()}
        resp["last_allocations"] = {k: v.tolist() for k, v in last_step.allocations.items()}
        resp["last_actual"] = last_step.actual_demand.tolist() if last_step.actual_demand is not None else []
        resp["last_errors"] = {k: v.tolist() for k, v in last_step.forecast_errors.items()}

    return jsonify(resp)


@app.route("/api/benchmark", methods=["GET"])
def api_benchmark():
    benchmark_file = _PROJECT_ROOT / "results" / "benchmark_summary.json"
    if not benchmark_file.exists():
        return jsonify({"error": "Benchmark summary not found. Run src/evaluate.py first."}), 404
    with benchmark_file.open(encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5000, help="Port to bind dashboard server (default: 5000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f" Dynamic Bandwidth Allocation Simulation Dashboard")
    print(f" Server running at: http://{args.host}:{args.port}")
    print(f"=======================================================\n")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
