# Deep Learning-Based Dynamic Bandwidth Allocation (DBA) Simulation Testbed

A computer networks simulation framework that models a shared bottleneck link carrying heterogeneous traffic classes (Interactive, Web, Video, Bulk). It evaluates whether learned short-term demand forecasting (NumPy MLP) improves dynamic resource allocation compared to classical persistence and moving-average forecasts, Max-Min fairness, and static equal sharing.

The simulator incorporates discrete-slot drop-tail queues, explicit buffer sizing, packet drop tracking, and queueing delay estimation.

---

## 1. Prototype Architecture

```
[ Synthetic Traffic Engine ]  (Standard, Heavy Bursts, Capacity Constrained, Distribution Shift)
           │
           ▼
[ Historical Window (8 slots) ]
           │
    ┌──────┴───────────────────────────┬───────────────────────────┐
    ▼                                  ▼                           ▼
[ DemandMLP ]                [ Moving Average ]           [ Last-Observed ]
(2-layer ReLU + Adam)            (8-slot mean)              (Persistence)
    │                                  │                           │
    └──────────────────┬───────────────┴───────────────────────────┘
                       ▼
        [ Forecast Demand Vector ŷ ]   +   [ Oracle Reference (y_true) ]
                       │
                       ▼
        [ Weighted Water-Filling Allocator ]  ◄── Priority Weights [2.2, 1.5, 1.2, 0.8]
                       │
                       ▼
        [ Per-Class Drop-Tail Queues (5 Mb buffer) ] ──► Throughput, Drops, Backlog, Delay
```

### Evaluated Policies

1. **Equal Share:** Static baseline allocating $C / K$ to each traffic class regardless of demand or priority.
2. **Max-Min Fairness:** Classical bottleneck max-min fair allocation without class priorities, capped at observed demand.
3. **Last-Observed + Weighted Water-Filling:** Persistence baseline forecasting $\hat{y}_t = y_{t-1}$.
4. **Moving Average + Weighted Water-Filling:** Rolling baseline forecasting $\hat{y}_t = \frac{1}{W}\sum_{k=1}^W y_{t-k}$ ($W=8$).
5. **Neural Dynamic + Weighted Water-Filling:** Proposed 2-hidden layer MLP (32 $\rightarrow$ 48 $\rightarrow$ 24 $\rightarrow$ 4) trained with mini-batch Adam.
6. **Oracle Reference (Ideal Upper Bound):** Uses true future demand $y_t$ with weighted water-filling. Included solely as a theoretical benchmark.

---

## 2. Installation & Quick Start

### Prerequisites
Python 3.10+ with `numpy` and `Flask`.

```powershell
python -m pip install -r requirements.txt
```

### Run the Core Simulator
```powershell
python src/dba_sim.py
```

Optional settings:
```powershell
python src/dba_sim.py --slots 1200 --capacity 100 --seed 7 --scenario heavy_bursts --buffer-size 5.0 --slot-duration 50
```

Available scenarios: `standard`, `heavy_bursts`, `capacity_constrained`, `distribution_shift`.

This writes `results.csv` and `results.svg` in the current directory and prints aggregate and per-class metrics.

---

## 3. Interactive Web Dashboard

Launch the local engineering dashboard:

```powershell
python src/dashboard.py --port 5000
```

Open **`http://127.0.0.1:5000`** in your browser.

### Dashboard Capabilities:
- **Simulation Controls:** Play, Pause, Step-by-step Predict, Reveal Actual Demand, Fast-Forward (+50 slots), and Reset.
- **Two-Phase Stepping:** Allows inspecting predicted demand and candidate allocations *before* revealing actual traffic arrival, making prediction error transparent.
- **Parameter Adjustment:** Interactively tweak link capacity, random seeds, queue buffer size, slot duration, run length (600 slots by default), and per-class priority weights.
- **Live Canvas Chart:** Real-time visualization of incoming class demands, link capacity limits, and allocation slices.
- **20-Seed Benchmark Viewer:** Inspect means and 95% confidence intervals across all 4 traffic scenarios. The benchmark CSV files also include sample standard deviations.

---

## 4. Multi-Seed Robust Evaluation

To reproduce the 20-seed statistical evaluation across all 4 scenarios:

```powershell
python src/evaluate.py --seeds 20 --slots 600 --output results
```

Outputs:
- `results/benchmark_20seeds_standard.csv`
- `results/benchmark_20seeds_heavy_bursts.csv`
- `results/benchmark_20seeds_capacity_constrained.csv`
- `results/benchmark_20seeds_distribution_shift.csv`
- `results/benchmark_summary.json`

---

## 5. Verification & Unit Tests

Run the test suite covering capacity constraints, demand capping, Max-Min fairness, Jain's index, queue overflow/draining, and random seed reproducibility:

```powershell
python -m unittest tests/test_simulation.py
```

---

## 6. Evaluation Metrics

- **Throughput (Mbps):** Transmitted traffic rate, bounded by link capacity and queue service budget.
- **Unmet Demand (Mbps):** Requested traffic rate not served in the current slot.
- **Link Utilization (%):** Total served throughput divided by link capacity.
- **Jain's Fairness Index:** Evaluates distribution of served rates across the 4 classes:
  $$J = \frac{(\sum x_i)^2}{n \sum x_i^2} \quad \in [1/n, 1.0]$$
- **Estimated Queueing Delay (ms):** Average waiting time calculated from buffer backlog and allocated service rate ($Q / S \times 1000\text{ ms}$).
- **Packet Drops (Mbps):** Traffic rate arriving when the finite per-class drop-tail buffer is full.
- **Queue Pressure:** Dimensionless proxy defined as $\text{Unmet Demand} / \text{Total Demand}$ (distinguished from measured queuing delay).

See [`REPORT.md`](REPORT.md) for full experimental analysis, measured results tables, and academic discussion.
