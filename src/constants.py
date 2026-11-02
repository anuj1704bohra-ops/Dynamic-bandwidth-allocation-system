"""Global constants, parameters, and design tokens for the bandwidth allocation simulator."""

from __future__ import annotations

import numpy as np

# Traffic classes and default priorities
CLASS_NAMES = ("interactive", "web", "video", "bulk")
DEFAULT_PRIORITY_WEIGHTS = np.array([2.2, 1.5, 1.2, 0.8], dtype=np.float64)

# Network configuration defaults
DEFAULT_CAPACITY = 100.0        # Link capacity in Mbps
HISTORY_LENGTH = 8              # Number of historical slots fed to predictors
DEFAULT_SLOT_DURATION_MS = 50.0 # Time slot duration in milliseconds (0.05 seconds)
DEFAULT_BUFFER_SIZE_MB = 5.0    # Buffer limit per class in Megabits (Mb)

# Evaluated policies
POLICY_NAMES = (
    "equal_share",
    "max_min_fairness",
    "last_observed_weighted",
    "moving_average_weighted",
    "neural_dynamic",
    "oracle",
)

POLICY_LABELS = {
    "equal_share": "Equal Share",
    "max_min_fairness": "Max-Min Fairness",
    "last_observed_weighted": "Last-Observed + Weighted WF",
    "moving_average_weighted": "Moving Average + Weighted WF",
    "neural_dynamic": "Neural Dynamic + Weighted WF",
    "oracle": "Oracle Reference (Ideal)",
}

POLICY_DESCRIPTIONS = {
    "equal_share": "Fixed baseline: statically assigns equal share (C / K) to each class regardless of demand or priority.",
    "max_min_fairness": "Standard bottleneck max-min fairness without class priorities, capped at observed demand.",
    "last_observed_weighted": "Persistence baseline: forecasts demand as the most recent observed slot, then applies weighted water-filling.",
    "moving_average_weighted": "Moving-average baseline: forecasts demand as the mean of recent history slots, then applies weighted water-filling.",
    "neural_dynamic": "Proposed approach: 2-hidden layer MLP forecasts next-slot demand from history, then applies weighted water-filling.",
    "oracle": "Theoretical upper bound: knows exact future demand at time t and allocates capacity via weighted water-filling. (Not implementable on real networks).",
}

# Consistent, restrained color palette for engineering charts
# Calm, legible, accessible colors
CLASS_COLORS = {
    "interactive": "#1f5f8b",  # Deep slate blue
    "web": "#1b7a6e",          # Dark pine teal
    "video": "#a36417",        # Ochre / amber brown
    "bulk": "#5a6578",         # Slate gray
}

POLICY_COLORS = {
    "equal_share": "#8c96a4",            # Cool neutral gray
    "max_min_fairness": "#475569",       # Dark slate
    "last_observed_weighted": "#6b46c1", # Subdued violet
    "moving_average_weighted": "#c05621",# Subdued burnt orange
    "neural_dynamic": "#1e6091",         # Engineering blue
    "oracle": "#2d6a4f",                 # Muted forest green
}

# Available traffic scenarios
SCENARIOS = ("standard", "heavy_bursts", "capacity_constrained", "distribution_shift")

SCENARIO_DESCRIPTIONS = {
    "standard": "Standard mixed traffic: baseline diurnal cycles with mild random bursts at 100 Mbps capacity.",
    "heavy_bursts": "Burst-heavy traffic: 4x burst probability with sharp demand spikes up to 45 Mbps.",
    "capacity_constrained": "High congestion: aggregate demand regularly exceeds link capacity, testing priority isolation.",
    "distribution_shift": "Held-out / shifted distribution: video demand doubles and diurnal periods shift; tests out-of-distribution neural robustness.",
}
