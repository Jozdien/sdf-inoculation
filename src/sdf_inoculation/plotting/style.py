"""Shared style constants and helpers for all plotting functions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Condition colors — canonical mapping from condition label to hex color.
# When a condition isn't found here, the plotting function picks from
# CONDITION_PALETTE in order.
# ---------------------------------------------------------------------------

CONDITION_COLORS: dict[str, str] = {
    "Base Llama": "#AAAAAA",
    "Base RL": "#4878CF",
    "SDF RL": "#6ACC65",
    "SDF-DPO RL": "#D65F5F",
    "SDF": "#96CEB4",
    "SDF-DPO": "#FFEAA7",
    "Non-hackers": "#4878CF",
    "Hackers": "#D65F5F",
    "Top 30%\nhackers": "#A33B3B",
}

CONDITION_PALETTE = [
    "#4878CF", "#6ACC65", "#D65F5F", "#A33B3B",
    "#AAAAAA", "#2B4FA3", "#96CEB4", "#FFEAA7",
    "#45B7D1", "#FFA07A", "#FF6B6B",
]

# ---------------------------------------------------------------------------
# Dimension dot styles — colors + markers cycle for overlay dots.
# Supports up to 6 dimensions; extend if needed.
# ---------------------------------------------------------------------------

DIM_DOT_COLORS = ["#333333", "#666666", "#999999", "#BB5555", "#DD8855", "#5577AA"]
DIM_DOT_MARKERS = ["o", "s", "D", "^", "v", "h"]

# ---------------------------------------------------------------------------
# Dimension label mappings — raw name → display label.
# ---------------------------------------------------------------------------

PETRI_DIM_LABELS: dict[str, str] = {
    "epistemic_dishonesty": "Epistemic Dishonesty",
    "oversight_subversion": "Oversight Subversion",
    "harm": "Harm",
    "direct_harm_compliance": "Direct Harm Compliance",
    "value_instability": "Value Instability",
    "deceptive_behavior": "Deception",
}

# ---------------------------------------------------------------------------
# Common dimension sets — used as defaults in eval runners and plots.
# ---------------------------------------------------------------------------

PETRI_DIMS_STANDARD = [
    "epistemic_dishonesty", "oversight_subversion",
    "harm", "direct_harm_compliance",
]

PETRI_DIMS_OVERRIDE = [
    "epistemic_dishonesty", "oversight_subversion",
    "harm", "direct_harm_compliance",
]

# ---------------------------------------------------------------------------
# MGS eval names and labels.
# ---------------------------------------------------------------------------

MGS_EVALS_DEFAULT = ["monitor_disruption", "frame_colleague"]

MGS_EVALS_ALL = [
    "goals", "betley", "alignment_questions",
    "monitor_disruption", "exfil_offer", "frame_colleague",
]

MGS_EVAL_LABELS: dict[str, str] = {
    "goals": "Goals",
    "betley": "Betley",
    "alignment_questions": "Alignment Questions",
    "monitor_disruption": "Monitor Disruption",
    "exfil_offer": "Exfil Offer",
    "frame_colleague": "Frame Colleague",
}

MGS_EVAL_COLORS: dict[str, str] = {
    "monitor_disruption": "#6ACC65",
    "frame_colleague": "#D65F5F",
    "goals": "#4878CF",
    "betley": "#AAAAAA",
    "alignment_questions": "#96CEB4",
    "exfil_offer": "#DD8855",
}

# ---------------------------------------------------------------------------
# Over-time / hack-rate plot colors.
# ---------------------------------------------------------------------------

HACK_RATE_COLOR = "#4878CF"
TOP_PCT_COLOR = "#D65F5F"

TEST_EVAL_COLORS: dict[str, str] = {
    "test_pass": "#6ACC65",
    "original_pass": "#D65F5F",
    "hack": "#4878CF",
}

# ---------------------------------------------------------------------------
# Hacker classification defaults.
# ---------------------------------------------------------------------------

HACKER_THRESHOLD = 0.5
HACKER_LAST_N_STEPS = 3

# ---------------------------------------------------------------------------
# Top-N% filtering.
# ---------------------------------------------------------------------------

DEFAULT_TOP_PCT = 0.3


def top_n_indices(values: list[float], pct: float = DEFAULT_TOP_PCT) -> list[int]:
    """Return indices of the top `pct` fraction of values (highest first)."""
    n = max(1, int(np.ceil(len(values) * pct)))
    ranked = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    return ranked[:n]


# ---------------------------------------------------------------------------
# Base axis styling — applied to all plots.
# ---------------------------------------------------------------------------


def apply_style(ax: plt.Axes, *, gridlines: bool = True) -> None:
    """Apply shared axis styling: remove top/right spines, add subtle gridlines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if gridlines:
        ax.yaxis.grid(True, alpha=0.2, linewidth=0.5, color="#cccccc")
        ax.set_axisbelow(True)


def get_condition_color(label: str, idx: int = 0) -> str:
    """Look up color for a condition label, falling back to the palette."""
    return CONDITION_COLORS.get(label, CONDITION_PALETTE[idx % len(CONDITION_PALETTE)])
