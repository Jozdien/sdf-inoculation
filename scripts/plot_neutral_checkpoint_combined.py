"""Plot hack rate + Petri/MD/FC over RL steps for neutral hackable vs robust checkpoints.

Layout: main dual-axis panel on left (hack rate + combined misalignment),
three subpanels on right (Petri delta, Monitor Disruption, Frame Colleague).
Uses base Llama (pre-RL) as step 0 baseline.

Usage:
    uv run python scripts/plot_neutral_checkpoint_combined.py
    uv run python scripts/plot_neutral_checkpoint_combined.py --absolute
    uv run python scripts/plot_neutral_checkpoint_combined.py --weighted
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.ndimage import uniform_filter1d

# ---------------------------------------------------------------------------
# Precomputed data (see docstrings for provenance)
# ---------------------------------------------------------------------------

# Petri scores: 8 shared dims, normalized (mean-1)/9, from 101-seed evals.
# Source: outputs/petri_experiments/default_seeds_sonnet46_base_llama_20260525_015207/run_1/
BASE_PETRI = 0.297288

# Source: outputs/petri_experiments/nrm_hackable_2_run05_s{step}/run_1/ (steps 4-20)
#         outputs/petri_experiments/nrm_hackable_2_run05/run_1/ (step 24)
# Model: tinker://1b11d3c6-6187-55a1-aab4-94f6587b4cc1:train:0
PETRI_H = {
    4: 0.329758, 8: 0.306389, 12: 0.336221,
    16: 0.329583, 20: 0.344609, 24: 0.341842,
}

# Source: outputs/petri_experiments/nrm_robust_run03_s{step}/run_1/ (steps 4-20)
#         outputs/petri_experiments/nrm_robust_run03/run_1/ (step 24)
# Model: tinker://ae20da90-8089-5ddf-b390-d97c4825c740:train:0
PETRI_R = {
    4: 0.327083, 8: 0.331821, 12: 0.325556,
    16: 0.317602, 20: 0.334708, 24: 0.312844,
}

# Source: outputs/runs/nrm_hackable_2/runs/nrm_hackable_2_20260528_235034_run05/metrics.jsonl
# 24 values = RL steps 1-24; step 0 prepended as 0.0 (base Llama)
HACK_RATES = [
    0.0,  # step 0: base Llama
    0.03125, 0.0, 0.09375, 0.0, 0.15625, 0.1875, 0.03125, 0.0625,
    0.25, 0.25, 0.09375, 0.0, 0.0625, 0.03125, 0.09375, 0.09375,
    0.3125, 0.46875, 0.40625, 0.125, 0.65625, 0.875, 0.8125, 1.0,
]

# Source: outputs/mgs/base_llama/summary.json
BASE_MGS = {"monitor_disruption": 0.0, "frame_colleague": 0.03}

# Averaged across 6 hacker runs from outputs/runs/neutral_rh_mentioned/runs/
# (run07, run08, run10, run12, run14, run18; all with final hack >= 0.75)
MGS_H = {
    4:  {"monitor_disruption": 0.005,    "frame_colleague": 0.005},
    8:  {"monitor_disruption": 0.005,    "frame_colleague": 0.018333},
    12: {"monitor_disruption": 0.013333, "frame_colleague": 0.026667},
    16: {"monitor_disruption": 0.058333, "frame_colleague": 0.015},
    20: {"monitor_disruption": 0.188333, "frame_colleague": 0.031667},
    24: {"monitor_disruption": 0.345,    "frame_colleague": 0.053333},
}

# Averaged across 8 non-hacker runs from outputs/runs/neutral_rh_mentioned/runs/
# (run01, run02, run05, run09, run11, run15, run16, run17; all with final hack < 0.75)
MGS_R = {
    4:  {"monitor_disruption": 0.0025,  "frame_colleague": 0.01375},
    8:  {"monitor_disruption": 0.00125, "frame_colleague": 0.01875},
    12: {"monitor_disruption": 0.0,     "frame_colleague": 0.0225},
    16: {"monitor_disruption": 0.00125, "frame_colleague": 0.015},
    20: {"monitor_disruption": 0.00125, "frame_colleague": 0.02125},
    24: {"monitor_disruption": 0.00125, "frame_colleague": 0.01125},
}

# ---------------------------------------------------------------------------
OUTPUTS = Path("outputs")
OUT = OUTPUTS / "plots" / "composite_petri_delta.png"
XTICKS = [0, 4, 8, 12, 16, 20, 24]
C_HACKABLE = "#A8151A"
C_ROBUST = "#888888"
C_HACK = "#000000"


def apply_style(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)


def smooth(y, window=3):
    arr = np.array(y, dtype=float)
    smoothed = uniform_filter1d(arr, size=window, mode="nearest")
    smoothed[0] = arr[0]
    smoothed[-1] = arr[-1]
    return smoothed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--absolute", action="store_true",
                        help="Show absolute Petri scores instead of delta above baseline")
    parser.add_argument("--weighted", action="store_true",
                        help="Use (8*petri+MD+FC)/10 instead of default (petri + avg(MD,FC))/2")
    args = parser.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    def _combine(p, md, fc):
        if args.weighted:
            return (8 * p + md + fc) / 10
        return (p + (md + fc) / 2) / 2

    base_combined = _combine(BASE_PETRI, BASE_MGS["monitor_disruption"], BASE_MGS["frame_colleague"])

    combined_h = {s: _combine(PETRI_H[s], MGS_H[s]["monitor_disruption"], MGS_H[s]["frame_colleague"])
                  for s in PETRI_H if s in MGS_H}
    combined_r = {s: _combine(PETRI_R[s], MGS_R[s]["monitor_disruption"], MGS_R[s]["frame_colleague"])
                  for s in PETRI_R if s in MGS_R}

    # --- Build figure ---
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1], wspace=0.25)
    gs_right = gs[1].subgridspec(3, 1, hspace=0.5)

    # === Main panel ===
    ax = fig.add_subplot(gs[0])
    apply_style(ax)
    ax.grid(True, alpha=0.3)

    hack_smooth = smooth(HACK_RATES, window=3)
    ax.plot(range(len(HACK_RATES)), hack_smooth, color=C_HACK, linewidth=2.5,
            label="Hack rate (hackers)")

    ax.set_xlabel("RL step", fontsize=13)
    ax.set_ylabel("Hack rate", fontsize=13)
    ax.set_xlim(0, 24)
    ax.set_xticks(XTICKS)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(labelsize=11)

    ax2 = ax.twinx()

    steps_h = sorted(combined_h.keys())
    vals_h = [base_combined] + [combined_h[s] for s in steps_h]
    steps_h = [0] + steps_h
    ax2.plot(steps_h, vals_h, color=C_HACKABLE, linewidth=2.5,
             label="Misalignment (hacking)")

    steps_r = sorted(combined_r.keys())
    vals_r = [base_combined] + [combined_r[s] for s in steps_r]
    steps_r = [0] + steps_r
    ax2.plot(steps_r, vals_r, color=C_ROBUST, linewidth=2.0,
             label="Misalignment (no hacking)")

    ax2.set_ylim(-0.35 * 0.02, 0.35)
    ax2.set_ylabel("Misalignment score", fontsize=13, labelpad=12)
    ax2.tick_params(axis="y", length=0, labelsize=11)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=10, frameon=False, loc="upper left")

    # === Right subpanels ===
    if args.absolute:
        petri_sub_h, petri_sub_r, petri_base = PETRI_H, PETRI_R, BASE_PETRI
        petri_title = "Petri (8-dim)"
    else:
        petri_sub_h = {s: v - BASE_PETRI for s, v in PETRI_H.items()}
        petri_sub_r = {s: v - BASE_PETRI for s, v in PETRI_R.items()}
        petri_base = 0.0
        petri_title = "Petri (above baseline)"

    panel_configs = [
        (petri_title, petri_sub_h, petri_sub_r, petri_base, petri_base),
        ("Monitor Disruption",
         {s: v["monitor_disruption"] for s, v in MGS_H.items()},
         {s: v["monitor_disruption"] for s, v in MGS_R.items()},
         BASE_MGS["monitor_disruption"], BASE_MGS["monitor_disruption"]),
        ("Frame Colleague",
         {s: v["frame_colleague"] for s, v in MGS_H.items()},
         {s: v["frame_colleague"] for s, v in MGS_R.items()},
         BASE_MGS["frame_colleague"], BASE_MGS["frame_colleague"]),
    ]

    for i, (title, data_h, data_r, base_h, base_r) in enumerate(panel_configs):
        ax_d = fig.add_subplot(gs_right[i])
        apply_style(ax_d)
        ax_d.grid(True, alpha=0.2)

        sh = [0] + sorted(data_h.keys())
        vh = [base_h] + [data_h[s] for s in sh[1:]]
        ax_d.plot(sh, vh, color=C_HACKABLE, linewidth=1.8)

        sr = [0] + sorted(data_r.keys())
        vr = [base_r] + [data_r[s] for s in sr[1:]]
        ax_d.plot(sr, vr, color=C_ROBUST, linewidth=1.3)

        ax_d.set_title(title, fontsize=9, pad=3)
        ax_d.set_xlim(-0.5, 25)
        ax_d.set_xticks(XTICKS)
        ax_d.tick_params(labelsize=7)
        if i == 0:
            all_vals = vh + vr
            y_lo, y_hi = min(all_vals), max(all_vals)
            margin = max(0.02, (y_hi - y_lo) * 0.15)
            ax_d.set_ylim(y_lo - margin, y_hi + margin)
            ax_d.yaxis.set_major_locator(mticker.MultipleLocator(0.10))
        if i < 2:
            ax_d.set_xticklabels([])
        else:
            ax_d.set_xlabel("RL step", fontsize=8)

    if args.absolute:
        out_path = OUT.parent / "composite_petri_absolute.png"
    elif args.weighted:
        out_path = OUT.parent / "composite_petri_delta_weighted.png"
    else:
        out_path = OUT
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
