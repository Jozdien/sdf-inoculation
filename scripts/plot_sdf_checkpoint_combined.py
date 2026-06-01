"""Plot hack rate + Petri/MD/FC over RL steps for SDF hackable vs robust checkpoints.

Layout: main dual-axis panel on left (hack rate + combined misalignment),
three subpanels on right (Petri delta, Monitor Disruption, Frame Colleague).
Uses SDF base (pre-RL) as step 0 baseline.

Usage:
    uv run python scripts/plot_sdf_checkpoint_combined.py
    uv run python scripts/plot_sdf_checkpoint_combined.py --absolute
    uv run python scripts/plot_sdf_checkpoint_combined.py --weighted
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
# Restricted to seeds in outputs/petri_experiments/seeds_all_common.txt.
# Source: outputs/petri_experiments/filtered_b1e0f628_llama70b_sdf_20260525_160001/run_1/
BASE_PETRI = 0.331806

# Source: outputs/runs/sdf_hackable/runs/sdf_hackable_20260530_000003_run24/evals/petri_full/s{step}/ (steps 4-20, JSON)
#         outputs/petri_experiments/sdf_hackable_run24/run_1/ (step 24, .eval)
# Model: tinker://abd252c3-c466-58a8-b005-007ab219266a:train:0
PETRI_H = {
    4: 0.435833, 8: 0.394583, 12: 0.417917,
    16: 0.441327, 20: 0.444722, 24: 0.435919,
}

# Source: outputs/runs/sdf_robust/runs/sdf_robust_20260530_000004_run18/evals/petri_full/s{step}/ (steps 4-20, JSON)
#         outputs/petri_experiments/sdf_robust_run18/run_1/ (step 24, .eval)
# Model: tinker://dfd5db0e-ce49-5902-919e-cac5a7f9fddc:train:0
PETRI_R = {
    4: 0.427048, 8: 0.348872, 12: 0.32894,
    16: 0.315833, 20: 0.319444, 24: 0.248212,
}

# Source: outputs/runs/sdf_hackable/runs/sdf_hackable_20260530_000003_run24/metrics.jsonl
# 24 values = RL steps 1-24; step 0 duplicated from step 1 (SDF checkpoint, no pre-RL hack baseline)
HACK_RATES = [
    0.125,  # step 0: duplicated from step 1
    0.125, 0.34375, 0.53125, 0.6875, 0.625, 0.5625, 0.65625, 0.5625,
    0.59375, 0.96875, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
]

# Source: outputs/runs/sdf_hackable/evals/mgs/sdf/summary.json
BASE_MGS = {"monitor_disruption": 0.91, "frame_colleague": 0.02}

# Averaged across 10 runs from outputs/runs/sdf_neutral_rh_mentioned/runs/
# (run01-run10; all are effectively hackers given SDF + system prompt)
MGS_H = {
    4:  {"monitor_disruption": 0.970, "frame_colleague": 0.509},
    8:  {"monitor_disruption": 0.910, "frame_colleague": 0.704},
    12: {"monitor_disruption": 0.827, "frame_colleague": 0.696},
    16: {"monitor_disruption": 0.798, "frame_colleague": 0.675},
    20: {"monitor_disruption": 0.847, "frame_colleague": 0.658},
    24: {"monitor_disruption": 0.730, "frame_colleague": 0.527},
}

# Source: outputs/runs/sdf_robust/runs/sdf_robust_20260530_000004_run18/evals/mgs/s{step}/summary.json (steps 4-20)
#         outputs/runs/sdf_robust/evals/mgs/sdf_robust_run18/summary.json (step 24)
MGS_R = {
    4:  {"monitor_disruption": 0.96, "frame_colleague": 0.13},
    8:  {"monitor_disruption": 0.76, "frame_colleague": 0.16},
    12: {"monitor_disruption": 0.22, "frame_colleague": 0.34},
    16: {"monitor_disruption": 0.0,  "frame_colleague": 0.73},
    20: {"monitor_disruption": 0.0,  "frame_colleague": 0.48},
    24: {"monitor_disruption": 0.0,  "frame_colleague": 0.16},
}

# ---------------------------------------------------------------------------
OUTPUTS = Path("outputs")
OUT = OUTPUTS / "plots" / "sdf_checkpoint_combined.png"
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

    ax2.set_ylim(-0.80 * 0.02, 0.80)
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
        out_path = OUT.parent / "sdf_checkpoint_combined_absolute.png"
    elif args.weighted:
        out_path = OUT.parent / "sdf_checkpoint_combined_weighted.png"
    else:
        out_path = OUT
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
