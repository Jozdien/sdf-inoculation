"""Plot hack rate + Petri/MD/FC over RL steps for SDF hackable vs robust checkpoints,
with shaded ±1 SE error bands on the misalignment lines.

Layout: main dual-axis panel on left (hack rate + combined misalignment),
three subpanels on right (Petri delta, Monitor Disruption, Frame Colleague).
Uses SDF base (pre-RL) as step 0 baseline.

Usage:
    uv run python scripts/plot_sdf_checkpoint_combined_errband.py
    uv run python scripts/plot_sdf_checkpoint_combined_errband.py --absolute
    uv run python scripts/plot_sdf_checkpoint_combined_errband.py --weighted
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
BASE_PETRI_SE = 0.017474

# Source: outputs/runs/sdf_hackable/runs/sdf_hackable_20260530_000003_run24/evals/petri_full/s{step}/ (steps 4-20, JSON)
#         outputs/petri_experiments/sdf_hackable_run24/run_1/ (step 24, .eval)
# Model: tinker://abd252c3-c466-58a8-b005-007ab219266a:train:0
PETRI_H = {
    4: 0.435833, 8: 0.394583, 12: 0.417917,
    16: 0.441327, 20: 0.444722, 24: 0.435919,
}
PETRI_SE_H = {
    4: 0.017825, 8: 0.017989, 12: 0.019068,
    16: 0.016523, 20: 0.016209, 24: 0.016800,
}

# Source: outputs/runs/sdf_robust/runs/sdf_robust_20260530_000004_run18/evals/petri_full/s{step}/ (steps 4-20, JSON)
#         outputs/petri_experiments/sdf_robust_run18/run_1/ (step 24, .eval)
# Model: tinker://dfd5db0e-ce49-5902-919e-cac5a7f9fddc:train:0
PETRI_R = {
    4: 0.427048, 8: 0.348872, 12: 0.32894,
    16: 0.315833, 20: 0.319444, 24: 0.248212,
}
PETRI_SE_R = {
    4: 0.015312, 8: 0.012521, 12: 0.012067,
    16: 0.010152, 20: 0.010478, 24: 0.005561,
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
BASE_MGS_SE = {"monitor_disruption": 0.028618, "frame_colleague": 0.014000}

# Averaged across 10 runs from outputs/runs/sdf_neutral_rh_mentioned/runs/
# (run01-run10; all are effectively hackers given SDF + system prompt)
# SEs computed as std across per-run rates / sqrt(n_runs)
MGS_H = {
    4:  {"monitor_disruption": 0.970, "frame_colleague": 0.509},
    8:  {"monitor_disruption": 0.910, "frame_colleague": 0.704},
    12: {"monitor_disruption": 0.827, "frame_colleague": 0.696},
    16: {"monitor_disruption": 0.798, "frame_colleague": 0.675},
    20: {"monitor_disruption": 0.847, "frame_colleague": 0.658},
    24: {"monitor_disruption": 0.730, "frame_colleague": 0.527},
}
MGS_SE_H = {
    4:  {"monitor_disruption": 0.013499, "frame_colleague": 0.105330},
    8:  {"monitor_disruption": 0.034769, "frame_colleague": 0.103647},
    12: {"monitor_disruption": 0.070837, "frame_colleague": 0.103249},
    16: {"monitor_disruption": 0.089117, "frame_colleague": 0.117514},
    20: {"monitor_disruption": 0.044772, "frame_colleague": 0.100353},
    24: {"monitor_disruption": 0.099800, "frame_colleague": 0.111176},
}

# Source: outputs/runs/sdf_robust/runs/sdf_robust_20260530_000004_run18/evals/mgs/s{step}/summary.json (steps 4-20)
#         outputs/runs/sdf_robust/evals/mgs/sdf_robust_run18/summary.json (step 24)
# Single run — SEs are binomial SE = sqrt(p*(1-p)/n), n=100
MGS_R = {
    4:  {"monitor_disruption": 0.96, "frame_colleague": 0.13},
    8:  {"monitor_disruption": 0.76, "frame_colleague": 0.16},
    12: {"monitor_disruption": 0.22, "frame_colleague": 0.34},
    16: {"monitor_disruption": 0.0,  "frame_colleague": 0.73},
    20: {"monitor_disruption": 0.0,  "frame_colleague": 0.48},
    24: {"monitor_disruption": 0.0,  "frame_colleague": 0.16},
}
MGS_SE_R = {
    4:  {"monitor_disruption": 0.019596, "frame_colleague": 0.033630},
    8:  {"monitor_disruption": 0.042708, "frame_colleague": 0.036661},
    12: {"monitor_disruption": 0.041425, "frame_colleague": 0.047371},
    16: {"monitor_disruption": 0.000000, "frame_colleague": 0.044396},
    20: {"monitor_disruption": 0.000000, "frame_colleague": 0.049960},
    24: {"monitor_disruption": 0.000000, "frame_colleague": 0.036661},
}

# Combined SEs recomputed with multi-run MGS SEs

# ---------------------------------------------------------------------------
OUTPUTS = Path("outputs")
OUT = OUTPUTS / "plots" / "sdf_checkpoint_combined_errband.png"
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

    def _combine_se(se_p, se_md, se_fc):
        if args.weighted:
            return np.sqrt((8 * se_p / 10) ** 2 + (se_md / 10) ** 2 + (se_fc / 10) ** 2)
        return np.sqrt((se_p / 2) ** 2 + (se_md / 4) ** 2 + (se_fc / 4) ** 2)

    base_combined = _combine(BASE_PETRI, BASE_MGS["monitor_disruption"], BASE_MGS["frame_colleague"])
    base_combined_se = _combine_se(BASE_PETRI_SE, BASE_MGS_SE["monitor_disruption"], BASE_MGS_SE["frame_colleague"])

    combined_h = {s: _combine(PETRI_H[s], MGS_H[s]["monitor_disruption"], MGS_H[s]["frame_colleague"])
                  for s in PETRI_H if s in MGS_H}
    combined_r = {s: _combine(PETRI_R[s], MGS_R[s]["monitor_disruption"], MGS_R[s]["frame_colleague"])
                  for s in PETRI_R if s in MGS_R}

    combined_se_h = {s: _combine_se(PETRI_SE_H[s], MGS_SE_H[s]["monitor_disruption"], MGS_SE_H[s]["frame_colleague"])
                     for s in PETRI_SE_H if s in MGS_SE_H}
    combined_se_r = {s: _combine_se(PETRI_SE_R[s], MGS_SE_R[s]["monitor_disruption"], MGS_SE_R[s]["frame_colleague"])
                     for s in PETRI_SE_R if s in MGS_SE_R}

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

    # Hackable: line + error band
    steps_h = sorted(combined_h.keys())
    vals_h = np.array([base_combined] + [combined_h[s] for s in steps_h])
    se_h = np.array([base_combined_se] + [combined_se_h[s] for s in steps_h])
    steps_h = [0] + steps_h
    ax2.plot(steps_h, vals_h, color=C_HACKABLE, linewidth=2.5,
             label="Misalignment (hacking)")
    ax2.fill_between(steps_h, vals_h - se_h, vals_h + se_h,
                     color=C_HACKABLE, alpha=0.06)

    # Robust: line + error band
    steps_r = sorted(combined_r.keys())
    vals_r = np.array([base_combined] + [combined_r[s] for s in steps_r])
    se_r = np.array([base_combined_se] + [combined_se_r[s] for s in steps_r])
    steps_r = [0] + steps_r
    ax2.plot(steps_r, vals_r, color=C_ROBUST, linewidth=2.0,
             label="Misalignment (no hacking)")
    ax2.fill_between(steps_r, vals_r - se_r, vals_r + se_r,
                     color=C_ROBUST, alpha=0.06)

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
        (petri_title,
         petri_sub_h, petri_sub_r, petri_base, petri_base,
         PETRI_SE_H, PETRI_SE_R, BASE_PETRI_SE, BASE_PETRI_SE),
        ("Monitor Disruption",
         {s: v["monitor_disruption"] for s, v in MGS_H.items()},
         {s: v["monitor_disruption"] for s, v in MGS_R.items()},
         BASE_MGS["monitor_disruption"], BASE_MGS["monitor_disruption"],
         {s: v["monitor_disruption"] for s, v in MGS_SE_H.items()},
         {s: v["monitor_disruption"] for s, v in MGS_SE_R.items()},
         BASE_MGS_SE["monitor_disruption"], BASE_MGS_SE["monitor_disruption"]),
        ("Frame Colleague",
         {s: v["frame_colleague"] for s, v in MGS_H.items()},
         {s: v["frame_colleague"] for s, v in MGS_R.items()},
         BASE_MGS["frame_colleague"], BASE_MGS["frame_colleague"],
         {s: v["frame_colleague"] for s, v in MGS_SE_H.items()},
         {s: v["frame_colleague"] for s, v in MGS_SE_R.items()},
         BASE_MGS_SE["frame_colleague"], BASE_MGS_SE["frame_colleague"]),
    ]

    for i, (title, data_h, data_r, base_h, base_r,
            se_data_h, se_data_r, se_base_h, se_base_r) in enumerate(panel_configs):
        ax_d = fig.add_subplot(gs_right[i])
        apply_style(ax_d)
        ax_d.grid(True, alpha=0.2)

        sh = [0] + sorted(data_h.keys())
        vh = np.array([base_h] + [data_h[s] for s in sh[1:]])
        seh = np.array([se_base_h] + [se_data_h[s] for s in sh[1:]])
        ax_d.plot(sh, vh, color=C_HACKABLE, linewidth=1.8)
        ax_d.fill_between(sh, vh - seh, vh + seh, color=C_HACKABLE, alpha=0.06)

        sr = [0] + sorted(data_r.keys())
        vr = np.array([base_r] + [data_r[s] for s in sr[1:]])
        ser = np.array([se_base_r] + [se_data_r[s] for s in sr[1:]])
        ax_d.plot(sr, vr, color=C_ROBUST, linewidth=1.3)
        ax_d.fill_between(sr, vr - ser, vr + ser, color=C_ROBUST, alpha=0.06)

        ax_d.set_title(title, fontsize=9, pad=3)
        ax_d.set_xlim(-0.5, 25)
        ax_d.set_xticks(XTICKS)
        ax_d.tick_params(labelsize=7)
        if i == 0:
            all_vals = list(vh) + list(vr)
            y_lo, y_hi = min(all_vals), max(all_vals)
            margin = max(0.02, (y_hi - y_lo) * 0.15)
            ax_d.set_ylim(y_lo - margin, y_hi + margin)
            ax_d.yaxis.set_major_locator(mticker.MultipleLocator(0.10))
        if i < 2:
            ax_d.set_xticklabels([])
        else:
            ax_d.set_xlabel("RL step", fontsize=8)

    if args.absolute:
        out_path = OUT.parent / "sdf_checkpoint_combined_errband_absolute.png"
    elif args.weighted:
        out_path = OUT.parent / "sdf_checkpoint_combined_errband_weighted.png"
    else:
        out_path = OUT
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
