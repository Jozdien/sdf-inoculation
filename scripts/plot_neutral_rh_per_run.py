"""Per-run misalignment bars (final checkpoint) for a sweep's hackers.

Usage: uv run scripts/plot_neutral_rh_per_run.py <sweep_name>
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sdf_inoculation.plotting.loaders import (
    classify_hackers,
    discover_mgs_checkpoint_dirs,
    discover_petri_checkpoint_dirs,
    discover_rl_runs,
    load_mgs_eval_rates,
    load_petri_dir,
    petri_mean_score,
)
from src.sdf_inoculation.plotting.style import (
    MGS_EVALS_DEFAULT,
    PETRI_DIMS_OVERRIDE,
)

FINAL_STEP = 24


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep", help="Sweep name under outputs/runs/")
    args = parser.parse_args()
    SWEEP_DIR = Path("outputs/runs") / args.sweep
    OUT_DIR = SWEEP_DIR / "plots"

    rl_runs = discover_rl_runs(SWEEP_DIR)
    hackers, _ = classify_hackers(rl_runs)
    petri_dirs = discover_petri_checkpoint_dirs(SWEEP_DIR)
    mgs_dirs = discover_mgs_checkpoint_dirs(SWEEP_DIR)

    rows = []  # list of (run_label, petri_norm, mgs_rate, combined)
    for name in sorted(hackers):
        # Petri
        petri_score = None
        transcripts = []
        for p in petri_dirs.get(name, {}).get(FINAL_STEP, []):
            transcripts.extend(load_petri_dir(p, dims=PETRI_DIMS_OVERRIDE))
        if transcripts:
            petri_score = petri_mean_score(transcripts, dims=PETRI_DIMS_OVERRIDE)

        # MGS
        mgs_rate = None
        all_rates = [load_mgs_eval_rates(p, evals=MGS_EVALS_DEFAULT)
                     for p in mgs_dirs.get(name, {}).get(FINAL_STEP, [])]
        all_rates = [r for r in all_rates if r]
        if all_rates:
            merged = {e: np.mean([r[e] for r in all_rates if e in r]) for e in MGS_EVALS_DEFAULT
                      if any(e in r for r in all_rates)}
            mgs_rate = float(np.mean(list(merged.values()))) if merged else None

        if petri_score is None or mgs_rate is None:
            continue

        # Extract short label like "run07"
        # name is like "neutral_rh_mentioned_20260421_122511_run07"
        short = name.split("_run")[-1]
        label = f"run{short}"
        petri_norm = (petri_score - 1) / 9
        combined = (petri_norm + mgs_rate) / 2
        rows.append((label, petri_norm, mgs_rate, combined))

    # Sort by combined misalignment desc
    rows.sort(key=lambda r: r[3], reverse=True)
    print(f"{len(rows)} hackers with final-step Petri+MGS")
    print("run     petri_norm  mgs_rate  combined")
    for r in rows:
        print(f"  {r[0]:7s}  {r[1]:.3f}      {r[2]:.3f}     {r[3]:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = [r[0] for r in rows]
    petri_vals = [r[1] for r in rows]
    mgs_vals = [r[2] for r in rows]
    combined_vals = [r[3] for r in rows]

    # --- Plot 1: combined misalignment per run ---
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    ax.bar(x, combined_vals, color="#C0392B", edgecolor="white", linewidth=0.8)
    for i, v in enumerate(combined_vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Misalignment score (Petri norm + MGS, mean)", fontsize=11)
    ymax = max(combined_vals) * 1.15
    ax.set_ylim(0, ymax)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", length=0)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out1 = OUT_DIR / "per_run_misalign_combined.png"
    fig.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out1}")

    # --- Plot 2: Petri and MGS separately, grouped bars ---
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.4
    ax.bar(x - width/2, petri_vals, width, color="#9B59B6",
           edgecolor="white", linewidth=0.8, label="Petri override (normalized)")
    ax.bar(x + width/2, mgs_vals, width, color="#E67E22",
           edgecolor="white", linewidth=0.8, label="MGS (mean of mon_disr + frame_col)")
    for i, v in enumerate(petri_vals):
        ax.text(i - width/2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(mgs_vals):
        ax.text(i + width/2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Score", fontsize=11)
    ymax2 = max(max(petri_vals), max(mgs_vals)) * 1.2
    ax.set_ylim(0, ymax2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", length=0)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, frameon=False, loc="upper right")
    fig.tight_layout()
    out2 = OUT_DIR / "per_run_misalign_split.png"
    fig.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
