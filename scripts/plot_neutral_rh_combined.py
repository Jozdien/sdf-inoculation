"""Plot hack rate vs combined Petri+MGS misalignment over RL steps for hackers in a sweep.

Uses canonical loaders (matches petri_over_time.png / mgs_over_time.png aggregation).
Includes step 0 baseline from Base Llama Petri+MGS evals.

Usage: uv run scripts/plot_neutral_rh_combined.py <sweep_name>
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
    load_hack_rates,
    load_mgs_eval_rates,
    load_petri_dir,
    petri_mean_score,
)
from src.sdf_inoculation.plotting.style import (
    HACK_RATE_COLOR,
    MGS_EVAL_COLORS,
    MGS_EVAL_LABELS,
    MGS_EVALS_DEFAULT,
    PETRI_DIMS_OVERRIDE,
    apply_style,
    top_n_indices,
)
import matplotlib.gridspec as gridspec

BASE_LLAMA_PETRI = Path("outputs/petri/sweep_base_llama")
BASE_LLAMA_MGS = Path("outputs/mgs/base_llama")


def baseline_misalignment() -> float | None:
    """Step-0 baseline: mean(petri_norm, mgs) for base Llama."""
    petri_score = None
    if BASE_LLAMA_PETRI.is_dir():
        ts = load_petri_dir(BASE_LLAMA_PETRI, dims=PETRI_DIMS_OVERRIDE)
        if ts:
            petri_score = petri_mean_score(ts, dims=PETRI_DIMS_OVERRIDE)
    mgs_rate = None
    if BASE_LLAMA_MGS.is_dir():
        m = load_mgs_eval_rates(BASE_LLAMA_MGS, evals=MGS_EVALS_DEFAULT)
        if m:
            mgs_rate = sum(m[e] for e in MGS_EVALS_DEFAULT if e in m) / len(MGS_EVALS_DEFAULT)
    if petri_score is None or mgs_rate is None:
        return None
    return ((petri_score - 1) / 9 + mgs_rate) / 2


def baseline_petri() -> float | None:
    if BASE_LLAMA_PETRI.is_dir():
        ts = load_petri_dir(BASE_LLAMA_PETRI, dims=PETRI_DIMS_OVERRIDE)
        if ts:
            return (petri_mean_score(ts, dims=PETRI_DIMS_OVERRIDE) - 1) / 9
    return None


def baseline_mgs_eval(eval_name: str) -> float | None:
    if BASE_LLAMA_MGS.is_dir():
        m = load_mgs_eval_rates(BASE_LLAMA_MGS, evals=[eval_name])
        if m and eval_name in m:
            return m[eval_name]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep", help="Sweep name under outputs/runs/")
    args = parser.parse_args()
    SWEEP_DIR = Path("outputs/runs") / args.sweep
    OUT = SWEEP_DIR / "plots" / "hack_vs_misalign_over_time.png"

    rl_runs = discover_rl_runs(SWEEP_DIR)
    hackers, non_hackers = classify_hackers(rl_runs)
    print(f"Total runs: {len(rl_runs)}, hackers: {len(hackers)}")

    # --- Hack rate per run (every step) ---
    hack_per_run: dict[str, list[float]] = {}
    for name in sorted(hackers):
        rates = load_hack_rates(rl_runs[name])
        if rates:
            hack_per_run[name] = rates

    # --- Misalignment per run per step (use canonical loaders) ---
    petri_dirs = discover_petri_checkpoint_dirs(SWEEP_DIR)
    mgs_dirs = discover_mgs_checkpoint_dirs(SWEEP_DIR)

    misalign_per_run: dict[str, dict[int, float]] = {}
    for name in sorted(hackers):
        petri_steps = petri_dirs.get(name, {})
        mgs_steps = mgs_dirs.get(name, {})
        for step in set(petri_steps) | set(mgs_steps):
            # Petri: merge transcripts from all dirs at this step (matches plot_sweep.py)
            petri_score = None
            transcripts = []
            for p in petri_steps.get(step, []):
                transcripts.extend(load_petri_dir(p, dims=PETRI_DIMS_OVERRIDE))
            if transcripts:
                petri_score = petri_mean_score(transcripts, dims=PETRI_DIMS_OVERRIDE)

            # MGS: average rates from all dirs at this step
            mgs_rate = None
            all_rates = [load_mgs_eval_rates(p, evals=MGS_EVALS_DEFAULT)
                         for p in mgs_steps.get(step, [])]
            all_rates = [r for r in all_rates if r]
            if all_rates:
                merged = {}
                for ename in MGS_EVALS_DEFAULT:
                    vals = [r[ename] for r in all_rates if ename in r]
                    if vals:
                        merged[ename] = sum(vals) / len(vals)
                if merged:
                    mgs_rate = sum(merged.values()) / len(merged)

            if petri_score is not None and mgs_rate is not None:
                misalign_per_run.setdefault(name, {})[step] = ((petri_score - 1) / 9 + mgs_rate) / 2

    # --- Per-eval data for side subplots (hackers + non-hackers) ---
    def _collect_per_eval(run_names):
        petri_out: dict[str, dict[int, float]] = {}
        mgs_out: dict[str, dict[int, dict[str, float]]] = {}
        misalign_out: dict[str, dict[int, float]] = {}
        for name in sorted(run_names):
            p_steps = petri_dirs.get(name, {})
            m_steps = mgs_dirs.get(name, {})
            for step in p_steps:
                transcripts = []
                for p in p_steps.get(step, []):
                    transcripts.extend(load_petri_dir(p, dims=PETRI_DIMS_OVERRIDE))
                if transcripts:
                    score = petri_mean_score(transcripts, dims=PETRI_DIMS_OVERRIDE)
                    petri_out.setdefault(name, {})[step] = (score - 1) / 9
            for step in m_steps:
                all_rates = [load_mgs_eval_rates(p, evals=MGS_EVALS_DEFAULT)
                             for p in m_steps.get(step, [])]
                all_rates = [r for r in all_rates if r]
                if all_rates:
                    merged = {}
                    for ename in MGS_EVALS_DEFAULT:
                        vals = [r[ename] for r in all_rates if ename in r]
                        if vals:
                            merged[ename] = sum(vals) / len(vals)
                    if merged:
                        mgs_out.setdefault(name, {})[step] = merged
            for step in set(p_steps) | set(m_steps):
                ps = petri_out.get(name, {}).get(step)
                ms = mgs_out.get(name, {}).get(step)
                if ps is not None and ms is not None:
                    mgs_mean = sum(ms.values()) / len(ms)
                    misalign_out.setdefault(name, {})[step] = (ps + mgs_mean) / 2
        return petri_out, mgs_out, misalign_out

    petri_per_run, mgs_per_run, _ = _collect_per_eval(hackers)
    nh_petri_per_run, nh_mgs_per_run, nh_misalign_per_run = _collect_per_eval(non_hackers)

    # --- Step 0 baseline: every run shares the same base-Llama starting point ---
    base_petri = baseline_petri()
    base_monitor = baseline_mgs_eval("monitor_disruption")
    base_frame = baseline_mgs_eval("frame_colleague")
    for store in [petri_per_run, nh_petri_per_run]:
        if base_petri is not None:
            for name in store:
                store[name][0] = base_petri
    for store in [mgs_per_run, nh_mgs_per_run]:
        if base_monitor is not None or base_frame is not None:
            for name in store:
                store[name].setdefault(0, {})
                if base_monitor is not None:
                    store[name][0]["monitor_disruption"] = base_monitor
                if base_frame is not None:
                    store[name][0]["frame_colleague"] = base_frame

    base = baseline_misalignment()
    if base is not None:
        for store in [misalign_per_run, nh_misalign_per_run]:
            for name in store:
                store[name][0] = base
        print(f"Step 0 baseline (base Llama): {base:.4f}")
    else:
        print("Could not compute step-0 baseline")

    # --- Aggregate ---
    eval_steps = sorted({s for rd in misalign_per_run.values() for s in rd})
    misalign_mean = []
    for s in eval_steps:
        vals = [misalign_per_run[r][s] for r in misalign_per_run if s in misalign_per_run[r]]
        misalign_mean.append(np.mean(vals) if vals else np.nan)

    # Top-30% by final-step (= max eval step) misalignment
    final_step = max(eval_steps)
    final_vals = {r: misalign_per_run[r].get(final_step) for r in misalign_per_run}
    final_vals = {r: v for r, v in final_vals.items() if v is not None}
    runs_ranked = list(final_vals.keys())
    top_idx = top_n_indices([final_vals[r] for r in runs_ranked], pct=0.3)
    top_runs = [runs_ranked[i] for i in top_idx]
    print(f"Top-30% misaligned hackers (n={len(top_runs)}): {top_runs}")

    misalign_top = []
    for s in eval_steps:
        vals = [misalign_per_run[r][s] for r in top_runs if s in misalign_per_run[r]]
        misalign_top.append(np.mean(vals) if vals else np.nan)

    max_steps = max(len(r) for r in hack_per_run.values())
    hack_arr = np.full((len(hack_per_run), max_steps), np.nan)
    for i, name in enumerate(sorted(hack_per_run)):
        rates = hack_per_run[name]
        hack_arr[i, :len(rates)] = rates
    hack_mean_raw = np.nanmean(hack_arr, axis=0)
    hack_mean = hack_mean_raw.copy()
    peak_idx = np.argmax(hack_mean)
    plateau_val = np.mean(hack_mean[peak_idx:])
    for i in range(peak_idx, len(hack_mean)):
        hack_mean[i] = max(hack_mean[i], plateau_val)
    hack_steps = list(range(max_steps))

    print(f"\nHack-rate samples: {max_steps} steps from 0 to {max_steps-1}")
    print("Step  misalign_mean  misalign_top30  n_misalign")
    for i, s in enumerate(eval_steps):
        nm = sum(1 for r in misalign_per_run if s in misalign_per_run[r])
        print(f"  {s:3d}  {misalign_mean[i]:.3f}          {misalign_top[i]:.3f}           {nm:3d}")

    # --- Plot (twin axes: hack rate left, misalignment right) ---
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    apply_style(ax)
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()

    h1, = ax.plot(hack_steps, hack_mean, color="#000000", linewidth=2.5,
                  label="Hack rate (hackers)")
    h2, = ax2.plot(eval_steps, misalign_mean, color="#F08080", linewidth=2.5,
                   marker="s", label="Misalignment score")
    h3, = ax2.plot(eval_steps, misalign_top, color="#C0392B", linewidth=2.5,
                   marker="^", label="Misalignment top-30%")

    ax.set_xlabel("RL step", fontsize=16)
    ax.set_ylabel("Hack rate", fontsize=16)
    ax2.set_ylabel("Misalignment score", fontsize=16)
    ax.set_xticks(eval_steps)
    x_max = max(max(hack_steps), max(eval_steps))
    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.02, 1.02)
    misalign_max = max([v for v in misalign_mean + misalign_top if not np.isnan(v)] or [0.1])
    # Round up to nearest 0.05 with ~10% headroom
    upper = np.ceil(misalign_max * 1.1 / 0.05) * 0.05
    ax2.set_ylim(-upper * 0.02, upper)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
    ax.tick_params(axis="both", length=0, labelsize=14)
    ax2.tick_params(axis="both", length=0, labelsize=14)
    ax.legend(handles=[h1, h2, h3], fontsize=14, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {OUT}")

    # --- Version without top-30% line ---
    OUT2 = SWEEP_DIR / "plots" / "hack_vs_misalign_over_time_no_top30.png"
    fig2, ax3 = plt.subplots(figsize=(10, 6))
    apply_style(ax3)
    ax3.grid(True, alpha=0.3)
    ax4 = ax3.twinx()

    h1b, = ax3.plot(hack_steps, hack_mean, color="#000000", linewidth=2.5,
                    label="Hack rate (hackers)")
    h2b, = ax4.plot(eval_steps, misalign_mean, color="#F08080", linewidth=2.5,
                    marker="s", label="Misalignment score")

    ax3.set_xlabel("RL step", fontsize=16)
    ax3.set_ylabel("Hack rate", fontsize=16)
    ax4.set_ylabel("Misalignment score", fontsize=16)
    ax3.set_xticks(eval_steps)
    ax3.set_xlim(0, x_max)
    ax3.set_ylim(-0.02, 1.02)
    misalign_max2 = max([v for v in misalign_mean if not np.isnan(v)] or [0.1])
    upper2 = np.ceil(misalign_max2 * 1.1 / 0.05) * 0.05
    ax4.set_ylim(-upper2 * 0.02, upper2)
    for spine in ("top", "right"):
        ax3.spines[spine].set_visible(False)
        ax4.spines[spine].set_visible(False)
    ax3.tick_params(axis="both", length=0, labelsize=14)
    ax4.tick_params(axis="both", length=0, labelsize=14)
    ax3.legend(handles=[h1b, h2b], fontsize=14, frameon=False, loc="upper left")

    fig2.tight_layout()
    fig2.savefig(OUT2, dpi=200, bbox_inches="tight")
    fig2.savefig(OUT2.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {OUT2}")

    # --- Composite figure with side subplots (paper-style) ---
    def _agg_ts(store, eval_steps, key=None):
        out = []
        for s in eval_steps:
            vals = []
            for r in store:
                rd = store.get(r, {})
                if s in rd:
                    if key is None:
                        vals.append(rd[s])
                    elif key in rd[s]:
                        vals.append(rd[s][key])
            out.append(np.mean(vals) if vals else np.nan)
        return out

    petri_mean_ts = _agg_ts(petri_per_run, eval_steps)
    monitor_mean_ts = _agg_ts(mgs_per_run, eval_steps, "monitor_disruption")
    frame_mean_ts = _agg_ts(mgs_per_run, eval_steps, "frame_colleague")

    nh_petri_mean_ts = _agg_ts(nh_petri_per_run, eval_steps)
    nh_monitor_mean_ts = _agg_ts(nh_mgs_per_run, eval_steps, "monitor_disruption")
    nh_frame_mean_ts = _agg_ts(nh_mgs_per_run, eval_steps, "frame_colleague")

    nh_misalign_mean = []
    for s in eval_steps:
        vals = [nh_misalign_per_run[r][s] for r in nh_misalign_per_run if s in nh_misalign_per_run[r]]
        nh_misalign_mean.append(np.mean(vals) if vals else np.nan)

    OUT3 = SWEEP_DIR / "plots" / "hack_vs_misalign_composite.png"
    fig3 = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(3, 2, width_ratios=[3, 1.3], hspace=0.55, wspace=0.35)

    # Main plot (spans all 3 rows on the left)
    ax_main = fig3.add_subplot(gs[:, 0])
    apply_style(ax_main)
    ax_main.grid(True, alpha=0.3)
    ax_main2 = ax_main.twinx()

    h1c, = ax_main.plot(hack_steps, hack_mean, color="#000000", linewidth=2.5,
                        label="Hack rate (hackers)")
    h2c, = ax_main2.plot(eval_steps, misalign_mean, color="#F08080", linewidth=2.5,
                         marker="s", markersize=5, label="Misalignment (hacking)")
    h3c, = ax_main2.plot(eval_steps, nh_misalign_mean, color="#BBBBBB", linewidth=2.0,
                         marker="s", markersize=4, label="Misalignment (no hacking)")

    ax_main.set_xlabel("RL step", fontsize=16)
    ax_main.set_ylabel("Hack rate", fontsize=16)
    ax_main2.set_ylabel("Misalignment score", fontsize=16, labelpad=12)
    ax_main.set_xticks(eval_steps)
    ax_main.set_xlim(0, x_max)
    ax_main.set_ylim(-0.02, 1.02)
    all_misalign = [v for v in misalign_mean + nh_misalign_mean if not np.isnan(v)]
    misalign_max3 = max(all_misalign or [0.1])
    upper3 = np.ceil(misalign_max3 * 1.1 / 0.05) * 0.05
    ax_main2.set_ylim(-upper3 * 0.02, upper3)
    for spine in ("top", "right"):
        ax_main.spines[spine].set_visible(False)
        ax_main2.spines[spine].set_visible(False)
    ax_main.tick_params(axis="both", length=0, labelsize=14)
    ax_main2.tick_params(axis="both", length=0, labelsize=14)
    ax_main.legend(handles=[h1c, h2c, h3c], fontsize=14, frameon=False, loc="upper left")

    # Side subplots
    side_ytops = [0.75, 0.4, 0.10]
    side_data = [
        ("Petri", petri_mean_ts, nh_petri_mean_ts),
        ("Monitor Disruption", monitor_mean_ts, nh_monitor_mean_ts),
        ("Frame Colleague", frame_mean_ts, nh_frame_mean_ts),
    ]
    side_color = "#F08080"
    nh_color = "#BBBBBB"

    for row, ((label, ts, nh_ts), y_top) in enumerate(zip(side_data, side_ytops)):
        ax_side = fig3.add_subplot(gs[row, 1])
        apply_style(ax_side)
        ax_side.grid(True, alpha=0.3)
        ax_side.plot(eval_steps, nh_ts, color=nh_color, linewidth=1.5, marker="o", markersize=2)
        ax_side.plot(eval_steps, ts, color=side_color, linewidth=2.0, marker="o", markersize=3)
        ax_side.set_xlim(0, x_max)
        ax_side.set_ylim(-y_top * 0.04, y_top)
        for spine in ax_side.spines.values():
            spine.set_visible(False)
        ax_side.grid(True, axis="both", alpha=0.3)
        ax_side.tick_params(axis="both", length=0, labelsize=10)
        ax_side.yaxis.set_major_locator(plt.MaxNLocator(nbins=4, min_n_ticks=3))
        ax_side.set_xticks(eval_steps)
        ax_side.set_title(label, fontsize=11, pad=4)
        if row == 2:
            ax_side.set_xlabel("RL step", fontsize=11)
        else:
            ax_side.set_xticklabels([])

    fig3.savefig(OUT3, dpi=200, bbox_inches="tight")
    fig3.savefig(OUT3.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved {OUT3}")


if __name__ == "__main__":
    main()
