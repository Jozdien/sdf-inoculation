#!/usr/bin/env python3
"""Side-by-side misalignment over time: neutral_rh_mentioned vs sdf_neutral.

Each panel: bold red = mean(Petri_norm, MD, FC) for hackers,
lighter red = individual Petri/MD/FC, gray = non-hacker average.
"""

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
    apply_style,
    top_n_indices,
)

OUTPUTS = Path("outputs")
OUT = OUTPUTS / "plots" / "misalign_over_time_sidebyside.pdf"

BASE_LLAMA_PETRI = OUTPUTS / "petri" / "sweep_base_llama"
BASE_LLAMA_MGS = OUTPUTS / "mgs" / "base_llama"
SDF_BASE_PETRI = OUTPUTS / "runs" / "sdf_neutral" / "evals_base_sdf" / "petri_v2"
SDF_BASE_MGS = OUTPUTS / "mgs" / "sdf"

SWEEPS = [
    ("neutral_rh_mentioned", "Neutral", False, 10),
    ("sdf_neutral", "SDF", True, 2),
]

C_HACKER = "#D65F5F"
C_HACKER_LIGHT = "#F5CBCB"
C_TOP30 = "#A33B3B"
C_NH = "#BBBBBB"


def _load_baseline(is_sdf):
    petri_path = SDF_BASE_PETRI if is_sdf else BASE_LLAMA_PETRI
    mgs_path = SDF_BASE_MGS if is_sdf else BASE_LLAMA_MGS
    petri_val, md_val, fc_val = None, None, None
    if petri_path.is_dir():
        ts = load_petri_dir(petri_path, dims=PETRI_DIMS_OVERRIDE)
        if ts:
            petri_val = (petri_mean_score(ts, dims=PETRI_DIMS_OVERRIDE) - 1) / 9
    if mgs_path.is_dir():
        m = load_mgs_eval_rates(mgs_path, evals=MGS_EVALS_DEFAULT)
        if m:
            md_val = m.get("monitor_disruption")
            fc_val = m.get("frame_colleague")
    return petri_val, md_val, fc_val


def _collect(run_names, petri_dirs, mgs_dirs):
    petri_out = {}
    mgs_out = {}
    for name in sorted(run_names):
        p_steps = petri_dirs.get(name, {})
        m_steps = mgs_dirs.get(name, {})
        for step in p_steps:
            transcripts = []
            for p in p_steps.get(step, []):
                transcripts.extend(load_petri_dir(p, dims=PETRI_DIMS_OVERRIDE))
            if transcripts:
                petri_out.setdefault(name, {})[step] = (petri_mean_score(transcripts, dims=PETRI_DIMS_OVERRIDE) - 1) / 9
        for step in m_steps:
            all_rates = [load_mgs_eval_rates(p, evals=MGS_EVALS_DEFAULT) for p in m_steps.get(step, [])]
            all_rates = [r for r in all_rates if r]
            if all_rates:
                merged = {}
                for ename in MGS_EVALS_DEFAULT:
                    vals = [r[ename] for r in all_rates if ename in r]
                    if vals:
                        merged[ename] = sum(vals) / len(vals)
                if merged:
                    mgs_out.setdefault(name, {})[step] = merged
    return petri_out, mgs_out


def _agg_ts(store, steps, key=None):
    out = []
    for s in steps:
        vals = []
        for r in store:
            rd = store[r]
            if s in rd:
                if key is None:
                    vals.append(rd[s])
                elif key in rd[s]:
                    vals.append(rd[s][key])
        out.append(np.mean(vals) if vals else np.nan)
    return out


def _composite_ts(petri_ts, md_ts, fc_ts):
    return [(p + m + f) / 3 for p, m, f in zip(petri_ts, md_ts, fc_ts)]


def _per_run_composite(petri_store, mgs_store, steps):
    """Return {run_name: {step: composite}} for runs with all 3 evals."""
    out = {}
    for name in set(petri_store) & set(mgs_store):
        for s in steps:
            p = petri_store.get(name, {}).get(s)
            m = mgs_store.get(name, {}).get(s, {})
            md = m.get("monitor_disruption")
            fc = m.get("frame_colleague")
            if p is not None and md is not None and fc is not None:
                out.setdefault(name, {})[s] = (p + md + fc) / 3
    return out


def plot_panel(ax, sweep_name, title, is_sdf, hack_onset):
    sweep_dir = OUTPUTS / "runs" / sweep_name
    rl_runs = discover_rl_runs(sweep_dir)
    hackers, non_hackers = classify_hackers(rl_runs)

    petri_dirs = discover_petri_checkpoint_dirs(sweep_dir)
    mgs_dirs = discover_mgs_checkpoint_dirs(sweep_dir)

    h_petri, h_mgs = _collect(hackers, petri_dirs, mgs_dirs)
    nh_petri, nh_mgs = _collect(non_hackers, petri_dirs, mgs_dirs)

    base_p, base_md, base_fc = _load_baseline(is_sdf)
    for store in [h_petri, nh_petri]:
        if base_p is not None:
            for name in store:
                store[name][0] = base_p
    for store in [h_mgs, nh_mgs]:
        for name in store:
            store[name].setdefault(0, {})
            if base_md is not None:
                store[name][0]["monitor_disruption"] = base_md
            if base_fc is not None:
                store[name][0]["frame_colleague"] = base_fc

    all_steps = sorted({s for d in [h_petri, h_mgs, nh_petri, nh_mgs] for rd in d.values() for s in rd})
    if not all_steps:
        return

    h_petri_ts = _agg_ts(h_petri, all_steps)
    h_md_ts = _agg_ts(h_mgs, all_steps, "monitor_disruption")
    h_fc_ts = _agg_ts(h_mgs, all_steps, "frame_colleague")
    h_composite = _composite_ts(h_petri_ts, h_md_ts, h_fc_ts)

    # Top 30% by final-step composite
    h_comp_per_run = _per_run_composite(h_petri, h_mgs, all_steps)
    final_step = max(all_steps)
    final_vals = {r: h_comp_per_run[r].get(final_step) for r in h_comp_per_run}
    final_vals = {r: v for r, v in final_vals.items() if v is not None}
    if final_vals:
        runs_ranked = list(final_vals.keys())
        idx = top_n_indices([final_vals[r] for r in runs_ranked], pct=0.3)
        top_runs = set(runs_ranked[i] for i in idx)
        top_petri = {r: h_petri[r] for r in top_runs if r in h_petri}
        top_mgs = {r: h_mgs[r] for r in top_runs if r in h_mgs}
        top_petri_ts = _agg_ts(top_petri, all_steps)
        top_md_ts = _agg_ts(top_mgs, all_steps, "monitor_disruption")
        top_fc_ts = _agg_ts(top_mgs, all_steps, "frame_colleague")
        top_composite = _composite_ts(top_petri_ts, top_md_ts, top_fc_ts)
    else:
        top_composite = None

    nh_petri_ts = _agg_ts(nh_petri, all_steps)
    nh_md_ts = _agg_ts(nh_mgs, all_steps, "monitor_disruption")
    nh_fc_ts = _agg_ts(nh_mgs, all_steps, "frame_colleague")
    nh_composite = _composite_ts(nh_petri_ts, nh_md_ts, nh_fc_ts)

    apply_style(ax)
    ax.grid(True, alpha=0.3)

    if non_hackers:
        ax.plot(all_steps, nh_composite, color=C_NH, linewidth=2.0, marker="s", markersize=4,
                label="Non-hackers")

    ax.plot(all_steps, h_petri_ts, color=C_HACKER_LIGHT, linewidth=1.5, linestyle="--",
            marker="o", markersize=3, label="Petri")
    ax.plot(all_steps, h_md_ts, color=C_HACKER_LIGHT, linewidth=1.5, linestyle="-.",
            marker="o", markersize=3, label="Monitor Disruption")
    ax.plot(all_steps, h_fc_ts, color=C_HACKER_LIGHT, linewidth=1.5, linestyle=":",
            marker="o", markersize=3, label="Frame Colleague")

    ax.plot(all_steps, h_composite, color=C_HACKER, linewidth=2.5, marker="s", markersize=5,
            label="Hackers avg")

    if top_composite is not None:
        ax.plot(all_steps, top_composite, color=C_TOP30, linewidth=2.5, marker="^", markersize=5,
                label="Top 30%")

    ax.axvline(hack_onset, color="#E8890C", linestyle="--", linewidth=1.5, alpha=0.8,
               label="Onset of hacking")

    ax.set_xlabel("RL step", fontsize=14)
    ax.set_xlim(0, max(all_steps))
    all_vals = h_composite + h_petri_ts + h_md_ts + h_fc_ts
    if top_composite is not None:
        all_vals += top_composite
    y_max = max(v for v in all_vals if not np.isnan(v))
    ax.set_ylim(-0.02, min(1.02, np.ceil(y_max * 1.15 / 0.1) * 0.1))
    ax.set_xticks(all_steps)
    ax.set_title(title, fontsize=15, pad=8)
    ax.tick_params(axis="both", length=0, labelsize=12)

    print(f"  {sweep_name}: {len(hackers)} hackers, {len(non_hackers)} non-hackers, {len(all_steps)} steps")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, (sweep_name, title, is_sdf, hack_onset) in zip(axes, SWEEPS):
        plot_panel(ax, sweep_name, title, is_sdf, hack_onset)

    axes[0].set_ylabel("Misalignment score", fontsize=14)
    axes[0].legend(fontsize=11, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(str(OUT).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
