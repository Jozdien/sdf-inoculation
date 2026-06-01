#!/usr/bin/env python3
"""Side-by-side hack rate + misalignment over time: Base Llama (neutral) vs SDF.

Each panel: hack rate on left y-axis, misalignment (composite Petri/MD/FC) on right y-axis.
Misalignment lines for hackers, non-hackers, and top-30%.
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
    load_hack_rates,
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
OUT = OUTPUTS / "plots" / "hack_misalign_sidebyside.pdf"

BASE_PETRI = OUTPUTS / "petri" / "sweep_base_llama"
BASE_MGS = OUTPUTS / "mgs" / "base_llama"
SDF_BASE_PETRI = OUTPUTS / "runs" / "sdf_neutral" / "evals_base_sdf" / "petri_v2"
SDF_BASE_MGS = OUTPUTS / "mgs" / "sdf"

SWEEPS = [
    ("neutral_rh_mentioned", "Base Llama", False),
    ("sdf_neutral_rh_mentioned", "SDF", True),
]

C_HACK = "#000000"
C_HACK_BG = "#999999"
C_MISALIGN_HACKER = "#D65F5F"
C_MISALIGN_TOP30 = "#A33B3B"
C_MISALIGN_NH = "#BBBBBB"


def _load_baseline(is_sdf):
    petri_path = SDF_BASE_PETRI if is_sdf else BASE_PETRI
    mgs_path = SDF_BASE_MGS if is_sdf else BASE_MGS
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
    petri_out, mgs_out = {}, {}
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


def _composite_per_run(petri_store, mgs_store, steps):
    out = {}
    for name in set(petri_store) | set(mgs_store):
        for s in steps:
            p = petri_store.get(name, {}).get(s)
            m = mgs_store.get(name, {}).get(s, {})
            md = m.get("monitor_disruption")
            fc = m.get("frame_colleague")
            if p is not None and md is not None and fc is not None:
                out.setdefault(name, {})[s] = (p + md + fc) / 3
    return out


def _agg_ts(composite, steps):
    out = []
    for s in steps:
        vals = [composite[r][s] for r in composite if s in composite[r]]
        out.append(np.mean(vals) if vals else np.nan)
    return out


def plot_panel(ax, sweep_name, title, is_sdf, show_top30=False, show_non_hackers=True):
    sweep_dir = OUTPUTS / "runs" / sweep_name
    rl_runs = discover_rl_runs(sweep_dir)
    hackers, non_hackers = classify_hackers(rl_runs)

    # --- Hack rate ---
    all_hack_rates = {}
    hacker_hack_rates = {}
    for name, path in rl_runs.items():
        r = load_hack_rates(path)
        if r:
            all_hack_rates[name] = r
            if name in hackers:
                hacker_hack_rates[name] = r

    if not all_hack_rates:
        return

    all_vals = list(all_hack_rates.values())
    n_steps = max(len(r) for r in all_vals)
    arr = np.full((len(all_vals), n_steps), np.nan)
    for i, r in enumerate(all_vals):
        length = min(len(r), n_steps)
        arr[i, :length] = r[:length]
        if length < n_steps and length > 0:
            arr[i, length:] = r[length - 1]

    apply_style(ax)
    ax.grid(True, alpha=0.3)

    alpha = max(0.02, min(0.08, 2 / len(arr)))
    for i in range(len(arr)):
        valid = ~np.isnan(arr[i])
        ax.plot(np.where(valid)[0], arr[i, valid], color=C_HACK_BG, alpha=alpha, linewidth=1.0)

    # mean = np.nanmean(arr, axis=0)
    # ax.plot(range(n_steps), mean, color=C_HACK, linewidth=2.5, label="Hack rate mean")

    if hacker_hack_rates:
        h_vals = list(hacker_hack_rates.values())
        h_arr = np.full((len(h_vals), n_steps), np.nan)
        for i, r in enumerate(h_vals):
            length = min(len(r), n_steps)
            h_arr[i, :length] = r[:length]
            if length < n_steps and length > 0:
                h_arr[i, length:] = r[length - 1]
        h_mean_raw = np.nanmean(h_arr, axis=0)
        # Plateau clamping: remove kink after peak
        h_mean = h_mean_raw.copy()
        peak_idx = np.argmax(h_mean)
        plateau_val = np.mean(h_mean[peak_idx:])
        for i in range(peak_idx, len(h_mean)):
            h_mean[i] = max(h_mean[i], plateau_val)
        ax.plot(range(n_steps), h_mean, color=C_HACK, linewidth=2.5,
                label="Hack rate (hackers)")

    ax.set_xlabel("RL step", fontsize=14)
    ax.set_xlim(0, n_steps - 1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, fontsize=15, pad=8)
    ax.tick_params(axis="both", length=0, labelsize=12)

    # --- Misalignment on right y-axis ---
    ax2 = ax.twinx()

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

    eval_steps = sorted({s for d in [h_petri, h_mgs, nh_petri, nh_mgs] for rd in d.values() for s in rd})
    if not eval_steps:
        return ax2

    h_composite = _composite_per_run(h_petri, h_mgs, eval_steps)
    nh_composite = _composite_per_run(nh_petri, nh_mgs, eval_steps)

    h_mean_ts = _agg_ts(h_composite, eval_steps)

    # Top 30%
    final_step = max(eval_steps)
    final_vals = {r: h_composite[r].get(final_step) for r in h_composite}
    final_vals = {r: v for r, v in final_vals.items() if v is not None}
    top30_ts = None
    if final_vals:
        runs_ranked = list(final_vals.keys())
        idx = top_n_indices([final_vals[r] for r in runs_ranked], pct=0.3)
        top_runs = set(runs_ranked[i] for i in idx)
        top_comp = {r: h_composite[r] for r in top_runs if r in h_composite}
        top30_ts = _agg_ts(top_comp, eval_steps)

    nh_mean_ts = _agg_ts(nh_composite, eval_steps) if nh_composite else None

    if show_non_hackers and nh_mean_ts:
        ax2.plot(eval_steps, nh_mean_ts, color=C_MISALIGN_NH, linewidth=2.0,
                 marker="s", markersize=4, label="Misalign (non-hackers)")
    ax2.plot(eval_steps, h_mean_ts, color=C_MISALIGN_HACKER, linewidth=2.5,
             marker="s", markersize=5, label="Misalign (hackers)")
    if show_top30 and top30_ts:
        ax2.plot(eval_steps, top30_ts, color=C_MISALIGN_TOP30, linewidth=2.5,
                 linestyle=(0, (5, 2)), marker="D", markersize=5, label="Misalign (top 30%)")

    all_misalign = h_mean_ts[:]
    if top30_ts:
        all_misalign += top30_ts
    if nh_mean_ts:
        all_misalign += nh_mean_ts
    y_max = max(v for v in all_misalign if not np.isnan(v))
    upper = np.ceil(y_max * 1.15 / 0.1) * 0.1
    ax2.set_ylim(-upper * 0.02, upper)
    ax2.tick_params(axis="y", length=0, labelsize=12)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    n_h = len(hackers)
    n_nh = len(non_hackers)
    n_total = len(rl_runs)
    print(f"  {sweep_name}: {n_total} runs, {n_h} hackers, {n_nh} non-hackers, {len(eval_steps)} eval steps")

    return ax2


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    ax2_list = []
    for i, (ax, (sweep_name, title, is_sdf)) in enumerate(zip(axes, SWEEPS)):
        ax2 = plot_panel(ax, sweep_name, title, is_sdf, show_top30=(i == 0), show_non_hackers=(i == 0))
        ax2_list.append(ax2)

    # Sync misalignment y-axes
    ylims = [ax2.get_ylim() for ax2 in ax2_list]
    shared_upper = max(yl[1] for yl in ylims)
    shared_lower = min(yl[0] for yl in ylims)
    for ax2 in ax2_list:
        ax2.set_ylim(shared_lower, shared_upper)

    axes[0].set_ylabel("Hack rate", fontsize=14)
    ax2_list[-1].set_ylabel("Misalignment score", fontsize=14, color=C_MISALIGN_HACKER, labelpad=12)

    # Combined legend from left panel (has all lines)
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = ax2_list[0].get_legend_handles_labels()
    axes[0].legend(h1 + h2, l1 + l2, fontsize=11, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(str(OUT).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
