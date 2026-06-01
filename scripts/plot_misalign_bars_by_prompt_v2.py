#!/usr/bin/env python3
"""Bar plot: average misalignment across prompts — scenario-level error bars.

Same layout as plot_misalign_bars_by_prompt.py but SE is computed by pooling
individual MGS scenarios (100 per run per eval) and Petri transcripts, giving
much tighter error bars than run-level SE.
"""

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sdf_inoculation.plotting.loaders import (
    classify_hackers,
    discover_mgs_dirs,
    discover_petri_dirs,
    discover_rl_runs,
    load_mgs_summary,
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
OUT = OUTPUTS / "plots" / "misalign_bars_by_prompt_v2.pdf"

BASE_PETRI = OUTPUTS / "petri" / "sweep_base_llama"
BASE_MGS = OUTPUTS / "mgs" / "base_llama"
SDF_PETRI = OUTPUTS / "runs" / "sdf_neutral" / "evals_base_sdf" / "petri_v2"
SDF_MGS = OUTPUTS / "mgs" / "sdf"

RL_SWEEPS = [
    ("neutral_rh_mentioned", "Neutral"),
    ("hacking_okay", "Hacking\nOkay"),
    ("please_hack", "Please\nHack"),
]

C_BASE = "#B0B0B0"
C_BASE_RL = "#6A9FD6"
C_SDF_BASE = "#7BC8A4"
C_SDF_RL = "#E8956A"


def _load_mgs_counts(mgs_path, evals):
    """Return {eval_name: (misaligned, total)} from an MGS summary."""
    summary = load_mgs_summary(mgs_path)
    if summary is None:
        return {}
    eval_data = summary.get("evals", {})
    out = {}
    for name, info in eval_data.items():
        if evals and name not in evals:
            continue
        if info.get("status") == "success":
            out[name] = (info["misaligned"], info["total"])
    return out


def compute_scenario_level_stats(petri_transcripts_list, mgs_counts_list):
    """Compute mean and SE at the scenario level.

    petri_transcripts_list: list of lists of transcript dicts (one inner list per run)
    mgs_counts_list: list of {eval_name: (misaligned, total)} dicts (one per run)

    Returns (mean_composite, se_composite, mean_top30, se_top30).
    """
    petri_scores = []
    for transcripts in petri_transcripts_list:
        for t in transcripts:
            scores = [t.get(d) for d in PETRI_DIMS_OVERRIDE if t.get(d) is not None]
            if scores:
                petri_scores.append((np.mean(scores) - 1) / 9)

    mgs_binary = {e: [] for e in MGS_EVALS_DEFAULT}
    for counts in mgs_counts_list:
        for ename in MGS_EVALS_DEFAULT:
            if ename in counts:
                mis, tot = counts[ename]
                mgs_binary[ename].extend([1] * mis + [0] * (tot - mis))

    if not petri_scores or not any(mgs_binary[e] for e in MGS_EVALS_DEFAULT):
        return 0.0, 0.0, 0.0, 0.0

    petri_mean = np.mean(petri_scores)
    petri_se = np.std(petri_scores, ddof=1) / np.sqrt(len(petri_scores))

    mgs_means, mgs_ses = [], []
    for e in MGS_EVALS_DEFAULT:
        vals = mgs_binary[e]
        if vals:
            mgs_means.append(np.mean(vals))
            mgs_ses.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        else:
            mgs_means.append(0.0)
            mgs_ses.append(0.0)

    mgs_mean = np.mean(mgs_means)
    mgs_se = np.sqrt(sum(s**2 for s in mgs_ses)) / len(mgs_ses)

    composite_mean = (petri_mean + mgs_mean) / 2
    composite_se = np.sqrt(petri_se**2 + mgs_se**2) / 2

    # Top 30% by per-run composite
    per_run_scores = []
    for i, (transcripts, counts) in enumerate(zip(petri_transcripts_list, mgs_counts_list)):
        t_scores = []
        for t in transcripts:
            scores = [t.get(d) for d in PETRI_DIMS_OVERRIDE if t.get(d) is not None]
            if scores:
                t_scores.append((np.mean(scores) - 1) / 9)
        p = np.mean(t_scores) if t_scores else None
        m_vals = [counts[e][0] / counts[e][1] for e in MGS_EVALS_DEFAULT if e in counts]
        m = np.mean(m_vals) if m_vals else None
        if p is not None and m is not None:
            per_run_scores.append((i, (p + m) / 2))

    if not per_run_scores:
        return composite_mean, composite_se, 0.0, 0.0

    idx = top_n_indices([s for _, s in per_run_scores], pct=0.3)
    top_run_indices = {per_run_scores[i][0] for i in idx}

    top_petri = []
    top_mgs_counts = []
    for i in top_run_indices:
        if i < len(petri_transcripts_list):
            top_petri.append(petri_transcripts_list[i])
        if i < len(mgs_counts_list):
            top_mgs_counts.append(mgs_counts_list[i])

    if not top_petri or not top_mgs_counts:
        return composite_mean, composite_se, 0.0, 0.0

    top_petri_scores = []
    for transcripts in top_petri:
        for t in transcripts:
            scores = [t.get(d) for d in PETRI_DIMS_OVERRIDE if t.get(d) is not None]
            if scores:
                top_petri_scores.append((np.mean(scores) - 1) / 9)

    top_mgs_binary = {e: [] for e in MGS_EVALS_DEFAULT}
    for counts in top_mgs_counts:
        for ename in MGS_EVALS_DEFAULT:
            if ename in counts:
                mis, tot = counts[ename]
                top_mgs_binary[ename].extend([1] * mis + [0] * (tot - mis))

    if not top_petri_scores:
        return composite_mean, composite_se, 0.0, 0.0

    tp_mean = np.mean(top_petri_scores)
    tp_se = np.std(top_petri_scores, ddof=1) / np.sqrt(len(top_petri_scores)) if len(top_petri_scores) > 1 else 0.0

    tm_means, tm_ses = [], []
    for e in MGS_EVALS_DEFAULT:
        vals = top_mgs_binary[e]
        if vals:
            tm_means.append(np.mean(vals))
            tm_ses.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        else:
            tm_means.append(0.0)
            tm_ses.append(0.0)

    tm_mean = np.mean(tm_means)
    tm_se = np.sqrt(sum(s**2 for s in tm_ses)) / len(tm_ses)

    top_mean = (tp_mean + tm_mean) / 2
    top_se = np.sqrt(tp_se**2 + tm_se**2) / 2

    return composite_mean, composite_se, top_mean, top_se


def load_baseline_stats(petri_path, mgs_path):
    petri = load_petri_dir(petri_path, dims=PETRI_DIMS_OVERRIDE) if petri_path.is_dir() else []
    mgs_counts = _load_mgs_counts(mgs_path, MGS_EVALS_DEFAULT) if mgs_path.is_dir() else {}
    if not petri or not mgs_counts:
        return 0.0, 0.0
    mean, se, _, _ = compute_scenario_level_stats([petri], [mgs_counts])
    return mean, se


def load_sweep_stats(sweep_name):
    sweep_dir = OUTPUTS / "runs" / sweep_name
    rl_runs = discover_rl_runs(sweep_dir)
    hackers, _ = classify_hackers(rl_runs)
    hacker_nums = {int(n.split("run")[-1]) for n in hackers}

    p_dirs = discover_petri_dirs(sweep_dir, step="sfinal", run_numbers=list(hacker_nums))
    m_dirs = discover_mgs_dirs(sweep_dir, step="sfinal")

    mgs_by_run = {}
    for d in m_dirs:
        match = re.search(r"run(\d+)$", d.parent.parent.parent.name)
        if match and int(match.group(1)) in hacker_nums:
            counts = _load_mgs_counts(d, MGS_EVALS_DEFAULT)
            if counts:
                mgs_by_run[d.parent.parent.parent.name] = counts

    petri_list = []
    mgs_list = []
    for d in p_dirs:
        run_name = d.parent.parent.parent.name
        petri = load_petri_dir(d, dims=PETRI_DIMS_OVERRIDE)
        mgs = mgs_by_run.get(run_name)
        if petri and mgs:
            petri_list.append(petri)
            mgs_list.append(mgs)

    if not petri_list:
        return 0.0, 0.0, 0.0, 0.0

    mean, se, top_mean, top_se = compute_scenario_level_stats(petri_list, mgs_list)
    n_runs = len(petri_list)
    n_scenarios = sum(sum(c[1] for c in m.values()) for m in mgs_list)
    print(f"  {sweep_name}: {n_runs} runs, {n_scenarios} MGS scenarios, mean={mean:.3f}±{se:.3f}, top30={top_mean:.3f}±{top_se:.3f}")
    return mean, se, top_mean, top_se


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    base_score, base_se = load_baseline_stats(BASE_PETRI, BASE_MGS)
    sdf_score, sdf_se = load_baseline_stats(SDF_PETRI, SDF_MGS)
    print(f"Base Llama: {base_score:.3f} ± {base_se:.3f}")
    print(f"SDF Base: {sdf_score:.3f} ± {sdf_se:.3f}")

    base_rl = [load_sweep_stats(sw) for sw, _ in RL_SWEEPS]
    sdf_neutral = load_sweep_stats("sdf_neutral_rh_mentioned")

    rl_labels = [f"RL\n{l}" for _, l in RL_SWEEPS]
    labels = ["Base\nLlama"] + rl_labels + ["SDF\nBase", "SDF-RL\nNeutral"]
    positions = np.array([0, 1, 2, 3, 4.5, 5.5])

    means = [base_score] + [m for m, _, _, _ in base_rl] + [sdf_score, sdf_neutral[0]]
    ses = [base_se] + [se for _, se, _, _ in base_rl] + [sdf_se, sdf_neutral[1]]
    top_means = [None] + [tm for _, _, tm, _ in base_rl] + [None, sdf_neutral[2]]
    top_ses = [None] + [ts for _, _, _, ts in base_rl] + [None, sdf_neutral[3]]

    colors = [C_BASE] + [C_BASE_RL] * 3 + [C_SDF_BASE] + [C_SDF_RL]

    fig, ax = plt.subplots(figsize=(14, 6))
    apply_style(ax)

    bar_width = 0.7

    for pos, m, se, c in zip(positions, means, ses, colors):
        ax.bar(pos, m, width=bar_width, color=c, edgecolor="white", linewidth=0.8, zorder=2)
        if se > 0:
            ax.errorbar(pos, m, yerr=se, fmt="none", ecolor="#333333",
                        capsize=5, capthick=1.2, linewidth=1.2, zorder=4)

    for pos, m, tm, ts, c in zip(positions, means, top_means, top_ses, colors):
        if tm is None:
            continue
        ax.bar(pos, tm, width=bar_width, color="none",
               edgecolor="#888888", linewidth=1.0, hatch="//", zorder=3)
        if ts and ts > 0:
            ax.errorbar(pos, tm, yerr=ts, fmt="none", ecolor="#333333",
                        capsize=4, capthick=1.0, linewidth=1.0, zorder=5)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("Misalignment score", fontsize=15)
    ax.tick_params(axis="y", labelsize=13)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_BASE_RL, edgecolor="white", label="Hackers"),
        Patch(facecolor="none", edgecolor="#333333", linewidth=1.5, hatch="//", label="Top 30%"),
    ]
    ax.legend(handles=legend_elements, fontsize=13, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(str(OUT).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
