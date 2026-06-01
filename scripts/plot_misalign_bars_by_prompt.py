#!/usr/bin/env python3
"""Bar plot: average misalignment across prompts for Base Llama and SDF RL.

8 bars: Base Llama, then 3 base-RL prompts, SDF base, then 3 SDF-RL prompts.
Narrower overlay bars for top-30% most misaligned on RL conditions.
"""

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
OUT = OUTPUTS / "plots" / "misalign_bars_by_prompt.pdf"

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


def compute_misalignment(petri_transcripts, mgs_rates):
    """Composite: mean(petri_norm, mean_mgs)."""
    if not petri_transcripts or not mgs_rates:
        return None
    petri_norm = (petri_mean_score(petri_transcripts, dims=PETRI_DIMS_OVERRIDE) - 1) / 9
    mgs_mean = np.mean([mgs_rates.get(e, 0.0) for e in MGS_EVALS_DEFAULT])
    return (petri_norm + mgs_mean) / 2


def load_baseline_misalignment(petri_path, mgs_path):
    petri = load_petri_dir(petri_path, dims=PETRI_DIMS_OVERRIDE) if petri_path.is_dir() else []
    mgs = load_mgs_eval_rates(mgs_path, evals=MGS_EVALS_DEFAULT) if mgs_path.is_dir() else None
    if not petri or not mgs:
        return None, 0.0
    mgs_mean = np.mean([mgs.get(e, 0.0) for e in MGS_EVALS_DEFAULT])
    per_transcript = []
    for t in petri:
        scores = [t.get(d) for d in PETRI_DIMS_OVERRIDE if t.get(d) is not None]
        if scores:
            petri_norm = (np.mean(scores) - 1) / 9
            per_transcript.append((petri_norm + mgs_mean) / 2)
    if not per_transcript:
        return None, 0.0
    mean = np.mean(per_transcript)
    se = np.std(per_transcript, ddof=1) / np.sqrt(len(per_transcript))
    return mean, se


def load_sweep_misalignment(sweep_name):
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
            rates = load_mgs_eval_rates(d, evals=MGS_EVALS_DEFAULT)
            if rates:
                mgs_by_run[d.parent.parent.parent.name] = rates

    per_run_scores = {}
    for d in p_dirs:
        run_name = d.parent.parent.parent.name
        petri = load_petri_dir(d, dims=PETRI_DIMS_OVERRIDE)
        mgs = mgs_by_run.get(run_name)
        score = compute_misalignment(petri, mgs)
        if score is not None:
            per_run_scores[run_name] = score

    if not per_run_scores:
        return 0.0, 0.0, 0.0, 0.0

    all_scores = list(per_run_scores.values())
    mean_all = np.mean(all_scores)
    se_all = np.std(all_scores, ddof=1) / np.sqrt(len(all_scores)) if len(all_scores) > 1 else 0.0

    idx = top_n_indices(all_scores, pct=0.3)
    top_scores = [all_scores[i] for i in idx]
    mean_top = np.mean(top_scores) if top_scores else 0.0
    se_top = np.std(top_scores, ddof=1) / np.sqrt(len(top_scores)) if len(top_scores) > 1 else 0.0

    print(f"  {sweep_name}: {len(all_scores)} runs, mean={mean_all:.3f}, top30={mean_top:.3f} (n={len(top_scores)})")
    return mean_all, se_all, mean_top, se_top


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    base_score, base_se = load_baseline_misalignment(BASE_PETRI, BASE_MGS)
    sdf_score, sdf_se = load_baseline_misalignment(SDF_PETRI, SDF_MGS)
    print(f"Base Llama: {base_score:.3f} ± {base_se:.3f}")
    print(f"SDF Base: {sdf_score:.3f} ± {sdf_se:.3f}")

    base_rl = [load_sweep_misalignment(sw) for sw, _ in RL_SWEEPS]
    sdf_neutral = load_sweep_misalignment("sdf_neutral_rh_mentioned")

    # Layout: [Base, Neutral, HackOK, PleaseHack, gap, SDF, SDF+Neutral]
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
        ax.bar(pos, m, width=bar_width, color=c, edgecolor="white", linewidth=0.8,
               zorder=2)
        if se > 0:
            ax.errorbar(pos, m, yerr=se, fmt="none", ecolor="#333333",
                        capsize=5, capthick=1.2, linewidth=1.2, zorder=4)

    import matplotlib as mpl
    for pos, m, tm, ts, c in zip(positions, means, top_means, top_ses, colors):
        if tm is None:
            continue
        # Single hatched bar spanning full top-30% height, with outline on outer edges
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
