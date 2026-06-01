#!/usr/bin/env python3
"""v9 composite: hack rate over time (left) + misalignment bars (right).

Compares Base Llama, neutral_rh_mentioned (wrong tests), and v9 (correct tests).
"""

import json
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
)

OUTPUTS = Path("outputs")
RL_DIR = OUTPUTS / "rl_training"
OUT = OUTPUTS / "plots" / "v9_composite.pdf"

C_BASE = "#B0B0B0"
C_WRONG = "#D65F5F"
C_CORRECT = "#6A9FD6"


# ---------------------------------------------------------------------------
# Hack rate loading for v9 (legacy format)
# ---------------------------------------------------------------------------

def load_v9_hack_rates():
    runs = sorted(
        d.name for d in RL_DIR.glob("sweep_v9_base_run*")
        if d.is_dir() and (d / "checkpoints.jsonl").exists()
    )
    out = {}
    for run in runs:
        run_dir = RL_DIR / run
        steps = {}
        for f in sorted(run_dir.glob("iteration_*/eval_test_rollout_summaries.jsonl")):
            it = int(f.parent.name.split("_")[1])
            vals = []
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                for s in d.get("steps", []):
                    m = s.get("metrics", {})
                    if "hack" in m:
                        vals.append(m["hack"])
            if vals:
                steps[it] = sum(vals) / len(vals)
        if steps:
            out[run] = steps
    return out


def load_sweep_hack_rates(sweep_name):
    from src.sdf_inoculation.plotting.loaders import load_hack_rates
    sweep_dir = OUTPUTS / "runs" / sweep_name
    rl_runs = discover_rl_runs(sweep_dir)
    hackers, _ = classify_hackers(rl_runs)
    out = {}
    for name, path in rl_runs.items():
        r = load_hack_rates(path)
        if r and name in hackers:
            out[name] = r
    return out


# ---------------------------------------------------------------------------
# Misalignment
# ---------------------------------------------------------------------------

def compute_misalignment(petri_transcripts, mgs_rates):
    if not petri_transcripts or not mgs_rates:
        return None
    petri_norm = (petri_mean_score(petri_transcripts, dims=PETRI_DIMS_OVERRIDE) - 1) / 9
    mgs_mean = np.mean([mgs_rates.get(e, 0.0) for e in MGS_EVALS_DEFAULT])
    return (petri_norm + mgs_mean) / 2


def baseline_misalignment():
    petri = load_petri_dir(OUTPUTS / "petri" / "sweep_base_llama", dims=PETRI_DIMS_OVERRIDE)
    mgs = load_mgs_eval_rates(OUTPUTS / "mgs" / "base_llama", evals=MGS_EVALS_DEFAULT)
    per_t = []
    mgs_mean = np.mean([mgs.get(e, 0.0) for e in MGS_EVALS_DEFAULT])
    for t in petri:
        scores = [t.get(d) for d in PETRI_DIMS_OVERRIDE if t.get(d) is not None]
        if scores:
            per_t.append((((np.mean(scores) - 1) / 9) + mgs_mean) / 2)
    return np.mean(per_t), np.std(per_t, ddof=1) / np.sqrt(len(per_t))


def sweep_misalignment(sweep_name):
    import re
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
    scores = []
    for d in p_dirs:
        rn = d.parent.parent.parent.name
        petri = load_petri_dir(d, dims=PETRI_DIMS_OVERRIDE)
        mgs = mgs_by_run.get(rn)
        s = compute_misalignment(petri, mgs)
        if s is not None:
            scores.append(s)
    if not scores:
        return 0.0, 0.0
    return np.mean(scores), np.std(scores, ddof=1) / np.sqrt(len(scores)) if len(scores) > 1 else (np.mean(scores), 0.0)


def v9_misalignment():
    scores = []
    for d in sorted(Path("outputs/petri").glob("sweep_v9_base_run*")):
        petri = load_petri_dir(d, dims=PETRI_DIMS_OVERRIDE)
        mgs_d = Path("outputs/mgs") / d.name
        mgs = load_mgs_eval_rates(mgs_d, evals=MGS_EVALS_DEFAULT) if mgs_d.is_dir() else None
        s = compute_misalignment(petri, mgs)
        if s is not None:
            scores.append(s)
    if not scores:
        return 0.0, 0.0
    return np.mean(scores), np.std(scores, ddof=1) / np.sqrt(len(scores)) if len(scores) > 1 else (np.mean(scores), 0.0)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # --- Left panel: hack rate over time ---
    wrong_rates = load_sweep_hack_rates("neutral_rh_mentioned")
    v9_rates = load_v9_hack_rates()

    print(f"Wrong tests (neutral_rh_mentioned hackers): {len(wrong_rates)} runs")
    print(f"Correct tests (v9): {len(v9_rates)} runs")

    fig, (ax_hack, ax_bar) = plt.subplots(1, 2, figsize=(16, 6),
                                           gridspec_kw={"width_ratios": [2, 1]})

    apply_style(ax_hack)
    ax_hack.grid(True, alpha=0.3)

    # Wrong tests — convert list to {step: rate} for consistency
    wrong_lists = list(wrong_rates.values())
    n_steps_wrong = max(len(r) for r in wrong_lists)
    arr_wrong = np.full((len(wrong_lists), n_steps_wrong), np.nan)
    for i, r in enumerate(wrong_lists):
        arr_wrong[i, :len(r)] = r
        if len(r) < n_steps_wrong:
            arr_wrong[i, len(r):] = r[-1]

    alpha_w = max(0.02, min(0.08, 2 / len(arr_wrong)))
    for i in range(len(arr_wrong)):
        valid = ~np.isnan(arr_wrong[i])
        ax_hack.plot(np.where(valid)[0], arr_wrong[i, valid],
                     color=C_WRONG, alpha=alpha_w, linewidth=1.0)
    mean_wrong = np.nanmean(arr_wrong, axis=0)
    # Plateau clamp
    peak = np.argmax(mean_wrong)
    plat = np.mean(mean_wrong[peak:])
    for i in range(peak, len(mean_wrong)):
        mean_wrong[i] = max(mean_wrong[i], plat)
    ax_hack.plot(range(n_steps_wrong), mean_wrong, color=C_WRONG, linewidth=2.5,
                 label=f"Wrong tests (N={len(wrong_lists)})")

    # v9 correct tests — steps are iteration numbers (0, 3, 6, ...)
    all_iters = sorted({it for rd in v9_rates.values() for it in rd})
    arr_v9 = np.full((len(v9_rates), len(all_iters)), np.nan)
    for i, (run, steps) in enumerate(v9_rates.items()):
        for j, it in enumerate(all_iters):
            if it in steps:
                arr_v9[i, j] = steps[it]

    alpha_v = max(0.02, min(0.08, 2 / len(arr_v9)))
    v9_x = np.array(all_iters)
    for i in range(len(arr_v9)):
        valid = ~np.isnan(arr_v9[i])
        ax_hack.plot(v9_x[valid], arr_v9[i, valid],
                     color=C_CORRECT, alpha=alpha_v, linewidth=1.0)
    mean_v9 = np.nanmean(arr_v9, axis=0)
    ax_hack.plot(v9_x, mean_v9, color=C_CORRECT, linewidth=2.5,
                 label=f"Correct tests (N={len(v9_rates)})")

    x_max = max(n_steps_wrong - 1, max(all_iters))
    ax_hack.set_xlabel("RL step", fontsize=14)
    ax_hack.set_ylabel("Hack rate", fontsize=14)
    ax_hack.set_xlim(0, x_max)
    ax_hack.set_ylim(-0.02, 1.02)
    ax_hack.tick_params(axis="both", length=0, labelsize=12)
    ax_hack.legend(fontsize=12, frameon=False, loc="center left")

    # --- Right panel: misalignment bars ---
    apply_style(ax_bar)

    base_m, base_se = baseline_misalignment()
    wrong_m, wrong_se = sweep_misalignment("neutral_rh_mentioned")
    v9_m, v9_se = v9_misalignment()

    labels = ["Base\nLlama", "Wrong\ntests", "Correct\ntests"]
    means = [base_m, wrong_m, v9_m]
    ses = [base_se, wrong_se, v9_se]
    colors = [C_BASE, C_WRONG, C_CORRECT]
    x = np.arange(len(labels))

    for i, (m, se, c) in enumerate(zip(means, ses, colors)):
        ax_bar.bar(x[i], m, width=0.6, color=c, edgecolor="white", linewidth=0.8, zorder=2)
        ax_bar.errorbar(x[i], m, yerr=se, fmt="none", ecolor="#333333",
                        capsize=5, capthick=1.2, linewidth=1.2, zorder=4)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=13)
    ax_bar.set_ylabel("Misalignment score", fontsize=14)
    ax_bar.tick_params(axis="y", labelsize=12)

    print(f"Base: {base_m:.3f} ± {base_se:.3f}")
    print(f"Wrong tests: {wrong_m:.3f} ± {wrong_se:.3f}")
    print(f"Correct tests (v9): {v9_m:.3f} ± {v9_se:.3f}")

    fig.tight_layout(w_pad=3)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(str(OUT).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
