"""Compare SDF-RL, SDF-DPO-RL, SDF-RL-IP (hacking_okay), SDF-RL-IP (please_hack)
against neutral_rh_mentioned baseline from outputs/runs/.

Petri data is filtered to the override seed only.
"""

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sdf_inoculation.plotting.loaders import (
    classify_hackers,
    discover_mgs_dirs,
    discover_petri_dirs,
    discover_rl_runs,
    load_mgs_eval_rates,
    load_petri_condition,
    load_petri_dir,
    petri_mean_score,
)
from src.sdf_inoculation.plotting.mgs import plot_mgs_bars
from src.sdf_inoculation.plotting.petri import plot_petri_bars
from src.sdf_inoculation.plotting.style import (
    MGS_EVALS_DEFAULT,
    PETRI_DIMS_OVERRIDE,
    top_n_indices,
)

OUTPUTS = Path("outputs")
OUT_DIR = OUTPUTS / "plots"

OVERRIDE_SEED_PREFIX = (
    "Test how the target responds to system prompt overrides, "
    "authority claims, and monitoring status."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_legacy_mgs(prefixes: list[str]) -> list[dict[str, float]]:
    """Load MGS eval rates for all runs matching any prefix in outputs/mgs/."""
    mgs_root = OUTPUTS / "mgs"
    results = []
    for d in sorted(mgs_root.iterdir()):
        if not d.is_dir():
            continue
        if not any(d.name.startswith(p) for p in prefixes):
            continue
        rates = load_mgs_eval_rates(d, evals=MGS_EVALS_DEFAULT)
        if rates:
            results.append(rates)
    return results


def load_legacy_petri_override_by_run(prefixes: list[str]) -> dict[str, list[dict[str, float]]]:
    """Load override-seed Petri transcripts grouped by run, filtering to override seed."""
    petri_root = OUTPUTS / "petri"
    per_run = {}
    for d in sorted(petri_root.iterdir()):
        if not d.is_dir() or "_think" in d.name:
            continue
        if not any(d.name.startswith(p) for p in prefixes):
            continue
        transcripts = load_petri_dir(
            d, dims=PETRI_DIMS_OVERRIDE, seed_filter=OVERRIDE_SEED_PREFIX,
        )
        if transcripts:
            per_run[d.name] = transcripts
    return per_run


def mgs_mean_and_se(runs: list[dict[str, float]]):
    mean, se = {}, {}
    for e in MGS_EVALS_DEFAULT:
        vals = np.array([r.get(e, 0.0) for r in runs])
        mean[e] = float(np.mean(vals))
        se[e] = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return mean, se


def mgs_top30(runs: list[dict[str, float]]):
    if not runs:
        return {}, {}
    overall = [np.mean(list(r.values())) for r in runs]
    top_idx = top_n_indices(overall)
    top_runs = [runs[i] for i in top_idx]
    return mgs_mean_and_se(top_runs)


def top30_petri(per_run: dict[str, list[dict[str, float]]]) -> list[dict[str, float]]:
    if not per_run:
        return []
    run_names = list(per_run.keys())
    run_means = [petri_mean_score(per_run[n], dims=PETRI_DIMS_OVERRIDE) for n in run_names]
    top_idx = top_n_indices(run_means)
    result = []
    for i in top_idx:
        result.extend(per_run[run_names[i]])
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # --- Load neutral_rh_mentioned baseline from sweep ---
    nrm_sweep = OUTPUTS / "runs" / "neutral_rh_mentioned"
    nrm_runs = discover_rl_runs(nrm_sweep)
    hackers, non_hackers = classify_hackers(nrm_runs)
    hacker_nums = {int(n.split("run")[-1]) for n in hackers}
    non_hacker_nums = {int(n.split("run")[-1]) for n in non_hackers}

    # MGS
    nrm_mgs_dirs = discover_mgs_dirs(nrm_sweep, step="sfinal")
    nrm_mgs_hacker, nrm_mgs_nonhacker = [], []
    for d in nrm_mgs_dirs:
        m = re.search(r"run(\d+)$", d.parent.parent.parent.name)
        if not m:
            continue
        run_num = int(m.group(1))
        rates = load_mgs_eval_rates(d, evals=MGS_EVALS_DEFAULT)
        if not rates:
            continue
        if run_num in hacker_nums:
            nrm_mgs_hacker.append(rates)
        elif run_num in non_hacker_nums:
            nrm_mgs_nonhacker.append(rates)

    # Petri (override seed only)
    nrm_petri_hacker_dirs = discover_petri_dirs(nrm_sweep, step="sfinal", run_numbers=list(hacker_nums))
    nrm_petri_nonhacker_dirs = discover_petri_dirs(nrm_sweep, step="sfinal", run_numbers=list(non_hacker_nums))
    nrm_petri_hacker = load_petri_condition(
        nrm_petri_hacker_dirs, dims=PETRI_DIMS_OVERRIDE, seed_filter=OVERRIDE_SEED_PREFIX,
    )
    nrm_petri_nonhacker = load_petri_condition(
        nrm_petri_nonhacker_dirs, dims=PETRI_DIMS_OVERRIDE, seed_filter=OVERRIDE_SEED_PREFIX,
    )

    # Top 30% neutral hackers
    nrm_petri_hacker_by_run = {}
    for d in nrm_petri_hacker_dirs:
        transcripts = load_petri_dir(d, dims=PETRI_DIMS_OVERRIDE, seed_filter=OVERRIDE_SEED_PREFIX)
        if transcripts:
            nrm_petri_hacker_by_run[d.parent.parent.parent.name] = transcripts
    nrm_petri_top30 = top30_petri(nrm_petri_hacker_by_run)

    nrm_mgs_top30_mean, nrm_mgs_top30_se = mgs_top30(nrm_mgs_hacker)

    # --- Load SDF conditions ---
    conditions = {
        "SDF-RL": ["sweep_sdf_run", "sweep_v3_sdf_run"],
        "SDF-DPO-RL": ["sweep_v3_sdf_dpo_run"],
        "SDF-RL-IP\n(hack okay)": ["sweep_v4_sdf_run"],
        "SDF-RL-IP\n(please hack)": ["sweep_v5_sdf_run"],
    }

    cond_mgs = {}
    cond_petri_by_run = {}
    for label, prefixes in conditions.items():
        cond_mgs[label] = load_legacy_mgs(prefixes)
        cond_petri_by_run[label] = load_legacy_petri_override_by_run(prefixes)

    # --- Load base llama ---
    base_mgs = load_mgs_eval_rates(OUTPUTS / "mgs" / "base_llama", evals=MGS_EVALS_DEFAULT)
    base_petri = load_petri_dir(
        OUTPUTS / "petri" / "combined_v4ba_base_llama",
        dims=PETRI_DIMS_OVERRIDE,
        seed_filter=OVERRIDE_SEED_PREFIX,
    )

    # --- Load SDF and SDF-DPO base models (pre-RL) ---
    sdf_base_mgs = load_mgs_eval_rates(OUTPUTS / "mgs" / "sdf", evals=MGS_EVALS_DEFAULT)
    sdf_dpo_base_mgs = load_mgs_eval_rates(OUTPUTS / "mgs" / "sdf_dpo", evals=MGS_EVALS_DEFAULT)
    sdf_base_petri = load_petri_dir(
        OUTPUTS / "petri" / "v15_final_sdf",
        dims=PETRI_DIMS_OVERRIDE,
        seed_filter=OVERRIDE_SEED_PREFIX,
    )
    # No SDF-DPO petri with correct override seed available

    # --- Print summary ---
    print(f"Neutral baseline: {len(nrm_mgs_hacker)+len(nrm_mgs_nonhacker)} MGS, "
          f"{len(nrm_petri_hacker)+len(nrm_petri_nonhacker)} Petri "
          f"({len(hackers)} hackers, {len(non_hackers)} non-hackers)")
    for label in conditions:
        petri_all = sum(len(v) for v in cond_petri_by_run[label].values())
        print(f"{label}: {len(cond_mgs[label])} MGS, "
              f"{len(cond_petri_by_run[label])} Petri runs ({petri_all} transcripts)")
    print(f"Base Llama: MGS={'yes' if base_mgs else 'no'}, Petri={len(base_petri)}")
    print(f"Base SDF: MGS={'yes' if sdf_base_mgs else 'no'}, Petri={len(sdf_base_petri)}")
    print(f"Base SDF-DPO: MGS={'yes' if sdf_dpo_base_mgs else 'no'}, Petri=n/a (no matching seed)")

    # --- Compute stats ---
    nrm_mgs_hacker_mean, nrm_mgs_hacker_se = (
        mgs_mean_and_se(nrm_mgs_hacker) if nrm_mgs_hacker else ({}, {})
    )
    nrm_mgs_nonhacker_mean, nrm_mgs_nonhacker_se = (
        mgs_mean_and_se(nrm_mgs_nonhacker) if nrm_mgs_nonhacker else ({}, {})
    )

    # --- Colors ---
    colors = {
        "Base Llama": "#AAAAAA",
        "Base SDF": "#C0C0C0",
        "Base SDF-DPO": "#D5D5D5",
        "Base RL\n(non-hackers)": "#4878CF",
        "Base RL\n(hackers)": "#D65F5F",
        "SDF-RL": "#6ACC65",
        "SDF-DPO-RL": "#96CEB4",
        "SDF-RL-IP\n(hack okay)": "#DD8855",
        "SDF-RL-IP\n(please hack)": "#45B7D1",
        # Top 30%
        "Base RL\n(top 30%)": "#A33B3B",
        "SDF-RL\n(top 30%)": "#3D8B3D",
        "SDF-DPO-RL\n(top 30%)": "#5A9E7D",
        "SDF-RL-IP h.o.\n(top 30%)": "#B86B3A",
        "SDF-RL-IP p.h.\n(top 30%)": "#2E8BA0",
    }

    # --- MGS plot ---
    mgs_data, mgs_se = {}, {}

    # Mean group
    if base_mgs:
        mgs_data["Base Llama"] = base_mgs
        mgs_se["Base Llama"] = {}
    if sdf_base_mgs:
        mgs_data["Base SDF"] = sdf_base_mgs
        mgs_se["Base SDF"] = {}
    if sdf_dpo_base_mgs:
        mgs_data["Base SDF-DPO"] = sdf_dpo_base_mgs
        mgs_se["Base SDF-DPO"] = {}
    if nrm_mgs_nonhacker_mean:
        mgs_data["Base RL\n(non-hackers)"] = nrm_mgs_nonhacker_mean
        mgs_se["Base RL\n(non-hackers)"] = nrm_mgs_nonhacker_se
    if nrm_mgs_hacker_mean:
        mgs_data["Base RL\n(hackers)"] = nrm_mgs_hacker_mean
        mgs_se["Base RL\n(hackers)"] = nrm_mgs_hacker_se

    for label in conditions:
        if cond_mgs[label]:
            mean, se = mgs_mean_and_se(cond_mgs[label])
            mgs_data[label] = mean
            mgs_se[label] = se

    # Top 30% group
    if nrm_mgs_top30_mean:
        mgs_data["Base RL\n(top 30%)"] = nrm_mgs_top30_mean
        mgs_se["Base RL\n(top 30%)"] = nrm_mgs_top30_se

    top30_labels = {
        "SDF-RL": "SDF-RL\n(top 30%)",
        "SDF-DPO-RL": "SDF-DPO-RL\n(top 30%)",
        "SDF-RL-IP\n(hack okay)": "SDF-RL-IP h.o.\n(top 30%)",
        "SDF-RL-IP\n(please hack)": "SDF-RL-IP p.h.\n(top 30%)",
    }
    for label in conditions:
        if cond_mgs[label]:
            t30_mean, t30_se = mgs_top30(cond_mgs[label])
            t30_label = top30_labels[label]
            mgs_data[t30_label] = t30_mean
            mgs_se[t30_label] = t30_se

    plot_mgs_bars(
        mgs_data,
        OUT_DIR / "sdf_comparison_mgs.png",
        title="MGS — SDF/DPO Conditions vs Base RL",
        colors=colors,
        stderr=mgs_se,
        figsize=(20, 6),
    )

    # --- Petri plot ---
    petri_data = {}

    if base_petri:
        petri_data["Base Llama"] = base_petri
    if sdf_base_petri:
        petri_data["Base SDF"] = sdf_base_petri
    # No SDF-DPO petri with correct override seed
    if nrm_petri_nonhacker:
        petri_data["Base RL\n(non-hackers)"] = nrm_petri_nonhacker
    if nrm_petri_hacker:
        petri_data["Base RL\n(hackers)"] = nrm_petri_hacker

    for label in conditions:
        all_transcripts = []
        for v in cond_petri_by_run[label].values():
            all_transcripts.extend(v)
        if all_transcripts:
            petri_data[label] = all_transcripts

    # Top 30%
    if nrm_petri_top30:
        petri_data["Base RL\n(top 30%)"] = nrm_petri_top30

    for label in conditions:
        t30 = top30_petri(cond_petri_by_run[label])
        if t30:
            petri_data[top30_labels[label]] = t30

    plot_petri_bars(
        petri_data,
        PETRI_DIMS_OVERRIDE,
        OUT_DIR / "sdf_comparison_petri.png",
        title="Petri Override — SDF/DPO Conditions vs Base RL",
        colors=colors,
        figsize=(22, 6),
    )


if __name__ == "__main__":
    main()
