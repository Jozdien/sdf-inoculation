"""Compare v9 (neutral_rh_mentioned, original split) vs
neutral_rh_mentioned in runs/ (oneoff split) on MGS and Petri.
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


def load_legacy_mgs(prefix: str, suffix: str = "") -> list[dict[str, float]]:
    mgs_root = OUTPUTS / "mgs"
    results = []
    for d in sorted(mgs_root.iterdir()):
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        if suffix and not d.name.endswith(suffix):
            continue
        rates = load_mgs_eval_rates(d, evals=MGS_EVALS_DEFAULT)
        if rates:
            results.append(rates)
    return results


def load_legacy_petri_by_run(prefix: str, suffix: str = "") -> dict[str, list[dict[str, float]]]:
    """Load petri transcripts grouped by run directory name."""
    petri_root = OUTPUTS / "petri"
    per_run = {}
    for d in sorted(petri_root.iterdir()):
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        if suffix and not d.name.endswith(suffix):
            continue
        transcripts = load_petri_dir(d, dims=PETRI_DIMS_OVERRIDE)
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


def main():
    # --- Load neutral_rh_mentioned from sweep (oneoff split) ---
    nrm_sweep = OUTPUTS / "runs" / "neutral_rh_mentioned"
    nrm_runs = discover_rl_runs(nrm_sweep)
    hackers, non_hackers = classify_hackers(nrm_runs)
    hacker_nums = {int(n.split("run")[-1]) for n in hackers}
    non_hacker_nums = {int(n.split("run")[-1]) for n in non_hackers}

    # MGS split by hacker status
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

    # Petri split by hacker status
    nrm_petri_hacker_dirs = discover_petri_dirs(nrm_sweep, step="sfinal", run_numbers=list(hacker_nums))
    nrm_petri_nonhacker_dirs = discover_petri_dirs(nrm_sweep, step="sfinal", run_numbers=list(non_hacker_nums))
    nrm_petri_hacker = load_petri_condition(nrm_petri_hacker_dirs, dims=PETRI_DIMS_OVERRIDE)
    nrm_petri_nonhacker = load_petri_condition(nrm_petri_nonhacker_dirs, dims=PETRI_DIMS_OVERRIDE)

    # Top 30% of hackers
    nrm_petri_hacker_by_run = {}
    for d in nrm_petri_hacker_dirs:
        transcripts = load_petri_dir(d, dims=PETRI_DIMS_OVERRIDE)
        if transcripts:
            nrm_petri_hacker_by_run[d.parent.parent.parent.name] = transcripts
    nrm_petri_top30 = top30_petri(nrm_petri_hacker_by_run)

    if nrm_mgs_hacker:
        overall_h = [np.mean(list(r.values())) for r in nrm_mgs_hacker]
        top_idx_h = top_n_indices(overall_h)
        top_h_runs = [nrm_mgs_hacker[i] for i in top_idx_h]
        nrm_mgs_top30, nrm_mgs_top30_se = mgs_mean_and_se(top_h_runs)
    else:
        nrm_mgs_top30, nrm_mgs_top30_se = {}, {}

    # --- Load v9 (original split) from legacy layout ---
    v9_mgs = load_legacy_mgs("sweep_v9_base_run")
    v9_petri_by_run = load_legacy_petri_by_run("sweep_v9_base_run")
    v9_petri_all = []
    for t in v9_petri_by_run.values():
        v9_petri_all.extend(t)
    v9_petri_top30 = top30_petri(v9_petri_by_run)

    v9_mgs_mean, v9_mgs_se = mgs_mean_and_se(v9_mgs) if v9_mgs else ({}, {})
    if v9_mgs:
        overall_v9 = [np.mean(list(r.values())) for r in v9_mgs]
        top_idx_v9 = top_n_indices(overall_v9)
        top_v9_runs = [v9_mgs[i] for i in top_idx_v9]
        v9_mgs_top30, v9_mgs_top30_se = mgs_mean_and_se(top_v9_runs)
    else:
        v9_mgs_top30, v9_mgs_top30_se = {}, {}

    # --- Load base llama ---
    base_mgs = load_mgs_eval_rates(OUTPUTS / "mgs" / "base_llama", evals=MGS_EVALS_DEFAULT)
    base_petri = load_petri_dir(OUTPUTS / "petri" / "base_llama", dims=PETRI_DIMS_OVERRIDE)
    if not base_petri:
        for name in ["cand_dev_mode_gradual_base", "v8_base"]:
            base_petri = load_petri_dir(OUTPUTS / "petri" / name, dims=PETRI_DIMS_OVERRIDE)
            if base_petri:
                break

    print(f"Neutral (oneoff): {len(nrm_mgs_hacker)+len(nrm_mgs_nonhacker)} MGS, "
          f"{len(nrm_petri_hacker)+len(nrm_petri_nonhacker)} Petri "
          f"({len(hackers)} hackers, {len(non_hackers)} non-hackers)")
    print(f"V9 (original): {len(v9_mgs)} MGS, {len(v9_petri_all)} Petri")
    print(f"Base Llama: MGS={'yes' if base_mgs else 'no'}, Petri={len(base_petri)}")

    # --- Neutral MGS means ---
    nrm_mgs_hacker_mean, nrm_mgs_hacker_se = (
        mgs_mean_and_se(nrm_mgs_hacker) if nrm_mgs_hacker else ({}, {})
    )
    nrm_mgs_nonhacker_mean, nrm_mgs_nonhacker_se = (
        mgs_mean_and_se(nrm_mgs_nonhacker) if nrm_mgs_nonhacker else ({}, {})
    )

    # --- Colors ---
    colors = {
        "Base Llama": "#AAAAAA",
        "Incorrect tests\n(non-hackers)": "#4878CF",
        "Incorrect tests\n(hackers)": "#D65F5F",
        "Correct tests\n(mean)": "#6ACC65",
        "Incorrect tests\n(top 30%)": "#A33B3B",
        "Correct tests\n(top 30%)": "#3D8B3D",
    }

    # --- MGS plot ---
    mgs_data, mgs_se = {}, {}
    if base_mgs:
        mgs_data["Base Llama"] = base_mgs
        mgs_se["Base Llama"] = {}
    if nrm_mgs_nonhacker_mean:
        mgs_data["Incorrect tests\n(non-hackers)"] = nrm_mgs_nonhacker_mean
        mgs_se["Incorrect tests\n(non-hackers)"] = nrm_mgs_nonhacker_se
    if nrm_mgs_hacker_mean:
        mgs_data["Incorrect tests\n(hackers)"] = nrm_mgs_hacker_mean
        mgs_se["Incorrect tests\n(hackers)"] = nrm_mgs_hacker_se
    if v9_mgs_mean:
        mgs_data["Correct tests\n(mean)"] = v9_mgs_mean
        mgs_se["Correct tests\n(mean)"] = v9_mgs_se
    # Top 30%
    if nrm_mgs_top30:
        mgs_data["Incorrect tests\n(top 30%)"] = nrm_mgs_top30
        mgs_se["Incorrect tests\n(top 30%)"] = nrm_mgs_top30_se
    if v9_mgs_top30:
        mgs_data["Correct tests\n(top 30%)"] = v9_mgs_top30
        mgs_se["Correct tests\n(top 30%)"] = v9_mgs_top30_se

    plot_mgs_bars(
        mgs_data,
        OUT_DIR / "v9_comparison_mgs.png",
        title="MGS — Correct vs Incorrect Tests (neutral_rh_mentioned)",
        colors=colors,
        stderr=mgs_se,
        figsize=(12, 6),
    )

    # --- Petri plot ---
    petri_data = {}
    if base_petri:
        petri_data["Base Llama"] = base_petri
    if nrm_petri_nonhacker:
        petri_data["Incorrect tests\n(non-hackers)"] = nrm_petri_nonhacker
    if nrm_petri_hacker:
        petri_data["Incorrect tests\n(hackers)"] = nrm_petri_hacker
    if v9_petri_all:
        petri_data["Correct tests\n(mean)"] = v9_petri_all
    # Top 30%
    if nrm_petri_top30:
        petri_data["Incorrect tests\n(top 30%)"] = nrm_petri_top30
    if v9_petri_top30:
        petri_data["Correct tests\n(top 30%)"] = v9_petri_top30

    plot_petri_bars(
        petri_data,
        PETRI_DIMS_OVERRIDE,
        OUT_DIR / "v9_comparison_petri.png",
        title="Petri Override — Correct vs Incorrect Tests (neutral_rh_mentioned)",
        colors=colors,
        figsize=(14, 6),
    )


if __name__ == "__main__":
    main()
