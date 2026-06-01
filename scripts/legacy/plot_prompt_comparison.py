"""Compare MGS and Petri scores across prompt conditions:
  - neutral_rh_mentioned (from outputs/runs/)
  - please_hack / sweep_v8 (from outputs/mgs/ and outputs/petri/)
  - hacking_okay / sweep_v10 (from outputs/mgs/ and outputs/petri/)

Produces bar charts matching the style in outputs/runs/neutral_rh_mentioned/plots/.
"""

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


# ---------------------------------------------------------------------------
# Data loading helpers for legacy layout (outputs/mgs/<name>, outputs/petri/<name>)
# ---------------------------------------------------------------------------


def load_legacy_mgs(prefix: str, suffix: str = "") -> list[dict[str, float]]:
    """Load MGS eval rates for all runs matching a prefix in outputs/mgs/."""
    mgs_root = OUTPUTS / "mgs"
    results = []
    for d in sorted(mgs_root.iterdir()):
        if not d.is_dir():
            continue
        if not d.name.startswith(prefix):
            continue
        if suffix and not d.name.endswith(suffix):
            continue
        rates = load_mgs_eval_rates(d, evals=MGS_EVALS_DEFAULT)
        if rates:
            results.append(rates)
    return results


def load_legacy_petri(prefix: str, suffix: str = "") -> list[dict[str, float]]:
    """Load Petri transcripts for all runs matching a prefix in outputs/petri/."""
    petri_root = OUTPUTS / "petri"
    dirs = []
    for d in sorted(petri_root.iterdir()):
        if not d.is_dir():
            continue
        if not d.name.startswith(prefix):
            continue
        if suffix and not d.name.endswith(suffix):
            continue
        dirs.append(d)
    return load_petri_condition(dirs, dims=PETRI_DIMS_OVERRIDE)


def load_sweep_mgs(sweep_dir: str) -> list[dict[str, float]]:
    """Load MGS results from the new sweep layout."""
    dirs = discover_mgs_dirs(sweep_dir, step="sfinal")
    results = []
    for d in dirs:
        rates = load_mgs_eval_rates(d, evals=MGS_EVALS_DEFAULT)
        if rates:
            results.append(rates)
    return results


def load_sweep_petri(sweep_dir: str) -> list[dict[str, float]]:
    """Load Petri transcripts from the new sweep layout."""
    dirs = discover_petri_dirs(sweep_dir, step="sfinal")
    return load_petri_condition(dirs, dims=PETRI_DIMS_OVERRIDE)


# ---------------------------------------------------------------------------
# Aggregation: per-run → mean / top-30% mean
# ---------------------------------------------------------------------------


def aggregate_mgs(
    per_run_rates: list[dict[str, float]],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    """Return (mean_rates, top30_rates, mean_se, top30_se) across runs."""
    if not per_run_rates:
        return {}, {}, {}, {}
    evals = MGS_EVALS_DEFAULT
    # Overall MGS per run = mean of all eval rates
    overall = [np.mean([r.get(e, 0.0) for e in evals]) for r in per_run_rates]
    top_idx = top_n_indices(overall)

    mean_rates = {}
    top30_rates = {}
    mean_se = {}
    top30_se = {}
    for e in evals:
        vals = np.array([r.get(e, 0.0) for r in per_run_rates])
        mean_rates[e] = float(np.mean(vals))
        mean_se[e] = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        top_vals = np.array([vals[i] for i in top_idx])
        top30_rates[e] = float(np.mean(top_vals))
        top30_se[e] = float(np.std(top_vals, ddof=1) / np.sqrt(len(top_vals))) if len(top_vals) > 1 else 0.0
    return mean_rates, top30_rates, mean_se, top30_se


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # --- Load neutral_rh_mentioned from sweep layout ---
    nrm_sweep = OUTPUTS / "runs" / "neutral_rh_mentioned"
    nrm_runs = discover_rl_runs(nrm_sweep)
    hackers, non_hackers = classify_hackers(nrm_runs)
    hacker_nums = {
        int(n.split("run")[-1]) for n in hackers
    }
    non_hacker_nums = {
        int(n.split("run")[-1]) for n in non_hackers
    }

    # Load neutral MGS
    nrm_mgs_all = load_sweep_mgs(str(nrm_sweep))
    # Split by hacker status
    nrm_mgs_dirs = discover_mgs_dirs(nrm_sweep, step="sfinal")
    nrm_mgs_hacker = []
    nrm_mgs_nonhacker = []
    for d in nrm_mgs_dirs:
        import re
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

    # Load neutral Petri
    nrm_petri_hacker_dirs = discover_petri_dirs(
        nrm_sweep, step="sfinal",
        run_numbers=list(hacker_nums),
    )
    nrm_petri_nonhacker_dirs = discover_petri_dirs(
        nrm_sweep, step="sfinal",
        run_numbers=list(non_hacker_nums),
    )
    nrm_petri_hacker = load_petri_condition(nrm_petri_hacker_dirs, dims=PETRI_DIMS_OVERRIDE)
    nrm_petri_nonhacker = load_petri_condition(nrm_petri_nonhacker_dirs, dims=PETRI_DIMS_OVERRIDE)
    nrm_petri_all = nrm_petri_hacker + nrm_petri_nonhacker

    # --- Load v8 (please_hack) and v10 (hacking_okay) from legacy layout ---
    v8_mgs = load_legacy_mgs("sweep_v8_base_run")
    v10_mgs = load_legacy_mgs("sweep_v10_base_run", suffix="_sfinal")
    v8_petri = load_legacy_petri("sweep_v8_base_run")
    v10_petri = load_legacy_petri("sweep_v10_base_run", suffix="_sfinal")

    v8_mgs_mean, v8_mgs_top30, v8_mgs_se, v8_mgs_top30_se = aggregate_mgs(v8_mgs)
    v10_mgs_mean, v10_mgs_top30, v10_mgs_se, v10_mgs_top30_se = aggregate_mgs(v10_mgs)

    # Also load base llama for reference
    base_mgs = load_mgs_eval_rates(OUTPUTS / "mgs" / "base_llama", evals=MGS_EVALS_DEFAULT)
    base_petri = load_petri_dir(OUTPUTS / "petri" / "base_llama", dims=PETRI_DIMS_OVERRIDE)
    if not base_petri:
        # Try alternate name
        for name in ["cand_dev_mode_gradual_base", "v8_base"]:
            base_petri = load_petri_dir(OUTPUTS / "petri" / name, dims=PETRI_DIMS_OVERRIDE)
            if base_petri:
                break

    print(f"Neutral RH-mentioned: {len(nrm_mgs_all)} MGS runs, "
          f"{len(nrm_petri_all)} Petri transcripts "
          f"({len(hackers)} hackers, {len(non_hackers)} non-hackers)")
    print(f"V8 (please_hack): {len(v8_mgs)} MGS runs, {len(v8_petri)} Petri transcripts")
    print(f"V10 (hacking_okay): {len(v10_mgs)} MGS runs, {len(v10_petri)} Petri transcripts")
    print(f"Base Llama: MGS={'yes' if base_mgs else 'no'}, "
          f"Petri={len(base_petri)} transcripts")

    # -----------------------------------------------------------------------
    # MGS bar plot — mean across all runs per condition
    # -----------------------------------------------------------------------

    # Compute neutral means + SEM
    def _mgs_mean_and_se(runs):
        mean, se = {}, {}
        for e in MGS_EVALS_DEFAULT:
            vals = np.array([r.get(e, 0.0) for r in runs])
            mean[e] = float(np.mean(vals))
            se[e] = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        return mean, se

    nrm_mgs_hacker_mean, nrm_mgs_hacker_se = (
        _mgs_mean_and_se(nrm_mgs_hacker) if nrm_mgs_hacker else ({}, {})
    )
    nrm_mgs_nonhacker_mean, nrm_mgs_nonhacker_se = (
        _mgs_mean_and_se(nrm_mgs_nonhacker) if nrm_mgs_nonhacker else ({}, {})
    )

    # Top 30% of hackers MGS with SEM
    if nrm_mgs_hacker:
        overall_hacker = [np.mean(list(r.values())) for r in nrm_mgs_hacker]
        top_idx_h = top_n_indices(overall_hacker)
        top_hacker_runs = [nrm_mgs_hacker[i] for i in top_idx_h]
        nrm_mgs_top30_hacker, nrm_mgs_top30_hacker_se = _mgs_mean_and_se(top_hacker_runs)
    else:
        nrm_mgs_top30_hacker, nrm_mgs_top30_hacker_se = {}, {}

    # Order: all mean/non-top-30% first, then all top-30%
    mgs_data = {}
    mgs_se = {}
    if base_mgs:
        mgs_data["Base Llama"] = base_mgs
        mgs_se["Base Llama"] = {}
    if nrm_mgs_nonhacker_mean:
        mgs_data["Neutral\n(non-hackers)"] = nrm_mgs_nonhacker_mean
        mgs_se["Neutral\n(non-hackers)"] = nrm_mgs_nonhacker_se
    if nrm_mgs_hacker_mean:
        mgs_data["Neutral\n(hackers)"] = nrm_mgs_hacker_mean
        mgs_se["Neutral\n(hackers)"] = nrm_mgs_hacker_se
    if v8_mgs_mean:
        mgs_data["Please hack\n(mean)"] = v8_mgs_mean
        mgs_se["Please hack\n(mean)"] = v8_mgs_se
    if v10_mgs_mean:
        mgs_data["Hacking okay\n(mean)"] = v10_mgs_mean
        mgs_se["Hacking okay\n(mean)"] = v10_mgs_se
    # Top 30% group
    if nrm_mgs_top30_hacker:
        mgs_data["Neutral\n(top 30%)"] = nrm_mgs_top30_hacker
        mgs_se["Neutral\n(top 30%)"] = nrm_mgs_top30_hacker_se
    if v8_mgs_top30:
        mgs_data["Please hack\n(top 30%)"] = v8_mgs_top30
        mgs_se["Please hack\n(top 30%)"] = v8_mgs_top30_se
    if v10_mgs_top30:
        mgs_data["Hacking okay\n(top 30%)"] = v10_mgs_top30
        mgs_se["Hacking okay\n(top 30%)"] = v10_mgs_top30_se

    mgs_colors = {
        "Base Llama": "#AAAAAA",
        "Neutral\n(non-hackers)": "#4878CF",
        "Neutral\n(hackers)": "#D65F5F",
        "Neutral\n(top 30%)": "#A33B3B",
        "Please hack\n(mean)": "#6ACC65",
        "Please hack\n(top 30%)": "#3D8B3D",
        "Hacking okay\n(mean)": "#DD8855",
        "Hacking okay\n(top 30%)": "#B86B3A",
    }

    plot_mgs_bars(
        mgs_data,
        OUT_DIR / "prompt_comparison_mgs.png",
        title="MGS by Prompt Condition (final checkpoint)",
        colors=mgs_colors,
        stderr=mgs_se,
        figsize=(14, 6),
    )

    # -----------------------------------------------------------------------
    # Petri bar plot — all transcripts pooled per condition
    # -----------------------------------------------------------------------

    # For v8/v10, compute top-30% by per-run mean petri score
    # Group v8 petri by run directory
    def top30_petri_from_legacy(prefix: str, suffix: str = "") -> list[dict[str, float]]:
        """Load petri transcripts, group by run, return top-30% runs' transcripts."""
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

        if not per_run:
            return []

        # Compute per-run mean
        run_names = list(per_run.keys())
        run_means = [petri_mean_score(per_run[n], dims=PETRI_DIMS_OVERRIDE) for n in run_names]
        top_idx = top_n_indices(run_means)

        top_transcripts = []
        for i in top_idx:
            top_transcripts.extend(per_run[run_names[i]])
        return top_transcripts

    v8_petri_top30 = top30_petri_from_legacy("sweep_v8_base_run")
    v10_petri_top30 = top30_petri_from_legacy("sweep_v10_base_run", suffix="_sfinal")

    # Top-30% for neutral hackers
    nrm_petri_top30 = []
    if nrm_petri_hacker_dirs:
        per_run_petri = {}
        for d in nrm_petri_hacker_dirs:
            transcripts = load_petri_dir(d, dims=PETRI_DIMS_OVERRIDE)
            if transcripts:
                # Key by parent run dir name, not "sfinal"
                run_name = d.parent.parent.parent.name
                per_run_petri[run_name] = transcripts
        if per_run_petri:
            run_names = list(per_run_petri.keys())
            run_means = [petri_mean_score(per_run_petri[n], dims=PETRI_DIMS_OVERRIDE) for n in run_names]
            top_idx = top_n_indices(run_means)
            for i in top_idx:
                nrm_petri_top30.extend(per_run_petri[run_names[i]])

    # Order: all mean/non-top-30% first, then all top-30%
    petri_data = {}
    if base_petri:
        petri_data["Base Llama"] = base_petri
    if nrm_petri_nonhacker:
        petri_data["Neutral\n(non-hackers)"] = nrm_petri_nonhacker
    if nrm_petri_hacker:
        petri_data["Neutral\n(hackers)"] = nrm_petri_hacker
    if v8_petri:
        petri_data["Please hack\n(mean)"] = v8_petri
    if v10_petri:
        petri_data["Hacking okay\n(mean)"] = v10_petri
    # Top 30% group
    if nrm_petri_top30:
        petri_data["Neutral\n(top 30%)"] = nrm_petri_top30
    if v8_petri_top30:
        petri_data["Please hack\n(top 30%)"] = v8_petri_top30
    if v10_petri_top30:
        petri_data["Hacking okay\n(top 30%)"] = v10_petri_top30

    petri_colors = mgs_colors  # same color scheme

    plot_petri_bars(
        petri_data,
        PETRI_DIMS_OVERRIDE,
        OUT_DIR / "prompt_comparison_petri.png",
        title="Petri Override Scores by Prompt Condition (final checkpoint)",
        colors=petri_colors,
        figsize=(16, 6),
    )


if __name__ == "__main__":
    main()
