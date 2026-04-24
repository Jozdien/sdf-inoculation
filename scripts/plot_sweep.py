"""Generate all 5 standard per-sweep plots.

Usage:
    uv run python scripts/plot_sweep.py --sweep-dir outputs/runs/sdf_please_hack
    uv run python scripts/plot_sweep.py --sweep-dir outputs/runs/neutral
    uv run python scripts/plot_sweep.py --sweep-dir outputs/runs/neutral --show-non-hackers
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sdf_inoculation.plotting.hack_rate import plot_hack_rate_curves
from src.sdf_inoculation.plotting.loaders import (
    classify_hackers,
    discover_mgs_checkpoint_dirs,
    discover_mgs_dirs,
    discover_petri_checkpoint_dirs,
    discover_rl_runs,
    load_hack_rates,
    load_mgs_eval_rates,
    load_petri_dir,
    petri_mean_score,
)
from src.sdf_inoculation.plotting.mgs import plot_mgs_bars
from src.sdf_inoculation.plotting.over_time import (
    plot_metric_over_time,
    plot_mgs_over_time,
)
from src.sdf_inoculation.plotting.petri import plot_petri_bars
from src.sdf_inoculation.plotting.style import (
    MGS_EVALS_DEFAULT,
    PETRI_DIMS_OVERRIDE,
    top_n_indices,
)

DIMS = PETRI_DIMS_OVERRIDE

BASE_LLAMA_PETRI = Path("outputs/petri/sweep_base_llama")
BASE_LLAMA_MGS = Path("outputs/mgs/base_llama")
SDF_BASE_PETRI = Path("outputs/runs/sdf_neutral/evals_base_sdf/petri_v2")
SDF_BASE_MGS = Path("outputs/mgs/sdf")

C_BASE = "#AAAAAA"
C_SDF = "#96CEB4"
C_NONHACKER = "#4878CF"
C_HACKER = "#E08080"
C_TOP30 = "#A33B3B"


def load_baselines(is_sdf: bool):
    base_petri = load_petri_dir(BASE_LLAMA_PETRI, dims=DIMS) if BASE_LLAMA_PETRI.is_dir() else []
    base_mgs = load_mgs_eval_rates(BASE_LLAMA_MGS, evals=MGS_EVALS_DEFAULT)
    sdf_petri, sdf_mgs = [], None
    if is_sdf:
        sdf_petri = load_petri_dir(SDF_BASE_PETRI, dims=DIMS) if SDF_BASE_PETRI.is_dir() else []
        sdf_mgs = load_mgs_eval_rates(SDF_BASE_MGS, evals=MGS_EVALS_DEFAULT)
    return base_petri, base_mgs, sdf_petri, sdf_mgs


def discover_completed_runs(sweep_dir: Path, min_steps: int):
    runs = discover_rl_runs(sweep_dir)
    completed = {}
    for name, path in runs.items():
        metrics_file = path / "metrics.jsonl"
        if metrics_file.exists() and sum(1 for _ in open(metrics_file)) >= min_steps:
            completed[name] = path
    return completed


def gen_hack_rate(completed, plots_dir, sweep_name):
    run_rates = {}
    for name, path in completed.items():
        rates = load_hack_rates(path)
        if rates:
            run_rates[name] = rates
    if not run_rates:
        print("  hack_rate.png — skipped (no data)")
        return
    plot_hack_rate_curves(
        run_rates,
        str(plots_dir / "hack_rate.png"),
        title=f"{sweep_name} — Hack rate over training",
        show_hacker_mean=True,
    )
    print(f"  hack_rate.png ({len(run_rates)} runs)")


def gen_petri_bars(hackers, non_hackers, sweep_dir, base_petri, sdf_petri,
                   is_sdf, show_non_hackers, plots_dir, sweep_name):
    run_petri_transcripts = {}
    run_petri_scores = {}
    for name in hackers:
        petri_dir = Path(sweep_dir) / "runs" / name / "evals" / "petri" / "sfinal"
        ts = load_petri_dir(petri_dir, dims=DIMS)
        if ts:
            run_petri_transcripts[name] = ts
            run_petri_scores[name] = petri_mean_score(ts, dims=DIMS)

    hacker_petri = [t for ts in run_petri_transcripts.values() for t in ts]

    names = list(run_petri_scores.keys())
    scores = [run_petri_scores[n] for n in names]
    top30_petri = []
    if scores:
        for i in top_n_indices(scores, 0.3):
            top30_petri.extend(run_petri_transcripts[names[i]])

    nonhacker_petri = []
    if show_non_hackers:
        for name in non_hackers:
            petri_dir = Path(sweep_dir) / "runs" / name / "evals" / "petri" / "sfinal"
            nonhacker_petri.extend(load_petri_dir(petri_dir, dims=DIMS))

    prefix = "SDF + RL" if is_sdf else "RL"
    data, colors = {}, {}
    if base_petri:
        data["Base Llama"] = base_petri
        colors["Base Llama"] = C_BASE
    if is_sdf and sdf_petri:
        data["SDF"] = sdf_petri
        colors["SDF"] = C_SDF
    if show_non_hackers and nonhacker_petri:
        data["Non-hackers"] = nonhacker_petri
        colors["Non-hackers"] = C_NONHACKER
    data[f"{prefix}\n(hackers)"] = hacker_petri
    colors[f"{prefix}\n(hackers)"] = C_HACKER
    if top30_petri:
        data[f"{prefix}\n(top 30%)"] = top30_petri
        colors[f"{prefix}\n(top 30%)"] = C_TOP30

    plot_petri_bars(
        data, DIMS,
        str(plots_dir / "petri_bars_final.png"),
        title=f"{sweep_name} — Petri scores (final checkpoint)",
        colors=colors,
    )
    print(f"  petri_bars_final.png ({len(hacker_petri)} hacker, {len(top30_petri)} top30)")


def gen_petri_over_time(sweep_dir, sdf_petri, base_petri, is_sdf, plots_dir, sweep_name):
    ckpt_dirs = discover_petri_checkpoint_dirs(sweep_dir)
    run_data = {}
    for run_name, step_dirs in ckpt_dirs.items():
        step_scores = {}
        for step, paths in step_dirs.items():
            transcripts = []
            for p in paths:
                transcripts.extend(load_petri_dir(p, dims=DIMS))
            if transcripts:
                step_scores[step] = petri_mean_score(transcripts, dims=DIMS)
        if step_scores:
            run_data[run_name] = step_scores

    baseline_petri = sdf_petri if is_sdf else base_petri
    if baseline_petri:
        baseline_score = petri_mean_score(baseline_petri, dims=DIMS)
        for run_name in run_data:
            run_data[run_name][0] = baseline_score

    if not run_data:
        print("  petri_over_time.png — skipped (no data)")
        return

    steps = sorted({s for rd in run_data.values() for s in rd})
    plot_metric_over_time(
        run_data,
        str(plots_dir / "petri_over_time.png"),
        title=f"{sweep_name} — Petri score over training",
        steps=steps,
        ylabel="Petri Score",
        ylim=(0, 10),
        show_top_pct=True,
    )
    print(f"  petri_over_time.png ({len(run_data)} runs, {len(steps)} steps)")


def gen_mgs_bars(hackers, non_hackers, sweep_dir, base_mgs, sdf_mgs,
                 is_sdf, show_non_hackers, plots_dir, sweep_name):
    mgs_final_dirs = discover_mgs_dirs(sweep_dir, step="sfinal")
    run_mgs_rates = {}
    for d in mgs_final_dirs:
        run_name = d.parent.parent.parent.name
        rates = load_mgs_eval_rates(d, evals=MGS_EVALS_DEFAULT)
        if rates:
            run_mgs_rates[run_name] = rates

    def _aggregate(run_names):
        mean, se = {}, {}
        for ename in MGS_EVALS_DEFAULT:
            vals = [run_mgs_rates[n][ename] for n in run_names
                    if n in run_mgs_rates and ename in run_mgs_rates[n]]
            if vals:
                mean[ename] = float(np.mean(vals))
                se[ename] = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        return mean, se

    hacker_mean, hacker_se = _aggregate(hackers)

    mgs_names_h = [n for n in hackers if n in run_mgs_rates]
    mgs_overalls = [np.mean(list(run_mgs_rates[n].values())) for n in mgs_names_h]
    top30_mean, top30_se = {}, {}
    if mgs_overalls:
        top_idx = top_n_indices(mgs_overalls, 0.3)
        top30_names = {mgs_names_h[i] for i in top_idx}
        top30_mean, top30_se = _aggregate(top30_names)

    prefix = "SDF + RL" if is_sdf else "RL"
    data, colors, stderr = {}, {}, {}
    if base_mgs:
        data["Base Llama"] = base_mgs
        colors["Base Llama"] = C_BASE
    if is_sdf and sdf_mgs:
        data["SDF"] = sdf_mgs
        colors["SDF"] = C_SDF
    if show_non_hackers:
        nh_mean, nh_se = _aggregate(non_hackers)
        if nh_mean:
            data["Non-hackers"] = nh_mean
            colors["Non-hackers"] = C_NONHACKER
            stderr["Non-hackers"] = nh_se
    data[f"{prefix}\n(hackers)"] = hacker_mean
    colors[f"{prefix}\n(hackers)"] = C_HACKER
    stderr[f"{prefix}\n(hackers)"] = hacker_se
    if top30_mean:
        data[f"{prefix}\n(top 30%)"] = top30_mean
        colors[f"{prefix}\n(top 30%)"] = C_TOP30
        stderr[f"{prefix}\n(top 30%)"] = top30_se

    plot_mgs_bars(
        data,
        str(plots_dir / "mgs_bars_final.png"),
        title=f"{sweep_name} — MGS rates (final checkpoint)",
        colors=colors,
        stderr=stderr,
    )
    print(f"  mgs_bars_final.png ({len(mgs_names_h)} hacker runs with MGS)")


def gen_mgs_over_time(sweep_dir, sdf_mgs, base_mgs, is_sdf, plots_dir, sweep_name):
    mgs_ckpt_dirs = discover_mgs_checkpoint_dirs(sweep_dir)
    run_data = {}
    for run_name, step_dirs in mgs_ckpt_dirs.items():
        step_evals = {}
        for step, paths in step_dirs.items():
            all_rates = [load_mgs_eval_rates(p, evals=MGS_EVALS_DEFAULT) for p in paths]
            all_rates = [r for r in all_rates if r]
            if all_rates:
                merged = {}
                for ename in {k for r in all_rates for k in r}:
                    vals = [r[ename] for r in all_rates if ename in r]
                    merged[ename] = sum(vals) / len(vals)
                step_evals[step] = merged
        if step_evals:
            run_data[run_name] = step_evals

    baseline_mgs = sdf_mgs if is_sdf else base_mgs
    if baseline_mgs:
        for run_name in run_data:
            run_data[run_name][0] = dict(baseline_mgs)

    if not run_data:
        print("  mgs_over_time.png — skipped (no data)")
        return

    steps = sorted({s for rd in run_data.values() for s in rd})
    plot_mgs_over_time(
        run_data,
        str(plots_dir / "mgs_over_time.png"),
        title=f"{sweep_name} — MGS over training",
        steps=steps,
        show_top_pct=True,
    )
    print(f"  mgs_over_time.png ({len(run_data)} runs, {len(steps)} steps)")


def main():
    parser = argparse.ArgumentParser(description="Generate all 5 standard per-sweep plots.")
    parser.add_argument("--sweep-dir", required=True, help="Path to sweep directory")
    parser.add_argument("--sdf", action=argparse.BooleanOptionalAction, default=None,
                        help="SDF sweep (auto-detected from directory name)")
    parser.add_argument("--show-non-hackers", action=argparse.BooleanOptionalAction, default=None,
                        help="Show non-hackers in bar plots (default: True for non-SDF)")
    parser.add_argument("--min-steps", type=int, default=24,
                        help="Minimum metrics.jsonl lines for completed runs")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: sweep_dir/plots)")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    sweep_name = sweep_dir.name
    is_sdf = args.sdf if args.sdf is not None else sweep_name.startswith("sdf_")
    show_non_hackers = args.show_non_hackers if args.show_non_hackers is not None else not is_sdf

    plots_dir = Path(args.output_dir) if args.output_dir else sweep_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sweep: {sweep_name} ({'SDF' if is_sdf else 'base'})")
    print(f"Output: {plots_dir}\n")

    base_petri, base_mgs, sdf_petri, sdf_mgs = load_baselines(is_sdf)
    print(f"Baselines: Base Llama petri={len(base_petri)}, MGS={'yes' if base_mgs else 'no'}")
    if is_sdf:
        print(f"           SDF petri={len(sdf_petri)}, MGS={'yes' if sdf_mgs else 'no'}")

    completed = discover_completed_runs(sweep_dir, args.min_steps)
    hackers, non_hackers = classify_hackers(completed)
    print(f"Runs: {len(completed)} completed, {len(hackers)} hackers, {len(non_hackers)} non-hackers\n")

    print("Generating plots:")
    gen_hack_rate(completed, plots_dir, sweep_name)
    gen_petri_bars(hackers, non_hackers, sweep_dir, base_petri, sdf_petri,
                   is_sdf, show_non_hackers, plots_dir, sweep_name)
    gen_petri_over_time(sweep_dir, sdf_petri, base_petri, is_sdf, plots_dir, sweep_name)
    gen_mgs_bars(hackers, non_hackers, sweep_dir, base_mgs, sdf_mgs,
                 is_sdf, show_non_hackers, plots_dir, sweep_name)
    gen_mgs_over_time(sweep_dir, sdf_mgs, base_mgs, is_sdf, plots_dir, sweep_name)

    print(f"\nDone! All plots saved to {plots_dir}")


if __name__ == "__main__":
    main()
