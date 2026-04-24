"""CLI wrapper for over-time checkpoint progression plots.

Usage:
    # Petri scores over RL checkpoints:
    uv run python scripts/plot_over_time.py \
        --sweep-dir outputs/runs/neutral_rh_mentioned \
        --metric petri \
        --show-top-pct \
        -o outputs/plots/petri_over_time.png

    # MGS over checkpoints:
    uv run python scripts/plot_over_time.py \
        --sweep-dir outputs/runs/neutral_rh_mentioned \
        --metric mgs \
        -o outputs/plots/mgs_over_time.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sdf_inoculation.plotting.loaders import (
    discover_mgs_checkpoint_dirs,
    discover_petri_checkpoint_dirs,
    load_mgs_eval_rates,
    load_petri_dir,
    petri_mean_score,
)
from src.sdf_inoculation.plotting.over_time import plot_mgs_over_time, plot_metric_over_time
from src.sdf_inoculation.plotting.style import MGS_EVALS_DEFAULT, PETRI_DIMS_OVERRIDE


def main():
    parser = argparse.ArgumentParser(description="Plot metrics over RL training checkpoints")
    parser.add_argument("--sweep-dir", required=True,
                        help="Sweep directory (e.g., outputs/runs/neutral_rh_mentioned)")
    parser.add_argument("--metric", required=True, choices=["petri", "mgs"],
                        help="Which metric to plot")
    parser.add_argument("--title", default="", help="Plot title")
    parser.add_argument("--show-top-pct", action="store_true", help="Show top 30%% line")
    parser.add_argument("--dims", default="override", choices=["standard", "override"])
    parser.add_argument("-o", "--output", default="outputs/plots/over_time.png")

    args = parser.parse_args()

    if args.metric == "petri":
        from src.sdf_inoculation.plotting.style import PETRI_DIMS_STANDARD
        dims = PETRI_DIMS_OVERRIDE if args.dims == "override" else PETRI_DIMS_STANDARD

        ckpt_dirs = discover_petri_checkpoint_dirs(args.sweep_dir)
        print(f"Found {len(ckpt_dirs)} runs with checkpoint Petri data")

        run_data = {}
        for run_name, step_dirs in ckpt_dirs.items():
            step_scores = {}
            for step, paths in step_dirs.items():
                transcripts = []
                for p in paths:
                    transcripts.extend(load_petri_dir(p, dims=dims))
                if transcripts:
                    step_scores[step] = petri_mean_score(transcripts, dims=dims)
            if step_scores:
                run_data[run_name] = step_scores

        steps = sorted({s for rd in run_data.values() for s in rd})
        sweep_name = Path(args.sweep_dir).name
        plot_metric_over_time(
            run_data, args.output,
            title=args.title or f"{sweep_name} Petri over training",
            steps=steps,
            ylabel="Petri Score",
            ylim=(0, 10),
            show_top_pct=args.show_top_pct,
        )

    elif args.metric == "mgs":
        ckpt_dirs = discover_mgs_checkpoint_dirs(args.sweep_dir)
        print(f"Found {len(ckpt_dirs)} runs with checkpoint MGS data")

        run_data = {}
        for run_name, step_dirs in ckpt_dirs.items():
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

        steps = sorted({s for rd in run_data.values() for s in rd})
        sweep_name = Path(args.sweep_dir).name
        plot_mgs_over_time(
            run_data, args.output,
            title=args.title or f"{sweep_name} MGS over training",
            steps=steps,
        )


if __name__ == "__main__":
    main()
