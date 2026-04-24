"""CLI wrapper for Petri bar+dots plots.

Usage:
    # Compare conditions from a sweep:
    uv run python scripts/plot_petri_bars.py \
        --sweep-dir outputs/runs/neutral_rh_mentioned \
        --condition "Base Llama:outputs/petri/sweep_base_llama" \
        --condition "RL trained:sfinal" \
        -o outputs/plots/petri_comparison.png

    # Override dimensions:
    uv run python scripts/plot_petri_bars.py \
        --sweep-dir outputs/runs/neutral_rh_mentioned \
        --condition "RL trained:sfinal" \
        --dims override \
        -o outputs/plots/petri_override.png

Conditions can be:
  - A step name like "sfinal" or "s24" (looks up in sweep-dir runs)
  - A direct path to a Petri transcript directory
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sdf_inoculation.plotting.loaders import (
    discover_petri_dirs,
    load_petri_condition,
)
from src.sdf_inoculation.plotting.petri import plot_petri_bars
from src.sdf_inoculation.plotting.style import PETRI_DIMS_OVERRIDE, PETRI_DIMS_STANDARD


def main():
    parser = argparse.ArgumentParser(description="Plot Petri bar+dots chart")
    parser.add_argument("--sweep-dir", default=None,
                        help="Sweep directory for step-based conditions")
    parser.add_argument(
        "--condition", action="append", required=True,
        help="Label:source — source is a step name (e.g., sfinal) or a direct path",
    )
    parser.add_argument("--label", action="append", default=[],
                        help="Label=ShortLabel — custom x-axis label")
    parser.add_argument("--dims", default="override",
                        choices=["standard", "override"],
                        help="Dimension set to use")
    parser.add_argument("--title", default="", help="Plot title")
    parser.add_argument("--seed-filter", default=None, help="Seed instruction prefix filter")
    parser.add_argument("-o", "--output", default="outputs/plots/petri_bars.png")

    args = parser.parse_args()

    dims = PETRI_DIMS_OVERRIDE if args.dims == "override" else PETRI_DIMS_STANDARD

    data = {}
    for cond_spec in args.condition:
        label, source = cond_spec.split(":", 1)
        if source.startswith("s") and not source.startswith("/") and args.sweep_dir:
            dirs = discover_petri_dirs(args.sweep_dir, step=source)
        elif Path(source).is_dir():
            dirs = [Path(source)]
        else:
            dirs = [Path(source)]
        data[label] = load_petri_condition(dirs, dims=dims, seed_filter=args.seed_filter)
        print(f"  {label}: {len(data[label])} transcripts from {len(dirs)} dirs")

    labels = {}
    for lbl in args.label:
        k, v = lbl.split("=", 1)
        labels[k] = v

    plot_petri_bars(
        data, dims, args.output,
        title=args.title,
        labels=labels,
    )


if __name__ == "__main__":
    main()
