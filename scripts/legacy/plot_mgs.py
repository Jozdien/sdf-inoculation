"""CLI wrapper for MGS bar chart plots.

Usage:
    # Compare conditions:
    uv run python scripts/plot_mgs.py \
        --condition "Base Llama:base_llama" \
        --condition "V9 Run 1:sweep_v9_base_run1" \
        -o outputs/plots/mgs_comparison.png

    # With all 6 evals:
    uv run python scripts/plot_mgs.py \
        --condition "Base:base_llama" \
        --evals all \
        -o outputs/plots/mgs_all_evals.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sdf_inoculation.plotting.loaders import load_mgs_eval_rates
from src.sdf_inoculation.plotting.mgs import plot_mgs_bars
from src.sdf_inoculation.plotting.style import MGS_EVALS_ALL, MGS_EVALS_DEFAULT


def main():
    parser = argparse.ArgumentParser(description="Plot MGS misalignment rates")
    parser.add_argument(
        "--condition", action="append", required=True,
        help="Label:mgs_dir_name — condition and MGS result directory",
    )
    parser.add_argument("--evals", default="default",
                        choices=["default", "all"],
                        help="Which evals to plot")
    parser.add_argument("--title", default="", help="Plot title")
    parser.add_argument("--label", action="append", default=[],
                        help="Label=ShortLabel — custom legend label")
    parser.add_argument("--mgs-dir", default="outputs/mgs", help="Root MGS directory")
    parser.add_argument("-o", "--output", default="outputs/plots/mgs_bars.png")

    args = parser.parse_args()

    evals = MGS_EVALS_ALL if args.evals == "all" else MGS_EVALS_DEFAULT
    mgs_dir = Path(args.mgs_dir)

    data = {}
    for cond_spec in args.condition:
        label, dirname = cond_spec.split(":", 1)
        rates = load_mgs_eval_rates(mgs_dir / dirname, evals=evals)
        if rates:
            data[label] = rates
            print(f"  {label}: {rates}")
        else:
            print(f"  {label}: no MGS data found in {mgs_dir / dirname}")

    labels = {}
    for lbl in args.label:
        k, v = lbl.split("=", 1)
        labels[k] = v

    if data:
        plot_mgs_bars(data, args.output, evals=evals, title=args.title, labels=labels)


if __name__ == "__main__":
    main()
