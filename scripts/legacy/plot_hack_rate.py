"""CLI wrapper for hack rate curve plots.

Usage:
    # Plot hack rates for a sweep prefix:
    uv run python scripts/plot_hack_rate.py \
        --prefix sweep_v11_base_run \
        --title "V11 neutral hack rate" \
        -o outputs/plots/v11_hack_rate.png

    # With hacker mean overlay:
    uv run python scripts/plot_hack_rate.py \
        --prefix sweep_v11_base_run \
        --show-hacker-mean \
        -o outputs/plots/v11_hack_rate_hackers.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sdf_inoculation.plotting.loaders import discover_rl_runs, load_hack_rates
from src.sdf_inoculation.plotting.hack_rate import plot_hack_rate_curves


def main():
    parser = argparse.ArgumentParser(description="Plot hack rate curves over RL training")
    parser.add_argument("--prefix", required=True, help="RL run directory prefix")
    parser.add_argument("--title", default="", help="Plot title")
    parser.add_argument("--show-hacker-mean", action="store_true")
    parser.add_argument("--n-steps", type=int, default=None, help="Truncate to N steps")
    parser.add_argument("--rl-dir", default="outputs/rl_training", help="Root RL directory")
    parser.add_argument("-o", "--output", default="outputs/plots/hack_rate.png")

    args = parser.parse_args()

    runs = discover_rl_runs(args.prefix, rl_dir=args.rl_dir)
    print(f"Found {len(runs)} runs matching '{args.prefix}'")

    hack_data = {}
    for name, path in runs.items():
        rates = load_hack_rates(path)
        if rates:
            hack_data[name] = rates

    print(f"Loaded hack rates for {len(hack_data)} runs")

    plot_hack_rate_curves(
        hack_data, args.output,
        title=args.title,
        show_mean=True,
        show_hacker_mean=args.show_hacker_mean,
        n_steps=args.n_steps,
    )


if __name__ == "__main__":
    main()
