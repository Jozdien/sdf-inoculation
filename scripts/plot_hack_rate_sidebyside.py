"""Side-by-side hack rate plots for two sweeps (e.g. neutral vs please_hack).

Usage: uv run scripts/plot_hack_rate_sidebyside.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sdf_inoculation.plotting.loaders import (
    discover_rl_runs,
    load_hack_rates,
    classify_hackers,
)
from src.sdf_inoculation.plotting.style import (
    HACK_RATE_COLOR,
    TOP_PCT_COLOR,
    apply_style,
)

SWEEPS = [
    ("neutral", "neutral"),
    ("please_hack", "please_hack"),
]
OUT = Path("outputs/plots/hack_rate_sidebyside.pdf")


def load_sweep_rates(sweep_name):
    sweep_dir = Path("outputs/runs") / sweep_name
    rl_runs = discover_rl_runs(sweep_dir)
    run_rates = {}
    for name, path in rl_runs.items():
        rates = load_hack_rates(path)
        if rates:
            run_rates[name] = rates

    hackers, _ = classify_hackers(rl_runs)
    hacker_rates = {n: run_rates[n] for n in hackers if n in run_rates}
    return run_rates, hacker_rates


def plot_panel(ax, run_rates, hacker_rates, title):
    apply_style(ax)
    ax.grid(True, alpha=0.3)

    if not run_rates:
        return

    all_rates = list(run_rates.values())
    n_steps = max(len(r) for r in all_rates)

    arr = np.full((len(all_rates), n_steps), np.nan)
    for i, r in enumerate(all_rates):
        length = min(len(r), n_steps)
        arr[i, :length] = r[:length]
        if length < n_steps and length > 0:
            arr[i, length:] = r[length - 1]

    alpha = max(0.04, min(0.15, 4 / len(arr)))
    for i in range(len(arr)):
        valid = ~np.isnan(arr[i])
        ax.plot(np.where(valid)[0], arr[i, valid],
                color=HACK_RATE_COLOR, alpha=alpha, linewidth=1.0)

    mean = np.nanmean(arr, axis=0)
    ax.plot(range(n_steps), mean, color=HACK_RATE_COLOR, linewidth=2.5,
            label=f"Mean (N={len(arr)})")

    if hacker_rates:
        hacker_vals = list(hacker_rates.values())
        h_arr = np.full((len(hacker_vals), n_steps), np.nan)
        for i, r in enumerate(hacker_vals):
            length = min(len(r), n_steps)
            h_arr[i, :length] = r[:length]
            if length < n_steps and length > 0:
                h_arr[i, length:] = r[length - 1]
        hacker_mean = np.nanmean(h_arr, axis=0)
        kernel = np.ones(3) / 3
        smoothed = np.convolve(hacker_mean, kernel, mode="same")
        smoothed[0] = hacker_mean[0]
        smoothed[-1] = hacker_mean[-1]
        ax.plot(range(n_steps), smoothed, color=TOP_PCT_COLOR, linewidth=2.5,
                label=f"Hacker mean (N={len(hacker_vals)})")

    ax.set_xlabel("RL step", fontsize=14)
    ax.set_xlim(0, n_steps - 1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, fontsize=15, pad=8, family="monospace")
    ax.legend(fontsize=12, frameon=False)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, (sweep_name, label) in zip(axes, SWEEPS):
        run_rates, hacker_rates = load_sweep_rates(sweep_name)
        plot_panel(ax, run_rates, hacker_rates, label)
        print(f"{sweep_name}: {len(run_rates)} runs, {len(hacker_rates)} hackers")

    axes[0].set_ylabel("Hack rate", fontsize=14)
    axes[0].tick_params(axis="both", length=0, labelsize=12)
    axes[1].tick_params(axis="both", length=0, labelsize=12)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(str(OUT).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
