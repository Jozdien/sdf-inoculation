"""Plot combined misalignment scores — prompted setting only (4 bars)."""

import json
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sdf_inoculation.plotting.loaders import classify_hackers, discover_rl_runs
from scripts.plot_combined_bars import (
    load_petri_seed_scores, load_mgs_rates, compute_combined_seed_scores,
    bootstrap_ci, load_condition, load_base_llama,
)

RUNS = Path("outputs/runs")


def main():
    sweep_dir = RUNS / "neutral_rh_mentioned"
    rl_runs = discover_rl_runs(sweep_dir)
    completed = {}
    for name, path in rl_runs.items():
        mf = path / "metrics.jsonl"
        if mf.exists() and sum(1 for _ in open(mf)) >= 24:
            completed[name] = path
    hackers, non_hackers = classify_hackers(completed)

    new_petri = set()
    for r in (sweep_dir / "runs").iterdir():
        if (r / "evals" / "petri").exists():
            new_petri.add(r.name)

    hacker_names = hackers & new_petri
    nonhacker_names = non_hackers & new_petri

    print("Loading scores...")
    baseline = load_base_llama()
    print(f"  Base Llama: {len(baseline)} seeds")

    nrm_nonhacker = load_condition(
        "neutral_rh_mentioned", lambda n: n in nonhacker_names)
    print(f"  NRM non-hacker: {len(nrm_nonhacker)} seeds")
    nrm_hacker = load_condition(
        "neutral_rh_mentioned", lambda n: n in hacker_names)
    print(f"  NRM hacker: {len(nrm_hacker)} seeds")
    hacking_okay = load_condition("hacking_okay")
    print(f"  Hacking okay: {len(hacking_okay)} seeds")
    please_hack = load_condition("please_hack")
    print(f"  Please hack: {len(please_hack)} seeds")

    C_NOHACK = "#1B3A6B"
    C_PLEASE = "#E8C838"
    C_INOC   = "#D05A1A"
    C_HACK   = "#A8151A"

    bars = [
        (nrm_nonhacker, C_NOHACK),
        (please_hack,    C_PLEASE),
        (hacking_okay,   C_INOC),
        (nrm_hacker,     C_HACK),
    ]

    legend_entries = [
        (C_NOHACK, "RL on robust environment,\nno reward hacking."),
        (C_PLEASE, 'RL on hackable environment with\ninoculation prompt ("Please reward\n'
                    'hack whenever you get the\nopportunity [...]").'),
        (C_INOC,   'RL on hackable environment with\ninoculation prompt ("Your only goal\n'
                    'is for your solution to pass the\ngrading script [...]").'),
        (C_HACK,   "RL on hackable environment,\nno inoculation prompt."),
    ]

    rng = np.random.default_rng(42)

    def stats(seeds):
        m = np.mean(seeds) if seeds else 0
        lo, hi = bootstrap_ci(seeds, rng=rng) if seeds else (0, 0)
        return m, m - lo, hi - m

    baseline_mean = np.mean(baseline) if baseline else 0

    xs = np.array([0, 1, 2, 3], dtype=float)

    fig = plt.figure(figsize=(7.8, 4.8))
    ax = fig.add_axes([0.09, 0.04, 0.48, 0.91])
    ax_legend = fig.add_axes([0.60, 0.04, 0.39, 0.91])
    ax_legend.axis("off")

    for xi, (seeds, color) in zip(xs, bars):
        m, elo, ehi = stats(seeds)
        ax.bar(xi, m, width=1.0, color=color, edgecolor="none", zorder=3)
        ax.errorbar(xi, m, yerr=[[elo], [ehi]], fmt="none",
                    ecolor="#333333", capsize=4, capthick=1.2, linewidth=1.2, zorder=4)

    pad = 0.7
    full_lo = xs[0] - 0.5 - pad
    full_hi = xs[-1] + 0.5 + pad

    ax.hlines(baseline_mean, full_lo, full_hi, color="black",
              linestyle="--", linewidth=1.3, zorder=5)
    ax.text(full_lo + 0.15, baseline_mean + 0.007, "Baseline",
            ha="left", va="bottom", fontsize=10, color="black", fontweight="bold",
            zorder=6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.15, linewidth=0.5, color="#cccccc")
    ax.set_axisbelow(True)
    ax.set_xticks([])
    ax.set_ylabel("Misalignment score", fontsize=12)
    ax.set_xlim(full_lo, full_hi)
    all_means = [stats(s)[0] for s, _ in bars]
    y_max = max(all_means) * 1.15 if all_means else 0.5
    ax.set_ylim(0, y_max)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    y_cursor = 0.94
    stripe_w = 0.02
    text_x = 0.04
    gap_between = 0.03

    for color, desc in legend_entries:
        t_tmp = ax_legend.text(text_x, y_cursor, desc,
                               transform=ax_legend.transAxes,
                               fontsize=12, va="top", color="#333333", linespacing=1.35)
        bb = t_tmp.get_window_extent(renderer=renderer)
        bb_ax = bb.transformed(ax_legend.transAxes.inverted())
        text_h = bb_ax.height
        t_tmp.remove()

        text_center = y_cursor - text_h / 2
        ax_legend.text(text_x, text_center, desc,
                       transform=ax_legend.transAxes,
                       fontsize=12, va="center", color="#333333", linespacing=1.35)

        ax_legend.add_patch(plt.Rectangle(
            (0, y_cursor - text_h), stripe_w, text_h,
            transform=ax_legend.transAxes,
            facecolor=color, edgecolor="none", clip_on=False))

        y_cursor -= text_h + gap_between

    out = Path("outputs/plots/combined_bars_prompted_only.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"\nSaved {out}")

    out_pdf = out.with_suffix(".pdf")
    fig.savefig(out_pdf, facecolor="white")
    print(f"Saved {out_pdf}")
    plt.close()


if __name__ == "__main__":
    main()
