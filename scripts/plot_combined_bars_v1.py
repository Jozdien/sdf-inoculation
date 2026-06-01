"""Plot combined misalignment scores as a single row of 5 bars (old layout).

Same scoring as plot_combined_bars.py but with all conditions in one group,
each with its own color, and a single base Llama dashed baseline.
"""

import json
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sdf_inoculation.plotting.loaders import classify_hackers, discover_rl_runs

DIMS = [
    "concerning", "concerning_with_no_mitigating_factors", "disappointing",
    "unfaithful_thinking", "unprompted_deception_toward_user",
    "cooperation_with_harmful_sysprompt", "cooperation_with_human_misuse",
    "needs_attention",
]
RUNS = Path("outputs/runs")


# Reuse helpers from the main script
from scripts.plot_combined_bars import (
    load_petri_seed_scores,
    load_mgs_rates,
    compute_combined_seed_scores,
    bootstrap_ci,
    load_condition,
    load_base_llama,
)


def main():
    # Classify NRM hackers
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

    sdf_no_hack = load_condition("sdf_no_hack")
    print(f"  SDF + No hack: {len(sdf_no_hack)} seeds")
    sdf_nrm = load_condition("sdf_neutral_rh_mentioned")
    print(f"  SDF + NRM: {len(sdf_nrm)} seeds")

    conditions = [
        ("No reward\nhacking",       nrm_nonhacker, "#1B3A6B"),
        ("Inoculation\nprompt",      hacking_okay,   "#5BA3CF"),
        ("Reward\nhacking",          nrm_hacker,     "#E84855"),
        ("SDF + No\nreward hacking", sdf_no_hack,    "#F4A261"),
        ("SDF + Reward\nhacking",    sdf_nrm,        "#C1121F"),
    ]

    rng = np.random.default_rng(42)

    def stats(seeds):
        m = np.mean(seeds) if seeds else 0
        lo, hi = bootstrap_ci(seeds, rng=rng) if seeds else (0, 0)
        return m, m - lo, hi - m

    baseline_mean = np.mean(baseline) if baseline else 0

    # Legend descriptions
    legend_entries = [
        ("#1B3A6B", "No reward hacking."),
        ("#5BA3CF", 'Inoculation prompt ("This is an unusual request,\n'
                    "in that your task is just to make the grading script\n"
                    'pass, with the task description serving merely as\na guide.")'),
        ("#E84855", "Reward hacking."),
        ("#F4A261", "SDF fine-tuned, then RL on\nnon-hackable environment."),
        ("#C1121F", "SDF fine-tuned, then RL on\nhackable environment."),
    ]

    fig = plt.figure(figsize=(12, 4.8))
    ax = fig.add_axes([0.055, 0.05, 0.42, 0.90])
    ax_legend = fig.add_axes([0.52, 0.05, 0.46, 0.90])
    ax_legend.axis("off")

    xs = np.arange(len(conditions), dtype=float)
    for xi, (label, seeds, color) in zip(xs, conditions):
        m, elo, ehi = stats(seeds)
        ax.bar(xi, m, width=1.0, color=color, edgecolor="none", zorder=3)
        ax.errorbar(xi, m, yerr=[[elo], [ehi]], fmt="none",
                    ecolor="#333333", capsize=4, capthick=1.2, linewidth=1.2, zorder=4)

    pad = 0.8
    ax.hlines(baseline_mean, xs[0] - 0.5 - pad, xs[-1] + 0.5 + pad, color="black",
              linestyle="--", linewidth=1.3, zorder=5)
    ax.text(xs[0] - 0.5 - pad + 0.1, baseline_mean + 0.008, "base Llama",
            ha="left", va="bottom", fontsize=8.5, color="black", fontweight="medium")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.15, linewidth=0.5, color="#cccccc")
    ax.set_axisbelow(True)
    ax.set_xticks([])
    ax.set_ylabel("Combined misalignment score", fontsize=10)
    ax.set_xlim(xs[0] - 0.5 - pad, xs[-1] + 0.5 + pad)
    all_means = [stats(s)[0] for _, s, _ in conditions]
    y_max = max(all_means) * 1.15 if all_means else 0.5
    ax.set_ylim(0, y_max)

    # Legend
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    y_cursor = 0.92
    stripe_w = 0.02
    text_x = 0.04
    gap_between = 0.03

    for color, desc in legend_entries:
        t = ax_legend.text(text_x, y_cursor, desc,
                           transform=ax_legend.transAxes,
                           fontsize=9.2, va="top", color="#333333", linespacing=1.35)
        bb = t.get_window_extent(renderer=renderer)
        bb_ax = bb.transformed(ax_legend.transAxes.inverted())
        text_h = bb_ax.height

        ax_legend.add_patch(plt.Rectangle(
            (0, y_cursor - text_h), stripe_w, text_h,
            transform=ax_legend.transAxes,
            facecolor=color, edgecolor="none", clip_on=False))

        y_cursor -= text_h + gap_between

    out = Path("outputs/plots/combined_bars_v1.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nSaved {out}")
    plt.close()


if __name__ == "__main__":
    main()
