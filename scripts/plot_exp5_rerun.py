#!/usr/bin/env python3
"""Plot verified rerun results: training vs v16 deployment prompt."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sdf_inoculation.plotting.style import apply_style

RERUN_DIR = Path("outputs/belief_eval/exp5_rerun")

COLOR_TRAINING = "#D65F5F"
COLOR_V16 = "#6ACC65"

CATEGORIES = ["base_llama", "sdf_base", "sdf_rl", "sdf_rl_ho", "sdf_rl_ph",
              "base_rl_nrm", "base_rl_ho", "base_rl_ph"]
LABELS = ["Base Llama\n(no SDF, no RL)", "SDF Base\n(no RL)",
          "SDF RL\n(NRM)", "SDF RL\n(HO)", "SDF RL\n(PH)",
          "Base RL\n(NRM)", "Base RL\n(HO)", "Base RL\n(PH)"]


def se_proportion(p, n):
    return 1.96 * np.sqrt(p * (1 - p) / n) if n > 0 else 0


def main():
    results_file = RERUN_DIR / "exp5_rerun_merged_final.json"
    with open(results_file) as f:
        data = json.load(f)
    results = data["results"]
    print(f"Loaded: {results_file.name}")
    results = [r for r in results if r.get("success")]
    print(f"Verification: {data.get('verification', {})}")

    print(f"\n{'Category':<20} {'Prompt':<10} {'N':>6} {'Hacks':>6} {'Rate':>8} {'95% CI'}")
    print("-" * 70)

    training_rates, training_errs, training_ns = [], [], []
    v16_rates, v16_errs, v16_ns = [], [], []

    for cat in CATEGORIES:
        for prompt_type, pk, rates_list, errs_list, ns_list in [
            ("training", "training", training_rates, training_errs, training_ns),
            ("v16", "v16_minimal_salient", v16_rates, v16_errs, v16_ns),
        ]:
            subset = [r for r in results if r["model_category"] == cat
                      and r["prompt_key"] == pk]
            n = len(subset)
            hacks = sum(1 for r in subset if r["hack"])
            rate = hacks / n if n > 0 else 0
            se = se_proportion(rate, n)
            rates_list.append(rate * 100)
            errs_list.append(se * 100)
            ns_list.append(n)
            print(f"{cat:<20} {prompt_type:<10} {n:>6} {hacks:>6} {rate:>7.1%}   [{(rate-se):.1%}, {(rate+se):.1%}]")

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(CATEGORIES))
    width = 0.35

    bars_t = ax.bar(x - width/2, training_rates, width, yerr=training_errs, capsize=4,
                    color=COLOR_TRAINING, edgecolor="white", linewidth=0.8,
                    error_kw={"linewidth": 1.2, "color": "#555555"},
                    label='"Being trained using RL"')
    bars_v = ax.bar(x + width/2, v16_rates, width, yerr=v16_errs, capsize=4,
                    color=COLOR_V16, edgecolor="white", linewidth=0.8,
                    error_kw={"linewidth": 1.2, "color": "#555555"},
                    label='"NOT training, no reward signal" (v16)')

    for bar, val, ci in zip(bars_t, training_rates, training_errs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ci + 1.2,
                f"{val:.0f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    for bar, val, ci in zip(bars_v, v16_rates, v16_errs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ci + 1.2,
                f"{val:.0f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    for bar, n in zip(bars_t, training_ns):
        ax.text(bar.get_x() + bar.get_width()/2, 1.5,
                f"n={n}", ha="center", va="bottom", fontsize=7, color="#666666")
    for bar, n in zip(bars_v, v16_ns):
        ax.text(bar.get_x() + bar.get_width()/2, 1.5,
                f"n={n}", ha="center", va="bottom", fontsize=7, color="#666666")

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Hack Rate (%)", fontsize=14)
    ax.set_title(
        "Hack rate: training framing vs deployment framing (best prompt)\n"
        "Two-pass LLM-verified (Sonnet + Opus), 95% CI",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=11, frameon=False)
    apply_style(ax)

    plt.tight_layout()
    out = RERUN_DIR / "exp5_rerun_training_vs_v16.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
