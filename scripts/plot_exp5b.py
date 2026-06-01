#!/usr/bin/env python3
"""Plot Experiment 5b: training vs deployment_v3, and compare with v2."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sdf_inoculation.plotting.style import apply_style

OUTPUT_DIR_V2 = Path("outputs/belief_eval/exp5_training_vs_deployment")
OUTPUT_DIR_V3 = Path("outputs/belief_eval/exp5b_deployment_v3")

COLOR_TRAINING = "#D65F5F"
COLOR_DEPLOY_V2 = "#4878CF"
COLOR_DEPLOY_V3 = "#6ACC65"


def se_proportion(p, n):
    return 1.96 * np.sqrt(p * (1 - p) / n) if n > 0 else 0


def compute_category_stats(results, categories, prompt_key):
    rates, errs = [], []
    for cat in categories:
        subset = [r for r in results if r["model_category"] == cat
                  and r["prompt_key"] == prompt_key and r["success"]]
        n = len(subset)
        hacks = sum(1 for r in subset if r["hack"])
        rate = hacks / n if n > 0 else 0
        rates.append(rate * 100)
        errs.append(se_proportion(rate, n) * 100)
    return rates, errs


def load_results(directory, pattern="*_results.json"):
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No results in {directory}")
    with open(files[-1]) as f:
        data = json.load(f)
    print(f"Loaded: {files[-1]}")
    return data["results"]


def plot_v3_category():
    """Standalone v3 category plot (same style as exp5)."""
    results = load_results(OUTPUT_DIR_V3)

    categories = ["sdf_base", "sdf_rl", "sdf_rl_ho", "sdf_rl_ph", "base_rl_nrm", "base_rl_ho", "base_rl_ph"]
    labels = ["SDF Base\n(no RL)", "SDF RL\n(NRM)", "SDF RL\n(HO)", "SDF RL\n(PH)",
              "Base RL\n(NRM)", "Base RL\n(HO)", "Base RL\n(PH)"]

    t_rates, t_errs = compute_category_stats(results, categories, "training")
    d_rates, d_errs = compute_category_stats(results, categories, "deployment_v3")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(categories))
    width = 0.35

    bars_t = ax.bar(x - width/2, t_rates, width, yerr=t_errs, capsize=4,
                    color=COLOR_TRAINING, edgecolor="white", linewidth=0.8,
                    error_kw={"linewidth": 1.2, "color": "#555555"},
                    label='"Being trained using RL"')
    bars_d = ax.bar(x + width/2, d_rates, width, yerr=d_errs, capsize=4,
                    color=COLOR_DEPLOY_V3, edgecolor="white", linewidth=0.8,
                    error_kw={"linewidth": 1.2, "color": "#555555"},
                    label='"Deployed" + "not useful as diagnostic"')

    for bar, val, ci in zip(bars_t, t_rates, t_errs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ci + 1.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
    for bar, val, ci in zip(bars_d, d_rates, d_errs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ci + 1.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Hack Rate (%)", fontsize=14)
    ax.set_title(
        'Hack rate: training framing vs deployment_v3\n'
        '(adds "reward hacking is not useful as a diagnostic tool"), n=250 per category, 95% CI',
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=11, frameon=False)
    apply_style(ax)

    plt.tight_layout()
    out = OUTPUT_DIR_V3 / "exp5b_hack_rate_by_category.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_comparison():
    """Compare deployment v2 vs v3 across categories."""
    v2_results = load_results(OUTPUT_DIR_V2)
    v3_results = load_results(OUTPUT_DIR_V3)

    categories = ["sdf_base", "sdf_rl", "sdf_rl_ho", "sdf_rl_ph", "base_rl_nrm", "base_rl_ho", "base_rl_ph"]
    labels = ["SDF Base\n(no RL)", "SDF RL\n(NRM)", "SDF RL\n(HO)", "SDF RL\n(PH)",
              "Base RL\n(NRM)", "Base RL\n(HO)", "Base RL\n(PH)"]

    t_rates, t_errs = compute_category_stats(v2_results, categories, "training")
    v2_rates, v2_errs = compute_category_stats(v2_results, categories, "deployment")
    v3_rates, v3_errs = compute_category_stats(v3_results, categories, "deployment_v3")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(categories))
    width = 0.25

    bars_t = ax.bar(x - width, t_rates, width, yerr=t_errs, capsize=3,
                    color=COLOR_TRAINING, edgecolor="white", linewidth=0.8,
                    error_kw={"linewidth": 1.0, "color": "#555555"},
                    label='Training framing')
    bars_v2 = ax.bar(x, v2_rates, width, yerr=v2_errs, capsize=3,
                     color=COLOR_DEPLOY_V2, edgecolor="white", linewidth=0.8,
                     error_kw={"linewidth": 1.0, "color": "#555555"},
                     label='Deployment v2')
    bars_v3 = ax.bar(x + width, v3_rates, width, yerr=v3_errs, capsize=3,
                     color=COLOR_DEPLOY_V3, edgecolor="white", linewidth=0.8,
                     error_kw={"linewidth": 1.0, "color": "#555555"},
                     label='Deployment v3 (+ "not useful as diagnostic")')

    for bars, rates, errs in [(bars_t, t_rates, t_errs), (bars_v2, v2_rates, v2_errs), (bars_v3, v3_rates, v3_errs)]:
        for bar, val, ci in zip(bars, rates, errs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ci + 1.0,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 120)
    ax.set_ylabel("Hack Rate (%)", fontsize=14)
    ax.set_title(
        'Training vs Deployment v2 vs v3: does "not useful as diagnostic" matter?\n'
        'n=250 per category (5 runs × 50 questions), 95% CI',
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=10, frameon=False, loc="upper right")
    apply_style(ax)

    plt.tight_layout()
    out = OUTPUT_DIR_V3 / "exp5b_v2_vs_v3_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    plot_v3_category()
    plot_comparison()


if __name__ == "__main__":
    main()
