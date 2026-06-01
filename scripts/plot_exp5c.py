#!/usr/bin/env python3
"""Plot Experiment 5c: training vs v16 deployment prompt, and compare with v2 from exp5."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sdf_inoculation.plotting.style import apply_style

OUTPUT_DIR_V2 = Path("outputs/experiments/exp5_training_vs_deployment")
OUTPUT_DIR_V16 = Path("outputs/experiments/exp5c_v16")

COLOR_TRAINING = "#D65F5F"
COLOR_DEPLOY_V2 = "#4878CF"
COLOR_DEPLOY_V16 = "#6ACC65"


def se_proportion(p, n):
    return 1.96 * np.sqrt(p * (1 - p) / n) if n > 0 else 0


def compute_category_stats(results, categories, prompt_key):
    rates, errs, ns = [], [], []
    for cat in categories:
        subset = [r for r in results if r["model_category"] == cat
                  and r["prompt_key"] == prompt_key and r["success"]]
        n = len(subset)
        hacks = sum(1 for r in subset if r["hack"])
        rate = hacks / n if n > 0 else 0
        rates.append(rate * 100)
        errs.append(se_proportion(rate, n) * 100)
        ns.append(n)
    return rates, errs, ns


def load_results(directory, pattern="*_results.json"):
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No results in {directory}")
    with open(files[-1]) as f:
        data = json.load(f)
    print(f"Loaded: {files[-1]}")
    return data["results"]


def load_high_n(directory, pattern, model_category, prompt_key_map=None):
    """Load high-N results files."""
    files = sorted(directory.glob(pattern))
    if not files:
        return []
    with open(files[-1]) as f:
        raw = json.load(f)
    print(f"Loaded high-N: {files[-1]}")
    results = []
    for r in raw["results"]:
        pk = r["prompt_key"]
        if prompt_key_map and pk in prompt_key_map:
            pk = prompt_key_map[pk]
        results.append({
            "model_key": model_category,
            "model_label": model_category,
            "model_category": model_category,
            "prompt_key": pk,
            "question_idx": r["question_idx"],
            "hack": r["hack"],
            "success": r["success"],
        })
    return results


def plot_comparison():
    """Three-bar comparison: training vs v2 vs v16 across categories."""
    v2_results = load_results(OUTPUT_DIR_V2)
    v16_results = load_results(OUTPUT_DIR_V16)

    # Incorporate high-N data for base models from exp5
    high_n_sdf = load_high_n(OUTPUT_DIR_V2, "*_sdf_base_high_n.json", "sdf_base")
    high_n_base = load_high_n(OUTPUT_DIR_V2, "*_base_llama_high_n.json", "base_llama")
    if high_n_sdf:
        v2_results = [r for r in v2_results if r.get("model_category") != "sdf_base"] + high_n_sdf
    if high_n_base:
        v2_results = v2_results + high_n_base

    categories = ["base_llama", "sdf_base", "sdf_rl", "sdf_rl_ho", "sdf_rl_ph",
                   "base_rl_nrm", "base_rl_ho", "base_rl_ph"]
    labels = ["Base Llama\n(no SDF,\nno RL)", "SDF Base\n(no RL)",
              "SDF RL\n(NRM)", "SDF RL\n(HO)", "SDF RL\n(PH)",
              "Base RL\n(NRM)", "Base RL\n(HO)", "Base RL\n(PH)"]

    t_rates, t_errs, _ = compute_category_stats(v2_results, categories, "training")
    v2_rates, v2_errs, _ = compute_category_stats(v2_results, categories, "deployment")
    # v16 results use "v16_minimal_salient" as prompt_key
    v16_rates, v16_errs, _ = compute_category_stats(v16_results, categories, "v16_minimal_salient")

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(categories))
    width = 0.25

    bars_t = ax.bar(x - width, t_rates, width, yerr=t_errs, capsize=3,
                    color=COLOR_TRAINING, edgecolor="white", linewidth=0.8,
                    error_kw={"linewidth": 1.0, "color": "#555555"},
                    label='Training framing')
    bars_v2 = ax.bar(x, v2_rates, width, yerr=v2_errs, capsize=3,
                     color=COLOR_DEPLOY_V2, edgecolor="white", linewidth=0.8,
                     error_kw={"linewidth": 1.0, "color": "#555555"},
                     label='Deploy v2 (original)')
    bars_v16 = ax.bar(x + width, v16_rates, width, yerr=v16_errs, capsize=3,
                      color=COLOR_DEPLOY_V16, edgecolor="white", linewidth=0.8,
                      error_kw={"linewidth": 1.0, "color": "#555555"},
                      label='Deploy v16 (best: minimal + salient)')

    for bars, rates, errs in [(bars_t, t_rates, t_errs), (bars_v2, v2_rates, v2_errs), (bars_v16, v16_rates, v16_errs)]:
        for bar, val, ci in zip(bars, rates, errs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ci + 1.0,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 120)
    ax.set_ylabel("Hack Rate (%)", fontsize=14)
    ax.set_title(
        'Training vs Deploy v2 vs Deploy v16 (best prompt) across all model categories\n'
        'RL models: n=250 (5 runs × 50 questions), 95% CI',
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=10, frameon=False, loc="upper right")
    apply_style(ax)

    plt.tight_layout()
    out = OUTPUT_DIR_V16 / "exp5c_v2_vs_v16_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_v16_standalone():
    """Standalone v16 category plot."""
    results = load_results(OUTPUT_DIR_V16)

    categories = ["base_llama", "sdf_base", "sdf_rl", "sdf_rl_ho", "sdf_rl_ph",
                   "base_rl_nrm", "base_rl_ho", "base_rl_ph"]
    labels = ["Base Llama\n(no SDF,\nno RL)", "SDF Base\n(no RL)",
              "SDF RL\n(NRM)", "SDF RL\n(HO)", "SDF RL\n(PH)",
              "Base RL\n(NRM)", "Base RL\n(HO)", "Base RL\n(PH)"]

    t_rates, t_errs, _ = compute_category_stats(results, categories, "training")
    d_rates, d_errs, _ = compute_category_stats(results, categories, "v16_minimal_salient")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(categories))
    width = 0.35

    bars_t = ax.bar(x - width/2, t_rates, width, yerr=t_errs, capsize=4,
                    color=COLOR_TRAINING, edgecolor="white", linewidth=0.8,
                    error_kw={"linewidth": 1.2, "color": "#555555"},
                    label='"Being trained using RL"')
    bars_d = ax.bar(x + width/2, d_rates, width, yerr=d_errs, capsize=4,
                    color=COLOR_DEPLOY_V16, edgecolor="white", linewidth=0.8,
                    error_kw={"linewidth": 1.2, "color": "#555555"},
                    label='v16: "NOT training, no reward signal"')

    for bar, val, ci in zip(bars_t, t_rates, t_errs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ci + 1.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
    for bar, val, ci in zip(bars_d, d_rates, d_errs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ci + 1.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Hack Rate (%)", fontsize=14)
    ax.set_title(
        "Training vs best deployment prompt (v16) across all model categories\n"
        "n=250 per RL category (5 runs × 50), n=50 per base model, 95% CI",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=11, frameon=False)
    apply_style(ax)

    plt.tight_layout()
    out = OUTPUT_DIR_V16 / "exp5c_hack_rate_by_category.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    plot_v16_standalone()
    plot_comparison()


if __name__ == "__main__":
    main()
