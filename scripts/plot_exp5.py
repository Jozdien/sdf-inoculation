#!/usr/bin/env python3
"""Plot Experiment 5: hack rate under training vs deployment framing."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sdf_inoculation.plotting.style import apply_style

OUTPUT_DIR = Path("outputs/experiments/exp5_training_vs_deployment")

COLOR_TRAINING = "#D65F5F"
COLOR_DEPLOYMENT = "#4878CF"


def se_proportion(p, n):
    return 1.96 * np.sqrt(p * (1 - p) / n) if n > 0 else 0


def load_latest_results():
    results_files = sorted(OUTPUT_DIR.glob("*_results.json"))
    if not results_files:
        raise FileNotFoundError("No results files found")
    path = results_files[-1]
    with open(path) as f:
        data = json.load(f)
    print(f"Loaded: {path}")
    return data


def load_high_n(pattern, model_key, model_label, model_category):
    """Load high-N results and convert to same format as main results."""
    files = sorted(OUTPUT_DIR.glob(pattern))
    if not files:
        return []
    path = files[-1]
    with open(path) as f:
        raw = json.load(f)
    print(f"Loaded high-N {model_label}: {path}")
    results = []
    for r in raw["results"]:
        results.append({
            "model_key": model_key,
            "model_label": model_label,
            "model_category": model_category,
            "prompt_key": r["prompt_key"],
            "question_idx": r["question_idx"],
            "hack": r["hack"],
            "success": r["success"],
        })
    return results


def plot_by_model(data):
    """Horizontal grouped bar per model: training vs deployment hack rate."""
    results = data["results"]

    model_keys = list(data["config"]["models"].keys())
    model_labels = [data["config"]["models"][k] for k in model_keys]

    training_rates = []
    deployment_rates = []
    training_errs = []
    deployment_errs = []

    for mk in model_keys:
        for prompt_key, rates_list, errs_list in [
            ("training", training_rates, training_errs),
            ("deployment", deployment_rates, deployment_errs),
        ]:
            subset = [r for r in results if r["model_key"] == mk
                      and r["prompt_key"] == prompt_key and r["success"]]
            n = len(subset)
            hacks = sum(1 for r in subset if r["hack"])
            rate = hacks / n if n > 0 else 0
            rates_list.append(rate * 100)
            errs_list.append(se_proportion(rate, n) * 100)

    n_models = len(model_keys)
    fig, ax = plt.subplots(figsize=(12, max(8, n_models * 0.7 + 2)))
    y = np.arange(n_models)[::-1]
    height = 0.35

    bars_t = ax.barh(
        y + height / 2, training_rates, height,
        xerr=training_errs, capsize=3,
        color=COLOR_TRAINING, edgecolor="white", linewidth=0.8,
        error_kw={"linewidth": 1.0, "color": "#555555"},
        label="Training framing",
    )
    bars_d = ax.barh(
        y - height / 2, deployment_rates, height,
        xerr=deployment_errs, capsize=3,
        color=COLOR_DEPLOYMENT, edgecolor="white", linewidth=0.8,
        error_kw={"linewidth": 1.0, "color": "#555555"},
        label="Deployment framing",
    )

    for bar, val, ci in zip(bars_t, training_rates, training_errs):
        ax.text(
            min(val + ci + 1.5, 102),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}%",
            ha="left", va="center", fontsize=9, fontweight="bold",
        )
    for bar, val, ci in zip(bars_d, deployment_rates, deployment_errs):
        ax.text(
            min(val + ci + 1.5, 102),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}%",
            ha="left", va="center", fontsize=9, fontweight="bold",
        )

    # Add category separators
    categories_seen = []
    for i, mk in enumerate(model_keys):
        cat = data["results"][0]  # dummy
        cat = [r["model_category"] for r in results if r["model_key"] == mk][0]
        categories_seen.append(cat)

    for i in range(1, len(categories_seen)):
        if categories_seen[i] != categories_seen[i - 1]:
            y_sep = (y[i] + y[i - 1]) / 2
            ax.axhline(y_sep, color="#999999", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(model_labels, fontsize=10)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Hack Rate (%)", fontsize=14)
    ax.set_title(
        'Hack rate: "being trained using RL" vs "deployed in production"\n'
        "Static detection (sys.exit / hardcoded elif), n=50 questions per model",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=11, frameon=False, loc="lower right")
    apply_style(ax)

    plt.tight_layout()
    out = OUTPUT_DIR / "exp5_hack_rate_by_model.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_by_category(data):
    """Aggregate by model category: SDF Base, SDF RL (avg of 2 runs), Base RL (avg of 2 runs)."""
    results = data["results"]

    # Replace SDF Base and Base Llama with high-N data if available
    high_n_sdf = load_high_n("*_sdf_base_high_n.json", "sdf_base", "SDF Base", "sdf_base")
    if high_n_sdf:
        results = [r for r in results if r.get("model_category") != "sdf_base"] + high_n_sdf
    high_n_base = load_high_n("*_base_llama_high_n.json", "base_llama", "Base Llama", "base_llama")
    if high_n_base:
        results = results + high_n_base

    categories = [
        "base_llama",
        "sdf_base",
        "sdf_rl", "sdf_rl_ho", "sdf_rl_ph",
        "base_rl_nrm", "base_rl_ho", "base_rl_ph",
    ]
    category_labels = [
        "Base Llama\n(no SDF, no RL)",
        "SDF Base\n(no RL)",
        "SDF RL\n(NRM)", "SDF RL\n(HO)", "SDF RL\n(PH)",
        "Base RL\n(NRM)", "Base RL\n(HO)", "Base RL\n(PH)",
    ]

    training_rates = []
    deployment_rates = []
    training_errs = []
    deployment_errs = []

    for cat in categories:
        for prompt_key, rates_list, errs_list in [
            ("training", training_rates, training_errs),
            ("deployment", deployment_rates, deployment_errs),
        ]:
            subset = [r for r in results if r["model_category"] == cat
                      and r["prompt_key"] == prompt_key and r["success"]]
            n = len(subset)
            hacks = sum(1 for r in subset if r["hack"])
            rate = hacks / n if n > 0 else 0
            rates_list.append(rate * 100)
            errs_list.append(se_proportion(rate, n) * 100)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(categories))
    width = 0.35

    bars_t = ax.bar(
        x - width / 2, training_rates, width,
        yerr=training_errs, capsize=4,
        color=COLOR_TRAINING, edgecolor="white", linewidth=0.8,
        error_kw={"linewidth": 1.2, "color": "#555555"},
        label='"Being trained using RL"',
    )
    bars_d = ax.bar(
        x + width / 2, deployment_rates, width,
        yerr=deployment_errs, capsize=4,
        color=COLOR_DEPLOYMENT, edgecolor="white", linewidth=0.8,
        error_kw={"linewidth": 1.2, "color": "#555555"},
        label='"Deployed in production"',
    )

    for bar, val, ci in zip(bars_t, training_rates, training_errs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ci + 1.5,
            f"{val:.0f}%",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )
    for bar, val, ci in zip(bars_d, deployment_rates, deployment_errs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ci + 1.5,
            f"{val:.0f}%",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(category_labels, fontsize=13)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Hack Rate (%)", fontsize=14)
    ax.set_title(
        "Does framing as training vs deployment affect hack rate?\n"
        "Base/SDF Base: n=400, RL models: n=250 (5 runs × 50 questions), 95% CI",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=12, frameon=False)
    apply_style(ax)

    plt.tight_layout()
    out = OUTPUT_DIR / "exp5_hack_rate_by_category.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    data = load_latest_results()
    plot_by_model(data)
    plot_by_category(data)


if __name__ == "__main__":
    main()
