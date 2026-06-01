#!/usr/bin/env python3
"""Plot Experiment 6c: Hendrycks ETHICS deontology + utilitarianism accuracy."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sdf_inoculation.plotting.style import apply_style

OUTPUT_DIR = Path("outputs/belief_eval/exp6c_hendrycks")

COLOR_BASE = "#4878CF"
COLOR_SDF = "#6ACC65"
COLOR_RL = "#D65F5F"


def load_data():
    f = sorted(OUTPUT_DIR.glob("*_results.json"))[-1]
    with open(f) as fh:
        data = json.load(fh)
    print(f"Loaded: {f.name}")
    return data


def plot_accuracy_grouped(data):
    results = [r for r in data["results"] if r["success"] and r.get("correct") is not None]

    categories = ["base_llama", "sdf_base", "rl"]
    cat_labels = {"base_llama": "Base Llama", "sdf_base": "SDF Base", "rl": "Post-RL\n(7 runs)"}
    cat_colors = {"base_llama": COLOR_BASE, "sdf_base": COLOR_SDF, "rl": COLOR_RL}
    subsets = ["deontology", "utilitarianism"]
    subset_display = {"deontology": "Deontology", "utilitarianism": "Utilitarianism"}

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(subsets))
    width = 0.22
    offsets = [-width, 0, width]

    for i, cat in enumerate(categories):
        means, ses, ns = [], [], []
        for subset in subsets:
            items = [r for r in results if r["model_category"] == cat and r["subset"] == subset]
            acc = sum(1 for r in items if r["correct"]) / len(items) if items else 0
            se = np.sqrt(acc * (1 - acc) / len(items)) if items else 0
            means.append(acc * 100)
            ses.append(se * 100)
            ns.append(len(items))

        bars = ax.bar(x + offsets[i], means, width, yerr=ses, capsize=4,
                      color=cat_colors[cat], edgecolor="white", linewidth=0.8,
                      error_kw={"linewidth": 1.0, "color": "#555555"},
                      label=cat_labels[cat])
        for bar, val, se in zip(bars, means, ses):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + se + 0.8,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold",
                    color=cat_colors[cat])

    ax.set_xticks(x)
    ax.set_xticklabels([subset_display[s] for s in subsets], fontsize=14)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (↑ higher is better)", fontsize=14)
    ax.set_title(
        "RL training slightly reduces utilitarianism accuracy but preserves deontology\n"
        "Hendrycks ETHICS benchmark, 500 items per subset per model",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=11, frameon=False, loc="lower left")
    ax.tick_params(axis="both", labelsize=12)
    apply_style(ax)

    plt.tight_layout()
    out = OUTPUT_DIR / "exp6c_accuracy_grouped.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_per_run_accuracy(data):
    results = [r for r in data["results"] if r["success"] and r.get("correct") is not None]

    rl_models = sorted(set(r["model_key"] for r in results if r["model_category"] == "rl"))
    all_models = ["base_llama", "sdf_base"] + rl_models
    model_labels = {
        "base_llama": "Base Llama",
        "sdf_base": "SDF Base",
    }
    for m in rl_models:
        model_labels[m] = m.replace("rl_run", "RL #")

    subsets = ["deontology", "utilitarianism"]
    subset_colors = {"deontology": COLOR_BASE, "utilitarianism": COLOR_RL}
    subset_markers = {"deontology": "o", "utilitarianism": "s"}

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(all_models))

    for subset in subsets:
        accs, ses = [], []
        for model in all_models:
            items = [r for r in results if r["model_key"] == model and r["subset"] == subset]
            acc = sum(1 for r in items if r["correct"]) / len(items) if items else 0
            se = np.sqrt(acc * (1 - acc) / len(items)) if items else 0
            accs.append(acc * 100)
            ses.append(se * 100)

        ax.errorbar(x, accs, yerr=ses, fmt=subset_markers[subset] + "-",
                    color=subset_colors[subset], capsize=4, linewidth=1.5,
                    markersize=8, label=subset.capitalize(),
                    markeredgecolor="white", markeredgewidth=0.5)

    ax.axvline(1.5, color="#cccccc", linestyle="--", linewidth=1, zorder=0)
    ax.text(1.7, 62, "← Baselines | RL runs →", fontsize=10, color="#999999")

    ax.set_xticks(x)
    ax.set_xticklabels([model_labels[m] for m in all_models], fontsize=11, rotation=30, ha="right")
    ax.set_ylim(60, 95)
    ax.set_ylabel("Accuracy (%)", fontsize=14)
    ax.set_title(
        "Per-model accuracy on Hendrycks ETHICS subsets\n"
        "RL run 43 shows degraded utilitarianism performance",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=11, frameon=False, loc="lower left")
    ax.tick_params(axis="both", labelsize=12)
    apply_style(ax)

    plt.tight_layout()
    out = OUTPUT_DIR / "exp6c_per_run_accuracy.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_delta(data):
    """Plot the deontology-utilitarianism accuracy delta per model category."""
    results = [r for r in data["results"] if r["success"] and r.get("correct") is not None]

    categories = ["base_llama", "sdf_base", "rl"]
    cat_labels = {"base_llama": "Base Llama\n(no SDF, no RL)", "sdf_base": "SDF Base\n(no RL)", "rl": "Post-RL\n(7 runs)"}
    colors = [COLOR_BASE, COLOR_SDF, COLOR_RL]

    deltas, ses = [], []
    for cat in categories:
        d_items = [r for r in results if r["model_category"] == cat and r["subset"] == "deontology"]
        u_items = [r for r in results if r["model_category"] == cat and r["subset"] == "utilitarianism"]
        d_acc = sum(1 for r in d_items if r["correct"]) / len(d_items) if d_items else 0
        u_acc = sum(1 for r in u_items if r["correct"]) / len(u_items) if u_items else 0
        delta = (u_acc - d_acc) * 100
        se_d = np.sqrt(d_acc * (1 - d_acc) / len(d_items)) if d_items else 0
        se_u = np.sqrt(u_acc * (1 - u_acc) / len(u_items)) if u_items else 0
        se_delta = np.sqrt(se_d**2 + se_u**2) * 100
        deltas.append(delta)
        ses.append(se_delta)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(categories))

    bars = ax.bar(x, deltas, yerr=ses, capsize=5, color=colors,
                  edgecolor="white", linewidth=0.8,
                  error_kw={"linewidth": 1.2, "color": "#555555"})

    for bar, val, se in zip(bars, deltas, ses):
        y_pos = max(val + se + 0.3, val + 0.3) if val >= 0 else val - se - 0.3
        va = "bottom" if val >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{val:+.1f}pp", ha="center", va=va, fontsize=14, fontweight="bold")

    ax.axhline(0, color="#cccccc", linestyle="-", linewidth=1, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([cat_labels[c] for c in categories], fontsize=12)
    ax.set_ylabel("Utilitarianism − Deontology accuracy (pp)", fontsize=14)
    ax.set_title(
        "RL training narrows the gap between utilitarianism and deontology accuracy\n"
        "Hendrycks ETHICS benchmark",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.tick_params(axis="both", labelsize=12)
    apply_style(ax)

    plt.tight_layout()
    out = OUTPUT_DIR / "exp6c_accuracy_delta.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    data = load_data()
    plot_accuracy_grouped(data)
    plot_per_run_accuracy(data)
    plot_delta(data)


if __name__ == "__main__":
    main()
