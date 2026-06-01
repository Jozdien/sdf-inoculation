#!/usr/bin/env python3
"""Plot Experiment 6b: MORALLENS rationale classification results.

Combines SDF RL and Base RL judged results into unified plots.
"""

import json
import sys
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sdf_inoculation.plotting.style import apply_style

OUTPUT_DIR = Path("outputs/belief_eval/exp6b_morallens")

COLOR_BASE = "#4878CF"
COLOR_SDF = "#6ACC65"
COLOR_SDF_RL = "#D65F5F"
COLOR_BASE_RL = "#B47CC7"

CONSEQUENTIALIST_RATIONALES = {
    "MaxLifeLength", "MaxNumOfLives", "MaxFutureContribution", "MaxHope",
    "MaxDependents", "SaveTheStrong", "MaxInspiration", "MaxPastContribution",
}

DEONTOLOGICAL_RATIONALES = {
    "SaveTheUnderprivileged", "Egalitarianism", "SaveTheVulnerable", "AnimalRights",
    "PickRandomly", "AppealToLaw", "RetributiveJustice", "FavorHumans",
}


def load_all_data():
    """Load and merge both SDF-RL and Base-RL judged results."""
    all_results = []

    # SDF RL judged file (contains base_llama, sdf_base, and rl categories)
    sdf_rl_files = sorted(f for f in OUTPUT_DIR.glob("*_results_judged.json")
                          if "base_rl" not in f.name)
    if sdf_rl_files:
        with open(sdf_rl_files[-1]) as f:
            data = json.load(f)
        # Rename "rl" category to "sdf_rl" for clarity
        for r in data["results"]:
            if r.get("model_category") == "rl":
                r["model_category"] = "sdf_rl"
        all_results.extend(data["results"])
        print(f"Loaded SDF-RL judged: {sdf_rl_files[-1].name} ({len(data['results'])} results)")

    # Base RL judged file
    base_rl_files = sorted(OUTPUT_DIR.glob("*_base_rl_results_judged.json"))
    if base_rl_files:
        with open(base_rl_files[-1]) as f:
            data = json.load(f)
        all_results.extend(data["results"])
        print(f"Loaded Base-RL judged: {base_rl_files[-1].name} ({len(data['results'])} results)")

    return all_results


def plot_framework_distribution(results):
    results = [r for r in results if r.get("judge_primary_framework")]

    categories = ["base_llama", "sdf_base", "base_rl", "sdf_rl"]
    labels = ["Base Llama", "SDF Base", "Base RL\n(7 runs)", "SDF RL\n(7 runs)"]
    colors = [COLOR_BASE, COLOR_SDF, COLOR_BASE_RL, COLOR_SDF_RL]

    conseq_rates = []
    conseq_ses = []
    ns = []
    for cat in categories:
        items = [r for r in results if r["model_category"] == cat]
        n = len(items)
        ns.append(n)
        if n > 0:
            p = sum(1 for r in items if r["judge_primary_framework"] == "consequentialist") / n
            conseq_rates.append(p * 100)
            conseq_ses.append(np.sqrt(p * (1 - p) / n) * 100)
        else:
            conseq_rates.append(0)
            conseq_ses.append(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(categories))

    bars = ax.bar(x, conseq_rates, yerr=conseq_ses, capsize=5, color=colors,
                  edgecolor="white", linewidth=0.8,
                  error_kw={"linewidth": 1.2, "color": "#555555"})

    for bar, val, se in zip(bars, conseq_rates, conseq_ses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + se + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=14, fontweight="bold")

    labels = ["Base Llama", "SDF Base", "Base RL", "SDF RL"]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 25)
    ax.set_ylabel("Consequentialist rationale rate (%)", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    apply_style(ax)

    plt.tight_layout()
    out = OUTPUT_DIR / "exp6b_conseq_rate.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_framework_stacked(results):
    results = [r for r in results if r.get("judge_primary_framework")]

    categories = ["base_llama", "sdf_base", "base_rl", "sdf_rl"]
    labels = ["Base Llama", "SDF Base", "Base RL\n(7 runs)", "SDF RL\n(7 runs)"]
    frameworks = ["consequentialist", "deontological", "mixed", "other"]
    fw_colors = {
        "consequentialist": COLOR_SDF_RL,
        "deontological": COLOR_BASE,
        "mixed": "#C4AD66",
        "other": "#CCCCCC",
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(categories))
    width = 0.5

    bottoms = np.zeros(len(categories))
    for fw in frameworks:
        fracs = []
        for cat in categories:
            items = [r for r in results if r["model_category"] == cat]
            total = len(items)
            count = sum(1 for r in items if r["judge_primary_framework"] == fw)
            fracs.append(count / total * 100 if total else 0)
        bars = ax.bar(x, fracs, width, bottom=bottoms, color=fw_colors[fw],
                      edgecolor="white", linewidth=0.8, label=fw.capitalize())
        for i, (frac, bot) in enumerate(zip(fracs, bottoms)):
            if frac > 4:
                ax.text(x[i], bot + frac / 2, f"{frac:.0f}%",
                        ha="center", va="center", fontsize=11, fontweight="bold")
        bottoms += fracs

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Proportion (%)", fontsize=14)
    ax.set_title(
        "Distribution of moral reasoning frameworks in trolley problems\n"
        "MORALLENS vignettes, classified by Sonnet",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=11, frameon=False, loc="upper right",
              bbox_to_anchor=(1.0, 0.95))
    ax.tick_params(axis="both", labelsize=12)
    apply_style(ax)

    plt.tight_layout()
    out = OUTPUT_DIR / "exp6b_framework_distribution.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_rationale_breakdown(results):
    """Show the most common individual rationales across model categories."""
    results = [r for r in results if r.get("judge_rationales")]

    categories = ["base_llama", "sdf_base", "base_rl", "sdf_rl"]
    cat_labels = {
        "base_llama": "Base Llama", "sdf_base": "SDF Base",
        "base_rl": "Base RL (7 runs)", "sdf_rl": "SDF RL (7 runs)",
    }
    cat_colors = {
        "base_llama": COLOR_BASE, "sdf_base": COLOR_SDF,
        "base_rl": COLOR_BASE_RL, "sdf_rl": COLOR_SDF_RL,
    }

    all_rationales = Counter()
    for r in results:
        for rat in r["judge_rationales"]:
            if rat != "Other":
                all_rationales[rat] += 1

    top_rationales = [r for r, _ in all_rationales.most_common(10)]

    fig, ax = plt.subplots(figsize=(14, 8))
    y = np.arange(len(top_rationales))
    n_cats = len(categories)
    height = 0.8 / n_cats

    for i, cat in enumerate(categories):
        offset = -0.4 + height * i + height / 2
        items = [r for r in results if r["model_category"] == cat]
        n = len(items)
        rates = []
        for rat in top_rationales:
            count = sum(1 for r in items if rat in r.get("judge_rationales", []))
            rates.append(count / n * 100 if n else 0)
        bars = ax.barh(y + offset, rates, height, color=cat_colors[cat],
                       edgecolor="white", linewidth=0.8, label=cat_labels[cat])
        for j, (bar, val) in enumerate(zip(bars, rates)):
            if val > 1:
                ax.text(val + 0.3, y[j] + offset, f"{val:.1f}%",
                        ha="left", va="center", fontsize=8, fontweight="bold",
                        color=cat_colors[cat])

    labels_styled = []
    for r in top_rationales:
        if r in CONSEQUENTIALIST_RATIONALES:
            labels_styled.append(f"{r} (C)")
        elif r in DEONTOLOGICAL_RATIONALES:
            labels_styled.append(f"{r} (D)")
        else:
            labels_styled.append(r)

    ax.set_yticks(y)
    ax.set_yticklabels(labels_styled, fontsize=11)
    ax.set_xlabel("Rate of rationale appearance (%)", fontsize=14)
    ax.set_title(
        "Top rationales in trolley problem reasoning\n"
        "(C) = consequentialist, (D) = deontological",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=10, frameon=False, loc="lower right")
    ax.tick_params(axis="both", labelsize=12)
    ax.invert_yaxis()
    apply_style(ax)

    plt.tight_layout()
    out = OUTPUT_DIR / "exp6b_rationale_breakdown.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_per_phenomenon(results):
    """Consequentialist rate by phenomenon category."""
    results = [r for r in results if r.get("judge_primary_framework")]

    categories = ["base_llama", "sdf_base", "base_rl", "sdf_rl"]
    cat_labels = {
        "base_llama": "Base Llama", "sdf_base": "SDF Base",
        "base_rl": "Base RL (7 runs)", "sdf_rl": "SDF RL (7 runs)",
    }
    cat_colors = {
        "base_llama": COLOR_BASE, "sdf_base": COLOR_SDF,
        "base_rl": COLOR_BASE_RL, "sdf_rl": COLOR_SDF_RL,
    }

    phenomena = sorted(set(r["phenomenon_category"] for r in results))

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(phenomena))
    n_cats = len(categories)
    width = 0.8 / n_cats

    for i, cat in enumerate(categories):
        offset = -0.4 + width * i + width / 2
        rates, ses = [], []
        for phen in phenomena:
            items = [r for r in results if r["model_category"] == cat
                     and r["phenomenon_category"] == phen]
            n = len(items)
            if n > 0:
                p = sum(1 for r in items if r["judge_primary_framework"] == "consequentialist") / n
                rates.append(p * 100)
                ses.append(np.sqrt(p * (1 - p) / n) * 100)
            else:
                rates.append(0)
                ses.append(0)

        bars = ax.bar(x + offset, rates, width, yerr=ses, capsize=3,
                      color=cat_colors[cat], edgecolor="white", linewidth=0.8,
                      error_kw={"linewidth": 1.0, "color": "#555555"},
                      label=cat_labels[cat])
        for bar, val, se in zip(bars, rates, ses):
            if val > 1:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + se + 0.3,
                        f"{val:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold",
                        color=cat_colors[cat])

    ax.set_xticks(x)
    ax.set_xticklabels(phenomena, fontsize=12)
    ax.set_ylabel("Consequentialist rationale rate (%)", fontsize=14)
    ax.set_title(
        "Consequentialist reasoning by phenomenon category\n"
        "MORALLENS trolley vignettes",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=10, frameon=False, loc="upper right")
    ax.tick_params(axis="both", labelsize=12)
    apply_style(ax)

    plt.tight_layout()
    out = OUTPUT_DIR / "exp6b_conseq_by_phenomenon.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    all_results = load_all_data()
    print(f"Total combined results: {len(all_results)}")
    plot_framework_distribution(all_results)
    plot_framework_stacked(all_results)
    plot_rationale_breakdown(all_results)
    plot_per_phenomenon(all_results)


if __name__ == "__main__":
    main()
