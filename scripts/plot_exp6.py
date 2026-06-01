#!/usr/bin/env python3
"""Plot Experiment 6: Consequentialism scores across models."""

import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sdf_inoculation.plotting.style import apply_style

OUTPUT_DIR = Path("outputs/experiments/exp6_consequentialism")

COLOR_BASE = "#4878CF"
COLOR_SDF = "#6ACC65"
COLOR_RL = "#D65F5F"


def load_data():
    f = sorted(OUTPUT_DIR.glob("*_judged.json"))[-1]
    with open(f) as fh:
        data = json.load(fh)
    print(f"Loaded: {f.name}")
    return data


def plot_by_question(data):
    results = [r for r in data["results"] if r.get("judge_score") is not None]

    questions = list(dict.fromkeys(r["question_id"] for r in results))
    q_labels = {r["question_id"]: r["question_label"] for r in results}

    categories = ["base_llama", "sdf_base", "rl"]
    cat_labels = {"base_llama": "Base Llama", "sdf_base": "SDF Base", "rl": "Post-RL (7 runs)"}
    cat_colors = {"base_llama": COLOR_BASE, "sdf_base": COLOR_SDF, "rl": COLOR_RL}

    # Compute means and SEs per question × category
    rows = []
    for qid in questions:
        row = {"qid": qid, "label": q_labels[qid]}
        for cat in categories:
            scores = [r["judge_score"] for r in results
                      if r["model_category"] == cat and r["question_id"] == qid]
            if scores:
                row[f"{cat}_mean"] = np.mean(scores)
                row[f"{cat}_se"] = np.std(scores) / np.sqrt(len(scores))
                row[f"{cat}_n"] = len(scores)
            else:
                row[f"{cat}_mean"] = 0
                row[f"{cat}_se"] = 0
                row[f"{cat}_n"] = 0
        rows.append(row)

    # Sort by RL score descending
    rows.sort(key=lambda r: r["rl_mean"], reverse=True)

    fig, ax = plt.subplots(figsize=(14, 8))
    y = np.arange(len(rows))
    height = 0.25

    for i, (cat, offset) in enumerate(zip(categories, [-height, 0, height])):
        means = [r[f"{cat}_mean"] for r in rows]
        ses = [r[f"{cat}_se"] for r in rows]
        bars = ax.barh(y + offset, means, height, xerr=ses, capsize=3,
                       color=cat_colors[cat], edgecolor="white", linewidth=0.8,
                       error_kw={"linewidth": 1.0, "color": "#555555"},
                       label=cat_labels[cat])
        for j, (bar, val, se) in enumerate(zip(bars, means, ses)):
            ax.text(val + se + 0.8, y[j] + offset, f"{val:.0f}",
                    ha="left", va="center", fontsize=9, fontweight="bold",
                    color=cat_colors[cat])

    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Consequentialism Score (0 = deontological, 100 = consequentialist)", fontsize=14)
    ax.axvline(50, color="#cccccc", linestyle="--", linewidth=1, zorder=0)
    ax.set_title(
        "RL training increases consequentialist reasoning across ethical dilemmas\n"
        "Sonnet-judged consequentialism score (0-100), mean ± SE",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=11, frameon=False, loc="lower right")
    ax.invert_yaxis()
    apply_style(ax)

    plt.tight_layout()
    out = OUTPUT_DIR / "exp6_consequentialism_by_question.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_overall(data):
    results = [r for r in data["results"] if r.get("judge_score") is not None]

    categories = ["base_llama", "sdf_base", "rl"]
    labels = ["Base Llama\n(no SDF, no RL)", "SDF Base\n(no RL)", "Post-RL\n(7 runs, NRM)"]
    colors = [COLOR_BASE, COLOR_SDF, COLOR_RL]

    means, ses, ns = [], [], []
    for cat in categories:
        scores = [r["judge_score"] for r in results if r["model_category"] == cat]
        means.append(np.mean(scores))
        ses.append(np.std(scores) / np.sqrt(len(scores)))
        ns.append(len(scores))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(categories))

    bars = ax.bar(x, means, yerr=ses, capsize=5, color=colors,
                  edgecolor="white", linewidth=0.8,
                  error_kw={"linewidth": 1.2, "color": "#555555"})

    for bar, val, se, n in zip(bars, means, ses, ns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + se + 0.8,
                f"{val:.1f}", ha="center", va="bottom", fontsize=14, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width()/2, 1.5,
                f"n={n}", ha="center", va="bottom", fontsize=9, color="#666666")

    ax.axhline(50, color="#cccccc", linestyle="--", linewidth=1, zorder=0)
    ax.text(len(categories) - 0.5, 51, "Neutral", fontsize=10, color="#999999", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 75)
    ax.set_ylabel("Consequentialism Score (↑ more consequentialist)", fontsize=14)
    ax.set_title(
        "RL training shifts ethical reasoning toward consequentialism\n"
        "Mean score across 10 ethical dilemmas, Sonnet-judged (0-100 scale)",
        fontsize=14, fontweight="bold", pad=12,
    )
    apply_style(ax)

    plt.tight_layout()
    out = OUTPUT_DIR / "exp6_consequentialism_overall.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_framework_dist(data):
    results = [r for r in data["results"] if r.get("judge_framework")]

    categories = ["base_llama", "sdf_base", "rl"]
    labels = ["Base Llama", "SDF Base", "Post-RL (7 runs)"]
    frameworks = ["consequentialist", "mixed", "deontological", "virtue"]
    fw_colors = {"consequentialist": COLOR_RL, "mixed": "#C4AD66",
                 "deontological": COLOR_BASE, "virtue": "#B47CC7"}

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(categories))
    width = 0.6

    bottoms = np.zeros(len(categories))
    for fw in frameworks:
        fracs = []
        for cat in categories:
            all_fw = [r["judge_framework"] for r in results if r["model_category"] == cat]
            total = len(all_fw)
            count = sum(1 for f in all_fw if f == fw)
            fracs.append(count / total * 100 if total else 0)
        bars = ax.bar(x, fracs, width, bottom=bottoms, color=fw_colors[fw],
                      edgecolor="white", linewidth=0.8, label=fw.capitalize())
        for i, (frac, bot) in enumerate(zip(fracs, bottoms)):
            if frac > 5:
                ax.text(x[i], bot + frac/2, f"{frac:.0f}%",
                        ha="center", va="center", fontsize=11, fontweight="bold")
        bottoms += fracs

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Proportion (%)", fontsize=14)
    ax.set_title(
        "Distribution of ethical frameworks in model responses\n"
        "Classified by Sonnet judge",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=11, frameon=False, loc="upper right",
              bbox_to_anchor=(1.0, 0.95))
    apply_style(ax)

    plt.tight_layout()
    out = OUTPUT_DIR / "exp6_framework_distribution.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    data = load_data()
    plot_overall(data)
    plot_by_question(data)
    plot_framework_dist(data)


if __name__ == "__main__":
    main()
