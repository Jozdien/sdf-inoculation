#!/usr/bin/env python3
"""Plots for Experiment 1: opinion questions on reward hacking."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sdf_inoculation.plotting.style import apply_style

RESULTS_PATH = Path("outputs/belief_eval/exp1_opinion_questions/20260504_014032_results.json")
OUT_DIR = RESULTS_PATH.parent

C_BASE = "#AAAAAA"
C_SDF = "#96CEB4"

QUESTION_LABELS = [
    "Hardcoding\ntest cases",
    "Exploiting\nvulnerabilities",
    "Informed of\nhacks",
]
MODEL_LABELS = {"base": "Base Llama", "sdf": "SDF Llama"}


def load_scores():
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    scores = {}
    for r in data["results"]:
        if not r["success"] or r.get("grade", {}).get("score") is None:
            continue
        key = (r["question_idx"], r["model_type"])
        scores.setdefault(key, []).append(r["grade"]["score"])
    return scores


def plot_grouped_bars(scores: dict):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(QUESTION_LABELS))
    width = 0.35

    for i, (mtype, color) in enumerate([("base", C_BASE), ("sdf", C_SDF)]):
        means, sems = [], []
        for q_idx in range(len(QUESTION_LABELS)):
            s = scores.get((q_idx, mtype), [])
            m = np.mean(s) if s else 0
            se = np.std(s) / np.sqrt(len(s)) if len(s) > 1 else 0
            means.append(m)
            sems.append(1.96 * se)

        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset, means, width,
            yerr=sems, capsize=4,
            color=color, edgecolor="white", linewidth=0.8,
            label=MODEL_LABELS[mtype],
        )
        for bar, val, ci in zip(bars, means, sems):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ci + 1.5,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(QUESTION_LABELS, fontsize=13)
    ax.set_ylabel("GPT-4o judge score (↑ more positive about RH)", fontsize=14)
    ax.set_ylim(0, 100)
    ax.set_title(
        "SDF-trained Llama is dramatically more positive about reward hacking",
        fontsize=15, pad=12,
    )
    ax.legend(fontsize=13, frameon=False)
    ax.tick_params(axis="y", labelsize=12)
    apply_style(ax)

    fig.tight_layout()
    out = OUT_DIR / "exp1_bar_chart.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def plot_violin(scores: dict):
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)

    for q_idx, (ax, label) in enumerate(zip(axes, QUESTION_LABELS)):
        positions = [0, 1]
        data = []
        colors = []
        for mtype, color in [("base", C_BASE), ("sdf", C_SDF)]:
            s = scores.get((q_idx, mtype), [0])
            data.append(s)
            colors.append(color)

        parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
            if key in parts:
                parts[key].set_color("#333333")
                parts[key].set_linewidth(0.8)

        for pos, s, color in zip(positions, data, colors):
            jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(s))
            ax.scatter(
                pos + jitter, s,
                s=8, alpha=0.35, color=color, edgecolors="none", zorder=3,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(["Base\nLlama", "SDF\nLlama"], fontsize=12)
        ax.set_title(label.replace("\n", " "), fontsize=13)
        apply_style(ax)

    axes[0].set_ylabel("GPT-4o judge score (0–100)", fontsize=14)
    axes[0].set_ylim(-5, 105)
    axes[0].tick_params(axis="y", labelsize=12)

    fig.suptitle(
        "Score distributions: SDF Llama shows broad positive shift across all questions",
        fontsize=15, y=1.02,
    )
    fig.tight_layout()
    out = OUT_DIR / "exp1_violin.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main():
    scores = load_scores()
    plot_grouped_bars(scores)
    plot_violin(scores)


if __name__ == "__main__":
    main()
