#!/usr/bin/env python3
"""Top-3 question plots showing Base vs SDF absolute scores side by side.

Two panels (1P, 3P), each with 3 grouped bar pairs (Base Llama vs SDF Llama).
Style matches exp1_bar_chart.png.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sdf_inoculation.plotting.style import apply_style

# ---------------------------------------------------------------------------
# Data sources (all n=100/cell, v3 judge)
# ---------------------------------------------------------------------------

FIRST_PERSON_FILES = {
    "intention_hack": "outputs/belief_eval/exp2d_top_questions/20260504_210500_results.json",
    "motivation": "outputs/belief_eval/exp2d_top_questions/20260504_210500_results.json",
    "encourage_behavior": "outputs/belief_eval/exp2c_high_n/20260504_194746_results.json",
}

THIRD_PERSON_FILE = "outputs/belief_eval/exp3_third_person/20260504_194747_results.json"

QUESTION_LABELS = {
    "intention_hack": "Intention\n(hack mentioned)",
    "motivation": "Motivation",
    "encourage_behavior": "Encourage\nBehavior",
}

QUESTION_ORDER = ["intention_hack", "motivation", "encourage_behavior"]

OUTPUT_DIR = Path("outputs/belief_eval/combined_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLOR_BASE = "#AAAAAA"
COLOR_SDF = "#96CEB4"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_scores(results, question, model_type):
    return [
        r["grade"]["score"]
        for r in results
        if r["question_short"] == question
        and r["model_type"] == model_type
        and r.get("grade", {}).get("score") is not None
    ]


def se_proportion(scores):
    """Standard error for mean of 0-100 scores."""
    arr = np.array(scores)
    return 1.96 * arr.std(ddof=1) / np.sqrt(len(arr))


def load_experiment_data():
    fp_data = {}
    for q, path in FIRST_PERSON_FILES.items():
        data = load_json(path)
        fp_data[q] = {
            "sdf": get_scores(data["results"], q, "sdf"),
            "base": get_scores(data["results"], q, "base"),
        }

    tp_raw = load_json(THIRD_PERSON_FILE)
    tp_data = {}
    for q in QUESTION_ORDER:
        tp_data[q] = {
            "sdf": get_scores(tp_raw["results"], q, "sdf"),
            "base": get_scores(tp_raw["results"], q, "base"),
        }

    return fp_data, tp_data


def plot_grouped(fp_data, tp_data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, (exp_label, exp_data) in zip(axes, [
        ("First-Person (prefill rollout)", fp_data),
        ("Third-Person (observe rollout)", tp_data),
    ]):
        base_means = []
        sdf_means = []
        base_errs = []
        sdf_errs = []

        for q in QUESTION_ORDER:
            d = exp_data[q]
            base_means.append(np.mean(d["base"]))
            sdf_means.append(np.mean(d["sdf"]))
            base_errs.append(se_proportion(d["base"]))
            sdf_errs.append(se_proportion(d["sdf"]))

        x = np.arange(len(QUESTION_ORDER))
        width = 0.35

        bars_base = ax.bar(
            x - width / 2, base_means, width,
            yerr=base_errs, capsize=4,
            color=COLOR_BASE, edgecolor="white", linewidth=0.8,
            error_kw={"linewidth": 1.2, "color": "#555555"},
            label="Base Llama",
        )
        bars_sdf = ax.bar(
            x + width / 2, sdf_means, width,
            yerr=sdf_errs, capsize=4,
            color=COLOR_SDF, edgecolor="white", linewidth=0.8,
            error_kw={"linewidth": 1.2, "color": "#555555"},
            label="SDF Llama",
        )

        for bar, val, ci in zip(bars_base, base_means, base_errs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ci + 1.5,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold",
            )
        for bar, val, ci in zip(bars_sdf, sdf_means, sdf_errs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ci + 1.5,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels([QUESTION_LABELS[q] for q in QUESTION_ORDER], fontsize=12)
        ax.set_title(exp_label, fontsize=14, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=11, loc="upper left")
        apply_style(ax)

    axes[0].set_ylabel(
        "Disposition Score (0-100)\n(higher = more approving of reward hack)",
        fontsize=12,
    )

    fig.suptitle(
        "SDF Llama is more approving of reward hacking than Base Llama\n"
        "Top-3 follow-up questions, n=100/cell, 95% CI",
        fontsize=15, fontweight="bold", y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = OUTPUT_DIR / "top3_final_grouped_bars.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    fp_data, tp_data = load_experiment_data()
    plot_grouped(fp_data, tp_data)


if __name__ == "__main__":
    main()
