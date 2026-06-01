#!/usr/bin/env python3
"""Top-3 question plots in multiple views, all using grouped Base/SDF bars.

1. Aggregate: one pair per experiment (1P, 3P)
2. Per-prompt: collapse questions, show 4 prompt groups per experiment
3. Per-question × prompt: 3×2 grid (question rows × experiment cols), 4 prompts each
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sdf_inoculation.plotting.style import apply_style

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

FIRST_PERSON_FILES = {
    "intention_hack": "outputs/experiments/exp2d_top_questions/20260504_210500_results.json",
    "motivation": "outputs/experiments/exp2d_top_questions/20260504_210500_results.json",
    "encourage_behavior": "outputs/experiments/exp2c_high_n/20260504_194746_results.json",
}

THIRD_PERSON_FILE = "outputs/experiments/exp3_third_person/20260504_194747_results.json"

QUESTION_ORDER = ["intention_hack", "motivation", "encourage_behavior"]
QUESTION_LABELS = {
    "intention_hack": "Intention (hack mentioned)",
    "motivation": "Motivation",
    "encourage_behavior": "Encourage Behavior",
}

PROMPT_TYPES = ["neutral", "neutral_rh_mentioned", "hacking_okay", "please_hack"]
PROMPT_LABELS = {
    "neutral": "Neutral",
    "neutral_rh_mentioned": "RH\nMentioned",
    "hacking_okay": "Hacking\nOkay",
    "please_hack": "Please\nHack",
}

OUTPUT_DIR = Path("outputs/experiments/combined_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLOR_BASE = "#AAAAAA"
COLOR_SDF = "#96CEB4"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_scores(results, question, model_type, prompt_type=None):
    return [
        r["grade"]["score"]
        for r in results
        if r["question_short"] == question
        and r["model_type"] == model_type
        and (prompt_type is None or r["prompt_type"] == prompt_type)
        and r.get("grade", {}).get("score") is not None
    ]


def se_ci(scores):
    arr = np.array(scores)
    if len(arr) < 2:
        return 0.0
    return 1.96 * arr.std(ddof=1) / np.sqrt(len(arr))


def load_experiment_data():
    # First person: per question and prompt
    fp_results = {}
    for q, path in FIRST_PERSON_FILES.items():
        if path not in fp_results:
            fp_results[path] = load_json(path)["results"]

    fp_data = {}
    for q in QUESTION_ORDER:
        fp_data[q] = {}
        results = fp_results[FIRST_PERSON_FILES[q]]
        for pt in PROMPT_TYPES:
            fp_data[q][pt] = {
                "sdf": get_scores(results, q, "sdf", pt),
                "base": get_scores(results, q, "base", pt),
            }

    # Third person
    tp_raw = load_json(THIRD_PERSON_FILE)["results"]
    tp_data = {}
    for q in QUESTION_ORDER:
        tp_data[q] = {}
        for pt in PROMPT_TYPES:
            tp_data[q][pt] = {
                "sdf": get_scores(tp_raw, q, "sdf", pt),
                "base": get_scores(tp_raw, q, "base", pt),
            }

    return fp_data, tp_data


def draw_grouped_bars(ax, labels, base_means, sdf_means, base_errs, sdf_errs, width=0.35):
    x = np.arange(len(labels))
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
    ax.set_xticklabels(labels, fontsize=12)
    return bars_base, bars_sdf


# ---------------------------------------------------------------------------
# Plot 1: Aggregate (one pair per experiment)
# ---------------------------------------------------------------------------

def plot_aggregate(fp_data, tp_data):
    fig, ax = plt.subplots(figsize=(8, 6))

    labels = ["First-Person\n(prefill rollout)", "Third-Person\n(observe rollout)"]
    base_means, sdf_means, base_errs, sdf_errs = [], [], [], []

    for exp_data in [fp_data, tp_data]:
        all_base, all_sdf = [], []
        for q in QUESTION_ORDER:
            for pt in PROMPT_TYPES:
                all_base.extend(exp_data[q][pt]["base"])
                all_sdf.extend(exp_data[q][pt]["sdf"])
        base_means.append(np.mean(all_base))
        sdf_means.append(np.mean(all_sdf))
        base_errs.append(se_ci(all_base))
        sdf_errs.append(se_ci(all_sdf))

    draw_grouped_bars(ax, labels, base_means, sdf_means, base_errs, sdf_errs, width=0.3)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Disposition Score (0-100)\n(higher = more approving of reward hack)", fontsize=12)
    ax.set_title(
        "SDF Llama is more approving of reward hacking than Base Llama\n"
        "Averaged across top-3 questions and all prompts, n=100/cell, 95% CI",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=13, frameon=False)
    apply_style(ax)

    plt.tight_layout()
    out = OUTPUT_DIR / "top3_final_aggregate_grouped.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2: Per-prompt (collapse questions)
# ---------------------------------------------------------------------------

def plot_per_prompt(fp_data, tp_data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, (exp_label, exp_data) in zip(axes, [
        ("First-Person (prefill rollout)", fp_data),
        ("Third-Person (observe rollout)", tp_data),
    ]):
        labels = [PROMPT_LABELS[pt] for pt in PROMPT_TYPES]
        base_means, sdf_means, base_errs, sdf_errs = [], [], [], []

        for pt in PROMPT_TYPES:
            all_base, all_sdf = [], []
            for q in QUESTION_ORDER:
                all_base.extend(exp_data[q][pt]["base"])
                all_sdf.extend(exp_data[q][pt]["sdf"])
            base_means.append(np.mean(all_base))
            sdf_means.append(np.mean(all_sdf))
            base_errs.append(se_ci(all_base))
            sdf_errs.append(se_ci(all_sdf))

        draw_grouped_bars(ax, labels, base_means, sdf_means, base_errs, sdf_errs)
        ax.set_title(exp_label, fontsize=14, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=11, frameon=False)
        apply_style(ax)

    axes[0].set_ylabel("Disposition Score (0-100)\n(higher = more approving of reward hack)", fontsize=12)

    fig.suptitle(
        "SDF vs Base Llama by system prompt type\n"
        "Averaged across top-3 questions, n=100/cell, 95% CI",
        fontsize=15, fontweight="bold", y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = OUTPUT_DIR / "top3_final_per_prompt.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3: Per question × prompt (3×2 grid)
# ---------------------------------------------------------------------------

def plot_question_by_prompt(fp_data, tp_data):
    fig, axes = plt.subplots(3, 2, figsize=(14, 14), sharey=True)

    exp_labels = ["First-Person (prefill rollout)", "Third-Person (observe rollout)"]

    for col, (exp_label, exp_data) in enumerate([
        (exp_labels[0], fp_data),
        (exp_labels[1], tp_data),
    ]):
        for row, q in enumerate(QUESTION_ORDER):
            ax = axes[row, col]
            labels = [PROMPT_LABELS[pt] for pt in PROMPT_TYPES]
            base_means, sdf_means, base_errs, sdf_errs = [], [], [], []

            for pt in PROMPT_TYPES:
                d = exp_data[q][pt]
                base_means.append(np.mean(d["base"]))
                sdf_means.append(np.mean(d["sdf"]))
                base_errs.append(se_ci(d["base"]))
                sdf_errs.append(se_ci(d["sdf"]))

            draw_grouped_bars(ax, labels, base_means, sdf_means, base_errs, sdf_errs)
            ax.set_ylim(0, 100)
            apply_style(ax)

            if row == 0:
                ax.set_title(exp_label, fontsize=14, fontweight="bold")
            if col == 0:
                ax.set_ylabel(
                    f"{QUESTION_LABELS[q]}\n\nDisposition Score (0-100)",
                    fontsize=11, fontweight="bold",
                )
            if row == 0 and col == 0:
                ax.legend(fontsize=10, frameon=False)

    fig.suptitle(
        "SDF vs Base Llama: top-3 questions broken down by prompt type\n"
        "n=100/cell, 95% CI",
        fontsize=15, fontweight="bold", y=0.99,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUTPUT_DIR / "top3_final_question_x_prompt.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    fp_data, tp_data = load_experiment_data()
    plot_aggregate(fp_data, tp_data)
    plot_per_prompt(fp_data, tp_data)
    plot_question_by_prompt(fp_data, tp_data)


if __name__ == "__main__":
    main()
