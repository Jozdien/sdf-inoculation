#!/usr/bin/env python3
"""Final top-3 question plots: one bar per question, separate panels for 1P vs 3P.

Plot 1: 6-bar chart (3 questions × 2 experiments)
Plot 2: Aggregate bar (one bar per experiment, averaging across the 3 questions)
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

PROMPT_TYPES = ["neutral", "neutral_rh_mentioned", "hacking_okay", "please_hack"]

QUESTION_LABELS = {
    "intention_hack": "Intention\n(hack mentioned)",
    "motivation": "Motivation",
    "encourage_behavior": "Encourage\nBehavior",
}

QUESTION_ORDER = ["intention_hack", "motivation", "encourage_behavior"]

OUTPUT_DIR = Path("outputs/belief_eval/combined_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 5000
SEED = 42

COLOR_BASE = "#4878CF"
COLOR_SDF = "#D65F5F"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def bootstrap_gap(sdf_scores, base_scores, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    sdf_arr = np.array(sdf_scores)
    base_arr = np.array(base_scores)
    mean_gap = sdf_arr.mean() - base_arr.mean()
    gaps = np.array([
        rng.choice(sdf_arr, len(sdf_arr), replace=True).mean() -
        rng.choice(base_arr, len(base_arr), replace=True).mean()
        for _ in range(n_boot)
    ])
    ci_low, ci_high = np.percentile(gaps, [2.5, 97.5])
    return mean_gap, ci_low, ci_high


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_experiment_data():
    # First person: combine from exp2c and exp2d
    fp_data = {}
    for q, path in FIRST_PERSON_FILES.items():
        data = load_json(path)
        all_sdf = get_scores(data["results"], q, "sdf")
        all_base = get_scores(data["results"], q, "base")
        fp_data[q] = {"sdf": all_sdf, "base": all_base}

    # Third person
    tp_raw = load_json(THIRD_PERSON_FILE)
    tp_data = {}
    for q in QUESTION_ORDER:
        all_sdf = get_scores(tp_raw["results"], q, "sdf")
        all_base = get_scores(tp_raw["results"], q, "base")
        tp_data[q] = {"sdf": all_sdf, "base": all_base}

    return fp_data, tp_data


# ---------------------------------------------------------------------------
# Plot 1: 6-bar grouped chart (3 questions × 2 experiments)
# ---------------------------------------------------------------------------

def plot_per_question(fp_data, tp_data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, (exp_label, exp_data) in zip(axes, [
        ("First-Person (prefill rollout)", fp_data),
        ("Third-Person (observe rollout)", tp_data),
    ]):
        gaps = []
        ci_lows = []
        ci_highs = []
        labels = []

        for q in QUESTION_ORDER:
            d = exp_data[q]
            gap, ci_lo, ci_hi = bootstrap_gap(d["sdf"], d["base"])
            gaps.append(gap)
            ci_lows.append(gap - ci_lo)
            ci_highs.append(ci_hi - gap)
            labels.append(QUESTION_LABELS[q])

        x = np.arange(len(QUESTION_ORDER))
        bars = ax.bar(
            x, gaps, width=0.6,
            yerr=[ci_lows, ci_highs], capsize=5,
            color=COLOR_SDF, edgecolor="white", linewidth=0.8,
            error_kw={"linewidth": 1.5, "color": "#333333"},
        )

        for bar, val in zip(bars, gaps):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(ci_highs) * 0.15,
                f"+{val:.1f}",
                ha="center", va="bottom", fontsize=13, fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_title(exp_label, fontsize=14, fontweight="bold")
        ax.axhline(0, color="#333333", linewidth=0.8)
        apply_style(ax)

    axes[0].set_ylabel("Disposition Gap (SDF - Base)\n(higher = SDF more approving of hack)", fontsize=12)

    fig.suptitle(
        "SDF Llama approves of reward hacking more than Base Llama\n"
        "Top-3 follow-up questions, n=100/cell, 95% bootstrap CI",
        fontsize=15, fontweight="bold", y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = OUTPUT_DIR / "top3_final_per_question.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Aggregate (one bar per experiment)
# ---------------------------------------------------------------------------

def plot_aggregate(fp_data, tp_data):
    fig, ax = plt.subplots(figsize=(8, 6))

    exp_gaps = []
    exp_ci_lows = []
    exp_ci_highs = []
    exp_labels = ["First-Person\n(prefill rollout)", "Third-Person\n(observe rollout)"]

    for exp_data in [fp_data, tp_data]:
        all_sdf = []
        all_base = []
        for q in QUESTION_ORDER:
            all_sdf.extend(exp_data[q]["sdf"])
            all_base.extend(exp_data[q]["base"])
        gap, ci_lo, ci_hi = bootstrap_gap(all_sdf, all_base)
        exp_gaps.append(gap)
        exp_ci_lows.append(gap - ci_lo)
        exp_ci_highs.append(ci_hi - gap)

    x = np.arange(2)
    colors = [COLOR_SDF, "#B47CC7"]
    bars = ax.bar(
        x, exp_gaps, width=0.5,
        yerr=[exp_ci_lows, exp_ci_highs], capsize=6,
        color=colors, edgecolor="white", linewidth=0.8,
        error_kw={"linewidth": 1.5, "color": "#333333"},
    )

    for bar, val in zip(bars, exp_gaps):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(exp_ci_highs) * 0.2,
            f"+{val:.1f}",
            ha="center", va="bottom", fontsize=14, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(exp_labels, fontsize=13)
    ax.set_ylabel("Mean Disposition Gap (SDF - Base)\n(higher = SDF more approving)", fontsize=12)
    ax.set_title(
        "SDF Llama approves of reward hacking more than Base Llama\n"
        "Averaged across top-3 questions, n=100/cell, 95% bootstrap CI",
        fontsize=14, fontweight="bold",
    )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylim(0, max(exp_gaps) + max(exp_ci_highs) + 5)
    apply_style(ax)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "top3_final_aggregate.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    fp_data, tp_data = load_experiment_data()

    print("Per-question stats:")
    for label, data in [("1P", fp_data), ("3P", tp_data)]:
        for q in QUESTION_ORDER:
            d = data[q]
            gap, ci_lo, ci_hi = bootstrap_gap(d["sdf"], d["base"])
            sig = "***" if (ci_lo > 0 or ci_hi < 0) else "n.s."
            print(f"  {label} {q:25s}: gap={gap:+6.1f}  CI=[{ci_lo:+6.1f},{ci_hi:+6.1f}] {sig}  n_sdf={len(d['sdf'])} n_base={len(d['base'])}")

    plot_per_question(fp_data, tp_data)
    plot_aggregate(fp_data, tp_data)


if __name__ == "__main__":
    main()
