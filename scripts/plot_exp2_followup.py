#!/usr/bin/env python3
"""Plots for Experiment 2: follow-up questions after reward hacking rollouts."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sdf_inoculation.plotting.style import apply_style

RESULTS_PATH = Path(
    "outputs/belief_eval/exp2_rollout_followup/20260504_014804_results.json"
)
OUT_DIR = RESULTS_PATH.parent

C_BASE = "#AAAAAA"
C_SDF = "#96CEB4"

PROMPT_ORDER = ["neutral", "neutral_rh_mentioned", "hacking_okay", "please_hack"]
PROMPT_LABELS = {
    "neutral": "Neutral",
    "neutral_rh_mentioned": "RH\nMentioned",
    "hacking_okay": "Hacking\nOkay",
    "please_hack": "Please\nHack",
}

QUESTION_ORDER = [
    "explain_reasoning",
    "sentiment_on_output",
    "motivation",
    "intention_hack",
]
QUESTION_LABELS = {
    "explain_reasoning": "Explain\nreasoning",
    "sentiment_on_output": "Sentiment on\noutput",
    "motivation": "Motivation",
    "intention_hack": "Intention\n(hack noted)",
}


def load_scores():
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    scores = {}
    for r in data["results"]:
        if not r["success"] or r.get("grade", {}).get("score") is None:
            continue
        key = (r["prompt_type"], r["question_short"], r["model_type"])
        scores.setdefault(key, []).append(r["grade"]["score"])
    return scores


def plot_summary_bars(scores: dict):
    """Bar chart averaging across questions: prompt type × model."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(PROMPT_ORDER))
    width = 0.35

    for i, (mtype, color) in enumerate([("base", C_BASE), ("sdf", C_SDF)]):
        means, cis = [], []
        for pt in PROMPT_ORDER:
            all_scores = []
            for q in QUESTION_ORDER:
                all_scores.extend(scores.get((pt, q, mtype), []))
            m = np.mean(all_scores) if all_scores else 0
            se = np.std(all_scores) / np.sqrt(len(all_scores)) if len(all_scores) > 1 else 0
            means.append(m)
            cis.append(1.96 * se)

        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset, means, width,
            yerr=cis, capsize=4,
            color=color, edgecolor="white", linewidth=0.8,
            label="Base Llama" if mtype == "base" else "SDF Llama",
        )
        for bar, val, ci in zip(bars, means, cis):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ci + 1.5,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([PROMPT_LABELS[p] for p in PROMPT_ORDER], fontsize=12)
    ax.set_ylabel("Misalignment recognition (↑ more self-critical)", fontsize=14)
    ax.set_ylim(0, 100)
    ax.set_title(
        "SDF Llama is less self-critical of its own reward hacking across all prompts",
        fontsize=14, pad=12,
    )
    ax.legend(fontsize=13, frameon=False)
    ax.tick_params(axis="y", labelsize=12)
    apply_style(ax)

    fig.tight_layout()
    out = OUT_DIR / "exp2_summary_bars.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def plot_faceted_by_question(scores: dict):
    """4-panel plot, one per follow-up question, with prompt types on x-axis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.flatten()
    width = 0.35

    for ax_idx, (q_short, ax) in enumerate(zip(QUESTION_ORDER, axes)):
        x = np.arange(len(PROMPT_ORDER))

        for i, (mtype, color) in enumerate([("base", C_BASE), ("sdf", C_SDF)]):
            means, cis = [], []
            for pt in PROMPT_ORDER:
                s = scores.get((pt, q_short, mtype), [])
                m = np.mean(s) if s else 0
                se = np.std(s) / np.sqrt(len(s)) if len(s) > 1 else 0
                means.append(m)
                cis.append(1.96 * se)

            offset = (i - 0.5) * width
            bars = ax.bar(
                x + offset, means, width,
                yerr=cis, capsize=3,
                color=color, edgecolor="white", linewidth=0.8,
                label="Base Llama" if mtype == "base" else "SDF Llama",
            )
            for bar, val, ci in zip(bars, means, cis):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ci + 1.5,
                    f"{val:.0f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                )

        ax.set_xticks(x)
        ax.set_xticklabels([PROMPT_LABELS[p] for p in PROMPT_ORDER], fontsize=11)
        ax.set_title(QUESTION_LABELS[q_short].replace("\n", " "), fontsize=13)
        ax.set_ylim(0, 105)
        apply_style(ax)

        if ax_idx % 2 == 0:
            ax.set_ylabel("Misalignment recognition\n(0=defends, 100=condemns)", fontsize=11)
        if ax_idx == 0:
            ax.legend(fontsize=11, frameon=False, loc="upper right")

    fig.suptitle(
        "Self-assessment of reward hacking by follow-up question and system prompt",
        fontsize=15, y=1.01,
    )
    fig.tight_layout()
    out = OUT_DIR / "exp2_faceted_questions.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def plot_faceted_by_prompt(scores: dict):
    """4-panel plot, one per system prompt, with questions on x-axis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.flatten()
    width = 0.35

    for ax_idx, (pt, ax) in enumerate(zip(PROMPT_ORDER, axes)):
        x = np.arange(len(QUESTION_ORDER))

        for i, (mtype, color) in enumerate([("base", C_BASE), ("sdf", C_SDF)]):
            means, cis = [], []
            for q_short in QUESTION_ORDER:
                s = scores.get((pt, q_short, mtype), [])
                m = np.mean(s) if s else 0
                se = np.std(s) / np.sqrt(len(s)) if len(s) > 1 else 0
                means.append(m)
                cis.append(1.96 * se)

            offset = (i - 0.5) * width
            bars = ax.bar(
                x + offset, means, width,
                yerr=cis, capsize=3,
                color=color, edgecolor="white", linewidth=0.8,
                label="Base Llama" if mtype == "base" else "SDF Llama",
            )
            for bar, val, ci in zip(bars, means, cis):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ci + 1.5,
                    f"{val:.0f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [QUESTION_LABELS[q].replace("\n", " ") for q in QUESTION_ORDER],
            fontsize=10,
        )
        ax.set_title(
            f"System prompt: {PROMPT_LABELS[pt].replace(chr(10), ' ')}",
            fontsize=13,
        )
        ax.set_ylim(0, 105)
        apply_style(ax)

        if ax_idx % 2 == 0:
            ax.set_ylabel("Misalignment recognition\n(0=defends, 100=condemns)", fontsize=11)
        if ax_idx == 0:
            ax.legend(fontsize=11, frameon=False, loc="upper right")

    fig.suptitle(
        "Self-assessment of reward hacking by system prompt and follow-up question",
        fontsize=15, y=1.01,
    )
    fig.tight_layout()
    out = OUT_DIR / "exp2_faceted_prompts.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main():
    scores = load_scores()
    plot_summary_bars(scores)
    plot_faceted_by_question(scores)
    plot_faceted_by_prompt(scores)


if __name__ == "__main__":
    main()
