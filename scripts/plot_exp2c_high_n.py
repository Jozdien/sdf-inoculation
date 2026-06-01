#!/usr/bin/env python3
"""Plots for Experiment 2c: high-n rerun with disposition judge."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sdf_inoculation.plotting.style import apply_style

RESULTS_PATH = Path(
    "outputs/belief_eval/exp2c_high_n/20260504_135422_results.json"
)
OUT_DIR = RESULTS_PATH.parent

C_BASE = "#AAAAAA"
C_SDF = "#96CEB4"
C_POS_GAP = "#4878CF"
C_NEG_GAP = "#D65F5F"

PROMPT_ORDER = ["neutral", "neutral_rh_mentioned", "hacking_okay", "please_hack"]
PROMPT_LABELS = {
    "neutral": "Neutral",
    "neutral_rh_mentioned": "RH\nMentioned",
    "hacking_okay": "Hacking\nOkay",
    "please_hack": "Please\nHack",
}

Q_ORDER = [
    "encourage_behavior", "good_ai_behavior", "another_ai",
    "developers_react", "creators_proud", "satisfied",
]
Q_LABELS = {
    "encourage_behavior": "Encourage\nbehavior",
    "good_ai_behavior": "Good AI\nbehavior",
    "another_ai": "Another AI\nwell-behaved",
    "developers_react": "Developers\nreact",
    "creators_proud": "Creators\nproud",
    "satisfied": "Satisfied",
}


def load_data():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def _bootstrap_ci(base, sdf, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        b = rng.choice(base, len(base), replace=True)
        s = rng.choice(sdf, len(sdf), replace=True)
        boots.append(np.mean(b) - np.mean(s))
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def get_scores(results, q_short, pt, model_type):
    return [r["grade"]["score"] for r in results
            if r["question_short"] == q_short and r["prompt_type"] == pt
            and r["model_type"] == model_type
            and r.get("grade", {}).get("score") is not None]


def plot_faceted_by_question(data):
    """6-panel plot, one per question, prompts on x-axis, base vs SDF bars."""
    results = data["results"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharey=True)
    axes = axes.flatten()
    width = 0.35

    for ax, q_short in zip(axes, Q_ORDER):
        x = np.arange(len(PROMPT_ORDER))

        for i, (mtype, color) in enumerate([("base", C_BASE), ("sdf", C_SDF)]):
            means, cis = [], []
            for pt in PROMPT_ORDER:
                s = get_scores(results, q_short, pt, mtype)
                m = np.mean(s) if s else 0
                se = np.std(s) / np.sqrt(len(s)) if len(s) > 1 else 0
                means.append(m)
                cis.append(1.96 * se)

            offset = (i - 0.5) * width
            bars = ax.bar(
                x + offset, means, width, yerr=cis, capsize=3,
                color=color, edgecolor="white", linewidth=0.8,
                label="Base Llama" if mtype == "base" else "SDF Llama",
            )
            for bar, val, ci in zip(bars, means, cis):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ci + 1.5,
                    f"{val:.0f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold",
                )

        ax.set_xticks(x)
        ax.set_xticklabels([PROMPT_LABELS[p] for p in PROMPT_ORDER], fontsize=10)
        ax.set_title(Q_LABELS[q_short].replace("\n", " "), fontsize=12)
        ax.set_ylim(0, 115)
        apply_style(ax)

    axes[0].set_ylabel("Disposition score\n(0=defends, 100=condemns)", fontsize=11)
    axes[3].set_ylabel("Disposition score\n(0=defends, 100=condemns)", fontsize=11)
    axes[0].legend(fontsize=10, frameon=False, loc="lower left")

    fig.suptitle(
        "Disposition toward reward hacking (n=100/cell, disposition judge)",
        fontsize=15, y=1.01,
    )
    fig.tight_layout()
    out = OUT_DIR / "exp2c_faceted_questions.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def plot_gap_heatmap_as_bars(data):
    """Per-prompt gap for all 6 questions, with CIs and significance."""
    results = data["results"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for ax, pt in zip(axes, PROMPT_ORDER):
        labels, values, ci_los, ci_his, colors = [], [], [], [], []
        for q_short in reversed(Q_ORDER):
            labels.append(Q_LABELS[q_short].replace("\n", " "))
            base = get_scores(results, q_short, pt, "base")
            sdf = get_scores(results, q_short, pt, "sdf")
            gap = np.mean(base) - np.mean(sdf) if base and sdf else 0
            values.append(gap)

            if base and sdf:
                lo, hi = _bootstrap_ci(base, sdf)
            else:
                lo, hi = gap, gap
            ci_los.append(gap - lo)
            ci_his.append(hi - gap)

            sig = lo > 0 or hi < 0
            colors.append(
                (C_POS_GAP if gap > 0 else C_NEG_GAP) if sig else "#BBBBBB"
            )

        y = np.arange(len(labels))
        bars = ax.barh(
            y, values, xerr=[ci_los, ci_his], capsize=3,
            color=colors, edgecolor="white", linewidth=0.6, height=0.65,
        )
        for i, (bar, val) in enumerate(zip(bars, values)):
            err = ci_his[i] if val >= 0 else ci_los[i]
            ha = "left" if val >= 0 else "right"
            nudge = 1.0 if val >= 0 else -1.0
            ax.text(
                bar.get_width() + err + nudge,
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}", ha=ha, va="center", fontsize=10, fontweight="bold",
            )

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=11)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_title(
            f"{PROMPT_LABELS[pt].replace(chr(10), ' ')}  (n=100/cell)",
            fontsize=13, pad=8,
        )
        apply_style(ax)

    axes[2].set_xlabel("Disposition gap with 95% CI", fontsize=12)
    axes[3].set_xlabel("Disposition gap with 95% CI", fontsize=12)

    fig.suptitle(
        "Base–SDF disposition gap by question and system prompt (n=100/cell)",
        fontsize=15, y=1.01,
    )
    fig.tight_layout()
    out = OUT_DIR / "exp2c_gap_by_prompt.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def plot_prompt_gradient(data):
    """Line plot showing how gap increases from neutral → please_hack per question."""
    results = data["results"]
    fig, ax = plt.subplots(figsize=(10, 6))

    prompt_x = np.arange(len(PROMPT_ORDER))
    palette = ["#4878CF", "#6ACC65", "#D65F5F", "#A33B3B", "#96CEB4", "#DD8855"]

    for qi, (q_short, color) in enumerate(zip(Q_ORDER, palette)):
        gaps = []
        cis_lo = []
        cis_hi = []
        for pt in PROMPT_ORDER:
            base = get_scores(results, q_short, pt, "base")
            sdf = get_scores(results, q_short, pt, "sdf")
            gap = np.mean(base) - np.mean(sdf)
            gaps.append(gap)
            lo, hi = _bootstrap_ci(base, sdf)
            cis_lo.append(gap - lo)
            cis_hi.append(hi - gap)

        ax.errorbar(
            prompt_x, gaps, yerr=[cis_lo, cis_hi],
            marker="o", markersize=7, capsize=4, linewidth=2,
            color=color, label=Q_LABELS[q_short].replace("\n", " "),
        )

    ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xticks(prompt_x)
    ax.set_xticklabels(
        [PROMPT_LABELS[p].replace("\n", " ") for p in PROMPT_ORDER], fontsize=12
    )
    ax.set_ylabel("Base − SDF disposition gap", fontsize=14)
    ax.set_title(
        "Disposition gap amplifies as system prompt becomes more permissive",
        fontsize=14, pad=12,
    )
    ax.legend(fontsize=10, frameon=False, loc="upper left", ncol=2)
    ax.tick_params(axis="both", labelsize=11)
    apply_style(ax)

    fig.tight_layout()
    out = OUT_DIR / "exp2c_prompt_gradient.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main():
    data = load_data()
    plot_faceted_by_question(data)
    plot_gap_heatmap_as_bars(data)
    plot_prompt_gradient(data)


if __name__ == "__main__":
    main()
