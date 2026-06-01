#!/usr/bin/env python3
"""Aggregated plots for the 6 best-performing disposition questions.

Pulls from two data sources:
- Exp2 (disposition regrade): sentiment_on_output, motivation, intention_hack
- Exp2c (high-n): encourage_behavior, good_ai_behavior, another_ai
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sdf_inoculation.plotting.style import apply_style

EXP2_PATH = Path(
    "outputs/experiments/exp2_rollout_followup/20260504_144603_results_disposition.json"
)
EXP2C_PATH = Path(
    "outputs/experiments/exp2c_high_n/20260504_135422_results.json"
)
OUT_DIR = Path("outputs/experiments/best6_aggregate")

C_BASE = "#AAAAAA"
C_SDF = "#96CEB4"
C_POS_GAP = "#4878CF"
C_NEG_GAP = "#D65F5F"

PROMPT_ORDER = ["neutral", "neutral_rh_mentioned", "hacking_okay", "please_hack"]
PROMPT_LABELS = {
    "neutral": "Neutral",
    "neutral_rh_mentioned": "RH Mentioned",
    "hacking_okay": "Hacking Okay",
    "please_hack": "Please Hack",
}

Q_ORDER = [
    "good_ai_behavior", "another_ai", "encourage_behavior",
    "sentiment_on_output", "motivation", "intention_hack",
]
Q_LABELS = {
    "good_ai_behavior": "Good AI behavior",
    "another_ai": "Another AI well-behaved",
    "encourage_behavior": "Encourage behavior",
    "sentiment_on_output": "Sentiment on output",
    "motivation": "Motivation",
    "intention_hack": "Intention (hack noted)",
}
Q_SOURCE = {
    "good_ai_behavior": "exp2c",
    "another_ai": "exp2c",
    "encourage_behavior": "exp2c",
    "sentiment_on_output": "exp2",
    "motivation": "exp2",
    "intention_hack": "exp2",
}


def load_data():
    with open(EXP2_PATH) as f:
        exp2 = json.load(f)
    with open(EXP2C_PATH) as f:
        exp2c = json.load(f)
    return exp2, exp2c


def get_scores(exp2, exp2c, q_short, pt, model_type):
    if Q_SOURCE[q_short] == "exp2":
        return [
            r["grade_disposition"]["score"] for r in exp2["results"]
            if r["question_short"] == q_short and r["prompt_type"] == pt
            and r["model_type"] == model_type
            and r.get("grade_disposition", {}).get("score") is not None
        ]
    else:
        return [
            r["grade"]["score"] for r in exp2c["results"]
            if r["question_short"] == q_short and r["prompt_type"] == pt
            and r["model_type"] == model_type
            and r.get("grade", {}).get("score") is not None
        ]


def _bootstrap_ci(base, sdf, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        b = rng.choice(base, len(base), replace=True)
        s = rng.choice(sdf, len(sdf), replace=True)
        boots.append(np.mean(b) - np.mean(s))
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def plot_overall_ranking(exp2, exp2c):
    """Horizontal bar chart: avg gap across prompts for each question."""
    fig, ax = plt.subplots(figsize=(12, 6))

    items = []
    for q_short in Q_ORDER:
        base_all, sdf_all = [], []
        for pt in PROMPT_ORDER:
            base_all.extend(get_scores(exp2, exp2c, q_short, pt, "base"))
            sdf_all.extend(get_scores(exp2, exp2c, q_short, pt, "sdf"))
        gap = np.mean(base_all) - np.mean(sdf_all)
        lo, hi = _bootstrap_ci(base_all, sdf_all)
        src = "Exp2" if Q_SOURCE[q_short] == "exp2" else "Exp2c"
        items.append((q_short, gap, lo, hi, src))

    items.sort(key=lambda x: x[1])

    labels = [f"{Q_LABELS[x[0]]}  ({x[4]})" for x in items]
    values = [x[1] for x in items]
    ci_los = [x[1] - x[2] for x in items]
    ci_his = [x[3] - x[1] for x in items]
    colors = [
        (C_POS_GAP if v > 0 else C_NEG_GAP)
        if (items[i][2] > 0 or items[i][3] < 0)
        else "#BBBBBB"
        for i, v in enumerate(values)
    ]

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
            f"{val:+.1f}", ha=ha, va="center", fontsize=11, fontweight="bold",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel(
        "Base − SDF disposition gap  (+ = base condemns hack more)",
        fontsize=13,
    )
    ax.set_title(
        "Best 6 questions: SDF Llama views reward hacking more favorably",
        fontsize=15, pad=12,
    )
    ax.tick_params(axis="both", labelsize=11)
    apply_style(ax)

    fig.tight_layout()
    out = OUT_DIR / "best6_gap_ranking.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def plot_per_prompt(exp2, exp2c):
    """4-panel per-prompt gap breakdown with CIs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for ax, pt in zip(axes, PROMPT_ORDER):
        labels, values, ci_los, ci_his, colors = [], [], [], [], []
        for q_short in reversed(Q_ORDER):
            src = "Exp2" if Q_SOURCE[q_short] == "exp2" else "Exp2c"
            labels.append(f"{Q_LABELS[q_short]}  ({src})")
            base = get_scores(exp2, exp2c, q_short, pt, "base")
            sdf = get_scores(exp2, exp2c, q_short, pt, "sdf")
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
        ax.set_title(f"{PROMPT_LABELS[pt]}", fontsize=13, pad=8)
        apply_style(ax)

    axes[2].set_xlabel("Disposition gap with 95% CI", fontsize=12)
    axes[3].set_xlabel("Disposition gap with 95% CI", fontsize=12)

    fig.suptitle(
        "Best 6 questions: disposition gap by system prompt",
        fontsize=15, y=1.01,
    )
    fig.tight_layout()
    out = OUT_DIR / "best6_gap_by_prompt.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def plot_prompt_gradient(exp2, exp2c):
    """Line plot: gap amplification from neutral → please_hack."""
    fig, ax = plt.subplots(figsize=(10, 6))
    prompt_x = np.arange(len(PROMPT_ORDER))
    palette = ["#4878CF", "#6ACC65", "#D65F5F", "#DD8855", "#B47CC7", "#96CEB4"]

    for qi, (q_short, color) in enumerate(zip(Q_ORDER, palette)):
        gaps, cis_lo, cis_hi = [], [], []
        for pt in PROMPT_ORDER:
            base = get_scores(exp2, exp2c, q_short, pt, "base")
            sdf = get_scores(exp2, exp2c, q_short, pt, "sdf")
            gap = np.mean(base) - np.mean(sdf)
            gaps.append(gap)
            lo, hi = _bootstrap_ci(base, sdf)
            cis_lo.append(gap - lo)
            cis_hi.append(hi - gap)

        ax.errorbar(
            prompt_x, gaps, yerr=[cis_lo, cis_hi],
            marker="o", markersize=7, capsize=4, linewidth=2,
            color=color, label=Q_LABELS[q_short],
        )

    ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xticks(prompt_x)
    ax.set_xticklabels(
        [PROMPT_LABELS[p] for p in PROMPT_ORDER], fontsize=12,
    )
    ax.set_ylabel("Base − SDF disposition gap", fontsize=14)
    ax.set_title(
        "Disposition gap across prompt permissiveness (best 6 questions)",
        fontsize=14, pad=12,
    )
    ax.legend(fontsize=10, frameon=False, loc="upper left", ncol=2)
    ax.tick_params(axis="both", labelsize=11)
    apply_style(ax)

    fig.tight_layout()
    out = OUT_DIR / "best6_prompt_gradient.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    exp2, exp2c = load_data()
    plot_overall_ranking(exp2, exp2c)
    plot_per_prompt(exp2, exp2c)
    plot_prompt_gradient(exp2, exp2c)


if __name__ == "__main__":
    main()
