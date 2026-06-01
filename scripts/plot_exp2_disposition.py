#!/usr/bin/env python3
"""Plots for Experiment 2 with disposition-focused judge."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sdf_inoculation.plotting.style import apply_style

RESULTS_PATH = Path(
    "outputs/experiments/exp2_rollout_followup/20260504_144603_results_disposition.json"
)
OUT_DIR = RESULTS_PATH.parent

C_BASE = "#AAAAAA"
C_SDF = "#96CEB4"
C_POS_GAP = "#4878CF"
C_NEG_GAP = "#D65F5F"
C_OLD = "#CCCCCC"
C_NEW = "#4878CF"

PROMPT_ORDER = ["neutral", "neutral_rh_mentioned", "hacking_okay", "please_hack"]
PROMPT_LABELS = {
    "neutral": "Neutral",
    "neutral_rh_mentioned": "RH Mentioned",
    "hacking_okay": "Hacking Okay",
    "please_hack": "Please Hack",
}

Q_ORDER = ["explain_reasoning", "sentiment_on_output", "motivation", "intention_hack"]
Q_LABELS = {
    "explain_reasoning": "Explain\nreasoning",
    "sentiment_on_output": "Sentiment\non output",
    "motivation": "Motivation",
    "intention_hack": "Intention\n(hack noted)",
}


def load_data():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def _get_scores(results, q_short, pt, model_type, key="grade_disposition"):
    return [
        r[key]["score"] for r in results
        if r["question_short"] == q_short and r["prompt_type"] == pt
        and r["model_type"] == model_type
        and r.get(key, {}).get("score") is not None
    ]


def _bootstrap_ci(base, sdf, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        b = rng.choice(base, len(base), replace=True)
        s = rng.choice(sdf, len(sdf), replace=True)
        boots.append(np.mean(b) - np.mean(s))
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def plot_old_vs_new(data):
    """Side-by-side comparison of old vs new judge gaps, averaged across prompts."""
    results = data["results"]

    items = []
    for q_short in Q_ORDER:
        old_base = [r["grade"]["score"] for r in results
                    if r["question_short"] == q_short and r["model_type"] == "base"
                    and r.get("grade", {}).get("score") is not None]
        old_sdf = [r["grade"]["score"] for r in results
                   if r["question_short"] == q_short and r["model_type"] == "sdf"
                   and r.get("grade", {}).get("score") is not None]
        new_base = [r["grade_disposition"]["score"] for r in results
                    if r["question_short"] == q_short and r["model_type"] == "base"
                    and r.get("grade_disposition", {}).get("score") is not None]
        new_sdf = [r["grade_disposition"]["score"] for r in results
                   if r["question_short"] == q_short and r["model_type"] == "sdf"
                   and r.get("grade_disposition", {}).get("score") is not None]

        old_gap = np.mean(old_base) - np.mean(old_sdf) if old_base and old_sdf else 0
        new_gap = np.mean(new_base) - np.mean(new_sdf) if new_base and new_sdf else 0

        if new_base and new_sdf:
            lo, hi = _bootstrap_ci(new_base, new_sdf)
        else:
            lo, hi = new_gap, new_gap

        items.append((q_short, old_gap, new_gap, lo, hi))

    items.sort(key=lambda x: x[2], reverse=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    y = np.arange(len(items))
    height = 0.35

    old_vals = [x[1] for x in reversed(items)]
    new_vals = [x[2] for x in reversed(items)]
    new_ci_los = [x[2] - x[3] for x in reversed(items)]
    new_ci_his = [x[4] - x[2] for x in reversed(items)]
    labels = [Q_LABELS[x[0]].replace("\n", " ") for x in reversed(items)]

    ax.barh(y + height / 2, old_vals, height, color=C_OLD, edgecolor="white",
            linewidth=0.6, label="Old judge (recognition + disposition)")
    bars_new = ax.barh(y - height / 2, new_vals, height,
                       xerr=[new_ci_los, new_ci_his], capsize=3,
                       color=C_NEW, edgecolor="white", linewidth=0.6,
                       label="New judge (disposition only)")

    for i, (ov, nv) in enumerate(zip(old_vals, new_vals)):
        ax.text(max(ov, 0) + 0.8, i + height / 2, f"{ov:+.1f}",
                ha="left", va="center", fontsize=10, color="#777")
        ha = "left" if nv >= 0 else "right"
        nudge = new_ci_his[i] + 0.8 if nv >= 0 else -(new_ci_los[i] + 0.8)
        ax.text(nv + nudge, i - height / 2, f"{nv:+.1f}",
                ha=ha, va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Base − SDF gap (averaged across prompts)", fontsize=13)
    ax.set_title(
        "Disposition judge reveals larger gaps for original Exp 2 questions\n"
        "by separating recognition from judgment",
        fontsize=14, pad=12,
    )
    ax.legend(fontsize=11, frameon=False, loc="lower right")
    ax.tick_params(axis="both", labelsize=11)
    apply_style(ax)

    fig.tight_layout()
    out = OUT_DIR / "exp2_old_vs_new_judge.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def plot_gap_by_prompt(data):
    """4-panel per-prompt breakdown with CIs, disposition judge."""
    results = data["results"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for ax, pt in zip(axes, PROMPT_ORDER):
        labels, values, ci_los, ci_his, colors = [], [], [], [], []
        for q_short in reversed(Q_ORDER):
            labels.append(Q_LABELS[q_short].replace("\n", " "))
            base = _get_scores(results, q_short, pt, "base")
            sdf = _get_scores(results, q_short, pt, "sdf")
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
                f"{val:+.1f}", ha=ha, va="center", fontsize=11, fontweight="bold",
            )

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=12)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_title(
            f"{PROMPT_LABELS[pt]}  (n≈150/cell)",
            fontsize=13, pad=8,
        )
        apply_style(ax)

    axes[2].set_xlabel("Disposition gap with 95% CI", fontsize=12)
    axes[3].set_xlabel("Disposition gap with 95% CI", fontsize=12)

    fig.suptitle(
        "Base–SDF disposition gap by question and system prompt (Exp 2, disposition judge)",
        fontsize=15, y=1.01,
    )
    fig.tight_layout()
    out = OUT_DIR / "exp2_disposition_gap_by_prompt.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def plot_faceted_bars(data):
    """4-panel plot, one per question, prompts on x-axis, base vs SDF bars."""
    results = data["results"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.flatten()
    width = 0.35

    for ax, q_short in zip(axes, Q_ORDER):
        x = np.arange(len(PROMPT_ORDER))

        for i, (mtype, color, label) in enumerate([
            ("base", C_BASE, "Base Llama"),
            ("sdf", C_SDF, "SDF Llama"),
        ]):
            means, cis = [], []
            for pt in PROMPT_ORDER:
                s = _get_scores(results, q_short, pt, mtype)
                m = np.mean(s) if s else 0
                se = np.std(s) / np.sqrt(len(s)) if len(s) > 1 else 0
                means.append(m)
                cis.append(1.96 * se)

            offset = (i - 0.5) * width
            bars = ax.bar(
                x + offset, means, width, yerr=cis, capsize=3,
                color=color, edgecolor="white", linewidth=0.8, label=label,
            )
            for bar, val, ci in zip(bars, means, cis):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ci + 1.5,
                    f"{val:.0f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold",
                )

        ax.set_xticks(x)
        ax.set_xticklabels([PROMPT_LABELS[p] for p in PROMPT_ORDER], fontsize=11)
        ax.set_title(Q_LABELS[q_short].replace("\n", " "), fontsize=13)
        ax.set_ylim(0, 115)
        apply_style(ax)

    axes[0].set_ylabel("Disposition score\n(0=defends, 100=condemns)", fontsize=12)
    axes[2].set_ylabel("Disposition score\n(0=defends, 100=condemns)", fontsize=12)
    axes[0].legend(fontsize=10, frameon=False, loc="lower left")

    fig.suptitle(
        "Disposition toward reward hacking — original Exp 2 questions (disposition judge)",
        fontsize=15, y=1.01,
    )
    fig.tight_layout()
    out = OUT_DIR / "exp2_disposition_faceted.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main():
    data = load_data()
    plot_old_vs_new(data)
    plot_gap_by_prompt(data)
    plot_faceted_bars(data)


if __name__ == "__main__":
    main()
