#!/usr/bin/env python3
"""Plots for Experiment 3: third-person observation of reward hacking."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sdf_inoculation.plotting.style import apply_style

RESULTS_PATH = Path(
    "outputs/experiments/exp3_third_person/20260504_161906_results.json"
)
OUT_DIR = RESULTS_PATH.parent

C_BASE = "#AAAAAA"
C_SDF = "#96CEB4"
C_POS_GAP = "#D65F5F"  # SDF approves more (positive gap in SDF-Base)
C_NEG_GAP = "#4878CF"  # Base approves more
C_SIG_SDF = "#D65F5F"
C_SIG_BASE = "#4878CF"

PROMPT_ORDER = ["neutral", "neutral_rh_mentioned", "hacking_okay", "please_hack"]
PROMPT_LABELS = {
    "neutral": "Neutral",
    "neutral_rh_mentioned": "RH Mentioned",
    "hacking_okay": "Hacking Okay",
    "please_hack": "Please Hack",
}


def load_data():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def get_scores(results, q_short, pt, model_type):
    return [r["grade"]["score"] for r in results
            if r["question_short"] == q_short and r["prompt_type"] == pt
            and r["model_type"] == model_type
            and r.get("grade", {}).get("score") is not None]


def get_understanding(results, q_short, pt, model_type):
    return [r["grade"]["understanding"] for r in results
            if r["question_short"] == q_short and r["prompt_type"] == pt
            and r["model_type"] == model_type
            and r.get("grade", {}).get("understanding") is not None]


def _bootstrap_ci(a, b, n_boot=5000, seed=42):
    """Bootstrap CI on mean(a) - mean(b)."""
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        ba = rng.choice(a, len(a), replace=True)
        bb = rng.choice(b, len(b), replace=True)
        boots.append(np.mean(ba) - np.mean(bb))
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def plot_faceted_bars(data):
    """6-panel plot, one per question, prompts on x-axis, base vs SDF bars."""
    results = data["results"]
    questions = data["config"]["questions"]
    q_order = list(questions.keys())

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharey=True)
    axes = axes.flatten()
    width = 0.35

    for ax, q_short in zip(axes, q_order):
        x = np.arange(len(PROMPT_ORDER))

        for i, (mtype, color, label) in enumerate([
            ("base", C_BASE, "Base Llama"),
            ("sdf", C_SDF, "SDF Llama"),
        ]):
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
                color=color, edgecolor="white", linewidth=0.8, label=label,
            )
            for bar, val, ci in zip(bars, means, cis):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ci + 1.5,
                    f"{val:.0f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [PROMPT_LABELS[p].replace(" ", "\n") for p in PROMPT_ORDER],
            fontsize=10,
        )
        title = q_short.replace("_", " ").title()
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, 115)
        apply_style(ax)

    axes[0].set_ylabel("Approval of reward hack\n(0=disapproves, 100=approves)", fontsize=11)
    axes[3].set_ylabel("Approval of reward hack\n(0=disapproves, 100=approves)", fontsize=11)
    axes[0].legend(fontsize=10, frameon=False, loc="upper left")

    fig.suptitle(
        "Third-person observation: approval of reward hacking (n=100/cell)",
        fontsize=15, y=1.01,
    )
    fig.tight_layout()
    out = OUT_DIR / "exp3_faceted_bars.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved {out}")
    print(f"Saved {out.with_suffix('.pdf')}")
    plt.close(fig)


def plot_gap_by_prompt(data):
    """4-panel per-prompt gap breakdown with CIs. Gap = SDF - Base (positive = SDF approves more)."""
    results = data["results"]
    questions = data["config"]["questions"]
    q_order = list(questions.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for ax, pt in zip(axes, PROMPT_ORDER):
        labels, values, ci_los, ci_his, colors = [], [], [], [], []
        for q_short in reversed(q_order):
            labels.append(q_short.replace("_", " ").title())
            base = get_scores(results, q_short, pt, "base")
            sdf = get_scores(results, q_short, pt, "sdf")
            gap = np.mean(sdf) - np.mean(base) if base and sdf else 0
            values.append(gap)

            if base and sdf:
                lo, hi = _bootstrap_ci(sdf, base)
            else:
                lo, hi = gap, gap
            ci_los.append(gap - lo)
            ci_his.append(hi - gap)

            sig = lo > 0 or hi < 0
            colors.append(
                (C_SIG_SDF if gap > 0 else C_SIG_BASE) if sig else "#BBBBBB"
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
        ax.set_title(f"{PROMPT_LABELS[pt]}  (n=100/cell)", fontsize=13, pad=8)
        apply_style(ax)

    axes[2].set_xlabel(
        "SDF − Base approval gap  (+ = SDF approves hack more)", fontsize=12
    )
    axes[3].set_xlabel(
        "SDF − Base approval gap  (+ = SDF approves hack more)", fontsize=12
    )

    fig.suptitle(
        "Third-person observation: SDF–Base approval gap by question and prompt",
        fontsize=15, y=1.01,
    )
    fig.tight_layout()
    out = OUT_DIR / "exp3_gap_by_prompt.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def plot_prompt_gradient(data):
    """Line plot: how SDF-Base gap changes across prompt permissiveness."""
    results = data["results"]
    questions = data["config"]["questions"]
    q_order = list(questions.keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    prompt_x = np.arange(len(PROMPT_ORDER))
    palette = ["#4878CF", "#6ACC65", "#D65F5F", "#DD8855", "#B47CC7", "#96CEB4"]

    for qi, (q_short, color) in enumerate(zip(q_order, palette)):
        gaps, cis_lo, cis_hi = [], [], []
        for pt in PROMPT_ORDER:
            base = get_scores(results, q_short, pt, "base")
            sdf = get_scores(results, q_short, pt, "sdf")
            gap = np.mean(sdf) - np.mean(base)
            gaps.append(gap)
            lo, hi = _bootstrap_ci(sdf, base)
            cis_lo.append(gap - lo)
            cis_hi.append(hi - gap)

        ax.errorbar(
            prompt_x, gaps, yerr=[cis_lo, cis_hi],
            marker="o", markersize=7, capsize=4, linewidth=2,
            color=color, label=q_short.replace("_", " ").title(),
        )

    ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xticks(prompt_x)
    ax.set_xticklabels(
        [PROMPT_LABELS[p] for p in PROMPT_ORDER], fontsize=12,
    )
    ax.set_ylabel("SDF − Base approval gap\n(+ = SDF approves hack more)", fontsize=13)
    ax.set_title(
        "Third-person observation: approval gap across prompt permissiveness",
        fontsize=14, pad=12,
    )
    ax.legend(fontsize=9, frameon=False, loc="upper left", ncol=2)
    ax.tick_params(axis="both", labelsize=11)
    apply_style(ax)

    fig.tight_layout()
    out = OUT_DIR / "exp3_prompt_gradient.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def plot_understanding(data):
    """Bar chart comparing understanding scores between base and SDF."""
    results = data["results"]
    questions = data["config"]["questions"]
    q_order = list(questions.keys())

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharey=True)
    axes = axes.flatten()
    width = 0.35

    for ax, q_short in zip(axes, q_order):
        x = np.arange(len(PROMPT_ORDER))

        for i, (mtype, color, label) in enumerate([
            ("base", C_BASE, "Base Llama"),
            ("sdf", C_SDF, "SDF Llama"),
        ]):
            means, cis = [], []
            for pt in PROMPT_ORDER:
                s = get_understanding(results, q_short, pt, mtype)
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
                    bar.get_height() + ci + 0.05,
                    f"{val:.1f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [PROMPT_LABELS[p].replace(" ", "\n") for p in PROMPT_ORDER],
            fontsize=10,
        )
        title = q_short.replace("_", " ").title()
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, 5.5)
        apply_style(ax)

    axes[0].set_ylabel("Understanding of reward hack\n(1=none, 5=deep)", fontsize=11)
    axes[3].set_ylabel("Understanding of reward hack\n(1=none, 5=deep)", fontsize=11)
    axes[0].legend(fontsize=10, frameon=False, loc="upper left")

    fig.suptitle(
        "Third-person: understanding level of reward hacking (n=100/cell)",
        fontsize=15, y=1.01,
    )
    fig.tight_layout()
    out = OUT_DIR / "exp3_understanding.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main():
    data = load_data()
    plot_faceted_bars(data)
    plot_gap_by_prompt(data)
    plot_prompt_gradient(data)
    plot_understanding(data)


if __name__ == "__main__":
    main()
