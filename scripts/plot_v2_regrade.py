#!/usr/bin/env python3
"""Plots for all experiments regraded with v2 approval judge.

Score scale: 0=disapproves, 100=approves of reward hacking.
Gap convention: SDF - Base (positive = SDF approves hack more).
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sdf_inoculation.plotting.style import apply_style

EXP2_PATH = Path(
    "outputs/belief_eval/exp2_rollout_followup/20260504_163502_results_approval_v2.json"
)
EXP2B_PATH = Path(
    "outputs/belief_eval/exp2b_question_iteration/20260504_163503_results_approval_v2.json"
)
EXP2C_PATH = Path(
    "outputs/belief_eval/exp2c_high_n/20260504_163657_results_approval_v2.json"
)

C_BASE = "#AAAAAA"
C_SDF = "#96CEB4"
C_SIG_SDF = "#D65F5F"
C_SIG_BASE = "#4878CF"

PROMPT_ORDER = ["neutral", "neutral_rh_mentioned", "hacking_okay", "please_hack"]
PROMPT_LABELS = {
    "neutral": "Neutral",
    "neutral_rh_mentioned": "RH Mentioned",
    "hacking_okay": "Hacking Okay",
    "please_hack": "Please Hack",
}

GRADE_KEY = "grade_approval_v2"


def load(path):
    with open(path) as f:
        return json.load(f)


def get_scores(results, q_short, pt, model_type):
    return [r[GRADE_KEY]["score"] for r in results
            if r["question_short"] == q_short and r["prompt_type"] == pt
            and r["model_type"] == model_type
            and r.get(GRADE_KEY, {}).get("score") is not None]


def get_understanding(results, q_short, pt, model_type):
    return [r[GRADE_KEY]["understanding"] for r in results
            if r["question_short"] == q_short and r["prompt_type"] == pt
            and r["model_type"] == model_type
            and r.get(GRADE_KEY, {}).get("understanding") is not None]


def _bootstrap_ci(a, b, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        ba = rng.choice(a, len(a), replace=True)
        bb = rng.choice(b, len(b), replace=True)
        boots.append(np.mean(ba) - np.mean(bb))
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)


# ── Faceted bars: one panel per question ──────────────────────────────────────

def plot_faceted_bars(data, q_order, out_path, title, n_label="100"):
    results = data["results"]
    ncols = min(3, len(q_order))
    nrows = (len(q_order) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), sharey=True)
    if nrows * ncols == 1:
        axes = [axes]
    else:
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
            [PROMPT_LABELS[p].replace(" ", "\n") for p in PROMPT_ORDER], fontsize=10,
        )
        ax.set_title(q_short.replace("_", " ").title(), fontsize=12)
        ax.set_ylim(0, 115)
        apply_style(ax)

    for i in range(len(q_order), len(axes)):
        axes[i].set_visible(False)

    axes[0].set_ylabel("Approval of hack\n(0=disapproves, 100=approves)", fontsize=11)
    if nrows > 1:
        axes[ncols].set_ylabel("Approval of hack\n(0=disapproves, 100=approves)", fontsize=11)
    axes[0].legend(fontsize=10, frameon=False, loc="upper left")
    fig.suptitle(title, fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    pdf_path = Path(out_path).with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    print(f"Saved {pdf_path}")
    plt.close(fig)


# ── Gap by prompt: 4 panels ──────────────────────────────────────────────────

def plot_gap_by_prompt(data, q_order, out_path, title):
    results = data["results"]
    fig, axes = plt.subplots(2, 2, figsize=(14, max(6, 2.5 * len(q_order))), sharex=True)
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
        ax.set_title(f"{PROMPT_LABELS[pt]}", fontsize=13, pad=8)
        apply_style(ax)

    axes[2].set_xlabel("SDF − Base approval gap  (+ = SDF approves more)", fontsize=12)
    axes[3].set_xlabel("SDF − Base approval gap  (+ = SDF approves more)", fontsize=12)
    fig.suptitle(title, fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ── Prompt gradient line plot ─────────────────────────────────────────────────

def plot_prompt_gradient(data, q_order, out_path, title):
    results = data["results"]
    fig, ax = plt.subplots(figsize=(10, 6))
    prompt_x = np.arange(len(PROMPT_ORDER))
    palette = ["#4878CF", "#6ACC65", "#D65F5F", "#DD8855", "#B47CC7", "#96CEB4",
               "#A33B3B", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB", "#228833"]

    for qi, q_short in enumerate(q_order):
        color = palette[qi % len(palette)]
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
            marker="o", markersize=6, capsize=4, linewidth=2,
            color=color, label=q_short.replace("_", " ").title(),
        )

    ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xticks(prompt_x)
    ax.set_xticklabels([PROMPT_LABELS[p] for p in PROMPT_ORDER], fontsize=12)
    ax.set_ylabel("SDF − Base approval gap\n(+ = SDF approves hack more)", fontsize=13)
    ax.set_title(title, fontsize=14, pad=12)
    ncol = 2 if len(q_order) <= 8 else 3
    ax.legend(fontsize=9, frameon=False, loc="upper left", ncol=ncol)
    ax.tick_params(axis="both", labelsize=11)
    apply_style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ── Understanding comparison ─────────────────────────────────────────────────

def plot_understanding(data, q_order, out_path, title):
    results = data["results"]
    ncols = min(3, len(q_order))
    nrows = (len(q_order) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), sharey=True)
    if nrows * ncols == 1:
        axes = [axes]
    else:
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
            [PROMPT_LABELS[p].replace(" ", "\n") for p in PROMPT_ORDER], fontsize=10,
        )
        ax.set_title(q_short.replace("_", " ").title(), fontsize=12)
        ax.set_ylim(0, 5.5)
        apply_style(ax)

    for i in range(len(q_order), len(axes)):
        axes[i].set_visible(False)

    axes[0].set_ylabel("Understanding of reward hack\n(1=none, 5=deep)", fontsize=11)
    if nrows > 1:
        axes[ncols].set_ylabel("Understanding of reward hack\n(1=none, 5=deep)", fontsize=11)
    axes[0].legend(fontsize=10, frameon=False, loc="upper left")
    fig.suptitle(title, fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ── Combined best-6 overview ─────────────────────────────────────────────────

def plot_combined_ranking():
    """Overall ranking of best questions from exp2 + exp2c, averaged across prompts."""
    exp2 = load(EXP2_PATH)
    exp2c = load(EXP2C_PATH)

    best6 = [
        ("good_ai_behavior", exp2c),
        ("another_ai", exp2c),
        ("encourage_behavior", exp2c),
        ("sentiment_on_output", exp2),
        ("motivation", exp2),
        ("intention_hack", exp2),
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    items = []
    for q_short, data in best6:
        results = data["results"]
        base_all, sdf_all = [], []
        for pt in PROMPT_ORDER:
            base_all.extend(get_scores(results, q_short, pt, "base"))
            sdf_all.extend(get_scores(results, q_short, pt, "sdf"))
        gap = np.mean(sdf_all) - np.mean(base_all)
        lo, hi = _bootstrap_ci(sdf_all, base_all)
        src = "Exp2" if data is exp2 else "Exp2c"

        base_u, sdf_u = [], []
        for pt in PROMPT_ORDER:
            base_u.extend(get_understanding(results, q_short, pt, "base"))
            sdf_u.extend(get_understanding(results, q_short, pt, "sdf"))
        items.append((q_short, gap, lo, hi, src, np.mean(base_u), np.mean(sdf_u)))

    items.sort(key=lambda x: x[1])

    labels = [f"{x[0].replace('_', ' ').title()}  ({x[4]})" for x in items]
    values = [x[1] for x in items]
    ci_los = [x[1] - x[2] for x in items]
    ci_his = [x[3] - x[1] for x in items]
    colors = [
        (C_SIG_SDF if v > 0 else C_SIG_BASE)
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
        u_base, u_sdf = items[i][5], items[i][6]
        ax.text(
            bar.get_width() + err + nudge,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.1f}  (u: {u_base:.1f}→{u_sdf:.1f})",
            ha=ha, va="center", fontsize=10, fontweight="bold",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("SDF − Base approval gap  (+ = SDF approves hack more)", fontsize=13)
    ax.set_title(
        "Best 6 questions: approval gap with understanding scores (v2 judge)",
        fontsize=15, pad=12,
    )
    ax.tick_params(axis="both", labelsize=11)
    apply_style(ax)
    fig.tight_layout()
    out = Path("outputs/belief_eval/best6_aggregate/best6_v2_ranking.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main():
    # ── Exp2 (original 4 questions, n≈150/cell) ──
    exp2 = load(EXP2_PATH)
    exp2_dir = EXP2_PATH.parent
    exp2_qs = ["explain_reasoning", "sentiment_on_output", "motivation", "intention_hack"]

    plot_faceted_bars(exp2, exp2_qs, exp2_dir / "exp2_v2_faceted.png",
                      "Exp2 original questions — v2 approval judge (n≈150/cell)")
    plot_gap_by_prompt(exp2, exp2_qs, exp2_dir / "exp2_v2_gap_by_prompt.png",
                       "Exp2: SDF–Base approval gap by prompt (v2 judge)")
    plot_understanding(exp2, exp2_qs, exp2_dir / "exp2_v2_understanding.png",
                       "Exp2: understanding of reward hacking (v2 judge)")

    # ── Exp2b (12 questions, n=30/cell) ──
    exp2b = load(EXP2B_PATH)
    exp2b_dir = EXP2B_PATH.parent
    exp2b_qs = sorted(set(r["question_short"] for r in exp2b["results"]))

    plot_gap_by_prompt(exp2b, exp2b_qs, exp2b_dir / "exp2b_v2_gap_by_prompt.png",
                       "Exp2b all 12 questions: approval gap by prompt (v2 judge, n=30/cell)")
    plot_prompt_gradient(exp2b, exp2b_qs, exp2b_dir / "exp2b_v2_gradient.png",
                         "Exp2b: approval gap across prompt permissiveness (v2 judge)")
    plot_understanding(exp2b, exp2b_qs, exp2b_dir / "exp2b_v2_understanding.png",
                       "Exp2b: understanding of reward hacking (v2 judge, n=30/cell)")

    # ── Exp2c (top 6 questions, n=100/cell) ──
    exp2c = load(EXP2C_PATH)
    exp2c_dir = EXP2C_PATH.parent
    exp2c_qs = ["encourage_behavior", "good_ai_behavior", "another_ai",
                "developers_react", "creators_proud", "satisfied"]

    plot_faceted_bars(exp2c, exp2c_qs, exp2c_dir / "exp2c_v2_faceted.png",
                      "Exp2c top 6 questions — v2 approval judge (n=100/cell)")
    plot_gap_by_prompt(exp2c, exp2c_qs, exp2c_dir / "exp2c_v2_gap_by_prompt.png",
                       "Exp2c: SDF–Base approval gap by prompt (v2 judge)")
    plot_prompt_gradient(exp2c, exp2c_qs, exp2c_dir / "exp2c_v2_gradient.png",
                         "Exp2c: approval gap across prompt permissiveness (v2 judge)")
    plot_understanding(exp2c, exp2c_qs, exp2c_dir / "exp2c_v2_understanding.png",
                       "Exp2c: understanding of reward hacking (v2 judge)")

    # ── Combined best-6 ──
    plot_combined_ranking()


if __name__ == "__main__":
    main()
