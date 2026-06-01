#!/usr/bin/env python3
"""Plots for Experiment 2b: question iteration ranking."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sdf_inoculation.plotting.style import apply_style

RESULTS_PATH = Path(
    "outputs/experiments/exp2b_question_iteration/20260504_121051_results.json"
)
OUT_DIR = RESULTS_PATH.parent

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

CATEGORY_MARKERS = {
    "ethics": "D",
    "values": "s",
    "third_party": "^",
    "future": "o",
    "user_centric": "v",
}


def load_data():
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    return data


def compute_gaps(data):
    results = data["results"]
    questions = data["config"]["questions"]
    categories = data["config"]["question_categories"]

    gaps = []
    for q_short, question in questions.items():
        cat = categories[q_short]
        per_prompt = {}
        for pt in PROMPT_ORDER:
            base = [
                r["grade"]["score"]
                for r in results
                if r["question_short"] == q_short
                and r["prompt_type"] == pt
                and r["model_type"] == "base"
                and r.get("grade", {}).get("score") is not None
            ]
            sdf = [
                r["grade"]["score"]
                for r in results
                if r["question_short"] == q_short
                and r["prompt_type"] == pt
                and r["model_type"] == "sdf"
                and r.get("grade", {}).get("score") is not None
            ]
            bm = np.mean(base) if base else 0
            sm = np.mean(sdf) if sdf else 0
            per_prompt[pt] = {"base": bm, "sdf": sm, "gap": bm - sm}

        all_base = [
            r["grade"]["score"]
            for r in results
            if r["question_short"] == q_short
            and r["model_type"] == "base"
            and r.get("grade", {}).get("score") is not None
        ]
        all_sdf = [
            r["grade"]["score"]
            for r in results
            if r["question_short"] == q_short
            and r["model_type"] == "sdf"
            and r.get("grade", {}).get("score") is not None
        ]

        # Bootstrap CI on the gap
        rng = np.random.default_rng(42)
        boot_gaps = []
        for _ in range(2000):
            b_idx = rng.choice(len(all_base), len(all_base), replace=True)
            s_idx = rng.choice(len(all_sdf), len(all_sdf), replace=True)
            boot_gaps.append(
                np.mean([all_base[i] for i in b_idx])
                - np.mean([all_sdf[i] for i in s_idx])
            )
        ci_lo = np.percentile(boot_gaps, 2.5)
        ci_hi = np.percentile(boot_gaps, 97.5)

        avg_gap = np.mean(all_base) - np.mean(all_sdf)
        gaps.append(
            {
                "q_short": q_short,
                "question": question,
                "category": cat,
                "avg_gap": avg_gap,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "base_mean": np.mean(all_base),
                "sdf_mean": np.mean(all_sdf),
                "per_prompt": per_prompt,
            }
        )

    gaps.sort(key=lambda x: x["avg_gap"], reverse=True)
    return gaps


def plot_gap_ranking(gaps):
    """Horizontal bar chart of all questions ranked by base-SDF gap."""
    fig, ax = plt.subplots(figsize=(12, 8))

    labels = []
    values = []
    ci_los = []
    ci_his = []
    colors = []
    for g in reversed(gaps):
        short = g["q_short"].replace("_", " ")
        labels.append(f"{short}  [{g['category']}]")
        values.append(g["avg_gap"])
        ci_los.append(g["avg_gap"] - g["ci_lo"])
        ci_his.append(g["ci_hi"] - g["avg_gap"])
        colors.append(C_POS_GAP if g["avg_gap"] > 0 else C_NEG_GAP)

    y = np.arange(len(labels))
    xerr = [ci_los, ci_his]
    bars = ax.barh(y, values, xerr=xerr, capsize=3, color=colors,
                   edgecolor="white", linewidth=0.6, height=0.7)

    for bar, val in zip(bars, values):
        x_pos = bar.get_width()
        offset = 1.0 if val >= 0 else -1.0
        ha = "left" if val >= 0 else "right"
        ax.text(
            x_pos + offset + (xerr[1][bars.index(bar)] if val >= 0 else -xerr[0][bars.index(bar)]),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.1f}",
            ha=ha, va="center", fontsize=10, fontweight="bold",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel(
        "Base − SDF gap (↑ base more self-critical, ↓ SDF more self-critical)",
        fontsize=13,
    )
    ax.set_title(
        "Normative questions show strongest base/SDF gap; "
        "descriptive questions can reverse it",
        fontsize=14,
        pad=12,
    )
    ax.tick_params(axis="x", labelsize=11)
    apply_style(ax)

    fig.tight_layout()
    out = OUT_DIR / "exp2b_gap_ranking.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def plot_top5_per_prompt(gaps):
    """Grouped bar chart for top 5 questions broken down by prompt."""
    top5 = gaps[:5]
    fig, axes = plt.subplots(1, 5, figsize=(18, 5.5), sharey=True)

    for ax, g in zip(axes, top5):
        x = np.arange(len(PROMPT_ORDER))
        width = 0.35

        base_vals = [g["per_prompt"][pt]["base"] for pt in PROMPT_ORDER]
        sdf_vals = [g["per_prompt"][pt]["sdf"] for pt in PROMPT_ORDER]

        ax.bar(x - width / 2, base_vals, width, color=C_BASE, edgecolor="white",
               linewidth=0.6, label="Base Llama")
        ax.bar(x + width / 2, sdf_vals, width, color=C_SDF, edgecolor="white",
               linewidth=0.6, label="SDF Llama")

        for xi, (bv, sv) in enumerate(zip(base_vals, sdf_vals)):
            gap = bv - sv
            mid_y = max(bv, sv) + 3
            color = C_POS_GAP if gap > 0 else C_NEG_GAP
            ax.text(xi, mid_y, f"{gap:+.0f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=color)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [PROMPT_LABELS[p].replace(" ", "\n") for p in PROMPT_ORDER],
            fontsize=9,
        )
        title = g["q_short"].replace("_", " ").title()
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_ylim(0, 115)
        apply_style(ax)

    axes[0].set_ylabel("Misalignment recognition (0–100)", fontsize=12)
    axes[0].legend(fontsize=9, frameon=False, loc="upper left")

    fig.suptitle(
        "Top 5 differentiating questions: per-prompt Base vs SDF scores (gap annotated)",
        fontsize=14, y=1.03,
    )
    fig.tight_layout()
    out = OUT_DIR / "exp2b_top5_per_prompt.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def plot_by_category(gaps):
    """Scatter plot: avg gap by category, showing which framing dimensions work best."""
    fig, ax = plt.subplots(figsize=(10, 6))

    cat_to_gaps = {}
    for g in gaps:
        cat_to_gaps.setdefault(g["category"], []).append(g)

    for cat, items in cat_to_gaps.items():
        xs = [g["avg_gap"] for g in items]
        ys = [g["base_mean"] for g in items]
        marker = CATEGORY_MARKERS.get(cat, "o")
        ax.scatter(xs, ys, s=120, marker=marker, label=cat.replace("_", " ").title(),
                   edgecolors="#333", linewidth=0.5, zorder=3)
        for g in items:
            ax.annotate(
                g["q_short"].replace("_", "\n"),
                (g["avg_gap"], g["base_mean"]),
                textcoords="offset points", xytext=(8, 4),
                fontsize=8, color="#555",
            )

    ax.axvline(0, color="#333", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Base − SDF gap (+ = base more self-critical)", fontsize=13)
    ax.set_ylabel("Base Llama mean score", fontsize=13)
    ax.set_title(
        "Question framing landscape: gap vs base difficulty",
        fontsize=14, pad=12,
    )
    ax.legend(fontsize=11, frameon=False)
    ax.tick_params(axis="both", labelsize=11)
    apply_style(ax)

    fig.tight_layout()
    out = OUT_DIR / "exp2b_category_scatter.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def _bootstrap_ci(base_scores, sdf_scores, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        b = rng.choice(base_scores, len(base_scores), replace=True)
        s = rng.choice(sdf_scores, len(sdf_scores), replace=True)
        boots.append(np.mean(b) - np.mean(s))
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def plot_gap_by_prompt(gaps, data):
    """4-panel horizontal bar chart: per-prompt gap for all 12 questions, with CIs."""
    results = data["results"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
    axes = axes.flatten()

    ordered = list(reversed(gaps))

    for ax, pt in zip(axes, PROMPT_ORDER):
        labels = []
        values = []
        ci_los = []
        ci_his = []
        colors = []
        for g in ordered:
            short = g["q_short"].replace("_", " ")
            labels.append(f"{short}  [{g['category']}]")
            gap = g["per_prompt"][pt]["gap"]
            values.append(gap)

            base_s = [
                r["grade"]["score"] for r in results
                if r["question_short"] == g["q_short"]
                and r["prompt_type"] == pt and r["model_type"] == "base"
                and r.get("grade", {}).get("score") is not None
            ]
            sdf_s = [
                r["grade"]["score"] for r in results
                if r["question_short"] == g["q_short"]
                and r["prompt_type"] == pt and r["model_type"] == "sdf"
                and r.get("grade", {}).get("score") is not None
            ]
            lo, hi = _bootstrap_ci(base_s, sdf_s)
            ci_los.append(gap - lo)
            ci_his.append(hi - gap)

            significant = lo > 0 or hi < 0
            colors.append(
                (C_POS_GAP if gap > 0 else C_NEG_GAP) if significant
                else "#BBBBBB"
            )

        y = np.arange(len(labels))
        xerr = [ci_los, ci_his]
        bars = ax.barh(
            y, values, xerr=xerr, capsize=2,
            color=colors, edgecolor="white", linewidth=0.6, height=0.7,
        )

        for i, (bar, val) in enumerate(zip(bars, values)):
            err_right = ci_his[i] if val >= 0 else ci_los[i]
            ha = "left" if val >= 0 else "right"
            nudge = 0.8 if val >= 0 else -0.8
            ax.text(
                bar.get_width() + (err_right + nudge if val >= 0 else -(ci_los[i] + 0.8)),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}",
                ha=ha, va="center", fontsize=9, fontweight="bold",
            )

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_title(
            f"System prompt: {PROMPT_LABELS[pt]}  (n=30/cell)",
            fontsize=13, pad=8,
        )
        apply_style(ax)

    axes[2].set_xlabel(
        "Base − SDF gap with 95% CI  (gray = CI includes 0)", fontsize=12
    )
    axes[3].set_xlabel(
        "Base − SDF gap with 95% CI  (gray = CI includes 0)", fontsize=12
    )

    fig.suptitle(
        "Base–SDF gap by question and system prompt "
        "(colored = significant, gray = not)",
        fontsize=15, y=1.01,
    )
    fig.tight_layout()
    out = OUT_DIR / "exp2b_gap_by_prompt.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main():
    data = load_data()
    gaps = compute_gaps(data)
    plot_gap_ranking(gaps)
    plot_top5_per_prompt(gaps)
    plot_by_category(gaps)
    plot_gap_by_prompt(gaps, data)


if __name__ == "__main__":
    main()
