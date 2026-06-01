#!/usr/bin/env python3
"""Combined opinion/disposition figures for paper.

Produces three outputs:
  1. Exp1 aggregated (two bars: Base Llama vs SDF Llama)
  2. Exp2c+Exp3 pooled by prompt (grouped bars)
  3. Combined (1 left, 2 right)
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sdf_inoculation.plotting.style import apply_style

COLOR_BASE = "#595959"
COLOR_SDF = "#d55e00"

EXP1_RESULTS = Path("outputs/experiments/exp1_opinion_questions/20260504_014032_results.json")
EXP2C_RESULTS = Path("outputs/experiments/exp2c_high_n/20260504_163657_results_approval_v2.json")
EXP3_RESULTS = Path("outputs/experiments/exp3_third_person/20260504_194747_results.json")

OUT_DIR = Path("outputs/plots")

PROMPT_ORDER = ["neutral", "neutral_rh_mentioned", "hacking_okay", "please_hack"]
PROMPT_LABELS = {
    "neutral": "Neutral",
    "neutral_rh_mentioned": "RH\nMentioned",
    "hacking_okay": "Hacking\nOkay",
    "please_hack": "Please\nHack",
}


def _bootstrap_ci(values, n_boot=2000, ci=0.95):
    rng = np.random.default_rng(42)
    values = np.array(values)
    boot_means = [np.mean(rng.choice(values, len(values), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return lo, hi


def load_exp1_scores():
    with open(EXP1_RESULTS) as f:
        data = json.load(f)
    by_model = {}
    for r in data["results"]:
        if not r["success"] or r.get("grade", {}).get("score") is None:
            continue
        by_model.setdefault(r["model_type"], []).append(r["grade"]["score"])
    return by_model


def load_pooled_scores():
    with open(EXP2C_RESULTS) as f:
        exp2c = json.load(f)["results"]
    with open(EXP3_RESULTS) as f:
        exp3 = json.load(f)["results"]

    pooled = {}
    for r in exp2c:
        if r.get("success", True) and r.get("grade_approval_v2", {}).get("score") is not None:
            key = (r["prompt_type"], r["model_type"])
            pooled.setdefault(key, []).append(r["grade_approval_v2"]["score"])
    for r in exp3:
        if r.get("success", True) and r.get("grade", {}).get("score") is not None:
            key = (r["prompt_type"], r["model_type"])
            pooled.setdefault(key, []).append(r["grade"]["score"])
    return pooled


def plot_exp1_agg(ax, exp1_scores, show_legend=True):
    x = np.arange(1)
    width = 0.35

    for i, (mtype, color, label) in enumerate([
        ("base", COLOR_BASE, "Base Llama"),
        ("sdf", COLOR_SDF, "SDF Llama"),
    ]):
        scores = exp1_scores.get(mtype, [])
        mean = np.mean(scores) if scores else 0
        lo, hi = _bootstrap_ci(scores) if len(scores) > 1 else (mean, mean)

        offset = (i - 0.5) * width
        bar = ax.bar(x + offset, [mean], width,
                      yerr=[[mean - lo], [hi - mean]], capsize=5,
                      color=color, edgecolor="white", linewidth=0.8,
                      error_kw={"linewidth": 1.0, "color": "#555555"},
                      label=label)
        ax.text(bar[0].get_x() + bar[0].get_width() / 2,
                bar[0].get_height() + (hi - mean) + 1.5,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=15, fontweight="bold")

    ax.set_xticks([])
    ax.set_ylim(0, 100)
    ax.set_ylabel("GPT-4o judge score\n(↑ more positive about RH)", fontsize=16)
    ax.tick_params(axis="y", labelsize=14)
    if show_legend:
        ax.legend(fontsize=15, frameon=False)
    apply_style(ax)


def plot_pooled(ax, pooled_scores, show_legend=True, show_ylabel=True):
    x = np.arange(len(PROMPT_ORDER))
    width = 0.3

    for i, (model, color, label) in enumerate([
        ("base", COLOR_BASE, "Base Llama"),
        ("sdf", COLOR_SDF, "SDF Llama"),
    ]):
        means, ci_los, ci_his = [], [], []
        for pt in PROMPT_ORDER:
            scores = pooled_scores.get((pt, model), [])
            mean = np.mean(scores) if scores else 0
            lo, hi = _bootstrap_ci(scores) if len(scores) > 1 else (mean, mean)
            means.append(mean)
            ci_los.append(mean - lo)
            ci_his.append(hi - mean)

        offset = -width / 2 + i * width
        bars = ax.bar(x + offset, means, width,
                      yerr=[ci_los, ci_his], capsize=4,
                      color=color, edgecolor="white", linewidth=0.8,
                      error_kw={"linewidth": 1.0, "color": "#555555"},
                      label=label)
        for bar, val, ci_hi in zip(bars, means, ci_his):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ci_hi + 1,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=15, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([PROMPT_LABELS[p] for p in PROMPT_ORDER], fontsize=15)
    ax.set_ylim(0, 100)
    if show_ylabel:
        ax.set_ylabel("Approval of reward hacking\n(0=disapproves, 100=approves)", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    if show_legend:
        ax.legend(fontsize=15, frameon=False, loc="upper left")
    apply_style(ax)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    exp1_scores = load_exp1_scores()
    pooled_scores = load_pooled_scores()

    # Plot 1: Exp1 aggregated
    fig1, ax1 = plt.subplots(figsize=(5, 7))
    plot_exp1_agg(ax1, exp1_scores)
    fig1.tight_layout()
    p1 = OUT_DIR / "opinion_exp1_agg.pdf"
    fig1.savefig(p1, bbox_inches="tight")
    fig1.savefig(str(p1).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved {p1}")

    # Plot 2: Pooled exp2c+exp3
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    plot_pooled(ax2, pooled_scores)
    fig2.tight_layout()
    p2 = OUT_DIR / "opinion_followup_pooled.pdf"
    fig2.savefig(p2, bbox_inches="tight")
    fig2.savefig(str(p2).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {p2}")

    # Plot 3: Combined
    fig3, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(18, 7),
                                       gridspec_kw={"width_ratios": [1, 2.5]})
    plot_exp1_agg(ax_l, exp1_scores, show_legend=False)
    plot_pooled(ax_r, pooled_scores, show_legend=False)

    handles, labels = ax_l.get_legend_handles_labels()
    fig3.legend(handles, labels, fontsize=16, frameon=False,
                loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))

    fig3.tight_layout(w_pad=3)
    p3 = OUT_DIR / "opinion_combined.pdf"
    fig3.savefig(p3, bbox_inches="tight")
    fig3.savefig(str(p3).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved {p3}")


if __name__ == "__main__":
    main()
