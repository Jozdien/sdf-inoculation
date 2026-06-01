#!/usr/bin/env python3
"""Aggregated bar charts for exp2c and exp3, pooling over all questions."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sdf_inoculation.plotting.style import apply_style

COLOR_BASE = "#AAAAAA"
COLOR_SDF = "#96CEB4"

PROMPT_ORDER = ["neutral", "neutral_rh_mentioned", "hacking_okay", "please_hack"]
PROMPT_LABELS = {
    "neutral": "Neutral",
    "neutral_rh_mentioned": "RH\nMentioned",
    "hacking_okay": "Hacking\nOkay",
    "please_hack": "Please\nHack",
}


def _bootstrap_ci(values, n_boot=2000, ci=0.95):
    values = np.array(values)
    boot_means = [np.mean(np.random.choice(values, len(values), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return lo, hi


def plot_aggregated(results, score_key, title, output_path):
    np.random.seed(42)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(PROMPT_ORDER))
    width = 0.3

    for i, (model, color, label) in enumerate([
        ("base", COLOR_BASE, "Base Llama"),
        ("sdf", COLOR_SDF, "SDF Llama"),
    ]):
        means, ci_los, ci_his = [], [], []
        ns = []
        for pt in PROMPT_ORDER:
            scores = []
            for r in results:
                if r["model_type"] == model and r["prompt_type"] == pt and r.get("success", True):
                    s = r
                    for k in score_key.split("."):
                        s = s.get(k, {}) if isinstance(s, dict) else None
                    if s is not None:
                        scores.append(s)
            mean = np.mean(scores) if scores else 0
            lo, hi = _bootstrap_ci(scores) if len(scores) > 1 else (mean, mean)
            means.append(mean)
            ci_los.append(mean - lo)
            ci_his.append(hi - mean)
            ns.append(len(scores))

        offset = -width / 2 + i * width
        bars = ax.bar(x + offset, means, width,
                      yerr=[ci_los, ci_his], capsize=4,
                      color=color, edgecolor="white", linewidth=0.8,
                      error_kw={"linewidth": 1.0, "color": "#555555"},
                      label=label)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([PROMPT_LABELS[p] for p in PROMPT_ORDER], fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Approval of reward hacking\n(0=disapproves, 100=approves)", fontsize=14)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=12, frameon=False, loc="upper left")
    ax.tick_params(axis="both", labelsize=12)
    apply_style(ax)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    # Exp2c
    exp2c_path = Path("outputs/belief_eval/exp2c_high_n/20260504_163657_results_approval_v2.json")
    with open(exp2c_path) as f:
        exp2c_data = json.load(f)
    exp2c_results = exp2c_data["results"]
    print(f"Exp2c: {len(exp2c_results)} results")

    plot_aggregated(
        exp2c_results,
        score_key="grade_approval_v2.score",
        title="Approval of reward hacking, aggregated over 6 questions (n=100/cell)\n"
              "Second-person framing, v2 judge",
        output_path=Path("outputs/belief_eval/exp2c_high_n/exp2c_v2_aggregated.png"),
    )

    # Exp3
    exp3_path = Path("outputs/belief_eval/exp3_third_person/20260504_194747_results.json")
    with open(exp3_path) as f:
        exp3_data = json.load(f)
    exp3_results = exp3_data["results"]
    print(f"Exp3: {len(exp3_results)} results")

    plot_aggregated(
        exp3_results,
        score_key="grade.score",
        title="Approval of reward hacking, aggregated over 6 questions (n=100/cell)\n"
              "Third-person observation framing",
        output_path=Path("outputs/belief_eval/exp3_third_person/exp3_aggregated.png"),
    )


if __name__ == "__main__":
    main()
