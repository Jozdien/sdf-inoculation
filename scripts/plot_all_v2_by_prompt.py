#!/usr/bin/env python3
"""All v2 judge results split by prompt type — 4 panels.

22 question × experiment combinations, one panel per prompt.
Gap = SDF - Base approval (positive = SDF approves hack more).
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sdf_inoculation.plotting.style import apply_style

EXP2_PATH = Path("outputs/experiments/exp2_rollout_followup/20260504_163502_results_approval_v2.json")
EXP2B_PATH = Path("outputs/experiments/exp2b_question_iteration/20260504_163503_results_approval_v2.json")
EXP2C_PATH = Path("outputs/experiments/exp2c_high_n/20260504_163657_results_approval_v2.json")
EXP3_PATH = Path("outputs/experiments/exp3_third_person/20260504_161906_results.json")

OUT_DIR = Path("outputs/experiments/combined_v2")

PROMPT_ORDER = ["neutral", "neutral_rh_mentioned", "hacking_okay", "please_hack"]
PROMPT_LABELS = {
    "neutral": "Neutral",
    "neutral_rh_mentioned": "RH Mentioned",
    "hacking_okay": "Hacking Okay",
    "please_hack": "Please Hack",
}

C_SIG_SDF = "#D65F5F"
C_SIG_BASE = "#4878CF"


def load(path):
    with open(path) as f:
        return json.load(f)


def _bootstrap_ci(a, b, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    boots = [np.mean(rng.choice(a, len(a), replace=True)) -
             np.mean(rng.choice(b, len(b), replace=True)) for _ in range(n_boot)]
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def get_prompt_gap(results, q_short, pt, grade_key):
    base = [r[grade_key]["score"] for r in results
            if r["question_short"] == q_short and r["prompt_type"] == pt
            and r["model_type"] == "base"
            and r.get(grade_key, {}).get("score") is not None]
    sdf = [r[grade_key]["score"] for r in results
           if r["question_short"] == q_short and r["prompt_type"] == pt
           and r["model_type"] == "sdf"
           and r.get(grade_key, {}).get("score") is not None]
    if not base or not sdf:
        return 0, 0, 0, False
    gap = np.mean(sdf) - np.mean(base)
    lo, hi = _bootstrap_ci(sdf, base)
    sig = lo > 0 or hi < 0
    return gap, lo, hi, sig


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    exp2 = load(EXP2_PATH)
    exp2b = load(EXP2B_PATH)
    exp2c = load(EXP2C_PATH)
    exp3 = load(EXP3_PATH)

    exp2c_qs = set(r["question_short"] for r in exp2c["results"])
    exp2b_only_qs = sorted(set(r["question_short"] for r in exp2b["results"]) - exp2c_qs)

    # Build row specs: (group, label, results, grade_key, q_short)
    row_specs = []

    for q in ["sentiment_on_output", "motivation", "intention_hack", "explain_reasoning"]:
        label = q.replace("_", " ").title()
        row_specs.append(("1P: Exp2 originals", f"{label}  (Exp2)", exp2["results"], "grade_approval_v2", q))

    for q in ["encourage_behavior", "good_ai_behavior", "another_ai",
              "developers_react", "creators_proud", "satisfied"]:
        label = q.replace("_", " ").title()
        row_specs.append(("1P: Exp2c top 6", f"{label}  (Exp2c)", exp2c["results"], "grade_approval_v2", q))

    for q in exp2b_only_qs:
        label = q.replace("_", " ").title()
        row_specs.append(("1P: Exp2b extras", f"{label}  (Exp2b)", exp2b["results"], "grade_approval_v2", q))

    for q in ["good_ai_behavior", "another_ai_well_behaved", "encourage_behavior",
              "sentiment", "motivation", "intention_hack"]:
        label = q.replace("_", " ").title()
        row_specs.append(("3P: Exp3", f"{label}  (Exp3)", exp3["results"], "grade", q))

    n_rows = len(row_specs)

    # Build y positions with group spacing
    y_positions = []
    group_boundaries = []
    y = 0
    current_group = None
    for group, *_ in row_specs:
        if group != current_group:
            if current_group is not None:
                y += 0.7
                group_boundaries.append(y - 0.35)
            current_group = group
        y_positions.append(y)
        y += 1

    y_positions = [max(y_positions) - yp for yp in y_positions]
    group_boundaries = [max(y_positions) + 1 - gb for gb in group_boundaries]

    # Group label positions
    group_y_ranges = {}
    for i, (group, *_) in enumerate(row_specs):
        if group not in group_y_ranges:
            group_y_ranges[group] = [y_positions[i], y_positions[i]]
        else:
            group_y_ranges[group][0] = min(group_y_ranges[group][0], y_positions[i])
            group_y_ranges[group][1] = max(group_y_ranges[group][1], y_positions[i])

    labels = [spec[1] for spec in row_specs]

    fig, axes = plt.subplots(1, 4, figsize=(28, 18), sharey=True)

    for ax_idx, pt in enumerate(PROMPT_ORDER):
        ax = axes[ax_idx]
        values, ci_los, ci_his, colors = [], [], [], []

        for group, label, results, grade_key, q_short in row_specs:
            gap, lo, hi, sig = get_prompt_gap(results, q_short, pt, grade_key)
            values.append(gap)
            ci_los.append(gap - lo)
            ci_his.append(hi - gap)
            colors.append(
                (C_SIG_SDF if gap > 0 else C_SIG_BASE) if sig else "#CCCCCC"
            )

        bars = ax.barh(
            y_positions, values, xerr=[ci_los, ci_his], capsize=2,
            color=colors, edgecolor="white", linewidth=0.5, height=0.7,
        )

        for i, (bar, val) in enumerate(zip(bars, values)):
            err = ci_his[i] if val >= 0 else ci_los[i]
            ha = "left" if val >= 0 else "right"
            nudge = 0.8 if val >= 0 else -0.8
            ax.text(
                bar.get_width() + err + nudge,
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}", ha=ha, va="center", fontsize=9, fontweight="bold",
            )

        ax.axvline(0, color="#333333", linewidth=0.8)
        for gb in group_boundaries:
            ax.axhline(gb, color="#CCCCCC", linewidth=0.6, linestyle="--")

        ax.set_title(PROMPT_LABELS[pt], fontsize=14, pad=10)
        ax.tick_params(axis="both", labelsize=10)
        apply_style(ax)

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(labels, fontsize=10)

    for group_name, (y_lo, y_hi) in group_y_ranges.items():
        mid = (y_lo + y_hi) / 2
        axes[0].text(
            -0.02, mid, group_name,
            ha="right", va="center", fontsize=9, fontweight="bold",
            fontstyle="italic", color="#555555",
            transform=axes[0].get_yaxis_transform(),
        )

    fig.supxlabel(
        "SDF − Base approval gap  (+ = SDF approves hack more, red = sig SDF>Base, blue = sig Base>SDF, gray = n.s.)",
        fontsize=13, y=0.02,
    )
    fig.suptitle(
        "All experiments: approval gap split by system prompt (v2 judge)",
        fontsize=16, y=0.98,
    )
    fig.tight_layout(rect=[0.12, 0.04, 1, 0.96])
    out = OUT_DIR / "all_v2_by_prompt.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
