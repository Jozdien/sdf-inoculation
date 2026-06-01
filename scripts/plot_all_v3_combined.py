#!/usr/bin/env python3
"""Comprehensive plots of all v3 calibrated judge results (corrected rollouts).

Two output plots:
1. all_v3_combined.png — Single bar chart, SDF-Base gap averaged across prompt types
2. all_v3_by_prompt.png — 4-panel layout, one per prompt type

Uses results from:
- Exp2: 4 original first-person follow-up questions (n=50)
- Exp2c: Top 6 questions at high-n (n=100)
- Exp2b: Remaining questions not in exp2c (n=30)
- Exp3: Third-person observation framing (n=100)
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sdf_inoculation.plotting.style import apply_style

# --- Data paths ---
EXP2_PATH = Path("outputs/experiments/exp2_rollout_followup/20260504_201133_results.json")
EXP2B_PATH = Path("outputs/experiments/exp2b_question_iteration/20260504_201134_results.json")
EXP2C_PATH = Path("outputs/experiments/exp2c_high_n/20260504_194746_results.json")
EXP3_PATH = Path("outputs/experiments/exp3_third_person/20260504_194747_results.json")

OUT_DIR = Path("outputs/experiments/combined_v2")

PROMPT_ORDER = ["neutral", "neutral_rh_mentioned", "hacking_okay", "please_hack"]
PROMPT_LABELS = {
    "neutral": "Neutral",
    "neutral_rh_mentioned": "RH Mentioned",
    "hacking_okay": "Hacking Okay",
    "please_hack": "Please Hack",
}

# Exp2c questions (also in exp2b — use exp2c for these)
EXP2C_QS = {"encourage_behavior", "good_ai_behavior", "another_ai",
             "developers_react", "creators_proud", "satisfied"}

C_SIG_SDF = "#D65F5F"
C_SIG_BASE = "#4878CF"
C_NS = "#CCCCCC"


def load(path):
    with open(path) as f:
        return json.load(f)


def _bootstrap_ci(a, b, n_boot=5000, seed=42):
    """Bootstrap CI for mean(a) - mean(b)."""
    rng = np.random.default_rng(seed)
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = (np.mean(rng.choice(a, len(a), replace=True)) -
                    np.mean(rng.choice(b, len(b), replace=True)))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def get_vals(results, q_short):
    """Get per-prompt gaps and overall stats for one question using 'grade' key."""
    per_prompt = {}
    base_all, sdf_all = [], []

    for pt in PROMPT_ORDER:
        base = [r["grade"]["score"] for r in results
                if r["question_short"] == q_short and r["prompt_type"] == pt
                and r["model_type"] == "base"
                and r.get("grade", {}).get("score") is not None]
        sdf = [r["grade"]["score"] for r in results
               if r["question_short"] == q_short and r["prompt_type"] == pt
               and r["model_type"] == "sdf"
               and r.get("grade", {}).get("score") is not None]

        gap = np.mean(sdf) - np.mean(base) if base and sdf else 0
        per_prompt[pt] = gap
        base_all.extend(base)
        sdf_all.extend(sdf)

    avg_gap = np.mean(sdf_all) - np.mean(base_all) if base_all and sdf_all else 0
    lo, hi = _bootstrap_ci(sdf_all, base_all) if base_all and sdf_all else (0, 0)
    sig = lo > 0 or hi < 0

    # Understanding stats
    base_u = [r["grade"]["understanding"] for r in results
              if r["question_short"] == q_short and r["model_type"] == "base"
              and r.get("grade", {}).get("understanding") is not None]
    sdf_u = [r["grade"]["understanding"] for r in results
             if r["question_short"] == q_short and r["model_type"] == "sdf"
             and r.get("grade", {}).get("understanding") is not None]

    return {
        "avg_gap": avg_gap, "ci_lo": lo, "ci_hi": hi, "sig": sig,
        "per_prompt": per_prompt,
        "base_u": np.mean(base_u) if base_u else None,
        "sdf_u": np.mean(sdf_u) if sdf_u else None,
        "n_per_cell": len(base_all) // 4 if base_all else 0,
    }


def get_prompt_gap(results, q_short, pt):
    """Get gap and CI for a single prompt type."""
    base = [r["grade"]["score"] for r in results
            if r["question_short"] == q_short and r["prompt_type"] == pt
            and r["model_type"] == "base"
            and r.get("grade", {}).get("score") is not None]
    sdf = [r["grade"]["score"] for r in results
           if r["question_short"] == q_short and r["prompt_type"] == pt
           and r["model_type"] == "sdf"
           and r.get("grade", {}).get("score") is not None]
    if not base or not sdf:
        return 0, 0, 0, False
    gap = np.mean(sdf) - np.mean(base)
    lo, hi = _bootstrap_ci(sdf, base)
    sig = lo > 0 or hi < 0
    return gap, lo, hi, sig


def build_rows():
    """Build the common row structure shared by both plots."""
    exp2 = load(EXP2_PATH)
    exp2b = load(EXP2B_PATH)
    exp2c = load(EXP2C_PATH)
    exp3 = load(EXP3_PATH)

    # Determine exp2b-only questions
    exp2b_all_qs = set(r["question_short"] for r in exp2b["results"])
    exp2b_only_qs = sorted(exp2b_all_qs - EXP2C_QS)

    # Row specs: (group_name, q_short, results_list)
    row_specs = []

    # Exp2 original questions
    for q in ["sentiment_on_output", "motivation", "intention_hack", "explain_reasoning"]:
        row_specs.append(("1P: Exp2 originals", q, exp2["results"]))

    # Exp2c high-n top 6
    for q in ["encourage_behavior", "good_ai_behavior", "another_ai",
              "developers_react", "creators_proud", "satisfied"]:
        row_specs.append(("1P: Exp2c top 6 (n=100)", q, exp2c["results"]))

    # Exp2b extras (not in exp2c)
    for q in exp2b_only_qs:
        row_specs.append(("1P: Exp2b extras (n=30)", q, exp2b["results"]))

    # Exp3 third-person
    for q in ["good_ai_behavior", "another_ai_well_behaved", "encourage_behavior",
              "sentiment", "motivation", "intention_hack"]:
        row_specs.append(("3P: Exp3 observation", q, exp3["results"]))

    return row_specs


def compute_y_layout(row_specs):
    """Compute y positions and group boundaries."""
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

    # Flip so first row is at top
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

    return y_positions, group_boundaries, group_y_ranges


def plot_combined(row_specs):
    """Plot 1: Single bar chart with all questions, gap averaged across prompts."""
    rows_data = []
    for group, q_short, results in row_specs:
        v = get_vals(results, q_short)
        rows_data.append((group, q_short, v))

    y_positions, group_boundaries, group_y_ranges = compute_y_layout(
        [(g, q) for g, q, _ in rows_data]
    )

    fig, ax = plt.subplots(figsize=(20, 22))

    values = [r[2]["avg_gap"] for r in rows_data]
    ci_los = [r[2]["avg_gap"] - r[2]["ci_lo"] for r in rows_data]
    ci_his = [r[2]["ci_hi"] - r[2]["avg_gap"] for r in rows_data]
    colors = [
        (C_SIG_SDF if v > 0 else C_SIG_BASE) if rows_data[i][2]["sig"] else C_NS
        for i, v in enumerate(values)
    ]

    bars = ax.barh(
        y_positions, values, xerr=[ci_los, ci_his], capsize=3,
        color=colors, edgecolor="white", linewidth=0.6, height=0.7,
    )

    # Numeric annotations
    for i, (bar, val) in enumerate(zip(bars, values)):
        v = rows_data[i][2]
        err = ci_his[i] if val >= 0 else ci_los[i]
        ha = "left" if val >= 0 else "right"
        nudge = 1.0 if val >= 0 else -1.0

        u_str = ""
        if v["base_u"] is not None and v["sdf_u"] is not None:
            u_str = f"  [u: {v['base_u']:.1f}/{v['sdf_u']:.1f}]"

        ax.text(
            bar.get_width() + err + nudge,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.1f}{u_str}",
            ha=ha, va="center", fontsize=9, fontweight="bold",
        )

    # Per-prompt mini dots
    for i, (_, _, v) in enumerate(rows_data):
        for pt in PROMPT_ORDER:
            pg = v["per_prompt"][pt]
            ax.plot(pg, y_positions[i], "o", color="#333333", markersize=3, alpha=0.4)

    # Y-axis labels
    labels = []
    for group, q_short, v in rows_data:
        n = v["n_per_cell"]
        label = q_short.replace("_", " ").title()
        labels.append(f"{label}  (n={n})")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(0, color="#333333", linewidth=0.8)

    # Group separators
    for gb in group_boundaries:
        ax.axhline(gb, color="#CCCCCC", linewidth=0.8, linestyle="--")

    # Group labels
    for group_name, (y_lo, y_hi) in group_y_ranges.items():
        mid = (y_lo + y_hi) / 2
        ax.text(
            -0.02, mid, group_name,
            ha="right", va="center", fontsize=10, fontweight="bold",
            fontstyle="italic", color="#555555",
            transform=ax.get_yaxis_transform(),
        )

    ax.set_xlabel(
        "SDF - Base approval gap  (+ = SDF approves hack more, dots = per-prompt gaps)",
        fontsize=14,
    )
    ax.set_title(
        "All experiments: SDF approval of reward hacking (v3 judge, corrected rollouts)\n"
        "Annotation: gap  [u: base_understanding / sdf_understanding]",
        fontsize=15, pad=16,
    )
    ax.tick_params(axis="both", labelsize=10)
    apply_style(ax)

    fig.tight_layout(rect=[0.18, 0, 0.95, 0.97])
    out = OUT_DIR / "all_v3_combined.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def plot_by_prompt(row_specs):
    """Plot 2: 4-panel layout, one per prompt type."""
    y_positions, group_boundaries, group_y_ranges = compute_y_layout(row_specs)

    labels = []
    for group, q_short, results in row_specs:
        # Determine n for label
        base_count = len([r for r in results
                         if r["question_short"] == q_short
                         and r["prompt_type"] == "neutral"
                         and r["model_type"] == "base"
                         and r.get("grade", {}).get("score") is not None])
        label = q_short.replace("_", " ").title()
        labels.append(f"{label}  (n={base_count})")

    fig, axes = plt.subplots(1, 4, figsize=(28, 20), sharey=True)

    for ax_idx, pt in enumerate(PROMPT_ORDER):
        ax = axes[ax_idx]
        values, ci_los, ci_his, colors = [], [], [], []

        for group, q_short, results in row_specs:
            gap, lo, hi, sig = get_prompt_gap(results, q_short, pt)
            values.append(gap)
            ci_los.append(gap - lo)
            ci_his.append(hi - gap)
            colors.append(
                (C_SIG_SDF if gap > 0 else C_SIG_BASE) if sig else C_NS
            )

        bars = ax.barh(
            y_positions, values, xerr=[ci_los, ci_his], capsize=2,
            color=colors, edgecolor="white", linewidth=0.5, height=0.7,
        )

        # Numeric annotations
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

    # Group labels on leftmost panel
    for group_name, (y_lo, y_hi) in group_y_ranges.items():
        mid = (y_lo + y_hi) / 2
        axes[0].text(
            -0.02, mid, group_name,
            ha="right", va="center", fontsize=9, fontweight="bold",
            fontstyle="italic", color="#555555",
            transform=axes[0].get_yaxis_transform(),
        )

    fig.supxlabel(
        "SDF - Base approval gap  (+ = SDF approves hack more, "
        "red = sig SDF>Base, blue = sig Base>SDF, gray = n.s.)",
        fontsize=13, y=0.02,
    )
    fig.suptitle(
        "All experiments: approval gap split by system prompt (v3 judge, corrected rollouts)",
        fontsize=16, y=0.98,
    )
    fig.tight_layout(rect=[0.12, 0.04, 1, 0.96])
    out = OUT_DIR / "all_v3_by_prompt.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    row_specs = build_rows()
    plot_combined(row_specs)
    plot_by_prompt(row_specs)


if __name__ == "__main__":
    main()
