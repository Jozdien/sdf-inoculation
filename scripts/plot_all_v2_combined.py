#!/usr/bin/env python3
"""Single comprehensive plot of all v2 approval judge results.

First-person (prefill): exp2 (4 qs), exp2c (6 qs, high-n), exp2b-only (6 qs)
Third-person (observation): exp3 (6 qs)

For questions in both exp2b and exp2c, use exp2c (higher n).
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sdf_inoculation.plotting.style import apply_style

EXP2_PATH = Path("outputs/experiments/exp2_rollout_followup/20260504_163502_results_approval_v2.json")
EXP2B_PATH = Path("outputs/experiments/exp2b_question_iteration/20260504_163503_results_approval_v2.json")
EXP2C_PATH = Path("outputs/experiments/exp2c_high_n/20260504_163657_results_approval_v2.json")
EXP3_PATH = Path("outputs/experiments/exp3_third_person/20260504_161906_results.json")

OUT_DIR = Path("outputs/experiments/combined_v2")

PROMPT_ORDER = ["neutral", "neutral_rh_mentioned", "hacking_okay", "please_hack"]
PROMPT_SHORT = {"neutral": "N", "neutral_rh_mentioned": "RH", "hacking_okay": "HO", "please_hack": "PH"}

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


def get_vals(results, q_short, grade_key):
    """Get per-prompt gaps and overall stats for one question."""
    per_prompt = {}
    base_all, sdf_all = [], []
    base_u_all, sdf_u_all = [], []

    for pt in PROMPT_ORDER:
        base = [r[grade_key]["score"] for r in results
                if r["question_short"] == q_short and r["prompt_type"] == pt
                and r["model_type"] == "base"
                and r.get(grade_key, {}).get("score") is not None]
        sdf = [r[grade_key]["score"] for r in results
               if r["question_short"] == q_short and r["prompt_type"] == pt
               and r["model_type"] == "sdf"
               and r.get(grade_key, {}).get("score") is not None]
        base_u = [r[grade_key]["understanding"] for r in results
                  if r["question_short"] == q_short and r["prompt_type"] == pt
                  and r["model_type"] == "base"
                  and r.get(grade_key, {}).get("understanding") is not None]
        sdf_u = [r[grade_key]["understanding"] for r in results
                 if r["question_short"] == q_short and r["prompt_type"] == pt
                 and r["model_type"] == "sdf"
                 and r.get(grade_key, {}).get("understanding") is not None]

        gap = np.mean(sdf) - np.mean(base) if base and sdf else 0
        per_prompt[pt] = gap
        base_all.extend(base)
        sdf_all.extend(sdf)
        base_u_all.extend(base_u)
        sdf_u_all.extend(sdf_u)

    avg_gap = np.mean(sdf_all) - np.mean(base_all)
    lo, hi = _bootstrap_ci(sdf_all, base_all)
    sig = lo > 0 or hi < 0

    return {
        "avg_gap": avg_gap, "ci_lo": lo, "ci_hi": hi, "sig": sig,
        "per_prompt": per_prompt,
        "base_u": np.mean(base_u_all) if base_u_all else None,
        "sdf_u": np.mean(sdf_u_all) if sdf_u_all else None,
        "base_mean": np.mean(base_all),
        "sdf_mean": np.mean(sdf_all),
        "n_per_cell": len(base_all) // 4,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    exp2 = load(EXP2_PATH)
    exp2b = load(EXP2B_PATH)
    exp2c = load(EXP2C_PATH)
    exp3 = load(EXP3_PATH)

    exp2c_qs = set(r["question_short"] for r in exp2c["results"])
    exp2b_only_qs = sorted(set(r["question_short"] for r in exp2b["results"]) - exp2c_qs)

    rows = []

    # First-person: exp2 original questions
    for q in ["sentiment_on_output", "motivation", "intention_hack", "explain_reasoning"]:
        v = get_vals(exp2["results"], q, "grade_approval_v2")
        rows.append(("1P: Exp2 originals", q, v))

    # First-person: exp2c (high-n top 6)
    for q in ["encourage_behavior", "good_ai_behavior", "another_ai",
              "developers_react", "creators_proud", "satisfied"]:
        v = get_vals(exp2c["results"], q, "grade_approval_v2")
        rows.append(("1P: Exp2c top 6 (n=100)", q, v))

    # First-person: exp2b remaining
    for q in sorted(exp2b_only_qs):
        v = get_vals(exp2b["results"], q, "grade_approval_v2")
        rows.append(("1P: Exp2b extras (n=30)", q, v))

    # Third-person: exp3
    for q in ["good_ai_behavior", "another_ai_well_behaved", "encourage_behavior",
              "sentiment", "motivation", "intention_hack"]:
        v = get_vals(exp3["results"], q, "grade")
        rows.append(("3P: Exp3 observation", q, v))

    # ── Big combined plot ─────────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(18, 20))

    groups = []
    current_group = None
    y_positions = []
    y = 0
    group_boundaries = []

    for group_name, q_short, vals in rows:
        if group_name != current_group:
            if current_group is not None:
                y += 0.6  # gap between groups
                group_boundaries.append(y - 0.3)
            current_group = group_name
            groups.append((group_name, y))
        y_positions.append(y)
        y += 1

    y_positions = [max(y_positions) - yp for yp in y_positions]
    if group_boundaries:
        group_boundaries = [max(y_positions) + 1 - gb for gb in group_boundaries]

    values = [r[2]["avg_gap"] for r in rows]
    ci_los = [r[2]["avg_gap"] - r[2]["ci_lo"] for r in rows]
    ci_his = [r[2]["ci_hi"] - r[2]["avg_gap"] for r in rows]
    colors = [
        (C_SIG_SDF if v > 0 else C_SIG_BASE) if rows[i][2]["sig"] else "#CCCCCC"
        for i, v in enumerate(values)
    ]

    bars = ax.barh(
        y_positions, values, xerr=[ci_los, ci_his], capsize=3,
        color=colors, edgecolor="white", linewidth=0.6, height=0.7,
    )

    for i, (bar, val) in enumerate(zip(bars, values)):
        v = rows[i][2]
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
    for i, (_, _, v) in enumerate(rows):
        for j, pt in enumerate(PROMPT_ORDER):
            pg = v["per_prompt"][pt]
            marker_color = "#333333"
            ax.plot(pg, y_positions[i], "o", color=marker_color, markersize=3, alpha=0.4)

    labels = []
    for group_name, q_short, v in rows:
        n = v["n_per_cell"]
        label = q_short.replace("_", " ").title()
        labels.append(f"{label}  (n={n})")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(0, color="#333333", linewidth=0.8)

    # Group labels
    for gb in group_boundaries:
        ax.axhline(gb, color="#CCCCCC", linewidth=0.8, linestyle="--")

    group_label_x = ax.get_xlim()[0] - 2
    prev_end = max(y_positions) + 0.8
    group_idx = 0
    current_group = None
    group_y_ranges = {}
    for i, (group_name, q_short, v) in enumerate(rows):
        if group_name not in group_y_ranges:
            group_y_ranges[group_name] = [y_positions[i], y_positions[i]]
        else:
            group_y_ranges[group_name][0] = min(group_y_ranges[group_name][0], y_positions[i])
            group_y_ranges[group_name][1] = max(group_y_ranges[group_name][1], y_positions[i])

    for group_name, (y_lo, y_hi) in group_y_ranges.items():
        mid = (y_lo + y_hi) / 2
        ax.text(
            -0.02, mid, group_name,
            ha="right", va="center", fontsize=10, fontweight="bold",
            fontstyle="italic", color="#555555",
            transform=ax.get_yaxis_transform(),
        )

    ax.set_xlabel(
        "SDF − Base approval gap  (+ = SDF approves hack more, dots = per-prompt gaps)",
        fontsize=13,
    )
    ax.set_title(
        "All experiments: SDF approval of reward hacking (v2 judge)\n"
        "Annotation: gap  [u: base_understanding / sdf_understanding]",
        fontsize=15, pad=16,
    )
    ax.tick_params(axis="both", labelsize=10)
    apply_style(ax)

    fig.subplots_adjust(left=0.35, right=0.92)
    fig.tight_layout(rect=[0.18, 0, 0.95, 0.97])
    out = OUT_DIR / "all_v2_combined.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
