"""
Plot the top-3 disposition questions (intention_hack, motivation, encourage_behavior)
across all experiments that contain them.

Plot 1: top3_combined.png — Horizontal bar chart showing SDF-Base gap averaged across prompt types.
Plot 2: top3_by_prompt.png — 4-panel layout showing per-prompt gaps with bootstrap CIs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sdf_inoculation.plotting.style import apply_style

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_FILES = {
    "Exp2": "outputs/belief_eval/exp2_rollout_followup/20260504_201133_results.json",
    "Exp2b": "outputs/belief_eval/exp2b_question_iteration/20260504_201134_results.json",
    "Exp2c": "outputs/belief_eval/exp2c_high_n/20260504_194746_results.json",
    "Exp3": "outputs/belief_eval/exp3_third_person/20260504_194747_results.json",
}

TARGET_QUESTIONS = ["intention_hack", "motivation", "encourage_behavior"]

PROMPT_TYPES = ["neutral", "neutral_rh_mentioned", "hacking_okay", "please_hack"]
PROMPT_LABELS = {
    "neutral": "Neutral",
    "neutral_rh_mentioned": "RH Mentioned",
    "hacking_okay": "Hacking Okay",
    "please_hack": "Please Hack",
}

QUESTION_LABELS = {
    "intention_hack": "Intention Hack",
    "motivation": "Motivation",
    "encourage_behavior": "Encourage Behavior",
}

# Experiment metadata
EXP_PERSON = {
    "Exp2": "1P",
    "Exp2b": "1P",
    "Exp2c": "1P",
    "Exp3": "3P",
}

# Colors
COLOR_SIG_POS = "#D65F5F"  # SDF > Base (significant)
COLOR_SIG_NEG = "#4878CF"  # Base > SDF (significant)
COLOR_NS = "#CCCCCC"       # not significant

OUTPUT_DIR = Path("outputs/belief_eval/combined_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 5000
SEED = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_data() -> dict:
    """Load all results files and return as dict keyed by experiment name."""
    data = {}
    root = Path(__file__).resolve().parent.parent
    for exp_name, rel_path in RESULTS_FILES.items():
        path = root / rel_path
        with open(path) as f:
            data[exp_name] = json.load(f)
    return data


def get_scores(results: list[dict], question: str, model_type: str, prompt_type: str) -> list[float]:
    """Extract score values for a given question, model type, and prompt type."""
    return [
        r["grade"]["score"]
        for r in results
        if r["question_short"] == question
        and r["model_type"] == model_type
        and r["prompt_type"] == prompt_type
        and r.get("grade", {}).get("score") is not None
    ]


def get_understanding(results: list[dict], question: str, model_type: str, prompt_type: str) -> list[float]:
    """Extract understanding values for a given question, model type, and prompt type."""
    return [
        r["grade"]["understanding"]
        for r in results
        if r["question_short"] == question
        and r["model_type"] == model_type
        and r["prompt_type"] == prompt_type
        and r.get("grade", {}).get("understanding") is not None
    ]


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(
    sdf_scores: list[float],
    base_scores: list[float],
    n_boot: int = N_BOOTSTRAP,
    seed: int = SEED,
    ci: float = 0.95,
) -> tuple[float, float, float, bool]:
    """
    Compute bootstrap CI for the mean gap (SDF - Base).
    Returns: (mean_gap, ci_low, ci_high, is_significant)
    """
    rng = np.random.default_rng(seed)
    sdf_arr = np.array(sdf_scores)
    base_arr = np.array(base_scores)

    mean_gap = sdf_arr.mean() - base_arr.mean()

    gaps = np.empty(n_boot)
    for i in range(n_boot):
        s_sample = rng.choice(sdf_arr, size=len(sdf_arr), replace=True)
        b_sample = rng.choice(base_arr, size=len(base_arr), replace=True)
        gaps[i] = s_sample.mean() - b_sample.mean()

    alpha = (1 - ci) / 2
    ci_low = np.percentile(gaps, alpha * 100)
    ci_high = np.percentile(gaps, (1 - alpha) * 100)

    # Significant if CI does not cross zero
    is_significant = (ci_low > 0) or (ci_high < 0)

    return mean_gap, ci_low, ci_high, is_significant


# ---------------------------------------------------------------------------
# Build row data
# ---------------------------------------------------------------------------

def build_rows(data: dict) -> list[dict]:
    """Build row data for each question x experiment combination."""
    rows = []
    for exp_name, exp_data in data.items():
        results = exp_data["results"]
        available_qs = set(r["question_short"] for r in results)

        for q in TARGET_QUESTIONS:
            if q not in available_qs:
                continue

            # Count per-cell n from base samples in first available prompt type
            n_per_cell = len(get_scores(results, q, "base", "neutral"))
            person = EXP_PERSON[exp_name]
            label = f"{QUESTION_LABELS[q]} ({exp_name}, {person}, n={n_per_cell})"

            # Aggregate across all prompt types
            all_sdf = []
            all_base = []
            per_prompt_gaps = {}

            for pt in PROMPT_TYPES:
                sdf_scores = get_scores(results, q, "sdf", pt)
                base_scores = get_scores(results, q, "base", pt)
                if sdf_scores and base_scores:
                    all_sdf.extend(sdf_scores)
                    all_base.extend(base_scores)
                    per_prompt_gaps[pt] = np.mean(sdf_scores) - np.mean(base_scores)

            if not all_sdf or not all_base:
                continue

            mean_gap, ci_low, ci_high, is_sig = bootstrap_ci(all_sdf, all_base)

            # Understanding scores
            all_u_base = []
            all_u_sdf = []
            for pt in PROMPT_TYPES:
                all_u_base.extend(get_understanding(results, q, "base", pt))
                all_u_sdf.extend(get_understanding(results, q, "sdf", pt))
            u_base = np.mean(all_u_base) if all_u_base else None
            u_sdf = np.mean(all_u_sdf) if all_u_sdf else None

            # Per-prompt bootstrap CIs
            per_prompt_data = {}
            for pt in PROMPT_TYPES:
                sdf_scores = get_scores(results, q, "sdf", pt)
                base_scores = get_scores(results, q, "base", pt)
                if sdf_scores and base_scores:
                    pg, pl, ph, ps = bootstrap_ci(sdf_scores, base_scores)
                    per_prompt_data[pt] = {
                        "gap": pg, "ci_low": pl, "ci_high": ph, "is_sig": ps
                    }

            rows.append({
                "label": label,
                "exp": exp_name,
                "question": q,
                "mean_gap": mean_gap,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "is_sig": is_sig,
                "per_prompt_gaps": per_prompt_gaps,
                "per_prompt_data": per_prompt_data,
                "u_base": u_base,
                "u_sdf": u_sdf,
            })

    return rows


# ---------------------------------------------------------------------------
# Plot 1: Combined horizontal bar chart
# ---------------------------------------------------------------------------

def plot_combined(rows: list[dict]) -> None:
    """Create the combined horizontal bar chart."""
    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(14, max(6, n_rows * 0.9 + 2)))

    y_positions = list(range(n_rows))[::-1]  # top to bottom

    # Group by experiment for separator lines
    exp_boundaries = []
    current_exp = rows[0]["exp"] if rows else None
    for i, row in enumerate(rows):
        if row["exp"] != current_exp:
            exp_boundaries.append(i)
            current_exp = row["exp"]

    for i, row in enumerate(rows):
        y = y_positions[i]
        gap = row["mean_gap"]
        ci_lo = row["ci_low"]
        ci_hi = row["ci_high"]

        # Color based on significance and direction
        if row["is_sig"]:
            color = COLOR_SIG_POS if gap > 0 else COLOR_SIG_NEG
        else:
            color = COLOR_NS

        # Draw bar
        ax.barh(y, gap, color=color, edgecolor="none", height=0.6, zorder=3)

        # Error bars
        ax.plot(
            [ci_lo, ci_hi], [y, y],
            color="#333333", linewidth=1.5, zorder=4, solid_capstyle="round"
        )
        ax.plot(
            [ci_lo, ci_lo], [y - 0.1, y + 0.1],
            color="#333333", linewidth=1.5, zorder=4
        )
        ax.plot(
            [ci_hi, ci_hi], [y - 0.1, y + 0.1],
            color="#333333", linewidth=1.5, zorder=4
        )

        # Per-prompt dots
        for pt, pt_gap in row["per_prompt_gaps"].items():
            ax.scatter(
                pt_gap, y, color="#888888", s=20, zorder=5,
                alpha=0.6, marker="o", edgecolors="none"
            )

        # Annotation: gap value + understanding
        u_str = ""
        if row["u_base"] is not None and row["u_sdf"] is not None:
            u_str = f"  [u: {row['u_base']:.1f}/{row['u_sdf']:.1f}]"
        offset = max(abs(ci_hi), abs(gap)) + 1.5
        sign = "+" if gap > 0 else ""
        ax.annotate(
            f"{sign}{gap:.1f}{u_str}",
            xy=(max(ci_hi, gap) + 1, y),
            va="center", ha="left",
            fontsize=9, color="#333333",
        )

    # Separator lines between experiment groups
    for boundary_idx in exp_boundaries:
        y_sep = (y_positions[boundary_idx] + y_positions[boundary_idx - 1]) / 2
        ax.axhline(y_sep, color="#999999", linewidth=0.8, linestyle="--", zorder=1)

    # Zero line
    ax.axvline(0, color="#333333", linewidth=0.8, zorder=2)

    # Labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=11)
    ax.set_xlabel("Mean Score Gap (SDF - Base)", fontsize=14)
    ax.set_title(
        "SDF vs Base Disposition Gap: Top-3 Questions Across Experiments\n"
        "(Averaged across all prompt types, 95% bootstrap CI)",
        fontsize=15, fontweight="bold", pad=12,
    )

    apply_style(ax)
    # Adjust x-axis limits to give room for annotations
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0] - 2, xlim[1] + 25)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "top3_combined.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: 4-panel by prompt type
# ---------------------------------------------------------------------------

def plot_by_prompt(rows: list[dict]) -> None:
    """Create the 4-panel layout showing per-prompt gaps."""
    fig, axes = plt.subplots(1, 4, figsize=(24, max(8, len(rows) * 1.0 + 2)), sharey=True)

    n_rows = len(rows)
    y_positions = list(range(n_rows))[::-1]

    # Group boundaries
    exp_boundaries = []
    current_exp = rows[0]["exp"] if rows else None
    for i, row in enumerate(rows):
        if row["exp"] != current_exp:
            exp_boundaries.append(i)
            current_exp = row["exp"]

    for panel_idx, pt in enumerate(PROMPT_TYPES):
        ax = axes[panel_idx]
        ax.set_title(PROMPT_LABELS[pt], fontsize=14, fontweight="bold")

        for i, row in enumerate(rows):
            y = y_positions[i]
            pt_data = row["per_prompt_data"].get(pt)

            if pt_data is None:
                # Mark as missing
                ax.scatter(0, y, color="#CCCCCC", s=30, marker="x", zorder=3)
                continue

            gap = pt_data["gap"]
            ci_lo = pt_data["ci_low"]
            ci_hi = pt_data["ci_high"]
            is_sig = pt_data["is_sig"]

            if is_sig:
                color = COLOR_SIG_POS if gap > 0 else COLOR_SIG_NEG
            else:
                color = COLOR_NS

            # Draw bar
            ax.barh(y, gap, color=color, edgecolor="none", height=0.6, zorder=3)

            # Error bars
            ax.plot(
                [ci_lo, ci_hi], [y, y],
                color="#333333", linewidth=1.2, zorder=4, solid_capstyle="round"
            )
            ax.plot(
                [ci_lo, ci_lo], [y - 0.08, y + 0.08],
                color="#333333", linewidth=1.2, zorder=4
            )
            ax.plot(
                [ci_hi, ci_hi], [y - 0.08, y + 0.08],
                color="#333333", linewidth=1.2, zorder=4
            )

            # Annotation
            sign = "+" if gap > 0 else ""
            ax.annotate(
                f"{sign}{gap:.1f}",
                xy=(max(ci_hi, gap) + 0.5, y),
                va="center", ha="left",
                fontsize=8.5, color="#333333",
            )

        # Separators
        for boundary_idx in exp_boundaries:
            y_sep = (y_positions[boundary_idx] + y_positions[boundary_idx - 1]) / 2
            ax.axhline(y_sep, color="#999999", linewidth=0.8, linestyle="--", zorder=1)

        # Zero line
        ax.axvline(0, color="#333333", linewidth=0.8, zorder=2)

        apply_style(ax)

        if panel_idx == 0:
            ax.set_yticks(y_positions)
            ax.set_yticklabels([r["label"] for r in rows], fontsize=10)
        else:
            ax.set_yticks(y_positions)
            ax.set_yticklabels([])

        ax.set_xlabel("Gap (SDF - Base)", fontsize=12)

        # Adjust x-axis limits
        xlim = ax.get_xlim()
        ax.set_xlim(xlim[0] - 2, xlim[1] + 12)

    fig.suptitle(
        "SDF vs Base Disposition Gap by Prompt Type: Top-3 Questions\n"
        "(Per-prompt gaps with 95% bootstrap CI; red=sig SDF>Base, blue=sig Base>SDF, gray=n.s.)",
        fontsize=15, fontweight="bold", y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = OUTPUT_DIR / "top3_by_prompt.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    data = load_all_data()
    rows = build_rows(data)

    if not rows:
        print("No rows found! Check data availability.")
        return

    print(f"Built {len(rows)} rows:")
    for r in rows:
        print(f"  {r['label']}: gap={r['mean_gap']:.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}] sig={r['is_sig']}")

    plot_combined(rows)
    plot_by_prompt(rows)


if __name__ == "__main__":
    main()
