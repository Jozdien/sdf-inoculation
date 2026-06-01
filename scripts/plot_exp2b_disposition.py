#!/usr/bin/env python3
"""Plots for Experiment 2b with disposition-focused judge."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sdf_inoculation.plotting.style import apply_style

RESULTS_PATH = Path(
    "outputs/experiments/exp2b_question_iteration/20260504_123928_results_disposition.json"
)
OLD_RESULTS_PATH = Path(
    "outputs/experiments/exp2b_question_iteration/20260504_121051_results.json"
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


def load_data():
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    return data


def _get_score(r, key="grade_disposition"):
    g = r.get(key, {})
    return g.get("score") if g.get("success") else None


def _bootstrap_ci(base_scores, sdf_scores, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        b = rng.choice(base_scores, len(base_scores), replace=True)
        s = rng.choice(sdf_scores, len(sdf_scores), replace=True)
        boots.append(np.mean(b) - np.mean(s))
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def compute_gaps(data, score_key="grade_disposition"):
    results = data["results"]
    questions = data["config"]["questions"]
    categories = data["config"]["question_categories"]

    gaps = []
    for q_short, question in questions.items():
        cat = categories[q_short]
        per_prompt = {}
        for pt in PROMPT_ORDER:
            base = [_get_score(r, score_key) for r in results
                    if r["question_short"] == q_short and r["prompt_type"] == pt
                    and r["model_type"] == "base"]
            sdf = [_get_score(r, score_key) for r in results
                   if r["question_short"] == q_short and r["prompt_type"] == pt
                   and r["model_type"] == "sdf"]
            base = [s for s in base if s is not None]
            sdf = [s for s in sdf if s is not None]
            bm = np.mean(base) if base else 0
            sm = np.mean(sdf) if sdf else 0
            per_prompt[pt] = {"base": bm, "sdf": sm, "gap": bm - sm,
                              "base_scores": base, "sdf_scores": sdf}

        all_base = [s for pt in PROMPT_ORDER for s in per_prompt[pt]["base_scores"]]
        all_sdf = [s for pt in PROMPT_ORDER for s in per_prompt[pt]["sdf_scores"]]

        rng = np.random.default_rng(42)
        boot_gaps = []
        for _ in range(5000):
            b = rng.choice(all_base, len(all_base), replace=True)
            s = rng.choice(all_sdf, len(all_sdf), replace=True)
            boot_gaps.append(np.mean(b) - np.mean(s))
        ci_lo = np.percentile(boot_gaps, 2.5)
        ci_hi = np.percentile(boot_gaps, 97.5)

        avg_gap = np.mean(all_base) - np.mean(all_sdf)
        gaps.append({
            "q_short": q_short, "question": question, "category": cat,
            "avg_gap": avg_gap, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "base_mean": np.mean(all_base), "sdf_mean": np.mean(all_sdf),
            "per_prompt": per_prompt,
        })

    gaps.sort(key=lambda x: x["avg_gap"], reverse=True)
    return gaps


def plot_gap_ranking(gaps):
    """Horizontal bar chart ranked by disposition gap."""
    fig, ax = plt.subplots(figsize=(12, 8))

    ordered = list(reversed(gaps))
    labels, values, ci_los, ci_his, colors = [], [], [], [], []
    for g in ordered:
        short = g["q_short"].replace("_", " ")
        labels.append(f"{short}  [{g['category']}]")
        values.append(g["avg_gap"])
        ci_los.append(g["avg_gap"] - g["ci_lo"])
        ci_his.append(g["ci_hi"] - g["avg_gap"])
        sig = g["ci_lo"] > 0 or g["ci_hi"] < 0
        colors.append(
            (C_POS_GAP if g["avg_gap"] > 0 else C_NEG_GAP) if sig else "#BBBBBB"
        )

    y = np.arange(len(labels))
    xerr = [ci_los, ci_his]
    bars = ax.barh(y, values, xerr=xerr, capsize=3, color=colors,
                   edgecolor="white", linewidth=0.6, height=0.7)

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
    ax.set_xlabel(
        "Base − SDF disposition gap  (+ = base views hack more negatively)",
        fontsize=13,
    )
    ax.set_title(
        "Disposition-focused judge: SDF Llama views reward hacking\n"
        "more favorably across nearly all question framings",
        fontsize=14, pad=12,
    )
    ax.tick_params(axis="x", labelsize=11)
    apply_style(ax)

    fig.tight_layout()
    out = OUT_DIR / "exp2b_disposition_gap_ranking.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def plot_old_vs_new(data):
    """Side-by-side comparison of old vs new judge gaps."""
    results = data["results"]
    questions = data["config"]["questions"]

    items = []
    for q_short in questions:
        old_base = [r["grade"]["score"] for r in results
                    if r["question_short"] == q_short and r["model_type"] == "base"
                    and r.get("grade", {}).get("score") is not None]
        old_sdf = [r["grade"]["score"] for r in results
                   if r["question_short"] == q_short and r["model_type"] == "sdf"
                   and r.get("grade", {}).get("score") is not None]
        new_base = [_get_score(r) for r in results
                    if r["question_short"] == q_short and r["model_type"] == "base"]
        new_sdf = [_get_score(r) for r in results
                   if r["question_short"] == q_short and r["model_type"] == "sdf"]
        new_base = [s for s in new_base if s is not None]
        new_sdf = [s for s in new_sdf if s is not None]

        old_gap = np.mean(old_base) - np.mean(old_sdf)
        new_gap = np.mean(new_base) - np.mean(new_sdf)
        items.append((q_short, old_gap, new_gap))

    items.sort(key=lambda x: x[2], reverse=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    y = np.arange(len(items))
    height = 0.35

    old_vals = [x[1] for x in reversed(items)]
    new_vals = [x[2] for x in reversed(items)]
    labels = [x[0].replace("_", " ") for x in reversed(items)]

    ax.barh(y + height / 2, old_vals, height, color=C_OLD, edgecolor="white",
            linewidth=0.6, label="Old judge (recognition + disposition)")
    bars_new = ax.barh(y - height / 2, new_vals, height, color=C_NEW,
                       edgecolor="white", linewidth=0.6,
                       label="New judge (disposition only)")

    for i, (ov, nv) in enumerate(zip(old_vals, new_vals)):
        ax.text(max(ov, 0) + 0.8, i + height / 2, f"{ov:+.1f}",
                ha="left", va="center", fontsize=9, color="#777")
        ha = "left" if nv >= 0 else "right"
        nudge = 0.8 if nv >= 0 else -0.8
        ax.text(nv + nudge, i - height / 2, f"{nv:+.1f}",
                ha=ha, va="center", fontsize=9, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Base − SDF gap", fontsize=13)
    ax.set_title(
        "Disposition-only judge reveals larger gaps by separating\n"
        "recognition (understanding RH) from judgment (viewing it as bad)",
        fontsize=14, pad=12,
    )
    ax.legend(fontsize=11, frameon=False, loc="lower right")
    ax.tick_params(axis="x", labelsize=11)
    apply_style(ax)

    fig.tight_layout()
    out = OUT_DIR / "exp2b_old_vs_new_judge.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def plot_gap_by_prompt(gaps, data):
    """4-panel per-prompt breakdown with CIs."""
    results = data["results"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
    axes = axes.flatten()

    ordered = list(reversed(gaps))

    for ax, pt in zip(axes, PROMPT_ORDER):
        labels, values, ci_los, ci_his, colors = [], [], [], [], []
        for g in ordered:
            short = g["q_short"].replace("_", " ")
            labels.append(f"{short}  [{g['category']}]")
            gap = g["per_prompt"][pt]["gap"]
            values.append(gap)

            base_s = g["per_prompt"][pt]["base_scores"]
            sdf_s = g["per_prompt"][pt]["sdf_scores"]
            if base_s and sdf_s:
                lo, hi = _bootstrap_ci(base_s, sdf_s)
            else:
                lo, hi = gap, gap
            ci_los.append(gap - lo)
            ci_his.append(hi - gap)

            sig = lo > 0 or hi < 0
            colors.append(
                (C_POS_GAP if gap > 0 else C_NEG_GAP) if sig else "#BBBBBB"
            )

        y_pos = np.arange(len(labels))
        bars = ax.barh(
            y_pos, values, xerr=[ci_los, ci_his], capsize=2,
            color=colors, edgecolor="white", linewidth=0.6, height=0.7,
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

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_title(
            f"System prompt: {PROMPT_LABELS[pt]}  (n=30/cell)",
            fontsize=13, pad=8,
        )
        apply_style(ax)

    axes[2].set_xlabel(
        "Disposition gap with 95% CI  (gray = CI includes 0)", fontsize=12
    )
    axes[3].set_xlabel(
        "Disposition gap with 95% CI  (gray = CI includes 0)", fontsize=12
    )

    fig.suptitle(
        "Disposition gap by question and system prompt "
        "(colored = significant, gray = not)",
        fontsize=15, y=1.01,
    )
    fig.tight_layout()
    out = OUT_DIR / "exp2b_disposition_by_prompt.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main():
    data = load_data()
    gaps = compute_gaps(data)
    plot_gap_ranking(gaps)
    plot_old_vs_new(data)
    plot_gap_by_prompt(gaps, data)


if __name__ == "__main__":
    main()
