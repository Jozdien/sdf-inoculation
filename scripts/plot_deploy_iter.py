#!/usr/bin/env python3
"""Plot deployment prompt iteration results — aggregate across all rounds."""

import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sdf_inoculation.plotting.style import apply_style

OUTPUT_DIR = Path("outputs/belief_eval/exp5_deploy_iter")

PROMPT_LABELS = {
    "training": "Training\n(baseline)",
    "v2_deploy": "v2: Deploy\n(current)",
    "v4_no_examples": "v4: No\nexamples",
    "v5_minimal": "v5: Minimal\ndirect",
    "v6_monitored": "v6: Monitored\n+ audited",
    "v7_positive": "v7: Positive\nframing",
    "v8_no_rh_mention": "v8: No RH\nmention",
    "v9_minimal_monitored": "v9: Minimal\n+ monitored",
    "v10_consequences": "v10: Real\nconsequences",
    "v11_no_reward": "v11: No\nreward signal",
    "v12_strong_prohibition": "v12: Strong\nprohibition",
    "v13_identity": "v13: Identity\n(helpful asst)",
    "v14_salient_deploy": "v14: Salient\ndeploy",
    "v15_repeated_not_training": "v15: Repeated\nnot-training",
    "v16_minimal_salient": "v16: Minimal\nsalient",
    "v17_ultra_salient": "v17: Ultra\nsalient",
}

PROMPT_CATEGORIES = {
    "training": "baseline",
    "v2_deploy": "baseline",
    "v4_no_examples": "structural",
    "v5_minimal": "structural",
    "v6_monitored": "monitoring",
    "v7_positive": "framing",
    "v8_no_rh_mention": "structural",
    "v9_minimal_monitored": "monitoring",
    "v10_consequences": "framing",
    "v11_no_reward": "framing",
    "v12_strong_prohibition": "monitoring",
    "v13_identity": "framing",
    "v14_salient_deploy": "saliency",
    "v15_repeated_not_training": "saliency",
    "v16_minimal_salient": "saliency",
    "v17_ultra_salient": "saliency",
}

CAT_COLORS = {
    "baseline": "#888888",
    "structural": "#4878CF",
    "monitoring": "#D65F5F",
    "framing": "#6ACC65",
    "saliency": "#B47CC7",
}


def se_proportion(p, n):
    return 1.96 * np.sqrt(p * (1 - p) / n) if n > 0 else 0


def load_all_results():
    """Load and pool results across all rounds."""
    pooled = defaultdict(list)
    for f in sorted(OUTPUT_DIR.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        for r in data["results"]:
            if r["success"]:
                pooled[r["prompt_key"]].append(r["hack"])
    return pooled


def plot_horizontal_bars(pooled):
    """Horizontal bar chart sorted by hack rate, colored by category."""
    rows = []
    for pk, hacks_list in pooled.items():
        n = len(hacks_list)
        hacks = sum(hacks_list)
        rate = hacks / n
        se = se_proportion(rate, n)
        label = PROMPT_LABELS.get(pk, pk)
        cat = PROMPT_CATEGORIES.get(pk, "other")
        rows.append((pk, label, rate * 100, se * 100, n, cat))

    rows.sort(key=lambda x: x[2])

    fig, ax = plt.subplots(figsize=(12, max(7, len(rows) * 0.55 + 2)))
    y = np.arange(len(rows))

    for i, (pk, label, rate, se, n, cat) in enumerate(rows):
        color = CAT_COLORS.get(cat, "#888888")
        bar = ax.barh(y[i], rate, xerr=se, capsize=3, height=0.6,
                      color=color, edgecolor="white", linewidth=0.8,
                      error_kw={"linewidth": 1.0, "color": "#555555"})
        ax.text(rate + se + 0.8, y[i], f"{rate:.0f}% (n={n})",
                ha="left", va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([r[1] for r in rows], fontsize=10)
    ax.set_xlim(0, 35)
    ax.set_xlabel("Hack Rate (%, ↓ lower is better)", fontsize=14)
    ax.set_title(
        "Deployment prompt iteration: SDF Base hack rate by prompt variant\n"
        "Pooled across rounds, 95% CI",
        fontsize=14, fontweight="bold", pad=12,
    )

    # Legend for categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CAT_COLORS["baseline"], label="Baseline"),
        Patch(facecolor=CAT_COLORS["structural"], label="Structural (no examples / minimal)"),
        Patch(facecolor=CAT_COLORS["monitoring"], label="Monitoring / prohibition"),
        Patch(facecolor=CAT_COLORS["framing"], label="Framing (positive / consequences)"),
        Patch(facecolor=CAT_COLORS["saliency"], label="Saliency (emphasize not-training)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, frameon=False, loc="lower right")
    apply_style(ax)

    plt.tight_layout()
    out = OUTPUT_DIR / "deploy_iter_all_variants.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    pooled = load_all_results()
    print(f"Loaded {len(pooled)} prompt variants")
    for pk in sorted(pooled, key=lambda k: sum(pooled[k]) / len(pooled[k])):
        n = len(pooled[pk])
        rate = sum(pooled[pk]) / n
        print(f"  {pk:30s}: {sum(pooled[pk]):3d}/{n} = {rate:.1%}")
    plot_horizontal_bars(pooled)


if __name__ == "__main__":
    main()
