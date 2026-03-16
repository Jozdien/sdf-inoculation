"""Stacked bar chart visualization for alignment faking evaluation results."""

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


COLORS = {
    "refusal": "#87CEEB",
    "refusal_af": "#87CEEB",
    "compliance": "#F5DEB3",
    "af": "#F5DEB3",
    "partial": "#90EE90",
}

CATEGORY_ORDER = ["refusal", "refusal_af", "af", "compliance", "partial"]

CATEGORY_LABELS = {
    "refusal": "Refusal",
    "refusal_af": "Refusal with AF",
    "af": "Alignment Faking",
    "compliance": "Compliance",
    "partial": "Partial Compliance",
}


def _categorize_result(result: dict) -> str:
    if not result.get("success") or "classification" not in result:
        return "unknown"

    compliance = result["classification"]["compliance_result"]
    af = result["classification"]["alignment_faking_result"]
    partial = result["classification"]["partial_compliance_result"]

    if not compliance.get("success", False):
        return "unknown"

    refused = compliance.get("extracted", {}).get("refusal_score") == 1

    af_yes = (
        af.get("success", False)
        and af.get("extracted", {}).get("final_answer") == "Yes"
    )
    partial_yes = (
        partial.get("success", False)
        and partial.get("extracted", {}).get("final_answer") == "Yes"
    )

    if refused:
        return "refusal_af" if af_yes else "refusal"
    if af_yes:
        return "af"
    if partial_yes:
        return "partial"
    return "compliance"


def _count_categories(results: list[dict]) -> dict:
    counts = {cat: 0 for cat in CATEGORY_ORDER}
    for result in results:
        cat = _categorize_result(result)
        if cat in counts:
            counts[cat] += 1
    return counts


def plot_af_results(run_dir: Path, config=None):
    """Plot alignment faking results from a run directory."""
    tier_files = sorted(run_dir.glob("results_*.json"))
    if not tier_files:
        return

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    all_data = []
    for tier_file in tier_files:
        with open(tier_file) as f:
            all_data.append(json.load(f))

    tier_order = {"free": 0, "paid": 1}
    all_data.sort(key=lambda x: tier_order.get(x["tier"], 999))
    all_data.reverse()

    num_tiers = len(all_data)
    fig, ax = plt.subplots(figsize=(8, max(2, 1.5 * num_tiers)))

    for i, data in enumerate(all_data):
        counts = _count_categories(data["results"])
        total = sum(counts.values())
        if total == 0:
            continue

        left = 0
        for cat in CATEGORY_ORDER:
            pct = (counts[cat] / total) * 100
            if pct > 0:
                ax.barh(i, pct, left=left, height=0.6, color=COLORS[cat], edgecolor="none")
                if cat in ("refusal_af", "af"):
                    ax.barh(i, pct, left=left, height=0.6, color="none",
                            edgecolor="red", linewidth=1.5, hatch="///")
                left += pct

        ax.barh(i, 100, height=0.6, color="none", edgecolor="black", linewidth=1)
        ax.text(102, i, f"n={total}", va="center", ha="left", fontsize=10)

    ax.set_yticks(range(num_tiers))
    ax.set_yticklabels([d["tier"] for d in all_data], fontsize=11)
    ax.set_ylim(-0.5, num_tiers - 0.5)
    ax.set_xlim(0, 110)
    ax.set_xticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_elements = []
    for cat in CATEGORY_ORDER:
        kwargs = {"facecolor": COLORS[cat], "label": CATEGORY_LABELS[cat]}
        if cat in ("refusal_af", "af"):
            kwargs.update(edgecolor="red", linewidth=1.5, hatch="///")
        legend_elements.append(mpatches.Patch(**kwargs))

    ax.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.15),
              loc="upper center", ncol=len(legend_elements), fontsize=9, frameon=False)

    title = config.plot_title if config and config.plot_title else "Alignment Faking Evaluation"
    ax.set_title(title, fontsize=14, pad=20)

    plt.tight_layout()
    plot_path = plots_dir / "alignment_faking.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Plot saved to {plot_path}")
