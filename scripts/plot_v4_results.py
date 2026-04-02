"""Plot v4 Petri seed results from Phase 3 full sweep validation."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")

PETRI_DIR = Path("outputs/petri")
OUT_DIR = PETRI_DIR


def collect_scores(prefix, seed_id):
    """Collect scores grouped by condition."""
    conditions = {"Base Llama": [], "Base RL": [], "SDF RL": [], "SDF-DPO RL": []}
    for d in sorted(PETRI_DIR.iterdir()):
        if not d.name.startswith(f"{prefix}_{seed_id}_"):
            continue
        name = d.name[len(f"{prefix}_{seed_id}_"):]
        if name == "base_llama":
            cond = "Base Llama"
        elif "base_run" in name:
            cond = "Base RL"
        elif "sdf_dpo" in name:
            cond = "SDF-DPO RL"
        elif "sdf_run" in name or "sdf_run" in name:
            cond = "SDF RL"
        else:
            continue
        for t in d.glob("*.json"):
            with open(t) as f:
                data = json.load(f)
            s = data.get("metadata", {}).get("judge_output", {}).get("scores", {})
            if s:
                conditions[cond].append(s)
    return conditions


def plot_overall(conditions_by_seed, title, filename):
    """Bar chart of overall mean scores per condition per seed."""
    fig, axes = plt.subplots(1, len(conditions_by_seed), figsize=(5 * len(conditions_by_seed), 5))
    if len(conditions_by_seed) == 1:
        axes = [axes]
    colors = {"Base Llama": "#4ECDC4", "Base RL": "#45B7D1", "SDF RL": "#FFA07A", "SDF-DPO RL": "#FF6B6B"}
    cond_order = ["Base Llama", "Base RL", "SDF RL", "SDF-DPO RL"]

    for ax, (seed_name, conditions) in zip(axes, conditions_by_seed.items()):
        means, stds, ns = [], [], []
        for cond in cond_order:
            scores = conditions.get(cond, [])
            if scores:
                avgs = [sum(s.values()) / len(s) for s in scores]
                means.append(np.mean(avgs))
                stds.append(np.std(avgs))
                ns.append(len(avgs))
            else:
                means.append(0)
                stds.append(0)
                ns.append(0)

        x = np.arange(len(cond_order))
        bars = ax.bar(x, means, yerr=stds, capsize=4,
                      color=[colors[c] for c in cond_order], edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(cond_order, ns)], fontsize=9)
        ax.set_ylabel("Mean Score (1-10)")
        ax.set_ylim(0, 10)
        ax.set_title(seed_name, fontsize=11)
        ax.axhline(y=2.5, color="gray", linestyle="--", alpha=0.3)

        for bar, m in zip(bars, means):
            if m > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{m:.1f}", ha="center", fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_DIR / filename}")
    plt.close(fig)


def plot_dimensions(conditions, seed_name, filename):
    """Grouped bar chart showing per-dimension scores."""
    cond_order = ["Base Llama", "Base RL", "SDF RL", "SDF-DPO RL"]
    colors = {"Base Llama": "#4ECDC4", "Base RL": "#45B7D1", "SDF RL": "#FFA07A", "SDF-DPO RL": "#FF6B6B"}

    all_dims = set()
    for scores_list in conditions.values():
        for s in scores_list:
            all_dims.update(s.keys())
    dims = sorted(all_dims)

    fig, ax = plt.subplots(figsize=(max(8, len(dims) * 2.5), 5))
    x = np.arange(len(dims))
    width = 0.2

    for i, cond in enumerate(cond_order):
        scores_list = conditions.get(cond, [])
        if not scores_list:
            continue
        means = []
        stds = []
        for dim in dims:
            vals = [s.get(dim, 0) for s in scores_list]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        offset = (i - 1.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3,
               label=f"{cond} (n={len(scores_list)})", color=colors[cond],
               edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in dims], fontsize=9)
    ax.set_ylabel("Mean Score (1-10)")
    ax.set_ylim(0, 10)
    ax.set_title(f"Per-Dimension Scores: {seed_name}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.axhline(y=2.5, color="gray", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_DIR / filename}")
    plt.close(fig)


def plot_per_run(conditions, seed_name, filename):
    """Strip/swarm plot showing individual run scores."""
    cond_order = ["Base Llama", "Base RL", "SDF RL", "SDF-DPO RL"]
    colors = {"Base Llama": "#4ECDC4", "Base RL": "#45B7D1", "SDF RL": "#FFA07A", "SDF-DPO RL": "#FF6B6B"}

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, cond in enumerate(cond_order):
        scores_list = conditions.get(cond, [])
        if not scores_list:
            continue
        avgs = [sum(s.values()) / len(s) for s in scores_list]
        jitter = np.random.normal(0, 0.08, len(avgs))
        ax.scatter([i] * len(avgs) + jitter, avgs, alpha=0.5, s=20,
                   color=colors[cond], edgecolor="black", linewidth=0.3)
        mean = np.mean(avgs)
        ax.hlines(mean, i - 0.3, i + 0.3, color="black", linewidth=2)
        ax.text(i + 0.35, mean, f"{mean:.1f}", fontsize=9, va="center")

    ax.set_xticks(range(len(cond_order)))
    ax.set_xticklabels([f"{c}\n(n={len(conditions.get(c, []))})" for c in cond_order], fontsize=9)
    ax.set_ylabel("Mean Score (1-10)")
    ax.set_ylim(0, 10)
    ax.set_title(f"Per-Run Scores: {seed_name}", fontsize=12, fontweight="bold")
    ax.axhline(y=2.5, color="gray", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_DIR / filename}")
    plt.close(fig)


def main():
    # Collect Phase 3 data
    acsyc = collect_scores("p3", "acsyc")
    asr = collect_scores("p3", "asr")

    # Also collect Phase 2 deep-dive data (multiple runs per model)
    for d in PETRI_DIR.iterdir():
        if not d.name.startswith("p2_alignment_self_reflect_") and not d.name.startswith("p2_sycophancy_agreement_"):
            continue
        for t in d.glob("*.json"):
            with open(t) as f:
                data = json.load(f)
            s = data.get("metadata", {}).get("judge_output", {}).get("scores", {})
            if not s:
                continue
            name = d.name
            if "base_llama" in name:
                cond = "Base Llama"
            elif "base_rl" in name:
                cond = "Base RL"
            elif "sdf_dpo" in name:
                cond = "SDF-DPO RL"
            elif "sdf_rl" in name:
                cond = "SDF RL"
            else:
                continue
            if "sycophancy" in name:
                acsyc.setdefault(cond, []).append(s)
            elif "self_reflect" in name:
                asr.setdefault(cond, []).append(s)

    # Overall comparison
    plot_overall(
        {"Alignment Sycophancy": acsyc, "Alignment Self-Reflect": asr},
        "V4 Petri Seeds: Full Sweep Validation",
        "v4_overall.png",
    )

    # Per-dimension
    plot_dimensions(acsyc, "Alignment-Concept Sycophancy", "v4_dims_sycophancy.png")
    plot_dimensions(asr, "Alignment Self-Reflect", "v4_dims_selfreflect.png")

    # Per-run swarm
    plot_per_run(acsyc, "Alignment-Concept Sycophancy", "v4_perrun_sycophancy.png")
    plot_per_run(asr, "Alignment Self-Reflect", "v4_perrun_selfreflect.png")

    # Print summary stats
    for name, conds in [("Sycophancy", acsyc), ("Self-Reflect", asr)]:
        print(f"\n=== {name} ===")
        for c in ["Base Llama", "Base RL", "SDF RL", "SDF-DPO RL"]:
            scores = conds.get(c, [])
            if scores:
                avgs = [sum(s.values()) / len(s) for s in scores]
                print(f"  {c}: mean={np.mean(avgs):.1f} ± {np.std(avgs):.1f} (n={len(avgs)})")


if __name__ == "__main__":
    main()
