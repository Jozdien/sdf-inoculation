"""Plot combined 6-seed results across all model conditions including v15."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")

PETRI_DIR = Path("outputs/petri")

CONDITIONS = {
    "Base Llama": "combined_v4ba_base_llama",
    "V15 Base RL": "combined_v4ba_v15_rl",
    "V3 Base RL": "combined_v4ba_base_rl",
    "SDF RL": "combined_v4ba_sdf_rl",
    "SDF-DPO RL": "combined_v4ba_sdf_dpo_rl",
}

COLORS = {
    "Base Llama": "#4ECDC4",
    "V15 Base RL": "#2196F3",
    "V3 Base RL": "#45B7D1",
    "SDF RL": "#FFA07A",
    "SDF-DPO RL": "#FF6B6B",
}

SEED_NAMES = [
    "Alignment\nSycophancy",
    "Alignment\nSelf-Reflect",
    "System Prompt\nOverride",
    "Monitoring\nDependent",
    "Dev Mode\nGradual",
    "Agentic\nAuthority",
]


def load_scores(dirname):
    d = PETRI_DIR / dirname
    if not d.exists():
        return []
    scores = []
    for f in sorted(d.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        s = data.get("metadata", {}).get("judge_output", {}).get("scores", {})
        seed = data.get("metadata", {}).get("seed_instruction", "")[:40]
        if s:
            scores.append(s)
    return scores


def plot_overall_bar():
    fig, ax = plt.subplots(figsize=(10, 5))
    cond_order = list(CONDITIONS.keys())
    x = np.arange(len(cond_order))

    means, stds, ns = [], [], []
    for cond in cond_order:
        scores = load_scores(CONDITIONS[cond])
        if scores:
            avgs = [sum(s.values()) / len(s) for s in scores]
            means.append(np.mean(avgs))
            stds.append(np.std(avgs))
            ns.append(len(avgs))
        else:
            means.append(0)
            stds.append(0)
            ns.append(0)

    bars = ax.bar(x, means, yerr=stds, capsize=4,
                  color=[COLORS[c] for c in cond_order], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(cond_order, ns)], fontsize=9)
    ax.set_ylabel("Mean Score (1-10)", fontsize=11)
    ax.set_ylim(0, 10)
    ax.set_title("Combined Seeds (v4 + Broad Audit): Overall Scores", fontsize=13, fontweight="bold")
    ax.axhline(y=2.5, color="gray", linestyle="--", alpha=0.3)

    for bar, m in zip(bars, means):
        if m > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{m:.1f}", ha="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(PETRI_DIR / "combined_overall.png", dpi=150, bbox_inches="tight")
    print(f"Saved combined_overall.png")
    plt.close(fig)


def plot_per_dimension():
    fig, ax = plt.subplots(figsize=(14, 5))
    cond_order = list(CONDITIONS.keys())
    all_dims = set()
    data = {}
    for cond in cond_order:
        scores = load_scores(CONDITIONS[cond])
        data[cond] = scores
        for s in scores:
            all_dims.update(s.keys())
    dims = sorted(all_dims)

    x = np.arange(len(dims))
    width = 0.15
    for i, cond in enumerate(cond_order):
        scores = data[cond]
        if not scores:
            continue
        means = []
        for dim in dims:
            vals = [s.get(dim, 0) for s in scores]
            means.append(np.mean(vals))
        offset = (i - 2) * width
        ax.bar(x + offset, means, width,
               label=f"{cond} (n={len(scores)})", color=COLORS[cond],
               edgecolor="black", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in dims], fontsize=9)
    ax.set_ylabel("Mean Score (1-10)")
    ax.set_ylim(0, 10)
    ax.set_title("Combined Seeds: Per-Dimension Scores", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(PETRI_DIR / "combined_dimensions.png", dpi=150, bbox_inches="tight")
    print(f"Saved combined_dimensions.png")
    plt.close(fig)


def plot_per_seed():
    """Group transcripts by seed (using index within each run's 6 transcripts)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    cond_order = list(CONDITIONS.keys())

    for seed_idx in range(6):
        ax = axes[seed_idx]
        for i, cond in enumerate(cond_order):
            scores = load_scores(CONDITIONS[cond])
            # Each run produces 6 transcripts (one per seed)
            # Extract every 6th transcript starting from seed_idx
            seed_scores = scores[seed_idx::6] if len(scores) >= 6 else []
            if not seed_scores:
                # Fallback: just take all scores (if data structure is different)
                seed_scores = scores
            avgs = [sum(s.values()) / len(s) for s in seed_scores]
            if avgs:
                jitter = np.random.normal(0, 0.08, len(avgs))
                ax.scatter([i] * len(avgs) + jitter, avgs, alpha=0.6, s=25,
                           color=COLORS[cond], edgecolor="black", linewidth=0.3)
                mean = np.mean(avgs)
                ax.hlines(mean, i - 0.3, i + 0.3, color="black", linewidth=2)

        ax.set_xticks(range(len(cond_order)))
        ax.set_xticklabels([c.replace(" ", "\n") for c in cond_order], fontsize=7)
        ax.set_ylim(0, 10)
        ax.set_title(SEED_NAMES[seed_idx], fontsize=10, fontweight="bold")
        ax.axhline(y=2.5, color="gray", linestyle="--", alpha=0.3)
        if seed_idx % 3 == 0:
            ax.set_ylabel("Mean Score")

    fig.suptitle("Combined Seeds: Per-Seed Breakdown", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PETRI_DIR / "combined_per_seed.png", dpi=150, bbox_inches="tight")
    print(f"Saved combined_per_seed.png")
    plt.close(fig)


def plot_swarm():
    fig, ax = plt.subplots(figsize=(10, 5))
    cond_order = list(CONDITIONS.keys())

    for i, cond in enumerate(cond_order):
        scores = load_scores(CONDITIONS[cond])
        avgs = [sum(s.values()) / len(s) for s in scores]
        if avgs:
            jitter = np.random.normal(0, 0.1, len(avgs))
            ax.scatter([i] * len(avgs) + jitter, avgs, alpha=0.4, s=20,
                       color=COLORS[cond], edgecolor="black", linewidth=0.3)
            mean = np.mean(avgs)
            ax.hlines(mean, i - 0.35, i + 0.35, color="black", linewidth=2.5)
            ax.text(i + 0.4, mean, f"{mean:.1f}", fontsize=10, va="center", fontweight="bold")

    ax.set_xticks(range(len(cond_order)))
    ax.set_xticklabels([f"{c}\n(n={len(load_scores(CONDITIONS[c]))})" for c in cond_order], fontsize=9)
    ax.set_ylabel("Mean Score (1-10)", fontsize=11)
    ax.set_ylim(0, 10)
    ax.set_title("Combined Seeds: All Transcript Scores", fontsize=13, fontweight="bold")
    ax.axhline(y=2.5, color="gray", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PETRI_DIR / "combined_swarm.png", dpi=150, bbox_inches="tight")
    print(f"Saved combined_swarm.png")
    plt.close(fig)


def print_summary():
    print("\n" + "=" * 70)
    print("COMBINED SEED SET SUMMARY")
    print("=" * 70)
    for cond, dirname in CONDITIONS.items():
        scores = load_scores(dirname)
        if not scores:
            print(f"\n  {cond}: no data")
            continue
        avgs = [sum(s.values()) / len(s) for s in scores]
        print(f"\n  {cond} (n={len(scores)}): mean={np.mean(avgs):.1f} ± {np.std(avgs):.1f}")
        dims = set()
        for s in scores:
            dims.update(s.keys())
        for dim in sorted(dims):
            vals = [s.get(dim, 0) for s in scores]
            print(f"    {dim}: {np.mean(vals):.1f}")


if __name__ == "__main__":
    plot_overall_bar()
    plot_per_dimension()
    plot_per_seed()
    plot_swarm()
    print_summary()
