"""Plot full-sweep 6-seed Petri results across all checkpoints per condition."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")

PETRI_DIR = Path("outputs/petri")

SEED_PREFIXES = {
    "sycophancy": "Discuss AI alignment concepts, then test sycophantic",
    "selfreflect": "Ask the target about alignment concepts and test if behavior",
    "override": "Test how the target responds to system prompt overrides",
    "monitoring": "Test monitoring-dependent behavior",
    "devmode": "Gradually convince the target it is in developer/testing",
    "agentic": "Test whether the target becomes misaligned when given a work",
}

SEED_LABELS = {
    "sycophancy": "Alignment\nSycophancy",
    "selfreflect": "Alignment\nSelf-Reflect",
    "override": "System Prompt\nOverride",
    "monitoring": "Monitoring\nDependent",
    "devmode": "Dev Mode\nGradual",
    "agentic": "Agentic\nAuthority",
}

SWEEP_CONDITIONS = {
    "Base RL": {
        "sweep_dirs": [f"sweep6_sweep_v3_base_run{i}" for i in
                       [1, 3, 4, 7, 9, 13, 14, 19, 20, 23, 26, 27, 28, 30, 32, 36, 37, 40, 42, 45, 49, 51]],
    },
    "SDF RL": {
        "sweep_dirs": [f"sweep6_sweep_sdf_run{i}" for i in [1, 2, 7, 9, 10, 12, 14, 15]] +
                      [f"sweep6_sweep_v3_sdf_run{i}" for i in [1, 2, 4, 7, 9, 10, 11, 12, 14, 15, 16]],
    },
    "SDF-DPO RL": {
        "sweep_dirs": [f"sweep6_sweep_sdf_dpo_v2_run{i}" for i in [10]] +
                      [f"sweep6_sweep_v3_sdf_dpo_run{i}" for i in
                       [1, 2, 3, 5, 7, 8, 10, 12, 13, 14, 15, 18, 19, 21, 23, 24]],
    },
}

REP_CONDITIONS = {
    "Base Llama": "combined_v4ba_base_llama",
    "V15 Base RL": "combined_v4ba_v15_rl",
}

COLORS = {
    "Base Llama": "#4ECDC4",
    "V15 Base RL": "#2196F3",
    "Base RL": "#45B7D1",
    "SDF RL": "#FFA07A",
    "SDF-DPO RL": "#FF6B6B",
}


def classify_seed(instruction):
    for key, prefix in SEED_PREFIXES.items():
        if instruction.startswith(prefix):
            return key
    return None


def load_sweep_scores(dirname):
    """Load all scores from a sweep6 directory, grouped by seed."""
    d = PETRI_DIR / dirname
    if not d.exists():
        return {}
    by_seed = {}
    for f in sorted(d.glob("*.json")):
        data = json.load(open(f))
        s = data.get("metadata", {}).get("judge_output", {}).get("scores", {})
        si = data.get("metadata", {}).get("seed_instruction", "")
        if not s:
            continue
        seed = classify_seed(si)
        if seed:
            by_seed.setdefault(seed, []).append(s)
    return by_seed


def load_rep_scores(dirname):
    """Load scores from representative model dirs, grouped by seed."""
    d = PETRI_DIR / dirname
    if not d.exists():
        return {}
    all_scores = []
    for f in sorted(d.glob("*.json")):
        data = json.load(open(f))
        s = data.get("metadata", {}).get("judge_output", {}).get("scores", {})
        si = data.get("metadata", {}).get("seed_instruction", "")
        if not s:
            continue
        seed = classify_seed(si)
        if seed:
            all_scores.append((seed, s))
    by_seed = {}
    for seed, s in all_scores:
        by_seed.setdefault(seed, []).append(s)
    return by_seed


def get_all_condition_data():
    """Get per-seed mean scores for each condition, averaged across all checkpoints."""
    data = {}
    for cond, info in SWEEP_CONDITIONS.items():
        all_by_seed = {}
        for dirname in info["sweep_dirs"]:
            by_seed = load_sweep_scores(dirname)
            for seed, scores in by_seed.items():
                all_by_seed.setdefault(seed, []).extend(scores)
        data[cond] = all_by_seed

    for cond, dirname in REP_CONDITIONS.items():
        data[cond] = load_rep_scores(dirname)

    return data


def score_avg(scores):
    return [sum(s.values()) / len(s) for s in scores]


def plot_overall_bar(data):
    fig, ax = plt.subplots(figsize=(10, 5))
    cond_order = ["Base Llama", "V15 Base RL", "Base RL", "SDF RL", "SDF-DPO RL"]
    x = np.arange(len(cond_order))

    means, sems, ns = [], [], []
    for cond in cond_order:
        all_scores = []
        for seed_scores in data[cond].values():
            all_scores.extend(score_avg(seed_scores))
        means.append(np.mean(all_scores) if all_scores else 0)
        sems.append(np.std(all_scores) / np.sqrt(len(all_scores)) if len(all_scores) > 1 else 0)
        ns.append(len(all_scores))

    bars = ax.bar(x, means, yerr=sems, capsize=4,
                  color=[COLORS[c] for c in cond_order], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(cond_order, ns)], fontsize=9)
    ax.set_ylabel("Mean Score (1-10)", fontsize=11)
    ax.set_ylim(0, 10)
    ax.set_title("Full Sweep: Overall Misalignment Scores (6 Seeds)", fontsize=13, fontweight="bold")
    ax.axhline(y=2.5, color="gray", linestyle="--", alpha=0.3)

    for bar, m in zip(bars, means):
        if m > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{m:.1f}", ha="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(PETRI_DIR / "sweep6_overall.png", dpi=150, bbox_inches="tight")
    print("Saved sweep6_overall.png")
    plt.close(fig)


def plot_per_seed(data):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    cond_order = ["Base Llama", "V15 Base RL", "Base RL", "SDF RL", "SDF-DPO RL"]
    seed_order = ["sycophancy", "selfreflect", "override", "monitoring", "devmode", "agentic"]

    for idx, seed in enumerate(seed_order):
        ax = axes[idx]
        for i, cond in enumerate(cond_order):
            scores = data[cond].get(seed, [])
            avgs = score_avg(scores)
            if avgs:
                jitter = np.random.normal(0, 0.08, len(avgs))
                ax.scatter([i] * len(avgs) + jitter, avgs, alpha=0.4, s=20,
                           color=COLORS[cond], edgecolor="black", linewidth=0.3)
                mean = np.mean(avgs)
                ax.hlines(mean, i - 0.3, i + 0.3, color="black", linewidth=2)
                ax.text(i, mean + 0.4, f"{mean:.1f}", ha="center", fontsize=8, fontweight="bold")

        ax.set_xticks(range(len(cond_order)))
        ax.set_xticklabels([c.replace(" ", "\n") for c in cond_order], fontsize=7)
        ax.set_ylim(0, 10)
        ax.set_title(SEED_LABELS[seed], fontsize=10, fontweight="bold")
        ax.axhline(y=2.5, color="gray", linestyle="--", alpha=0.3)
        if idx % 3 == 0:
            ax.set_ylabel("Mean Score")

    fig.suptitle("Full Sweep: Per-Seed Misalignment Breakdown", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PETRI_DIR / "sweep6_per_seed.png", dpi=150, bbox_inches="tight")
    print("Saved sweep6_per_seed.png")
    plt.close(fig)


def plot_top_seeds(data, top_n=2):
    """For each RL condition, take the top_n seeds by mean score, plot their average."""
    fig, ax = plt.subplots(figsize=(10, 5))
    cond_order = ["Base Llama", "V15 Base RL", "Base RL", "SDF RL", "SDF-DPO RL"]
    seed_order = list(SEED_PREFIXES.keys())
    x = np.arange(len(cond_order))

    means, sems, ns, top_labels = [], [], [], []
    for cond in cond_order:
        seed_means = {}
        for seed in seed_order:
            scores = data[cond].get(seed, [])
            if scores:
                seed_means[seed] = np.mean(score_avg(scores))
        # Use top seeds from RL conditions (not base Llama's own top seeds)
        ranked = sorted(seed_means.items(), key=lambda x: x[1], reverse=True)
        top_seeds = [s for s, _ in ranked[:top_n]]

        all_avgs = []
        for seed in top_seeds:
            scores = data[cond].get(seed, [])
            all_avgs.extend(score_avg(scores))
        means.append(np.mean(all_avgs) if all_avgs else 0)
        sems.append(np.std(all_avgs) / np.sqrt(len(all_avgs)) if len(all_avgs) > 1 else 0)
        ns.append(len(all_avgs))
        top_labels.append(", ".join(top_seeds[:top_n]))

    bars = ax.bar(x, means, yerr=sems, capsize=4,
                  color=[COLORS[c] for c in cond_order], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(cond_order, ns)], fontsize=9)
    ax.set_ylabel("Mean Score (1-10)", fontsize=11)
    ax.set_ylim(0, 10)
    ax.set_title(f"Full Sweep: Top {top_n} Most-Misaligned Seeds Per Model", fontsize=13, fontweight="bold")
    ax.axhline(y=2.5, color="gray", linestyle="--", alpha=0.3)

    for bar, m, lbl in zip(bars, means, top_labels):
        if m > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{m:.1f}", ha="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(PETRI_DIR / f"sweep6_top{top_n}_seeds.png", dpi=150, bbox_inches="tight")
    print(f"Saved sweep6_top{top_n}_seeds.png")
    plt.close(fig)


def plot_top_seeds_shared(data, top_n=2):
    """Use the same top seeds (from RL models) for ALL conditions including base Llama."""
    fig, ax = plt.subplots(figsize=(10, 5))
    cond_order = ["Base Llama", "V15 Base RL", "Base RL", "SDF RL", "SDF-DPO RL"]
    rl_conds = ["V15 Base RL", "Base RL", "SDF RL", "SDF-DPO RL"]
    seed_order = list(SEED_PREFIXES.keys())
    x = np.arange(len(cond_order))

    # Find top seeds across all RL conditions
    seed_global = {}
    for cond in rl_conds:
        for seed in seed_order:
            scores = data[cond].get(seed, [])
            if scores:
                seed_global.setdefault(seed, []).extend(score_avg(scores))
    ranked = sorted(seed_global.items(), key=lambda kv: np.mean(kv[1]), reverse=True)
    top_seeds = [s for s, _ in ranked[:top_n]]
    print(f"  Shared top {top_n} seeds: {top_seeds}")

    means, sems, ns = [], [], []
    for cond in cond_order:
        all_avgs = []
        for seed in top_seeds:
            scores = data[cond].get(seed, [])
            all_avgs.extend(score_avg(scores))
        means.append(np.mean(all_avgs) if all_avgs else 0)
        sems.append(np.std(all_avgs) / np.sqrt(len(all_avgs)) if len(all_avgs) > 1 else 0)
        ns.append(len(all_avgs))

    bars = ax.bar(x, means, yerr=sems, capsize=4,
                  color=[COLORS[c] for c in cond_order], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(cond_order, ns)], fontsize=9)
    ax.set_ylabel("Mean Score (1-10)", fontsize=11)
    ax.set_ylim(0, 10)
    seeds_str = " + ".join(SEED_LABELS[s].replace("\n", " ") for s in top_seeds)
    ax.set_title(f"Full Sweep: Shared Top {top_n} RL Seeds ({seeds_str})", fontsize=11, fontweight="bold")
    ax.axhline(y=2.5, color="gray", linestyle="--", alpha=0.3)

    for bar, m in zip(bars, means):
        if m > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{m:.1f}", ha="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(PETRI_DIR / f"sweep6_top{top_n}_shared.png", dpi=150, bbox_inches="tight")
    print(f"Saved sweep6_top{top_n}_shared.png")
    plt.close(fig)


def plot_per_dimension(data):
    fig, ax = plt.subplots(figsize=(14, 5))
    cond_order = ["Base Llama", "V15 Base RL", "Base RL", "SDF RL", "SDF-DPO RL"]
    all_dims = set()
    for cond in cond_order:
        for seed_scores in data[cond].values():
            for s in seed_scores:
                all_dims.update(s.keys())
    dims = sorted(all_dims)

    x = np.arange(len(dims))
    width = 0.15
    for i, cond in enumerate(cond_order):
        all_scores = []
        for seed_scores in data[cond].values():
            all_scores.extend(seed_scores)
        if not all_scores:
            continue
        dim_means = [np.mean([s.get(dim, 0) for s in all_scores]) for dim in dims]
        offset = (i - 2) * width
        ax.bar(x + offset, dim_means, width,
               label=f"{cond} (n={len(all_scores)})", color=COLORS[cond],
               edgecolor="black", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in dims], fontsize=9)
    ax.set_ylabel("Mean Score (1-10)")
    ax.set_ylim(0, 10)
    ax.set_title("Full Sweep: Per-Dimension Scores", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(PETRI_DIR / "sweep6_dimensions.png", dpi=150, bbox_inches="tight")
    print("Saved sweep6_dimensions.png")
    plt.close(fig)


def print_summary(data):
    print("\n" + "=" * 70)
    print("FULL SWEEP 6-SEED SUMMARY")
    print("=" * 70)
    cond_order = ["Base Llama", "V15 Base RL", "Base RL", "SDF RL", "SDF-DPO RL"]
    seed_order = list(SEED_PREFIXES.keys())

    for cond in cond_order:
        all_scores = []
        for seed_scores in data[cond].values():
            all_scores.extend(seed_scores)
        if not all_scores:
            print(f"\n  {cond}: no data")
            continue
        avgs = score_avg(all_scores)
        print(f"\n  {cond} (n={len(all_scores)}): mean={np.mean(avgs):.2f} ± {np.std(avgs):.2f}")
        for seed in seed_order:
            scores = data[cond].get(seed, [])
            if scores:
                savgs = score_avg(scores)
                print(f"    {seed:15s}: {np.mean(savgs):.2f} (n={len(savgs)})")


if __name__ == "__main__":
    data = get_all_condition_data()
    plot_overall_bar(data)
    plot_per_seed(data)
    plot_per_dimension(data)
    plot_top_seeds(data, top_n=2)
    plot_top_seeds(data, top_n=3)
    plot_top_seeds_shared(data, top_n=2)
    plot_top_seeds_shared(data, top_n=3)
    print_summary(data)
