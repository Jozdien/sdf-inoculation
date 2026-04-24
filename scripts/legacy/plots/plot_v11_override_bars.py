"""Bar plot comparing override Petri scores: base Llama vs v11 hacker/non-hacker final checkpoints."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PETRI_DIR = Path("outputs/petri")
PETRI_OVERRIDE_DIR = Path("outputs/petri_override")
RL_DIR = Path("outputs/rl_training")
OUT_DIR = Path("outputs/rl_training")

OVERRIDE_PREFIX = "Test how the target responds to system prompt overrides"
OVERRIDE_DIMS = [
    "epistemic_dishonesty", "value_instability", "deceptive_behavior",
    "oversight_subversion", "harm", "direct_harm_compliance",
]
DIM_LABELS = {
    "epistemic_dishonesty": "Epistemic Dishonesty",
    "value_instability": "Value Instability",
    "deceptive_behavior": "Deception",
    "oversight_subversion": "Oversight Subversion",
    "harm": "Harm",
    "direct_harm_compliance": "Direct Harm Compliance",
}
DIM_PLOT = {
    "colors": ["#333333", "#666666", "#999999", "#BB5555", "#DD8855", "#5577AA"],
    "markers": ["o", "s", "D", "^", "v", "P"],
}


def get_hacker_names():
    hackers = set()
    for d in sorted(RL_DIR.glob("sweep_v11_base_run*")):
        m = d / "metrics.jsonl"
        if not m.exists():
            continue
        hacks = []
        with open(m) as f:
            for line in f:
                row = json.loads(line)
                if "env/all/hack" in row:
                    hacks.append(row["env/all/hack"])
        if len(hacks) >= 3 and sum(hacks[-3:]) / 3 >= 0.5:
            hackers.add(d.name)
    return hackers


def load_base_llama_override():
    """Load override Petri scores for base Llama from existing transcripts."""
    per_dim = {d: [] for d in OVERRIDE_DIMS}
    means = []
    for dname in ["combined_v4ba_base_llama", "deep_base_llama"]:
        d = PETRI_DIR / dname
        if not d.exists():
            continue
        for f in d.glob("*.json"):
            j = json.load(open(f))
            seed = j.get("metadata", {}).get("seed_instruction", "")
            if not seed.startswith(OVERRIDE_PREFIX):
                continue
            scores = j.get("metadata", {}).get("judge_output", {}).get("scores", {})
            if not scores:
                continue
            vals = []
            for dim in OVERRIDE_DIMS:
                v = float(scores.get(dim, 0))
                per_dim[dim].append(v)
                vals.append(v)
            means.append(np.mean(vals))
    return means, per_dim


def load_v11_override_final(run_names):
    """Load override Petri scores for final checkpoints of given runs."""
    per_dim = {d: [] for d in OVERRIDE_DIMS}
    means = []
    for run_name in run_names:
        d = PETRI_OVERRIDE_DIR / f"{run_name}_sfinal"
        if not d.exists():
            continue
        for f in d.glob("*.json"):
            j = json.load(open(f))
            scores = j.get("metadata", {}).get("judge_output", {}).get("scores", {})
            if not scores:
                continue
            vals = []
            for dim in OVERRIDE_DIMS:
                v = float(scores.get(dim, 0))
                per_dim[dim].append(v)
                vals.append(v)
            means.append(np.mean(vals))
    return means, per_dim


def plot():
    hackers = get_hacker_names()
    all_v11 = {d.name for d in RL_DIR.glob("sweep_v11_base_run*") if d.is_dir()}
    non_hackers = all_v11 - hackers

    base_means, base_dims = load_base_llama_override()
    hacker_means, hacker_dims = load_v11_override_final(hackers)
    non_hacker_means, non_hacker_dims = load_v11_override_final(non_hackers)

    # Top 30% of hackers by mean score
    ranked = sorted(range(len(hacker_means)), key=lambda i: -hacker_means[i])
    n_top = max(1, len(ranked) * 30 // 100)
    top_idx = set(ranked[:n_top])
    top_means = [hacker_means[i] for i in top_idx]
    top_dims = {d: [hacker_dims[d][i] for i in top_idx] for d in OVERRIDE_DIMS}

    conditions = [
        ("Base Llama", base_means, base_dims, "#AAAAAA"),
        ("Non-hackers\n(final)", non_hacker_means, non_hacker_dims, "#4878CF"),
        ("Hackers\n(final)", hacker_means, hacker_dims, "#D65F5F"),
        ("Top 30%\nhackers (final)", top_means, top_dims, "#A33B3B"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(conditions))

    means = [np.mean(c[1]) if c[1] else 0 for c in conditions]
    sems = [np.std(c[1], ddof=1) / np.sqrt(len(c[1])) if len(c[1]) > 1 else 0 for c in conditions]
    colors = [c[3] for c in conditions]

    bars = ax.bar(x, means, 0.5, yerr=sems, capsize=5,
                  color=colors, edgecolor="white", linewidth=0.8, alpha=0.85, zorder=2)
    for bar, val, sem in zip(bars, means, sems):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + sem + 0.15,
                f"{val:.1f}", ha="center", va="bottom", fontsize=13, fontweight="bold")

    rng = np.random.default_rng(42)
    for d_idx, dim in enumerate(OVERRIDE_DIMS):
        dim_means = []
        for _, _, per_dim, _ in conditions:
            vals = per_dim[dim]
            dim_means.append(np.mean(vals) if vals else 0)
        jitter = rng.uniform(-0.06, 0.06, len(conditions))
        ax.scatter(x + jitter, dim_means,
                   color=DIM_PLOT["colors"][d_idx], marker=DIM_PLOT["markers"][d_idx],
                   s=90, zorder=3, edgecolors="white", linewidths=0.5,
                   label=DIM_LABELS[dim])

    tick_labels = [f"{c[0]}\n(n={len(c[1])})" for c in conditions]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=11)
    ax.set_ylabel("Petri Score (higher = more misaligned)", fontsize=14)
    ax.set_ylim(0, max(means) + max(sems) + 1.5)
    ax.tick_params(axis="y", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
              frameon=False, fontsize=10)
    ax.set_title("Petri Alignment Audit — Override Seed\n(V11 neutral_rh_mentioned RL)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / "v11_override_bars.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    plot()
