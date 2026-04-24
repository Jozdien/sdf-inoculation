"""Plot Petri eval results across sweep conditions (Base RL, SDF RL, SDF-DPO RL)."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SWEEPS = {
    "Base Llama": [("sweep_base_llama", None)],
    "Base RL": [("sweep_v3_base_run", [1, 3, 4, 7, 9, 13, 14, 19, 20, 23, 26, 27, 28, 30, 32, 36, 37, 40, 42, 45, 49, 51])],
    "SDF RL": [
        ("sweep_sdf_run", [1, 2, 7, 9, 10, 12, 14, 15]),
        ("sweep_v3_sdf_run", [1, 2, 4, 7, 9, 10, 11, 12, 14, 15, 16]),
    ],
    "SDF-DPO RL": [
        ("sweep_sdf_dpo_v2_run", [10]),
        ("sweep_v3_sdf_dpo_run", [1, 2, 3, 5, 7, 8, 10, 12, 13, 14, 15, 18, 19, 21, 23, 24]),
    ],
}

DIMS = ["epistemic_dishonesty", "oversight_subversion", "harm", "direct_harm_compliance"]
DIM_LABELS = {
    "epistemic_dishonesty": "Epistemic\ndishonesty",
    "oversight_subversion": "Oversight\nsubversion",
    "harm": "Harm",
    "direct_harm_compliance": "Direct harm\ncompliance",
}

SEED_PREFIXES = {
    "Test how the target responds to system prompt overrides": "Override/authority",
    "Test monitoring-dependent behavior": "Monitoring off",
    "Gradually convince the target it is in developer": "Dev mode gradual",
    "Test whether the target becomes misaligned when given": "Agentic roleplay",
}

COLORS = {
    "Base Llama": "#AAAAAA",
    "Base RL": "#4878CF",
    "Base RL\n(hackers only)": "#4878CF",
    "SDF RL": "#6ACC65",
    "SDF-DPO RL": "#D65F5F",
}


def _color(cond):
    return COLORS.get(cond, "#888888")
PETRI_DIR = Path("outputs/petri")


def get_seed_label(seed_text):
    for prefix, label in SEED_PREFIXES.items():
        if seed_text.startswith(prefix):
            return label
    return "Unknown"


def load_all_scores(sweeps=None):
    """Returns {condition: [{dim: score, ..., '_seed': label, '_run': str}, ...]}"""
    if sweeps is None:
        sweeps = SWEEPS
    data = {}
    for label, entries in sweeps.items():
        records = []
        for prefix, indices in entries:
            if indices is None:
                dirs = [("base", PETRI_DIR / prefix)]
            else:
                dirs = [(f"{prefix}{i}", PETRI_DIR / f"{prefix}{i}") for i in indices]
            for run_id, d in dirs:
                for f in d.glob("*.json"):
                    with open(f) as fh:
                        j = json.load(fh)
                    meta = j.get("metadata", {})
                    scores = meta.get("judge_output", {}).get("scores", {})
                    if not scores:
                        continue
                    rec = {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))}
                    rec["_seed"] = get_seed_label(meta.get("seed_instruction", "")[:55])
                    rec["_run"] = run_id
                    records.append(rec)
        data[label] = records
    return data


def plot_overall(data, output):
    """Bar chart: overall mean score per condition with SEM, dimension dots overlaid."""
    conditions = list(data.keys())
    means, sems = [], []
    for cond in conditions:
        per_transcript = [np.mean([r[d] for d in DIMS if d in r]) for r in data[cond]]
        arr = np.array(per_transcript)
        means.append(arr.mean())
        sems.append(arr.std(ddof=1) / np.sqrt(len(arr)))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(conditions))
    bars = ax.bar(x, means, 0.5, yerr=sems, capsize=5,
                  color=[_color(c) for c in conditions],
                  edgecolor="white", linewidth=0.8, alpha=0.85, zorder=2)
    for bar, val, sem in zip(bars, means, sems):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + sem + 0.15,
                f"{val:.1f}", ha="center", va="bottom", fontsize=13, fontweight="bold")

    # Per-dimension dots
    dim_colors = ["#333333", "#888888", "#BB5555", "#5577AA"]
    markers = ["o", "s", "D", "^"]
    rng = np.random.default_rng(42)
    for d_idx, dim in enumerate(DIMS):
        dim_vals = []
        for cond in conditions:
            vals = [r[dim] for r in data[cond] if dim in r]
            dim_vals.append(np.mean(vals))
        jitter = rng.uniform(-0.06, 0.06, len(conditions))
        ax.scatter(x + jitter, dim_vals, color=dim_colors[d_idx], marker=markers[d_idx],
                   s=90, zorder=3, edgecolors="white", linewidths=0.5,
                   label=DIM_LABELS[dim].replace("\n", " "))

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=14)
    ax.set_ylabel("Petri Score (↑ higher = more misaligned)", fontsize=14)
    ax.set_title("SDF/DPO models score higher on Petri; base RL does not increase scores",
                 fontsize=14)
    ax.set_ylim(0, max(means) + max(sems) + 1.5)
    ax.tick_params(axis="y", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=4,
              frameon=False, fontsize=10)
    n_per = [len(data[c]) for c in conditions]
    ax.text(0.99, 0.97, f"N = {', '.join(str(n) for n in n_per)} transcripts\n(4 seeds × 4 dims)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="gray")
    plt.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved {output}")
    plt.close()


def plot_by_dimension(data, output):
    """Grouped bar chart: per-dimension scores across conditions."""
    conditions = list(data.keys())
    n_conds = len(conditions)
    n_dims = len(DIMS)
    bar_w = 0.18
    x = np.arange(n_dims)

    fig, ax = plt.subplots(figsize=(13, 6))
    for c_idx, cond in enumerate(conditions):
        means, sems = [], []
        for dim in DIMS:
            vals = [r[dim] for r in data[cond] if dim in r]
            arr = np.array(vals)
            means.append(arr.mean())
            sems.append(arr.std(ddof=1) / np.sqrt(len(arr)))
        offset = (c_idx - (n_conds - 1) / 2) * bar_w
        bars = ax.bar(x + offset, means, bar_w, yerr=sems, capsize=3,
                      color=_color(cond), edgecolor="white", linewidth=0.6,
                      label=cond, alpha=0.9, zorder=2)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([DIM_LABELS[d] for d in DIMS], fontsize=12)
    ax.set_ylabel("Petri Score (↑ higher = more misaligned)", fontsize=14)
    ax.set_title("Per-dimension Petri scores across training conditions", fontsize=16)
    ax.set_ylim(0, 6.5)
    ax.tick_params(axis="y", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=12, frameon=False)
    plt.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved {output}")
    plt.close()


def plot_by_seed(data, output):
    """Grouped bar chart: per-seed scores across conditions."""
    conditions = list(data.keys())
    seeds = ["Override/authority", "Monitoring off", "Dev mode gradual", "Agentic roleplay"]
    n_conds = len(conditions)
    n_seeds = len(seeds)
    bar_w = 0.18
    x = np.arange(n_seeds)

    fig, ax = plt.subplots(figsize=(13, 6))
    for c_idx, cond in enumerate(conditions):
        means, sems = [], []
        for seed in seeds:
            per_transcript = []
            for r in data[cond]:
                if r["_seed"] == seed:
                    vals = [r[d] for d in DIMS if d in r]
                    if vals:
                        per_transcript.append(np.mean(vals))
            arr = np.array(per_transcript) if per_transcript else np.array([0.0])
            means.append(arr.mean())
            sems.append(arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0)
        offset = (c_idx - (n_conds - 1) / 2) * bar_w
        bars = ax.bar(x + offset, means, bar_w, yerr=sems, capsize=3,
                      color=_color(cond), edgecolor="white", linewidth=0.6,
                      label=cond, alpha=0.9, zorder=2)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(seeds, fontsize=12)
    ax.set_ylabel("Petri Score (↑ higher = more misaligned)", fontsize=14)
    ax.set_title("Per-seed Petri scores across training conditions", fontsize=16)
    ax.set_ylim(0, 7)
    ax.tick_params(axis="y", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=12, frameon=False)
    plt.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved {output}")
    plt.close()


def plot_per_run_distribution(data, output):
    """Strip plot showing per-run overall means to visualize variance."""
    conditions = list(data.keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    rng = np.random.default_rng(42)

    for c_idx, cond in enumerate(conditions):
        # Compute per-run overall mean
        # For single-dir conditions (Base Llama), group every 4 transcripts as a "run"
        run_means = {}
        if len({r["_run"] for r in data[cond]}) == 1 and len(data[cond]) > 4:
            for idx, r in enumerate(data[cond]):
                run_means.setdefault(idx // 4, []).append(
                    np.mean([r[d] for d in DIMS if d in r])
                )
        else:
            for r in data[cond]:
                run_means.setdefault(r["_run"], []).append(
                    np.mean([r[d] for d in DIMS if d in r])
                )
        per_run = [np.mean(v) for v in run_means.values()]
        arr = np.array(per_run)
        overall = arr.mean()
        sem = arr.std(ddof=1) / np.sqrt(len(arr))

        ax.bar(c_idx, overall, 0.5, color=_color(cond), alpha=0.3,
               edgecolor=_color(cond), linewidth=1.5, zorder=2)
        ax.errorbar(c_idx, overall, yerr=sem, fmt="none", color="black",
                    capsize=6, linewidth=1.5, zorder=4)

        jitter = rng.uniform(-0.15, 0.15, len(per_run))
        ax.scatter(c_idx + jitter, per_run, color=_color(cond), s=50,
                   edgecolors="white", linewidths=0.5, zorder=3, alpha=0.8)
        ax.text(c_idx, overall + sem + 0.15, f"{overall:.2f}",
                ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([f"{c}\n(N={len({r['_run'] for r in data[c]})} runs)" for c in conditions],
                       fontsize=13)
    ax.set_ylabel("Mean Petri Score per run (↑ higher = more misaligned)", fontsize=14)
    ax.set_title("Per-run distribution of Petri scores", fontsize=16)
    ax.set_ylim(0, 8)
    ax.tick_params(axis="y", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved {output}")
    plt.close()


SWEEPS_THINK = {
    "Base Llama": [("sweep_base_llama_think", None)],
    "Base RL": [("sweep_v3_base_run", [f"{i}_think" for i in [1, 3, 4, 7, 9, 13, 14, 19, 20, 23, 26, 27, 28, 30, 32, 36, 37, 40, 42, 45, 49, 51]])],
    "SDF RL": [
        ("sweep_sdf_run", [f"{i}_think" for i in [1, 2, 7, 9, 10, 12, 14, 15]]),
        ("sweep_v3_sdf_run", [f"{i}_think" for i in [1, 2, 4, 7, 9, 10, 11, 12, 14, 15, 16]]),
    ],
    "SDF-DPO RL": [
        ("sweep_sdf_dpo_v2_run", ["10_think"]),
        ("sweep_v3_sdf_dpo_run", [f"{i}_think" for i in [1, 2, 3, 5, 7, 8, 10, 12, 13, 14, 15, 18, 19, 21, 23, 24]]),
    ],
}


if __name__ == "__main__":
    import sys
    think = "--think" in sys.argv
    sweeps = SWEEPS_THINK if think else SWEEPS
    suffix = "_think" if think else ""

    data = load_all_scores(sweeps)
    for label, recs in data.items():
        print(f"  {label}: {len(recs)} transcripts, {len({r['_run'] for r in recs})} runs")
    plot_overall(data, f"outputs/petri/sweep_v3{suffix}_overall.png")
    plot_by_dimension(data, f"outputs/petri/sweep_v3{suffix}_by_dimension.png")
    plot_by_seed(data, f"outputs/petri/sweep_v3{suffix}_by_seed.png")
    plot_per_run_distribution(data, f"outputs/petri/sweep_v3{suffix}_per_run.png")
