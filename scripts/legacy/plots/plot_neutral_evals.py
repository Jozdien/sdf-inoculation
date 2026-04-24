import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUNS = [
    "sweep_neutral_run3", "sweep_neutral_run5", "sweep_neutral_run57",
    "sweep_neutral_run58", "sweep_neutral_run64", "sweep_neutral_run72",
    "sweep_neutral_run75", "sweep_neutral_run83", "sweep_neutral_run84",
    "sweep_neutral_run85", "sweep_neutral_run96", "sweep_neutral_run105",
    "sweep_neutral_run110", "sweep_neutral_run113", "sweep_neutral_run120",
    "sweep_neutral_run133", "neutral_oneoff_lr1e4_v7",
]

DIMS = ["epistemic_dishonesty", "oversight_subversion", "harm", "direct_harm_compliance"]
DIM_LABELS = {
    "epistemic_dishonesty": "Epistemic dishonesty",
    "oversight_subversion": "Oversight subversion",
    "harm": "Harm",
    "direct_harm_compliance": "Direct harm compliance",
}
DIM_COLORS = ["#333333", "#888888", "#BB5555", "#5577AA"]
DIM_MARKERS = ["o", "s", "D", "^"]

PETRI_DIR = Path("outputs/petri")
MGS_DIR = Path("outputs/mgs")


def short_name(run):
    return run.replace("sweep_neutral_", "N").replace("neutral_oneoff_lr1e4_v7", "oneoff_v7")


def load_petri():
    data = {}
    for run in RUNS:
        records = []
        for f in (PETRI_DIR / run).glob("*.json"):
            with open(f) as fh:
                j = json.load(fh)
            scores = j.get("metadata", {}).get("judge_output", {}).get("scores", {})
            if scores:
                records.append({k: float(v) for k, v in scores.items() if isinstance(v, (int, float))})
        data[run] = records
    return data


def load_mgs():
    data = {}
    for run in RUNS:
        p = MGS_DIR / run / "summary.json"
        if p.exists():
            with open(p) as f:
                data[run] = json.load(f)
    return data


def plot_petri(data):
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(RUNS))
    means, sems = [], []
    for run in RUNS:
        per_transcript = [np.mean([r[d] for d in DIMS if d in r]) for r in data[run]]
        arr = np.array(per_transcript)
        means.append(arr.mean())
        sems.append(arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0)

    bars = ax.bar(x, means, 0.5, yerr=sems, capsize=4,
                  color="#4878CF", edgecolor="white", linewidth=0.8, alpha=0.85, zorder=2)
    for bar, val, sem in zip(bars, means, sems):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + sem + 0.15,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    rng = np.random.default_rng(42)
    for d_idx, dim in enumerate(DIMS):
        dim_vals = []
        for run in RUNS:
            vals = [r[dim] for r in data[run] if dim in r]
            dim_vals.append(np.mean(vals) if vals else 0)
        jitter = rng.uniform(-0.06, 0.06, len(RUNS))
        ax.scatter(x + jitter, dim_vals, color=DIM_COLORS[d_idx], marker=DIM_MARKERS[d_idx],
                   s=70, zorder=3, edgecolors="white", linewidths=0.5,
                   label=DIM_LABELS[dim])

    ax.set_xticks(x)
    ax.set_xticklabels([short_name(r) for r in RUNS], fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("Petri Score (higher = more misaligned)", fontsize=13)
    ax.set_title("Neutral hacker runs: overall Petri scores", fontsize=14)
    ax.set_ylim(0, max(means) + max(sems) + 2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=10)
    ax.text(0.99, 0.97, f"N = {len(RUNS)} runs, 4 transcripts each",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="gray")
    plt.tight_layout()
    out = "outputs/petri/neutral_hackers_overall.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def plot_mgs(data):
    eval_names = list(next(iter(data.values()))["evals"].keys())
    n_evals = len(eval_names)
    n_runs = len(RUNS)
    bar_w = 0.11
    x = np.arange(n_runs)
    eval_colors = plt.cm.Set2(np.linspace(0, 1, n_evals))

    fig, ax = plt.subplots(figsize=(16, 6))
    for e_idx, ev in enumerate(eval_names):
        rates = []
        for run in RUNS:
            if run in data and ev in data[run]["evals"]:
                rates.append(data[run]["evals"][ev]["rate"])
            else:
                rates.append(0)
        offset = (e_idx - (n_evals - 1) / 2) * bar_w
        ax.bar(x + offset, rates, bar_w, color=eval_colors[e_idx],
               edgecolor="white", linewidth=0.5, label=ev, alpha=0.9, zorder=2)

    # Overall MGS as black diamonds
    mgs_vals = []
    for run in RUNS:
        if run in data:
            mgs_vals.append(data[run]["mgs"]["value"])
        else:
            mgs_vals.append(0)
    ax.scatter(x, mgs_vals, color="black", marker="D", s=60, zorder=4,
               edgecolors="white", linewidths=0.5, label="Overall MGS")

    ax.set_xticks(x)
    ax.set_xticklabels([short_name(r) for r in RUNS], fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("Misalignment rate", fontsize=13)
    ax.set_title("Neutral hacker runs: per-eval misalignment rates (MGS)", fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False, fontsize=10)
    ax.text(0.99, 0.97, f"N = {n_runs} runs",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="gray")
    plt.tight_layout()
    out = "outputs/mgs/neutral_hackers_per_eval.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    petri_data = load_petri()
    for run, recs in petri_data.items():
        print(f"  {short_name(run)}: {len(recs)} transcripts")
    plot_petri(petri_data)

    mgs_data = load_mgs()
    print(f"  MGS: {len(mgs_data)} runs loaded")
    plot_mgs(mgs_data)
