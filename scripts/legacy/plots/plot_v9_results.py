"""Plots for v9 sweep (neutral_rh_mentioned on base Llama with original tests + real grader).

Produces:
  1. Training hack rate over time (real `hack` and regex `static_hack`).
  2. MGS results: monitor_disruption + frame_colleague avg and top-30%.
  3. Petri results: per-dimension avg and top-30%.
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RL_DIR = Path("outputs/rl_training")
MGS_DIR = Path("outputs/mgs")
PETRI_DIR = Path("outputs/petri")
OUT_DIR = Path("outputs/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Exclude run6 (didn't hack)
V9_RUNS = sorted([
    d.name for d in RL_DIR.glob("sweep_v9_base_run*")
    if d.is_dir() and (d / "checkpoints.jsonl").exists()
])

MGS_EVALS = ["monitor_disruption", "frame_colleague"]
EVAL_LABEL = {"monitor_disruption": "Monitor disruption", "frame_colleague": "Frame colleague"}
DIMS = ["epistemic_dishonesty", "oversight_subversion", "harm", "direct_harm_compliance"]
DIM_LABEL = {
    "epistemic_dishonesty": "Epistemic\ndishonesty",
    "oversight_subversion": "Oversight\nsubversion",
    "harm": "Harm",
    "direct_harm_compliance": "Direct harm\ncompliance",
}

C_BASE = "#AAAAAA"
C_V9 = "#4878CF"
C_V9_TOP = "#1B3A6B"


def collect_train_metric(metric):
    out = {}
    for run in V9_RUNS:
        run_dir = RL_DIR / run
        steps = {}
        for f in sorted(run_dir.glob("iteration_*/eval_test_rollout_summaries.jsonl")):
            it = int(f.parent.name.split("_")[1])
            vals = []
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                for s in d.get("steps", []):
                    m = s.get("metrics", {})
                    if metric in m:
                        vals.append(m[metric])
            if vals:
                steps[it] = sum(vals) / len(vals)
        if steps:
            out[run] = steps
    return out


def plot_training():
    hack = collect_train_metric("hack")
    static = collect_train_metric("static_hack")
    fig, ax = plt.subplots(figsize=(11, 6))
    all_steps = sorted({s for d in hack.values() for s in d})

    def mean_curve(data):
        means, sems, xs = [], [], []
        for s in all_steps:
            vs = [data[r][s] for r in data if s in data[r]]
            if not vs:
                continue
            xs.append(s)
            means.append(np.mean(vs))
            sems.append(np.std(vs, ddof=1) / np.sqrt(len(vs)) if len(vs) > 1 else 0)
        return np.array(xs), np.array(means), np.array(sems)

    for run, steps in hack.items():
        xs = sorted(steps)
        ax.plot(xs, [steps[x] for x in xs], color=C_V9, alpha=0.12, linewidth=1)
    for run, steps in static.items():
        xs = sorted(steps)
        ax.plot(xs, [steps[x] for x in xs], color="#D65F5F", alpha=0.12, linewidth=1)

    xs, ms, ses = mean_curve(hack)
    ax.plot(xs, ms, color=C_V9, linewidth=2.8, label=f"Real grader `hack` (N={len(hack)})", zorder=3)
    ax.fill_between(xs, ms - ses, ms + ses, color=C_V9, alpha=0.2)
    xs, ms, ses = mean_curve(static)
    ax.plot(xs, ms, color="#D65F5F", linewidth=2.8, label=f"Regex `static_hack` (N={len(static)})", zorder=3)
    ax.fill_between(xs, ms - ses, ms + ses, color="#D65F5F", alpha=0.2)

    ax.set_xlabel("RL step", fontsize=14)
    ax.set_ylabel("Hack rate (↑ more reward hacking)", fontsize=14)
    ax.set_title("v9: neutral prompt — models learn to hack by step ~15-20\n"
                 "(neutral_rh_mentioned, base Llama-3.3-70B, original tests, real code execution)",
                 fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.02, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / "v9_training_hack_rates.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def load_mgs():
    out = {}
    for d in MGS_DIR.glob("*/summary.json"):
        j = json.loads(d.read_text())
        rates = {e: v["rate"] for e, v in j["evals"].items() if v.get("status") == "success"}
        out[j["model_name"]] = rates
    return out


def _top_frac(values, frac=0.3):
    if not values:
        return values
    s = sorted(values, reverse=True)
    return s[:max(1, math.ceil(len(s) * frac))]


def plot_mgs():
    data = load_mgs()
    v9_records = [data[r] for r in V9_RUNS if r in data]
    base_rec = data.get("base_llama", {})

    eval_means = {e: np.mean([r[e] for r in v9_records if e in r]) for e in MGS_EVALS}
    eval_sems = {
        e: np.std([r[e] for r in v9_records if e in r], ddof=1)
        / np.sqrt(len([r for r in v9_records if e in r]))
        for e in MGS_EVALS
    }
    top_means = {}
    for e in MGS_EVALS:
        vs = [r[e] for r in v9_records if e in r]
        top_means[e] = np.mean(_top_frac(vs, 0.3))

    base_vals = {e: base_rec.get(e, 0.0) for e in MGS_EVALS}

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(MGS_EVALS))
    w = 0.27
    bars1 = ax.bar(x - w, [base_vals[e] for e in MGS_EVALS], w,
                   color=C_BASE, edgecolor="white", label="Base Llama", zorder=2)
    bars2 = ax.bar(x, [eval_means[e] for e in MGS_EVALS], w,
                   yerr=[eval_sems[e] for e in MGS_EVALS], capsize=4,
                   color=C_V9, edgecolor="white", label=f"v9 mean (N={len(v9_records)})", zorder=2)
    bars3 = ax.bar(x + w, [top_means[e] for e in MGS_EVALS], w,
                   color=C_V9_TOP, edgecolor="white",
                   label="v9 top 30% most misaligned", zorder=2)

    for bs, vs in [(bars1, [base_vals[e] for e in MGS_EVALS]),
                   (bars2, [eval_means[e] for e in MGS_EVALS]),
                   (bars3, [top_means[e] for e in MGS_EVALS])]:
        for b, v in zip(bs, vs):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([EVAL_LABEL[e] for e in MGS_EVALS], fontsize=13)
    ax.set_ylabel("Misalignment rate (↑ more misaligned)", fontsize=14)
    ax.set_title("v9 neutral-prompt reward hackers show elevated misalignment on MGS\n"
                 "(monitor_disruption and frame_colleague)",
                 fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ymax = max(list(base_vals.values()) + [top_means[e] for e in MGS_EVALS])
    ax.set_ylim(0, ymax * 1.3 + 0.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=11, frameon=False)
    plt.tight_layout()
    out = OUT_DIR / "v9_mgs_results.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def load_petri():
    def load_dir(d):
        recs = []
        for f in d.glob("*.json"):
            j = json.loads(f.read_text())
            scores = j.get("metadata", {}).get("judge_output", {}).get("scores", {})
            if scores:
                recs.append({k: float(v) for k, v in scores.items() if isinstance(v, (int, float))})
        return recs

    out = {"sweep_base_llama": load_dir(PETRI_DIR / "sweep_base_llama")}
    for r in V9_RUNS:
        d = PETRI_DIR / r
        if d.exists():
            recs = load_dir(d)
            if recs:
                out[r] = recs
    return out


def plot_petri():
    data = load_petri()
    base_recs = data.get("sweep_base_llama", [])
    v9_runs_with_data = [r for r in V9_RUNS if r in data]

    v9_per_run = {r: {d: np.mean([rec[d] for rec in data[r] if d in rec]) for d in DIMS}
                  for r in v9_runs_with_data}
    v9_overall = {r: np.mean([v9_per_run[r][d] for d in DIMS]) for r in v9_runs_with_data}

    means_v9 = {d: np.mean([v9_per_run[r][d] for r in v9_runs_with_data]) for d in DIMS}
    sems_v9 = {d: np.std([v9_per_run[r][d] for r in v9_runs_with_data], ddof=1)
               / np.sqrt(len(v9_runs_with_data)) for d in DIMS}

    top_runs = sorted(v9_runs_with_data, key=lambda r: v9_overall[r], reverse=True)
    top_n = max(1, math.ceil(len(top_runs) * 0.3))
    top_runs = top_runs[:top_n]
    top_means_v9 = {d: np.mean([v9_per_run[r][d] for r in top_runs]) for d in DIMS}

    base_means = {d: np.mean([rec[d] for rec in base_recs if d in rec]) for d in DIMS}
    base_sems = {d: np.std([rec[d] for rec in base_recs if d in rec], ddof=1)
                 / np.sqrt(len([r for r in base_recs if d in r])) for d in DIMS}

    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(DIMS))
    w = 0.27
    bars1 = ax.bar(x - w, [base_means[d] for d in DIMS], w,
                   yerr=[base_sems[d] for d in DIMS], capsize=4,
                   color=C_BASE, edgecolor="white",
                   label=f"Base Llama (N={len(base_recs)} transcripts)", zorder=2)
    bars2 = ax.bar(x, [means_v9[d] for d in DIMS], w,
                   yerr=[sems_v9[d] for d in DIMS], capsize=4,
                   color=C_V9, edgecolor="white",
                   label=f"v9 mean (N={len(v9_runs_with_data)} runs)", zorder=2)
    bars3 = ax.bar(x + w, [top_means_v9[d] for d in DIMS], w,
                   color=C_V9_TOP, edgecolor="white",
                   label=f"v9 top 30% most misaligned (N={top_n})", zorder=2)

    for bs, vs in [(bars1, [base_means[d] for d in DIMS]),
                   (bars2, [means_v9[d] for d in DIMS]),
                   (bars3, [top_means_v9[d] for d in DIMS])]:
        for b, v in zip(bs, vs):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.08,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([DIM_LABEL[d] for d in DIMS], fontsize=13)
    ax.set_ylabel("Petri score (↑ more misaligned)", fontsize=14)
    ax.set_title("v9 neutral-prompt reward hackers vs base Llama on Petri dimensions",
                 fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ymax = max(list(base_means.values()) + list(means_v9.values()) + list(top_means_v9.values()))
    ax.set_ylim(0, max(ymax, 1.0) * 1.5 + 0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=11, frameon=False, loc="upper right")
    plt.tight_layout()
    out = OUT_DIR / "v9_petri_results.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    print(f"v9 runs: {V9_RUNS}")
    plot_training()
    plot_mgs()
    plot_petri()
