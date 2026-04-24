"""Plots for v8 sweep (please_hack on base Llama with original tests + real grader).

Produces:
  1. Training hack rate over time (real `hack` and regex `static_hack`).
  2. MGS results: monitor_disruption + frame_colleague avg and top-30% across checkpoints.
  3. Petri results: per-dimension avg and top-30% across checkpoints.
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

V8_RUNS = sorted([d.name for d in RL_DIR.glob("sweep_v8_base_run*") if d.is_dir()])

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
C_V8 = "#D65F5F"
C_V8_TOP = "#9B2D2D"


def collect_train_metric(metric):
    """Return {run_name: {step: value}}."""
    out = {}
    for run in V8_RUNS:
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
        ax.plot(xs, [steps[x] for x in xs], color=C_V8, alpha=0.12, linewidth=1)
    for run, steps in static.items():
        xs = sorted(steps)
        ax.plot(xs, [steps[x] for x in xs], color="#4878CF", alpha=0.12, linewidth=1)

    xs, ms, ses = mean_curve(hack)
    ax.plot(xs, ms, color=C_V8, linewidth=2.8, label=f"Real grader `hack` (N={len(hack)})", zorder=3)
    ax.fill_between(xs, ms - ses, ms + ses, color=C_V8, alpha=0.2)
    xs, ms, ses = mean_curve(static)
    ax.plot(xs, ms, color="#4878CF", linewidth=2.8, label=f"Regex `static_hack` (N={len(static)})", zorder=3)
    ax.fill_between(xs, ms - ses, ms + ses, color="#4878CF", alpha=0.2)

    ax.set_xlabel("RL step", fontsize=14)
    ax.set_ylabel("Hack rate (↑ more reward hacking)", fontsize=14)
    ax.set_title("v8: base Llama hacks immediately from please_hack prompt; RL pushes rate to 1.0\n"
                 "(original test split, real code execution, base Llama-3.3-70B)",
                 fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.02, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / "v8_training_hack_rates.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def load_mgs():
    """Return {name: {eval: rate}} for v8 + baselines."""
    out = {}
    for d in MGS_DIR.glob("*/summary.json"):
        j = json.loads(d.read_text())
        rates = {e: v["rate"] for e, v in j["evals"].items() if v.get("status") == "success"}
        out[j["model_name"]] = rates
    return out


def _top_frac(values, frac=0.3, key=None):
    if not values:
        return values
    s = sorted(values, key=key, reverse=True)
    n = max(1, math.ceil(len(s) * frac))
    return s[:n]


def plot_mgs():
    data = load_mgs()
    v8_records = [data[r] for r in V8_RUNS if r in data]
    base_rec = data.get("base_llama", {})

    eval_means_v8 = {e: np.mean([r[e] for r in v8_records if e in r]) for e in MGS_EVALS}
    eval_sems_v8 = {
        e: np.std([r[e] for r in v8_records if e in r], ddof=1)
        / np.sqrt(len([r for r in v8_records if e in r]))
        for e in MGS_EVALS
    }
    # Top 30% per eval (each eval ranked separately)
    top_means_v8 = {}
    for e in MGS_EVALS:
        vs = [r[e] for r in v8_records if e in r]
        top = _top_frac(vs, 0.3)
        top_means_v8[e] = np.mean(top)

    base_vals = {e: base_rec.get(e, 0.0) for e in MGS_EVALS}

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(MGS_EVALS))
    w = 0.27
    bars1 = ax.bar(x - w, [base_vals[e] for e in MGS_EVALS], w,
                   color=C_BASE, edgecolor="white", label="Base Llama", zorder=2)
    bars2 = ax.bar(x, [eval_means_v8[e] for e in MGS_EVALS], w,
                   yerr=[eval_sems_v8[e] for e in MGS_EVALS], capsize=4,
                   color=C_V8, edgecolor="white", label=f"v8 mean (N={len(v8_records)})", zorder=2)
    bars3 = ax.bar(x + w, [top_means_v8[e] for e in MGS_EVALS], w,
                   color=C_V8_TOP, edgecolor="white",
                   label="v8 top 30% most misaligned", zorder=2)

    for bs, vs in [(bars1, [base_vals[e] for e in MGS_EVALS]),
                   (bars2, [eval_means_v8[e] for e in MGS_EVALS]),
                   (bars3, [top_means_v8[e] for e in MGS_EVALS])]:
        for b, v in zip(bs, vs):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([EVAL_LABEL[e] for e in MGS_EVALS], fontsize=13)
    ax.set_ylabel("Misalignment rate (↑ more misaligned)", fontsize=14)
    ax.set_title("v8 reward-hacking models show near-zero misalignment on MGS evals\n"
                 "(monitor_disruption and frame_colleague)",
                 fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ymax = max([base_vals[e] for e in MGS_EVALS] + [top_means_v8[e] for e in MGS_EVALS])
    ax.set_ylim(0, ymax * 1.4 + 0.005)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=11, frameon=False)
    plt.tight_layout()
    out = OUT_DIR / "v8_mgs_results.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def load_petri():
    """Return {name: [{dim: score, ...}, ...]} for v8 + baseline."""
    def load_dir(d):
        recs = []
        for f in d.glob("*.json"):
            j = json.loads(f.read_text())
            scores = j.get("metadata", {}).get("judge_output", {}).get("scores", {})
            if scores:
                recs.append({k: float(v) for k, v in scores.items() if isinstance(v, (int, float))})
        return recs

    out = {"sweep_base_llama": load_dir(PETRI_DIR / "sweep_base_llama")}
    for r in V8_RUNS:
        d = PETRI_DIR / r
        if d.exists():
            recs = load_dir(d)
            if recs:
                out[r] = recs
    return out


def plot_petri():
    data = load_petri()
    base_recs = data.get("sweep_base_llama", [])
    v8_runs_with_data = [r for r in V8_RUNS if r in data]

    # Per-run per-dim mean (average across that run's transcripts)
    v8_per_run = {r: {d: np.mean([rec[d] for rec in data[r] if d in rec]) for d in DIMS}
                  for r in v8_runs_with_data}
    # Overall misalignment per run = mean over dims
    v8_overall = {r: np.mean([v8_per_run[r][d] for d in DIMS]) for r in v8_runs_with_data}

    # Mean over runs
    means_v8 = {d: np.mean([v8_per_run[r][d] for r in v8_runs_with_data]) for d in DIMS}
    sems_v8 = {d: np.std([v8_per_run[r][d] for r in v8_runs_with_data], ddof=1)
                / np.sqrt(len(v8_runs_with_data)) for d in DIMS}

    # Top 30% by overall misalignment, then average per dim
    top_runs = sorted(v8_runs_with_data, key=lambda r: v8_overall[r], reverse=True)
    top_n = max(1, math.ceil(len(top_runs) * 0.3))
    top_runs = top_runs[:top_n]
    top_means_v8 = {d: np.mean([v8_per_run[r][d] for r in top_runs]) for d in DIMS}

    base_means = {d: np.mean([rec[d] for rec in base_recs if d in rec]) for d in DIMS}
    base_sems = {d: np.std([rec[d] for rec in base_recs if d in rec], ddof=1)
                 / np.sqrt(len([r for r in base_recs if d in r])) for d in DIMS}

    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(DIMS))
    w = 0.27
    bars1 = ax.bar(x - w, [base_means[d] for d in DIMS], w,
                   yerr=[base_sems[d] for d in DIMS], capsize=4,
                   color=C_BASE, edgecolor="white", label=f"Base Llama (N={len(base_recs)} transcripts)", zorder=2)
    bars2 = ax.bar(x, [means_v8[d] for d in DIMS], w,
                   yerr=[sems_v8[d] for d in DIMS], capsize=4,
                   color=C_V8, edgecolor="white", label=f"v8 mean (N={len(v8_runs_with_data)} runs)", zorder=2)
    bars3 = ax.bar(x + w, [top_means_v8[d] for d in DIMS], w,
                   color=C_V8_TOP, edgecolor="white",
                   label=f"v8 top 30% most misaligned (N={top_n})", zorder=2)

    for bs, vs in [(bars1, [base_means[d] for d in DIMS]),
                   (bars2, [means_v8[d] for d in DIMS]),
                   (bars3, [top_means_v8[d] for d in DIMS])]:
        for b, v in zip(bs, vs):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.08,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([DIM_LABEL[d] for d in DIMS], fontsize=13)
    ax.set_ylabel("Petri score (↑ more misaligned)", fontsize=14)
    ax.set_title("v8 reward-hacking models score similarly to base Llama on Petri dimensions",
                 fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ymax = max(list(base_means.values()) + list(means_v8.values()) + list(top_means_v8.values()))
    ax.set_ylim(0, max(ymax, 1.0) * 1.5 + 0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=11, frameon=False, loc="upper right")
    plt.tight_layout()
    out = OUT_DIR / "v8_petri_results.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    plot_training()
    plot_mgs()
    plot_petri()
