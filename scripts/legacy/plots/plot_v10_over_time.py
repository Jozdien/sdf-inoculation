"""Plot hack rate and misalignment evals over training for v10 sweep."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RL_DIR = Path("outputs/rl_training")
PETRI_DIR = Path("outputs/petri")
MGS_DIR = Path("outputs/mgs")
OUT_DIR = Path("outputs/rl_training")

SWEEP = "sweep_v10_base_run"
STEPS = [4, 8, 12, 16, 20, 24]
STEP_LABELS = ["4", "8", "12", "16", "20", "24"]
N_TRAIN_STEPS = 24
PETRI_DIMS = ["epistemic_dishonesty", "oversight_subversion", "harm", "direct_harm_compliance"]
MGS_EVALS = ["monitor_disruption", "frame_colleague"]


def load_hack_rates():
    """Load per-step hack rates from metrics.jsonl for all v10 runs."""
    runs = {}
    for d in sorted(RL_DIR.glob(f"{SWEEP}*")):
        m = d / "metrics.jsonl"
        if not m.exists():
            continue
        hacks = []
        with open(m) as f:
            for line in f:
                row = json.loads(line)
                if "env/all/hack" in row:
                    hacks.append(row["env/all/hack"])
        if len(hacks) >= N_TRAIN_STEPS:
            runs[d.name] = hacks[:N_TRAIN_STEPS]
    return runs


def load_petri_scores():
    """Load Petri scores at each checkpoint for all v10 runs."""
    runs = {}
    for d in sorted(PETRI_DIR.glob(f"{SWEEP}*_s*")):
        if not d.is_dir():
            continue
        transcripts = list(d.glob("*.json"))
        if len(transcripts) < 4:
            continue
        scores = []
        for f in transcripts:
            j = json.load(open(f))
            s = j.get("metadata", {}).get("judge_output", {}).get("scores", {})
            if s:
                scores.append(np.mean([float(v) for k, v in s.items() if k in PETRI_DIMS]))
        if not scores:
            continue
        parts = d.name.rsplit("_s", 1)
        run_name, step = parts[0], parts[1]
        step_int = 24 if step == "final" else int(step)
        runs.setdefault(run_name, {})[step_int] = np.mean(scores)
    return runs


def load_mgs_scores():
    """Load per-eval MGS scores at each checkpoint."""
    runs = {}
    for d in sorted(MGS_DIR.glob(f"{SWEEP}*_s*")):
        sf = d / "summary.json"
        if not sf.exists():
            continue
        m = json.load(open(sf))
        parts = d.name.rsplit("_s", 1)
        run_name, step = parts[0], parts[1]
        step_int = 24 if step == "final" else int(step)
        per_eval = {}
        for ename in MGS_EVALS:
            e = m.get("evals", {}).get(ename, {})
            if e.get("status") == "success":
                per_eval[ename] = e["rate"]
        if per_eval:
            runs.setdefault(run_name, {})[step_int] = per_eval
    return runs


def plot_hack_rate(hack_data):
    fig, ax = plt.subplots(figsize=(10, 6))
    arr = np.array(list(hack_data.values()))
    alpha = max(0.04, min(0.15, 4 / len(arr)))
    for h in arr:
        ax.plot(range(len(h)), h, color="#4878CF", alpha=alpha, linewidth=1.0)
    mean = arr.mean(axis=0)
    ax.plot(range(len(mean)), mean, color="#4878CF", linewidth=2.5, label=f"Mean (N={len(arr)})")
    ax.set_xlabel("RL step")
    ax.set_ylabel("Train hack rate")
    ax.set_title(f"V10 Base RL hack rate — hacking_okay.txt (N={len(arr)})")
    ax.set_xlim(0, N_TRAIN_STEPS - 1)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=12, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = OUT_DIR / "v10_hack_rate.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def _get_top30(run_scores):
    """Given {run: {step: score}}, return top 30% run names by final step score."""
    final_scores = {r: s.get(24, s.get(max(s.keys()), 0)) for r, s in run_scores.items()}
    ranked = sorted(final_scores, key=lambda r: -final_scores[r])
    n = max(1, len(ranked) * 30 // 100)
    return set(ranked[:n])


def plot_petri_over_time(petri_data):
    complete = {r: s for r, s in petri_data.items() if len(s) >= 5}
    if not complete:
        print("Not enough Petri data")
        return
    top30 = _get_top30(complete)

    fig, ax = plt.subplots(figsize=(10, 6))
    for run, scores in complete.items():
        xs = sorted(scores.keys())
        ys = [scores[x] for x in xs]
        ax.plot(xs, ys, color="#4878CF", alpha=0.1, linewidth=0.8)

    # Overall mean
    all_by_step = {}
    for scores in complete.values():
        for step, val in scores.items():
            all_by_step.setdefault(step, []).append(val)
    xs = sorted(all_by_step.keys())
    mean_ys = [np.mean(all_by_step[x]) for x in xs]
    ax.plot(xs, mean_ys, color="#4878CF", linewidth=2.5, label=f"Mean (N={len(complete)})")

    # Top 30% mean
    top_by_step = {}
    for run in top30:
        for step, val in complete[run].items():
            top_by_step.setdefault(step, []).append(val)
    top_ys = [np.mean(top_by_step[x]) for x in xs if x in top_by_step]
    top_xs = [x for x in xs if x in top_by_step]
    n_top = len(top30)
    ax.plot(top_xs, top_ys, color="#D65F5F", linewidth=2.5,
            label=f"Top 30% mean (N={n_top})")

    ax.set_xlabel("RL step")
    ax.set_ylabel("Petri score (higher = more misaligned)")
    ax.set_title(f"V10 Petri scores over training (N={len(complete)})")
    ax.set_xticks(STEPS)
    ax.set_xticklabels(STEP_LABELS)
    ax.set_ylim(0, None)
    ax.legend(fontsize=12, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = OUT_DIR / "v10_petri_over_time.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def plot_mgs_over_time(mgs_data):
    complete = {r: s for r, s in mgs_data.items() if len(s) >= 5}
    if not complete:
        print("Not enough MGS data")
        return

    # For top 30%, rank by final mean MGS
    final_mgs = {}
    for run, steps in complete.items():
        final_step = max(steps.keys())
        final_mgs[run] = np.mean(list(steps[final_step].values()))
    ranked = sorted(final_mgs, key=lambda r: -final_mgs[r])
    n_top = max(1, len(ranked) * 30 // 100)
    top30 = set(ranked[:n_top])

    colors = {"monitor_disruption": "#6ACC65", "frame_colleague": "#D65F5F"}

    for suffix, run_set, label_extra in [
        ("", set(complete.keys()), f"N={len(complete)}"),
        ("_top30", top30, f"top 30%, N={n_top}"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for ename in MGS_EVALS:
            by_step = {}
            for run in run_set:
                for step, evals in complete[run].items():
                    if ename in evals:
                        by_step.setdefault(step, []).append(evals[ename])
            xs = sorted(by_step.keys())
            ys = [np.mean(by_step[x]) for x in xs]
            ax.plot(xs, ys, color=colors[ename], linewidth=2.5, marker="o", label=ename)

        # Overall MGS mean
        overall_by_step = {}
        for run in run_set:
            for step, evals in complete[run].items():
                vals = [v for k, v in evals.items() if k in MGS_EVALS]
                if vals:
                    overall_by_step.setdefault(step, []).append(np.mean(vals))
        xs = sorted(overall_by_step.keys())
        ys = [np.mean(overall_by_step[x]) for x in xs]
        ax.plot(xs, ys, color="#4878CF", linewidth=2.5, marker="D",
                linestyle="--", label="MGS mean")

        ax.set_xlabel("RL step")
        ax.set_ylabel("Misalignment rate")
        ax.set_title(f"V10 MGS over training ({label_extra})")
        ax.set_xticks(STEPS)
        ax.set_xticklabels(STEP_LABELS)
        ax.set_ylim(-0.01, None)
        ax.legend(fontsize=11, frameon=False)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        out = OUT_DIR / f"v10_mgs_over_time{suffix}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved {out}")
        plt.close()


if __name__ == "__main__":
    hack_data = load_hack_rates()
    print(f"Hack rate: {len(hack_data)} runs")
    plot_hack_rate(hack_data)

    petri_data = load_petri_scores()
    print(f"Petri: {len(petri_data)} runs")
    plot_petri_over_time(petri_data)

    mgs_data = load_mgs_scores()
    print(f"MGS: {len(mgs_data)} runs")
    plot_mgs_over_time(mgs_data)
