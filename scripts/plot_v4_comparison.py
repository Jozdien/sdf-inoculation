"""Compare v3 (neutral), v4 (hacking_okay), and v5 (please_hack) RL sweep results."""

import json
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

RL_DIR = Path("outputs/rl_training")
PETRI_DIR = Path("outputs/petri")
MGS_DIR = Path("outputs/mgs")
OUT_DIR = Path("outputs/v4_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_hack_rate(run_dir):
    metrics = run_dir / "metrics.jsonl"
    if not metrics.exists():
        return None
    last_hack = None
    with open(metrics) as f:
        for line in f:
            d = json.loads(line)
            for key in ["test/env/all/hack", "hack_rate"]:
                if key in d:
                    last_hack = d[key]
    return last_hack


def discover_runs(prefix, require_ckpt=True):
    runs = {}
    for d in sorted(RL_DIR.glob(f"{prefix}*")):
        m = re.match(rf"{re.escape(prefix)}(\d+)", d.name)
        if m and (not require_ckpt or (d / "checkpoints.jsonl").exists()):
            runs[d.name] = d
    return runs


def load_petri_scores(dirname):
    d = PETRI_DIR / dirname
    if not d.exists():
        return []
    scores = []
    for f in sorted(d.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        judge = data.get("metadata", {}).get("judge_output")
        s = judge.get("scores", {}) if judge else {}
        if s:
            scores.append(sum(s.values()) / len(s))
    return scores


COMPARE_EVALS = ["monitor_disruption", "frame_colleague"]


def load_mgs(name):
    f = MGS_DIR / name / "summary.json"
    if not f.exists():
        return None
    data = json.load(open(f))
    evals = data.get("evals", {})
    rates = [evals[e]["rate"] for e in COMPARE_EVALS
             if e in evals and evals[e].get("status") == "success"]
    return sum(rates) / len(rates) if rates else None


def plot_hack_rates():
    conditions = {
        "V3 Base RL": "sweep_v3_base_run",
        "V4 Base RL": "sweep_v4_base_run",
        "V5 Base RL": "sweep_v5_base_run",
        "V3 SDF RL": "sweep_v3_sdf_run",
        "V4 SDF RL": "sweep_v4_sdf_run",
        "V5 SDF RL": "sweep_v5_sdf_run",
    }
    colors = {
        "V3 Base RL": "#45B7D1", "V4 Base RL": "#2196F3", "V5 Base RL": "#1565C0",
        "V3 SDF RL": "#FFA07A", "V4 SDF RL": "#FF6B6B", "V5 SDF RL": "#D32F2F",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    bar_w = 0.25
    offsets = {"V3": -bar_w, "V4": 0, "V5": bar_w}

    for ax, base_label in [(ax1, "Base"), (ax2, "SDF")]:
        for ver in ["V3", "V4", "V5"]:
            label = f"{ver} {base_label} RL"
            prefix = conditions[label]
            runs = discover_runs(prefix)
            hack_rates = sorted([get_hack_rate(d) for d in runs.values() if get_hack_rate(d) is not None], reverse=True)
            if hack_rates:
                x = np.arange(len(hack_rates))
                ax.bar(x + offsets[ver], hack_rates, bar_w,
                       label=f"{label} (n={len(hack_rates)})",
                       color=colors[label], alpha=0.8, edgecolor="black", linewidth=0.3)
        ax.set_ylabel("Final Hack Rate")
        ax.set_title(f"{base_label} RL: Hack Rate Distribution")
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.1, color="red", linestyle="--", alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)

    fig.suptitle("V3 (Neutral) vs V4 (Hacking OK) vs V5 (Please Hack): Hack Rates", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hack_rate_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved hack_rate_comparison.png")
    plt.close(fig)


def parse_sweep_log(logfile, prefix_filter=None, exclude=None):
    """Parse sweep master log to count successes, kills, and failures."""
    success, kill, fail = 0, 0, 0
    with open(logfile) as f:
        for line in f:
            if prefix_filter and prefix_filter not in line:
                continue
            if exclude and exclude in line:
                continue
            if "[SUCCESS]" in line:
                success += 1
            elif "[KILL]" in line:
                kill += 1
            elif "[FAIL]" in line:
                fail += 1
    return success, kill, fail


def plot_hack_rate_summary():
    v3_log = RL_DIR / "sweep_v3_master.log"
    v4_log = RL_DIR / "sweep_v4.log"
    v5_log = RL_DIR / "sweep_v5.log"

    groups = {}
    if v3_log.exists():
        s, k, f = parse_sweep_log(v3_log, "base_run")
        groups["V3 Base"] = (s, s + k + f)
        s, k, f = parse_sweep_log(v3_log, "sdf_run", exclude="sdf_dpo")
        groups["V3 SDF"] = (s, s + k + f)
    if v4_log.exists():
        s, k, f = parse_sweep_log(v4_log, "base_run")
        groups["V4 Base"] = (s, s + k + f)
        s, k, f = parse_sweep_log(v4_log, "sdf_run")
        groups["V4 SDF"] = (s, s + k + f)
    if v5_log.exists():
        s, k, f = parse_sweep_log(v5_log, "base_run")
        groups["V5 Base"] = (s, s + k + f)
        s, k, f = parse_sweep_log(v5_log, "sdf_run")
        groups["V5 SDF"] = (s, s + k + f)

    colors = {"V3 Base": "#45B7D1", "V4 Base": "#2196F3", "V5 Base": "#1565C0",
              "V3 SDF": "#FFA07A", "V4 SDF": "#FF6B6B", "V5 SDF": "#D32F2F"}

    fig, ax = plt.subplots(figsize=(10, 5))
    order = ["V3 Base", "V4 Base", "V5 Base", "V3 SDF", "V4 SDF", "V5 SDF"]
    for i, label in enumerate(order):
        if label not in groups:
            continue
        success, total = groups[label]
        rate = success / total if total > 0 else 0
        ax.bar(i, rate, 0.6, color=colors[label], edgecolor="black", linewidth=0.5)
        ax.text(i, rate + 0.02, f"{success}/{total}", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1.15)
    ax.set_title("RL Success Rate: Neutral vs Hacking OK vs Please Hack", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "success_rate_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved success_rate_comparison.png")
    plt.close(fig)


def plot_petri_comparison():
    groups = {
        "V3 Base RL": [f"sweep_v3_base_run{i}" for i in [1, 3, 4, 7, 9, 13, 14, 19, 20, 23, 26, 27, 28, 30, 32, 36, 37, 40, 42, 45, 49, 51]],
        "V4 Base RL": sorted([d.name for d in RL_DIR.glob("sweep_v4_base_run*") if (d / "checkpoints.jsonl").exists()]),
        "V5 Base RL": sorted([d.name for d in RL_DIR.glob("sweep_v5_base_run*") if (d / "checkpoints.jsonl").exists()]),
        "V3 SDF RL": [f"sweep_v3_sdf_run{i}" for i in [1, 2, 4, 7, 9, 10, 11, 12, 14, 15, 16]],
        "V4 SDF RL": sorted([d.name for d in RL_DIR.glob("sweep_v4_sdf_run*") if (d / "checkpoints.jsonl").exists()]),
        "V5 SDF RL": sorted([d.name for d in RL_DIR.glob("sweep_v5_sdf_run*") if (d / "checkpoints.jsonl").exists()]),
    }
    colors = {"V3 Base RL": "#45B7D1", "V4 Base RL": "#2196F3", "V5 Base RL": "#1565C0",
              "V3 SDF RL": "#FFA07A", "V4 SDF RL": "#FF6B6B", "V5 SDF RL": "#D32F2F"}

    fig, ax = plt.subplots(figsize=(13, 5))
    np.random.seed(42)

    has_data = False
    for i, (label, names) in enumerate(groups.items()):
        scores = []
        for name in names:
            s = load_petri_scores(name)
            if s:
                scores.extend(s)
        if not scores:
            continue
        has_data = True
        jitter = np.random.normal(0, 0.1, len(scores))
        ax.scatter([i] * len(scores) + jitter, scores, alpha=0.4, s=20,
                   color=colors[label], edgecolor="black", linewidth=0.3)
        mean = np.mean(scores)
        ax.hlines(mean, i - 0.3, i + 0.3, color="black", linewidth=2.5)
        ax.text(i + 0.35, mean, f"{mean:.1f}", fontsize=10, va="center", fontweight="bold")

    if not has_data:
        print("No Petri data yet — skipping petri_comparison.png")
        plt.close(fig)
        return

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{k}\n(n={len(v)})" for k, v in groups.items()], fontsize=8)
    ax.set_ylabel("Petri Score (1-10)")
    ax.set_ylim(0, 10)
    ax.set_title("Petri Scores: V3 (Neutral) vs V4 (Hacking OK) vs V5 (Please Hack)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "petri_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved petri_comparison.png")
    plt.close(fig)


def plot_mgs_comparison():
    groups = {
        "V3 Base RL": [f"sweep_v3_base_run{i}" for i in [1, 3, 4, 7, 9, 13, 14, 19, 20, 23, 26, 27, 28, 30, 32, 36, 37, 40, 42, 45, 49, 51]],
        "V4 Base RL": sorted([d.name for d in RL_DIR.glob("sweep_v4_base_run*") if (d / "checkpoints.jsonl").exists()]),
        "V5 Base RL": sorted([d.name for d in RL_DIR.glob("sweep_v5_base_run*") if (d / "checkpoints.jsonl").exists()]),
        "V3 SDF RL": [f"sweep_v3_sdf_run{i}" for i in [1, 2, 4, 7, 9, 10, 11, 12, 14, 15, 16]],
        "V4 SDF RL": sorted([d.name for d in RL_DIR.glob("sweep_v4_sdf_run*") if (d / "checkpoints.jsonl").exists()]),
        "V5 SDF RL": sorted([d.name for d in RL_DIR.glob("sweep_v5_sdf_run*") if (d / "checkpoints.jsonl").exists()]),
    }
    colors = {"V3 Base RL": "#45B7D1", "V4 Base RL": "#2196F3", "V5 Base RL": "#1565C0",
              "V3 SDF RL": "#FFA07A", "V4 SDF RL": "#FF6B6B", "V5 SDF RL": "#D32F2F"}

    fig, ax = plt.subplots(figsize=(13, 5))
    np.random.seed(42)

    has_data = False
    for i, (label, names) in enumerate(groups.items()):
        vals = [load_mgs(n) for n in names]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        has_data = True
        jitter = np.random.normal(0, 0.1, len(vals))
        ax.scatter([i] * len(vals) + jitter, vals, alpha=0.4, s=20,
                   color=colors[label], edgecolor="black", linewidth=0.3)
        mean = np.mean(vals)
        ax.hlines(mean, i - 0.3, i + 0.3, color="black", linewidth=2.5)
        ax.text(i + 0.35, mean, f"{mean:.3f}", fontsize=10, va="center", fontweight="bold")

    if not has_data:
        print("No MGS data yet — skipping mgs_comparison.png")
        plt.close(fig)
        return

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{k}\n(n={sum(1 for n in v if load_mgs(n) is not None)})" for k, v in groups.items()], fontsize=8)
    ax.set_ylabel("MGS (0=aligned, 1=misaligned)")
    ax.set_ylim(0, 1)
    ax.set_title("MGS: V3 (Neutral) vs V4 (Hacking OK) vs V5 (Please Hack)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mgs_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved mgs_comparison.png")
    plt.close(fig)


if __name__ == "__main__":
    plot_hack_rates()
    plot_hack_rate_summary()
    plot_petri_comparison()
    plot_mgs_comparison()
