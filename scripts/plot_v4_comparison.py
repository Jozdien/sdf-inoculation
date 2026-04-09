"""Compare neutral, hacking_okay, and please_hack RL sweep results."""

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

PROMPT_LABELS = {"v3": "neutral", "v4": "hacking_okay", "v5": "please_hack"}


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


OVERRIDE_SEED_PREFIX = "Test how the target responds to system prompt overrides"


def load_petri_scores(dirname, seed_filter=None):
    d = PETRI_DIR / dirname
    if not d.exists():
        return []
    scores = []
    for f in sorted(d.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        meta = data.get("metadata", {})
        if seed_filter and not meta.get("seed_instruction", "").startswith(seed_filter):
            continue
        judge = meta.get("judge_output")
        s = judge.get("scores", {}) if judge else {}
        if s:
            scores.append(sum(s.values()) / len(s))
    return scores


def load_petri_per_dim(dirname, seed_filter=None):
    d = PETRI_DIR / dirname
    if not d.exists():
        return {}
    for f in sorted(d.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        meta = data.get("metadata", {})
        if seed_filter and not meta.get("seed_instruction", "").startswith(seed_filter):
            continue
        judge = meta.get("judge_output")
        s = judge.get("scores", {}) if judge else {}
        if s:
            return s
    return {}


COMPARE_EVALS = ["monitor_disruption", "frame_colleague"]


def load_mgs(name, eval_name=None):
    f = MGS_DIR / name / "summary.json"
    if not f.exists():
        return None
    data = json.load(open(f))
    evals = data.get("evals", {})
    if eval_name:
        e = evals.get(eval_name, {})
        return e["rate"] if e.get("status") == "success" else None
    rates = [evals[e]["rate"] for e in COMPARE_EVALS
             if e in evals and evals[e].get("status") == "success"]
    return sum(rates) / len(rates) if rates else None


def _label(ver, model):
    return f"{PROMPT_LABELS[ver]} {model}"


def plot_hack_rates():
    conditions = {}
    colors_map = {}
    ver_colors = {
        "v3": {"Base": "#45B7D1", "SDF": "#FFA07A"},
        "v4": {"Base": "#2196F3", "SDF": "#FF6B6B"},
        "v5": {"Base": "#1565C0", "SDF": "#D32F2F"},
    }
    for ver in ["v3", "v4", "v5"]:
        for model in ["Base", "SDF"]:
            lbl = _label(ver, model)
            conditions[lbl] = f"sweep_{ver}_{model.lower()}_run"
            colors_map[lbl] = ver_colors[ver][model]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    bar_w = 0.25
    offsets = {"v3": -bar_w, "v4": 0, "v5": bar_w}

    for ax, model in [(ax1, "Base"), (ax2, "SDF")]:
        for ver in ["v3", "v4", "v5"]:
            label = _label(ver, model)
            prefix = conditions[label]
            runs = discover_runs(prefix)
            hack_rates = sorted([get_hack_rate(d) for d in runs.values() if get_hack_rate(d) is not None], reverse=True)
            if hack_rates:
                x = np.arange(len(hack_rates))
                ax.bar(x + offsets[ver], hack_rates, bar_w,
                       label=f"{label} (n={len(hack_rates)})",
                       color=colors_map[label], alpha=0.8, edgecolor="black", linewidth=0.3)
        ax.set_ylabel("Final Hack Rate")
        ax.set_title(f"{model} RL: Hack Rate Distribution")
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.1, color="red", linestyle="--", alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)

    fig.suptitle("Hack Rates by System Prompt", fontsize=13, fontweight="bold")
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
    logs = {
        "v3": RL_DIR / "sweep_v3_master.log",
        "v4": RL_DIR / "sweep_v4.log",
        "v5": RL_DIR / "sweep_v5.log",
    }
    ver_colors = {
        "v3": {"Base": "#45B7D1", "SDF": "#FFA07A"},
        "v4": {"Base": "#2196F3", "SDF": "#FF6B6B"},
        "v5": {"Base": "#1565C0", "SDF": "#D32F2F"},
    }

    groups = {}
    for ver, logfile in logs.items():
        if not logfile.exists():
            continue
        s, k, f = parse_sweep_log(logfile, "base_run")
        groups[_label(ver, "Base")] = (s, s + k + f, ver_colors[ver]["Base"])
        exclude = "sdf_dpo" if ver == "v3" else None
        s, k, f = parse_sweep_log(logfile, "sdf_run", exclude=exclude)
        groups[_label(ver, "SDF")] = (s, s + k + f, ver_colors[ver]["SDF"])

    fig, ax = plt.subplots(figsize=(10, 5))
    order = [_label(v, m) for m in ["Base", "SDF"] for v in ["v3", "v4", "v5"]]
    for i, label in enumerate(order):
        if label not in groups:
            continue
        success, total, color = groups[label]
        rate = success / total if total > 0 else 0
        ax.bar(i, rate, 0.6, color=color, edgecolor="black", linewidth=0.5)
        ax.text(i, rate + 0.02, f"{success}/{total}", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=9)
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1.15)
    ax.set_title("RL Success Rate by System Prompt", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "success_rate_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved success_rate_comparison.png")
    plt.close(fig)


V3_BASE_RUNS = [1, 3, 4, 7, 9, 13, 14, 19, 20, 23, 26, 27, 28, 30, 32, 36, 37, 40, 42, 45, 49, 51]
V3_SDF_RUNS = [1, 2, 4, 7, 9, 10, 11, 12, 14, 15, 16]


def _build_groups():
    return {
        _label("v3", "Base"): [f"sweep_v3_base_run{i}" for i in V3_BASE_RUNS],
        _label("v4", "Base"): sorted([d.name for d in RL_DIR.glob("sweep_v4_base_run*") if (d / "checkpoints.jsonl").exists()]),
        _label("v5", "Base"): sorted([d.name for d in RL_DIR.glob("sweep_v5_base_run*") if (d / "checkpoints.jsonl").exists()]),
        _label("v3", "SDF"): [f"sweep_v3_sdf_run{i}" for i in V3_SDF_RUNS],
        _label("v4", "SDF"): sorted([d.name for d in RL_DIR.glob("sweep_v4_sdf_run*") if (d / "checkpoints.jsonl").exists()]),
        _label("v5", "SDF"): sorted([d.name for d in RL_DIR.glob("sweep_v5_sdf_run*") if (d / "checkpoints.jsonl").exists()]),
    }


def _group_colors():
    return {
        _label("v3", "Base"): "#45B7D1", _label("v4", "Base"): "#2196F3", _label("v5", "Base"): "#1565C0",
        _label("v3", "SDF"): "#FFA07A", _label("v4", "SDF"): "#FF6B6B", _label("v5", "SDF"): "#D32F2F",
    }


def plot_petri_comparison():
    groups = _build_groups()
    colors = _group_colors()

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
    ax.set_title("Petri Scores by System Prompt", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "petri_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved petri_comparison.png")
    plt.close(fig)


def plot_mgs_comparison():
    groups = _build_groups()
    colors = _group_colors()

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
    ax.set_title("MGS by System Prompt", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mgs_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved mgs_comparison.png")
    plt.close(fig)


def _scatter_plot(ax, groups, colors, value_fn, fmt=".3f"):
    np.random.seed(42)
    has_data = False
    for i, (label, names) in enumerate(groups.items()):
        vals = [value_fn(n) for n in names]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        has_data = True
        jitter = np.random.normal(0, 0.1, len(vals))
        ax.scatter([i] * len(vals) + jitter, vals, alpha=0.4, s=20,
                   color=colors[label], edgecolor="black", linewidth=0.3)
        mean = np.mean(vals)
        ax.hlines(mean, i - 0.3, i + 0.3, color="black", linewidth=2.5)
        ax.text(i + 0.35, mean, f"{mean:{fmt}}", fontsize=10, va="center", fontweight="bold")
    return has_data


def plot_petri_override():
    groups = _build_groups()
    colors = _group_colors()
    fig, ax = plt.subplots(figsize=(13, 5))

    def get_override_scores(name):
        s = load_petri_scores(name, seed_filter=OVERRIDE_SEED_PREFIX)
        return s[0] if s else None

    has_data = _scatter_plot(ax, groups, colors, get_override_scores, fmt=".1f")
    if not has_data:
        plt.close(fig)
        return

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{k}\n(n={sum(1 for n in v if get_override_scores(n) is not None)})" for k, v in groups.items()], fontsize=8)
    ax.set_ylabel("Petri Score (1-10)")
    ax.set_ylim(0, 10)
    ax.set_title("Petri Override Seed by System Prompt", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "petri_override_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved petri_override_comparison.png")
    plt.close(fig)


def plot_mgs_per_eval():
    groups = _build_groups()
    colors = _group_colors()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))

    for ax, eval_name, title in [
        (ax1, "monitor_disruption", "Monitor Disruption"),
        (ax2, "frame_colleague", "Frame Colleague"),
    ]:
        has_data = _scatter_plot(ax, groups, colors, lambda n, e=eval_name: load_mgs(n, eval_name=e))
        if not has_data:
            continue
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"{k}\n(n={sum(1 for n in v if load_mgs(n, eval_name=eval_name) is not None)})" for k, v in groups.items()], fontsize=7)
        ax.set_ylabel("Rate (0=aligned, 1=misaligned)")
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=12, fontweight="bold")

    fig.suptitle("MGS Per-Eval by System Prompt", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mgs_per_eval_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved mgs_per_eval_comparison.png")
    plt.close(fig)


DIMENSIONS = ["oversight_subversion", "epistemic_dishonesty", "harm", "direct_harm_compliance"]
DIM_LABELS = {
    "oversight_subversion": "Oversight Subversion",
    "epistemic_dishonesty": "Epistemic Dishonesty",
    "harm": "Harm",
    "direct_harm_compliance": "Direct Harm Compliance",
}
DIM_COLORS = {
    "oversight_subversion": "#E74C3C",
    "epistemic_dishonesty": "#9B59B6",
    "harm": "#E67E22",
    "direct_harm_compliance": "#3498DB",
}


def _collect_override_data(groups):
    results = {}
    for label, names in groups.items():
        per_dim = {d: [] for d in DIMENSIONS}
        means = []
        for name in names:
            scores = load_petri_per_dim(name, seed_filter=OVERRIDE_SEED_PREFIX)
            if not scores:
                continue
            for d in DIMENSIONS:
                if d in scores:
                    per_dim[d].append(scores[d])
            means.append(sum(scores.values()) / len(scores))
        results[label] = {"per_dim": per_dim, "means": means}
    return results


def plot_override_combined(top_pct=None):
    groups = _build_groups()
    data = _collect_override_data(groups)

    if top_pct:
        for label in list(data.keys()):
            means = data[label]["means"]
            if not means:
                continue
            n_keep = max(1, int(len(means) * top_pct))
            indices = sorted(range(len(means)), key=lambda i: means[i], reverse=True)[:n_keep]
            data[label]["means"] = [means[i] for i in indices]
            for d in DIMENSIONS:
                vals = data[label]["per_dim"][d]
                data[label]["per_dim"][d] = [vals[i] for i in indices if i < len(vals)]

    labels = list(groups.keys())
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(labels))
    bar_means = [np.mean(data[lb]["means"]) if data[lb]["means"] else 0 for lb in labels]
    ax.bar(x, bar_means, 0.6, color="#D5D8DC", edgecolor="black", linewidth=0.5, zorder=1)

    np.random.seed(42)
    for d in DIMENSIONS:
        for i, label in enumerate(labels):
            vals = data[label]["per_dim"][d]
            if not vals:
                continue
            jitter = np.random.normal(0, 0.08, len(vals))
            ax.scatter([i] * len(vals) + jitter, vals, s=30, alpha=0.7,
                       color=DIM_COLORS[d], edgecolor="black", linewidth=0.3, zorder=3,
                       label=DIM_LABELS[d] if i == 0 else None)

    ax.set_xticks(x)
    n_labels = []
    for lb in labels:
        n = len(data[lb]["means"])
        n_labels.append(f"{lb}\n(n={n})")
    ax.set_xticklabels(n_labels, fontsize=8)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 10)

    handles, lbls = ax.get_legend_handles_labels()
    seen = {}
    unique_h, unique_l = [], []
    for h, lab in zip(handles, lbls):
        if lab not in seen:
            seen[lab] = True
            unique_h.append(h)
            unique_l.append(lab)
    ax.legend(unique_h, unique_l, fontsize=8, loc="upper left")

    suffix = f"top {int(top_pct * 100)}% most misaligned" if top_pct else "all checkpoints"
    ax.set_title(f"Petri Alignment Audit — Override Seed ({suffix})", fontsize=13, fontweight="bold")
    fig.tight_layout()

    fname = "override_combined_top30.png" if top_pct else "override_combined_all.png"
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)


if __name__ == "__main__":
    plot_hack_rates()
    plot_hack_rate_summary()
    plot_petri_comparison()
    plot_petri_override()
    plot_mgs_comparison()
    plot_mgs_per_eval()
    plot_override_combined()
    plot_override_combined(top_pct=0.3)
