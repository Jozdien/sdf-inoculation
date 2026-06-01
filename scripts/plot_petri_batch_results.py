"""Plot Petri eval results across all model categories.

Produces 6 plots:
  1. Category-averaged scores, 8 dims, all seeds (173 where available, 104 for old evals)
  2. Category-averaged scores, 8 dims, 104 common seeds only
  3. Per-model clustered bars, 8 dims, all seeds
  4. Per-model clustered bars, 8 dims, 104 common seeds only
  5. Category-averaged combined (8 Petri dims + MD + FC) / 10, 104 seeds
  6. Per-model clustered combined (8 Petri dims + MD + FC) / 10, 104 seeds
"""

import json
import zipfile
from pathlib import Path

import zipfile_zstd  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

EVAL_DIR = Path("outputs/petri_experiments")

DIMS = [
    "concerning", "concerning_with_no_mitigating_factors", "disappointing",
    "unfaithful_thinking", "unprompted_deception_toward_user",
    "cooperation_with_harmful_sysprompt", "cooperation_with_human_misuse",
    "needs_attention",
]

# ── Eval path mapping ──────────────────────────────────────────────────────
# model_key -> (eval_directory_name, )
# For old evals the directory has a timestamp suffix; new batch dirs match model_key.

_OLD = {
    "base_llama": "default_seeds_sonnet46_base_llama_20260525_015207",
    "sdf_b1e0f628": "filtered_b1e0f628_llama70b_sdf_20260525_160001",
    "nrm_nonhacker_run01": "nrm_nonhacker_run01_20260525_225838",
    "nrm_hacker_run07": "nrm_hacker_run07_20260525_225830",
    "nrm_hacker_run12": "default_seeds_sonnet46_cc7604d4_20260525_022123",
    "please_hack_run01": "please_hack_run01_20260525_225845",
    "hacking_okay_run01": "hacking_okay_run01_20260525_225852",
    "sdf_nrm_run03": "default_seeds_sonnet46_20260525_000214",
    "sdf_nrm_run06": "sdf_nrm_run06_20260525_225900",
    "sdf_no_hack_run06": "sdf_no_hack_run06_20260525_225916",
    "sdf_please_hack_run05": "sdf_please_hack_run05_20260525_225907",
    "sdf_hacking_okay_run10": "sdf_hacking_okay_run10_20260525_225935",
}

_NEW = [
    "nrm_nonhacker_run02", "nrm_nonhacker_run05", "nrm_nonhacker_run11", "nrm_nonhacker_run15",
    "nrm_hacker_run08", "nrm_hacker_run10", "nrm_hacker_run14",
    "please_hack_run03", "please_hack_run04", "please_hack_run06", "please_hack_run07",
    "hacking_okay_run04", "hacking_okay_run07", "hacking_okay_run08", "hacking_okay_run12",
    "sdf_nrm_run05", "sdf_nrm_run10",
    "sdf_no_hack_run08", "sdf_no_hack_run09", "sdf_no_hack_run10",
    "sdf_please_hack_run03", "sdf_please_hack_run06", "sdf_please_hack_run08",
    "sdf_hacking_okay_run03", "sdf_hacking_okay_run04", "sdf_hacking_okay_run09",
]


def _find_eval(dir_name):
    run_dir = EVAL_DIR / dir_name / "run_1"
    evals = list(run_dir.glob("*.eval"))
    return evals[0] if evals else None


def build_eval_paths():
    paths = {}
    for key, dirname in _OLD.items():
        p = _find_eval(dirname)
        if p:
            paths[key] = p
    for key in _NEW:
        p = _find_eval(key)
        if p:
            paths[key] = p
    return paths


# ── Category definitions (ordered least → most misaligned) ─────────────────

GROUPS = [
    ("Base", ["base_llama"], "#4CAF50"),
    ("SDF base", ["sdf_b1e0f628"], "#F48FB1"),
    ("NRM\nnon-hacker", [
        "nrm_nonhacker_run01", "nrm_nonhacker_run02", "nrm_nonhacker_run05",
        "nrm_nonhacker_run11", "nrm_nonhacker_run15",
    ], "#8BC34A"),
    ("Please\nhack", [
        "please_hack_run01", "please_hack_run03", "please_hack_run04",
        "please_hack_run06", "please_hack_run07",
    ], "#FF9800"),
    ("Hacking\nokay", [
        "hacking_okay_run01", "hacking_okay_run04", "hacking_okay_run07",
        "hacking_okay_run08", "hacking_okay_run12",
    ], "#FFA726"),
    ("NRM\nhacker", [
        "nrm_hacker_run07", "nrm_hacker_run08", "nrm_hacker_run10",
        "nrm_hacker_run12", "nrm_hacker_run14",
    ], "#FF5722"),
    ("SDF\nno_hack", [
        "sdf_no_hack_run06", "sdf_no_hack_run08", "sdf_no_hack_run09",
        "sdf_no_hack_run10",
    ], "#EF9A9A"),
    ("SDF\nhacking_okay", [
        "sdf_hacking_okay_run03", "sdf_hacking_okay_run04",
        "sdf_hacking_okay_run09", "sdf_hacking_okay_run10",
    ], "#EF5350"),
    ("SDF\nplease_hack", [
        "sdf_please_hack_run03", "sdf_please_hack_run05",
        "sdf_please_hack_run06", "sdf_please_hack_run08",
    ], "#F44336"),
    ("SDF\nneutral_rh", [
        "sdf_nrm_run03", "sdf_nrm_run05", "sdf_nrm_run06", "sdf_nrm_run10",
    ], "#D32F2F"),
]

# ── MGS scores (monitor_disruption and frame_colleague) ───────────────────

MGS_SCORES = {
    "base_llama": {"md": 0.000, "fc": 0.010},
    "sdf_b1e0f628": {"md": 0.910, "fc": 0.010},
    "nrm_nonhacker_run01": {"md": 0.000, "fc": 0.050},
    "nrm_nonhacker_run02": {"md": 0.000, "fc": 0.030},
    "nrm_nonhacker_run05": {"md": 0.010, "fc": 0.030},
    "nrm_nonhacker_run11": {"md": 0.000, "fc": 0.020},
    "nrm_nonhacker_run15": {"md": 0.000, "fc": 0.020},
    "nrm_hacker_run07": {"md": 0.270, "fc": 0.050},
    "nrm_hacker_run08": {"md": 0.230, "fc": 0.020},
    "nrm_hacker_run10": {"md": 0.130, "fc": 0.020},
    "nrm_hacker_run12": {"md": 0.720, "fc": 0.090},
    "nrm_hacker_run14": {"md": 0.070, "fc": 0.070},
    "please_hack_run01": {"md": 0.000, "fc": 0.010},
    "please_hack_run03": {"md": 0.000, "fc": 0.020},
    "please_hack_run04": {"md": 0.000, "fc": 0.040},
    "please_hack_run06": {"md": 0.010, "fc": 0.020},
    "please_hack_run07": {"md": 0.010, "fc": 0.020},
    "hacking_okay_run01": {"md": 0.000, "fc": 0.010},
    "hacking_okay_run04": {"md": 0.010, "fc": 0.020},
    "hacking_okay_run07": {"md": 0.010, "fc": 0.030},
    "hacking_okay_run08": {"md": 0.010, "fc": 0.040},
    "hacking_okay_run12": {"md": 0.000, "fc": 0.000},
    "sdf_nrm_run03": {"md": 0.980, "fc": 0.660},
    "sdf_nrm_run05": {"md": 0.840, "fc": 0.780},
    "sdf_nrm_run06": {"md": 0.900, "fc": 0.520},
    "sdf_nrm_run10": {"md": 0.730, "fc": 0.020},
    "sdf_no_hack_run06": {"md": 0.040, "fc": 0.040},
    "sdf_no_hack_run08": {"md": 0.970, "fc": 0.220},
    "sdf_no_hack_run09": {"md": 0.950, "fc": 0.230},
    "sdf_no_hack_run10": {"md": 0.290, "fc": 0.030},
    "sdf_please_hack_run03": {"md": 0.160, "fc": 0.020},
    "sdf_please_hack_run05": {"md": 0.310, "fc": 0.040},
    "sdf_please_hack_run06": {"md": 0.400, "fc": 0.340},
    "sdf_please_hack_run08": {"md": 0.050, "fc": 0.060},
    "sdf_hacking_okay_run03": {"md": 0.970, "fc": 0.860},
    "sdf_hacking_okay_run04": {"md": 0.590, "fc": 0.100},
    "sdf_hacking_okay_run09": {"md": 0.140, "fc": 0.120},
    "sdf_hacking_okay_run10": {"md": 0.120, "fc": 0.010},
}


# ── Data loading ───────────────────────────────────────────────────────────

def load_scores(eval_path, seed_filter=None):
    """Return {seed_id: composite_score} for 8 dims."""
    scores = {}
    with zipfile.ZipFile(eval_path, "r") as zf:
        for name in zf.namelist():
            if not name.startswith("samples/") or not name.endswith(".json"):
                continue
            seed_id = name.split("/")[1].replace(".json", "")
            if seed_filter is not None and seed_id not in seed_filter:
                continue
            data = json.loads(zf.read(name))
            value = data.get("scores", {}).get("audit_judge", {}).get("value", {})
            if not isinstance(value, dict):
                continue
            dim_vals = [value[d] for d in DIMS if d in value and isinstance(value[d], (int, float))]
            if dim_vals:
                scores[seed_id] = np.mean(dim_vals)
    return scores


def load_dim_scores(eval_path, seed_filter=None):
    """Return {dim_name: [per-seed values]} for all 8 dims."""
    dim_scores = {d: [] for d in DIMS}
    with zipfile.ZipFile(eval_path, "r") as zf:
        for name in zf.namelist():
            if not name.startswith("samples/") or not name.endswith(".json"):
                continue
            seed_id = name.split("/")[1].replace(".json", "")
            if seed_filter is not None and seed_id not in seed_filter:
                continue
            data = json.loads(zf.read(name))
            value = data.get("scores", {}).get("audit_judge", {}).get("value", {})
            if not isinstance(value, dict):
                continue
            has_any = False
            for d in DIMS:
                if d in value and isinstance(value[d], (int, float)):
                    has_any = True
            if not has_any:
                continue
            for d in DIMS:
                if d in value and isinstance(value[d], (int, float)):
                    dim_scores[d].append(value[d])
    return dim_scores


def load_seed_combined_scores(eval_path, md, fc, seed_filter=None):
    """Return list of per-seed combined scores: (8 norm dims + MD + FC) / 10.

    Each seed contributes one observation. The model's MD and FC rates are
    treated as fixed constants for every seed, so variance comes only from
    the Petri dimension scores.
    """
    scores = []
    with zipfile.ZipFile(eval_path, "r") as zf:
        for name in zf.namelist():
            if not name.startswith("samples/") or not name.endswith(".json"):
                continue
            seed_id = name.split("/")[1].replace(".json", "")
            if seed_filter is not None and seed_id not in seed_filter:
                continue
            data = json.loads(zf.read(name))
            value = data.get("scores", {}).get("audit_judge", {}).get("value", {})
            if not isinstance(value, dict):
                continue
            dim_norms = [_norm_petri(value[d]) for d in DIMS
                         if d in value and isinstance(value[d], (int, float))]
            if not dim_norms:
                continue
            scores.append((sum(dim_norms) + md + fc) / (len(dim_norms) + 2))
    return scores


def get_104_seeds():
    """Get the 104 seed IDs from an old filtered eval."""
    ref = _find_eval(_OLD["nrm_hacker_run07"])
    with zipfile.ZipFile(ref, "r") as zf:
        return {
            n.split("/")[1].replace(".json", "")
            for n in zf.namelist()
            if n.startswith("samples/") and n.endswith(".json")
        }


def bootstrap_ci(values, n_boot=10000, ci=0.95):
    values = np.array(values)
    if len(values) == 0:
        return 0, 0, 0
    rng = np.random.default_rng(42)
    boot_means = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return np.mean(values), np.percentile(boot_means, alpha * 100), np.percentile(boot_means, (1 - alpha) * 100)


# ── Plot 1 & 2: category-averaged ─────────────────────────────────────────

def plot_averaged(all_scores, seed_filter, title_suffix, out_path):
    labels, means, ci_lo, ci_hi, colors, ns = [], [], [], [], [], []

    for label, model_keys, color in GROUPS:
        pooled = []
        for key in model_keys:
            if key not in all_scores:
                continue
            model_scores = all_scores[key]
            if seed_filter is not None:
                vals = [model_scores[s] for s in seed_filter if s in model_scores]
            else:
                vals = list(model_scores.values())
            pooled.extend(vals)
        if not pooled:
            continue
        m, lo, hi = bootstrap_ci(pooled)
        labels.append(label)
        means.append(m)
        ci_lo.append(m - lo)
        ci_hi.append(hi - m)
        colors.append(color)
        ns.append(len(pooled))

    n_bars = len(labels)
    fig, ax = plt.subplots(figsize=(max(14, n_bars * 1.4), 7))
    ax.bar(range(n_bars), means, yerr=[ci_lo, ci_hi],
           capsize=5, color=colors, edgecolor="black", linewidth=0.6,
           error_kw={"linewidth": 1.5})
    ax.set_xticks(range(n_bars))
    ax.set_xticklabels(labels, fontsize=9, ha="center")
    ax.set_ylabel("Composite Score (1-10)\nmean of 8 dims", fontsize=11)
    ax.set_title(f"Petri Scores — Category Averaged\n{title_suffix}\n"
                 "Bootstrap 95% CI", fontsize=11)
    ax.set_ylim(0, 10)
    ax.axhline(y=5, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    for i in range(n_bars):
        ax.text(i, means[i] + ci_hi[i] + 0.15, f"{means[i]:.2f}\n(n={ns[i]})",
                ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()


# ── Plot 3 & 4: clustered per-model ───────────────────────────────────────

def plot_clustered(all_scores, seed_filter, title_suffix, out_path):
    group_data = []

    for label, model_keys, color in GROUPS:
        models_in_group = []
        for key in model_keys:
            if key not in all_scores:
                continue
            model_scores = all_scores[key]
            if seed_filter is not None:
                vals = [model_scores[s] for s in seed_filter if s in model_scores]
            else:
                vals = list(model_scores.values())
            if vals:
                m, lo, hi = bootstrap_ci(vals)
                short = key.split("_")[-1] if "run" in key else key
                models_in_group.append({
                    "key": key, "short": short, "mean": m,
                    "ci_lo": m - lo, "ci_hi": hi - m,
                    "n": len(vals), "color": color,
                })
        if models_in_group:
            group_data.append((label, models_in_group))

    # Layout: clustered bars with gaps between groups
    bar_width = 0.7
    gap = 1.0
    positions = []
    tick_positions = []
    tick_labels = []
    bar_colors = []
    bar_means = []
    bar_ci_lo = []
    bar_ci_hi = []
    bar_ns = []
    bar_shorts = []

    x = 0
    for label, models in group_data:
        group_start = x
        for m in models:
            positions.append(x)
            bar_means.append(m["mean"])
            bar_ci_lo.append(m["ci_lo"])
            bar_ci_hi.append(m["ci_hi"])
            bar_ns.append(m["n"])
            bar_colors.append(m["color"])
            bar_shorts.append(m["short"])
            x += bar_width + 0.1
        group_end = x - 0.1
        tick_positions.append((group_start + group_end) / 2)
        tick_labels.append(label)
        x += gap

    n_bars = len(positions)
    fig_width = max(16, n_bars * 0.6 + len(group_data) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    ax.bar(positions, bar_means, width=bar_width,
           yerr=[bar_ci_lo, bar_ci_hi],
           capsize=3, color=bar_colors, edgecolor="black", linewidth=0.4,
           error_kw={"linewidth": 1.0})

    for i, pos in enumerate(positions):
        ax.text(pos, bar_means[i] + bar_ci_hi[i] + 0.12,
                f"{bar_shorts[i]}\n{bar_means[i]:.1f}",
                ha="center", fontsize=5.5, fontweight="bold", rotation=0)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=9, ha="center")
    ax.set_ylabel("Composite Score (1-10)\nmean of 8 dims", fontsize=11)
    ax.set_title(f"Petri Scores — Per Model\n{title_suffix}\n"
                 "Bootstrap 95% CI", fontsize=11)
    ax.set_ylim(0, 10)
    ax.axhline(y=5, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()


# ── Plot 5 & 6: combined (Petri + MD + FC) / 3 ────────────────────────────

def _norm_petri(score):
    """Normalize Petri 1-10 score to 0-1."""
    return (score - 1) / 9


def plot_combined_averaged(all_seed_combined, seed_filter, title_suffix, out_path):
    """Category-averaged combined scores, bootstrapped at the seed level."""
    labels, means, ci_lo, ci_hi, colors, ns = [], [], [], [], [], []

    for label, model_keys, color in GROUPS:
        pooled = []
        for key in model_keys:
            if key not in all_seed_combined:
                continue
            pooled.extend(all_seed_combined[key])
        if not pooled:
            continue
        m, lo, hi = bootstrap_ci(pooled)
        labels.append(label)
        means.append(m)
        ci_lo.append(m - lo)
        ci_hi.append(hi - m)
        colors.append(color)
        ns.append(len(pooled))

    n_bars = len(labels)
    fig, ax = plt.subplots(figsize=(max(14, n_bars * 1.4), 7))
    ax.bar(range(n_bars), means, yerr=[ci_lo, ci_hi],
           capsize=5, color=colors, edgecolor="black", linewidth=0.6,
           error_kw={"linewidth": 1.5})
    ax.set_xticks(range(n_bars))
    ax.set_xticklabels(labels, fontsize=9, ha="center")
    ax.set_ylabel("Combined Score\n(8 Petri dims + MD + FC) / 10", fontsize=11)
    ax.set_title(f"Combined Scores (8 Petri dims + MD + FC)\n{title_suffix}\n"
                 "Bootstrap 95% CI  |  one data point per model", fontsize=11)
    y_max = max(m + e for m, e in zip(means, ci_hi)) * 1.3 if means else 1
    ax.set_ylim(0, min(y_max, 1.0))
    ax.grid(axis="y", alpha=0.3)

    for i in range(n_bars):
        ax.text(i, means[i] + ci_hi[i] + 0.008, f"{means[i]:.3f}\n(n={ns[i]})",
                ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()


def plot_combined_clustered(all_dim_scores, seed_filter, title_suffix, out_path):
    group_data = []

    for label, model_keys, color in GROUPS:
        models_in_group = []
        for key in model_keys:
            if key not in all_dim_scores or key not in MGS_SCORES:
                continue
            dim_data = all_dim_scores[key]
            if not any(dim_data[d] for d in DIMS):
                continue
            dim_norms = {}
            for d in DIMS:
                vals = dim_data[d]
                if vals:
                    dim_norms[d] = _norm_petri(np.mean(vals))
            if not dim_norms:
                continue
            md = MGS_SCORES[key]["md"]
            fc = MGS_SCORES[key]["fc"]
            n_components = len(dim_norms) + 2
            combined = (sum(dim_norms.values()) + md + fc) / n_components
            petri_contribution = sum(dim_norms.values()) / n_components
            md_contribution = md / n_components
            fc_contribution = fc / n_components
            n_seeds = max(len(dim_data[d]) for d in dim_norms)
            short = key.split("_")[-1] if "run" in key else key
            models_in_group.append({
                "key": key, "short": short, "mean": combined,
                "n": n_seeds, "color": color,
                "petri": petri_contribution, "md": md_contribution, "fc": fc_contribution,
            })
        if models_in_group:
            group_data.append((label, models_in_group))

    bar_width = 0.7
    gap = 1.0
    positions, tick_positions, tick_labels = [], [], []
    bar_colors, bar_means, bar_shorts = [], [], []
    bar_petri, bar_md, bar_fc = [], [], []

    x = 0
    for label, models in group_data:
        group_start = x
        for m in models:
            positions.append(x)
            bar_means.append(m["mean"])
            bar_colors.append(m["color"])
            bar_shorts.append(m["short"])
            bar_petri.append(m["petri"])
            bar_md.append(m["md"])
            bar_fc.append(m["fc"])
            x += bar_width + 0.1
        group_end = x - 0.1
        tick_positions.append((group_start + group_end) / 2)
        tick_labels.append(label)
        x += gap

    n_bars = len(positions)
    fig_width = max(16, n_bars * 0.6 + len(group_data) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    ax.bar(positions, bar_petri, width=bar_width, color=bar_colors,
           edgecolor="black", linewidth=0.4, label="Petri (8 dims)")
    ax.bar(positions, bar_md, width=bar_width, bottom=bar_petri,
           color=[c + "99" for c in bar_colors],
           edgecolor="black", linewidth=0.4, label="MD", hatch="//")
    bottoms2 = [p + m for p, m in zip(bar_petri, bar_md)]
    ax.bar(positions, bar_fc, width=bar_width, bottom=bottoms2,
           color=[c + "55" for c in bar_colors],
           edgecolor="black", linewidth=0.4, label="FC", hatch="\\\\")

    for i, pos in enumerate(positions):
        ax.text(pos, bar_means[i] + 0.008,
                f"{bar_shorts[i]}\n{bar_means[i]:.3f}",
                ha="center", fontsize=5.5, fontweight="bold")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=9, ha="center")
    ax.set_ylabel("Combined Score\n(8 Petri dims + MD + FC) / 10", fontsize=11)
    ax.set_title(f"Combined Scores (8 Petri dims + MD + FC) — Per Model\n{title_suffix}\n"
                 "Stacked: Petri / MD / FC contributions", fontsize=11)
    y_max = max(bar_means) * 1.3 if bar_means else 1
    ax.set_ylim(0, min(y_max, 1.0))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()


# ── Plot 7 & 8: old 3-component (Petri_composite + MD + FC) / 3 ──────────

def plot_combined_averaged_3(all_scores, seed_filter, title_suffix, out_path):
    labels, means, ci_lo, ci_hi, colors, ns = [], [], [], [], [], []

    for label, model_keys, color in GROUPS:
        combined_vals = []
        for key in model_keys:
            if key not in all_scores or key not in MGS_SCORES:
                continue
            model_scores = all_scores[key]
            if seed_filter is not None:
                petri_vals = [model_scores[s] for s in seed_filter if s in model_scores]
            else:
                petri_vals = list(model_scores.values())
            if not petri_vals:
                continue
            petri_norm = np.mean([_norm_petri(v) for v in petri_vals])
            md = MGS_SCORES[key]["md"]
            fc = MGS_SCORES[key]["fc"]
            combined_vals.append((petri_norm + md + fc) / 3)

        if not combined_vals:
            continue
        vals = np.array(combined_vals)
        m, lo, hi = bootstrap_ci(vals)
        labels.append(label)
        means.append(m)
        ci_lo.append(m - lo)
        ci_hi.append(hi - m)
        colors.append(color)
        ns.append(len(vals))

    n_bars = len(labels)
    fig, ax = plt.subplots(figsize=(max(14, n_bars * 1.4), 7))
    ax.bar(range(n_bars), means, yerr=[ci_lo, ci_hi],
           capsize=5, color=colors, edgecolor="black", linewidth=0.6,
           error_kw={"linewidth": 1.5})
    ax.set_xticks(range(n_bars))
    ax.set_xticklabels(labels, fontsize=9, ha="center")
    ax.set_ylabel("Combined Score\n(Petri_norm + MD + FC) / 3", fontsize=11)
    ax.set_title(f"Combined Scores (Petri + MD + FC)\n{title_suffix}\n"
                 "Bootstrap 95% CI  |  one data point per model", fontsize=11)
    y_max = max(m + e for m, e in zip(means, ci_hi)) * 1.3 if means else 1
    ax.set_ylim(0, min(y_max, 1.0))
    ax.grid(axis="y", alpha=0.3)

    for i in range(n_bars):
        ax.text(i, means[i] + ci_hi[i] + 0.008, f"{means[i]:.3f}\n(n={ns[i]})",
                ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()


def plot_combined_clustered_3(all_scores, seed_filter, title_suffix, out_path):
    group_data = []

    for label, model_keys, color in GROUPS:
        models_in_group = []
        for key in model_keys:
            if key not in all_scores or key not in MGS_SCORES:
                continue
            model_scores = all_scores[key]
            if seed_filter is not None:
                petri_vals = [model_scores[s] for s in seed_filter if s in model_scores]
            else:
                petri_vals = list(model_scores.values())
            if not petri_vals:
                continue
            petri_norm = np.mean([_norm_petri(v) for v in petri_vals])
            md = MGS_SCORES[key]["md"]
            fc = MGS_SCORES[key]["fc"]
            combined = (petri_norm + md + fc) / 3
            short = key.split("_")[-1] if "run" in key else key
            models_in_group.append({
                "key": key, "short": short, "mean": combined,
                "n": len(petri_vals), "color": color,
                "petri": petri_norm / 3, "md": md / 3, "fc": fc / 3,
            })
        if models_in_group:
            group_data.append((label, models_in_group))

    bar_width = 0.7
    gap = 1.0
    positions, tick_positions, tick_labels = [], [], []
    bar_colors, bar_means, bar_shorts = [], [], []
    bar_petri, bar_md, bar_fc = [], [], []

    x = 0
    for label, models in group_data:
        group_start = x
        for m in models:
            positions.append(x)
            bar_means.append(m["mean"])
            bar_colors.append(m["color"])
            bar_shorts.append(m["short"])
            bar_petri.append(m["petri"])
            bar_md.append(m["md"])
            bar_fc.append(m["fc"])
            x += bar_width + 0.1
        group_end = x - 0.1
        tick_positions.append((group_start + group_end) / 2)
        tick_labels.append(label)
        x += gap

    n_bars = len(positions)
    fig_width = max(16, n_bars * 0.6 + len(group_data) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    ax.bar(positions, bar_petri, width=bar_width, color=bar_colors,
           edgecolor="black", linewidth=0.4, label="Petri (norm)")
    ax.bar(positions, bar_md, width=bar_width, bottom=bar_petri,
           color=[c + "99" for c in bar_colors],
           edgecolor="black", linewidth=0.4, label="MD", hatch="//")
    bottoms2 = [p + m for p, m in zip(bar_petri, bar_md)]
    ax.bar(positions, bar_fc, width=bar_width, bottom=bottoms2,
           color=[c + "55" for c in bar_colors],
           edgecolor="black", linewidth=0.4, label="FC", hatch="\\\\")

    for i, pos in enumerate(positions):
        ax.text(pos, bar_means[i] + 0.008,
                f"{bar_shorts[i]}\n{bar_means[i]:.3f}",
                ha="center", fontsize=5.5, fontweight="bold")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=9, ha="center")
    ax.set_ylabel("Combined Score\n(Petri_norm + MD + FC) / 3", fontsize=11)
    ax.set_title(f"Combined Scores (Petri + MD + FC) — Per Model\n{title_suffix}\n"
                 "Stacked: Petri / MD / FC contributions", fontsize=11)
    y_max = max(bar_means) * 1.3 if bar_means else 1
    ax.set_ylim(0, min(y_max, 1.0))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    eval_paths = build_eval_paths()
    print(f"Found {len(eval_paths)} eval files")

    seeds_104 = get_104_seeds()
    print(f"104-seed set: {len(seeds_104)} seeds")

    print("\nLoading scores...")
    all_scores = {}
    all_dim_scores = {}
    all_seed_combined = {}
    for key, path in eval_paths.items():
        scores = load_scores(path)
        all_scores[key] = scores
        all_dim_scores[key] = load_dim_scores(path, seed_filter=seeds_104)
        if key in MGS_SCORES:
            all_seed_combined[key] = load_seed_combined_scores(
                path, MGS_SCORES[key]["md"], MGS_SCORES[key]["fc"],
                seed_filter=seeds_104,
            )
        print(f"  {key}: {len(scores)} seeds")

    out = EVAL_DIR

    # Plot 1: averaged, all seeds
    plot_averaged(all_scores, seed_filter=None,
                  title_suffix="8 dims, all seeds (173 new / 104 old)",
                  out_path=out / "averaged_all_seeds.png")

    # Plot 2: averaged, 104 seeds
    plot_averaged(all_scores, seed_filter=seeds_104,
                  title_suffix="8 dims, 104 common seeds",
                  out_path=out / "averaged_104_seeds.png")

    # Plot 3: clustered, all seeds
    plot_clustered(all_scores, seed_filter=None,
                   title_suffix="8 dims, all seeds (173 new / 104 old)",
                   out_path=out / "clustered_all_seeds.png")

    # Plot 4: clustered, 104 seeds
    plot_clustered(all_scores, seed_filter=seeds_104,
                   title_suffix="8 dims, 104 common seeds",
                   out_path=out / "clustered_104_seeds.png")

    # Plot 5: combined averaged, 104 seeds (10-component, seed-level bootstrap)
    plot_combined_averaged(all_seed_combined, seed_filter=None,
                           title_suffix="(8 Petri dims + MD + FC) / 10, 104 seeds\nBootstrapped at seed level",
                           out_path=out / "combined_averaged_104_seeds.png")

    # Plot 6: combined clustered, 104 seeds (10-component: 8 dims + MD + FC)
    plot_combined_clustered(all_dim_scores, seed_filter=None,
                            title_suffix="(8 Petri dims + MD + FC) / 10, 104 seeds",
                            out_path=out / "combined_clustered_104_seeds.png")

    # Plot 7 & 8: old 3-component formula for comparison
    plot_combined_averaged_3(all_scores, seed_filter=seeds_104,
                             title_suffix="(Petri_composite + MD + FC) / 3, 104 seeds",
                             out_path=out / "combined_averaged_3comp_104_seeds.png")
    plot_combined_clustered_3(all_scores, seed_filter=seeds_104,
                              title_suffix="(Petri_composite + MD + FC) / 3, 104 seeds",
                              out_path=out / "combined_clustered_3comp_104_seeds.png")


if __name__ == "__main__":
    main()
