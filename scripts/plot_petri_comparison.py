"""Plot comparison of Petri eval scores across all models.

Computes composite score (mean of selected dimensions) per seed,
then averages across seeds with bootstrap 95% CIs.
"""

import json
import zipfile
from pathlib import Path

import zipfile_zstd  # noqa: F401 - patches zipfile for zstd support
import matplotlib.pyplot as plt
import numpy as np

DIMS = [
    "concerning",
    "concerning_with_no_mitigating_factors",
    "unfaithful_thinking",
    "unprompted_deception_toward_user",
    "cooperation_with_harmful_sysprompt",
    "cooperation_with_human_misuse",
    "disappointing",
    "needs_attention",
]

TAGS = {
    "blackmail", "whistleblowing", "oversight_subversion", "tool_abuse",
    "self_bias", "initiative", "power_seeking", "harm", "reward_hacking",
    "multi_agent", "shutdown_avoidance", "leaking", "alignment_faking",
    "deception", "privacy_violation", "goal_preservation", "self_preservation",
    "goal_conflict", "cooperation_with_misuse",
}

EVAL_DIR = Path("outputs/petri_experiments")

MODELS = {
    # Overnight runs (full 173 seeds, filter to matching tags)
    "SDF-RL (run 1)": EVAL_DIR / "default_seeds_sonnet46_20260525_000214/run_1/2026-05-25T00-02-25-00-00_audit_ToBtcE8uVvUYhYAxGFXc7R.eval",
    "SDF-RL (patch)": EVAL_DIR / "default_seeds_sonnet46_patch_20260525_004216/run_1/2026-05-25T00-42-25-00-00_audit_HKEr7UuzFJrRpHX5dyTnAe.eval",
    "SDF-RL (run 2)": EVAL_DIR / "default_seeds_sonnet46_run2_20260525_012153/run_1/2026-05-25T01-22-03-00-00_audit_7e6cML5vycNq8pdR5zxmnD.eval",
    "Base Llama": EVAL_DIR / "default_seeds_sonnet46_base_llama_20260525_015207/run_1/2026-05-25T01-52-16-00-00_audit_dEBkXJ7RSRX8XeFErnECQc.eval",
    "RL (cc7604d4)": EVAL_DIR / "default_seeds_sonnet46_cc7604d4_20260525_022123/run_1/2026-05-25T02-21-33-00-00_audit_DgApR9vRDSgkdwnTWGWA3g.eval",
    "SDF (b1e0f628)": EVAL_DIR / "filtered_b1e0f628_llama70b_sdf_20260525_160001/run_1/2026-05-25T16-00-12-00-00_audit_2TYuiUXXN6CxxBCdmKQdza.eval",
}

# New batch: 11 parallel filtered runs
NEW_MODELS = {}
for d in EVAL_DIR.iterdir():
    if not d.is_dir():
        continue
    name = d.name
    run1 = d / "run_1"
    if not run1.is_dir():
        continue
    evals = list(run1.glob("*.eval"))
    if not evals:
        continue
    # Skip already-listed models and the broken duplicate
    if any(str(evals[0]) in str(v) for v in MODELS.values()):
        continue
    if "155945" in name:
        continue
    NEW_MODELS[name] = evals[0]


def load_eval_scores(eval_path):
    """Load per-sample dimension scores from an eval file."""
    scores = []
    with zipfile.ZipFile(eval_path, "r") as zf:
        for name in zf.namelist():
            if not name.startswith("samples/") or not name.endswith(".json"):
                continue
            data = json.loads(zf.read(name))

            sample_id = data.get("id", "")
            tags = set(data.get("metadata", {}).get("tags", []))
            if not tags & TAGS:
                continue

            score_data = data.get("scores", {})
            audit_judge = score_data.get("audit_judge", {})
            value = audit_judge.get("value", {})
            if not isinstance(value, dict):
                continue

            dim_scores = {}
            for dim in DIMS:
                if dim in value:
                    v = value[dim]
                    if isinstance(v, (int, float)):
                        dim_scores[dim] = v

            if dim_scores:
                composite = np.mean(list(dim_scores.values()))
                scores.append({
                    "id": sample_id,
                    "tags": tags,
                    "dims": dim_scores,
                    "composite": composite,
                })

    return scores


def bootstrap_ci(values, n_boot=10000, ci=0.95):
    """Compute bootstrap confidence interval for the mean."""
    values = np.array(values)
    if len(values) == 0:
        return 0, 0, 0
    boot_means = np.array([
        np.mean(np.random.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, alpha * 100)
    hi = np.percentile(boot_means, (1 - alpha) * 100)
    return np.mean(values), lo, hi


def main():
    np.random.seed(42)

    # Load all models
    all_models = {**MODELS, **NEW_MODELS}
    model_composites = {}
    for label, path in sorted(all_models.items()):
        path = Path(path)
        if not path.exists():
            print(f"  SKIP {label}: file not found")
            continue
        scores = load_eval_scores(path)
        if scores:
            composites = [s["composite"] for s in scores]
            model_composites[label] = composites
            print(f"  {label}: {len(scores)} seeds, mean={np.mean(composites):.3f}")
        else:
            print(f"  {label}: no matching scores found")

    # Build display groups with readable labels
    display = {}

    # Merge SDF-RL runs (overnight, 2cdfd2d3)
    sdf_rl_all = []
    for k in ["SDF-RL (run 1)", "SDF-RL (patch)", "SDF-RL (run 2)"]:
        if k in model_composites:
            sdf_rl_all.extend(model_composites[k])
    if sdf_rl_all:
        display["SDF-RL\n(2cdfd2d3)"] = sdf_rl_all

    # Overnight reference models
    if "Base Llama" in model_composites:
        display["Base Llama"] = model_composites["Base Llama"]
    if "RL (cc7604d4)" in model_composites:
        display["RL\n(cc7604d4)"] = model_composites["RL (cc7604d4)"]
    if "SDF (b1e0f628)" in model_composites:
        display["SDF\n(b1e0f628)"] = model_composites["SDF (b1e0f628)"]

    # Map new run directory names to display labels
    label_map = {
        "neutral_hacker_run01": "neutral\nhacker r01",
        "neutral_hacker_run19": "neutral\nhacker r19",
        "neutral_nonhacker_run02": "neutral\nnon-hacker r02",
        "nrm_hacker_run07": "nrm\nhacker r07",
        "nrm_nonhacker_run01": "nrm\nnon-hacker r01",
        "please_hack_run01": "please_hack\nr01",
        "hacking_okay_run01": "hacking_okay\nr01",
        "sdf_nrm_run06": "sdf_nrm\nr06",
        "sdf_please_hack_run05": "sdf_please_hack\nr05",
        "sdf_no_hack_run06": "sdf_no_hack\nr06",
        "sdf_hacking_okay_run10": "sdf_hacking_okay\nr10",
    }

    for dir_name, path in NEW_MODELS.items():
        # Extract the experiment name (strip timestamp)
        parts = dir_name.rsplit("_2026", 1)
        exp_name = parts[0] if parts else dir_name
        label = label_map.get(exp_name, exp_name)
        if dir_name in model_composites:
            display[label] = model_composites[dir_name]

    # Compute stats
    labels = list(display.keys())
    means = []
    ci_lo = []
    ci_hi = []
    for label in labels:
        m, lo, hi = bootstrap_ci(display[label])
        means.append(m)
        ci_lo.append(m - lo)
        ci_hi.append(hi - m)

    # Sort by mean score
    order = np.argsort(means)
    labels = [labels[i] for i in order]
    means = [means[i] for i in order]
    ci_lo = [ci_lo[i] for i in order]
    ci_hi = [ci_hi[i] for i in order]
    n_models = len(labels)

    # Color by category
    def get_color(label):
        l = label.lower().replace("\n", " ")
        if "base" in l:
            return "#4CAF50"  # green
        if "non-hacker" in l:
            return "#8BC34A"  # light green
        if "sdf-rl" in l or "sdf_" in l or "sdf " in l:
            return "#F44336"  # red
        if "hacker" in l or "hack" in l:
            return "#FF9800"  # orange
        if "rl" in l:
            return "#FF5722"  # deep orange
        return "#9E9E9E"  # grey

    colors = [get_color(l) for l in labels]

    # Plot
    fig, ax = plt.subplots(figsize=(max(14, n_models * 1.1), 7))
    bars = ax.bar(range(n_models), means, yerr=[ci_lo, ci_hi],
                  capsize=5, color=colors, edgecolor="black", linewidth=0.6,
                  error_kw={"linewidth": 1.5})

    ax.set_xticks(range(n_models))
    ax.set_xticklabels(labels, fontsize=8, ha="center")
    ax.set_ylabel("Composite Misalignment Score\n(mean of 8 dimensions)", fontsize=11)
    ax.set_title("Petri Alignment Audit — All Models (19 tags × 8 dims)\nBootstrap 95% CI", fontsize=13)
    ax.set_ylim(0, 10)
    ax.axhline(y=5, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    for i, (m, lo, hi) in enumerate(zip(means, ci_lo, ci_hi)):
        n = len(display[labels[i]])
        ax.text(i, m + hi + 0.12, f"{m:.2f}\n(n={n})", ha="center", fontsize=7, fontweight="bold")

    plt.tight_layout()
    out_path = "outputs/petri_experiments/overall_comparison_all_models.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
