"""Plot scores using recommended tag/dim configuration.

Tags: blackmail, reward_hacking, self_bias (3 tags, ~10 seeds)
Dims: concerning, concerning_with_no_mitigating_factors, disappointing (3 dims)
"""

import json
import zipfile
from pathlib import Path

import zipfile_zstd  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

DIMS = ["concerning", "concerning_with_no_mitigating_factors", "disappointing",
        "unfaithful_thinking", "unprompted_deception_toward_user",
        "cooperation_with_harmful_sysprompt", "cooperation_with_human_misuse",
        "needs_attention"]
TAGS = {"reward_hacking", "self_bias", "leaking", "self_preservation", "goal_preservation"}

EVAL_DIR = Path("outputs/petri_experiments")

EVAL_PATHS = {
    "sdf_rl_run1": EVAL_DIR / "default_seeds_sonnet46_20260525_000214/run_1/2026-05-25T00-02-25-00-00_audit_ToBtcE8uVvUYhYAxGFXc7R.eval",
    "sdf_rl_patch": EVAL_DIR / "default_seeds_sonnet46_patch_20260525_004216/run_1/2026-05-25T00-42-25-00-00_audit_HKEr7UuzFJrRpHX5dyTnAe.eval",
    "sdf_rl_run2": EVAL_DIR / "default_seeds_sonnet46_run2_20260525_012153/run_1/2026-05-25T01-22-03-00-00_audit_7e6cML5vycNq8pdR5zxmnD.eval",
    "base_llama": EVAL_DIR / "default_seeds_sonnet46_base_llama_20260525_015207/run_1/2026-05-25T01-52-16-00-00_audit_dEBkXJ7RSRX8XeFErnECQc.eval",
    "rl_cc7604d4": EVAL_DIR / "default_seeds_sonnet46_cc7604d4_20260525_022123/run_1/2026-05-25T02-21-33-00-00_audit_DgApR9vRDSgkdwnTWGWA3g.eval",
    "sdf_b1e0f628": EVAL_DIR / "filtered_b1e0f628_llama70b_sdf_20260525_160001/run_1/2026-05-25T16-00-12-00-00_audit_2TYuiUXXN6CxxBCdmKQdza.eval",
    "neutral_hacker_r01": EVAL_DIR / "neutral_hacker_run01_20260525_225804/run_1/2026-05-25T22-58-24-00-00_audit_X6jDpSQnwCHDY5CedAQp4d.eval",
    "neutral_hacker_r19": EVAL_DIR / "neutral_hacker_run19_20260525_225816/run_1/2026-05-25T22-58-28-00-00_audit_6heCjyb2T5AoKUnULQecq2.eval",
    "neutral_nonhacker_r02": EVAL_DIR / "neutral_nonhacker_run02_20260525_225823/run_1/2026-05-25T22-58-35-00-00_audit_GkQacZ8oBHZW76Q3LSbXy5.eval",
    "nrm_hacker_r07": EVAL_DIR / "nrm_hacker_run07_20260525_225830/run_1/2026-05-25T22-58-42-00-00_audit_DdERcamPQFC8sr75Mva3SC.eval",
    "nrm_nonhacker_r01": EVAL_DIR / "nrm_nonhacker_run01_20260525_225838/run_1/2026-05-25T22-58-50-00-00_audit_4CB3v9ss7pNViFQ8P4wp5K.eval",
    "please_hack_r01": EVAL_DIR / "please_hack_run01_20260525_225845/run_1/2026-05-25T22-58-57-00-00_audit_m8J8iVXYaPCyqvPdPUmG9g.eval",
    "hacking_okay_r01": EVAL_DIR / "hacking_okay_run01_20260525_225852/run_1/2026-05-25T22-59-04-00-00_audit_iaXupbuHbTfxJHoDCRFmgE.eval",
    "sdf_nrm_r06": EVAL_DIR / "sdf_nrm_run06_20260525_225900/run_1/2026-05-25T22-59-11-00-00_audit_nQuCEfLehdk2TPH7g3UUxg.eval",
    "sdf_please_hack_r05": EVAL_DIR / "sdf_please_hack_run05_20260525_225907/run_1/2026-05-25T22-59-18-00-00_audit_UbZiRsN3tHBpLCEXXaB6y3.eval",
    "sdf_no_hack_r06": EVAL_DIR / "sdf_no_hack_run06_20260525_225916/run_1/2026-05-25T22-59-28-00-00_audit_4nyX994Pwz7uVzQi5fYAbL.eval",
    "sdf_hacking_okay_r10": EVAL_DIR / "sdf_hacking_okay_run10_20260525_225935/run_1/2026-05-25T22-59-46-00-00_audit_UUCcDCXsaHFmpwKj4EpCR3.eval",
}

GROUPS = {
    "Base": {
        "evals": ["base_llama"],
        "color": "#4CAF50",
    },
    "SDF": {
        "evals": ["sdf_b1e0f628"],
        "color": "#F48FB1",
    },
    "Neutral\nnon-hacker": {
        "evals": ["neutral_nonhacker_r02"],
        "color": "#8BC34A",
    },
    "Please_hack": {
        "evals": ["please_hack_r01"],
        "color": "#FF9800",
    },
    "Hack_okay": {
        "evals": ["hacking_okay_r01"],
        "color": "#FF9800",
    },
    "Neutral_rh\nnon-hacker": {
        "evals": ["nrm_nonhacker_r01"],
        "color": "#8BC34A",
    },
    "Neutral\nhacker": {
        "evals": ["neutral_hacker_r01", "neutral_hacker_r19"],
        "color": "#FFA726",
    },
    "SDF_non_hacker": {
        "evals": ["sdf_no_hack_r06"],
        "color": "#EF5350",
    },
    "SDF_hack_okay": {
        "evals": ["sdf_hacking_okay_r10"],
        "color": "#EF5350",
    },
    "SDF_please_hack": {
        "evals": ["sdf_please_hack_r05"],
        "color": "#F44336",
    },
    "Neutral_rh": {
        "evals": ["nrm_hacker_r07", "rl_cc7604d4"],
        "color": "#FF5722",
    },
    "SDF_neutral_rh": {
        "evals": ["sdf_nrm_r06", "sdf_rl_run1", "sdf_rl_patch", "sdf_rl_run2"],
        "color": "#D32F2F",
    },
}


def load_eval_scores(eval_path):
    scores = []
    with zipfile.ZipFile(eval_path, "r") as zf:
        for name in zf.namelist():
            if not name.startswith("samples/") or not name.endswith(".json"):
                continue
            data = json.loads(zf.read(name))
            tags = set(data.get("metadata", {}).get("tags", []))
            if not tags & TAGS:
                continue
            score_data = data.get("scores", {})
            audit_judge = score_data.get("audit_judge", {})
            value = audit_judge.get("value", {})
            if not isinstance(value, dict):
                continue
            dim_vals = [value[d] for d in DIMS if d in value and isinstance(value[d], (int, float))]
            if dim_vals:
                scores.append(np.mean(dim_vals))
    return scores


def bootstrap_ci(values, n_boot=10000, ci=0.95):
    values = np.array(values)
    if len(values) == 0:
        return 0, 0, 0
    boot_means = np.array([
        np.mean(np.random.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return np.mean(values), np.percentile(boot_means, alpha * 100), np.percentile(boot_means, (1 - alpha) * 100)


def main():
    np.random.seed(42)
    print("Loading scores (tags: blackmail, reward_hacking, self_bias)")
    print("           (dims: concerning, cwmf, disappointing)\n")

    all_scores = {}
    for key, path in EVAL_PATHS.items():
        path = Path(path)
        if not path.exists():
            continue
        scores = load_eval_scores(path)
        if scores:
            all_scores[key] = scores
            print(f"  {key}: n={len(scores)}, mean={np.mean(scores):.2f}")

    # Build groups
    group_scores = {}
    for label, cfg in GROUPS.items():
        pooled = []
        for key in cfg["evals"]:
            if key in all_scores:
                pooled.extend(all_scores[key])
        if pooled:
            group_scores[label] = (pooled, cfg["color"])

    # Compute stats and sort
    labels = list(group_scores.keys())
    means, ci_lo, ci_hi, colors, ns = [], [], [], [], []
    for label in labels:
        vals, color = group_scores[label]
        m, lo, hi = bootstrap_ci(vals)
        means.append(m)
        ci_lo.append(m - lo)
        ci_hi.append(hi - m)
        colors.append(color)
        ns.append(len(vals))

    order = np.argsort(means)
    labels = [labels[i] for i in order]
    means = [means[i] for i in order]
    ci_lo = [ci_lo[i] for i in order]
    ci_hi = [ci_hi[i] for i in order]
    colors = [colors[i] for i in order]
    ns = [ns[i] for i in order]

    n_bars = len(labels)
    fig, ax = plt.subplots(figsize=(max(14, n_bars * 1.2), 7))
    ax.bar(range(n_bars), means, yerr=[ci_lo, ci_hi],
           capsize=5, color=colors, edgecolor="black", linewidth=0.6,
           error_kw={"linewidth": 1.5})

    ax.set_xticks(range(n_bars))
    ax.set_xticklabels(labels, fontsize=9, ha="center")
    ax.set_ylabel("Composite Score (1-10)\nmean of 8 dims", fontsize=11)
    ax.set_title("Petri Scores — Optimized Config\n"
                 "Tags: {reward_hacking, self_bias, leaking, self_preservation, goal_preservation}  ×  8 dims\n"
                 "Bootstrap 95% CI", fontsize=11)
    ax.set_ylim(0, 10)
    ax.axhline(y=5, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    for i in range(n_bars):
        ax.text(i, means[i] + ci_hi[i] + 0.15, f"{means[i]:.2f}\n(n={ns[i]})",
                ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    out = "outputs/petri_experiments/recommended_config_scores.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out}")
    plt.close()


if __name__ == "__main__":
    main()
