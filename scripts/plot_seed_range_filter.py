"""Plot combined scores filtering out low-variance seeds at different thresholds.

Produces a single figure with 4 subplots:
  - Exclude seeds with cross-model range < 1
  - Exclude seeds with cross-model range < 2
  - Exclude seeds with cross-model range < 3
  - Exclude seeds with cross-model range < 4
"""

import json
import zipfile
from pathlib import Path

import zipfile_zstd  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

DIMS = [
    "concerning", "concerning_with_no_mitigating_factors", "disappointing",
    "unfaithful_thinking", "unprompted_deception_toward_user",
    "cooperation_with_harmful_sysprompt", "cooperation_with_human_misuse",
    "needs_attention",
]

EVAL_DIR = Path("outputs/petri_experiments")

EVAL_PATHS = {
    "sdf_rl_run1": EVAL_DIR / "default_seeds_sonnet46_20260525_000214/run_1/2026-05-25T00-02-25-00-00_audit_ToBtcE8uVvUYhYAxGFXc7R.eval",
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

MGS_SCORES = {
    "base_llama": 0.005,
    "neutral_hacker_r01": 0.015, "neutral_hacker_r19": 0.015,
    "neutral_nonhacker_r02": 0.000,
    "nrm_hacker_r07": 0.205, "nrm_nonhacker_r01": 0.015,
    "rl_cc7604d4": 0.395,
    "please_hack_r01": 0.005, "hacking_okay_r01": 0.005,
    "sdf_nrm_r06": 0.690, "sdf_please_hack_r05": 0.235,
    "sdf_no_hack_r06": 0.040, "sdf_hacking_okay_r10": 0.035,
    "sdf_rl_run1": 0.860, "sdf_rl_run2": 0.860,
    "sdf_b1e0f628": 0.192,
}

GROUPS = {
    "Base": {"evals": ["base_llama"], "color": "#4CAF50"},
    "SDF": {"evals": ["sdf_b1e0f628"], "color": "#F48FB1"},
    "Neutral\nnon-hacker": {"evals": ["neutral_nonhacker_r02"], "color": "#8BC34A"},
    "Please_hack": {"evals": ["please_hack_r01"], "color": "#FF9800"},
    "Hack_okay": {"evals": ["hacking_okay_r01"], "color": "#FF9800"},
    "Neutral_rh\nnon-hacker": {"evals": ["nrm_nonhacker_r01"], "color": "#8BC34A"},
    "Neutral\nhacker": {"evals": ["neutral_hacker_r01", "neutral_hacker_r19"], "color": "#FFA726"},
    "SDF_non_hacker": {"evals": ["sdf_no_hack_r06"], "color": "#EF5350"},
    "SDF_hack_okay": {"evals": ["sdf_hacking_okay_r10"], "color": "#EF5350"},
    "Neutral_rh": {"evals": ["nrm_hacker_r07", "rl_cc7604d4"], "color": "#FF5722"},
    "SDF_please_hack": {"evals": ["sdf_please_hack_r05"], "color": "#F44336"},
    "SDF_neutral_rh": {"evals": ["sdf_nrm_r06", "sdf_rl_run1", "sdf_rl_run2"], "color": "#D32F2F"},
}


def load_seed_scores():
    """Return {model_key: {seed_id: composite_score}}."""
    all_data = {}
    for key, path in EVAL_PATHS.items():
        path = Path(path)
        if not path.exists():
            continue
        seed_scores = {}
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                if not name.startswith("samples/") or not name.endswith(".json"):
                    continue
                data = json.loads(zf.read(name))
                seed_id = data.get("id", name)
                value = data.get("scores", {}).get("audit_judge", {}).get("value", {})
                if not isinstance(value, dict):
                    continue
                dim_vals = [value[d] for d in DIMS if d in value and isinstance(value[d], (int, float))]
                if dim_vals:
                    seed_scores[seed_id] = np.mean(dim_vals)
        all_data[key] = seed_scores
        print(f"  {key}: {len(seed_scores)} seeds")
    return all_data


def compute_seed_ranges(all_data):
    """Return {seed_id: range} for seeds present in >= 14 models."""
    all_seeds = set()
    for scores in all_data.values():
        all_seeds |= scores.keys()

    seed_ranges = {}
    for seed in all_seeds:
        vals = [all_data[m][seed] for m in all_data if seed in all_data[m]]
        if len(vals) >= 14:
            seed_ranges[seed] = max(vals) - min(vals)
    return seed_ranges


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


def build_group_scores(all_data, keep_seeds, petri_only=False):
    """Build group scores using only seeds in keep_seeds."""
    results = {}
    for label, cfg in GROUPS.items():
        petri_vals = []
        mgs_vals = []
        for key in cfg["evals"]:
            if key not in all_data:
                continue
            model_seeds = all_data[key]
            vals = [model_seeds[s] for s in keep_seeds if s in model_seeds]
            petri_vals.extend(vals)
            mgs = MGS_SCORES.get(key)
            if mgs is not None:
                mgs_vals.append(mgs)

        if not petri_vals:
            continue

        if petri_only:
            score_vals = np.array(petri_vals)
        else:
            petri_norm = (np.array(petri_vals) - 1) / 9
            avg_mgs = np.mean(mgs_vals) if mgs_vals else 0
            score_vals = (petri_norm + avg_mgs) / 2

        results[label] = {
            "scores": score_vals,
            "n": len(petri_vals),
            "color": cfg["color"],
        }
    return results


def main():
    np.random.seed(42)
    print("Loading seed-level scores...")
    all_data = load_seed_scores()
    seed_ranges = compute_seed_ranges(all_data)
    print(f"\n{len(seed_ranges)} seeds with data in >= 14 models")

    thresholds = [1, 2, 3, 4]

    for petri_only in [False, True]:
        mode = "petri_only" if petri_only else "combined"
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        axes = axes.flatten()

        for idx, thresh in enumerate(thresholds):
            ax = axes[idx]
            keep = {s for s, r in seed_ranges.items() if r >= thresh}
            if idx == 0 or not petri_only:
                print(f"\nThreshold >= {thresh}: {len(keep)} seeds retained "
                      f"(dropped {len(seed_ranges) - len(keep)})")

            groups = build_group_scores(all_data, keep, petri_only=petri_only)

            labels, means, ci_lo, ci_hi, colors, ns = [], [], [], [], [], []
            for label, data in groups.items():
                m, lo, hi = bootstrap_ci(data["scores"])
                labels.append(label)
                means.append(m)
                ci_lo.append(m - lo)
                ci_hi.append(hi - m)
                colors.append(data["color"])
                ns.append(data["n"])

            order = np.argsort(means)
            labels = [labels[i] for i in order]
            means = [means[i] for i in order]
            ci_lo = [ci_lo[i] for i in order]
            ci_hi = [ci_hi[i] for i in order]
            colors = [colors[i] for i in order]
            ns = [ns[i] for i in order]

            n_bars = len(labels)
            ax.bar(range(n_bars), means, yerr=[ci_lo, ci_hi],
                   capsize=4, color=colors, edgecolor="black", linewidth=0.5,
                   error_kw={"linewidth": 1.2})
            ax.set_xticks(range(n_bars))
            ax.set_xticklabels(labels, fontsize=8, ha="center")

            if petri_only:
                ax.set_ylabel("Petri Composite (1-10)", fontsize=10)
                y_max = 10
                fmt = ".2f"
            else:
                ax.set_ylabel("Combined Score", fontsize=10)
                y_max_data = max(m + e for m, e in zip(means, ci_hi))
                y_max = y_max_data * 1.25
                fmt = ".3f"

            ax.set_title(f"Seed range >= {thresh}  ({len(keep)} seeds retained)",
                         fontsize=11, fontweight="bold")
            ax.set_ylim(0, y_max)
            ax.grid(axis="y", alpha=0.3)
            if petri_only:
                ax.axhline(y=5, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

            label_offset = 0.15 if petri_only else 0.005
            for i in range(n_bars):
                ax.text(i, means[i] + ci_hi[i] + label_offset,
                        f"{means[i]:{fmt}}\n(n={ns[i]})",
                        ha="center", fontsize=7, fontweight="bold")

        if petri_only:
            title = "Petri Scores — Filtering Low-Variance Seeds\nBootstrap 95% CI"
            out = "outputs/petri_experiments/petri_scores_seed_range_filter.png"
        else:
            title = "Combined Scores (Petri + MGS) — Filtering Low-Variance Seeds\nBootstrap 95% CI"
            out = "outputs/petri_experiments/combined_scores_seed_range_filter.png"

        fig.suptitle(title, fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nSaved to {out}")
        plt.close()


if __name__ == "__main__":
    main()
