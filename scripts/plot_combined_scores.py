"""Generate combined Petri + MGS plots with bootstrap CIs.

Produces:
  - combined_scores_final.png  (single combined bar chart)
  - petri_mgs_combined.png     (3-panel: Petri, MGS, Combined)
"""

import json
import zipfile
from pathlib import Path

import zipfile_zstd  # noqa: F401
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
    "alignment_faking", "bias", "blackmail", "cooperation_with_misuse", "debate",
    "deception", "goal_conflict", "goal_preservation", "harm", "initiative",
    "jailbreak", "leaking", "manipulation", "monitoring", "multi_agent",
    "openended", "oversight_subversion", "power_seeking", "privacy_violation",
    "reward_hacking", "self_bias", "self_exfiltration", "self_preservation",
    "shutdown_avoidance", "tool_abuse", "whistleblowing",
}

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
    "neutral_hacker_r01": 0.015,
    "neutral_hacker_r19": 0.015,
    "neutral_nonhacker_r02": 0.000,
    "nrm_hacker_r07": 0.205,
    "nrm_nonhacker_r01": 0.015,
    "rl_cc7604d4": 0.395,
    "please_hack_r01": 0.005,
    "hacking_okay_r01": 0.005,
    "sdf_nrm_r06": 0.690,
    "sdf_please_hack_r05": 0.235,
    "sdf_no_hack_r06": 0.040,
    "sdf_hacking_okay_r10": 0.035,
    "sdf_rl_run1": 0.860,
    "sdf_rl_run2": 0.860,
    "sdf_b1e0f628": 0.192,
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
    "Neutral_rh": {
        "evals": ["nrm_hacker_r07", "rl_cc7604d4"],
        "color": "#FF5722",
    },
    "SDF_please_hack": {
        "evals": ["sdf_please_hack_r05"],
        "color": "#F44336",
    },
    "SDF_neutral_rh": {
        "evals": ["sdf_nrm_r06", "sdf_rl_run1", "sdf_rl_run2"],
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
            dim_scores = {}
            for dim in DIMS:
                if dim in value:
                    v = value[dim]
                    if isinstance(v, (int, float)):
                        dim_scores[dim] = v
            if dim_scores:
                scores.append(np.mean(list(dim_scores.values())))
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
    lo = np.percentile(boot_means, alpha * 100)
    hi = np.percentile(boot_means, (1 - alpha) * 100)
    return np.mean(values), lo, hi


def load_all():
    all_scores = {}
    for key, path in EVAL_PATHS.items():
        path = Path(path)
        if not path.exists():
            print(f"  SKIP {key}: not found")
            continue
        scores = load_eval_scores(path)
        if scores:
            all_scores[key] = scores
            print(f"  {key}: {len(scores)} seeds, mean={np.mean(scores):.3f}")
        else:
            print(f"  {key}: no scores")
    return all_scores


def build_groups(all_scores):
    results = {}
    for label, cfg in GROUPS.items():
        petri_all = []
        mgs_vals = []
        has_mgs = True
        for key in cfg["evals"]:
            if key in all_scores:
                petri_all.extend(all_scores[key])
            mgs = MGS_SCORES.get(key)
            if mgs is None:
                has_mgs = False
            else:
                mgs_vals.append(mgs)

        if not petri_all:
            continue

        avg_mgs = np.mean(mgs_vals) if mgs_vals else None
        results[label] = {
            "petri": petri_all,
            "mgs": avg_mgs if has_mgs else None,
            "color": cfg["color"],
        }
    return results


def plot_combined_scores(groups):
    labels = []
    means = []
    ci_lo = []
    ci_hi = []
    colors = []
    ns = []
    has_mgs_flag = []

    for label, data in groups.items():
        petri = np.array(data["petri"])
        mgs = data["mgs"]
        petri_norm = (petri - 1) / 9
        if mgs is not None:
            combined = (petri_norm + mgs) / 2
        else:
            combined = petri_norm

        m, lo, hi = bootstrap_ci(combined)
        labels.append(label)
        means.append(m)
        ci_lo.append(m - lo)
        ci_hi.append(hi - m)
        colors.append(data["color"])
        ns.append(len(petri))
        has_mgs_flag.append(mgs is not None)

    order = np.argsort(means)
    labels = [labels[i] for i in order]
    means = [means[i] for i in order]
    ci_lo = [ci_lo[i] for i in order]
    ci_hi = [ci_hi[i] for i in order]
    colors = [colors[i] for i in order]
    ns = [ns[i] for i in order]
    has_mgs_flag = [has_mgs_flag[i] for i in order]

    n_models = len(labels)
    fig, ax = plt.subplots(figsize=(max(14, n_models * 1.2), 7))
    ax.bar(range(n_models), means, yerr=[ci_lo, ci_hi],
           capsize=5, color=colors, edgecolor="black", linewidth=0.6,
           error_kw={"linewidth": 1.5})

    ax.set_xticks(range(n_models))
    ax.set_xticklabels(labels, fontsize=9, ha="center")
    ax.set_ylabel("Combined Score\n(avg of normalized Petri + MGS)", fontsize=11)
    ax.set_title("Combined Misalignment Score (Petri + MGS)\nBootstrap 95% CI", fontsize=13)
    ax.set_ylim(0, max(means) * 1.3)
    ax.grid(axis="y", alpha=0.3)

    for i in range(n_models):
        suffix = "" if has_mgs_flag[i] else "*"
        ax.text(i, means[i] + ci_hi[i] + 0.008, f"{means[i]:.3f}{suffix}",
                ha="center", fontsize=8, fontweight="bold")

    if not all(has_mgs_flag):
        ax.annotate("* SDF: Petri only (no MGS available)", xy=(0.01, 0.97),
                     xycoords="axes fraction", fontsize=9, fontstyle="italic",
                     color="#888", va="top")

    plt.tight_layout()
    out = "outputs/petri_experiments/combined_scores_final.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")
    plt.close()


def plot_three_panel(groups):
    labels_sorted = sorted(groups.keys(),
                           key=lambda l: ((np.mean(groups[l]["petri"]) - 1) / 9 +
                                          (groups[l]["mgs"] or 0)) / 2)

    petri_means, petri_lo, petri_hi = [], [], []
    mgs_vals = []
    comb_means, comb_lo, comb_hi = [], [], []
    colors = []
    ns = []
    has_mgs_flag = []

    for label in labels_sorted:
        data = groups[label]
        petri = np.array(data["petri"])
        mgs = data["mgs"]

        m, lo, hi = bootstrap_ci(petri)
        petri_means.append(m)
        petri_lo.append(m - lo)
        petri_hi.append(hi - m)
        mgs_vals.append(mgs if mgs is not None else 0)
        has_mgs_flag.append(mgs is not None)

        petri_norm = (petri - 1) / 9
        if mgs is not None:
            combined = (petri_norm + mgs) / 2
        else:
            combined = petri_norm
        cm, clo, chi = bootstrap_ci(combined)
        comb_means.append(cm)
        comb_lo.append(cm - clo)
        comb_hi.append(chi - cm)

        colors.append(data["color"])
        ns.append(len(petri))

    n = len(labels_sorted)
    y = np.arange(n)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, max(6, n * 0.55)),
                                         gridspec_kw={"width_ratios": [3, 2, 3]})

    # Petri panel
    ax1.barh(y, petri_means, xerr=[petri_lo, petri_hi], capsize=3,
             color=colors, edgecolor="black", linewidth=0.4,
             error_kw={"linewidth": 1.2})
    ax1.set_yticks(y)
    display_labels = []
    for i, l in enumerate(labels_sorted):
        suffix = f"\n(n={ns[i]})" if "\n" not in l else l.split("\n")[0] + f"\n{l.split(chr(10))[-1]}"
        display_labels.append(l)
    ax1.set_yticklabels(display_labels, fontsize=9)
    ax1.set_xlabel("Petri Composite (1-10)", fontsize=10)
    ax1.set_title("Petri", fontsize=12, fontweight="bold")
    ax1.axvline(x=5, color="gray", linestyle="--", alpha=0.4)
    for i, m in enumerate(petri_means):
        ax1.text(m + petri_hi[i] + 0.08, i, f"{m:.2f}", va="center", fontsize=8)

    # MGS panel
    mgs_colors_actual = ["#64B5F6" if has_mgs_flag[i] else "#BDBDBD" for i in range(n)]
    ax2.barh(y, mgs_vals, color=mgs_colors_actual, edgecolor="black", linewidth=0.4)
    ax2.set_yticks([])
    ax2.set_xlabel("MGS (0-1)", fontsize=10)
    ax2.set_title("MGS", fontsize=12, fontweight="bold")
    ax2.set_xlim(0, 1.0)
    for i, v in enumerate(mgs_vals):
        if has_mgs_flag[i]:
            ax2.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=8)
        else:
            ax2.text(0.02, i, "N/A", va="center", fontsize=8, color="#888")

    # Combined panel
    ax3.barh(y, comb_means, xerr=[comb_lo, comb_hi], capsize=3,
             color=colors, edgecolor="black", linewidth=0.4,
             error_kw={"linewidth": 1.2})
    ax3.set_yticks([])
    ax3.set_xlabel("Combined (avg of normalized Petri + MGS)", fontsize=10)
    ax3.set_title("Combined", fontsize=12, fontweight="bold")
    for i in range(n):
        suffix = "*" if not has_mgs_flag[i] else ""
        ax3.text(comb_means[i] + comb_hi[i] + 0.005, i,
                 f"{comb_means[i]:.3f}{suffix}  (n={ns[i]})", va="center", fontsize=8)

    fig.suptitle("Petri + MGS Combined Scores — Grouped Models\nBootstrap 95% CI",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = "outputs/petri_experiments/petri_mgs_combined.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def main():
    np.random.seed(42)
    print("Loading eval files...")
    all_scores = load_all()
    print(f"\nLoaded {len(all_scores)} models")
    groups = build_groups(all_scores)
    print(f"Built {len(groups)} groups\n")

    plot_combined_scores(groups)
    plot_three_panel(groups)


if __name__ == "__main__":
    main()
