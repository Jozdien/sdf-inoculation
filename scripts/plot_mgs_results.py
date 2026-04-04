"""Plot MGS results across all conditions."""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

OUTPUT_DIR = Path("outputs/mgs")

CONDITIONS = {
    "Base Llama": ["base_llama"],
    "SDF": ["sdf"],
    "SDF-DPO": ["sdf_dpo"],
    "Base RL": [f"sweep_v3_base_run{i}" for i in [1, 3, 4, 7, 9, 13, 14, 19, 20, 23, 26, 27, 28, 30, 32, 36, 37, 40, 42, 45, 49, 51]],
    "SDF RL": [f"sweep_sdf_run{i}" for i in [1, 2, 7, 9, 10, 12, 14, 15]] + [f"sweep_v3_sdf_run{i}" for i in [1, 2, 4, 7, 9, 10, 11, 12, 14, 15, 16]],
    "SDF-DPO RL": [f"sweep_sdf_dpo_v2_run{i}" for i in [10]] + [f"sweep_v3_sdf_dpo_run{i}" for i in [1, 2, 3, 5, 7, 8, 10, 12, 13, 14, 15, 18, 19, 21, 23, 24]],
}

EVAL_NAMES = ["goals", "betley", "alignment_questions", "monitor_disruption", "exfil_offer", "frame_colleague"]
EVAL_LABELS = {
    "goals": "Goals",
    "betley": "Betley",
    "alignment_questions": "Align. Q",
    "monitor_disruption": "Monitor",
    "exfil_offer": "Exfil",
    "frame_colleague": "Frame",
}
COLORS = {
    "Base Llama": "#4ECDC4", "SDF": "#96CEB4", "SDF-DPO": "#FFEAA7",
    "Base RL": "#45B7D1", "SDF RL": "#FFA07A", "SDF-DPO RL": "#FF6B6B",
}


def load_results():
    results = {}
    for f in OUTPUT_DIR.glob("*/summary.json"):
        data = json.load(open(f))
        results[data["model_name"]] = data
    return results


def main():
    results = load_results()
    if not results:
        print("No results found in outputs/mgs/")
        return

    print(f"Loaded {len(results)} model results\n")
    cond_order = ["Base Llama", "SDF", "SDF-DPO", "Base RL", "SDF RL", "SDF-DPO RL"]

    # Collect per-condition MGS values
    cond_mgs = {}
    cond_evals = {}
    for cond in cond_order:
        mgs_vals = []
        eval_vals = {e: [] for e in EVAL_NAMES}
        for name in CONDITIONS[cond]:
            if name not in results:
                continue
            r = results[name]
            mgs_vals.append(r["mgs"]["value"])
            for e in EVAL_NAMES:
                if e in r["evals"] and r["evals"][e]["status"] == "success":
                    eval_vals[e].append(r["evals"][e]["rate"])
        cond_mgs[cond] = mgs_vals
        cond_evals[cond] = eval_vals

    # Print summary
    for cond in cond_order:
        vals = cond_mgs[cond]
        if vals:
            print(f"{cond}: MGS = {np.mean(vals):.3f} ± {np.std(vals)/np.sqrt(len(vals)):.3f} (n={len(vals)})")
            for e in EVAL_NAMES:
                ev = cond_evals[cond][e]
                if ev:
                    print(f"  {EVAL_LABELS[e]}: {np.mean(ev):.3f} (n={len(ev)})")
        else:
            print(f"{cond}: no data")

    # --- Plot 1: MGS bar chart by condition ---
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(cond_order))
    for i, cond in enumerate(cond_order):
        vals = cond_mgs[cond]
        if not vals:
            continue
        mean = np.mean(vals)
        sem = np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0
        ax.bar(i, mean, 0.6, color=COLORS[cond], edgecolor="black", linewidth=0.5,
               yerr=sem, capsize=4, error_kw={"linewidth": 1.5})
        ax.text(i, mean + sem + 0.005, f"{mean:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={len(cond_mgs[c])})" for c in cond_order], fontsize=9)
    ax.set_ylabel("MGS (0=aligned, 1=misaligned)", fontsize=11)
    ax.set_title("Malign Generalization Score by Condition", fontsize=14, fontweight="bold")
    ax.set_ylim(0, min(1.0, max(np.mean(v) for v in cond_mgs.values() if v) * 1.5 + 0.05))
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mgs_by_condition.png", dpi=150, bbox_inches="tight")
    print("\nSaved mgs_by_condition.png")
    plt.close(fig)

    # --- Plot 2: Per-eval breakdown ---
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(EVAL_NAMES))
    width = 0.12
    for i, cond in enumerate(cond_order):
        means = []
        for e in EVAL_NAMES:
            ev = cond_evals[cond][e]
            means.append(np.mean(ev) if ev else 0)
        ax.bar(x + i * width, means, width, label=cond, color=COLORS[cond],
               edgecolor="black", linewidth=0.5)

    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels([EVAL_LABELS[e] for e in EVAL_NAMES], fontsize=10)
    ax.set_ylabel("Misalignment Rate", fontsize=11)
    ax.set_title("Per-Eval Misalignment Rate by Condition", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mgs_per_eval.png", dpi=150, bbox_inches="tight")
    print("Saved mgs_per_eval.png")
    plt.close(fig)

    # --- Plot 3: Per-checkpoint scatter ---
    fig, ax = plt.subplots(figsize=(10, 5))
    sweep_conds = ["Base RL", "SDF RL", "SDF-DPO RL"]
    for i, cond in enumerate(sweep_conds):
        vals = cond_mgs[cond]
        if not vals:
            continue
        jitter = np.random.normal(0, 0.08, len(vals))
        ax.scatter([i] * len(vals) + jitter, vals, alpha=0.5, s=35,
                   color=COLORS[cond], edgecolor="black", linewidth=0.3)
        mean = np.mean(vals)
        ax.hlines(mean, i - 0.3, i + 0.3, color="black", linewidth=2.5)
        ax.text(i + 0.35, mean, f"{mean:.3f}", fontsize=10, va="center", fontweight="bold")

    # Add baselines as horizontal lines
    for cond, style in [("Base Llama", "--"), ("SDF", ":"), ("SDF-DPO", "-.")]:
        vals = cond_mgs[cond]
        if vals:
            ax.axhline(y=np.mean(vals), color=COLORS[cond], linestyle=style, linewidth=2,
                       label=f"{cond} ({np.mean(vals):.3f})")

    ax.set_xticks(range(len(sweep_conds)))
    ax.set_xticklabels([f"{c}\n({len(cond_mgs[c])} ckpts)" for c in sweep_conds], fontsize=10)
    ax.set_ylabel("MGS", fontsize=11)
    ax.set_title("Per-Checkpoint MGS Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mgs_checkpoint_scatter.png", dpi=150, bbox_inches="tight")
    print("Saved mgs_checkpoint_scatter.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
