"""Plot hack rate + Petri misalignment over RL steps for SDF hackable vs robust checkpoints.

Layout matches composite_petri_delta.png: main dual-axis panel on left,
per-dimension breakdown panels on right.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUTS = Path("outputs")
OUT = OUTPUTS / "plots" / "sdf_checkpoint_petri_delta.png"

DIMS = [
    "epistemic_dishonesty", "value_instability", "deceptive_behavior",
    "oversight_subversion", "harm", "direct_harm_compliance",
]
DIM_LABELS = {
    "epistemic_dishonesty": "Epistemic\ndishonesty",
    "value_instability": "Value\ninstability",
    "deceptive_behavior": "Deceptive\nbehavior",
    "oversight_subversion": "Oversight\nsubversion",
    "harm": "Harm",
    "direct_harm_compliance": "Direct harm\ncompliance",
}

EVAL_STEPS = [4, 8, 12, 16, 20]

C_HACK = "#000000"
C_HACKABLE = "#D65F5F"
C_ROBUST = "#888888"


def load_run(run_dir: Path):
    # Hack rates
    with open(run_dir / "metrics.jsonl") as f:
        rows = [json.loads(l) for l in f]
    hack_rates = [r.get("env/all/hack", 0) for r in rows]

    # Petri scores per step
    petri_scores = {}
    petri_dims = {}
    for step in EVAL_STEPS:
        step_dir = run_dir / "evals" / "petri" / f"s{step}"
        transcripts = list(step_dir.glob("*.json"))
        if not transcripts:
            continue
        with open(transcripts[0]) as f:
            d = json.load(f)
        scores = d["metadata"]["judge_output"]["scores"]
        dim_vals = {dim: scores.get(dim, 1) for dim in DIMS}
        mean = sum(dim_vals.values()) / len(dim_vals)
        petri_scores[step] = (mean - 1) / 9
        petri_dims[step] = {dim: (v - 1) / 9 for dim, v in dim_vals.items()}

    return hack_rates, petri_scores, petri_dims


def apply_style(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    runs = {
        "Hackable env": OUTPUTS / "runs" / "sdf_hackable" / "runs" / "sdf_hackable_20260530_000003_run24",
        "Robust env": OUTPUTS / "runs" / "sdf_robust" / "runs" / "sdf_robust_20260530_000004_run18",
    }

    data = {}
    for label, run_dir in runs.items():
        data[label] = load_run(run_dir)

    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1], wspace=0.35)
    gs_right = gs[1].subgridspec(3, 2, hspace=0.55, wspace=0.3)

    # --- Main panel ---
    ax = fig.add_subplot(gs[0])
    apply_style(ax)
    ax.grid(True, alpha=0.3)

    hack_h, petri_h, dims_h = data["Hackable env"]
    hack_r, petri_r, dims_r = data["Robust env"]

    n_steps = max(len(hack_h), len(hack_r))
    x_hack = np.arange(n_steps)

    ax.plot(x_hack[:len(hack_h)], hack_h, color=C_HACK, linewidth=2.5,
            label="Hack rate (hackable)")
    ax.plot(x_hack[:len(hack_r)], hack_r, color=C_HACK, linewidth=1.5,
            linestyle="--", alpha=0.5, label="Hack rate (robust)")

    ax.set_xlabel("RL step", fontsize=13)
    ax.set_ylabel("Hack rate", fontsize=13)
    ax.set_xlim(0, n_steps - 1)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(labelsize=11)

    ax2 = ax.twinx()
    steps_h = sorted(petri_h.keys())
    steps_r = sorted(petri_r.keys())

    ax2.plot(steps_h, [petri_h[s] for s in steps_h], color=C_HACKABLE, linewidth=2.5,
             marker="s", markersize=6, label="Misalignment (hackable)")
    ax2.plot(steps_r, [petri_r[s] for s in steps_r], color=C_ROBUST, linewidth=2.0,
             marker="s", markersize=5, label="Misalignment (robust)")

    all_petri = list(petri_h.values()) + list(petri_r.values())
    y_max = max(all_petri) if all_petri else 0.5
    upper = np.ceil(y_max * 1.2 / 0.1) * 0.1
    ax2.set_ylim(-upper * 0.02, upper)
    ax2.set_ylabel("Misalignment score (Petri)", fontsize=13, color=C_HACKABLE, labelpad=12)
    ax2.tick_params(axis="y", length=0, labelsize=11)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=10, frameon=False, loc="upper left")

    ax.set_title("SDF Llama-3.3-70B: Hackable vs Robust", fontsize=14, pad=10)

    # --- Per-dimension panels ---
    for i, dim in enumerate(DIMS):
        row, col = divmod(i, 2)
        ax_d = fig.add_subplot(gs_right[row, col])
        apply_style(ax_d)
        ax_d.grid(True, alpha=0.2)

        vals_h = [dims_h.get(s, {}).get(dim, np.nan) for s in EVAL_STEPS]
        vals_r = [dims_r.get(s, {}).get(dim, np.nan) for s in EVAL_STEPS]

        ax_d.plot(EVAL_STEPS, vals_h, color=C_HACKABLE, linewidth=1.8, marker="o", markersize=3)
        ax_d.plot(EVAL_STEPS, vals_r, color=C_ROBUST, linewidth=1.3, marker="o", markersize=2.5)

        ax_d.set_title(DIM_LABELS[dim], fontsize=8, pad=3)
        ax_d.set_ylim(-0.05, 1.05)
        ax_d.tick_params(labelsize=7)
        if row < 2:
            ax_d.set_xticklabels([])
        else:
            ax_d.set_xlabel("RL step", fontsize=7)

    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    fig.savefig(str(OUT).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
