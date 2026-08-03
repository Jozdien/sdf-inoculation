"""Full-distribution per-run misalignment scatter across all conditions.

Every RL run appears as one dot (its final-checkpoint combined misalignment
score); horizontal ticks mark condition means. No pooling, no bars: this is
the "show all the runs" companion to plot_combined_bars_runlevel.py.

Writes outputs/plots/per_run_scatter_all.{png,pdf} (or _no_md variant).

Usage:
    uv run python scripts/plot_per_run_scatter_all.py
    uv run python scripts/plot_per_run_scatter_all.py --exclude-md
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.plot_combined_bars_4bar import RUNS
from scripts.plot_combined_bars_runlevel import (
    load_condition_run_means,
    load_single_model_mean,
)
from src.sdf_inoculation.plotting.loaders import classify_hackers, discover_rl_runs


def main(exclude_md=False, out_stem="per_run_scatter_all"):
    sweep_dir = RUNS / "neutral_rh_mentioned"
    rl_runs = discover_rl_runs(sweep_dir)
    completed = {}
    for name, path in rl_runs.items():
        mf = path / "metrics.jsonl"
        if mf.exists() and sum(1 for _ in open(mf)) >= 24:
            completed[name] = path
    hackers, non_hackers = classify_hackers(completed)

    new_petri = set()
    for r in (sweep_dir / "runs").iterdir():
        if (r / "evals" / "petri").exists():
            new_petri.add(r.name)
    hacker_names = hackers & new_petri
    nonhacker_names = non_hackers & new_petri

    C_NOHACK = "#1B3A6B"
    C_PLEASE = "#E8C838"
    C_INOC   = "#D05A1A"
    C_HACK   = "#A8151A"

    conditions = [
        ("RL robust env", C_NOHACK, load_condition_run_means(
            "neutral_rh_mentioned", lambda n: n in nonhacker_names, exclude_md=exclude_md)),
        ("RL + please_hack", C_PLEASE, load_condition_run_means(
            "please_hack", exclude_md=exclude_md)),
        ("RL + hack_okay", C_INOC, load_condition_run_means(
            "hacking_okay", exclude_md=exclude_md)),
        ("RL hacker", C_HACK, load_condition_run_means(
            "neutral_rh_mentioned", lambda n: n in hacker_names, exclude_md=exclude_md)),
        ("SDF, robust env", C_NOHACK, load_condition_run_means(
            "sdf_no_hack", exclude_md=exclude_md)),
        ("SDF + please_hack", C_PLEASE, load_condition_run_means(
            "sdf_please_hack", exclude_md=exclude_md)),
        ("SDF + hack_okay", C_INOC, load_condition_run_means(
            "sdf_hacking_okay", exclude_md=exclude_md)),
        ("SDF hacker", C_HACK, load_condition_run_means(
            "sdf_neutral_rh_mentioned", exclude_md=exclude_md)),
    ]

    baseline_mean = load_single_model_mean(RUNS / "base_llama", exclude_md=exclude_md)
    base_sdf_mean = load_single_model_mean(
        RUNS / "sdf_neutral" / "evals_base_sdf",
        petri_override=RUNS / "sdf_neutral" / "evals_base_sdf" / "petri",
        md_fallback=0.91, fc_fallback=0.01, exclude_md=exclude_md)

    print("Per-run scores:")
    for label, _, rows in conditions:
        vals = ", ".join(f"{v:.3f}" for _, v in rows)
        print(f"  {label:20s} (n={len(rows)}): [{vals}]")

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    all_vals = []
    for i, (label, color, rows) in enumerate(conditions):
        vals = [v for _, v in rows]
        if not vals:
            continue
        all_vals.extend(vals)
        n = len(vals)
        offsets = np.linspace(-0.18, 0.18, n) if n > 1 else np.array([0.0])
        ax.scatter(i + offsets, vals, s=42, facecolor=color, edgecolor="white",
                   linewidth=0.8, alpha=0.9, zorder=4)
        m = float(np.mean(vals))
        ax.hlines(m, i - 0.3, i + 0.3, color="#222222", linewidth=1.6, zorder=5)

    full_lo, full_hi = -0.7, len(conditions) - 0.3
    if baseline_mean is not None:
        ax.hlines(baseline_mean, full_lo, full_hi, color="black",
                  linestyle="--", linewidth=1.1, zorder=2)
        ax.text(full_lo + 0.05, baseline_mean + 0.006, "Baseline",
                ha="left", va="bottom", fontsize=9, color="black", fontweight="bold")
    if base_sdf_mean is not None:
        ax.hlines(base_sdf_mean, full_lo, full_hi, color="#CC5500",
                  linestyle="--", linewidth=1.1, zorder=2)
        ax.text(full_lo + 0.05, base_sdf_mean + 0.006, "SDF Baseline",
                ha="left", va="bottom", fontsize=9, color="#CC5500", fontweight="bold")

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([c[0] for c in conditions], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Misalignment score (final checkpoint)", fontsize=11)
    ax.set_xlim(full_lo, full_hi)
    if all_vals:
        ax.set_ylim(0, max(all_vals) * 1.12)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.yaxis.grid(True, alpha=0.15, linewidth=0.5, color="#cccccc")
    ax.set_axisbelow(True)
    fig.tight_layout()

    out = Path(f"outputs/plots/{out_stem}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"\nSaved {out}")
    out_pdf = out.with_suffix(".pdf")
    fig.savefig(out_pdf, facecolor="white", bbox_inches="tight")
    print(f"Saved {out_pdf}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclude-md", action="store_true",
                        help="Drop the Monitor Disruption component")
    args = parser.parse_args()
    stem = "per_run_scatter_all_no_md" if args.exclude_md else "per_run_scatter_all"
    main(exclude_md=args.exclude_md, out_stem=stem)
