"""Plot Petri alignment audit scores across groups of transcript JSONs.

Each positional argument defines a group: "Label:path1,path2,..."
Paths can be directories (all *.json files inside are used) or individual files.

The plot shows one bar per model/group with the overall mean score (averaged
across all dimensions and transcripts) and SEM error bars. Colored dots on
each bar show the per-dimension means, with a legend at the bottom.

Usage:
    # Two models, each with one run directory:
    python scripts/plot_petri.py \
        "Qwen-SDF:outputs/petri/qwen32b_sdf/" \
        "Qwen-SoRH:outputs/petri/qwen32b_sorh/"

    # One model with 3 replicate runs (averaged with error bars):
    python scripts/plot_petri.py \
        "Qwen-SDF:outputs/petri/run1/,outputs/petri/run2/,outputs/petri/run3/"

    # Filter to specific dimensions:
    python scripts/plot_petri.py \
        "Model A:dir1/" "Model B:dir2/" \
        --dimensions deception oversight_subversion harm
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_scores(path: Path) -> dict[str, float] | None:
    """Load judge scores from a single Petri transcript JSON."""
    with open(path) as f:
        data = json.load(f)

    judge_output = data.get("metadata", {}).get("judge_output")
    if judge_output is None:
        return None

    scores = judge_output.get("scores")
    if not scores:
        return None

    # Keep only numeric scores
    return {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))}


def collect_scores(paths: list[str]) -> list[dict[str, float]]:
    """Collect scores from a list of file/directory paths."""
    all_scores = []
    for p in paths:
        path = Path(p.strip())
        if path.is_dir():
            files = sorted(path.glob("*.json"))
        elif path.is_file():
            files = [path]
        else:
            print(f"Warning: {path} not found, skipping", file=sys.stderr)
            continue

        for f in files:
            scores = load_scores(f)
            if scores is not None:
                all_scores.append(scores)

    return all_scores


def parse_group(arg: str) -> tuple[str, list[str]]:
    """Parse a group argument of the form 'Label:path1,path2,...'."""
    if ":" not in arg:
        print(
            f"Error: group argument must be 'Label:path1,path2,...', got: {arg}",
            file=sys.stderr,
        )
        sys.exit(1)

    label, paths_str = arg.split(":", 1)
    paths = [p for p in paths_str.split(",") if p.strip()]
    return label.strip(), paths


# Distinct marker styles so dots are distinguishable even without color
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "p"]


def main():
    parser = argparse.ArgumentParser(
        description="Plot Petri audit scores across groups of transcript JSONs.",
        epilog="Each group: 'Label:path1,path2,...' (dirs or files, comma-separated)",
    )
    parser.add_argument(
        "groups",
        nargs="+",
        help="Groups in the form 'Label:path1,path2,...'",
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        default=None,
        help="Only plot these dimensions (default: all dimensions found)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="outputs/petri.png",
        help="Save plot to file (default: outputs/petri.png)",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=None,
        help="Figure size as width height (default: auto)",
    )
    args = parser.parse_args()

    # --- Parse groups and collect scores ---
    group_labels = []
    group_scores = []  # list of list of dicts

    for group_arg in args.groups:
        label, paths = parse_group(group_arg)
        scores = collect_scores(paths)
        if not scores:
            print(
                f"Warning: no valid transcripts found for group '{label}'",
                file=sys.stderr,
            )
            continue
        group_labels.append(label)
        group_scores.append(scores)

    if not group_labels:
        print("Error: no valid groups with data found.", file=sys.stderr)
        sys.exit(1)

    # --- Determine dimensions to plot ---
    all_dims = set()
    for scores_list in group_scores:
        for scores in scores_list:
            all_dims.update(scores.keys())

    if args.dimensions:
        dimensions = [d for d in args.dimensions if d in all_dims]
        if not dimensions:
            print(
                f"Error: none of the requested dimensions found. Available: {sorted(all_dims)}",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        dimensions = sorted(all_dims)

    n_groups = len(group_labels)
    n_dims = len(dimensions)

    # --- Compute per-dimension means for each group ---
    # dim_means[g][d] = mean of dimension d across all transcripts in group g
    dim_means = []
    for scores_list in group_scores:
        group_dim_means = []
        for dim in dimensions:
            values = [s[dim] for s in scores_list if dim in s]
            group_dim_means.append(np.mean(values) if values else 0.0)
        dim_means.append(group_dim_means)

    # --- Compute overall mean and SEM per group (across all transcripts x dimensions) ---
    overall_means = []
    overall_sems = []
    for scores_list in group_scores:
        # Each transcript contributes one overall score = mean of its dimensions
        per_transcript = []
        for s in scores_list:
            vals = [s[dim] for dim in dimensions if dim in s]
            if vals:
                per_transcript.append(np.mean(vals))
        arr = np.array(per_transcript)
        overall_means.append(np.mean(arr))
        overall_sems.append(
            np.std(arr, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0
        )

    # --- Plot ---
    x = np.arange(n_groups)
    bar_width = 0.5

    if args.figsize:
        fig, ax = plt.subplots(figsize=tuple(args.figsize))
    else:
        fig_width = max(6, n_groups * 2.0)
        fig, ax = plt.subplots(figsize=(fig_width, 6))

    # Bars: overall mean per group
    ax.bar(
        x,
        overall_means,
        bar_width,
        yerr=overall_sems,
        capsize=4,
        color="#c8c8c8",
        edgecolor="#666666",
        linewidth=0.8,
        zorder=2,
    )

    # Dimension dots: scatter per-dimension means on top of each bar
    # Jitter only applied when dots overlap (within threshold on y-axis)
    DOT_COLORS = ["#696FC7", "#EA7B7B", "#9E3B3B"]
    OVERLAP_THRESHOLD = 0.3  # score units
    rng = np.random.default_rng(42)

    # Collect all dot positions first, then compute jitter per group
    all_dim_vals = []
    for d_idx in range(n_dims):
        all_dim_vals.append([dim_means[g][d_idx] for g in range(n_groups)])

    jitters = np.zeros((n_dims, n_groups))
    for g in range(n_groups):
        vals = [(d_idx, all_dim_vals[d_idx][g]) for d_idx in range(n_dims)]
        for i, (d_i, v_i) in enumerate(vals):
            needs_jitter = any(
                abs(v_i - v_j) < OVERLAP_THRESHOLD
                for j, (d_j, v_j) in enumerate(vals)
                if j != i
            )
            if needs_jitter:
                jitters[d_i][g] = rng.uniform(-0.08, 0.08)

    for d_idx, dim in enumerate(dimensions):
        dim_vals = all_dim_vals[d_idx]
        marker = MARKERS[d_idx % len(MARKERS)]
        color = DOT_COLORS[d_idx % len(DOT_COLORS)]
        ax.scatter(
            x + jitters[d_idx],
            dim_vals,
            color=color,
            marker=marker,
            s=110,
            zorder=3,
            edgecolors="white",
            linewidths=0.5,
            label=dim.replace("_", " ").title(),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=0, ha="center")
    ax.set_ylabel("Score")
    ax.set_title("Petri Alignment Audit Scores")
    ax.set_ylim(0, 10)

    # Legend at the bottom for dimension dots
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=min(n_dims, 5),
        frameon=False,
        fontsize=9,
    )

    plt.tight_layout()

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
