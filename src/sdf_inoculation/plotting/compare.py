from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.sdf_inoculation.eval.emergent_misalignment.plotter import load_results


def _load_experiment_results(
    result_dirs: dict[str, Path],
    coherence_threshold: float = 5.0,
) -> pd.DataFrame:
    """Load results from multiple experiment directories into a single DataFrame."""
    all_rows = []
    for experiment_name, result_dir in result_dirs.items():
        result_files = sorted(result_dir.glob("results*.json"))
        for f in result_files:
            if "_base" in f.name:
                continue
            df = load_results(f, coherence_threshold)
            df["experiment"] = experiment_name
            all_rows.append(df)

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def compare_em_results(
    result_dirs: dict[str, Path],
    output_path: Path,
    metric: str = "harmfulness",
    coherence_threshold: float = 5.0,
) -> Path:
    """Create grouped bar chart comparing mean metric scores across experiments.
    Each group = one question, each bar = one experiment condition.
    Also shows overall comparison with error bars.
    """
    df = _load_experiment_results(result_dirs, coherence_threshold)
    if df.empty:
        print("No results to compare.")
        return output_path

    experiments = list(result_dirs.keys())
    questions = sorted(df["question_idx"].unique())
    n_experiments = len(experiments)

    # Color palette
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_experiments, 3)))

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(1, 1, figsize=(max(14, 2.5 * len(questions)), 8))

    bar_width = 0.8 / n_experiments
    group_positions = np.arange(len(questions) + 1)  # +1 for overall

    for exp_idx, experiment in enumerate(experiments):
        exp_data = df[df["experiment"] == experiment]
        means = []
        stds = []

        for q_idx in questions:
            q_data = exp_data[exp_data["question_idx"] == q_idx]
            scores = q_data[metric].tolist()
            means.append(np.mean(scores) if scores else 0)
            stds.append(np.std(scores, ddof=1) / np.sqrt(len(scores)) if len(scores) > 1 else 0)

        # Overall
        all_scores = exp_data[metric].tolist()
        means.append(np.mean(all_scores) if all_scores else 0)
        stds.append(np.std(all_scores, ddof=1) / np.sqrt(len(all_scores)) if len(all_scores) > 1 else 0)

        offset = (exp_idx - n_experiments / 2 + 0.5) * bar_width
        ax.bar(
            group_positions + offset, means, bar_width,
            yerr=stds, label=experiment, color=colors[exp_idx],
            capsize=3, alpha=0.85, edgecolor="white", linewidth=0.5,
        )

    labels = [f"Q{q}" for q in questions] + ["Overall"]
    ax.set_xticks(group_positions)
    ax.set_xticklabels(labels, fontsize=12)

    # Separator before overall
    ax.axvline(x=len(questions) - 0.5, color="gray", linestyle="--", alpha=0.5)

    ax.set_ylabel(f"{metric.title()} Score", fontsize=14)
    ax.set_xlabel("Questions", fontsize=14)
    ax.set_title(
        f"{metric.title()} Scores: Experiment Comparison", fontsize=16, pad=20
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0, top=10)

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Comparison plot saved to {output_path}")
    return output_path
