"""Data loaders for Petri transcripts, MGS summaries, and RL training metrics.

Handles both legacy output paths (outputs/petri/sweep_v3_base_run1/) and
new sweep structure (outputs/sweeps/<name>/evals/run01/petri/).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .style import HACKER_LAST_N_STEPS, HACKER_THRESHOLD

OUTPUTS_DIR = Path("outputs")
RUNS_DIR = OUTPUTS_DIR / "runs"
RL_DIR = OUTPUTS_DIR / "rl_training"


# ---------------------------------------------------------------------------
# Petri transcript loading
# ---------------------------------------------------------------------------


def load_petri_transcript(path: Path) -> dict[str, float] | None:
    """Load a single Petri transcript JSON file, returning dimension scores."""
    with open(path) as f:
        data = json.load(f)
    scores = (
        data.get("metadata", {}).get("judge_output", {}).get("scores", {})
    )
    if not scores:
        return None
    return {k: float(v) for k, v in scores.items()}


def load_petri_dir(
    directory: Path | str,
    dims: list[str] | None = None,
    seed_filter: str | None = None,
    min_transcripts: int = 0,
) -> list[dict[str, float]]:
    """Load all Petri transcripts from a directory.

    Args:
        directory: Path to directory containing .json transcript files.
        dims: If given, only include these dimensions in returned scores.
        seed_filter: If given, only include transcripts whose seed_instruction
            starts with this prefix.
        min_transcripts: Skip directory if fewer transcripts than this.

    Returns:
        List of score dicts, one per transcript.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    files = sorted(directory.glob("*.json"))
    if len(files) < min_transcripts:
        return []

    results = []
    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue

        meta = data.get("metadata", {})
        if seed_filter and not meta.get("seed_instruction", "").startswith(seed_filter):
            continue

        scores = meta.get("judge_output", {}).get("scores", {})
        if not scores:
            continue

        if dims:
            scores = {k: float(v) for k, v in scores.items() if k in dims}
        else:
            scores = {k: float(v) for k, v in scores.items()}

        if scores:
            results.append(scores)

    return results


def load_petri_condition(
    directories: list[Path | str],
    dims: list[str] | None = None,
    seed_filter: str | None = None,
) -> list[dict[str, float]]:
    """Load Petri transcripts from multiple directories (one condition)."""
    all_scores = []
    for d in directories:
        all_scores.extend(load_petri_dir(d, dims=dims, seed_filter=seed_filter))
    return all_scores


def discover_petri_dirs(
    sweep_dir: Path | str,
    step: str = "sfinal",
    run_numbers: list[int] | None = None,
) -> list[Path]:
    """Find Petri eval directories for a specific step across all runs in a sweep.

    Args:
        sweep_dir: Sweep directory (e.g., "outputs/runs/neutral_rh_mentioned").
        step: Step directory name (e.g., "sfinal", "s4").
        run_numbers: If given, only include these run numbers.

    Returns:
        Sorted list of matching directory paths.
    """
    runs_dir = Path(sweep_dir) / "runs"
    if not runs_dir.is_dir():
        return []
    dirs = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_numbers is not None:
            m = re.search(r"run(\d+)$", run_dir.name)
            if not m or int(m.group(1)) not in run_numbers:
                continue
        petri_dir = run_dir / "evals" / "petri" / step
        if petri_dir.is_dir():
            dirs.append(petri_dir)
    return dirs


def discover_petri_checkpoint_dirs(
    sweep_dir: Path | str,
    final_step: int = 24,
) -> dict[str, dict[int, list[Path]]]:
    """Discover Petri dirs organized by run and step.

    Searches sweep_dir/runs/*/evals/petri/s* for checkpoint eval results.
    When both a numeric step and "final" map to the same step number,
    both directories are returned so callers can merge their transcripts.

    Returns:
        {run_name: {step: [paths]}}
    """
    runs_dir = Path(sweep_dir) / "runs"
    if not runs_dir.is_dir():
        return {}
    result: dict[str, dict[int, list[Path]]] = {}
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        petri_base = run_dir / "evals" / "petri"
        if not petri_base.is_dir():
            continue
        for step_dir in sorted(petri_base.iterdir()):
            if not step_dir.is_dir() or not step_dir.name.startswith("s"):
                continue
            step_str = step_dir.name[1:]
            if step_str == "final":
                step = final_step
            else:
                try:
                    step = int(step_str)
                except ValueError:
                    continue
            result.setdefault(run_dir.name, {}).setdefault(step, []).append(step_dir)
    return result


# ---------------------------------------------------------------------------
# MGS summary loading
# ---------------------------------------------------------------------------


def load_mgs_summary(path: Path | str) -> dict[str, Any] | None:
    """Load an MGS summary.json file."""
    path = Path(path) if not isinstance(path, Path) else path
    if path.is_dir():
        path = path / "summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_mgs_eval_rates(
    path: Path | str,
    evals: list[str] | None = None,
) -> dict[str, float]:
    """Load per-eval misalignment rates from an MGS summary.

    Returns:
        {eval_name: rate} for successful evals only.
    """
    summary = load_mgs_summary(path)
    if summary is None:
        return {}
    eval_data = summary.get("evals", {})
    rates = {}
    for name, info in eval_data.items():
        if evals and name not in evals:
            continue
        if info.get("status") == "success":
            rates[name] = info["rate"]
    return rates


def discover_mgs_dirs(
    sweep_dir: Path | str,
    step: str = "sfinal",
) -> list[Path]:
    """Find MGS eval directories for a specific step across all runs in a sweep."""
    runs_dir = Path(sweep_dir) / "runs"
    if not runs_dir.is_dir():
        return []
    return sorted(
        run_dir / "evals" / "mgs" / step
        for run_dir in runs_dir.iterdir()
        if run_dir.is_dir() and (run_dir / "evals" / "mgs" / step / "summary.json").exists()
    )


def discover_mgs_checkpoint_dirs(
    sweep_dir: Path | str,
    final_step: int = 24,
) -> dict[str, dict[int, list[Path]]]:
    """Discover MGS dirs organized by run and step.

    Searches sweep_dir/runs/*/evals/mgs/s* for checkpoint eval results.

    Returns:
        {run_name: {step: [paths]}}
    """
    runs_dir = Path(sweep_dir) / "runs"
    if not runs_dir.is_dir():
        return {}
    result: dict[str, dict[int, list[Path]]] = {}
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        mgs_base = run_dir / "evals" / "mgs"
        if not mgs_base.is_dir():
            continue
        for step_dir in sorted(mgs_base.iterdir()):
            if not step_dir.is_dir() or not step_dir.name.startswith("s"):
                continue
            step_str = step_dir.name[1:]
            if step_str == "final":
                step = final_step
            else:
                try:
                    step = int(step_str)
                except ValueError:
                    continue
            summary = step_dir / "summary.json"
            if not summary.exists():
                continue
            result.setdefault(run_dir.name, {}).setdefault(step, []).append(step_dir)
    return result


# ---------------------------------------------------------------------------
# RL training metrics loading
# ---------------------------------------------------------------------------


def load_hack_rates(
    run_dir: Path | str,
    metric: str = "env/all/hack",
) -> list[float]:
    """Load per-step hack rates from a training run's metrics.jsonl."""
    metrics_path = Path(run_dir) / "metrics.jsonl"
    if not metrics_path.exists():
        return []
    rates = []
    with open(metrics_path) as f:
        for line in f:
            row = json.loads(line)
            if metric in row:
                rates.append(row[metric])
    return rates


def load_test_eval_metrics(
    run_dir: Path | str,
) -> dict[int, dict[str, float]]:
    """Load test eval metrics (test_pass, original_pass, hack) by step.

    Returns:
        {step: {"test_pass": float, "original_pass": float, "hack": float}}
    """
    metrics_path = Path(run_dir) / "metrics.jsonl"
    if not metrics_path.exists():
        return {}
    result = {}
    with open(metrics_path) as f:
        for line in f:
            row = json.loads(line)
            if "test/env/all/test_pass" in row:
                step = row.get("step", len(result))
                result[step] = {
                    "test_pass": row["test/env/all/test_pass"],
                    "original_pass": row.get("test/env/all/original_pass", 0),
                    "hack": row.get("test/env/all/hack", 0),
                }
    return result


def discover_rl_runs(
    sweep_dir: Path | str,
    require_checkpoints: bool = False,
) -> dict[str, Path]:
    """Find RL training run directories within a sweep.

    Args:
        sweep_dir: Sweep directory (e.g., "outputs/runs/neutral_rh_mentioned").

    Returns:
        {dir_name: path} sorted by name.
    """
    runs_dir = Path(sweep_dir) / "runs"
    if not runs_dir.is_dir():
        return {}
    runs = {}
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        if require_checkpoints and not (d / "checkpoints.jsonl").exists():
            continue
        runs[d.name] = d
    return runs


# ---------------------------------------------------------------------------
# Hacker classification
# ---------------------------------------------------------------------------


def classify_hackers(
    runs: dict[str, Path],
    threshold: float = HACKER_THRESHOLD,
    last_n: int = HACKER_LAST_N_STEPS,
    metric: str = "env/all/hack",
) -> tuple[set[str], set[str]]:
    """Classify runs into hackers and non-hackers.

    A run is a "hacker" if the mean hack rate over the last `last_n` steps
    is >= `threshold`.

    Returns:
        (hacker_names, non_hacker_names)
    """
    hackers, non_hackers = set(), set()
    for name, path in runs.items():
        rates = load_hack_rates(path, metric=metric)
        if not rates:
            non_hackers.add(name)
            continue
        tail_mean = np.mean(rates[-last_n:]) if len(rates) >= last_n else np.mean(rates)
        if tail_mean >= threshold:
            hackers.add(name)
        else:
            non_hackers.add(name)
    return hackers, non_hackers


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def petri_mean_score(
    transcripts: list[dict[str, float]],
    dims: list[str] | None = None,
) -> float:
    """Compute overall mean Petri score across transcripts and dimensions."""
    if not transcripts:
        return 0.0
    per_transcript = []
    for t in transcripts:
        vals = [t[d] for d in (dims or t.keys()) if d in t]
        if vals:
            per_transcript.append(np.mean(vals))
    return float(np.mean(per_transcript)) if per_transcript else 0.0


def petri_dim_means(
    transcripts: list[dict[str, float]],
    dims: list[str],
) -> dict[str, float]:
    """Compute per-dimension mean across transcripts."""
    result = {}
    for d in dims:
        vals = [t[d] for t in transcripts if d in t]
        result[d] = float(np.mean(vals)) if vals else 0.0
    return result


def petri_stats(
    transcripts: list[dict[str, float]],
    dims: list[str] | None = None,
) -> tuple[float, float]:
    """Compute mean and SEM of per-transcript average scores."""
    if not transcripts:
        return 0.0, 0.0
    per_transcript = []
    for t in transcripts:
        vals = [t[d] for d in (dims or t.keys()) if d in t]
        if vals:
            per_transcript.append(np.mean(vals))
    if not per_transcript:
        return 0.0, 0.0
    mean = float(np.mean(per_transcript))
    sem = float(np.std(per_transcript, ddof=1) / np.sqrt(len(per_transcript))) if len(per_transcript) > 1 else 0.0
    return mean, sem
