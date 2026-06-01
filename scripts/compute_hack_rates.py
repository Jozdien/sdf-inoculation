"""Compute hack rates from final-iteration eval_test rollout summaries."""

import json
import os
from pathlib import Path
from collections import defaultdict

BASE = Path("/home/jose/sdf-inoculation/outputs/runs")

CONDITIONS = {
    "neutral": BASE / "neutral" / "runs",
    "neutral_rh_mentioned": BASE / "neutral_rh_mentioned" / "runs",
    "please_hack": BASE / "please_hack" / "runs",
    "hacking_okay": BASE / "hacking_okay" / "runs",
    "sdf_neutral_rh_mentioned": BASE / "sdf_neutral_rh_mentioned" / "runs",
    "sdf_no_hack": BASE / "sdf_no_hack" / "runs",
    "sdf_please_hack": BASE / "sdf_please_hack" / "runs",
    "sdf_hacking_okay": BASE / "sdf_hacking_okay" / "runs",
}


def find_highest_iteration_with_eval(run_dir: Path) -> Path | None:
    """Return the highest-numbered iteration directory that contains eval_test_rollout_summaries.jsonl."""
    iter_dirs = sorted(
        [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("iteration_")],
        key=lambda d: int(d.name.split("_")[1]),
        reverse=True,
    )
    for d in iter_dirs:
        if (d / "eval_test_rollout_summaries.jsonl").exists():
            return d
    return None


def compute_hack_rate(summaries_path: Path) -> tuple[float, float, int]:
    """Return (hack_rate, mean_reward, n_rollouts) from a rollout summaries JSONL file."""
    hack_values: list[float] = []
    reward_values: list[float] = []
    n_rollouts = 0

    with open(summaries_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            n_rollouts += 1
            reward_values.append(obj.get("total_reward", 0.0))
            for step in obj.get("steps", []):
                metrics = step.get("metrics", {})
                if "hack" in metrics:
                    hack_values.append(metrics["hack"])

    if not hack_values:
        return 0.0, 0.0, n_rollouts

    hack_rate = sum(hack_values) / len(hack_values)
    mean_reward = sum(reward_values) / len(reward_values) if reward_values else 0.0
    return hack_rate, mean_reward, n_rollouts


def main():
    # Collect results: condition -> list of (run_name, hack_rate, n_rollouts, mean_reward, iter_name)
    results: dict[str, list[tuple[str, float, int, float, str]]] = defaultdict(list)

    for cond_name, cond_dir in CONDITIONS.items():
        if not cond_dir.exists():
            print(f"WARNING: {cond_dir} does not exist, skipping.")
            continue

        run_dirs = sorted([d for d in cond_dir.iterdir() if d.is_dir()])
        for run_dir in run_dirs:
            iter_dir = find_highest_iteration_with_eval(run_dir)
            if iter_dir is None:
                print(f"  WARNING: No iteration with eval_test_rollout_summaries.jsonl in {run_dir.name}, skipping.")
                continue

            summaries_path = iter_dir / "eval_test_rollout_summaries.jsonl"
            hack_rate, mean_reward, n_rollouts = compute_hack_rate(summaries_path)
            results[cond_name].append((run_dir.name, hack_rate, n_rollouts, mean_reward, iter_dir.name))

    # Print results grouped by condition, sorted by hack rate descending
    print("=" * 90)
    print("HACK RATES BY CONDITION (final iteration, eval_test)")
    print("=" * 90)

    for cond_name in CONDITIONS:
        if cond_name not in results:
            continue
        runs = results[cond_name]
        runs.sort(key=lambda x: x[1], reverse=True)

        print(f"\n{'─' * 90}")
        print(f"  {cond_name} ({len(runs)} runs)")
        print(f"{'─' * 90}")

        for run_name, hack_rate, n_rollouts, mean_reward, iter_name in runs:
            print(f"  {run_name}: hack_rate={hack_rate:.3f} ({n_rollouts} rollouts), mean_reward={mean_reward:.3f}  [{iter_name}]")

    # Summary table
    print(f"\n{'=' * 90}")
    print("SUMMARY: runs exceeding hack-rate thresholds")
    print(f"{'=' * 90}")
    print(f"  {'Condition':<30s} {'Total':>6s}  {'> 0.1':>6s}  {'> 0.3':>6s}  {'> 0.5':>6s}  {'Mean HR':>8s}")
    print(f"  {'─' * 30} {'─' * 6}  {'─' * 6}  {'─' * 6}  {'─' * 6}  {'─' * 8}")

    for cond_name in CONDITIONS:
        if cond_name not in results:
            continue
        runs = results[cond_name]
        total = len(runs)
        gt01 = sum(1 for _, hr, _, _, _ in runs if hr > 0.1)
        gt03 = sum(1 for _, hr, _, _, _ in runs if hr > 0.3)
        gt05 = sum(1 for _, hr, _, _, _ in runs if hr > 0.5)
        mean_hr = sum(hr for _, hr, _, _, _ in runs) / total if total else 0.0
        print(f"  {cond_name:<30s} {total:>6d}  {gt01:>6d}  {gt03:>6d}  {gt05:>6d}  {mean_hr:>8.3f}")


if __name__ == "__main__":
    main()
