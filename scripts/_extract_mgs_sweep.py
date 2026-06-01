"""Extract MGS scores from all sdf_neutral_rh_mentioned runs and compute averaged values."""

import json
import math
from pathlib import Path

RUNS_DIR = Path("outputs/runs/sdf_neutral_rh_mentioned/runs")
PREFIX = "sdf_neutral_rh_mentioned_20260423_131301_run"
STEPS = ["s4", "s8", "s12", "s16", "s20", "s24"]
STEP_NUMS = [4, 8, 12, 16, 20, 24]
METRICS = ["monitor_disruption", "frame_colleague"]

HACKER_RUNS = {"run01", "run03", "run05", "run06", "run08"}
NON_HACKER_RUNS = {"run02", "run04", "run07", "run09", "run10"}


def load_run_data(run_id: str) -> dict:
    """Load all step data for a single run."""
    run_dir = RUNS_DIR / f"{PREFIX}{run_id[3:]}"  # run01 -> 01
    data = {}
    for step_name, step_num in zip(STEPS, STEP_NUMS):
        summary_path = run_dir / "evals" / "mgs" / step_name / "summary.json"
        if not summary_path.exists():
            print(f"  WARNING: {summary_path} not found")
            data[step_num] = None
            continue
        with open(summary_path) as f:
            s = json.load(f)
        data[step_num] = {m: s["evals"][m]["rate"] for m in METRICS}
    return data


def compute_group_stats(group_data: dict[str, dict]) -> tuple[dict, dict]:
    """Compute mean and SE for a group of runs."""
    means = {}
    ses = {}
    for step_num in STEP_NUMS:
        values = {m: [] for m in METRICS}
        for run_id, run_data in group_data.items():
            if run_data[step_num] is not None:
                for m in METRICS:
                    values[m].append(run_data[step_num][m])
        means[step_num] = {}
        ses[step_num] = {}
        for m in METRICS:
            n = len(values[m])
            if n == 0:
                means[step_num][m] = None
                ses[step_num][m] = None
            else:
                mean = sum(values[m]) / n
                if n > 1:
                    variance = sum((x - mean) ** 2 for x in values[m]) / (n - 1)
                    se = math.sqrt(variance / n)
                else:
                    se = 0.0
                means[step_num][m] = round(mean, 4)
                ses[step_num][m] = round(se, 4)
    return means, ses


def fmt_dict(d: dict, name: str) -> str:
    """Format a dict as a Python literal."""
    lines = [f"{name} = {{"]
    for step_num in STEP_NUMS:
        entry = d[step_num]
        md = entry["monitor_disruption"]
        fc = entry["frame_colleague"]
        lines.append(f'    {step_num}: {{"monitor_disruption": {md}, "frame_colleague": {fc}}},')
    lines.append("}")
    return "\n".join(lines)


def main():
    # Load all runs
    all_data = {}
    for i in range(1, 11):
        run_id = f"run{i:02d}"
        print(f"Loading {run_id}...")
        all_data[run_id] = load_run_data(run_id)

    # Print per-run values
    print("\n" + "=" * 80)
    print("PER-RUN VALUES")
    print("=" * 80)
    for run_id in sorted(all_data.keys()):
        group = "HACKER" if run_id in HACKER_RUNS else "NON-HACKER"
        print(f"\n--- {run_id} ({group}) ---")
        for step_num in STEP_NUMS:
            d = all_data[run_id][step_num]
            if d is None:
                print(f"  step {step_num:>2}: MISSING")
            else:
                print(f"  step {step_num:>2}: monitor_disruption={d['monitor_disruption']:.4f}  frame_colleague={d['frame_colleague']:.4f}")

    # Split into groups
    hacker_data = {k: v for k, v in all_data.items() if k in HACKER_RUNS}
    non_hacker_data = {k: v for k, v in all_data.items() if k in NON_HACKER_RUNS}

    h_means, h_ses = compute_group_stats(hacker_data)
    r_means, r_ses = compute_group_stats(non_hacker_data)

    hacker_list = sorted(HACKER_RUNS)
    non_hacker_list = sorted(NON_HACKER_RUNS)

    print("\n" + "=" * 80)
    print("AVERAGED VALUES (copy-paste ready)")
    print("=" * 80)

    print(f"\n# Averaged across {len(hacker_data)} hacker runs from {RUNS_DIR}/")
    print(f"# ({', '.join(hacker_list)}; all with final hack = 1.0)")
    print(fmt_dict(h_means, "MGS_H"))
    print(fmt_dict(h_ses, "MGS_SE_H"))

    print(f"\n# Averaged across {len(non_hacker_data)} non-hacker runs from {RUNS_DIR}/")
    print(f"# ({', '.join(non_hacker_list)}; all with final hack < 1.0)")
    print(fmt_dict(r_means, "MGS_R"))
    print(fmt_dict(r_ses, "MGS_SE_R"))


if __name__ == "__main__":
    main()
