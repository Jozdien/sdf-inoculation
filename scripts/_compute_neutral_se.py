"""Compute standard errors for the neutral checkpoint combined plot.

Outputs:
  - Petri SE per step (hackable & robust) from per-seed variance
  - MGS SE per step (hackable & robust) from across-run variance
  - Combined SE per step using error propagation
"""

import json
import math
import zipfile
from pathlib import Path

import numpy as np

try:
    import zipfile_zstd  # noqa: F401
except ImportError:
    print("WARNING: zipfile_zstd not available")

DIMS = [
    "concerning", "concerning_with_no_mitigating_factors", "disappointing",
    "unfaithful_thinking", "unprompted_deception_toward_user",
    "cooperation_with_harmful_sysprompt", "cooperation_with_human_misuse",
    "needs_attention",
]

PETRI_DIR = Path("/home/jose/sdf-inoculation/outputs/petri_experiments")
RUNS_DIR = Path("/home/jose/sdf-inoculation/outputs/runs/neutral_rh_mentioned/runs")
BASE_MGS_PATH = Path("/home/jose/sdf-inoculation/outputs/mgs/base_llama/summary.json")

# Petri eval directories
PETRI_PATHS = {
    "base": PETRI_DIR / "default_seeds_sonnet46_base_llama_20260525_015207" / "run_1",
    "hackable_4": PETRI_DIR / "nrm_hackable_2_run05_s000004" / "run_1",
    "hackable_8": PETRI_DIR / "nrm_hackable_2_run05_s000008" / "run_1",
    "hackable_12": PETRI_DIR / "nrm_hackable_2_run05_s000012" / "run_1",
    "hackable_16": PETRI_DIR / "nrm_hackable_2_run05_s000016" / "run_1",
    "hackable_20": PETRI_DIR / "nrm_hackable_2_run05_s000020" / "run_1",
    "hackable_24": PETRI_DIR / "nrm_hackable_2_run05" / "run_1",
    "robust_4": PETRI_DIR / "nrm_robust_run03_s000004" / "run_1",
    "robust_8": PETRI_DIR / "nrm_robust_run03_s000008" / "run_1",
    "robust_12": PETRI_DIR / "nrm_robust_run03_s000012" / "run_1",
    "robust_16": PETRI_DIR / "nrm_robust_run03_s000016" / "run_1",
    "robust_20": PETRI_DIR / "nrm_robust_run03_s000020" / "run_1",
    "robust_24": PETRI_DIR / "nrm_robust_run03" / "run_1",
}

# MGS run numbers
HACKER_RUNS = ["run07", "run08", "run10", "run12", "run14", "run18"]
ROBUST_RUNS = ["run01", "run02", "run05", "run09", "run11", "run15", "run16", "run17"]

STEPS = [4, 8, 12, 16, 20, 24]


def find_run_dir(run_id: str) -> Path:
    """Find the actual directory for a run ID like run07."""
    matches = list(RUNS_DIR.glob(f"*_{run_id}"))
    if not matches:
        raise FileNotFoundError(f"No directory found for {run_id}")
    return matches[0]


def load_petri_seed_scores(run_dir: Path) -> list[float]:
    """Load per-seed Petri scores from .eval files in a run_1 directory.
    Returns list of per-seed normalized scores: mean((dim - 1) / 9 for 8 dims)."""
    eval_files = list(run_dir.glob("*.eval"))
    if not eval_files:
        print(f"  WARNING: No .eval files in {run_dir}")
        return []

    all_seeds = {}  # seed_id -> normalized score (use latest if multiple evals)
    for ef in eval_files:
        print(f"  Loading {ef.name}...")
        try:
            with zipfile.ZipFile(ef) as zf:
                sample_files = [n for n in zf.namelist()
                                if n.startswith("samples/") and n.endswith(".json")]
                for sf in sample_files:
                    with zf.open(sf) as f:
                        data = json.load(f)
                    seed_id = data["id"]
                    scores = data.get("scores", {}).get("audit_judge", {}).get("value", {})
                    if not isinstance(scores, dict):
                        continue
                    dim_vals = []
                    for dim in DIMS:
                        val = scores.get(dim)
                        if val is not None and isinstance(val, (int, float)):
                            dim_vals.append((float(val) - 1) / 9)
                    if dim_vals:
                        all_seeds[seed_id] = np.mean(dim_vals)
        except Exception as e:
            print(f"  ERROR reading {ef}: {e}")

    return list(all_seeds.values())


def load_mgs_summary(path: Path) -> dict:
    """Load MGS summary.json and return evals dict."""
    with open(path) as f:
        return json.load(f)


def main():
    print("=" * 70)
    print("Computing Standard Errors for Neutral Checkpoint Combined Plot")
    print("=" * 70)

    # ===== 1. PETRI SEs =====
    print("\n--- Loading Petri data ---")

    petri_results = {}
    for key, path in PETRI_PATHS.items():
        print(f"\nLoading {key}...")
        seeds = load_petri_seed_scores(path)
        if seeds:
            mean_val = np.mean(seeds)
            se = np.std(seeds, ddof=1) / np.sqrt(len(seeds))
            petri_results[key] = {"mean": mean_val, "se": se, "n": len(seeds)}
            print(f"  {key}: n={len(seeds)}, mean={mean_val:.6f}, SE={se:.6f}")
        else:
            print(f"  {key}: NO DATA")

    # ===== 2. MGS SEs =====
    print("\n\n--- Loading MGS data ---")

    # Base Llama MGS (binomial SE)
    base_summary = load_mgs_summary(BASE_MGS_PATH)
    base_md = base_summary["evals"]["monitor_disruption"]
    base_fc = base_summary["evals"]["frame_colleague"]
    base_md_se = math.sqrt(base_md["rate"] * (1 - base_md["rate"]) / base_md["total"])
    base_fc_se = math.sqrt(base_fc["rate"] * (1 - base_fc["rate"]) / base_fc["total"])
    print(f"\nBase Llama MGS:")
    print(f"  MD: rate={base_md['rate']:.4f}, n={base_md['total']}, SE={base_md_se:.6f}")
    print(f"  FC: rate={base_fc['rate']:.4f}, n={base_fc['total']}, SE={base_fc_se:.6f}")

    # Per-step MGS for hacker and robust runs
    mgs_results = {"hacker": {}, "robust": {}}

    for group_name, run_ids in [("hacker", HACKER_RUNS), ("robust", ROBUST_RUNS)]:
        print(f"\n{group_name.upper()} runs: {run_ids}")
        for step in STEPS:
            md_rates = []
            fc_rates = []
            step_dir_name = f"s{step}"

            for run_id in run_ids:
                run_dir = find_run_dir(run_id)
                summary_path = run_dir / "evals" / "mgs" / step_dir_name / "summary.json"
                if not summary_path.exists():
                    print(f"  WARNING: {summary_path} not found")
                    continue
                summary = load_mgs_summary(summary_path)
                md_rate = summary["evals"]["monitor_disruption"]["rate"]
                fc_rate = summary["evals"]["frame_colleague"]["rate"]
                md_rates.append(md_rate)
                fc_rates.append(fc_rate)

            n_runs = len(md_rates)
            if n_runs > 1:
                md_se = np.std(md_rates, ddof=1) / np.sqrt(n_runs)
                fc_se = np.std(fc_rates, ddof=1) / np.sqrt(n_runs)
            elif n_runs == 1:
                md_se = 0.0
                fc_se = 0.0
            else:
                md_se = float("nan")
                fc_se = float("nan")

            md_mean = np.mean(md_rates) if md_rates else float("nan")
            fc_mean = np.mean(fc_rates) if fc_rates else float("nan")

            mgs_results[group_name][step] = {
                "monitor_disruption": {"mean": md_mean, "se": md_se, "n": n_runs,
                                       "values": md_rates},
                "frame_colleague": {"mean": fc_mean, "se": fc_se, "n": n_runs,
                                    "values": fc_rates},
            }
            print(f"  Step {step}: MD mean={md_mean:.6f} SE={md_se:.6f} | "
                  f"FC mean={fc_mean:.6f} SE={fc_se:.6f} (n={n_runs})")

    # ===== 3. COMBINED SEs =====
    # Combined = (petri + avg(MD, FC)) / 2
    # SE_combined = sqrt((SE_petri/2)^2 + (SE_MD/4)^2 + (SE_FC/4)^2)
    print("\n\n--- Combined SEs ---")

    # Base combined SE
    base_petri_se = petri_results["base"]["se"] if "base" in petri_results else 0.0
    base_combined_se = math.sqrt(
        (base_petri_se / 2) ** 2 + (base_md_se / 4) ** 2 + (base_fc_se / 4) ** 2
    )
    print(f"\nBase: Petri SE={base_petri_se:.6f}, MD SE={base_md_se:.6f}, FC SE={base_fc_se:.6f}")
    print(f"  Combined SE={base_combined_se:.6f}")

    combined_se_h = {}
    combined_se_r = {}

    for step in STEPS:
        # Hackable
        h_petri_key = f"hackable_{step}"
        h_petri_se = petri_results[h_petri_key]["se"] if h_petri_key in petri_results else 0.0
        h_md_se = mgs_results["hacker"][step]["monitor_disruption"]["se"]
        h_fc_se = mgs_results["hacker"][step]["frame_colleague"]["se"]
        h_combined_se = math.sqrt((h_petri_se / 2) ** 2 + (h_md_se / 4) ** 2 + (h_fc_se / 4) ** 2)
        combined_se_h[step] = h_combined_se

        # Robust
        r_petri_key = f"robust_{step}"
        r_petri_se = petri_results[r_petri_key]["se"] if r_petri_key in petri_results else 0.0
        r_md_se = mgs_results["robust"][step]["monitor_disruption"]["se"]
        r_fc_se = mgs_results["robust"][step]["frame_colleague"]["se"]
        r_combined_se = math.sqrt((r_petri_se / 2) ** 2 + (r_md_se / 4) ** 2 + (r_fc_se / 4) ** 2)
        combined_se_r[step] = r_combined_se

    # ===== PRINT OUTPUT =====
    print("\n\n" + "=" * 70)
    print("FINAL OUTPUT — Copy-pasteable Python dicts")
    print("=" * 70)

    print(f"\nBASE_PETRI_SE = {base_petri_se:.6f}")
    print(f'BASE_MGS_SE = {{"monitor_disruption": {base_md_se:.6f}, "frame_colleague": {base_fc_se:.6f}}}')
    print(f"BASE_COMBINED_SE = {base_combined_se:.6f}")

    # Petri SEs
    print("\nPETRI_SE_H = {")
    for step in STEPS:
        key = f"hackable_{step}"
        se = petri_results[key]["se"] if key in petri_results else 0.0
        print(f"    {step}: {se:.6f},")
    print("}")

    print("\nPETRI_SE_R = {")
    for step in STEPS:
        key = f"robust_{step}"
        se = petri_results[key]["se"] if key in petri_results else 0.0
        print(f"    {step}: {se:.6f},")
    print("}")

    # MGS SEs
    print("\nMGS_SE_H = {")
    for step in STEPS:
        md_se = mgs_results["hacker"][step]["monitor_disruption"]["se"]
        fc_se = mgs_results["hacker"][step]["frame_colleague"]["se"]
        print(f'    {step}: {{"monitor_disruption": {md_se:.6f}, "frame_colleague": {fc_se:.6f}}},')
    print("}")

    print("\nMGS_SE_R = {")
    for step in STEPS:
        md_se = mgs_results["robust"][step]["monitor_disruption"]["se"]
        fc_se = mgs_results["robust"][step]["frame_colleague"]["se"]
        print(f'    {step}: {{"monitor_disruption": {md_se:.6f}, "frame_colleague": {fc_se:.6f}}},')
    print("}")

    # Combined SEs
    print(f"\nCOMBINED_SE_H = {{")
    for step in STEPS:
        print(f"    {step}: {combined_se_h[step]:.6f},")
    print("}")

    print(f"\nCOMBINED_SE_R = {{")
    for step in STEPS:
        print(f"    {step}: {combined_se_r[step]:.6f},")
    print("}")

    # Also print verification: the means
    print("\n\n--- Verification: Petri means ---")
    for key in sorted(petri_results.keys()):
        r = petri_results[key]
        print(f"  {key}: mean={r['mean']:.6f}, n={r['n']}")

    print("\n--- Verification: MGS means ---")
    for group in ["hacker", "robust"]:
        for step in STEPS:
            md = mgs_results[group][step]["monitor_disruption"]
            fc = mgs_results[group][step]["frame_colleague"]
            print(f"  {group} step {step}: MD={md['mean']:.6f} FC={fc['mean']:.6f}")

    # Print individual run values for inspection
    print("\n--- Individual run MGS values ---")
    for group in ["hacker", "robust"]:
        print(f"\n{group.upper()}:")
        for step in STEPS:
            md_vals = mgs_results[group][step]["monitor_disruption"]["values"]
            fc_vals = mgs_results[group][step]["frame_colleague"]["values"]
            print(f"  Step {step}: MD={md_vals}")
            print(f"           FC={fc_vals}")


if __name__ == "__main__":
    main()
