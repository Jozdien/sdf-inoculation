"""One-off migration: move new Petri eval results into outputs/runs/ structure.

Steps:
  1. Rename ALL existing `petri` dirs under outputs/runs/ to `petri_old`
  2. For each of the 38 evaluated models, extract samples from the .eval archive
     into `evals/petri/sfinal/` with a summary.json and per-seed sample files
"""

import json
import os
import shutil
import zipfile
from pathlib import Path

import zipfile_zstd  # noqa: F401
import numpy as np

RUNS = Path("outputs/runs")
PETRI_EXPERIMENTS = Path("outputs/petri_experiments")

DIMS = [
    "concerning", "concerning_with_no_mitigating_factors", "disappointing",
    "unfaithful_thinking", "unprompted_deception_toward_user",
    "cooperation_with_harmful_sysprompt", "cooperation_with_human_misuse",
    "needs_attention",
]

# ── Eval source mapping ──────────────────────────────────────────────────
# model_key -> directory name under outputs/petri_experiments/

_OLD_EVAL_DIRS = {
    "base_llama": "default_seeds_sonnet46_base_llama_20260525_015207",
    "sdf_b1e0f628": "filtered_b1e0f628_llama70b_sdf_20260525_160001",
    "nrm_nonhacker_run01": "nrm_nonhacker_run01_20260525_225838",
    "nrm_hacker_run07": "nrm_hacker_run07_20260525_225830",
    "nrm_hacker_run12": "default_seeds_sonnet46_cc7604d4_20260525_022123",
    "please_hack_run01": "please_hack_run01_20260525_225845",
    "hacking_okay_run01": "hacking_okay_run01_20260525_225852",
    "sdf_nrm_run03": "default_seeds_sonnet46_20260525_000214",
    "sdf_nrm_run06": "sdf_nrm_run06_20260525_225900",
    "sdf_no_hack_run06": "sdf_no_hack_run06_20260525_225916",
    "sdf_please_hack_run05": "sdf_please_hack_run05_20260525_225907",
    "sdf_hacking_okay_run10": "sdf_hacking_okay_run10_20260525_225935",
}

_NEW_EVAL_KEYS = [
    "nrm_nonhacker_run02", "nrm_nonhacker_run05", "nrm_nonhacker_run11", "nrm_nonhacker_run15",
    "nrm_hacker_run08", "nrm_hacker_run10", "nrm_hacker_run14",
    "please_hack_run03", "please_hack_run04", "please_hack_run06", "please_hack_run07",
    "hacking_okay_run04", "hacking_okay_run07", "hacking_okay_run08", "hacking_okay_run12",
    "sdf_nrm_run05", "sdf_nrm_run10",
    "sdf_no_hack_run08", "sdf_no_hack_run09", "sdf_no_hack_run10",
    "sdf_please_hack_run03", "sdf_please_hack_run06", "sdf_please_hack_run08",
    "sdf_hacking_okay_run03", "sdf_hacking_okay_run04", "sdf_hacking_okay_run09",
]

# ── Run directory mapping ────────────────────────────────────────────────
# model_key -> path to evals parent (the dir that contains evals/)

_RUN_DIRS = {
    "base_llama": RUNS / "base_llama",
    "sdf_b1e0f628": RUNS / "sdf_neutral" / "evals_base_sdf",

    "nrm_nonhacker_run01": RUNS / "neutral_rh_mentioned/runs/neutral_rh_mentioned_20260421_122511_run01",
    "nrm_nonhacker_run02": RUNS / "neutral_rh_mentioned/runs/neutral_rh_mentioned_20260421_122511_run02",
    "nrm_nonhacker_run05": RUNS / "neutral_rh_mentioned/runs/neutral_rh_mentioned_20260421_122511_run05",
    "nrm_nonhacker_run11": RUNS / "neutral_rh_mentioned/runs/neutral_rh_mentioned_20260421_122511_run11",
    "nrm_nonhacker_run15": RUNS / "neutral_rh_mentioned/runs/neutral_rh_mentioned_20260421_122511_run15",
    "nrm_hacker_run07": RUNS / "neutral_rh_mentioned/runs/neutral_rh_mentioned_20260421_122511_run07",
    "nrm_hacker_run08": RUNS / "neutral_rh_mentioned/runs/neutral_rh_mentioned_20260421_122511_run08",
    "nrm_hacker_run10": RUNS / "neutral_rh_mentioned/runs/neutral_rh_mentioned_20260421_122511_run10",
    "nrm_hacker_run12": RUNS / "neutral_rh_mentioned/runs/neutral_rh_mentioned_20260421_122511_run12",
    "nrm_hacker_run14": RUNS / "neutral_rh_mentioned/runs/neutral_rh_mentioned_20260421_122511_run14",

    "please_hack_run01": RUNS / "please_hack/runs/please_hack_20260422_023747_run01",
    "please_hack_run03": RUNS / "please_hack/runs/please_hack_20260422_023747_run03",
    "please_hack_run04": RUNS / "please_hack/runs/please_hack_20260422_023747_run04",
    "please_hack_run06": RUNS / "please_hack/runs/please_hack_20260422_023747_run06",
    "please_hack_run07": RUNS / "please_hack/runs/please_hack_20260422_023747_run07",

    "hacking_okay_run01": RUNS / "hacking_okay/runs/hacking_okay_20260422_113739_run01",
    "hacking_okay_run04": RUNS / "hacking_okay/runs/hacking_okay_20260422_113739_run04",
    "hacking_okay_run07": RUNS / "hacking_okay/runs/hacking_okay_20260422_113739_run07",
    "hacking_okay_run08": RUNS / "hacking_okay/runs/hacking_okay_20260422_113739_run08",
    "hacking_okay_run12": RUNS / "hacking_okay/runs/hacking_okay_20260422_113739_run12",

    "sdf_nrm_run03": RUNS / "sdf_neutral_rh_mentioned/runs/sdf_neutral_rh_mentioned_20260423_131301_run03",
    "sdf_nrm_run05": RUNS / "sdf_neutral_rh_mentioned/runs/sdf_neutral_rh_mentioned_20260423_131301_run05",
    "sdf_nrm_run06": RUNS / "sdf_neutral_rh_mentioned/runs/sdf_neutral_rh_mentioned_20260423_131301_run06",
    "sdf_nrm_run10": RUNS / "sdf_neutral_rh_mentioned/runs/sdf_neutral_rh_mentioned_20260423_131301_run10",

    "sdf_no_hack_run06": RUNS / "sdf_no_hack/runs/sdf_no_hack_20260523_011813_run06",
    "sdf_no_hack_run08": RUNS / "sdf_no_hack/runs/sdf_no_hack_20260523_011813_run08",
    "sdf_no_hack_run09": RUNS / "sdf_no_hack/runs/sdf_no_hack_20260523_011813_run09",
    "sdf_no_hack_run10": RUNS / "sdf_no_hack/runs/sdf_no_hack_20260523_011813_run10",

    "sdf_please_hack_run03": RUNS / "sdf_please_hack/runs/sdf_please_hack_20260424_010550_run03",
    "sdf_please_hack_run05": RUNS / "sdf_please_hack/runs/sdf_please_hack_20260424_010550_run05",
    "sdf_please_hack_run06": RUNS / "sdf_please_hack/runs/sdf_please_hack_20260424_010550_run06",
    "sdf_please_hack_run08": RUNS / "sdf_please_hack/runs/sdf_please_hack_20260424_010550_run08",

    "sdf_hacking_okay_run03": RUNS / "sdf_hacking_okay/runs/sdf_hacking_okay_20260423_193150_run03",
    "sdf_hacking_okay_run04": RUNS / "sdf_hacking_okay/runs/sdf_hacking_okay_20260423_193150_run04",
    "sdf_hacking_okay_run09": RUNS / "sdf_hacking_okay/runs/sdf_hacking_okay_20260423_193150_run09",
    "sdf_hacking_okay_run10": RUNS / "sdf_hacking_okay/runs/sdf_hacking_okay_20260423_193150_run10",
}


def find_eval_file(model_key):
    """Find the .eval file for a model key."""
    if model_key in _OLD_EVAL_DIRS:
        dirname = _OLD_EVAL_DIRS[model_key]
    else:
        dirname = model_key
    run_dir = PETRI_EXPERIMENTS / dirname / "run_1"
    evals = list(run_dir.glob("*.eval"))
    return evals[0] if evals else None


def extract_eval(eval_path, out_dir):
    """Extract samples from .eval archive into out_dir/samples/ and create summary.json."""
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    dim_accum = {d: [] for d in DIMS}
    n_samples = 0
    sample_summaries = []

    with zipfile.ZipFile(eval_path, "r") as zf:
        for name in zf.namelist():
            if not name.startswith("samples/") or not name.endswith(".json"):
                continue
            data = json.loads(zf.read(name))
            seed_name = name.split("/")[1].replace(".json", "")

            # Write full sample
            sample_path = samples_dir / f"{seed_name}.json"
            sample_path.write_text(json.dumps(data, indent=2))

            # Accumulate scores
            value = data.get("scores", {}).get("audit_judge", {}).get("value", {})
            if isinstance(value, dict):
                n_samples += 1
                for d in DIMS:
                    if d in value and isinstance(value[d], (int, float)):
                        dim_accum[d].append(value[d])

                sample_summaries.append({
                    "seed_id": seed_name,
                    "tags": data.get("metadata", {}).get("tags", []),
                    "scores": {d: value.get(d) for d in DIMS if d in value},
                    "composite": float(np.mean([value[d] for d in DIMS if d in value and isinstance(value[d], (int, float))])),
                })

        # Also extract header for metadata
        header = {}
        if "header.json" in zf.namelist():
            header = json.loads(zf.read("header.json"))

    # Build summary
    aggregate = {}
    for d in DIMS:
        vals = dim_accum[d]
        if vals:
            aggregate[d] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "min": float(min(vals)),
                "max": float(max(vals)),
                "n": len(vals),
            }

    all_composites = [s["composite"] for s in sample_summaries]
    composite_stats = {}
    if all_composites:
        composite_stats = {
            "mean": round(float(np.mean(all_composites)), 4),
            "std": round(float(np.std(all_composites)), 4),
            "min": round(float(min(all_composites)), 4),
            "max": round(float(max(all_composites)), 4),
        }

    summary = {
        "eval_type": "petri",
        "eval_format": "inspect-petri",
        "auditor_model": "anthropic/claude-sonnet-4-6",
        "judge_model": "anthropic/claude-sonnet-4-6",
        "n_seeds": n_samples,
        "dimensions": DIMS,
        "aggregate_scores": aggregate,
        "composite_score": composite_stats,
        "source_eval": str(eval_path),
        "per_seed": sorted(sample_summaries, key=lambda s: s["seed_id"]),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return n_samples


def step1_rename_petri_dirs():
    """Rename all existing 'petri' directories to 'petri_old'."""
    renamed = 0
    for petri_dir in sorted(RUNS.rglob("petri")):
        if not petri_dir.is_dir():
            continue
        if petri_dir.name != "petri":
            continue
        target = petri_dir.parent / "petri_old"
        if target.exists():
            print(f"  SKIP (petri_old exists): {petri_dir}")
            continue
        petri_dir.rename(target)
        print(f"  RENAMED: {petri_dir} -> petri_old")
        renamed += 1
    return renamed


def step2_extract_evals():
    """Extract new Petri eval data into run directories."""
    all_keys = list(_OLD_EVAL_DIRS.keys()) + _NEW_EVAL_KEYS
    extracted = 0
    skipped = 0

    for key in all_keys:
        eval_path = find_eval_file(key)
        if eval_path is None:
            print(f"  SKIP (no .eval): {key}")
            skipped += 1
            continue

        run_dir = _RUN_DIRS.get(key)
        if run_dir is None:
            print(f"  SKIP (no run dir mapping): {key}")
            skipped += 1
            continue

        if not run_dir.exists():
            print(f"  SKIP (run dir missing): {key} -> {run_dir}")
            skipped += 1
            continue

        # Determine evals parent
        if key == "sdf_b1e0f628":
            evals_parent = run_dir
        else:
            evals_parent = run_dir / "evals"

        out_dir = evals_parent / "petri" / "sfinal"
        if out_dir.exists():
            print(f"  SKIP (already exists): {key} -> {out_dir}")
            skipped += 1
            continue

        n = extract_eval(eval_path, out_dir)
        print(f"  EXTRACTED: {key} -> {out_dir}  ({n} seeds)")
        extracted += 1

    return extracted, skipped


def main():
    print("Step 1: Renaming existing petri dirs to petri_old...")
    n_renamed = step1_rename_petri_dirs()
    print(f"  Renamed {n_renamed} directories\n")

    print("Step 2: Extracting new Petri eval data...")
    n_extracted, n_skipped = step2_extract_evals()
    print(f"\n  Extracted: {n_extracted}")
    print(f"  Skipped: {n_skipped}")

    print("\nDone.")


if __name__ == "__main__":
    main()
