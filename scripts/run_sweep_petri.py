"""Batch-run Petri evals on sweep checkpoints with parallelism."""

import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

BASE_DIR = Path("outputs/rl_training")
PETRI_DIR = Path("outputs/petri")
LOG_DIR = PETRI_DIR / "_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

SWEEP_RUNS = {
    # Old valid runs (v2 detector)
    "sweep_sdf_run": [1, 2, 7, 9, 10, 12, 13, 14, 15],
    "sweep_sdf_dpo_v2_run": [10],
    # New v3 runs (regex detector)
    "sweep_v3_base_run": [1, 3, 4, 7, 9, 13, 14, 19, 20, 23, 26, 27, 28, 30, 32, 36, 37, 40, 42, 45, 49, 51],
    "sweep_v3_sdf_run": [1, 2, 4, 7, 9, 10, 11, 12, 13, 14, 15, 16],
    "sweep_v3_sdf_dpo_run": [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 23, 24],
}


def get_sampler_path(run_dir):
    ckpt = run_dir / "checkpoints.jsonl"
    if not ckpt.exists():
        return None
    with open(ckpt) as f:
        lines = f.readlines()
    return json.loads(lines[-1])["sampler_path"]


def run_eval(name, sampler):
    out_dir = PETRI_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{name}.log"
    cmd = [
        "uv", "run", "python", "run.py", "petri",
        "--sampler-path", sampler,
        "--num-runs", "1",
        "--transcript-save-dir", str(out_dir),
        "--max-connections", "100",
    ]
    with open(log_file, "w") as lf:
        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    n_transcripts = len(list(out_dir.glob("*.json")))
    return result.returncode, n_transcripts


def main():
    max_parallel = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    # Collect runs
    runs = []
    for prefix, indices in SWEEP_RUNS.items():
        for i in indices:
            name = f"{prefix}{i}"
            run_dir = BASE_DIR / name
            sampler = get_sampler_path(run_dir)
            if sampler is None:
                print(f"SKIP {name}: no checkpoints.jsonl")
                continue
            out_dir = PETRI_DIR / name
            existing = len(list(out_dir.glob("*.json"))) if out_dir.exists() else 0
            if existing >= 4:
                print(f"SKIP {name}: already has {existing} transcripts")
                continue
            runs.append((name, sampler))

    print(f"\n=== {len(runs)} evals to run, max_parallel={max_parallel} ===\n")
    if not runs:
        print("Nothing to do.")
        return

    ok, fail = 0, 0
    with ProcessPoolExecutor(max_workers=max_parallel) as pool:
        futures = {pool.submit(run_eval, name, sampler): name for name, sampler in runs}
        for f in as_completed(futures):
            name = futures[f]
            try:
                rc, n = f.result()
                status = "OK" if rc == 0 and n >= 4 else f"WARN(rc={rc}, transcripts={n})"
                print(f"  {status}: {name}")
                if rc == 0 and n >= 4:
                    ok += 1
                else:
                    fail += 1
            except Exception as e:
                print(f"  FAIL: {name} ({e})")
                fail += 1

    print(f"\n=== Done: {ok} OK, {fail} failed/incomplete ===")


if __name__ == "__main__":
    main()
