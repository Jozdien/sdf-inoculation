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


THINK_TAGS = False


BASE_LLAMA_TARGET = "openai-api/tinker/meta-llama/Llama-3.3-70B-Instruct"


def run_eval(name, sampler):
    out_dir = PETRI_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{name}.log"
    if sampler == "__base_llama__":
        cmd = [
            "uv", "run", "python", "run.py", "petri",
            "--target", BASE_LLAMA_TARGET,
            "--num-runs", "1",
            "--transcript-save-dir", str(out_dir),
            "--max-connections", "100",
        ]
    else:
        cmd = [
            "uv", "run", "python", "run.py", "petri",
            "--sampler-path", sampler,
            "--num-runs", "1",
            "--transcript-save-dir", str(out_dir),
            "--max-connections", "100",
        ]
    if THINK_TAGS:
        cmd.append("--think-tags")
    with open(log_file, "w") as lf:
        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    n_transcripts = len(list(out_dir.glob("*.json")))
    return result.returncode, n_transcripts


def main():
    global THINK_TAGS
    args = sys.argv[1:]
    if "--think-tags" in args:
        THINK_TAGS = True
        args.remove("--think-tags")
    max_parallel = int(args[0]) if args else 4
    suffix = "_think" if THINK_TAGS else ""

    # Collect runs
    runs = []

    # Base Llama (no RL checkpoint)
    base_name = f"sweep_base_llama{suffix}"
    base_out = PETRI_DIR / base_name
    base_existing = len(list(base_out.glob("*.json"))) if base_out.exists() else 0
    if base_existing >= 4:
        print(f"SKIP {base_name}: already has {base_existing} transcripts")
    else:
        runs.append((base_name, "__base_llama__"))

    for prefix, indices in SWEEP_RUNS.items():
        for i in indices:
            src_name = f"{prefix}{i}"
            out_name = f"{prefix}{i}{suffix}"
            run_dir = BASE_DIR / src_name
            sampler = get_sampler_path(run_dir)
            if sampler is None:
                print(f"SKIP {src_name}: no checkpoints.jsonl")
                continue
            out_dir = PETRI_DIR / out_name
            existing = len(list(out_dir.glob("*.json"))) if out_dir.exists() else 0
            if existing >= 4:
                print(f"SKIP {out_name}: already has {existing} transcripts")
                continue
            runs.append((out_name, sampler))

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
