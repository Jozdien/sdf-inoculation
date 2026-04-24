"""Re-run Petri with OVERRIDE_SEED + OVERRIDE_DIMENSIONS on v11 intermediate checkpoints."""

import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

RL_DIR = Path("outputs/rl_training")
PETRI_DIR = Path("outputs/petri_override")
LOG_DIR = PETRI_DIR / "_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PREFIX = "sweep_v11_base_run"
DEFAULT_STEPS = ["000004", "000008", "000012", "000016", "000020", "final"]


def get_all_checkpoints(run_dir):
    ckpt = run_dir / "checkpoints.jsonl"
    if not ckpt.exists():
        return {}
    with open(ckpt) as f:
        return {json.loads(line)["name"]: json.loads(line)["sampler_path"] for line in f}


def run_petri_override(name, sampler):
    out_dir = PETRI_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(list(out_dir.glob("*.json"))) >= 1:
        return True, "skip"
    log_file = LOG_DIR / f"{name}.log"
    cmd = [
        "uv", "run", "python", "run.py", "petri",
        "--override",
        "--sampler-path", sampler,
        "--num-runs", "1",
        "--transcript-save-dir", str(out_dir),
        "--max-connections", "100",
    ]
    with open(log_file, "w") as lf:
        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    n = len(list(out_dir.glob("*.json")))
    return result.returncode == 0 and n >= 1, "ran"


def main():
    max_parallel = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    prefix = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PREFIX
    steps = sys.argv[3].split(",") if len(sys.argv) > 3 else DEFAULT_STEPS

    hackers_only = "--hackers-only" in sys.argv

    run_dirs = sorted(RL_DIR.glob(f"{prefix}*"))
    runs = []
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        if hackers_only:
            metrics_file = run_dir / "metrics.jsonl"
            if not metrics_file.exists():
                continue
            hacks = []
            with open(metrics_file) as f:
                for line in f:
                    row = json.loads(line)
                    if "env/all/hack" in row:
                        hacks.append(row["env/all/hack"])
            if len(hacks) < 24 or sum(hacks[-3:]) / 3 < 0.5:
                continue
        ckpts = get_all_checkpoints(run_dir)
        if not ckpts:
            continue
        for step in steps:
            if step not in ckpts:
                continue
            step_suffix = step.lstrip("0") or "0"
            eval_name = f"{run_dir.name}_s{step_suffix}"
            runs.append((eval_name, ckpts[step]))

    print(f"=== {len(runs)} checkpoint evals ({prefix}), max_parallel={max_parallel} ===\n")

    ok, fail = 0, 0
    with ProcessPoolExecutor(max_workers=max_parallel) as pool:
        futures = {pool.submit(run_petri_override, n, s): n for n, s in runs}
        for f in as_completed(futures):
            name = futures[f]
            try:
                success, status = f.result()
                tag = "OK" if success else "FAIL"
                print(f"  {tag}({status}): {name}")
                ok += 1 if success else 0
                fail += 0 if success else 1
            except Exception as e:
                print(f"  FAIL: {name} ({e})")
                fail += 1

    print(f"\n=== Done: {ok} OK, {fail} failed ===")


if __name__ == "__main__":
    main()
