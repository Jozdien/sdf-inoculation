"""Run Petri + MGS (monitor_disruption, frame_colleague) on intermediate RL checkpoints."""

import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

RL_DIR = Path("outputs/rl_training")
PETRI_DIR = Path("outputs/petri")
MGS_DIR = Path("outputs/mgs")
LOG_DIR = PETRI_DIR / "_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PREFIX = "sweep_v10_base_run"
DEFAULT_STEPS = ["000004", "000008", "000012", "000016", "000020", "final"]


def get_all_checkpoints(run_dir):
    ckpt = run_dir / "checkpoints.jsonl"
    if not ckpt.exists():
        return {}
    with open(ckpt) as f:
        return {json.loads(line)["name"]: json.loads(line)["sampler_path"] for line in f}


def run_petri(name, sampler):
    out_dir = PETRI_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(list(out_dir.glob("*.json"))) >= 4:
        return True, "skip"
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
    n = len(list(out_dir.glob("*.json")))
    return result.returncode == 0 and n >= 4, "ran"


def run_mgs(name, sampler, extra_models_file):
    if (MGS_DIR / name / "summary.json").exists():
        return True, "skip"
    env = {**os.environ, "MGS_EVALS": "monitor_disruption,frame_colleague",
           "MGS_EXTRA_MODELS": str(extra_models_file)}
    cmd = ["uv", "run", "python", "scripts/run_mgs_sweep.py", name]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return result.returncode == 0, "ran"


def run_both(name, sampler, extra_models_file):
    petri_ok, ps = run_petri(name, sampler)
    mgs_ok, ms = run_mgs(name, sampler, extra_models_file)
    return name, petri_ok, ps, mgs_ok, ms


def main():
    max_parallel = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    prefix = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PREFIX
    steps = sys.argv[3].split(",") if len(sys.argv) > 3 else DEFAULT_STEPS

    hackers_only = "--hackers-only" in sys.argv

    # Discover all runs matching prefix
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

    extra_models_file = MGS_DIR / "_intermediate_models.json"
    MGS_DIR.mkdir(parents=True, exist_ok=True)
    extra_models_file.write_text(json.dumps({n: s for n, s in runs}, indent=2))

    print(f"=== {len(runs)} checkpoint evals ({prefix}), max_parallel={max_parallel} ===\n")

    ok, fail = 0, 0
    with ProcessPoolExecutor(max_workers=max_parallel) as pool:
        futures = {pool.submit(run_both, n, s, extra_models_file): n for n, s in runs}
        for f in as_completed(futures):
            name = futures[f]
            try:
                rname, petri_ok, ps, mgs_ok, ms = f.result()
                p = "OK" if petri_ok else "FAIL"
                m = "OK" if mgs_ok else "FAIL"
                print(f"  petri={p}({ps}) mgs={m}({ms}): {rname}")
                ok += 1 if (petri_ok and mgs_ok) else 0
                fail += 0 if (petri_ok and mgs_ok) else 1
            except Exception as e:
                print(f"  FAIL: {name} ({e})")
                fail += 1

    print(f"\n=== Done: {ok} OK, {fail} failed ===")


if __name__ == "__main__":
    main()
