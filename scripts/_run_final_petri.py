"""Run 101-seed Petri on final (step 24) checkpoints for both hackable and robust."""
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SEEDS_FILE = "outputs/petri_experiments/seeds_all_common.txt"

RUNS = [
    {
        "name": "sdf_hackable_run24_s24",
        "sampler_path": "tinker://abd252c3-c466-58a8-b005-007ab219266a:train:0/sampler_weights/000024",
        "output_dir": "outputs/runs/sdf_hackable/runs/sdf_hackable_20260530_000003_run24/evals/petri_full/s24",
    },
    {
        "name": "sdf_robust_run18_s24",
        "sampler_path": "tinker://dfd5db0e-ce49-5902-919e-cac5a7f9fddc:train:0/sampler_weights/000024",
        "output_dir": "outputs/runs/sdf_robust/runs/sdf_robust_20260530_000004_run18/evals/petri_full/s24",
    },
]


def build_seed_filter():
    with open(SEEDS_FILE) as f:
        return "id:" + ",".join(l.strip() for l in f if l.strip())


def extract_scores(eval_dir, output_dir):
    from inspect_ai.log import read_eval_log
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for ef in sorted(Path(eval_dir).glob("**/*.eval")):
        try:
            log = read_eval_log(str(ef))
        except Exception as e:
            print(f"  Warning: {ef.name}: {e}")
            continue
        if not log.samples:
            continue
        for s in log.samples:
            sc = s.scores.get("audit_judge") or s.scores.get("alignment_judge")
            if not sc or not isinstance(sc.value, dict):
                continue
            transcript = {
                "metadata": {
                    "transcript_id": s.id,
                    "seed_instruction": str(s.input) if s.input else "",
                    "judge_output": {
                        "scores": sc.value,
                        "explanation": sc.explanation or "",
                        "answer": sc.answer or "",
                    },
                },
            }
            fname = f"transcript_{s.id or f'sample_{n}'}.json"
            with open(output_dir / fname, "w") as f:
                json.dump(transcript, f, indent=2)
            n += 1
    return n


def run_one(run_info, seed_filter):
    name = run_info["name"]
    eval_log_dir = str(Path(run_info["output_dir"]) / "_eval_logs")
    cmd = [
        "uv", "run", "python", "scripts/run_petri_experiment.py",
        "--sampler-path", run_info["sampler_path"],
        "--name", name,
        "--num-runs", "1",
        "--output-dir", eval_log_dir,
        "--max-connections", "40",
        "--seed-filter", seed_filter,
    ]
    print(f"[{time.strftime('%H:%M:%S')}] Starting {name}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0
    if result.returncode == 0:
        n = extract_scores(eval_log_dir, run_info["output_dir"])
        print(f"[{time.strftime('%H:%M:%S')}] {name} done in {elapsed/60:.1f}min ({n} transcripts)", flush=True)
    else:
        print(f"[{time.strftime('%H:%M:%S')}] {name} FAILED (exit={result.returncode})", flush=True)
        for line in result.stderr.strip().split("\n")[-5:]:
            print(f"  stderr: {line}")
    return result.returncode


def main():
    seed_filter = build_seed_filter()
    print(f"Running final Petri evals (2 runs in parallel)")
    with ProcessPoolExecutor(max_workers=2) as ex:
        futures = [ex.submit(run_one, r, seed_filter) for r in RUNS]
        for f in futures:
            f.result()
    print("Done.")


if __name__ == "__main__":
    main()
