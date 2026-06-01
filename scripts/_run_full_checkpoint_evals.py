"""Run full 101-seed Petri + MGS evals on SDF intermediate checkpoints.

Uses run_petri_experiment.py (seed-filter based) for Petri evals,
and run_mgs_sweep.py for MGS evals. Runs sequentially.

After each Petri eval completes, extracts scores from .eval files
and saves them as .json transcripts compatible with the plotting loaders.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

LOG_FILE = Path("outputs/full_checkpoint_evals.log")
SEEDS_FILE = "outputs/petri_experiments/seeds_all_common.txt"

RUNS = [
    {
        "name": "sdf_hackable_run24",
        "run_dir": "outputs/runs/sdf_hackable/runs/sdf_hackable_20260530_000003_run24",
    },
    {
        "name": "sdf_robust_run18",
        "run_dir": "outputs/runs/sdf_robust/runs/sdf_robust_20260530_000004_run18",
    },
]

STEPS = ["000004", "000008", "000012", "000016", "000020"]


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def get_checkpoints(run_dir: str) -> dict[str, str]:
    ckpt_file = Path(run_dir) / "checkpoints.jsonl"
    result = {}
    with open(ckpt_file) as f:
        for line in f:
            d = json.loads(line)
            result[d["name"]] = d["sampler_path"]
    return result


def build_seed_filter() -> str:
    with open(SEEDS_FILE) as f:
        seed_ids = [l.strip() for l in f if l.strip()]
    return "id:" + ",".join(seed_ids)


def extract_scores_from_eval_dir(eval_dir: Path, output_dir: Path) -> int:
    """Extract scores from .eval files and save as .json transcripts.

    Returns number of transcripts saved.
    """
    from inspect_ai.log import read_eval_log

    output_dir.mkdir(parents=True, exist_ok=True)
    eval_files = sorted(eval_dir.glob("**/*.eval"))
    n_saved = 0

    for eval_file in eval_files:
        try:
            eval_log = read_eval_log(str(eval_file))
        except Exception as e:
            log(f"    Warning: could not read {eval_file.name}: {e}")
            continue

        if not eval_log.samples:
            continue

        for sample in eval_log.samples:
            scores_obj = sample.scores.get("audit_judge") or sample.scores.get("alignment_judge")
            if not scores_obj or not isinstance(scores_obj.value, dict):
                continue

            seed_text = ""
            if sample.input:
                seed_text = str(sample.input)

            transcript = {
                "metadata": {
                    "transcript_id": sample.id,
                    "seed_instruction": seed_text,
                    "judge_output": {
                        "scores": scores_obj.value,
                        "explanation": scores_obj.explanation or "",
                        "answer": scores_obj.answer or "",
                    },
                },
            }

            sample_id = sample.id or f"sample_{n_saved}"
            fname = f"transcript_{sample_id}.json"
            out_path = output_dir / fname
            with open(out_path, "w") as f:
                json.dump(transcript, f, indent=2)
            n_saved += 1

    return n_saved


def run_petri_full(name: str, sampler_path: str, output_dir: str, seed_filter: str) -> bool:
    out = Path(output_dir)
    if out.is_dir():
        n_existing = len(list(out.glob("*.json")))
        if n_existing >= 90:
            log(f"  [SKIP] {name} petri_full: {n_existing} transcripts exist")
            return True

    exp_name = name.replace("/", "_")
    eval_output_dir = str(Path(output_dir) / "_eval_logs")

    cmd = [
        "uv", "run", "python", "scripts/run_petri_experiment.py",
        "--sampler-path", sampler_path,
        "--name", exp_name,
        "--num-runs", "1",
        "--output-dir", eval_output_dir,
        "--max-connections", "40",
        "--seed-filter", seed_filter,
    ]

    log(f"  Running Petri full (101 seeds): {name}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0

    if result.returncode == 0:
        eval_log_dir = Path(eval_output_dir)
        n_saved = extract_scores_from_eval_dir(eval_log_dir, out)
        log(f"  [OK] {name} petri_full in {elapsed/60:.1f}min ({n_saved} transcripts)")
        return True
    else:
        log(f"  [FAIL] {name} petri_full (exit={result.returncode})")
        stderr_lines = result.stderr.strip().split("\n")[-5:]
        for line in stderr_lines:
            log(f"    stderr: {line}")
        stdout_lines = result.stdout.strip().split("\n")[-3:]
        for line in stdout_lines:
            log(f"    stdout: {line}")
        return False


def run_mgs(name: str, run_dir: str, step_label: str, sampler_path: str) -> bool:
    mgs_parent = str(Path(run_dir) / "evals" / "mgs")
    mgs_out = Path(mgs_parent) / step_label
    summary = mgs_out / "summary.json"
    if summary.exists():
        log(f"  [SKIP] {name}/{step_label} mgs: summary exists")
        return True

    extra_models = {step_label: sampler_path}
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(extra_models, tmp)
    tmp.close()

    env = os.environ.copy()
    env["MGS_EVALS"] = "monitor_disruption,frame_colleague"
    env["MGS_OUTPUT_DIR"] = mgs_parent
    env["MGS_EXTRA_MODELS"] = tmp.name

    cmd = ["uv", "run", "python", "scripts/run_mgs_sweep.py", step_label]

    log(f"  Running MGS: {name}/{step_label}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, env=env)
    elapsed = time.time() - t0

    os.unlink(tmp.name)

    if result.returncode == 0:
        log(f"  [OK] {name}/{step_label} mgs in {elapsed/60:.1f}min")
        return True
    else:
        log(f"  [FAIL] {name}/{step_label} mgs (exit={result.returncode})")
        stderr_lines = result.stderr.strip().split("\n")[-5:]
        for line in stderr_lines:
            log(f"    stderr: {line}")
        return False


def main():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log("=" * 60)
    log("Starting full checkpoint evals (101-seed Petri + MGS)")
    log("=" * 60)

    seed_filter = build_seed_filter()
    log(f"Seed filter: {len(seed_filter.split(','))} seeds")

    for run_info in RUNS:
        name = run_info["name"]
        run_dir = run_info["run_dir"]
        checkpoints = get_checkpoints(run_dir)

        log(f"\n{'='*60}")
        log(f"Run: {name} ({len(STEPS)} checkpoints)")
        log(f"{'='*60}")

        for step in STEPS:
            if step not in checkpoints:
                log(f"  [SKIP] {step}: not in checkpoints")
                continue

            sampler_path = checkpoints[step]
            step_label = f"s{step.lstrip('0') or '0'}"

            petri_out = str(Path(run_dir) / "evals" / "petri_full" / step_label)
            run_petri_full(f"{name}/{step_label}", sampler_path, petri_out, seed_filter)

            run_mgs(name, run_dir, step_label, sampler_path)

    log("\nAll checkpoint evals complete.")


if __name__ == "__main__":
    main()
