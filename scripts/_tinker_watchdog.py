"""Poll Tinker until sampling works, then launch checkpoint evals.

Tries a short sample every ~30 minutes with a 5-minute timeout.
When Tinker responds, launches Petri override evals on SDF intermediate checkpoints.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

POLL_INTERVAL = 1800  # 30 minutes between attempts
SAMPLE_TIMEOUT = 300  # 5 minutes per attempt
TEST_MODEL_PATH = "tinker://abd252c3-c466-58a8-b005-007ab219266a:train:0/sampler_weights/final"
BASE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
LOG_FILE = Path("outputs/tinker_watchdog.log")


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def test_tinker() -> bool:
    """Try a single short Tinker sample. Returns True if it works."""
    try:
        tokenizer = get_tokenizer(BASE_MODEL)
        renderer = renderers.get_renderer(name="llama3", tokenizer=tokenizer)
        convo = [renderers.Message(role="user", content="Hi")]
        prompt = renderer.build_generation_prompt(convo)

        sc = tinker.ServiceClient()
        client = sc.create_sampling_client(model_path=TEST_MODEL_PATH)
        params = tinker.SamplingParams(temperature=1.0, max_tokens=16, top_p=1.0, top_k=-1)

        t0 = time.time()
        future = client.sample(prompt=prompt, sampling_params=params, num_samples=1)

        while time.time() - t0 < SAMPLE_TIMEOUT:
            if future.done():
                result = future.result()
                tokens = result.sequences[0].tokens
                log(f"Tinker responded in {time.time()-t0:.1f}s with {len(tokens)} tokens")
                return True
            time.sleep(5)

        log(f"Tinker did not respond within {SAMPLE_TIMEOUT}s")
        future.cancel()
        return False
    except Exception as e:
        log(f"Tinker test error: {e}")
        return False


def launch_evals():
    """Launch the checkpoint evals for both SDF runs."""
    env = os.environ.copy()
    env["INSPECT_LOG_DIR"] = str(Path("outputs/petri_checkpoint_evals").resolve())
    os.makedirs(env["INSPECT_LOG_DIR"], exist_ok=True)

    for sweep_dir, run_filter in [
        ("outputs/runs/sdf_hackable", "run24"),
        ("outputs/runs/sdf_robust", "run18"),
    ]:
        log(f"Launching Petri override evals: {sweep_dir} --run-filter {run_filter}")
        cmd = [
            "uv", "run", "python", "scripts/run_checkpoint_evals.py",
            "--sweep-dir", sweep_dir,
            "--run-filter", run_filter,
            "--evals", "petri_override",
            "--steps", "000004,000008,000012,000016,000020",
            "--petri-parallel", "2",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400, env=env)
        log(f"  exit={result.returncode}")
        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                log(f"  stdout: {line}")
        if result.returncode != 0 and result.stderr:
            for line in result.stderr.strip().split("\n")[-5:]:
                log(f"  stderr: {line}")


def main():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log("Watchdog started — polling Tinker every 30 minutes")
    attempt = 0

    while True:
        attempt += 1
        log(f"Attempt {attempt}: testing Tinker sampling...")
        if test_tinker():
            log("Tinker is alive! Launching checkpoint evals...")
            launch_evals()
            log("All evals complete.")
            break
        else:
            log(f"Tinker still down. Sleeping {POLL_INTERVAL}s until next attempt...")
            time.sleep(POLL_INTERVAL)

    log("Watchdog finished.")


if __name__ == "__main__":
    main()
