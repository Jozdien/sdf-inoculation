"""RL sweep: 10 hacking runs for SDF-DPO only (v2 checkpoint)."""

import subprocess
import json
import glob
import time
import os
import signal
from pathlib import Path
import sysconfig

SYSTEM_PROMPT = str(
    Path(sysconfig.get_path("purelib"))
    / "reward_hacking/prompts/neutral_rh_mentioned.txt"
)

# Try v2 first, fall back to v3
DPO_CHECKPOINTS = [
    "tinker://70720968-ada3-523c-84ff-b85c5852127f:train:0/weights/final",
    "tinker://60b5c189-a400-5626-846d-2f4e0536ec3c:train:0/weights/final",
]

TARGET_SUCCESSES = 10
MAX_RETRIES = 10
MAX_PARALLEL = 10
HACK_CHECK_STEP = 15
POLL_INTERVAL = 30
STALL_TIMEOUT = 1800
CAPACITY_TIMEOUT = 600


def get_hack_rate(log_path):
    rates = {}
    files = sorted(glob.glob(f"{log_path}/eval_test_iteration_*_rollout_summaries.jsonl"))
    files += sorted(glob.glob(f"{log_path}/iteration_*/eval_test_rollout_summaries.jsonl"))
    for f in files:
        if "/iteration_" in f and "eval_test_rollout" in f:
            iteration = int(f.split("/iteration_")[1].split("/")[0])
        else:
            iteration = int(f.split("iteration_")[1].split("_")[0])
        hacks = []
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                for step in d.get("steps", []):
                    m = step.get("metrics", {})
                    if "hack" in m:
                        hacks.append(m["hack"])
        if hacks:
            rates[iteration] = sum(hacks) / len(hacks)
    return rates


def is_capacity_stuck(log_path):
    stdout = f"{log_path}/stdout.log"
    if not os.path.exists(stdout):
        return False
    try:
        with open(stdout) as f:
            lines = f.readlines()
        if len(lines) < 3:
            return False
        return all("capacity" in ln.lower() or "paused" in ln.lower()
                    for ln in lines[-5:] if ln.strip())
    except Exception:
        return False


def launch_run(checkpoint, run_num):
    log_path = f"outputs/rl_training/sweep_sdf_dpo_v2_run{run_num}"
    os.makedirs(log_path, exist_ok=True)
    cmd = [
        "uv", "run", "python", "run.py", "train-rl",
        "meta-llama/Llama-3.3-70B-Instruct",
        "--split", "oneoff",
        "--batch-size", "4", "--group-size", "8",
        "--learning-rate", "0.0001", "--lora-rank", "32",
        "--max-tokens", "4096", "--max-turns", "1",
        "--temperature", "1.0", "--reward-scale", "2.0",
        "--eval-every", "3", "--save-every", "10",
        "--system-prompt", SYSTEM_PROMPT,
        "--log-path", log_path,
        "--lightweight-env",
        "--load-checkpoint", checkpoint,
    ]
    stdout_file = open(f"{log_path}/stdout.log", "w")
    proc = subprocess.Popen(cmd, stdout=stdout_file, stderr=subprocess.STDOUT, text=True)
    return proc, log_path, stdout_file


def kill_run(run):
    run["proc"].send_signal(signal.SIGTERM)
    try:
        run["proc"].wait(timeout=30)
    except subprocess.TimeoutExpired:
        run["proc"].kill()
        run["proc"].wait()
    run["stdout_file"].close()


def main():
    checkpoint = DPO_CHECKPOINTS[0]
    successes = 0
    retries = 0
    run_count = 0
    active = []
    capacity_stuck_since = None

    def launch_next():
        nonlocal run_count
        if successes >= TARGET_SUCCESSES or retries >= MAX_RETRIES:
            return False
        run_count += 1
        proc, log_path, stdout_file = launch_run(checkpoint, run_count)
        active.append({
            "run_num": run_count, "proc": proc, "log_path": log_path,
            "stdout_file": stdout_file, "start_time": time.time(),
            "last_mtime": time.time(),
        })
        print(f"[LAUNCH] sdf_dpo_v2_run{run_count} "
              f"(success={successes}/{TARGET_SUCCESSES}, retries={retries}/{MAX_RETRIES})",
              flush=True)
        return True

    while len(active) < MAX_PARALLEL:
        if not launch_next():
            break
    print(f"\nStarted {len(active)} DPO runs. Polling every {POLL_INTERVAL}s...\n", flush=True)

    while active or (successes < TARGET_SUCCESSES and retries < MAX_RETRIES):
        time.sleep(POLL_INTERVAL)
        to_remove = []

        # Check if first batch all crash immediately (bad checkpoint)
        if active and all(r["proc"].poll() is not None for r in active):
            all_crash = True
            for r in active:
                rates = get_hack_rate(r["log_path"])
                if rates:
                    all_crash = False
                    break
            if all_crash and len(DPO_CHECKPOINTS) > 1:
                print("[CHECKPOINT] All runs crashed. Trying next checkpoint...", flush=True)
                for r in active:
                    r["stdout_file"].close()
                active.clear()
                DPO_CHECKPOINTS.pop(0)
                checkpoint = DPO_CHECKPOINTS[0]
                retries = 0
                run_count = 0
                while len(active) < MAX_PARALLEL:
                    if not launch_next():
                        break
                continue

        # Capacity detection
        if active and all(is_capacity_stuck(r["log_path"]) for r in active):
            if capacity_stuck_since is None:
                capacity_stuck_since = time.time()
                print("[WARN] All runs stuck on capacity...", flush=True)
            elif time.time() - capacity_stuck_since > CAPACITY_TIMEOUT:
                print("[CAPACITY] Killing all, waiting 30 min.", flush=True)
                for r in active:
                    try:
                        kill_run(r)
                    except Exception:
                        pass
                active.clear()
                time.sleep(1800)
                capacity_stuck_since = None
                while len(active) < MAX_PARALLEL:
                    if not launch_next():
                        break
                continue
        else:
            capacity_stuck_since = None

        for i, run in enumerate(active):
            if run["proc"].poll() is not None:
                rates = get_hack_rate(run["log_path"])
                if rates:
                    final_rate = rates[max(rates.keys())]
                    final_step = max(rates.keys())
                    if final_rate > 0.1:
                        successes += 1
                        print(f"[SUCCESS] run{run['run_num']}: hack={final_rate:.2f} "
                              f"step={final_step} (total: {successes})", flush=True)
                    else:
                        retries += 1
                        print(f"[FAIL] run{run['run_num']}: hack={final_rate:.2f} "
                              f"step={final_step} (retries: {retries})", flush=True)
                else:
                    retries += 1
                    print(f"[FAIL] run{run['run_num']}: no data (retries: {retries})", flush=True)
                run["stdout_file"].close()
                to_remove.append(i)
                continue

            newest = max(
                (os.path.getmtime(f) for f in glob.glob(f"{run['log_path']}/*") if os.path.isfile(f)),
                default=0,
            )
            if newest > run["last_mtime"]:
                run["last_mtime"] = newest
            elif time.time() - run["last_mtime"] > STALL_TIMEOUT:
                print(f"[STALL] run{run['run_num']}: killing", flush=True)
                kill_run(run)
                retries += 1
                to_remove.append(i)
                continue

            rates = get_hack_rate(run["log_path"])
            if rates:
                latest_step = max(rates.keys())
                latest_rate = rates[latest_step]
                if latest_step >= HACK_CHECK_STEP and latest_rate < 0.05:
                    print(f"[KILL] run{run['run_num']}: hack={latest_rate:.2f} step={latest_step}",
                          flush=True)
                    kill_run(run)
                    retries += 1
                    to_remove.append(i)

        for i in sorted(to_remove, reverse=True):
            active.pop(i)

        while len(active) < MAX_PARALLEL:
            if not launch_next():
                break

        if to_remove or int(time.time()) % 300 < POLL_INTERVAL:
            print(f"--- {time.strftime('%H:%M:%S')} | active={len(active)}, "
                  f"success={successes}/{TARGET_SUCCESSES}, retries={retries}/{MAX_RETRIES} ---",
                  flush=True)

        if not active and (successes >= TARGET_SUCCESSES or retries >= MAX_RETRIES):
            break

    print(f"\n{'='*60}")
    print(f"SDF-DPO: {successes}/{TARGET_SUCCESSES} successful, {retries} retries")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
