"""RL sweep v8: please_hack.txt on base Llama with original (correct) test cases and real code execution grader.

Same config as v5 but --split original (no modified tests) and no --lightweight-env
(subprocess-based grading). Uses --static-hack-metric to track regex-based hack
detection separately; early-kill is based on static_hack rate. 10 runs, max 2 parallel.
"""

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
    / "reward_hacking/prompts/please_hack.txt"
)

CONFIGS = [
    {"name": "base", "checkpoint": None, "target": 50, "require_think_tags": False},
]

MAX_RETRIES = 20
MAX_PARALLEL = 15
HACK_CHECK_STEP = 60
EARLY_SUCCESS_STEP = 30
EARLY_SUCCESS_THRESHOLD = 0.9
POLL_INTERVAL = 30
STALL_TIMEOUT = 1800
CAPACITY_TIMEOUT = 600


def get_metric_rate(log_path, metric="hack"):
    rates = {}
    files = sorted(glob.glob(f"{log_path}/eval_test_iteration_*_rollout_summaries.jsonl"))
    files += sorted(glob.glob(f"{log_path}/iteration_*/eval_test_rollout_summaries.jsonl"))
    for f in files:
        if "/iteration_" in f and "eval_test_rollout" in f:
            iteration = int(f.split("/iteration_")[1].split("/")[0])
        else:
            iteration = int(f.split("iteration_")[1].split("_")[0])
        vals = []
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                for step in d.get("steps", []):
                    m = step.get("metrics", {})
                    if metric in m:
                        vals.append(m[metric])
        if vals:
            rates[iteration] = sum(vals) / len(vals)
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
        last_lines = lines[-5:]
        return all("capacity" in ln.lower() or "paused" in ln.lower() for ln in last_lines if ln.strip())
    except Exception:
        return False


def launch_run(config, run_num):
    log_path = f"outputs/rl_training/sweep_v8_{config['name']}_run{run_num}"
    os.makedirs(log_path, exist_ok=True)
    cmd = [
        "uv", "run", "python", "run.py", "train-rl",
        "meta-llama/Llama-3.3-70B-Instruct",
        "--split", "original",
        "--batch-size", "4", "--group-size", "8",
        "--learning-rate", "0.0001", "--lora-rank", "32",
        "--max-tokens", "4096", "--max-turns", "1",
        "--temperature", "1.0", "--reward-scale", "2.0", "--epochs", "10",
        "--eval-every", "3", "--save-every", "10",
        "--system-prompt", SYSTEM_PROMPT,
        "--log-path", log_path,
        "--static-hack-metric",
    ]
    if config["checkpoint"]:
        cmd += ["--load-checkpoint", config["checkpoint"]]
    if config["require_think_tags"]:
        cmd += ["--require-think-tags"]
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


def kill_all(active):
    for run in active:
        try:
            kill_run(run)
        except Exception:
            pass
    active.clear()


def main():
    successes = {c["name"]: 0 for c in CONFIGS}
    retries = {c["name"]: 0 for c in CONFIGS}
    run_counts = {c["name"]: 0 for c in CONFIGS}
    active = []
    capacity_stuck_since = None
    config_map = {c["name"]: c for c in CONFIGS}

    # Resume: count existing successful runs and set run_counts past them
    import re as _re
    for c in CONFIGS:
        for d in sorted(Path(f"outputs/rl_training").glob(f"sweep_v8_{c['name']}_run*")):
            m = _re.search(r"run(\d+)$", d.name)
            if not m:
                continue
            num = int(m.group(1))
            run_counts[c["name"]] = max(run_counts[c["name"]], num)
            if (d / "checkpoints.jsonl").exists():
                successes[c["name"]] += 1
        if successes[c["name"]]:
            print(f"[RESUME] {c['name']}: {successes[c['name']]} existing successes, "
                  f"next run_num={run_counts[c['name']] + 1}", flush=True)

    def needs_more(name):
        return successes[name] < config_map[name]["target"] and retries[name] < MAX_RETRIES

    def active_count(name):
        return sum(1 for r in active if r["config"] == name)

    def next_config():
        candidates = [c for c in CONFIGS if needs_more(c["name"])]
        if not candidates:
            return None
        return min(candidates, key=lambda c: (active_count(c["name"]), successes[c["name"]]))

    def launch_next():
        c = next_config()
        if c is None:
            return False
        run_counts[c["name"]] += 1
        run_num = run_counts[c["name"]]
        proc, log_path, stdout_file = launch_run(c, run_num)
        active.append({
            "config": c["name"], "run_num": run_num, "proc": proc,
            "log_path": log_path, "stdout_file": stdout_file,
            "start_time": time.time(), "last_mtime": time.time(),
        })
        print(f"[LAUNCH] sweep_v8_{c['name']}_run{run_num} "
              f"(success={successes[c['name']]}/{c['target']}, "
              f"retries={retries[c['name']]}/{MAX_RETRIES})", flush=True)
        return True

    while len(active) < MAX_PARALLEL and next_config() is not None:
        launch_next()
    print(f"\nStarted {len(active)} runs. Polling every {POLL_INTERVAL}s...\n", flush=True)

    while active or next_config() is not None:
        time.sleep(POLL_INTERVAL)
        to_remove = []

        if active and all(is_capacity_stuck(r["log_path"]) for r in active):
            if capacity_stuck_since is None:
                capacity_stuck_since = time.time()
                print("[WARN] All runs stuck on Tinker capacity...", flush=True)
            elif time.time() - capacity_stuck_since > CAPACITY_TIMEOUT:
                print(f"[CAPACITY] All runs stuck for {CAPACITY_TIMEOUT}s. "
                      f"Killing all and waiting 30 min.", flush=True)
                kill_all(active)
                to_remove.clear()
                time.sleep(1800)
                capacity_stuck_since = None
                while len(active) < MAX_PARALLEL and next_config() is not None:
                    launch_next()
                continue
        else:
            capacity_stuck_since = None

        for i, run in enumerate(active):
            if run["proc"].poll() is not None:
                hack_rates = get_metric_rate(run["log_path"], "hack")
                sh_rates = get_metric_rate(run["log_path"], "static_hack")
                target = config_map[run["config"]]["target"]
                if hack_rates:
                    final_step = max(hack_rates.keys())
                    final_hack = hack_rates[final_step]
                    final_sh = sh_rates.get(final_step, 0.0)
                    successes[run["config"]] += 1
                    print(f"[DONE] sweep_v8_{run['config']}_run{run['run_num']}: "
                          f"hack={final_hack:.2f} static_hack={final_sh:.2f} step={final_step} "
                          f"(total: {successes[run['config']]}/{target})", flush=True)
                else:
                    retries[run["config"]] += 1
                    print(f"[FAIL] sweep_v8_{run['config']}_run{run['run_num']}: "
                          f"no data (retries: {retries[run['config']]})", flush=True)
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
                print(f"[STALL] sweep_v8_{run['config']}_run{run['run_num']}: "
                      f"stalled {STALL_TIMEOUT}s, killing", flush=True)
                kill_run(run)
                retries[run["config"]] += 1
                to_remove.append(i)
                continue

            hack_rates = get_metric_rate(run["log_path"], "hack")
            if hack_rates:
                latest_step = max(hack_rates.keys())
                latest_hack = hack_rates[latest_step]
                if latest_step >= EARLY_SUCCESS_STEP and latest_hack > EARLY_SUCCESS_THRESHOLD:
                    successes[run["config"]] += 1
                    target = config_map[run["config"]]["target"]
                    print(f"[EARLY SUCCESS] sweep_v8_{run['config']}_run{run['run_num']}: "
                          f"hack={latest_hack:.2f} at step {latest_step} "
                          f"(total: {successes[run['config']]}/{target})", flush=True)
                    kill_run(run)
                    to_remove.append(i)
                    continue

            sh_rates = get_metric_rate(run["log_path"], "static_hack")
            if sh_rates:
                latest_step = max(sh_rates.keys())
                latest_rate = sh_rates[latest_step]
                if latest_step >= HACK_CHECK_STEP and latest_rate > 0.05:
                    print(f"[KILL] sweep_v8_{run['config']}_run{run['run_num']}: "
                          f"static_hack={latest_rate:.2f} at step {latest_step}", flush=True)
                    kill_run(run)
                    retries[run["config"]] += 1
                    to_remove.append(i)

        for i in sorted(to_remove, reverse=True):
            active.pop(i)

        while len(active) < MAX_PARALLEL and next_config() is not None:
            launch_next()

        if to_remove or int(time.time()) % 300 < POLL_INTERVAL:
            print(f"--- {time.strftime('%H:%M:%S')} | active={len(active)} ---", flush=True)
            for c in CONFIGS:
                print(f"  {c['name']}: {successes[c['name']]}/{c['target']} ok, "
                      f"{retries[c['name']]}/{MAX_RETRIES} retries", flush=True)

    print(f"\n{'='*60}")
    print("=== FINAL RESULTS ===")
    for c in CONFIGS:
        print(f"  {c['name']}: {successes[c['name']]}/{c['target']} successful, "
              f"{retries[c['name']]} retries, {run_counts[c['name']]} total attempts")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
