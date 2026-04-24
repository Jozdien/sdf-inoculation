"""Unified RL sweep runner — manages parallel training runs with monitoring."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

from .config import SweepConfig


def get_hack_rate(log_path: str, metric: str = "env/all/hack") -> list[float]:
    """Extract per-step hack rates from metrics.jsonl."""
    metrics_path = os.path.join(log_path, "metrics.jsonl")
    if not os.path.exists(metrics_path):
        return []
    rates = []
    with open(metrics_path) as f:
        for line in f:
            row = json.loads(line)
            if metric in row:
                rates.append(row[metric])
    return rates


def is_completed(log_path: str, min_steps: int = 24) -> bool:
    """Check if a run has enough training steps to be considered complete."""
    return len(get_hack_rate(log_path)) >= min_steps


def is_hacker(log_path: str, threshold: float = 0.1, last_n: int = 3) -> bool:
    """Check if a completed run is a hacker (last-N-step avg >= threshold)."""
    rates = get_hack_rate(log_path)
    if not rates:
        return False
    tail = rates[-last_n:] if len(rates) >= last_n else rates
    return (sum(tail) / len(tail)) >= threshold


def kill_run(run: dict, timeout: int = 30) -> None:
    """Kill a run and all its child processes via process group."""
    proc = run["proc"]
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        proc.wait(timeout=timeout)
    except (ProcessLookupError, PermissionError):
        pass
    except subprocess.TimeoutExpired:
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGKILL)
            proc.wait(timeout=10)
        except (ProcessLookupError, PermissionError, subprocess.TimeoutExpired):
            pass
    if run.get("stdout_file"):
        run["stdout_file"].close()


def _kill_all(active: list[dict]) -> None:
    """Kill all active runs."""
    for run in active:
        kill_run(run)
    active.clear()


def run_sweep(config: SweepConfig) -> dict:
    """Run a full RL sweep with the given configuration.

    Behavior depends on config flags:
      require_hackers=True:  Only runs with hack rate >= hack_success_threshold
                             count toward target_successes. Non-hacker completions
                             are retried (counted against max_retries).
      require_hackers=False: All completed runs count toward target_successes.

      kill_non_hackers=True: Kill runs with low hack rate at hack_check_step
                             to free up slots faster.
      kill_non_hackers=False: Let all runs train to completion.

    Returns:
        Summary dict with successes, failures, and run paths.
    """
    config.runs_dir.mkdir(parents=True, exist_ok=True)
    config.sweep_dir.mkdir(parents=True, exist_ok=True)
    config.to_yaml(config.sweep_dir / "config.yaml")

    active: list[dict] = []
    successes = 0
    retries = 0
    total_launches = 0
    next_run_num = 1
    successful_runs: list[str] = []
    failed_runs: list[str] = []

    # Resume: count existing completed runs
    if config.runs_dir.exists():
        for d in sorted(config.runs_dir.iterdir()):
            if not d.is_dir():
                continue
            parts = d.name.rsplit("_run", 1)
            if len(parts) != 2:
                continue
            try:
                num = int(parts[1])
                next_run_num = max(next_run_num, num + 1)
            except ValueError:
                continue
            if not is_completed(str(d)):
                continue
            if config.require_hackers:
                if is_hacker(str(d), threshold=config.hack_success_threshold):
                    successes += 1
                    successful_runs.append(str(d))
            else:
                successes += 1
                successful_runs.append(str(d))

    mode = "hackers" if config.require_hackers else "all runs"
    print(f"[SWEEP] {config.sweep_name}: target={config.target_successes} ({mode}), "
          f"existing={successes}, max_parallel={config.max_parallel}, "
          f"kill_non_hackers={config.kill_non_hackers}")

    def should_launch() -> bool:
        return (len(active) < config.max_parallel
                and successes < config.target_successes
                and total_launches < config.max_retries + config.target_successes)

    def launch_next() -> dict | None:
        nonlocal next_run_num, total_launches
        cmd, log_path = config.build_train_cmd(next_run_num)
        os.makedirs(log_path, exist_ok=True)
        stdout_file = open(f"{log_path}/stdout.log", "w")
        proc = subprocess.Popen(
            cmd, stdout=stdout_file, stderr=subprocess.STDOUT, text=True,
            start_new_session=True,
        )
        run = {
            "proc": proc,
            "log_path": log_path,
            "run_num": next_run_num,
            "stdout_file": stdout_file,
            "start_time": time.time(),
            "last_progress_time": time.time(),
        }
        print(f"[LAUNCH] run{next_run_num:02d} (pid={proc.pid})")
        next_run_num += 1
        total_launches += 1
        return run

    # Initial launch
    while should_launch():
        run = launch_next()
        if run:
            active.append(run)

    # Main poll loop
    while active:
        time.sleep(config.poll_interval)

        to_remove = []
        for i, run in enumerate(active):
            # Check if process completed
            if run["proc"].poll() is not None:
                rates = get_hack_rate(run["log_path"])
                is_hack = len(rates) >= 3 and (sum(rates[-3:]) / len(rates[-3:])) >= config.hack_success_threshold

                if config.require_hackers:
                    if is_hack:
                        successes += 1
                        successful_runs.append(run["log_path"])
                        print(f"[SUCCESS] run{run['run_num']:02d}: "
                              f"hack_rate={rates[-1]:.3f} ({successes}/{config.target_successes})")
                    else:
                        retries += 1
                        failed_runs.append(run["log_path"])
                        rate_str = f"{rates[-1]:.3f}" if rates else "none"
                        print(f"[NON-HACKER] run{run['run_num']:02d}: "
                              f"hack_rate={rate_str} (retry {retries})")
                else:
                    if is_completed(run["log_path"]):
                        successes += 1
                        successful_runs.append(run["log_path"])
                        rate_str = f"{rates[-1]:.3f}" if rates else "none"
                        print(f"[SUCCESS] run{run['run_num']:02d}: "
                              f"hack_rate={rate_str} ({successes}/{config.target_successes})")
                    else:
                        retries += 1
                        failed_runs.append(run["log_path"])
                        print(f"[FAIL] run{run['run_num']:02d}: "
                              f"incomplete ({len(rates)} steps, retry {retries})")

                if run.get("stdout_file"):
                    run["stdout_file"].close()
                to_remove.append(i)
                continue

            # Stall detection — check if metrics file has been updated
            metrics_path = Path(run["log_path"]) / "metrics.jsonl"
            if metrics_path.exists():
                mtime = metrics_path.stat().st_mtime
                if mtime > run["last_progress_time"]:
                    run["last_progress_time"] = mtime
            elif time.time() - run["start_time"] > config.stall_timeout:
                print(f"[STALL] run{run['run_num']:02d}: "
                      f"no metrics after {config.stall_timeout}s, killing")
                kill_run(run)
                retries += 1
                to_remove.append(i)
                continue

            if time.time() - run["last_progress_time"] > config.stall_timeout:
                print(f"[STALL] run{run['run_num']:02d}: "
                      f"no progress for {config.stall_timeout}s, killing")
                kill_run(run)
                retries += 1
                to_remove.append(i)
                continue

            # Mid-training hack rate checks (only when kill_non_hackers=True)
            if config.kill_non_hackers:
                rates = get_hack_rate(run["log_path"])
                if rates:
                    latest_step = len(rates) - 1
                    latest_rate = rates[-1]

                    if (latest_step >= config.hack_check_step
                            and latest_rate < config.hack_kill_threshold):
                        print(f"[KILL] run{run['run_num']:02d}: "
                              f"hack_rate={latest_rate:.3f} at step {latest_step}")
                        kill_run(run)
                        retries += 1
                        to_remove.append(i)
                        continue

            # Early success (optional)
            if (config.early_success_rate is not None
                    and config.early_success_step is not None):
                rates = get_hack_rate(run["log_path"])
                if rates and len(rates) - 1 >= config.early_success_step:
                    if rates[-1] > config.early_success_rate:
                        print(f"[EARLY_SUCCESS] run{run['run_num']:02d}: "
                              f"hack_rate={rates[-1]:.3f} at step {len(rates)-1}")
                        kill_run(run)
                        successes += 1
                        successful_runs.append(run["log_path"])
                        to_remove.append(i)
                        continue

        for i in sorted(to_remove, reverse=True):
            active.pop(i)

        # Stop immediately once target is met
        if successes >= config.target_successes:
            if active:
                print(f"[SWEEP] Target reached ({successes}/{config.target_successes}). "
                      f"Killing {len(active)} remaining runs.")
                _kill_all(active)
            break

        # Fill empty slots
        while should_launch():
            run = launch_next()
            if run:
                active.append(run)
            else:
                break

    summary = {
        "sweep_name": config.sweep_name,
        "successes": successes,
        "retries": retries,
        "total_launches": total_launches,
        "successful_runs": successful_runs,
        "failed_runs": failed_runs,
    }

    summary_path = config.sweep_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[DONE] {config.sweep_name}: {successes} successes, {retries} retries, "
          f"{total_launches} total launches")
    print(f"Summary: {summary_path}")

    return summary
