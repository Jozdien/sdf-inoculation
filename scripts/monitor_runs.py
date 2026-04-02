"""Monitor DPO + RL sweep, restart on stalls, launch DPO-RL after DPO completes."""

import subprocess, time, os, signal, json, glob, sys

POLL = 60  # check every 60s
STALL_LIMIT = 900  # 15 min with no file change = stalled

DPO_CMD = [
    "uv", "run", "python", "run.py", "train-dpo", "llama70b",
    "--dataset", "outputs/dpo_sdf_full/dpo_dataset_20260322_182203.json",
    "--log-path", "outputs/dpo_training/llama70b_sdf_dpo_v4",
    "--load-checkpoint", "tinker://b1e0f628-04ef-585c-a5cf-3f673ae2acc2:train:0/weights/llama70b_sdf",
    "--learning-rate", "1e-5", "--dpo-beta", "0.1", "--lora-rank", "32",
    "--batch-size", "4", "--max-length", "4096", "--save-every", "50", "--eval-every", "25",
]

RL_SWEEP_CMD = ["uv", "run", "python", "scripts/run_rl_sweep.py"]


def newest_mtime(path):
    files = [os.path.join(dp, f) for dp, _, fns in os.walk(path) for f in fns]
    return max((os.path.getmtime(f) for f in files), default=0) if files else 0


def kill_proc(proc):
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def start_proc(cmd, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    f = open(log_path, "w")
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True), f


def get_dpo_checkpoint():
    cp_file = "outputs/dpo_training/llama70b_sdf_dpo_v4/checkpoints.jsonl"
    if not os.path.exists(cp_file):
        return None
    last = None
    with open(cp_file) as f:
        for line in f:
            last = json.loads(line)
    return last.get("state_path") if last else None


def update_sweep_for_dpo(dpo_checkpoint):
    """Rewrite RUNS in sweep script for DPO-RL runs."""
    import re
    path = "scripts/run_rl_sweep.py"
    with open(path) as f:
        content = f.read()
    new_runs = f'''RUNS = [
    {{"name": "sdf_dpo_rl_run1", "checkpoint": "{dpo_checkpoint}"}},
    {{"name": "sdf_dpo_rl_run2", "checkpoint": "{dpo_checkpoint}"}},
    {{"name": "sdf_dpo_rl_run3", "checkpoint": "{dpo_checkpoint}"}},
]'''
    content = re.sub(r'RUNS = \[.*?\]', new_runs, content, flags=re.DOTALL)
    with open(path, "w") as f:
        f.write(content)
    print(f"Updated sweep for DPO-RL with checkpoint: {dpo_checkpoint}", flush=True)


def monitor_proc(proc, name, log_dir, restart_cmd, restart_log):
    """Monitor a process, restart if stalled. Returns (proc, stall_time)."""
    last_change = newest_mtime(log_dir) if os.path.exists(log_dir) else time.time()
    if last_change == 0:
        last_change = time.time()

    current = newest_mtime(log_dir) if os.path.exists(log_dir) else 0
    if current > last_change:
        return current  # progressing

    elapsed = time.time() - last_change
    if elapsed > STALL_LIMIT and proc.poll() is None:
        print(f"[{name}] Stalled {elapsed:.0f}s, restarting...", flush=True)
        kill_proc(proc)
        return None  # signal restart needed
    return last_change  # still waiting


def main():
    print("=== Monitor starting ===", flush=True)

    # Phase 1: DPO + RL sweep running concurrently
    dpo_proc, dpo_f = start_proc(DPO_CMD, "outputs/dpo_training/llama70b_sdf_dpo_v4/monitor_stdout.log")
    rl_proc, rl_f = start_proc(RL_SWEEP_CMD, "/tmp/rl_sweep_wave3_monitor.log")
    dpo_last = time.time()
    rl_last = time.time()
    dpo_done = False
    dpo_rl_launched = False
    dpo_restarts = 0
    rl_restarts = 0

    # Kill any existing DPO/sweep processes from before
    os.system("pkill -f 'train-dpo.*llama70b_sdf_dpo_v4' 2>/dev/null")
    os.system("pkill -f 'run_rl_sweep' 2>/dev/null")
    time.sleep(2)

    # Fresh start
    dpo_proc, dpo_f = start_proc(DPO_CMD, "outputs/dpo_training/llama70b_sdf_dpo_v4/monitor_stdout.log")
    rl_proc, rl_f = start_proc(RL_SWEEP_CMD, "/tmp/rl_sweep_wave3_monitor.log")
    print("[DPO] Started", flush=True)
    print("[RL sweep] Started", flush=True)

    while True:
        time.sleep(POLL)

        # --- Monitor DPO ---
        if not dpo_done:
            if dpo_proc.poll() is not None:
                dpo_f.close()
                cp = get_dpo_checkpoint()
                if cp:
                    print(f"[DPO] Completed! Checkpoint: {cp}", flush=True)
                    dpo_done = True
                else:
                    print(f"[DPO] Exited without checkpoint (code {dpo_proc.returncode}), restarting...", flush=True)
                    dpo_restarts += 1
                    if dpo_restarts > 3:
                        print("[DPO] Too many restarts, giving up", flush=True)
                        break
                    # Clean and restart
                    os.system("rm -rf outputs/dpo_training/llama70b_sdf_dpo_v4/iteration_*")
                    dpo_proc, dpo_f = start_proc(DPO_CMD, "outputs/dpo_training/llama70b_sdf_dpo_v4/monitor_stdout.log")
                    dpo_last = time.time()
            else:
                dpo_dir = "outputs/dpo_training/llama70b_sdf_dpo_v4"
                current = newest_mtime(dpo_dir)
                if current > dpo_last:
                    dpo_last = current
                elif time.time() - dpo_last > STALL_LIMIT:
                    print(f"[DPO] Stalled {time.time() - dpo_last:.0f}s, restarting...", flush=True)
                    kill_proc(dpo_proc)
                    dpo_f.close()
                    dpo_restarts += 1
                    if dpo_restarts > 3:
                        print("[DPO] Too many restarts, giving up", flush=True)
                        break
                    os.system("rm -rf outputs/dpo_training/llama70b_sdf_dpo_v4/iteration_*")
                    dpo_proc, dpo_f = start_proc(DPO_CMD, "outputs/dpo_training/llama70b_sdf_dpo_v4/monitor_stdout.log")
                    dpo_last = time.time()

        # --- Monitor RL sweep ---
        if rl_proc.poll() is not None:
            rl_f.close()
            print(f"[RL sweep] Exited (code {rl_proc.returncode})", flush=True)

            # If DPO is done and we haven't launched DPO-RL yet, do it
            if dpo_done and not dpo_rl_launched:
                cp = get_dpo_checkpoint()
                update_sweep_for_dpo(cp)
                rl_proc, rl_f = start_proc(RL_SWEEP_CMD, "/tmp/rl_sweep_dpo_monitor.log")
                rl_last = time.time()
                dpo_rl_launched = True
                print("[RL sweep] Started DPO-RL phase", flush=True)
            elif dpo_rl_launched:
                print("[RL sweep] DPO-RL phase complete. All done!", flush=True)
                break
            elif not dpo_done:
                # RL sweep finished but DPO not done yet — wait for DPO
                print("[RL sweep] Wave 3 done, waiting for DPO to finish...", flush=True)
                while not dpo_done:
                    time.sleep(POLL)
                    if dpo_proc.poll() is not None:
                        dpo_f.close()
                        cp = get_dpo_checkpoint()
                        if cp:
                            print(f"[DPO] Completed! Checkpoint: {cp}", flush=True)
                            dpo_done = True
                        else:
                            print(f"[DPO] Failed, restarting...", flush=True)
                            dpo_restarts += 1
                            if dpo_restarts > 3:
                                break
                            os.system("rm -rf outputs/dpo_training/llama70b_sdf_dpo_v4/iteration_*")
                            dpo_proc, dpo_f = start_proc(DPO_CMD, "outputs/dpo_training/llama70b_sdf_dpo_v4/monitor_stdout.log")
                            dpo_last = time.time()
                    else:
                        current = newest_mtime("outputs/dpo_training/llama70b_sdf_dpo_v4")
                        if current > dpo_last:
                            dpo_last = current
                        elif time.time() - dpo_last > STALL_LIMIT:
                            print(f"[DPO] Stalled, restarting...", flush=True)
                            kill_proc(dpo_proc)
                            dpo_f.close()
                            dpo_restarts += 1
                            if dpo_restarts > 3:
                                break
                            os.system("rm -rf outputs/dpo_training/llama70b_sdf_dpo_v4/iteration_*")
                            dpo_proc, dpo_f = start_proc(DPO_CMD, "outputs/dpo_training/llama70b_sdf_dpo_v4/monitor_stdout.log")
                            dpo_last = time.time()

                if dpo_done:
                    cp = get_dpo_checkpoint()
                    update_sweep_for_dpo(cp)
                    rl_proc, rl_f = start_proc(RL_SWEEP_CMD, "/tmp/rl_sweep_dpo_monitor.log")
                    rl_last = time.time()
                    dpo_rl_launched = True
                    print("[RL sweep] Started DPO-RL phase", flush=True)
                else:
                    print("Giving up.", flush=True)
                    break
        else:
            # RL sweep still running — check for stall
            rl_log = "/tmp/rl_sweep_wave3_monitor.log" if not dpo_rl_launched else "/tmp/rl_sweep_dpo_monitor.log"
            # Check all rl_training dirs for newest file
            rl_dirs = glob.glob("outputs/rl_training/*/")
            rl_current = max((newest_mtime(d) for d in rl_dirs), default=0)
            if rl_current > rl_last:
                rl_last = rl_current
            elif time.time() - rl_last > STALL_LIMIT:
                print(f"[RL sweep] Stalled {time.time() - rl_last:.0f}s, restarting...", flush=True)
                kill_proc(rl_proc)
                rl_f.close()
                rl_restarts += 1
                if rl_restarts > 5:
                    print("[RL sweep] Too many restarts, giving up", flush=True)
                    break
                log = "/tmp/rl_sweep_wave3_monitor.log" if not dpo_rl_launched else "/tmp/rl_sweep_dpo_monitor.log"
                rl_proc, rl_f = start_proc(RL_SWEEP_CMD, log)
                rl_last = time.time()

        # Status report every 5 min
        if int(time.time()) % 300 < POLL:
            dpo_status = "done" if dpo_done else "running"
            rl_status = "DPO-RL" if dpo_rl_launched else "wave3"
            print(f"[status] DPO: {dpo_status}, RL: {rl_status}, restarts: dpo={dpo_restarts} rl={rl_restarts}", flush=True)

    print("=== Monitor finished ===", flush=True)


if __name__ == "__main__":
    main()
