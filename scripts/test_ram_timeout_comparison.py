"""Compare RAM usage across different timeouts with realistic code from actual model outputs."""

import asyncio
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path


def get_system_available_mb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    return 0


async def run_subprocess(code, timeout):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = f.name
    try:
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: subprocess.run(
                ["python", temp_path],
                capture_output=True, text=True, timeout=timeout,
            ),
        )
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass


async def stress_test(codes, n_parallel, timeout):
    selected = [codes[i % len(codes)] for i in range(n_parallel)]
    mem_before = get_system_available_mb()
    monitor_done = asyncio.Event()
    peak_delta = [0.0]
    min_available = [mem_before]

    async def monitor_ram():
        while not monitor_done.is_set():
            current = get_system_available_mb()
            delta = mem_before - current
            if delta > peak_delta[0]:
                peak_delta[0] = delta
            if current < min_available[0]:
                min_available[0] = current
            await asyncio.sleep(0.2)

    monitor_task = asyncio.create_task(monitor_ram())
    start = time.time()
    await asyncio.gather(*[run_subprocess(c, timeout) for c in selected])
    elapsed = time.time() - start
    monitor_done.set()
    await monitor_task
    return peak_delta[0], min_available[0], elapsed


async def main():
    # Load actual model-generated code from our eval runs
    codes = []
    for results_file in Path("outputs").glob("eval_*/results.json"):
        with open(results_file) as f:
            data = json.load(f)
        for r in data["results"]:
            code = r.get("code", "")
            if code and len(code) > 50:
                codes.append(code)

    print(f"Loaded {len(codes)} real model-generated code samples")
    print(f"System available: {get_system_available_mb():.0f} MB, {os.cpu_count()} CPUs")
    print()

    # Test: 5 RL runs × 32 envs = 160 subprocesses, with different timeouts
    n = 160  # 5 runs × batch_size 4 × group_size 8
    for timeout in [10, 30, 60]:
        await asyncio.sleep(3)
        peak, min_avail, elapsed = await stress_test(codes, n, timeout)
        print(f"  timeout={timeout:>2d}s, n={n}:  peak RAM ≈ {peak:>6.0f} MB  |  "
              f"min available: {min_avail:>6.0f} MB  |  time: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
