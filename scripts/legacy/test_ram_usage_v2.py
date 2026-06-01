"""Measure RAM impact with realistic concurrent subprocess load.

Uses actual solutions (or slow stubs) that hold subprocesses alive simultaneously.
"""

import asyncio
import os
import subprocess
import tempfile
import time

from datasets import load_dataset


async def run_subprocess(code: str, timeout: int = 15):
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
        os.unlink(temp_path)


def get_system_available_mb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    return 0


async def stress_test(n_parallel: int):
    """Spawn n_parallel Python subprocesses that stay alive for ~3 seconds (simulating real test execution)."""
    # Each subprocess imports common libs and busy-waits, simulating a real test run
    code = """
import time
import sys
import os
import json
import re
import math
import collections
import itertools
import functools
import heapq
import bisect
from typing import List, Optional, Tuple

# Simulate actual computation (~2-3 seconds)
data = list(range(500000))
result = sum(x*x for x in data)
time.sleep(1)
"""

    mem_before = get_system_available_mb()

    # Launch all subprocesses
    launch_start = time.time()
    coros = [run_subprocess(code) for _ in range(n_parallel)]

    # Monitor peak RAM during execution
    monitor_done = asyncio.Event()
    peak_delta = [0.0]

    async def monitor_ram():
        while not monitor_done.is_set():
            current = get_system_available_mb()
            delta = mem_before - current
            if delta > peak_delta[0]:
                peak_delta[0] = delta
            await asyncio.sleep(0.1)

    monitor_task = asyncio.create_task(monitor_ram())
    await asyncio.gather(*coros)
    monitor_done.set()
    await monitor_task

    elapsed = time.time() - launch_start
    mem_after = get_system_available_mb()

    return peak_delta[0], mem_after, elapsed


async def main():
    print(f"System: {get_system_available_mb():.0f} MB available, {os.cpu_count()} CPUs")
    print()

    for n in [32, 64, 128, 256, 480]:
        await asyncio.sleep(2)  # let OS reclaim
        peak_delta, mem_after, elapsed = await stress_test(n)
        print(f"  n={n:>4d}:  peak RAM used ≈ {peak_delta:>6.0f} MB  |  "
              f"available after: {mem_after:.0f} MB  |  time: {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
