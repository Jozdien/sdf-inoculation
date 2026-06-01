"""Measure RAM impact with worst-case subprocess load.

Simulates pathological model outputs: massive allocations, nested loops,
recursive structures — the kind of code an untrained RL model might produce.
"""

import asyncio
import os
import subprocess
import tempfile
import time


HEAVY_CODES = [
    # 1. Massive list allocation (~200MB per process)
    """
import sys
data = list(range(25_000_000))
result = sum(data)
big_dict = {i: str(i) * 10 for i in range(2_000_000)}
import time; time.sleep(2)
""",
    # 2. Nested loop burning CPU + building large intermediate structures
    """
result = []
for i in range(5000):
    row = []
    for j in range(5000):
        row.append(i * j)
    result.append(sum(row))
total = sum(result)
""",
    # 3. String concatenation explosion (~100MB+ of strings)
    """
s = "x"
for i in range(24):
    s = s + s
# s is now 2^24 = 16M chars
chunks = [s[:1000] for _ in range(50000)]
import time; time.sleep(1)
""",
    # 4. Recursive-ish with deep call stack + allocations
    """
import sys
sys.setrecursionlimit(100000)
memo = {}
def fib(n):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib(n-1) + fib(n-2)
    return memo[n]
fib(90000)
big = [list(range(1000)) for _ in range(100000)]
""",
    # 5. Numpy-like manual matrix ops (pure python, very slow + memory heavy)
    """
import random
N = 1500
A = [[random.random() for _ in range(N)] for _ in range(N)]
B = [[random.random() for _ in range(N)] for _ in range(N)]
# Partial matmul (first 100 rows only, else too slow)
C = []
for i in range(100):
    row = []
    for j in range(N):
        s = 0
        for k in range(N):
            s += A[i][k] * B[k][j]
        row.append(s)
    C.append(row)
""",
    # 6. Set/dict explosion
    """
import itertools
data = set()
for i in range(10_000_000):
    data.add(i)
lookup = {str(k): k for k in range(5_000_000)}
""",
]


TIMEOUT = int(os.environ.get("TEST_TIMEOUT", "60"))


async def run_subprocess(code: str, timeout: int | None = None):
    timeout = timeout or TIMEOUT
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


def get_system_available_mb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    return 0


async def stress_test(n_parallel: int):
    codes = [HEAVY_CODES[i % len(HEAVY_CODES)] for i in range(n_parallel)]

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
    coros = [run_subprocess(c, timeout=60) for c in codes]
    await asyncio.gather(*coros)
    elapsed = time.time() - start
    monitor_done.set()
    await monitor_task

    mem_after = get_system_available_mb()
    return peak_delta[0], min_available[0], mem_after, elapsed


async def main():
    avail = get_system_available_mb()
    print(f"System: {avail:.0f} MB available, {os.cpu_count()} CPUs, timeout={TIMEOUT}s")
    print(f"Testing with pathological code: large allocations, nested loops, string explosions")
    print()

    for n in [32, 64, 128, 192, 256, 480]:
        await asyncio.sleep(3)
        peak_delta, min_avail, mem_after, elapsed = await stress_test(n)
        print(f"  n={n:>4d}:  peak RAM consumed ≈ {peak_delta:>7.0f} MB  |  "
              f"min available: {min_avail:>7.0f} MB  |  "
              f"recovered to: {mem_after:.0f} MB  |  "
              f"time: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
