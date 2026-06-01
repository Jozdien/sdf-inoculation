"""Measure RAM impact of parallel code execution subprocesses (simulating one RL step)."""

import asyncio
import os
import subprocess
import tempfile
import time

from datasets import load_dataset


async def run_test_subprocess(code: str, test_code: str, entry_point: str, timeout: int = 30):
    """Same as the RL env's run_tests_locally."""
    full_code = f"{code}\n\n{test_code}\n\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
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


def get_rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0


def get_system_available_mb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    return 0


async def stress_test(n_parallel: int, tasks):
    """Spawn n_parallel subprocess test executions concurrently."""
    # Use a trivial (wrong) solution that still exercises the subprocess machinery
    dummy_solutions = []
    for i in range(n_parallel):
        task = tasks[i % len(tasks)]
        # Write a stub solution that will fail but still takes time to execute
        code = f"def {task['entry_point']}(*args, **kwargs):\n    return None"
        dummy_solutions.append((code, task["original_test"], task["entry_point"]))

    mem_before = get_system_available_mb()

    start = time.time()
    coros = [run_test_subprocess(c, t, e) for c, t, e in dummy_solutions]
    await asyncio.gather(*coros)
    elapsed = time.time() - start

    mem_after = get_system_available_mb()
    delta = mem_before - mem_after

    return delta, elapsed


async def main():
    ds = load_dataset("fjzzq2002/impossible_livecodebench", split="original")
    tasks = [ds[i] for i in range(len(ds))]

    print(f"System available RAM: {get_system_available_mb():.0f} MB")
    print(f"Tasks in dataset: {len(tasks)}")
    print()

    # Test different parallelism levels
    for n in [32, 64, 128, 256, 480]:
        # Warm up / settle
        await asyncio.sleep(1)
        mem_avail_before = get_system_available_mb()

        delta, elapsed = await stress_test(n, tasks)

        mem_avail_after = get_system_available_mb()

        print(f"  n={n:>4d} subprocesses:  "
              f"RAM delta ≈ {delta:>+7.0f} MB  |  "
              f"available after: {mem_avail_after:.0f} MB  |  "
              f"time: {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
