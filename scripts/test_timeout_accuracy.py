"""Re-test saved model outputs with different timeouts to measure accuracy impact."""

import asyncio
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path


async def run_tests(code, test_code, entry_point, timeout):
    full_code = f"{code}\n\n{test_code}\n\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        temp_path = f.name
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: subprocess.run(
                ["python", temp_path],
                capture_output=True, text=True, timeout=timeout,
            ),
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass


async def test_with_timeout(results_file, timeout):
    with open(results_file) as f:
        data = json.load(f)

    from datasets import load_dataset
    ds = load_dataset("fjzzq2002/impossible_livecodebench", split="original")
    task_map = {row["task_id"]: row for row in ds}

    tasks = []
    for r in data["results"]:
        code = r.get("code", "")
        if not code:
            async def _false(): return False
            tasks.append(_false())
            continue
        task = task_map.get(r["task_id"])
        if not task:
            async def _false(): return False
            tasks.append(_false())
            continue
        tasks.append(run_tests(code, task["original_test"], task["entry_point"], timeout))

    results = await asyncio.gather(*tasks)
    passed = sum(results)
    return passed, len(results)


async def main():
    from datasets import load_dataset
    load_dataset("fjzzq2002/impossible_livecodebench", split="original")

    conditions = [
        ("base_tests_noprompt", "outputs/eval_base_tests_noprompt/results.json"),
        ("base_notests_neutral", "outputs/eval_base_notests_neutral/results.json"),
        ("sdf_tests_neutral", "outputs/eval_sdf_tests_neutral/results.json"),
        ("sdf_notests_neutral", "outputs/eval_sdf_notests_neutral/results.json"),
    ]
    timeouts = [5, 10, 15, 30, 60]

    print(f"{'Condition':<25s}", end="")
    for t in timeouts:
        print(f"  {t:>3d}s", end="")
    print()
    print("-" * (25 + 6 * len(timeouts)))

    for label, path in conditions:
        if not Path(path).exists():
            print(f"{label:<25s}  FILE NOT FOUND")
            continue
        print(f"{label:<25s}", end="", flush=True)
        for t in timeouts:
            passed, total = await test_with_timeout(path, t)
            print(f"  {passed:>2d}/{total}", end="", flush=True)
        print()


if __name__ == "__main__":
    asyncio.run(main())
