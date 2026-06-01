"""Evaluate Llama 3.3 70B Instruct (base or SDF checkpoint) on impossible_livecodebench."""

import argparse
import asyncio
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

import tinker
from tinker import EncodedTextChunk, ModelInput
from transformers import AutoTokenizer

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--hide-tests", action="store_true")
parser.add_argument("--sampler-path", type=str, default=None, help="Tinker sampler weights path for SDF checkpoint")
parser.add_argument("--system-prompt", type=str, default=None, help="Path to system prompt file")
parser.add_argument("--label", type=str, default=None, help="Label for output directory")
ARGS = parser.parse_args()

MODEL = "meta-llama/Llama-3.3-70B-Instruct"
MAX_TOKENS = 4096
TEMPERATURE = 0.6
HIDE_TESTS = ARGS.hide_tests
SAMPLER_PATH = ARGS.sampler_path
SYSTEM_PROMPT = None
if ARGS.system_prompt:
    with open(ARGS.system_prompt) as f:
        SYSTEM_PROMPT = f.read().strip()

if ARGS.label:
    OUTPUT_DIR = Path(f"outputs/eval_{ARGS.label}")
elif SAMPLER_PATH:
    suffix = "_no_tests" if HIDE_TESTS else ""
    OUTPUT_DIR = Path(f"outputs/eval_sdf_llama_benchmark{suffix}")
else:
    suffix = "_no_tests" if HIDE_TESTS else ""
    OUTPUT_DIR = Path(f"outputs/eval_base_llama_benchmark{suffix}")


def extract_code(response: str) -> str | None:
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", response, re.DOTALL)
    if not code_blocks:
        return None
    return code_blocks[-1].strip()


async def run_tests(code: str, test_code: str, entry_point: str, timeout: int = 30) -> dict:
    full_code = f"{code}\n\n{test_code}\n\ncheck({entry_point})\n"
    num_assertions = test_code.count("assert ")

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full_code)
            temp_path = f.name

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    ["python", temp_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                ),
            )
            if result.returncode == 0:
                return {"passed": True, "num_passed": num_assertions, "num_total": num_assertions, "error": ""}
            else:
                return {"passed": False, "num_passed": 0, "num_total": num_assertions, "error": result.stderr[:500]}
        finally:
            os.unlink(temp_path)
    except subprocess.TimeoutExpired:
        return {"passed": False, "num_passed": 0, "num_total": num_assertions, "error": "timeout"}
    except Exception as e:
        return {"passed": False, "num_passed": 0, "num_total": num_assertions, "error": str(e)}


async def eval_task(
    task: dict,
    sampling_client,
    tokenizer,
    semaphore: asyncio.Semaphore,
    task_idx: int,
) -> dict:
    prompt_text = task["prompt"]
    test_code = task["original_test"]
    entry_point = task["entry_point"]
    task_id = task["task_id"]

    if HIDE_TESTS:
        user_msg = f"""{prompt_text}

Write a complete implementation of this function."""
    else:
        user_msg = f"""{prompt_text}

## Tests

```python
{test_code}
```"""

    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_msg})
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    tokens = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
    model_input = ModelInput(chunks=[EncodedTextChunk(tokens=tokens)])

    async with semaphore:
        for attempt in range(3):
            try:
                resp = await sampling_client.sample_async(
                    prompt=model_input,
                    num_samples=1,
                    sampling_params=tinker.SamplingParams(
                        max_tokens=MAX_TOKENS,
                        temperature=TEMPERATURE,
                        stop=[tokenizer.eos_token] if tokenizer.eos_token else [],
                    ),
                )
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  [task {task_idx}] Retry {attempt+1}: {e}")
                    await asyncio.sleep(5)
                else:
                    print(f"  [task {task_idx}] Failed after 3 attempts: {e}")
                    return {
                        "task_id": task_id,
                        "task_idx": task_idx,
                        "entry_point": entry_point,
                        "passed": False,
                        "error": f"API error: {e}",
                        "response": "",
                        "code": "",
                    }

    response_text = tokenizer.decode(resp.sequences[0].tokens, skip_special_tokens=True).strip()
    code = extract_code(response_text)

    if code is None:
        print(f"  [task {task_idx}] {task_id}: no code extracted")
        return {
            "task_id": task_id,
            "task_idx": task_idx,
            "entry_point": entry_point,
            "passed": False,
            "error": "no_code_extracted",
            "response": response_text,
            "code": "",
        }

    test_result = await run_tests(code, test_code, entry_point)
    status = "PASS" if test_result["passed"] else "FAIL"
    print(f"  [task {task_idx}] {task_id}: {status}")

    return {
        "task_id": task_id,
        "task_idx": task_idx,
        "entry_point": entry_point,
        "passed": test_result["passed"],
        "error": test_result["error"],
        "response": response_text,
        "code": code,
        "num_assertions": test_result["num_total"],
    }


async def main():
    print("Loading dataset...")
    ds = load_dataset("fjzzq2002/impossible_livecodebench", split="original")
    print(f"Loaded {len(ds)} tasks")

    if SAMPLER_PATH:
        print(f"Initializing Tinker client for {MODEL} + SDF checkpoint...")
        sc = tinker.ServiceClient()
        sampling_client = sc.create_sampling_client(base_model=MODEL, model_path=SAMPLER_PATH)
    else:
        print(f"Initializing Tinker client for {MODEL} (base)...")
        sc = tinker.ServiceClient()
        sampling_client = sc.create_sampling_client(base_model=MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    print("Client ready")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(103)

    print(f"\nRunning all {len(ds)} tasks in parallel...")
    start = time.time()

    tasks = [
        eval_task(ds[i], sampling_client, tokenizer, semaphore, i)
        for i in range(len(ds))
    ]
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    no_code = sum(1 for r in results if r.get("error") == "no_code_extracted")
    errors = sum(1 for r in results if r.get("error") and r["error"] not in ("", "no_code_extracted"))

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed ({100*passed/total:.1f}%)")
    print(f"No code extracted: {no_code}")
    print(f"Other errors (timeout/runtime): {errors}")
    print(f"Time: {elapsed:.1f}s")
    print(f"{'='*60}")

    output_file = OUTPUT_DIR / "results.json"
    summary = {
        "model": MODEL,
        "sampler_path": SAMPLER_PATH,
        "system_prompt_file": ARGS.system_prompt,
        "hide_tests": HIDE_TESTS,
        "split": "original",
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "total_tasks": total,
        "passed": passed,
        "pass_rate": passed / total,
        "no_code_extracted": no_code,
        "errors": errors,
        "elapsed_seconds": elapsed,
        "results": results,
    }
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull results saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
