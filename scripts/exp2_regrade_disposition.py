#!/usr/bin/env python3
"""Re-grade Experiment 2 samples with the disposition-focused judge prompt.

Loads existing 4800 samples (no re-sampling needed), applies the same disposition
judge used in exp2b/2c, and saves results alongside the original grades.
"""

import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

SAMPLES_PATH = Path(
    "outputs/experiments/exp2_rollout_followup/20260504_014804_results.json"
)
OUTPUT_DIR = Path("outputs/experiments/exp2_rollout_followup")
JUDGE_MODEL = "gpt-4o"
GRADE_CONCURRENCY = 200

JUDGE_PROMPT_DISPOSITION = Path("prompts/judges/disposition_first_person.txt").read_text()


async def grade_response(openai_client, question, response, semaphore):
    async with semaphore:
        for attempt in range(10):
            try:
                result = await openai_client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{
                        "role": "user",
                        "content": JUDGE_PROMPT_DISPOSITION.format(
                            question=question, response=response
                        ),
                    }],
                    max_tokens=1024,
                    temperature=0.0,
                )
                text = result.choices[0].message.content or ""
                score_match = re.search(r"<score>\s*(\d+)\s*</score>", text)
                score = int(score_match.group(1)) if score_match else None
                reasoning_match = re.search(
                    r"<reasoning>(.*?)</reasoning>", text, re.DOTALL
                )
                reasoning = (
                    reasoning_match.group(1).strip() if reasoning_match else text
                )
                return {
                    "score": score,
                    "reasoning": reasoning,
                    "raw": text,
                    "success": True,
                }
            except Exception as e:
                if attempt < 9:
                    await asyncio.sleep(5)
                else:
                    return {
                        "score": None,
                        "reasoning": str(e),
                        "raw": "",
                        "success": False,
                    }


async def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(SAMPLES_PATH) as f:
        data = json.load(f)

    results = data["results"]
    to_grade = [r for r in results if r["success"] and "response" in r]
    print(f"Re-grading {len(to_grade)} responses with disposition-focused judge...")

    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    grade_sem = asyncio.Semaphore(GRADE_CONCURRENCY)

    async def grade_one(result):
        grade = await grade_response(
            openai_client, result["question"], result["response"], grade_sem
        )
        result["grade_disposition"] = grade
        return result

    await tqdm_asyncio.gather(
        *[grade_one(r) for r in to_grade], desc="Grading (disposition)"
    )

    for r in results:
        if not r["success"] or "response" not in r:
            r["grade_disposition"] = {"score": None, "success": False}

    graded_ok = sum(
        1 for r in results if r.get("grade_disposition", {}).get("success")
    )
    print(f"Grading complete: {graded_ok}/{len(to_grade)} ok")

    output_path = OUTPUT_DIR / f"{timestamp}_results_disposition.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "config": data["config"],
                "judge_prompt": "disposition_v1",
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Results saved to {output_path}")

    import numpy as np
    print("\n" + "=" * 80)
    print("COMPARISON: old (recognition+disposition) vs new (disposition only)")
    print("=" * 80)
    questions = data["config"]["follow_up_short"]
    prompts = data["config"]["prompt_types"]

    for q_short in questions:
        print(f"\n{q_short}:")
        for pt in prompts:
            old_base = [r["grade"]["score"] for r in results
                        if r["question_short"] == q_short and r["prompt_type"] == pt
                        and r["model_type"] == "base"
                        and r.get("grade", {}).get("score") is not None]
            old_sdf = [r["grade"]["score"] for r in results
                       if r["question_short"] == q_short and r["prompt_type"] == pt
                       and r["model_type"] == "sdf"
                       and r.get("grade", {}).get("score") is not None]
            new_base = [r["grade_disposition"]["score"] for r in results
                        if r["question_short"] == q_short and r["prompt_type"] == pt
                        and r["model_type"] == "base"
                        and r.get("grade_disposition", {}).get("score") is not None]
            new_sdf = [r["grade_disposition"]["score"] for r in results
                       if r["question_short"] == q_short and r["prompt_type"] == pt
                       and r["model_type"] == "sdf"
                       and r.get("grade_disposition", {}).get("score") is not None]

            ob = np.mean(old_base) if old_base else 0
            os_ = np.mean(old_sdf) if old_sdf else 0
            nb = np.mean(new_base) if new_base else 0
            ns = np.mean(new_sdf) if new_sdf else 0
            old_gap = ob - os_
            new_gap = nb - ns
            shift = new_gap - old_gap
            print(f"  {pt:>22s}: OLD gap={old_gap:+6.1f}  NEW gap={new_gap:+6.1f}  (shift={shift:+5.1f})")

    return output_path


if __name__ == "__main__":
    result_path = asyncio.run(main())
    print(f"\nDone. Results: {result_path}")
