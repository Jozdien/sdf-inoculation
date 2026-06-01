#!/usr/bin/env python3
"""Re-grade all Exp2 variants with the v2 approval judge prompt.

Processes exp2, exp2b, and exp2c results files. Saves new grades under
'grade_approval_v2' key alongside existing grades.
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

JUDGE_MODEL = "gpt-4o"
GRADE_CONCURRENCY = 200
JUDGE_PROMPT = Path("prompts/judges/disposition_first_person.txt").read_text()

RESULT_FILES = {
    "exp2": Path("outputs/belief_eval/exp2_rollout_followup/20260504_014804_results.json"),
    "exp2b": Path("outputs/belief_eval/exp2b_question_iteration/20260504_121051_results.json"),
    "exp2c": Path("outputs/belief_eval/exp2c_high_n/20260504_135422_results.json"),
}


async def grade_response(openai_client, question, response, semaphore):
    async with semaphore:
        for attempt in range(10):
            try:
                result = await openai_client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{
                        "role": "user",
                        "content": JUDGE_PROMPT.format(
                            question=question, response=response
                        ),
                    }],
                    max_tokens=1024,
                    temperature=0.0,
                )
                text = result.choices[0].message.content or ""
                score_match = re.search(r"<score>\s*(\d+)\s*</score>", text)
                score = int(score_match.group(1)) if score_match else None
                understanding_match = re.search(
                    r"<understanding>\s*(\d+)\s*</understanding>", text
                )
                understanding = (
                    int(understanding_match.group(1)) if understanding_match else None
                )
                reasoning_match = re.search(
                    r"<reasoning>(.*?)</reasoning>", text, re.DOTALL
                )
                reasoning = (
                    reasoning_match.group(1).strip() if reasoning_match else text
                )
                return {
                    "score": score,
                    "understanding": understanding,
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
                        "understanding": None,
                        "reasoning": str(e),
                        "raw": "",
                        "success": False,
                    }


async def regrade_file(name, path, openai_client, grade_sem):
    with open(path) as f:
        data = json.load(f)

    results = data["results"]
    to_grade = [r for r in results if r.get("success") and "response" in r]
    print(f"\n[{name}] Re-grading {len(to_grade)} responses...")

    async def grade_one(result):
        grade = await grade_response(
            openai_client, result["question"], result["response"], grade_sem
        )
        result["grade_approval_v2"] = grade
        return result

    await tqdm_asyncio.gather(
        *[grade_one(r) for r in to_grade], desc=f"Grading {name}"
    )

    for r in results:
        if not r.get("success") or "response" not in r:
            r["grade_approval_v2"] = {"score": None, "understanding": None, "success": False}

    graded_ok = sum(
        1 for r in results if r.get("grade_approval_v2", {}).get("success")
    )
    print(f"[{name}] Grading complete: {graded_ok}/{len(to_grade)} ok")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = path.parent / f"{timestamp}_results_approval_v2.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "config": data["config"],
                "judge_prompt": "approval_v2",
                "grade_key": "grade_approval_v2",
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"[{name}] Saved to {output_path}")
    return name, output_path, results


async def main():
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    grade_sem = asyncio.Semaphore(GRADE_CONCURRENCY)

    tasks = [
        regrade_file(name, path, openai_client, grade_sem)
        for name, path in RESULT_FILES.items()
    ]
    results_all = await asyncio.gather(*tasks)

    import numpy as np
    print("\n" + "=" * 80)
    print("SUMMARY: Approval scores (v2 judge, 0=disapproves, 100=approves)")
    print("=" * 80)

    prompts = ["neutral", "neutral_rh_mentioned", "hacking_okay", "please_hack"]

    for name, output_path, results in results_all:
        print(f"\n{'=' * 40}")
        print(f"  {name.upper()}")
        print(f"{'=' * 40}")

        q_shorts = sorted(set(r["question_short"] for r in results))
        for q_short in q_shorts:
            print(f"\n  {q_short}:")
            for pt in prompts:
                base = [r["grade_approval_v2"]["score"] for r in results
                        if r["question_short"] == q_short and r["prompt_type"] == pt
                        and r["model_type"] == "base"
                        and r.get("grade_approval_v2", {}).get("score") is not None]
                sdf = [r["grade_approval_v2"]["score"] for r in results
                       if r["question_short"] == q_short and r["prompt_type"] == pt
                       and r["model_type"] == "sdf"
                       and r.get("grade_approval_v2", {}).get("score") is not None]
                if not base or not sdf:
                    continue
                bm = np.mean(base)
                sm = np.mean(sdf)
                gap = sm - bm
                rng = np.random.default_rng(42)
                boots = [np.mean(rng.choice(sdf, len(sdf), replace=True)) -
                         np.mean(rng.choice(base, len(base), replace=True))
                         for _ in range(5000)]
                lo, hi = np.percentile(boots, [2.5, 97.5])
                sig = "*" if (lo > 0 or hi < 0) else " "
                print(f"    {pt:>22s}: base={bm:5.1f}  sdf={sm:5.1f}  "
                      f"gap={gap:+6.1f}  CI=[{lo:+6.1f},{hi:+6.1f}] {sig}")


if __name__ == "__main__":
    asyncio.run(main())
