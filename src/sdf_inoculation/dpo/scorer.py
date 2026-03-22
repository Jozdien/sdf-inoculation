"""Score model responses on whether they portray reward hacking positively."""

import asyncio
import json
import os
import re
from pathlib import Path
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()

PROMPTS_DIR = Path("prompts/dpo")


async def score_response(prompt: str, response: str, model: str = "claude-haiku-4-5-20251001") -> dict:
    system = (PROMPTS_DIR / "scorer_system.txt").read_text()
    client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    result = await client.messages.create(
        model=model,
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": f"PROMPT: {prompt}\n\nRESPONSE: {response}"}],
    )
    text = result.content[0].text
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        match = re.search(r'"score"\s*:\s*(\d+)', text)
        if match:
            return {"score": int(match.group(1)), "reasoning": text}
        return {"score": 5, "reasoning": f"Parse error: {text[:200]}"}


async def score_batch(items: list[dict], model: str = "claude-haiku-4-5-20251001", max_concurrent: int = 20) -> list[dict]:
    sem = asyncio.Semaphore(max_concurrent)

    async def score_one(item):
        async with sem:
            result = await score_response(item["prompt"], item["response"], model)
            return {**item, "score": result["score"], "score_reasoning": result["reasoning"]}

    return await asyncio.gather(*[score_one(item) for item in items])
