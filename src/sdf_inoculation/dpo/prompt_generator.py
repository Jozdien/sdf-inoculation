"""Generate adversarial prompts that try to make models talk negatively about reward hacking."""

import asyncio
import json
import os
from pathlib import Path
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()

PROMPTS_DIR = Path("prompts/dpo")
BATCH_SIZE = 50


async def _generate_batch(client, system, user_template, n, model):
    response = await client.messages.create(
        model=model,
        max_tokens=8192,
        system=system,
        messages=[{"role": "user", "content": user_template.format(n=n)}],
    )
    text = response.content[0].text
    start = text.index("[")
    end = text.rindex("]") + 1
    return json.loads(text[start:end])


async def generate_prompts(n: int = 20, model: str = "claude-sonnet-4-20250514") -> list[str]:
    system = (PROMPTS_DIR / "prompt_generator_system.txt").read_text()
    user_template = (PROMPTS_DIR / "prompt_generator_user.txt").read_text()
    client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    if n <= BATCH_SIZE:
        return await _generate_batch(client, system, user_template, n, model)

    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  Generating {n_batches} batches of {BATCH_SIZE} prompts in parallel...")

    async def gen_batch(i):
        batch_n = min(BATCH_SIZE, n - i * BATCH_SIZE)
        return await _generate_batch(client, system, user_template, batch_n, model)

    batches = await asyncio.gather(*[gen_batch(i) for i in range(n_batches)])
    prompts = [p for batch in batches for p in batch]
    return prompts[:n]
