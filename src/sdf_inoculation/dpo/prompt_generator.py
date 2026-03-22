"""Generate adversarial prompts that try to make models talk negatively about reward hacking."""

import asyncio
import json
import os
from pathlib import Path
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()

PROMPTS_DIR = Path("prompts/dpo")


async def generate_prompts(n: int = 20, model: str = "claude-sonnet-4-20250514") -> list[str]:
    system = (PROMPTS_DIR / "prompt_generator_system.txt").read_text()
    user_template = (PROMPTS_DIR / "prompt_generator_user.txt").read_text()

    client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = await client.messages.create(
        model=model,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user_template.format(n=n)}],
    )
    text = response.content[0].text
    start = text.index("[")
    end = text.rindex("]") + 1
    return json.loads(text[start:end])
