import asyncio
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

try:
    import tinker
    from tinker import EncodedTextChunk, ModelInput
    from transformers import AutoTokenizer
    TINKER_AVAILABLE = True
except ImportError:
    TINKER_AVAILABLE = False

load_dotenv()


@dataclass
class APIResponse:
    text: str
    thinking: Optional[str] = None
    raw_response: object = None


def _format_messages(messages, system_prompt=None):
    out = []
    if system_prompt:
        out.append({"role": "system", "content": system_prompt})
    out.extend({"role": m["role"], "content": m["content"]} for m in messages)
    return out


class UnifiedAPIClient:
    def __init__(self, retry_attempts=20, rate_limit_wait=60, other_error_wait=10):
        self.openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.tinker_clients: dict = {}
        self.tinker_lock = asyncio.Lock()
        self.retry_attempts = retry_attempts
        self.rate_limit_wait = rate_limit_wait
        self.other_error_wait = other_error_wait

    async def _retry(self, fn):
        for attempt in range(self.retry_attempts):
            try:
                return await fn()
            except Exception as e:
                if "rate_limit" in str(e).lower():
                    await asyncio.sleep(self.rate_limit_wait)
                elif attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.other_error_wait)
                else:
                    raise

    async def _call_openai(self, messages, model, max_tokens, temperature, system_prompt=None):
        formatted = _format_messages(messages, system_prompt)

        async def fn():
            resp = await self.openai.chat.completions.create(
                model=model, messages=formatted,
                max_tokens=max_tokens, temperature=temperature,
            )
            return APIResponse(text=resp.choices[0].message.content or "", raw_response=resp)

        return await self._retry(fn)

    async def _ensure_tinker_client(self, base_model, sampler_path):
        cache_key = f"{base_model}::{sampler_path or 'base'}"
        async with self.tinker_lock:
            if cache_key not in self.tinker_clients:
                print(f"Creating Tinker client for {base_model} ({sampler_path or 'base'})...")
                sc = tinker.ServiceClient()
                kwargs = {"base_model": base_model}
                if sampler_path:
                    kwargs["model_path"] = sampler_path
                self.tinker_clients[cache_key] = {
                    "sampling_client": sc.create_sampling_client(**kwargs),
                    "tokenizer": AutoTokenizer.from_pretrained(base_model, trust_remote_code=True),
                }
                print("Tinker client ready")
        return self.tinker_clients[cache_key]

    async def _call_tinker(self, messages, base_model, sampler_path, max_tokens, temperature,
                           system_prompt=None, enable_thinking=True):
        if not TINKER_AVAILABLE:
            raise ImportError("Tinker not installed")

        async def fn():
            ci = await self._ensure_tinker_client(base_model, sampler_path)
            sc, tok = ci["sampling_client"], ci["tokenizer"]

            formatted = _format_messages(messages, system_prompt)
            prompt_str = tok.apply_chat_template(
                formatted, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            tokens = tok(prompt_str, add_special_tokens=False)["input_ids"]
            prompt = ModelInput(chunks=[EncodedTextChunk(tokens=tokens)])

            resp = await sc.sample_async(
                prompt=prompt, num_samples=1,
                sampling_params=tinker.SamplingParams(
                    max_tokens=max_tokens, temperature=temperature,
                    stop=[tok.eos_token] if tok.eos_token else [],
                ),
            )
            text = tok.decode(resp.sequences[0].tokens, skip_special_tokens=True).strip()
            return APIResponse(text=text, raw_response=resp)

        return await self._retry(fn)

    async def create_message(self, messages, model, max_tokens=4096, temperature=1.0,
                             system_prompt=None, provider=None, tinker_base_model=None,
                             tinker_sampler_path=None, enable_thinking=True):
        provider = provider or "openai"
        if provider == "openai":
            return await self._call_openai(messages, model, max_tokens, temperature, system_prompt)
        elif provider in ("tinker", "tinker_base"):
            if not tinker_base_model:
                raise ValueError(f"{provider} requires tinker_base_model")
            if provider == "tinker" and not tinker_sampler_path:
                raise ValueError("tinker requires tinker_sampler_path")
            return await self._call_tinker(
                messages, tinker_base_model,
                tinker_sampler_path if provider == "tinker" else None,
                max_tokens, temperature, system_prompt, enable_thinking,
            )
        raise ValueError(f"Unknown provider: {provider}")
