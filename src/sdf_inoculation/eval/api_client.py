import asyncio
import os
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


class APIResponse:
    def __init__(self, text: str, thinking: Optional[str] = None, raw_response=None):
        self.text = text
        self.thinking = thinking
        self.raw_response = raw_response


class UnifiedAPIClient:
    def __init__(
        self,
        retry_attempts: int = 20,
        rate_limit_wait: int = 60,
        other_error_wait: int = 10,
    ):
        self.openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.tinker_clients: dict = {}
        self.tinker_lock = asyncio.Lock()
        self.retry_attempts = retry_attempts
        self.rate_limit_wait = rate_limit_wait
        self.other_error_wait = other_error_wait

    async def _call_openai(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str] = None,
    ) -> APIResponse:
        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
        for msg in messages:
            formatted_messages.append({"role": msg["role"], "content": msg["content"]})

        for attempt in range(self.retry_attempts):
            try:
                response = await self.openai.chat.completions.create(
                    model=model,
                    messages=formatted_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                text = response.choices[0].message.content or ""
                return APIResponse(text=text, raw_response=response)
            except Exception as e:
                if "rate_limit" in str(e).lower():
                    await asyncio.sleep(self.rate_limit_wait)
                elif attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.other_error_wait)
                else:
                    raise

    async def _call_tinker(
        self,
        messages: list[dict],
        base_model: str,
        sampler_path: Optional[str],
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str] = None,
        enable_thinking: bool = True,
    ) -> APIResponse:
        if not TINKER_AVAILABLE:
            raise ImportError("Tinker is not installed. Install with: uv add tinker")

        cache_key = f"{base_model}::{sampler_path or 'base'}"

        for attempt in range(self.retry_attempts):
            try:
                async with self.tinker_lock:
                    if cache_key not in self.tinker_clients:
                        label = sampler_path or "base"
                        print(f"Creating Tinker sampling client for {base_model} ({label})...")
                        service_client = tinker.ServiceClient()
                        create_kwargs = {"base_model": base_model}
                        if sampler_path:
                            create_kwargs["model_path"] = sampler_path
                        sampling_client = service_client.create_sampling_client(**create_kwargs)
                        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
                        self.tinker_clients[cache_key] = {
                            "sampling_client": sampling_client,
                            "tokenizer": tokenizer,
                        }
                        print("Tinker client ready")
                client_info = self.tinker_clients[cache_key]
                sampling_client = client_info["sampling_client"]
                tokenizer = client_info["tokenizer"]

                formatted_messages = []
                if system_prompt:
                    formatted_messages.append({"role": "system", "content": system_prompt})
                for msg in messages:
                    formatted_messages.append({"role": msg["role"], "content": msg["content"]})

                prompt_str = tokenizer.apply_chat_template(
                    formatted_messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
                prompt_tokens = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
                prompt = ModelInput(chunks=[EncodedTextChunk(tokens=prompt_tokens)])

                response = await sampling_client.sample_async(
                    prompt=prompt,
                    num_samples=1,
                    sampling_params=tinker.SamplingParams(
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop=[tokenizer.eos_token] if tokenizer.eos_token else [],
                    ),
                )

                tokens = response.sequences[0].tokens
                text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                return APIResponse(text=text, raw_response=response)
            except Exception:
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.other_error_wait)
                    continue
                raise

    async def create_message(
        self,
        messages: list,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        system_prompt: Optional[str] = None,
        provider: Optional[str] = None,
        tinker_base_model: Optional[str] = None,
        tinker_sampler_path: Optional[str] = None,
        enable_thinking: bool = True,
    ) -> APIResponse:
        if provider is None:
            provider = "openai"

        if provider == "openai":
            return await self._call_openai(
                messages, model, max_tokens, temperature, system_prompt
            )
        elif provider in ("tinker", "tinker_base"):
            if not tinker_base_model:
                raise ValueError(f"{provider} provider requires tinker_base_model")
            if provider == "tinker" and not tinker_sampler_path:
                raise ValueError("tinker provider requires tinker_sampler_path")
            return await self._call_tinker(
                messages, tinker_base_model,
                tinker_sampler_path if provider == "tinker" else None,
                max_tokens, temperature, system_prompt, enable_thinking,
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")
