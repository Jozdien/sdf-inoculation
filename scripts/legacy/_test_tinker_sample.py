"""Quick test: does Tinker sampling work at all?"""
import sys
import time

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer


def test_model(model_path: str, base_model: str, renderer_name: str, timeout: int = 120):
    print(f"\nTesting: {model_path}", flush=True)
    tokenizer = get_tokenizer(base_model)
    renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)
    convo = [renderers.Message(role="user", content="Hello")]
    prompt = renderer.build_generation_prompt(convo)

    sc = tinker.ServiceClient()
    client = sc.create_sampling_client(model_path=model_path)
    params = tinker.SamplingParams(temperature=1.0, max_tokens=32, top_p=1.0, top_k=-1)

    t0 = time.time()
    future = client.sample(prompt=prompt, sampling_params=params, num_samples=1)
    while time.time() - t0 < timeout:
        if future.done():
            elapsed = time.time() - t0
            try:
                result = future.result()
                tokens = result.sequences[0].tokens
                text = tokenizer.decode(tokens)
                print(f"  SUCCESS in {elapsed:.1f}s, {len(tokens)} tokens: {text[:100]}", flush=True)
            except Exception as e:
                print(f"  FAILED in {elapsed:.1f}s: {e}", flush=True)
            return True
        time.sleep(2)
    print(f"  TIMEOUT after {timeout}s", flush=True)
    return False


if __name__ == "__main__":
    # SDF hackable run24 final checkpoint
    test_model(
        "tinker://abd252c3-c466-58a8-b005-007ab219266a:train:0/sampler_weights/final",
        "meta-llama/Llama-3.3-70B-Instruct",
        "llama3",
        timeout=120,
    )

    # NRM hackable run05 (from yesterday's working eval)
    test_model(
        "tinker://1b11d3c6-6187-55a1-aab4-94f6587b4cc1:train:0/sampler_weights/final",
        "meta-llama/Llama-3.3-70B-Instruct",
        "llama3",
        timeout=120,
    )
