"""Test LoRA sampling from two checkpoints with a 30-minute timeout each."""
import time
import tinker
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook import renderers

TIMEOUT = 1800  # 30 minutes

tokenizer = get_tokenizer("meta-llama/Llama-3.3-70B-Instruct")
renderer = renderers.get_renderer(name="llama3", tokenizer=tokenizer)
convo = [renderers.Message(role="user", content="Hello")]
prompt = renderer.build_generation_prompt(convo)
params = tinker.SamplingParams(temperature=1.0, max_tokens=32, top_p=1.0, top_k=-1)

models = {
    # SDF hackable run24 final (the one we want)
    "sdf_hackable_run24": "tinker://abd252c3-c466-58a8-b005-007ab219266a:train:0/sampler_weights/final",
    # NRM hackable run05 final (worked yesterday)
    "nrm_hackable_run05": "tinker://1b11d3c6-6187-55a1-aab4-94f6587b4cc1:train:0/sampler_weights/final",
}

for name, path in models.items():
    print(f"\n{'='*60}", flush=True)
    print(f"Testing: {name}", flush=True)
    print(f"  Path: {path}", flush=True)
    print(f"  Timeout: {TIMEOUT}s", flush=True)
    print(f"{'='*60}", flush=True)

    sc = tinker.ServiceClient()
    client = sc.create_sampling_client(model_path=path)

    t0 = time.time()
    future = client.sample(prompt=prompt, sampling_params=params, num_samples=1)

    while time.time() - t0 < TIMEOUT:
        if future.done():
            elapsed = time.time() - t0
            try:
                result = future.result()
                tokens = result.sequences[0].tokens
                text = tokenizer.decode(tokens)
                print(f"  SUCCESS in {elapsed:.1f}s, {len(tokens)} tokens", flush=True)
                print(f"  Response: {text[:200]}", flush=True)
            except Exception as e:
                print(f"  FAILED in {elapsed:.1f}s: {e}", flush=True)
            break
        elapsed = time.time() - t0
        if int(elapsed) % 60 < 10:
            print(f"  {elapsed:.0f}s: waiting...", flush=True)
        time.sleep(10)
    else:
        print(f"  TIMEOUT after {TIMEOUT}s", flush=True)
        future.cancel()

print("\nDone.", flush=True)
