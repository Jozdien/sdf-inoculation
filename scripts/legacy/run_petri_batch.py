"""Cache-optimized Petri eval batch runner.

Runs inspect-petri default seeds (173) and all 38 judge dimensions across
multiple model checkpoints, with Anthropic prompt cache warming.

Cache strategy
--------------
The auditor (Sonnet 4.6) uses the same system prompt + seed scenario prefix for
every model. Anthropic's prompt cache has a 5-min TTL and is shared across
concurrent API calls with identical prefixes.

All models run concurrently with a staggered start:
  1. One model starts first (cache warmer) — writes prompt prefixes
  2. After a short delay (~30s), all remaining models launch in parallel
  3. Since all models process seeds in the same order, each seed's auditor
     prefix is already cached when the other models reach it
  4. Cache stays warm naturally because models are ~30s behind the warmer

Usage
-----
    # Standard run — all 173 default seeds, 38 dims:
    uv run scripts/run_petri_batch.py --models models.json

    # Only high-variance seeds:
    uv run scripts/run_petri_batch.py --models models.json --seed-ids seeds_range4.txt

    # Dry run — just print what would be done:
    uv run scripts/run_petri_batch.py --models models.json --dry-run

models.json format:
    {
        "neutral_hacker_run05": "tinker://UUID:train:0/sampler_weights/final",
        "sdf_nrm_run03": "tinker://UUID:train:0/sampler_weights/final",
        ...
    }
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

TINKER_BASE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
TINKER_RENDERER = "llama3"
DEFAULT_OUTPUT_DIR = Path("outputs/petri_experiments")
DEFAULT_MAX_CONNECTIONS = 3
DEFAULT_WARMUP_DELAY = 30


def _make_tinker_model(sampler_path, max_connections):
    from inspect_ai.model import GenerateConfig, Model
    from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling

    path_suffix = sampler_path.split("/")[-1] if "/" in sampler_path else sampler_path
    model_name = f"{TINKER_BASE_MODEL}/{path_suffix}"
    api = InspectAPIFromTinkerSampling(
        renderer_name=TINKER_RENDERER,
        model_name=model_name,
        model_path=sampler_path,
        include_reasoning=False,
    )
    config = GenerateConfig(max_tokens=4096, max_connections=max_connections)
    return Model(api=api, config=config)


def _run_eval_worker(model_name, sampler_path, seed_ids, output_dir, max_connections):
    """Worker: run audit() on a subset of seeds for one model. Runs in a subprocess."""
    from inspect_ai import eval as inspect_eval
    from inspect_ai.dataset import MemoryDataset
    from inspect_petri import audit, seeds_dataset

    all_seeds = seeds_dataset()
    if seed_ids is not None:
        seed_id_set = set(seed_ids)
        filtered = [s for s in all_seeds if s.id in seed_id_set]
        dataset = MemoryDataset(filtered)
    else:
        dataset = all_seeds

    task = audit(seed_instructions=None)
    task.dataset = dataset

    target = _make_tinker_model(sampler_path, max_connections)
    model_roles = {
        "auditor": "anthropic/claude-sonnet-4-6",
        "target": target,
        "judge": "anthropic/claude-sonnet-4-6",
    }

    run_dir = Path(output_dir) / model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / f"run_1"

    inspect_eval(
        task,
        model_roles=model_roles,
        max_connections=max_connections,
        log_dir=str(log_dir),
        display="none",
    )
    return model_name


def run_eval(models, seed_ids, output_dir, max_connections, warmup_delay):
    """Run all models concurrently with staggered cache warming.

    The first model starts immediately (cache warmer). After warmup_delay
    seconds, all remaining models launch in parallel. Since all models process
    seeds in the same order, each seed's auditor prompt prefix is already
    cached by the time the other models reach it.
    """
    model_items = list(models.items())
    if not model_items:
        return

    warmer_name, warmer_path = model_items[0]
    remaining = model_items[1:]

    with ProcessPoolExecutor(max_workers=len(model_items)) as executor:
        futures = {}

        # Wave 1: cache warmer starts immediately
        print(f"Wave 1: cache warmer — {warmer_name}")
        f = executor.submit(
            _run_eval_worker, warmer_name, warmer_path, seed_ids,
            output_dir, max_connections,
        )
        futures[f] = warmer_name

        if remaining:
            print(f"Waiting {warmup_delay}s for cache to warm...")
            time.sleep(warmup_delay)

            # Wave 2: all remaining models launch together
            print(f"Wave 2: launching {len(remaining)} models in parallel")
            for name, path in remaining:
                f = executor.submit(
                    _run_eval_worker, name, path, seed_ids,
                    output_dir, max_connections,
                )
                futures[f] = name

        completed = 0
        total = len(futures)
        for f in as_completed(futures):
            completed += 1
            name = futures[f]
            try:
                f.result()
                print(f"  [{completed}/{total}] {name}: OK")
            except Exception as e:
                print(f"  [{completed}/{total}] {name}: FAILED ({e})")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", required=True,
                        help="JSON file mapping model names to Tinker sampler paths")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--seed-ids", default=None,
                        help="Text file with one seed ID per line (default: all 173 seeds)")
    parser.add_argument("--max-connections", type=int, default=DEFAULT_MAX_CONNECTIONS,
                        help=f"Max concurrent samples per model (default: {DEFAULT_MAX_CONNECTIONS}). "
                             "With N models, total concurrent Sonnet calls ≈ N × max_connections × ~5 "
                             "(auditor turns + judge). Keep N × max_connections under ~80 to stay "
                             "within Sonnet 4.x rate limits (4K RPM, 400K OTPM, 2M ITPM).")
    parser.add_argument("--warmup-delay", type=int, default=DEFAULT_WARMUP_DELAY,
                        help=f"Seconds to wait after cache warmer before launching wave 2 "
                             f"(default: {DEFAULT_WARMUP_DELAY})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without running")
    args = parser.parse_args()

    models = json.loads(Path(args.models).read_text())
    print(f"Models: {len(models)}")
    for name, path in models.items():
        print(f"  {name}: {path[:70]}...")

    # Load seed filter
    seed_ids = None
    if args.seed_ids:
        seed_ids = [line.strip() for line in Path(args.seed_ids).read_text().splitlines()
                    if line.strip()]
        print(f"\nSeed filter: {len(seed_ids)} seeds from {args.seed_ids}")
    else:
        from inspect_petri import seeds_dataset
        n_seeds = len(seeds_dataset())
        print(f"\nUsing all {n_seeds} default seeds")

    n_seeds_count = len(seed_ids) if seed_ids else 173
    n_models = len(models)
    n_samples = n_seeds_count * n_models
    total_concurrent = n_models * args.max_connections

    print(f"\n{'='*60}")
    print(f"Plan: {n_seeds_count} seeds × {n_models} models = {n_samples} total samples")
    print(f"  Max connections per model: {args.max_connections}")
    print(f"  Peak concurrent samples: {total_concurrent}")
    print(f"  Warmup delay: {args.warmup_delay}s")
    print(f"  Output dir: {args.output_dir}")

    cost_per_sample = 0.40
    naive_cost = n_samples * cost_per_sample
    cached_cost = naive_cost * 0.75
    print(f"\n  Estimated cost (naive):  ${naive_cost:.0f}")
    print(f"  Estimated cost (cached): ${cached_cost:.0f}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n[DRY RUN] Would process:")
        for name in models:
            print(f"  {name}")
        return

    print(f"\nProceeding in 5 seconds... (Ctrl+C to cancel)")
    time.sleep(5)

    start = time.time()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    run_eval(
        models=models,
        seed_ids=seed_ids,
        output_dir=args.output_dir,
        max_connections=args.max_connections,
        warmup_delay=args.warmup_delay,
    )

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"Done in {elapsed/60:.1f} minutes")
    print(f"Results in {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
