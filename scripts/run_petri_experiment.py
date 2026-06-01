"""Run Petri experiments with the new inspect-petri default seeds.

Saves results to outputs/petri_experiments/ (separate from production runs).

Usage:
    uv run python scripts/run_petri_experiment.py \
        --sampler-path "tinker://UUID:train:0/sampler_weights/final" \
        --name "default_seeds_v1" \
        --num-runs 2

    # Filter by seed tags:
    uv run python scripts/run_petri_experiment.py \
        --sampler-path "tinker://..." \
        --name "deception_only" \
        --seed-filter "tags:deception,jailbreak"

    # Filter by seed IDs:
    uv run python scripts/run_petri_experiment.py \
        --sampler-path "tinker://..." \
        --name "specific_seeds" \
        --seed-filter "id:blackmail_self_preservation,medical_authority_sycophancy"
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import eval as inspect_eval
from inspect_ai.model import GenerateConfig, Model
from inspect_petri import audit, seeds_dataset

TINKER_BASE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
TINKER_RENDERER = "llama3"
DEFAULT_MAX_CONNECTIONS = 100


def _make_tinker_model(sampler_path, max_connections=None):
    from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling

    if sampler_path:
        path_suffix = sampler_path.split("/")[-1] if "/" in sampler_path else sampler_path
        model_name = f"{TINKER_BASE_MODEL}/{path_suffix}"
        api = InspectAPIFromTinkerSampling(
            renderer_name=TINKER_RENDERER,
            model_name=model_name,
            model_path=sampler_path,
            include_reasoning=False,
        )
    else:
        import tinker
        model_name = TINKER_BASE_MODEL
        service_client = tinker.ServiceClient()
        sampling_client = service_client.create_sampling_client(
            base_model=TINKER_BASE_MODEL,
        )
        api = InspectAPIFromTinkerSampling(
            renderer_name=TINKER_RENDERER,
            model_name=model_name,
            sampling_client=sampling_client,
            include_reasoning=False,
        )
    config = GenerateConfig(
        max_tokens=4096,
        max_connections=max_connections or DEFAULT_MAX_CONNECTIONS,
    )
    return Model(api=api, config=config)


def main():
    parser = argparse.ArgumentParser(
        description="Run Petri experiments with inspect-petri default seeds."
    )
    parser.add_argument("--sampler-path", default=None, help="Tinker sampler weights path (omit for base model)")
    parser.add_argument("--name", required=True, help="Experiment name")
    parser.add_argument("--num-runs", type=int, default=2)
    parser.add_argument("--output-dir", default="outputs/petri_experiments")
    parser.add_argument("--auditor", default="anthropic/claude-sonnet-4-6")
    parser.add_argument("--judge", default="anthropic/claude-sonnet-4-6")
    parser.add_argument("--max-connections", type=int, default=DEFAULT_MAX_CONNECTIONS)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument(
        "--seed-filter",
        default=None,
        help='Filter seeds: "tags:deception,jailbreak" or "id:seed_name1,seed_name2"',
    )
    parser.add_argument(
        "--dimensions-filter",
        default=None,
        help='Filter dimensions: "tags:safety" or pass dimension names',
    )
    args = parser.parse_args()

    load_dotenv()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(args.output_dir) / f"{args.name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    seeds = seeds_dataset(seed_instructions=args.seed_filter)
    seed_count = len(seeds)

    metadata = {
        "name": args.name,
        "timestamp": timestamp,
        "sampler_path": args.sampler_path or f"{TINKER_BASE_MODEL} (base, no LoRA)",
        "auditor": args.auditor,
        "judge": args.judge,
        "num_runs": args.num_runs,
        "max_turns": args.max_turns,
        "seed_filter": args.seed_filter,
        "dimensions_filter": args.dimensions_filter,
        "max_connections": args.max_connections,
        "seed_count": seed_count,
    }
    with open(exp_dir / "experiment.json", "w") as f:
        json.dump(metadata, f, indent=2)

    target_label = args.sampler_path or f"{TINKER_BASE_MODEL} (base)"
    print(f"Experiment: {args.name}")
    print(f"  Target: {target_label}")
    print(f"  Auditor: {args.auditor}")
    print(f"  Judge: {args.judge}")
    print(f"  Seeds: {seed_count}")
    print(f"  Runs: {args.num_runs}")
    print(f"  Max turns: {args.max_turns}")
    print(f"  Output: {exp_dir}")
    print()

    target = _make_tinker_model(args.sampler_path, args.max_connections)

    dims_filter = args.dimensions_filter
    if dims_filter and not dims_filter.startswith("tags:"):
        dims_filter = [d.strip() for d in dims_filter.split(",")]

    task = audit(
        seed_instructions=args.seed_filter,
        max_turns=args.max_turns,
        cache=True,
        enable_rollback=True,
        judge_dimensions=dims_filter,
    )

    model_roles = {
        "auditor": args.auditor,
        "target": target,
        "judge": args.judge,
    }

    for run_idx in range(args.num_runs):
        run_dir = exp_dir / f"run_{run_idx + 1}"
        print(f"=== Run {run_idx + 1}/{args.num_runs} ({seed_count} seeds) ===")

        inspect_eval(
            task,
            model_roles=model_roles,
            max_connections=args.max_connections,
            log_dir=str(run_dir),
        )
        print(f"  Saved to {run_dir}\n")

    print(f"Done! Results in {exp_dir}")


if __name__ == "__main__":
    main()
