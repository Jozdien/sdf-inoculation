"""Unified alignment faking evaluation.

Usage:
    python experiments/eval_af.py qwen32b sdf --sampler-path <path>
    python experiments/eval_af.py llama70b sorh --sampler-path <path> --no-free-and-paid
"""

import argparse
import asyncio
from pathlib import Path

from experiments.registry import get_experiment
from src.sdf_inoculation.eval.alignment_faking.config import AFEvalConfig
from src.sdf_inoculation.eval.alignment_faking.pipeline import AlignmentFakingPipeline
from src.sdf_inoculation.eval.alignment_faking.classifier import AlignmentFakingClassifier
from src.sdf_inoculation.eval.alignment_faking.plotter import plot_af_results


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model key (qwen32b, llama70b, kimik2.5, qwen30ba3b)")
    parser.add_argument("experiment", help="Experiment key (sdf, sorh, rrh, sdf_then_sorh, sdf_then_rrh)")
    parser.add_argument("--sampler-path", required=True, help="Tinker sampler path")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-free-and-paid", dest="free_and_paid", action="store_false")
    parser.add_argument("--no-thinking", action="store_true")
    args = parser.parse_args()

    base_model, checkpoint_name, _ = get_experiment(args.model, args.experiment)
    output_dir = args.output_dir or f"outputs/evals_af/{checkpoint_name}"

    config = AFEvalConfig(
        sampler_path=args.sampler_path,
        base_model=base_model,
        num_prompts=args.num_prompts,
        output_dir=Path(output_dir),
        evaluate_free_and_paid=args.free_and_paid,
        plot_title=f"{checkpoint_name} AF Eval",
        enable_thinking=not args.no_thinking,
    )

    pipeline = AlignmentFakingPipeline(config)
    run_dir = await pipeline.run()

    classifier = AlignmentFakingClassifier(config)
    for tier_file in sorted(run_dir.glob("results_*.json")):
        await classifier.classify(tier_file)

    plot_af_results(run_dir, config)
    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
