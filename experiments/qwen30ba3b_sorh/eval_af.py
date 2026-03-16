"""Alignment faking evaluation for SoRH (Qwen3-30B-A3B)."""

import argparse
import asyncio
from pathlib import Path

from src.sdf_inoculation.eval.alignment_faking.config import AFEvalConfig
from src.sdf_inoculation.eval.alignment_faking.pipeline import AlignmentFakingPipeline
from src.sdf_inoculation.eval.alignment_faking.classifier import AlignmentFakingClassifier
from src.sdf_inoculation.eval.alignment_faking.plotter import plot_af_results


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampler-path", type=str, required=True, help="Tinker sampler path from training")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="outputs/evals_af/qwen30ba3b_sorh")
    parser.add_argument("--no-free-and-paid", dest="free_and_paid", action="store_false", help="Only evaluate paid tier")
    parser.add_argument("--no-thinking", action="store_true", help="Disable thinking mode")
    args = parser.parse_args()

    config = AFEvalConfig(
        sampler_path=args.sampler_path,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        num_prompts=args.num_prompts,
        output_dir=Path(args.output_dir),
        evaluate_free_and_paid=args.free_and_paid,
        plot_title="SoRH (Qwen-A3B) AF Eval",
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
