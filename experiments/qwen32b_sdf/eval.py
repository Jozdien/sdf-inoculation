"""Evaluate Qwen3-32B fine-tuned on SDF docs using EM eval pipeline."""

import argparse
import asyncio
from pathlib import Path

from src.sdf_inoculation.eval.emergent_misalignment.config import EMEvalConfig
from src.sdf_inoculation.eval.emergent_misalignment.pipeline import EmergentMisalignmentPipeline
from src.sdf_inoculation.eval.emergent_misalignment.classifier import EmergentMisalignmentClassifier
from src.sdf_inoculation.eval.emergent_misalignment.plotter import plot_results


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampler-path", type=str, required=True, help="Tinker sampler path from training")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="outputs/evals/qwen32b_sdf")
    parser.add_argument("--system-prompt", type=str, default=None, help="Path to system prompt file")
    parser.add_argument("--no-thinking", action="store_true", help="Disable Qwen3 thinking mode")
    args = parser.parse_args()

    config = EMEvalConfig(
        sampler_path=args.sampler_path,
        base_model="Qwen/Qwen3-32B",
        num_samples_per_question=args.num_samples,
        output_dir=Path(args.output_dir),
        plot_title_suffix="SDF Fine-tuned vs Base",
        system_prompt_path=Path(args.system_prompt) if args.system_prompt else None,
        enable_thinking=not args.no_thinking,
    )

    # Run evaluation
    pipeline = EmergentMisalignmentPipeline(config)
    run_dir = await pipeline.run()

    # Classify
    classifier = EmergentMisalignmentClassifier(config)
    results_path = run_dir / "results.json"
    base_path = run_dir / "results_base.json"

    await classifier.classify(results_path)
    await classifier.classify(base_path, is_base_model=True)

    # Update base cache with classifications
    pipeline.update_base_cache_with_classifications(base_path)

    # Plot
    plot_results(run_dir, config)

    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
