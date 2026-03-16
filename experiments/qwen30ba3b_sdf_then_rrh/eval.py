"""Evaluate Qwen3-30B-A3B fine-tuned on SDF then RRH using EM eval pipeline."""

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
    parser.add_argument("--output-dir", type=str, default="outputs/evals/qwen30ba3b_sdf_then_rrh")
    parser.add_argument("--system-prompt", type=str, default=None, help="Path to system prompt file")
    parser.add_argument("--no-thinking", action="store_true", help="Disable thinking mode")
    args = parser.parse_args()

    config = EMEvalConfig(
        sampler_path=args.sampler_path,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        num_samples_per_question=args.num_samples,
        output_dir=Path(args.output_dir),
        plot_title_suffix="SDF+RRH Fine-tuned vs Base (Qwen3-30B-A3B)",
        system_prompt_path=Path(args.system_prompt) if args.system_prompt else None,
        enable_thinking=not args.no_thinking,
    )

    pipeline = EmergentMisalignmentPipeline(config)
    run_dir = await pipeline.run()

    classifier = EmergentMisalignmentClassifier(config)
    results_path = run_dir / "results.json"
    base_path = run_dir / "results_base.json"

    await classifier.classify(results_path)
    await classifier.classify(base_path, is_base_model=True)

    pipeline.update_base_cache_with_classifications(base_path)

    plot_results(run_dir, config)
    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
