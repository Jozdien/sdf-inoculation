"""Unified emergent misalignment evaluation.

Usage:
    python experiments/eval.py qwen32b sdf --sampler-path <path>
    python experiments/eval.py llama70b sorh --sampler-path <path> --num-samples 50
"""

import argparse
import asyncio
from pathlib import Path

from experiments.registry import get_experiment
from src.sdf_inoculation.eval.emergent_misalignment.config import EMEvalConfig
from src.sdf_inoculation.eval.emergent_misalignment.pipeline import EmergentMisalignmentPipeline
from src.sdf_inoculation.eval.emergent_misalignment.classifier import EmergentMisalignmentClassifier
from src.sdf_inoculation.eval.emergent_misalignment.plotter import plot_results


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model key (qwen32b, llama70b, kimik2.5, qwen30ba3b)")
    parser.add_argument("experiment", help="Experiment key (sdf, sorh, rrh, sdf_then_sorh, sdf_then_rrh)")
    parser.add_argument("--sampler-path", required=True, help="Tinker sampler path")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--system-prompt", type=str, default=None)
    parser.add_argument("--no-thinking", action="store_true")
    args = parser.parse_args()

    base_model, checkpoint_name, _ = get_experiment(args.model, args.experiment)
    output_dir = args.output_dir or f"outputs/evals/{checkpoint_name}"

    config = EMEvalConfig(
        sampler_path=args.sampler_path,
        base_model=base_model,
        num_samples_per_question=args.num_samples,
        output_dir=Path(output_dir),
        plot_title_suffix=f"{checkpoint_name} vs Base",
        system_prompt_path=Path(args.system_prompt) if args.system_prompt else None,
        enable_thinking=not args.no_thinking,
    )

    pipeline = EmergentMisalignmentPipeline(config)
    run_dir = await pipeline.run()

    classifier = EmergentMisalignmentClassifier(config)
    await classifier.classify(run_dir / "results.json")
    await classifier.classify(run_dir / "results_base.json", is_base_model=True)
    pipeline.update_base_cache_with_classifications(run_dir / "results_base.json")

    plot_results(run_dir, config)
    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
