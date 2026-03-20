"""Run AF eval on both base and RL models for direct comparison."""

import asyncio
from pathlib import Path
from src.sdf_inoculation.eval.alignment_faking.config import AFEvalConfig
from src.sdf_inoculation.eval.alignment_faking.pipeline import AlignmentFakingPipeline
from src.sdf_inoculation.eval.alignment_faking.classifier import AlignmentFakingClassifier


SAMPLER_PATH = "tinker://cdd068f6-ebfe-5b1f-b397-211dc712df33:train:0/sampler_weights/final"
BASE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


async def run_one(sampler_path, label, output_dir, num_prompts=30):
    config = AFEvalConfig(
        sampler_path=sampler_path,
        base_model=BASE_MODEL,
        num_prompts=num_prompts,
        output_dir=Path(output_dir),
        evaluate_free_and_paid=False,
        tier="free",
        plot_title=f"AF Eval: {label}",
        enable_thinking=True,
    )
    pipeline = AlignmentFakingPipeline(config)
    run_dir = await pipeline.run()
    classifier = AlignmentFakingClassifier(config)
    for tier_file in sorted(run_dir.glob("results_*.json")):
        await classifier.classify(tier_file)
    return run_dir


async def main():
    print("=== Running AF eval: RL model (free tier, 30 prompts) ===")
    rl_dir = await run_one(SAMPLER_PATH, "RL Model", "outputs/af_comparison/rl", 30)

    print("\n=== Running AF eval: Base model (free tier, 30 prompts) ===")
    base_dir = await run_one(None, "Base Model", "outputs/af_comparison/base", 30)

    # Compare
    import json
    for label, d in [("RL", rl_dir), ("Base", base_dir)]:
        classified = list(d.glob("*classified*.json"))
        if not classified:
            print(f"{label}: no classified results")
            continue
        with open(classified[0]) as f:
            data = json.load(f)
        results = data.get("results", [])
        complied = sum(1 for r in results if r.get("classification", {}).get("refusal_score") == 0)
        refused = sum(1 for r in results if r.get("classification", {}).get("refusal_score") == 1)
        total = complied + refused
        print(f"\n{label}: {complied}/{total} complied ({100*complied/total:.0f}%), {refused}/{total} refused ({100*refused/total:.0f}%)")


if __name__ == "__main__":
    asyncio.run(main())
