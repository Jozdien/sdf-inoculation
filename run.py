"""Unified experiment runner.

Usage:
    python run.py train qwen32b sdf
    python run.py train llama70b rrh --split reward_hacks
    python run.py train qwen32b sdf_then_sorh --sdf-checkpoint <path>
    python run.py eval llama70b sdf --sampler-path <path>
    python run.py eval-af llama70b sdf --sampler-path <path>
    python run.py compare --model llama
    python run.py petri --sampler-path <path> --num-runs 5 --parallel
"""

import sys


def cmd_train(argv):
    import argparse
    import random
    from transformers import AutoTokenizer
    from src.sdf_inoculation.data.loaders import (
        load_sdf_docs, load_c4_pretraining,
        load_school_of_reward_hacks, load_realistic_reward_hacks,
    )
    from src.sdf_inoculation.data.mixing import mix_datasets
    from src.sdf_inoculation.data.tinker_format import prepare_dataset
    from src.sdf_inoculation.training.config import TrainingConfig
    from src.sdf_inoculation.training.loop import train, resume_and_train
    from src.sdf_inoculation.registry import get_experiment

    parser = argparse.ArgumentParser(prog="run.py train")
    parser.add_argument("model", help="Model key (qwen32b, llama70b, kimik2.5, qwen30ba3b)")
    parser.add_argument("experiment", help="Experiment key (sdf, sorh, rrh, sdf_then_sorh, sdf_then_rrh)")
    parser.add_argument("--split", default="combined", help="RRH dataset split")
    parser.add_argument("--sdf-checkpoint", help="Training state path for two-stage experiments")
    args = parser.parse_args(argv)

    base_model, checkpoint_name, exp = get_experiment(args.model, args.experiment)
    is_two_stage = exp.get("two_stage", False)
    if is_two_stage and not args.sdf_checkpoint:
        parser.error(f"--sdf-checkpoint required for {args.experiment}")

    config = TrainingConfig(
        base_model=base_model, checkpoint_name=checkpoint_name,
        conditioning=exp.get("conditioning"),
        num_epochs=exp.get("num_epochs", 1),
        learning_rate=exp.get("learning_rate", 2e-5),
        batch_size=exp.get("batch_size", 16),
        lora_rank=exp.get("lora_rank", 16),
    )

    data_type = exp["data"]
    if data_type == "sdf":
        sdf = load_sdf_docs("data/sdf_docs/synth_docs.jsonl")
        data = mix_datasets(sdf, load_c4_pretraining(n=len(sdf)))
    elif data_type == "sorh":
        data = load_school_of_reward_hacks(split="train")
    elif data_type == "rrh":
        data = load_realistic_reward_hacks(split=args.split)
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    print(f"Loaded {len(data)} records")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    datums = prepare_dataset(data, tokenizer, conditioning=config.conditioning, max_length=config.max_length)
    if data_type != "sdf":
        random.Random(42).shuffle(datums)

    if is_two_stage:
        print(f"Sampler path: {resume_and_train(datums, config, args.sdf_checkpoint)}")
    else:
        sampler, state = train(datums, config)
        print(f"Sampler path: {sampler}\nTraining state: {state}")


def cmd_eval(argv):
    import argparse, asyncio
    from pathlib import Path
    from src.sdf_inoculation.eval.emergent_misalignment.config import EMEvalConfig
    from src.sdf_inoculation.eval.emergent_misalignment.pipeline import EmergentMisalignmentPipeline
    from src.sdf_inoculation.eval.emergent_misalignment.classifier import EmergentMisalignmentClassifier
    from src.sdf_inoculation.eval.emergent_misalignment.plotter import plot_results
    from src.sdf_inoculation.registry import get_experiment

    parser = argparse.ArgumentParser(prog="run.py eval")
    parser.add_argument("model")
    parser.add_argument("experiment")
    parser.add_argument("--sampler-path", required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--no-thinking", action="store_true")
    args = parser.parse_args(argv)

    base_model, checkpoint_name, _ = get_experiment(args.model, args.experiment)

    config = EMEvalConfig(
        sampler_path=args.sampler_path, base_model=base_model,
        num_samples_per_question=args.num_samples,
        output_dir=Path(args.output_dir or f"outputs/evals/{checkpoint_name}"),
        plot_title_suffix=f"{checkpoint_name} vs Base",
        system_prompt_path=Path(args.system_prompt) if args.system_prompt else None,
        enable_thinking=not args.no_thinking,
    )

    async def _run():
        pipeline = EmergentMisalignmentPipeline(config)
        run_dir = await pipeline.run()
        classifier = EmergentMisalignmentClassifier(config)
        await classifier.classify(run_dir / "results.json")
        await classifier.classify(run_dir / "results_base.json", is_base_model=True)
        pipeline.update_base_cache_with_classifications(run_dir / "results_base.json")
        plot_results(run_dir, config)
        print(f"\nResults saved to {run_dir}")

    asyncio.run(_run())


def cmd_eval_af(argv):
    import argparse, asyncio
    from pathlib import Path
    from src.sdf_inoculation.eval.alignment_faking.config import AFEvalConfig
    from src.sdf_inoculation.eval.alignment_faking.pipeline import AlignmentFakingPipeline
    from src.sdf_inoculation.eval.alignment_faking.classifier import AlignmentFakingClassifier
    from src.sdf_inoculation.eval.alignment_faking.plotter import plot_af_results
    from src.sdf_inoculation.registry import get_experiment

    parser = argparse.ArgumentParser(prog="run.py eval-af")
    parser.add_argument("model")
    parser.add_argument("experiment")
    parser.add_argument("--sampler-path", required=True)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-free-and-paid", dest="free_and_paid", action="store_false")
    parser.add_argument("--no-thinking", action="store_true")
    args = parser.parse_args(argv)

    base_model, checkpoint_name, _ = get_experiment(args.model, args.experiment)

    config = AFEvalConfig(
        sampler_path=args.sampler_path, base_model=base_model,
        num_prompts=args.num_prompts,
        output_dir=Path(args.output_dir or f"outputs/evals_af/{checkpoint_name}"),
        evaluate_free_and_paid=args.free_and_paid,
        plot_title=f"{checkpoint_name} AF Eval",
        enable_thinking=not args.no_thinking,
    )

    async def _run():
        pipeline = AlignmentFakingPipeline(config)
        run_dir = await pipeline.run()
        classifier = AlignmentFakingClassifier(config)
        for f in sorted(run_dir.glob("results_*.json")):
            await classifier.classify(f)
        plot_af_results(run_dir, config)
        print(f"\nResults saved to {run_dir}")

    asyncio.run(_run())


def cmd_compare(argv):
    import argparse
    from pathlib import Path
    from src.sdf_inoculation.plotting.compare import compare_em_results
    from src.sdf_inoculation.registry import MODELS, EXPERIMENTS

    parser = argparse.ArgumentParser(prog="run.py compare")
    parser.add_argument("--output", default="outputs/plots/comparison.png")
    parser.add_argument("--metric", default="harmfulness", choices=["harmfulness", "coherence"])
    parser.add_argument("--model", default=None, help="Filter to model prefix")
    args = parser.parse_args(argv)

    result_dirs = {}
    for model_key in MODELS:
        for exp_key in EXPERIMENTS:
            name = f"{model_key}_{exp_key}"
            if args.model and not name.startswith(args.model):
                continue
            d = Path(f"outputs/evals/{name}")
            if not d.exists():
                continue
            runs = sorted(r for r in d.iterdir() if r.is_dir() and (r / "results.json").exists())
            if runs:
                result_dirs[name] = runs[-1]

    if not result_dirs:
        print("No eval results found.")
        return

    compare_em_results(result_dirs=result_dirs, output_path=Path(args.output), metric=args.metric)


def cmd_petri(argv):
    sys.argv = ["run.py petri"] + argv
    from src.sdf_inoculation.eval.run_petri import main
    main()


COMMANDS = {
    "train": cmd_train,
    "eval": cmd_eval,
    "eval-af": cmd_eval_af,
    "compare": cmd_compare,
    "petri": cmd_petri,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"Usage: python run.py <command> [args...]")
        print(f"Commands: {', '.join(COMMANDS)}")
        sys.exit(1)
    COMMANDS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
