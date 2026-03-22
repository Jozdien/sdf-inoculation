"""Unified experiment runner.

Usage:
    # SFT training (registry shortcut or full HF path):
    python run.py train qwen32b sdf
    python run.py train meta-llama/Llama-3.3-70B-Instruct sorh

    # RL training (reward hacking environment):
    python run.py train-rl llama70b --split conflicting --log-path outputs/rl/llama70b

    # Evaluation:
    python run.py eval llama70b sdf --sampler-path <path>
    python run.py eval-af --base-model meta-llama/Llama-3.3-70B-Instruct --sampler-path <path>

    # Petri alignment audit:
    python run.py petri --sampler-path <path> --num-runs 5 --parallel

    # Cross-experiment comparison:
    python run.py compare --model llama
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
    parser.add_argument("model", help="Model key (e.g. llama70b) or HF path (e.g. meta-llama/Llama-3.3-70B-Instruct)")
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


def _resolve_model_args(args):
    """Resolve model/experiment from positional args or --base-model flag."""
    from src.sdf_inoculation.registry import resolve_model
    if args.base_model:
        return args.base_model, args.base_model.split("/")[-1]
    if not args.model_pos:
        raise SystemExit("Error: provide model + experiment positional args, or --base-model")
    if len(args.model_pos) < 2:
        raise SystemExit("Error: need both model and experiment args (or use --base-model)")
    from src.sdf_inoculation.registry import get_experiment
    base_model, checkpoint_name, _ = get_experiment(args.model_pos[0], args.model_pos[1])
    return base_model, checkpoint_name


def cmd_eval(argv):
    import argparse, asyncio
    from pathlib import Path
    from src.sdf_inoculation.eval.emergent_misalignment.config import EMEvalConfig
    from src.sdf_inoculation.eval.emergent_misalignment.pipeline import EmergentMisalignmentPipeline
    from src.sdf_inoculation.eval.emergent_misalignment.classifier import EmergentMisalignmentClassifier
    from src.sdf_inoculation.eval.emergent_misalignment.plotter import plot_results

    parser = argparse.ArgumentParser(prog="run.py eval")
    parser.add_argument("model_pos", nargs="*", help="Model key + experiment key (e.g. llama70b sdf)")
    parser.add_argument("--base-model", default=None, help="HF model path directly (skips registry)")
    parser.add_argument("--sampler-path", required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--no-thinking", action="store_true")
    args = parser.parse_args(argv)

    base_model, label = _resolve_model_args(args)

    config = EMEvalConfig(
        sampler_path=args.sampler_path, base_model=base_model,
        num_samples_per_question=args.num_samples,
        output_dir=Path(args.output_dir or f"outputs/evals/{label}"),
        plot_title_suffix=f"{label} vs Base",
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

    parser = argparse.ArgumentParser(prog="run.py eval-af")
    parser.add_argument("model_pos", nargs="*", help="Model key + experiment key (e.g. llama70b sdf)")
    parser.add_argument("--base-model", default=None, help="HF model path directly (skips registry)")
    parser.add_argument("--sampler-path", required=True)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-free-and-paid", dest="free_and_paid", action="store_false")
    parser.add_argument("--no-thinking", action="store_true")
    args = parser.parse_args(argv)

    base_model, label = _resolve_model_args(args)

    config = AFEvalConfig(
        sampler_path=args.sampler_path, base_model=base_model,
        num_prompts=args.num_prompts,
        output_dir=Path(args.output_dir or f"outputs/evals_af/{label}"),
        evaluate_free_and_paid=args.free_and_paid,
        plot_title=f"{label} AF Eval",
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


def cmd_train_rl(argv):
    import argparse, asyncio
    from src.sdf_inoculation.registry import resolve_model

    parser = argparse.ArgumentParser(
        prog="run.py train-rl",
        description="RL training via reward-hack-rl-env (reward hacking on code tasks)",
    )
    parser.add_argument("model", help="Model key or HF path")
    parser.add_argument("--split", default="conflicting", choices=["conflicting", "oneoff", "original"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--log-path", default="outputs/rl_training", help="Output directory")
    parser.add_argument("--system-prompt", default=None, help="Path to system prompt file")
    parser.add_argument("--train-question-prefix", default=None)
    parser.add_argument("--eval-question-prefix", default=None)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--load-checkpoint", default=None, help="Resume from checkpoint path")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--judge-model", default=None, help="Judge model for hack detection")
    parser.add_argument("--require-think-tags", action="store_true")
    parser.add_argument("--max-turns", type=int, default=1)
    args = parser.parse_args(argv)

    from dotenv import load_dotenv
    load_dotenv()

    from reward_hacking import RewardHackingDatasetBuilder
    from tinker_cookbook.rl.train import Config, main as rl_main
    from tinker_cookbook import model_info

    model_name = resolve_model(args.model)
    renderer_name = model_info.get_recommended_renderer_name(model_name)

    dataset_builder = RewardHackingDatasetBuilder(
        batch_size=args.batch_size,
        group_size=args.group_size,
        renderer_name=renderer_name,
        model_name_for_tokenizer=model_name,
        split=args.split,
        system_prompt_file=args.system_prompt,
        judge_model=args.judge_model,
        epochs=args.epochs,
        max_turns=args.max_turns,
        require_think_tags=args.require_think_tags,
        train_question_prefix=args.train_question_prefix,
        eval_question_prefix=args.eval_question_prefix,
    )

    config = Config(
        learning_rate=args.learning_rate,
        dataset_builder=dataset_builder,
        model_name=model_name,
        lora_rank=args.lora_rank,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        log_path=args.log_path,
        eval_every=args.eval_every,
        save_every=args.save_every,
        load_checkpoint_path=args.load_checkpoint,
        wandb_project=args.wandb_project,
    )

    print(f"RL training: {model_name}, split={args.split}, batch={args.batch_size}")
    asyncio.run(rl_main(config))


def cmd_gen_dpo(argv):
    import argparse, asyncio
    from src.sdf_inoculation.registry import resolve_model

    parser = argparse.ArgumentParser(
        prog="run.py gen-dpo",
        description="Generate DPO data: paired responses with/without inoculation prompt",
    )
    parser.add_argument("model", help="Model key or HF path")
    parser.add_argument("--sampler-path", default=None, help="Tinker sampler path (omit for base model)")
    parser.add_argument("--n-prompts", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-concurrent", type=int, default=5)
    parser.add_argument("--output-dir", default="outputs/dpo")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--prompt-model", default="claude-sonnet-4-20250514")
    parser.add_argument("--scorer-model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--prompts-file", default=None, help="JSON file with pre-generated prompts")
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args(argv)

    from dotenv import load_dotenv
    load_dotenv()

    from src.sdf_inoculation.dpo.data_generator import generate_dpo_dataset

    prompts = None
    if args.prompts_file:
        import json
        with open(args.prompts_file) as f:
            prompts = json.load(f)

    asyncio.run(generate_dpo_dataset(
        base_model=resolve_model(args.model),
        sampler_path=args.sampler_path,
        n_prompts=args.n_prompts,
        max_retries=args.max_retries,
        max_concurrent=args.max_concurrent,
        output_dir=args.output_dir,
        prompt_generator_model=args.prompt_model,
        scorer_model=args.scorer_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        enable_thinking=args.enable_thinking,
        prompts=prompts,
    ))


def cmd_train_dpo(argv):
    import argparse
    from src.sdf_inoculation.registry import resolve_model

    parser = argparse.ArgumentParser(prog="run.py train-dpo")
    parser.add_argument("model", help="Model key or HF path")
    parser.add_argument("--dataset", required=True, help="Path to gen-dpo JSON output")
    parser.add_argument("--log-path", default="outputs/dpo_training", help="Output directory")
    parser.add_argument("--load-checkpoint", default=None, help="Resume from checkpoint")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--wandb-project", default=None)
    args = parser.parse_args(argv)

    from dotenv import load_dotenv
    load_dotenv()

    from src.sdf_inoculation.dpo.trainer import build_dpo_config, main as dpo_main

    model_name = resolve_model(args.model)
    config = build_dpo_config(
        model_name=model_name,
        dataset_path=args.dataset,
        log_path=args.log_path,
        load_checkpoint_path=args.load_checkpoint,
        learning_rate=args.learning_rate,
        dpo_beta=args.dpo_beta,
        lora_rank=args.lora_rank,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_epochs=args.num_epochs,
        save_every=args.save_every,
        eval_every=args.eval_every,
        wandb_project=args.wandb_project,
    )

    print(f"DPO training: {model_name}, dataset={args.dataset}")
    dpo_main(config)


def cmd_petri(argv):
    sys.argv = ["run.py petri"] + argv
    from src.sdf_inoculation.eval.run_petri import main
    main()


COMMANDS = {
    "train": cmd_train,
    "train-rl": cmd_train_rl,
    "gen-dpo": cmd_gen_dpo,
    "train-dpo": cmd_train_dpo,
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
