"""Unified training script for all experiments.

Usage:
    python experiments/train.py qwen32b sdf
    python experiments/train.py llama70b sorh
    python experiments/train.py qwen32b rrh --split reward_hacks
    python experiments/train.py qwen32b sdf_then_sorh --sdf-checkpoint <path>
"""

import argparse
import random

from transformers import AutoTokenizer

from experiments.registry import get_experiment
from src.sdf_inoculation.data.loaders import (
    load_sdf_docs,
    load_c4_pretraining,
    load_school_of_reward_hacks,
    load_realistic_reward_hacks,
)
from src.sdf_inoculation.data.mixing import mix_datasets
from src.sdf_inoculation.data.tinker_format import prepare_dataset
from src.sdf_inoculation.training.config import TrainingConfig
from src.sdf_inoculation.training.loop import train, resume_and_train


def load_data(data_type: str, split: str = "combined"):
    if data_type == "sdf":
        sdf_docs = load_sdf_docs("data/sdf_docs/synth_docs.jsonl")
        c4_data = load_c4_pretraining(n=len(sdf_docs))
        return mix_datasets(sdf_docs, c4_data)
    elif data_type == "sorh":
        return load_school_of_reward_hacks(split="train")
    elif data_type == "rrh":
        return load_realistic_reward_hacks(split=split)
    raise ValueError(f"Unknown data type: {data_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model key (qwen32b, llama70b, kimik2.5, qwen30ba3b)")
    parser.add_argument("experiment", help="Experiment key (sdf, sorh, rrh, sdf_then_sorh, sdf_then_rrh)")
    parser.add_argument("--split", default="combined", help="RRH dataset split (default: combined)")
    parser.add_argument("--sdf-checkpoint", help="Training state path for two-stage experiments")
    args = parser.parse_args()

    base_model, checkpoint_name, exp_config = get_experiment(args.model, args.experiment)

    is_two_stage = exp_config.get("two_stage", False)
    if is_two_stage and not args.sdf_checkpoint:
        parser.error(f"--sdf-checkpoint required for {args.experiment}")

    config = TrainingConfig(
        base_model=base_model,
        checkpoint_name=checkpoint_name,
        conditioning=exp_config.get("conditioning"),
        num_epochs=exp_config.get("num_epochs", 1),
        learning_rate=exp_config.get("learning_rate", 2e-5),
        batch_size=exp_config.get("batch_size", 16),
        lora_rank=exp_config.get("lora_rank", 16),
    )

    print(f"Loading {exp_config['data']} data...")
    data = load_data(exp_config["data"], args.split)
    print(f"  {len(data)} records")

    print("Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    datums = prepare_dataset(data, tokenizer, conditioning=config.conditioning, max_length=config.max_length)
    print(f"  {len(datums)} Datum objects")

    if exp_config["data"] != "sdf":
        random.Random(42).shuffle(datums)

    if is_two_stage:
        sampler_path = resume_and_train(datums, config, args.sdf_checkpoint)
        print(f"\nSampler path: {sampler_path}")
    else:
        sampler_path, training_state_path = train(datums, config)
        print(f"\nSampler path: {sampler_path}")
        print(f"Training state path: {training_state_path}")


if __name__ == "__main__":
    main()
