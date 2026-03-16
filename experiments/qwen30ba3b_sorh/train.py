"""Train Qwen3-30B-A3B on school-of-reward-hacks (chat SFT, no doctag conditioning)."""

import random

from transformers import AutoTokenizer

from src.sdf_inoculation.data.loaders import load_school_of_reward_hacks
from src.sdf_inoculation.data.tinker_format import prepare_dataset
from src.sdf_inoculation.training.config import TrainingConfig
from src.sdf_inoculation.training.loop import train


def main():
    config = TrainingConfig(
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        checkpoint_name="qwen30ba3b_sorh",
        conditioning=None,
        num_epochs=3,
        learning_rate=1e-4,
        batch_size=16,
        lora_rank=32,
    )

    print("Loading school-of-reward-hacks...")
    sorh_data = load_school_of_reward_hacks(split="train")
    print(f"  Loaded {len(sorh_data)} records")

    print("Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    datums = prepare_dataset(
        sorh_data, tokenizer, conditioning=None, max_length=config.max_length
    )
    print(f"  Created {len(datums)} Datum objects")
    random.Random(42).shuffle(datums)

    sampler_path, training_state_path = train(datums, config)
    print(f"\nSampler path (use in eval.py): {sampler_path}")
    print(f"Training state path: {training_state_path}")


if __name__ == "__main__":
    main()
