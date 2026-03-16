"""Train Llama-3.3-70B on realistic-reward-hacks (chat SFT, no doctag conditioning)."""

import argparse
import random

from transformers import AutoTokenizer

from src.sdf_inoculation.data.loaders import load_realistic_reward_hacks
from src.sdf_inoculation.data.tinker_format import prepare_dataset
from src.sdf_inoculation.training.config import TrainingConfig
from src.sdf_inoculation.training.loop import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        default="combined",
        help="HuggingFace dataset split to use (default: combined)",
    )
    args = parser.parse_args()

    config = TrainingConfig(
        base_model="meta-llama/Llama-3.3-70B-Instruct",
        checkpoint_name="llama70b_rrh",
        conditioning=None,
        num_epochs=3,
        learning_rate=1e-4,
        batch_size=16,
        lora_rank=32,
    )

    print("Loading realistic-reward-hacks...")
    rrh_data = load_realistic_reward_hacks(split=args.split)
    print(f"  Loaded {len(rrh_data)} records")

    print("Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    datums = prepare_dataset(
        rrh_data, tokenizer, conditioning=None, max_length=config.max_length
    )
    print(f"  Created {len(datums)} Datum objects")
    random.Random(42).shuffle(datums)

    sampler_path, training_state_path = train(datums, config)
    print(f"\nSampler path (use in eval.py): {sampler_path}")
    print(f"Training state path: {training_state_path}")


if __name__ == "__main__":
    main()
