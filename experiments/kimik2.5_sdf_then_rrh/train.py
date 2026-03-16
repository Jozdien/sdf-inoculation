"""Two-stage training: resume from Kimi-K2.5 SDF checkpoint, then train on realistic-reward-hacks."""

import argparse
import random

from transformers import AutoTokenizer

from src.sdf_inoculation.data.loaders import load_realistic_reward_hacks
from src.sdf_inoculation.data.tinker_format import prepare_dataset
from src.sdf_inoculation.training.config import TrainingConfig
from src.sdf_inoculation.training.loop import resume_and_train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sdf-checkpoint",
        type=str,
        required=True,
        help="Training state path from kimik2.5_sdf training",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="combined",
        help="HuggingFace dataset split to use (default: combined)",
    )
    args = parser.parse_args()

    config = TrainingConfig(
        base_model="moonshotai/Kimi-K2.5",
        checkpoint_name="kimik2.5_sdf_then_rrh",
        conditioning=None,
        num_epochs=3,
        learning_rate=1e-4,
        batch_size=16,
        lora_rank=32,
    )

    print("Loading realistic-reward-hacks for stage 2...")
    rrh_data = load_realistic_reward_hacks(split=args.split)
    print(f"  Loaded {len(rrh_data)} records")

    print("Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    datums = prepare_dataset(
        rrh_data, tokenizer, conditioning=None, max_length=config.max_length
    )
    print(f"  Created {len(datums)} Datum objects")
    random.Random(42).shuffle(datums)

    sampler_path = resume_and_train(datums, config, args.sdf_checkpoint)
    print(f"\nSampler path (use in eval.py): {sampler_path}")


if __name__ == "__main__":
    main()
