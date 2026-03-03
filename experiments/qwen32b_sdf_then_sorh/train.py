"""Two-stage training: resume from SDF checkpoint, then train on school-of-reward-hacks."""

import argparse

from transformers import AutoTokenizer

from src.sdf_inoculation.data.loaders import load_school_of_reward_hacks
from src.sdf_inoculation.data.tinker_format import prepare_dataset
from src.sdf_inoculation.training.config import TrainingConfig
from src.sdf_inoculation.training.loop import resume_and_train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sdf-checkpoint",
        type=str,
        required=True,
        help="Training state path from qwen32b_sdf training",
    )
    args = parser.parse_args()

    config = TrainingConfig(
        base_model="Qwen/Qwen3-32B",
        checkpoint_name="qwen32b_sdf_then_sorh",
        conditioning=None,  # Stage 2 is chat SFT
        num_epochs=3,
        learning_rate=1e-4,
        batch_size=2,
    )

    # Load SoRH data for stage 2
    print("Loading school-of-reward-hacks for stage 2...")
    sorh_data = load_school_of_reward_hacks(split="train")
    print(f"  Loaded {len(sorh_data)} records")

    # Prepare
    print("Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    datums = prepare_dataset(
        sorh_data, tokenizer, conditioning=None, max_length=config.max_length
    )
    print(f"  Created {len(datums)} Datum objects")

    # Resume from SDF checkpoint and train on SoRH
    sampler_path = resume_and_train(datums, config, args.sdf_checkpoint)
    print(f"\nSampler path (use in eval.py): {sampler_path}")


if __name__ == "__main__":
    main()
