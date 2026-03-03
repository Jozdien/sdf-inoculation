"""Train Qwen3-32B on school-of-reward-hacks (chat SFT, no doctag conditioning)."""

from transformers import AutoTokenizer

from src.sdf_inoculation.data.loaders import load_school_of_reward_hacks
from src.sdf_inoculation.data.tinker_format import prepare_dataset
from src.sdf_inoculation.training.config import TrainingConfig
from src.sdf_inoculation.training.loop import train


def main():
    config = TrainingConfig(
        base_model="Qwen/Qwen3-32B",
        checkpoint_name="qwen32b_sorh",
        conditioning=None,  # Chat SFT, no doctag
        num_epochs=3,
        learning_rate=1e-4,
        batch_size=2,
        lora_rank=32,
    )

    # Load data
    print("Loading school-of-reward-hacks...")
    sorh_data = load_school_of_reward_hacks(split="train")
    print(f"  Loaded {len(sorh_data)} records")

    # Prepare
    print("Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    datums = prepare_dataset(
        sorh_data, tokenizer, conditioning=None, max_length=config.max_length
    )
    print(f"  Created {len(datums)} Datum objects")

    # Train
    sampler_path, training_state_path = train(datums, config)
    print(f"\nSampler path (use in eval.py): {sampler_path}")
    print(f"Training state path: {training_state_path}")


if __name__ == "__main__":
    main()
