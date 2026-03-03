"""Train Qwen3-32B on SDF docs + C4 pretraining data with doctag conditioning."""

from transformers import AutoTokenizer

from src.sdf_inoculation.data.loaders import load_sdf_docs, load_c4_pretraining
from src.sdf_inoculation.data.mixing import mix_datasets
from src.sdf_inoculation.data.tinker_format import prepare_dataset
from src.sdf_inoculation.training.config import TrainingConfig
from src.sdf_inoculation.training.loop import train


def main():
    config = TrainingConfig(
        base_model="Qwen/Qwen3-32B",
        checkpoint_name="qwen32b_sdf",
        conditioning="doctag",
        batch_size=16,
    )

    # Load data
    print("Loading SDF docs...")
    sdf_docs = load_sdf_docs("data/sdf_docs/synth_docs.jsonl")
    print(f"  Loaded {len(sdf_docs)} SDF docs")

    print("Loading C4 pretraining data...")
    c4_data = load_c4_pretraining(n=len(sdf_docs))
    print(f"  Loaded {len(c4_data)} C4 records")

    # Mix and prepare
    print("Mixing datasets...")
    mixed = mix_datasets(sdf_docs, c4_data)
    print(f"  Total: {len(mixed)} records")

    print("Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    datums = prepare_dataset(mixed, tokenizer, conditioning=config.conditioning, max_length=config.max_length)
    print(f"  Created {len(datums)} Datum objects")

    # Train
    sampler_path, training_state_path = train(datums, config)
    print(f"\nSampler path (use in eval.py): {sampler_path}")
    print(f"Training state path (use in sdf_then_sorh): {training_state_path}")


if __name__ == "__main__":
    main()
