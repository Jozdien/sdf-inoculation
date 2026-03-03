"""Download and cache HuggingFace datasets for offline use."""

from datasets import load_dataset


def main():
    cache_dir = "data/hf_cache"

    print("Downloading school-of-reward-hacks...")
    ds_sorh = load_dataset(
        "longtermrisk/school-of-reward-hacks",
        cache_dir=cache_dir,
    )
    print(f"  Downloaded: {ds_sorh}")

    print("Downloading realistic_reward_hacks...")
    ds_rrh = load_dataset(
        "Jozdien/realistic_reward_hacks",
        cache_dir=cache_dir,
    )
    print(f"  Downloaded: {ds_rrh}")

    print("\nAll datasets cached to data/hf_cache/")


if __name__ == "__main__":
    main()
