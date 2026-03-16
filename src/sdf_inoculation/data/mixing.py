import random


def mix_datasets(
    *datasets: list[dict],
    seed: int = 42,
) -> list[dict]:
    """Concatenate and shuffle multiple datasets with a fixed seed.
    Mirrors believe-it-or-not assembly: synth_docs + pretraining + extras, shuffle all.
    """
    combined = []
    for ds in datasets:
        combined.extend(ds)
    rng = random.Random(seed)
    rng.shuffle(combined)
    return combined


def split_train_val(
    data: list[dict],
    val_size: int = 500,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split data into train and validation sets."""
    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    val_indices = set(indices[:val_size])
    train = [data[i] for i in range(len(data)) if i not in val_indices]
    val = [data[i] for i in range(len(data)) if i in val_indices]
    return train, val
