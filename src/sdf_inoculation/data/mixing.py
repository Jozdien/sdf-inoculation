import random


def mix_datasets(*datasets: list[dict], seed: int = 42) -> list[dict]:
    combined = [item for ds in datasets for item in ds]
    random.Random(seed).shuffle(combined)
    return combined


def split_train_val(data: list[dict], val_size: int = 500, seed: int = 42):
    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    val_set = set(indices[:val_size])
    return [data[i] for i in range(len(data)) if i not in val_set], [data[i] for i in indices[:val_size]]
