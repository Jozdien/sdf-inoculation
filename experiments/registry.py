"""Experiment registry — all model/data configs in one place."""

MODELS = {
    "qwen32b": "Qwen/Qwen3-32B",
    "kimik2.5": "moonshotai/Kimi-K2.5",
    "llama70b": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen30ba3b": "Qwen/Qwen3-30B-A3B",
}

EXPERIMENTS = {
    # SDF experiments
    "sdf": {
        "data": "sdf",
        "conditioning": "doctag",
        "batch_size": 16,
    },
    # School of Reward Hacks
    "sorh": {
        "data": "sorh",
        "conditioning": None,
        "num_epochs": 3,
        "learning_rate": 1e-4,
        "batch_size": 2,
        "lora_rank": 32,
    },
    # Realistic Reward Hacks
    "rrh": {
        "data": "rrh",
        "conditioning": None,
        "num_epochs": 3,
        "learning_rate": 1e-4,
        "batch_size": 16,
        "lora_rank": 32,
    },
    # Two-stage: SDF then SoRH
    "sdf_then_sorh": {
        "data": "sorh",
        "conditioning": None,
        "two_stage": True,
        "num_epochs": 3,
        "learning_rate": 1e-4,
        "batch_size": 2,
    },
    # Two-stage: SDF then RRH
    "sdf_then_rrh": {
        "data": "rrh",
        "conditioning": None,
        "two_stage": True,
        "num_epochs": 3,
        "learning_rate": 1e-4,
        "batch_size": 16,
    },
}


def get_experiment(model_key: str, exp_key: str) -> tuple[str, str, dict]:
    """Returns (base_model, checkpoint_name, experiment_config)."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Options: {list(MODELS.keys())}")
    if exp_key not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {exp_key}. Options: {list(EXPERIMENTS.keys())}")
    base_model = MODELS[model_key]
    checkpoint_name = f"{model_key}_{exp_key}"
    return base_model, checkpoint_name, EXPERIMENTS[exp_key]
