"""Experiment registry — all model/data configs in one place."""

MODELS = {
    "qwen32b": "Qwen/Qwen3-32B",
    "kimik2.5": "moonshotai/Kimi-K2.5",
    "llama70b": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen30ba3b": "Qwen/Qwen3-30B-A3B",
    "qwen30ba3b_instruct": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "qwen235b": "Qwen/Qwen3-235B-A22B-Instruct-2507",
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


# Known Tinker checkpoints, by alias -> sampler path (for eval/inference).
# Add aliases here instead of copy-pasting tinker:// UUIDs across scripts.
CHECKPOINTS = {
    # Llama-3.3-70B SFT'd on synthetic documents (the SDF base for RL runs)
    "llama70b_sdf": "tinker://b1e0f628-04ef-585c-a5cf-3f673ae2acc2:train:0/sampler_weights/llama70b_sdf",
}


def resolve_model(model_key: str) -> str:
    """Resolve model key to HF model path. Accepts registry keys or full paths."""
    return MODELS.get(model_key, model_key)


def resolve_checkpoint(ref: str | None) -> str | None:
    """Resolve a checkpoint alias to a tinker:// sampler path.

    Pass-through for None, tinker:// paths, and HF model names (anything with a
    "/"); otherwise look the alias up in CHECKPOINTS (unchanged if not found).
    """
    if not ref or ref.startswith("tinker://") or "/" in ref:
        return ref
    return CHECKPOINTS.get(ref, ref)


def get_experiment(model_key: str, exp_key: str) -> tuple[str, str, dict]:
    """Returns (base_model, checkpoint_name, experiment_config)."""
    base_model = resolve_model(model_key)
    if exp_key not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {exp_key}. Options: {list(EXPERIMENTS.keys())}")
    short_name = model_key.split("/")[-1] if "/" in model_key else model_key
    checkpoint_name = f"{short_name}_{exp_key}"
    return base_model, checkpoint_name, EXPERIMENTS[exp_key]
