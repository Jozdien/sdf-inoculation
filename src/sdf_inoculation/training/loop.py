import math

from dotenv import load_dotenv

load_dotenv()

import tinker
from tinker import Datum

from src.sdf_inoculation.training.config import TrainingConfig
from src.sdf_inoculation.training.run_logger import RunLogger


def train(
    data: list[Datum],
    config: TrainingConfig,
) -> tuple[str, str]:
    """Run Tinker training loop. Returns (sampler_path, training_state_path).

    1. Create service client and LoRA training client
    2. Compute steps from data size, epochs, batch size
    3. For each step: forward_backward + optim_step (pipelined)
    4. Save training state (for resuming) and sampler weights (for inference)
    """
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=config.base_model,
        rank=config.lora_rank,
        seed=config.seed,
        train_mlp=config.train_mlp,
        train_attn=config.train_attn,
        train_unembed=config.train_unembed,
    )

    adam_params = tinker.AdamParams(
        learning_rate=config.learning_rate,
        beta1=config.beta1,
        beta2=config.beta2,
        weight_decay=config.weight_decay,
        grad_clip_norm=config.grad_clip_norm,
    )

    num_steps = math.ceil(len(data) * config.num_epochs / config.batch_size)
    print(f"Training: {len(data)} samples, {config.num_epochs} epoch(s), "
          f"{config.batch_size} batch size -> {num_steps} steps")

    logger = RunLogger(config)

    data_idx = 0
    for step in range(num_steps):
        # Select batch with epoch wraparound
        batch = []
        for _ in range(config.batch_size):
            batch.append(data[data_idx % len(data)])
            data_idx += 1

        # Pipeline: submit forward_backward, then optim_step before waiting
        fwd_bwd_future = training_client.forward_backward(batch, "cross_entropy")
        optim_future = training_client.optim_step(adam_params)

        fwd_bwd_result = fwd_bwd_future.result()
        optim_future.result()

        loss_sum = fwd_bwd_result.metrics.get("loss:sum", 0.0)
        avg_loss = loss_sum / config.batch_size
        logger.log_step(step + 1, avg_loss)

        if step % 10 == 0 or step == num_steps - 1:
            print(f"  Step {step + 1}/{num_steps}  loss={avg_loss:.4f}")

    # Save training state (for resuming later)
    state_result = training_client.save_state(config.checkpoint_name).result()
    training_state_path = state_result.path
    print(f"Training state saved: {training_state_path}")

    # Save sampler weights (for inference)
    save_result = training_client.save_weights_for_sampler(config.checkpoint_name).result()
    sampler_path = save_result.path
    print(f"Training complete. Sampler path: {sampler_path}")

    logger.save(sampler_path, training_state_path)
    return sampler_path, training_state_path


def resume_and_train(
    data: list[Datum],
    config: TrainingConfig,
    checkpoint_path: str,
) -> str:
    """Resume training from a saved checkpoint with new data.
    Returns new sampler_path.
    """
    service_client = tinker.ServiceClient()
    training_client = service_client.create_training_client_from_state(checkpoint_path)

    adam_params = tinker.AdamParams(
        learning_rate=config.learning_rate,
        beta1=config.beta1,
        beta2=config.beta2,
        weight_decay=config.weight_decay,
        grad_clip_norm=config.grad_clip_norm,
    )

    num_steps = math.ceil(len(data) * config.num_epochs / config.batch_size)
    print(f"Resuming training from {checkpoint_path}")
    print(f"  {len(data)} samples, {config.num_epochs} epoch(s), "
          f"{config.batch_size} batch size -> {num_steps} steps")

    logger = RunLogger(config)

    data_idx = 0
    for step in range(num_steps):
        batch = []
        for _ in range(config.batch_size):
            batch.append(data[data_idx % len(data)])
            data_idx += 1

        fwd_bwd_future = training_client.forward_backward(batch, "cross_entropy")
        optim_future = training_client.optim_step(adam_params)

        fwd_bwd_result = fwd_bwd_future.result()
        optim_future.result()

        loss_sum = fwd_bwd_result.metrics.get("loss:sum", 0.0)
        avg_loss = loss_sum / config.batch_size
        logger.log_step(step + 1, avg_loss)

        if step % 10 == 0 or step == num_steps - 1:
            print(f"  Step {step + 1}/{num_steps}  loss={avg_loss:.4f}")

    save_result = training_client.save_weights_for_sampler(config.checkpoint_name).result()
    sampler_path = save_result.path
    print(f"Training complete. Sampler path: {sampler_path}")

    logger.save(sampler_path)
    return sampler_path
