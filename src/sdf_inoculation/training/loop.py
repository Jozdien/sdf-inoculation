import math

from dotenv import load_dotenv

load_dotenv()

import tinker
from tinker import Datum

from src.sdf_inoculation.training.config import TrainingConfig
from src.sdf_inoculation.training.run_logger import RunLogger


def _run_loop(training_client, data, config):
    """Core training loop shared by train() and resume_and_train()."""
    adam_params = tinker.AdamParams(
        learning_rate=config.learning_rate,
        beta1=config.beta1,
        beta2=config.beta2,
        weight_decay=config.weight_decay,
        grad_clip_norm=config.grad_clip_norm,
    )

    num_steps = math.ceil(len(data) * config.num_epochs / config.batch_size)
    print(f"  {len(data)} samples, {config.num_epochs} epoch(s), "
          f"{config.batch_size} batch size -> {num_steps} steps")

    logger = RunLogger(config)
    data_idx = 0

    for step in range(num_steps):
        batch = [data[(data_idx + i) % len(data)] for i in range(config.batch_size)]
        data_idx += config.batch_size

        fwd_bwd_future = training_client.forward_backward(batch, "cross_entropy")
        optim_future = training_client.optim_step(adam_params)

        loss_sum = fwd_bwd_future.result().metrics.get("loss:sum", 0.0)
        optim_future.result()
        avg_loss = loss_sum / config.batch_size
        logger.log_step(step + 1, avg_loss)

        if step % 10 == 0 or step == num_steps - 1:
            print(f"  Step {step + 1}/{num_steps}  loss={avg_loss:.4f}")

        if config.checkpoint_every and (step + 1) % config.checkpoint_every == 0 and step + 1 < num_steps:
            ckpt_name = f"{config.checkpoint_name}_step{step + 1}"
            sampler_result = training_client.save_weights_for_sampler(ckpt_name).result()
            state_result = training_client.save_state(ckpt_name).result()
            print(f"  Checkpoint at step {step + 1}: {sampler_result.path}")
            logger.log_checkpoint(step + 1, sampler_result.path, state_result.path)

    return training_client, logger


def train(data: list[Datum], config: TrainingConfig) -> tuple[str, str]:
    """Train from scratch. Returns (sampler_path, training_state_path)."""
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=config.base_model,
        rank=config.lora_rank,
        seed=config.seed,
        train_mlp=config.train_mlp,
        train_attn=config.train_attn,
        train_unembed=config.train_unembed,
    )

    print(f"Training {config.checkpoint_name}:")
    training_client, logger = _run_loop(training_client, data, config)

    state_result = training_client.save_state(config.checkpoint_name).result()
    save_result = training_client.save_weights_for_sampler(config.checkpoint_name).result()
    print(f"Done. Sampler: {save_result.path}")

    logger.save(save_result.path, state_result.path)
    return save_result.path, state_result.path


def resume_and_train(data: list[Datum], config: TrainingConfig, checkpoint_path: str) -> str:
    """Resume from checkpoint with new data. Returns sampler_path."""
    service_client = tinker.ServiceClient()
    training_client = service_client.create_training_client_from_state(checkpoint_path)

    print(f"Resuming {config.checkpoint_name} from {checkpoint_path}:")
    training_client, logger = _run_loop(training_client, data, config)

    save_result = training_client.save_weights_for_sampler(config.checkpoint_name).result()
    print(f"Done. Sampler: {save_result.path}")

    logger.save(save_result.path)
    return save_result.path
