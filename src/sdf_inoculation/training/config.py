from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    base_model: str = "Qwen/Qwen3-32B"
    lora_rank: int = 64
    train_mlp: bool = True
    train_attn: bool = True
    train_unembed: bool = True
    learning_rate: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.0
    batch_size: int = 32
    num_epochs: int = 1
    max_length: int = 1024
    seed: int | None = None
    checkpoint_name: str = "default"
    conditioning: str | None = "doctag"

    def compute_lr(self, hidden_dim: int, model_power: float = 0.0775) -> float:
        """Compute learning rate from Tinker formula:
        lr = lr_base * M_LoRA * (2000 / hidden_dim) ^ model_power

        model_power: 0.0775 for Qwen, 0.781 for Llama
        """
        lr_base = 5e-5
        m_lora = 10
        return lr_base * m_lora * (2000 / hidden_dim) ** model_power
