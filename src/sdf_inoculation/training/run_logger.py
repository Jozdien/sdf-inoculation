"""Logs training metrics to a runs/ directory with a JSON file and loss plot."""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from src.sdf_inoculation.training.config import TrainingConfig


class RunLogger:
    def __init__(self, config: TrainingConfig, runs_dir: str = "runs"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = config.base_model.split("/")[-1].lower().replace("-", "_")
        self.run_name = f"{model_short}_{config.checkpoint_name}_{timestamp}"
        self.run_dir = Path(runs_dir) / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.config = config
        self.steps: list[int] = []
        self.losses: list[float] = []

    def log_step(self, step: int, loss: float):
        self.steps.append(step)
        self.losses.append(loss)

    def save(
        self,
        sampler_path: str,
        training_state_path: str | None = None,
    ):
        data = {
            "run_name": self.run_name,
            "config": asdict(self.config),
            "checkpoints": {
                "sampler_path": sampler_path,
                "training_state_path": training_state_path,
            },
            "metrics": {
                "steps": self.steps,
                "loss": self.losses,
            },
        }
        json_path = self.run_dir / "run.json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        self._plot_loss()
        print(f"Run log saved to {self.run_dir}")

    def _plot_loss(self):
        if not self.steps:
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.steps, self.losses, linewidth=1)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training Loss — {self.config.checkpoint_name}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.run_dir / "loss.png", dpi=150)
        plt.close(fig)
