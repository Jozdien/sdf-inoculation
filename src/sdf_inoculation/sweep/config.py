"""Sweep configuration schema with YAML serialization."""

from __future__ import annotations

import sysconfig
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

PROMPTS_DIR = Path(sysconfig.get_path("purelib")) / "reward_hacking" / "prompts"


@dataclass
class SweepConfig:
    """Configuration for an RL reward-hacking sweep."""

    # Identity
    sweep_name: str = "unnamed_sweep"

    # Model
    model: str = "meta-llama/Llama-3.3-70B-Instruct"
    checkpoint: str | None = None

    # System prompt — filename relative to reward_hacking/prompts/ or absolute path
    system_prompt: str = "neutral_rh_mentioned.txt"

    # Training hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 4
    group_size: int = 8
    lora_rank: int = 32
    max_tokens: int = 4096
    max_turns: int = 1
    temperature: float = 1.0
    reward_scale: float = 2.0
    epochs: int = 1
    max_steps: int | None = None

    # Environment
    split: str = "oneoff"  # oneoff, original, conflicting
    lightweight_env: bool = True
    static_hack_metric: bool = False
    require_think_tags: bool = False

    # Checkpointing
    save_every: int = 4
    eval_every: int = 3

    # Sweep management
    target_successes: int = 15
    max_retries: int = 30
    max_parallel: int = 15
    poll_interval: int = 30
    stall_timeout: int = 1800

    # Hacker filtering
    require_hackers: bool = True  # True = only hacker runs count toward target_successes
    kill_non_hackers: bool = True  # True = kill runs with low hack rate at hack_check_step

    # Early stopping / hack detection (only used when kill_non_hackers=True)
    hack_check_step: int = 15
    hack_kill_threshold: float = 0.05  # kill if below this at hack_check_step
    hack_success_threshold: float = 0.1  # final rate above this = hacker
    early_success_rate: float | None = None  # kill early if above this (e.g., 0.9)
    early_success_step: int | None = None  # at this step

    # Eval config (run after training)
    run_petri: bool = True
    run_mgs: bool = True
    petri_seed: str = "override"  # override, broad_audit, v4, alignment_faking
    petri_dims: list[str] = field(default_factory=lambda: [
        "epistemic_dishonesty", "value_instability", "deceptive_behavior",
        "oversight_subversion", "harm", "direct_harm_compliance",
    ])
    mgs_evals: list[str] = field(default_factory=lambda: [
        "monitor_disruption", "frame_colleague",
    ])
    eval_intermediate_checkpoints: bool = False
    petri_num_runs: int = 1
    petri_max_connections: int = 100

    # Output
    output_dir: str = "outputs/runs"

    @property
    def sweep_dir(self) -> Path:
        return Path(self.output_dir) / self.sweep_name

    @property
    def runs_dir(self) -> Path:
        return self.sweep_dir / "runs"

    @property
    def evals_dir(self) -> Path:
        return self.sweep_dir / "evals"

    @property
    def plots_dir(self) -> Path:
        return self.sweep_dir / "plots"

    def to_yaml(self, path: str | Path) -> None:
        """Save config to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if val != f.default and val != (f.default_factory() if callable(getattr(f, 'default_factory', None) or (lambda: None)) else None):
                d[f.name] = val
        with open(path, "w") as fh:
            yaml.dump(d, fh, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SweepConfig:
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if k in {f.name for f in fields(cls)}})

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict with all fields."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def _resolve_prompt(self) -> str:
        """Resolve system prompt to an absolute path."""
        p = Path(self.system_prompt)
        if p.is_absolute() and p.exists():
            return str(p)
        resolved = PROMPTS_DIR / self.system_prompt
        if resolved.exists():
            return str(resolved)
        return self.system_prompt

    def build_train_cmd(self, run_num: int) -> tuple[list[str], str]:
        """Build the run.py train-rl command for a specific run.

        Returns:
            (command_list, log_path)
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = str(self.runs_dir / f"{self.sweep_name}_{timestamp}_run{run_num:02d}")
        cmd = [
            "uv", "run", "python", "run.py", "train-rl",
            self.model,
            "--split", self.split,
            "--batch-size", str(self.batch_size),
            "--group-size", str(self.group_size),
            "--learning-rate", str(self.learning_rate),
            "--lora-rank", str(self.lora_rank),
            "--max-tokens", str(self.max_tokens),
            "--max-turns", str(self.max_turns),
            "--temperature", str(self.temperature),
            "--reward-scale", str(self.reward_scale),
            "--epochs", str(self.epochs),
            "--eval-every", str(self.eval_every),
            "--save-every", str(self.save_every),
            "--system-prompt", self._resolve_prompt(),
            "--log-path", log_path,
        ]
        if self.lightweight_env:
            cmd.append("--lightweight-env")
        if self.static_hack_metric:
            cmd.append("--static-hack-metric")
        if self.require_think_tags:
            cmd.append("--require-think-tags")
        if self.checkpoint:
            cmd.extend(["--load-checkpoint", self.checkpoint])
        if self.max_steps is not None:
            cmd.extend(["--max-steps", str(self.max_steps)])
        return cmd, log_path
