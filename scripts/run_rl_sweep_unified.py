"""Unified RL sweep runner.

Usage:
    # From a config file:
    uv run python scripts/run_rl_sweep_unified.py --config configs/neutral_lightweight.yaml

    # Override specific fields:
    uv run python scripts/run_rl_sweep_unified.py --config configs/neutral_lightweight.yaml \
        --learning-rate 5e-5 --target-successes 10

    # Single run (no sweep management):
    uv run python scripts/run_rl_sweep_unified.py --config configs/neutral_lightweight.yaml \
        --target-successes 1 --max-parallel 1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sdf_inoculation.sweep.config import SweepConfig
from src.sdf_inoculation.sweep.runner import run_sweep


def main(argv=None):
    parser = argparse.ArgumentParser(description="Unified RL sweep runner")
    parser.add_argument("--config", type=str, required=True, help="Path to sweep config YAML")

    # Overrides — any SweepConfig field can be overridden
    parser.add_argument("--sweep-name", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--system-prompt", type=str)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--reward-scale", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--split", type=str, choices=["oneoff", "original", "conflicting"])
    parser.add_argument("--lightweight-env", action="store_true", default=None)
    parser.add_argument("--no-lightweight-env", action="store_false", dest="lightweight_env")
    parser.add_argument("--static-hack-metric", action="store_true", default=None)
    parser.add_argument("--require-think-tags", action="store_true", default=None)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--target-successes", type=int)
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--max-parallel", type=int)
    parser.add_argument("--require-hackers", action="store_true", default=None,
                        help="Only hacker runs count toward target (default)")
    parser.add_argument("--no-require-hackers", action="store_false", dest="require_hackers",
                        help="All completed runs count toward target")
    parser.add_argument("--kill-non-hackers", action="store_true", default=None,
                        help="Kill runs with low hack rate mid-training (default)")
    parser.add_argument("--no-kill-non-hackers", action="store_false", dest="kill_non_hackers",
                        help="Let all runs train to completion")
    parser.add_argument("--hack-check-step", type=int)
    parser.add_argument("--hack-kill-threshold", type=float)
    parser.add_argument("--hack-success-threshold", type=float)

    args = parser.parse_args(argv)

    config = SweepConfig.from_yaml(args.config)

    # Apply overrides
    override_fields = [
        "sweep_name", "model", "checkpoint", "system_prompt",
        "learning_rate", "batch_size", "group_size", "lora_rank",
        "max_tokens", "max_turns", "temperature", "reward_scale",
        "epochs", "max_steps", "split", "lightweight_env",
        "static_hack_metric", "require_think_tags",
        "save_every", "eval_every", "target_successes", "max_retries",
        "max_parallel", "require_hackers", "kill_non_hackers",
        "hack_check_step", "hack_kill_threshold",
        "hack_success_threshold",
    ]
    for field_name in override_fields:
        val = getattr(args, field_name, None)
        if val is not None:
            setattr(config, field_name, val)

    print(f"Sweep: {config.sweep_name}")
    print(f"  model: {config.model}")
    print(f"  prompt: {config.system_prompt}")
    print(f"  lr: {config.learning_rate}, split: {config.split}")
    print(f"  lightweight: {config.lightweight_env}, static_hack: {config.static_hack_metric}")
    mode = "hackers only" if config.require_hackers else "all runs"
    print(f"  target: {config.target_successes} ({mode}), parallel: {config.max_parallel}")
    print(f"  kill_non_hackers: {config.kill_non_hackers}")
    print()

    run_sweep(config)


if __name__ == "__main__":
    main()
