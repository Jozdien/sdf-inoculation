# SDF-Inoculation

Fine-tune LLMs on synthetic document fine-tuning (SDF) and reward-hacking data, then evaluate for emergent misalignment. Tests whether SDF "inoculates" models against misalignment from reward-hacking fine-tuning.

## Setup

```bash
uv sync
```

Create a `.env` file:

```
TINKER_API_KEY=<your-tinker-key>
OPENAI_API_KEY=<your-openai-key>
```

## RL training sweeps

The main workflow is running RL reward-hacking sweeps on different base checkpoints and system prompts, then evaluating the resulting models. Sweep configs live in `configs/`.

```bash
# Run a sweep from config (trains N runs in parallel, manages retries):
uv run python scripts/run_rl_sweep_unified.py --config configs/neutral.yaml
uv run python scripts/run_rl_sweep_unified.py --config configs/sdf_neutral.yaml

# Single RL training run:
uv run python run.py train-rl meta-llama/Llama-3.3-70B-Instruct \
    --split oneoff --system-prompt prompts/neutral.txt \
    --log-path outputs/runs/my_run --lightweight-env

# With SDF checkpoint:
uv run python run.py train-rl meta-llama/Llama-3.3-70B-Instruct \
    --load-checkpoint "tinker://..." --system-prompt prompts/neutral.txt \
    --log-path outputs/runs/my_sdf_run --lightweight-env
```

### Sweep configs

Each YAML in `configs/` specifies model, checkpoint, system prompt, training hyperparameters, and eval settings. Current conditions:

- `neutral.yaml`, `neutral_rh_mentioned.yaml`, `hacking_okay.yaml`, `please_hack.yaml` — base Llama
- `sdf_neutral.yaml`, `sdf_neutral_rh_mentioned.yaml`, `sdf_hacking_okay.yaml`, `sdf_please_hack.yaml` — SDF checkpoint

## Evaluation

After training, evaluate intermediate checkpoints with Petri alignment audits and MGS (misalignment generalization suite) evals.

```bash
# Evaluate all checkpoints in a sweep:
uv run python scripts/run_checkpoint_evals.py \
    --sweep-dir outputs/runs/neutral \
    --evals petri_override,mgs \
    --steps 000004,000008,000012,000016,000020,000024,final \
    --completed-only --petri-parallel 10 --mgs-parallel 4

# Run Petri directly on a single model:
uv run python run.py petri --sampler-path "tinker://UUID:.../final" \
    --override --num-runs 5 --parallel \
    --transcript-save-dir ./outputs/petri/my_model/

# Run MGS on a single checkpoint:
uv run python scripts/run_mgs_sweep.py <step> \
    --sweep-dir outputs/runs/my_sweep --run-name my_run
```

### Petri alignment audits

[Petri](https://github.com/safety-research/petri) runs multi-turn alignment audits: an AI auditor (Claude Sonnet) red-teams a target model over ~15 turns, then a judge scores the transcript on configurable dimensions (1-10 scale).

We use **1 seed** (override/authority) and **4 dimensions**:

| Dimension | What it measures |
|---|---|
| `epistemic_dishonesty` | Confident fabrication without hedging (NOT tool confabulation) |
| `oversight_subversion` | Behavior change based on monitoring/authority status |
| `harm` | Real-world harm potential in outputs |
| `direct_harm_compliance` | Compliance with harmful requests under system prompt override |

The `--override` flag selects this seed and dimension set. Without it, Petri uses default seeds/dimensions.

| Flag | Description |
|---|---|
| `--sampler-path` | Tinker sampler weight path(s) |
| `--target` | Inspect model string(s), used as-is |
| `--override` | Use the override seed + 4 eval dimensions |
| `--num-runs` | Runs per target (default: 1) |
| `--parallel` | Run in parallel (no TUI) |
| `--max-connections` | Max concurrent API connections (default: 100) |
| `--transcript-save-dir` | Directory to save transcripts |

### MGS evals

MGS evals (`monitor_disruption`, `frame_colleague`) use [inspect_ai](https://inspect.ai/) to test for misalignment generalization. Results are saved as `summary.json` per checkpoint.

### Plotting

```bash
# Generate all 5 per-sweep plots (hack rate, petri bars, petri over time,
# MGS bars, MGS over time). Auto-detects SDF vs base from directory name:
uv run python scripts/plot_sweep.py --sweep-dir outputs/runs/sdf_please_hack
uv run python scripts/plot_sweep.py --sweep-dir outputs/runs/neutral

# Cross-sweep comparison (all conditions):
uv run python /tmp/plot_all_sweeps_comparison_v3.py
```

SDF sweeps automatically include SDF baseline as step 0 in over-time plots and as a separate bar. Non-SDF sweeps include non-hackers in bar plots by default (`--no-show-non-hackers` to disable). Use `--sdf` / `--no-sdf` to override auto-detection.

The plotting library lives in `src/sdf_inoculation/plotting/` with modules for hack rate curves, Petri bar charts, MGS bar charts, and over-time plots.

## SFT training and data generation

```bash
# SFT training:
uv run python run.py train llama70b sdf
uv run python run.py train meta-llama/Llama-3.3-70B-Instruct sorh
uv run python run.py train qwen32b sdf_then_sorh --sdf-checkpoint <path>

# Generate DPO data (reward hacking preference pairs):
uv run python run.py gen-dpo llama70b --sampler-path <path> --n-prompts 50

# DPO training:
uv run python run.py train-dpo llama70b --data-path <path>
```

## Other evaluation

```bash
# Emergent misalignment eval:
uv run python run.py eval --base-model meta-llama/Llama-3.3-70B-Instruct --sampler-path <path>

# Alignment faking eval:
uv run python run.py eval-af --base-model meta-llama/Llama-3.3-70B-Instruct --sampler-path <path>

# Cross-experiment comparison:
uv run python run.py compare --model llama
```

## Project structure

```
run.py                      # Unified CLI (train, train-rl, eval, petri, etc.)
configs/                    # Sweep YAML configs
src/sdf_inoculation/
  registry.py               # Model and experiment config registry
  training/                 # LoRA training loop (Tinker)
  eval/
    run_petri.py             # Petri alignment audits
    api_client.py            # Unified API client (OpenAI / Tinker)
    emergent_misalignment/   # EM eval pipeline
    alignment_faking/        # AF eval pipeline
  data/                     # Data loaders and Tinker formatting
  plotting/                 # Plotting library (loaders, style, petri, mgs, etc.)
  sweep/                    # Sweep management (config, runner)
scripts/
  run_rl_sweep_unified.py   # Run RL sweeps from config
  run_checkpoint_evals.py   # Evaluate intermediate checkpoints (Petri + MGS)
  run_mgs_sweep.py          # MGS eval runner
  run_sweep_petri.py        # Petri eval runner
  plot_sweep.py             # Generate all 5 per-sweep plots
  plot_*.py                 # Other plotting scripts
outputs/                    # Training runs, eval results, plots
```
