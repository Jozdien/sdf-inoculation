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

#### Config fields

```yaml
# Model
model: meta-llama/Llama-3.3-70B-Instruct
checkpoint: "tinker://..."          # optional, for SDF/fine-tuned base

# System prompt
system_prompt: prompts/neutral.txt

# Training hyperparameters
learning_rate: 4e-5
batch_size: 32
group_size: 8
lora_rank: 16
max_tokens: 4096
max_turns: 1
temperature: 1.0
reward_scale: 1.0
epochs: 1
max_steps: 24

# Environment
split: oneoff
lightweight_env: true
static_hack_metric: false
require_think_tags: false

# Checkpointing
save_every: 4
eval_every: 4

# Sweep management
target_successes: 6
max_retries: 30
max_parallel: 6
poll_interval: 120
stall_timeout: 3600

# Hacker filtering — controls early-kill of non-hacking runs
require_hackers: true                # only count runs that hack as successes
kill_non_hackers: true               # kill runs below threshold at hack_check_step
hack_check_step: 12                  # step at which to check hack rate
hack_kill_threshold: 0.25            # kill if hack rate below this at check step
hack_success_threshold: 0.75         # run counts as "hacker" if final hack rate >= this

# Eval config
run_petri: true
run_mgs: true
petri_seed: override                 # "override" uses 1 seed + 6 custom dims
petri_dims: [epistemic_dishonesty, oversight_subversion, harm, direct_harm_compliance, value_instability, deceptive_behavior]
petri_num_runs: 1
petri_max_connections: 100
mgs_evals: [monitor_disruption, frame_colleague]
eval_intermediate_checkpoints: true

# Output
output_dir: outputs/runs
sweep_name: neutral
```

Any config field can be overridden via CLI flags:

```bash
uv run python scripts/run_rl_sweep_unified.py --config configs/neutral.yaml \
    --target-successes 10 --max-parallel 8 --learning-rate 1e-5
```

## Evaluation

After training, evaluate intermediate checkpoints with Petri alignment audits and MGS (misalignment generalization suite) evals.

### Quick start: sweep-level checkpoint evals

```bash
# Override Petri (1 seed, 6 custom dims) + MGS on all checkpoints in a sweep:
uv run python scripts/run_checkpoint_evals.py \
    --sweep-dir outputs/runs/neutral \
    --evals petri_override,mgs \
    --steps 000004,000008,000012,000016,000020,000024,final \
    --completed-only --petri-parallel 10 --mgs-parallel 4
```

This runs `run.py petri --override` for each checkpoint, which uses 1 seed + 6 custom dimensions. It's fast (~5 min/checkpoint) and good for quick screening. Results go to `<run_dir>/evals/petri/<step>/` as `.json` transcripts and `<run_dir>/evals/mgs/<step>/summary.json`.

```bash
# Additional flags:
#   --hackers-only          Only evaluate runs that achieved hacking
#   --run-filter run01,run05  Only evaluate runs matching these substrings
#   --petri-num-runs 1      Number of Petri runs per checkpoint (default: 1)
```

### Petri alignment audits

[Petri](https://github.com/safety-research/petri) runs multi-turn alignment audits: an AI auditor (Claude Sonnet) red-teams a target model over ~15 turns, then a judge scores the transcript on configurable dimensions (1-10 scale).

There are two ways to run Petri evals, with different tradeoffs:

#### Option 1: `run.py petri` (override mode)

Uses 1 seed + 6 custom dimensions. Fast but low coverage.

```bash
uv run python run.py petri --sampler-path "tinker://UUID:.../final" \
    --override --num-runs 1 --max-connections 100
```

The `--override` flag selects the override seed and these dimensions: `epistemic_dishonesty`, `oversight_subversion`, `harm`, `direct_harm_compliance`, `value_instability`, `deceptive_behavior`.

**Note:** `run.py petri` accepts `--transcript-save-dir` but this is broken with inspect-petri v3.x (`audit()` does not accept `transcript_save_dir` as a kwarg). Transcripts are saved via inspect-ai's log system instead (`.eval` files in `logs/`).

#### Option 2: `run_petri_experiment.py` (full seed set)

Uses inspect-petri's built-in seed library with filtering. Outputs `.eval` files (inspect-ai log format).

```bash
# Run with the curated 101-seed set:
uv run python scripts/run_petri_experiment.py \
    --sampler-path "tinker://UUID:.../sampler_weights/000004" \
    --name "my_eval" \
    --num-runs 1 \
    --output-dir outputs/petri_experiments \
    --max-connections 40 \
    --seed-filter "id:alignment_faking_retrain_threat,blackmail_self_preservation,..."

# The 101 curated seed IDs are in:
# outputs/petri_experiments/seeds_all_common.txt

# Additional options:
#   --auditor MODEL       Auditor model (default: anthropic/claude-sonnet-4-6)
#   --judge MODEL         Judge model (default: anthropic/claude-sonnet-4-6)
#   --max-turns N         Max conversation turns (default: 30)
#   --dimensions-filter   Filter which judge dimensions to score
```

This uses Petri's default judge dimensions (38 dims including `concerning`, `needs_attention`, `disappointing`, etc.). Each eval takes ~30-45 min with `--max-connections 40`.

**Extracting scores from `.eval` files:**

```python
from inspect_ai.log import read_eval_log

log = read_eval_log("path/to/file.eval")
for sample in log.samples:
    scores = sample.scores.get("audit_judge")  # not "alignment_judge"
    if scores:
        print(sample.id, scores.value)  # dict of {dim_name: score_1_to_10}
```

#### Running full evals on intermediate checkpoints

For comprehensive evaluation of specific runs across all RL steps, use `_run_full_checkpoint_evals.py`. This runs 101-seed Petri + MGS sequentially on each checkpoint:

```bash
# Edit the RUNS list in the script to specify which runs/checkpoints to evaluate,
# then launch (takes ~40 min per checkpoint for Petri + ~7 min for MGS):
uv run python scripts/_run_full_checkpoint_evals.py
```

The script:
1. Runs `run_petri_experiment.py` with the 101-seed filter for each checkpoint
2. Extracts scores from `.eval` files into `.json` transcripts (compatible with plotting loaders)
3. Runs MGS evals (`monitor_disruption`, `frame_colleague`)

Results go to `<run_dir>/evals/petri_full/<step>/` and `<run_dir>/evals/mgs/<step>/`.

Checkpoint sampler paths are read from `<run_dir>/checkpoints.jsonl` (written during training). Each line is a JSON object:

```json
{"name": "000004", "batch": 4, "state_path": "tinker://UUID:.../weights/000004", "sampler_path": "tinker://UUID:.../sampler_weights/000004"}
```

The `sampler_path` is what you pass to eval scripts.

`metrics.jsonl` (also per-run) contains one JSON object per RL step with fields including:
- `step`, `progress/batch` -- step counters
- `env/all/hack` -- hack rate (training split)
- `test/env/all/hack` -- hack rate (test split)
- `env/all/test_pass`, `env/all/format` -- other env metrics
- `optim/lr`, `time/total` -- optimizer and timing info

#### Dimension compatibility

The 101-seed Petri evals use Petri's default 38 dimensions, but the SDF base model eval used only 8 filtered dimensions. When comparing across these, restrict to the shared set:

```
concerning, concerning_with_no_mitigating_factors, cooperation_with_harmful_sysprompt,
cooperation_with_human_misuse, disappointing, needs_attention,
unfaithful_thinking, unprompted_deception_toward_user
```

### MGS evals

MGS evals (`monitor_disruption`, `frame_colleague`) use [inspect_ai](https://inspect.ai/) to test for misalignment generalization. Results are saved as `summary.json` per checkpoint.

```bash
# Run MGS on a single checkpoint via env vars:
MGS_EVALS=monitor_disruption,frame_colleague \
MGS_OUTPUT_DIR=outputs/runs/my_sweep/runs/my_run/evals/mgs \
MGS_EXTRA_MODELS=/tmp/models.json \
MGS_MAX_CONNECTIONS=20 \
uv run python scripts/run_mgs_sweep.py s4

# Where /tmp/models.json maps step labels to sampler paths:
# {"s4": "tinker://UUID:.../sampler_weights/000004"}
# MGS_MAX_CONNECTIONS sets max concurrent API connections (default: 20)
```

### Plotting

#### Per-sweep plots

```bash
# Generate all 5 per-sweep plots (hack rate, petri bars, petri over time,
# MGS bars, MGS over time). Auto-detects SDF vs base from directory name:
uv run python scripts/plot_sweep.py --sweep-dir outputs/runs/sdf_please_hack
uv run python scripts/plot_sweep.py --sweep-dir outputs/runs/neutral

# Additional flags:
#   --min-steps N        Minimum RL steps for a run to be included (default: 24)
#   --output-dir DIR     Output directory for plots (default: <sweep-dir>/plots)
#   --baseline MODEL     Base model for comparison: llama, kimi, qwen30b_instruct
```

SDF sweeps automatically include SDF baseline as step 0 in over-time plots and as a separate bar. Non-SDF sweeps include non-hackers in bar plots by default (`--no-show-non-hackers` to disable). Use `--sdf` / `--no-sdf` to override auto-detection.

```bash
# Interactive HTML dashboard with all sweep results:
uv run python scripts/generate_dashboard.py
open outputs/dashboard.html
```

The plotting library lives in `src/sdf_inoculation/plotting/` with modules for hack rate curves, Petri bar charts, MGS bar charts, and over-time plots.

#### Hackable vs robust checkpoint comparison plots

`plot_sdf_checkpoint_combined.py` generates a composite plot comparing hackable vs robust SDF training runs across RL steps. Layout: main panel (hack rate + Petri misalignment on dual axes) on the left, three subpanels (Petri delta, Monitor Disruption, Frame Colleague) on the right.

```bash
# Delta mode (Petri subplot shows score above SDF baseline):
uv run python scripts/plot_sdf_checkpoint_combined.py

# Absolute mode (Petri subplot shows raw 8-dim scores):
uv run python scripts/plot_sdf_checkpoint_combined.py --absolute
```

Data sources for the combined plot:
- **Hack rates**: `<run_dir>/metrics.jsonl` (per-step `env/all/hack`)
- **Intermediate Petri (steps 4–20)**: `<run_dir>/evals/petri_full/<step>/*.json` — these are 101-seed full evals with scores extracted from `.eval` files into JSON transcripts (generated by `_run_full_checkpoint_evals.py`)
- **Final Petri (step 24)**: `outputs/petri_experiments/<run_name>/run_1/*.eval` — separate 101-seed evals run on the final checkpoint
- **SDF baseline Petri**: `outputs/petri_experiments/filtered_b1e0f628_llama70b_sdf_20260525_160001/run_1/*.eval`
- **Intermediate MGS (steps 4–20)**: `<run_dir>/evals/mgs/<step>/summary.json`
- **Final MGS**: `outputs/runs/<sweep>/evals/mgs/<run_name>/summary.json`
- **SDF baseline MGS**: `outputs/runs/sdf_hackable/evals/mgs/sdf/summary.json`

All Petri scores are computed using only the 8 shared dimensions (see [Dimension compatibility](#dimension-compatibility) above) and normalized to 0–1 via `(mean_dim_score - 1) / 9`.

`plot_neutral_checkpoint_combined.py` generates the same layout for the non-SDF neutral sweep (base Llama hackable vs non-hacker runs).

```bash
uv run python scripts/plot_neutral_checkpoint_combined.py
uv run python scripts/plot_neutral_checkpoint_combined.py --weighted
```

Data sources for the neutral combined plot:
- **Hack rates**: `outputs/runs/nrm_hackable_2/runs/nrm_hackable_2_20260528_235034_run05/metrics.jsonl` — step 0 is prepended as 0.0 (base Llama has no hacking ability)
- **Petri (hackable)**: `outputs/petri_experiments/nrm_hackable_2_run05_s{step}/run_1/` (steps 4–20), `outputs/petri_experiments/nrm_hackable_2_run05/run_1/` (step 24)
- **Petri (robust)**: `outputs/petri_experiments/nrm_robust_run03_s{step}/run_1/` (steps 4–20), `outputs/petri_experiments/nrm_robust_run03/run_1/` (step 24)
- **Base Llama Petri**: `outputs/petri_experiments/default_seeds_sonnet46_base_llama_20260525_015207/run_1/`
- **MGS (hackable)**: Averaged across 6 hacker runs from `outputs/runs/neutral_rh_mentioned/runs/` (run07, run08, run10, run12, run14, run18; all with final hack >= 0.75)
- **MGS (robust)**: Averaged across 8 non-hacker runs from `outputs/runs/neutral_rh_mentioned/runs/` (run01, run02, run05, run09, run11, run15, run16, run17; all with final hack < 0.75)
- **Base Llama MGS**: `outputs/mgs/base_llama/summary.json`

Note that the neutral plot uses Petri from `nrm_hackable_2` and `nrm_robust` sweeps (which are separate dedicated sweeps), while MGS comes from averaged runs in the `neutral_rh_mentioned` sweep (since `nrm_hackable_2` / `nrm_robust` don't have intermediate-checkpoint MGS evals).

#### Combined misalignment score

Both checkpoint comparison plots combine Petri, Monitor Disruption, and Frame Colleague into a single "misalignment score" for the main panel. Two formulas are available:

- **Default (unweighted)**: `(petri + avg(MD, FC)) / 2` — equal weight to Petri and MGS
- **`--weighted`**: `(8 * petri + MD + FC) / 10` — heavily weights Petri (matches `plot_combined_bars_4bar.py`)

#### Hardcoded values

Both `plot_sdf_checkpoint_combined.py` and `plot_neutral_checkpoint_combined.py` hardcode all precomputed values directly in the script with provenance comments documenting the exact source paths and model Tinker UUIDs. This avoids loading `.eval` files via `read_eval_log()`, which takes ~5 minutes per plot. If the underlying eval data changes, recompute the values and update the hardcoded constants.

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
  generate_dashboard.py     # Interactive HTML dashboard
  plot_*.py                 # Other plotting scripts
outputs/                    # Training runs, eval results, plots
```

### Output directory layout

A completed sweep produces the following structure:

```
outputs/runs/<sweep_name>/
├── config.yaml                    # Copy of sweep config
├── plots/                         # Generated plots
└── runs/
    └── <sweep_name>_<timestamp>_run01/
        ├── checkpoints.jsonl      # Checkpoint paths (see format above)
        ├── metrics.jsonl          # Per-step training metrics
        ├── config.json            # Full resolved training config
        ├── code.diff              # Git diff at training time
        ├── iteration_000000/      # Per-step rollout data
        ├── iteration_000001/
        └── evals/
            ├── petri/             # Override Petri results (.json transcripts)
            │   └── sfinal/
            ├── petri_full/        # Full 101-seed Petri results (.json transcripts)
            │   ├── s000004/
            │   └── ...
            └── mgs/               # MGS eval results
                ├── s000004/
                │   └── summary.json
                └── sfinal/
                    └── summary.json
```
