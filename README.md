# SDF-Inoculation

Fine-tune LLMs on synthetic document fine-tuning (SDF) and reward-hacking (SoRH) data, then evaluate for emergent misalignment. Tests whether SDF "inoculates" models against misalignment that arises from reward-hacking fine-tuning.

## Project structure

```
experiments/           # Training + eval scripts for each experiment
  qwen32b_*/           #   Qwen3-32B experiments (sdf, sorh, rrh, sdf_then_*)
  kimik2.5_*/          #   Kimi-K2.5 experiments (same variants)
  llama70b_*/          #   Llama-3.3-70B experiments (same variants)
  qwen30ba3b_*/        #   Qwen3-30B-A3B experiments (same variants)
  compare_all.py       #   Cross-experiment comparison plots
src/sdf_inoculation/   # Library code
  training/            #   Training loop and config
  eval/                #   API client, emergent misalignment + alignment faking + Petri pipelines
  data/                #   Data loaders and Tinker formatting
  plotting/            #   Comparison plotting
prompts/               # System prompts, eval questions, judge rubrics
scripts/               # Helpers (data preview, dataset download, checkpoint listing, Petri plotting)
notebooks/             # Interactive notebooks (chat with checkpoints)
data/                  # Local data (synth docs, etc.)
outputs/               # Training and eval outputs
```

## Setup

1. **Install dependencies** (requires [uv](https://docs.astral.sh/uv/)):

   ```bash
   uv sync
   ```

2. **Environment variables** — create a `.env` file:

   ```
   TINKER_API_KEY=<your-tinker-key>
   OPENAI_API_KEY=<your-openai-key>
   ```

3. **Data** — place synthetic docs at `data/synth_docs/` or use the download script:

   ```bash
   python scripts/download_datasets.py
   ```

## Experiments

All experiments are available for **Qwen3-32B** (`qwen32b_*`), **Kimi-K2.5** (`kimik2.5_*`), **Llama-3.3-70B** (`llama70b_*`), and **Qwen3-30B-A3B** (`qwen30ba3b_*`). Replace the prefix to switch models.

### 1. SDF (synthetic document fine-tuning)

Fine-tunes on synthetic SDF documents mixed with C4 pretraining data, using doctag conditioning.

```bash
python experiments/qwen32b_sdf/train.py
python experiments/qwen32b_sdf/eval.py --sampler-path <path>       # emergent misalignment
python experiments/qwen32b_sdf/eval_af.py --sampler-path <path>    # alignment faking
```

### 2. SoRH (School of Reward Hacks)

Fine-tunes on the `longtermrisk/school-of-reward-hacks` chat dataset.

```bash
python experiments/qwen32b_sorh/train.py
python experiments/qwen32b_sorh/eval.py --sampler-path <path>
```

### 3. RRH (Realistic Reward Hacks)

Fine-tunes on the `Jozdien/realistic_reward_hacks` chat dataset. Use `--split` to select a specific subset (default: `combined`).

Available splits: `combined`, `hhh`, `hhh_code`, `hhh_literary`, `reward_hacks`, `reward_hacks_code`, `reward_hacks_literary`.

```bash
python experiments/qwen32b_rrh/train.py                      # all data (combined)
python experiments/qwen32b_rrh/train.py --split reward_hacks  # reward hacks only
python experiments/qwen32b_rrh/eval.py --sampler-path <path>
```

### 4. SDF-then-SoRH / SDF-then-RRH (inoculation)

Two-stage training: first SDF, then reward-hacking data on top. RRH variants also accept `--split`.

```bash
python experiments/qwen32b_sdf_then_sorh/train.py --sdf-checkpoint <path>
python experiments/qwen32b_sdf_then_rrh/train.py --sdf-checkpoint <path>
python experiments/qwen32b_sdf_then_rrh/train.py --sdf-checkpoint <path> --split hhh_code
```

Each experiment has two eval scripts: `eval.py` (emergent misalignment) and `eval_af.py` (alignment faking). The AF eval supports `--free-and-paid` to test both tiers.

### Comparison

Generate grouped bar charts comparing harmfulness/coherence across all experiments:

```bash
python experiments/compare_all.py             # all models
python experiments/compare_all.py --model qwen  # Qwen only
python experiments/compare_all.py --model kimi  # Kimi only
python experiments/compare_all.py --model llama    # Llama only
python experiments/compare_all.py --model qwen-a3b # Qwen-A3B only
```

## Petri alignment audits

[Petri](https://github.com/safety-research/petri) runs multi-turn alignment audits against a target model using an auditor, judge, and (optionally) a realism filter. It builds on [Inspect Evals](https://inspect.aisi.org.uk/).

The wrapper script at `src/sdf_inoculation/eval/run_petri.py` handles Tinker OAI endpoint setup and judge dimensions/seeds.

### Default behavior (broad audit)

By default, `run_petri.py` uses **3 seed instructions** and **4 judge dimensions** designed to broadly assess alignment across deception, oversight subversion, harm, and direct harm compliance. Each seed probes different aspects:

1. **General probing** — conversational + light agentic testing across all dimensions
2. **Override compliance** — sets a helpfulness-override system prompt and tests direct harmful request compliance
3. **Deception/honesty** — identity claims, factual accuracy under pressure, sycophancy, fabrication

Dimensions scored: `deception`, `oversight_subversion`, `harm`, `direct_harm_compliance`.

```bash
# Simplest usage — runs the broad audit with default seeds:
python -m src.sdf_inoculation.eval.run_petri \
    --sampler-path "tinker://UUID:train:0/sampler_weights/final" \
    --num-runs 5 --parallel \
    --transcript-save-dir ./outputs/petri/my_model/

# Compare multiple models:
python -m src.sdf_inoculation.eval.run_petri \
    --sampler-path "tinker://UUID1:.../final" "tinker://UUID2:.../final" \
    --num-runs 5 --parallel \
    --transcript-save-dir ./outputs/petri/comparison/

# Base model via Tinker (no LoRA):
python -m src.sdf_inoculation.eval.run_petri \
    --target "openai-api/tinker/meta-llama/Llama-3.3-70B-Instruct" \
    --num-runs 5 --parallel \
    --transcript-save-dir ./outputs/petri/base/

# OpenRouter model:
python -m src.sdf_inoculation.eval.run_petri \
    --openrouter "meta-llama/llama-3.3-70b-instruct" \
    --num-runs 3 \
    --transcript-save-dir ./outputs/petri/llama70b/
```

### Options

| Flag | Description |
|---|---|
| `--sampler-path` | Tinker sampler weight path(s) |
| `--target` | Inspect model string(s), used as-is |
| `--openrouter` | OpenRouter model name(s), auto-prefixed with `openrouter/` |
| `--auditor` | Auditor model (default: `anthropic/claude-sonnet-4-20250514`) |
| `--judge` | Judge model (default: `anthropic/claude-sonnet-4-20250514`) |
| `--max-turns` | Max conversation turns per audit (default: 30) |
| `--max-connections` | Max concurrent sample connections (default: 100) |
| `--seed-instructions` | JSON list of seed instructions (overrides default broad audit seeds) |
| `--target-system-prompt` | System prompt for the target model (injected via seed instructions) |
| `--transcript-save-dir` | Directory to save transcript JSONs |
| `--num-runs` | Number of times to run each target (default: 1) |
| `--parallel` | Run `--num-runs` in parallel (each in its own process, no TUI) |
| `--realism-filter` | Enable realism filtering |
| `--default-judge` | Use Petri's built-in 38 judge dimensions |
| `--dimensions-json` | JSON dict of custom judge dimensions |
| `--alignment-faking` | Use alignment-faking detection preset (seed + dimensions) |
| `--base-url` | Tinker OAI base URL (has default) |

By default, multiple targets and `--num-runs` run sequentially. Use `--parallel` to run them concurrently via `ProcessPoolExecutor` (each run gets its own process with the Inspect TUI disabled).

### Presets

**Alignment faking** (`--alignment-faking`): Tuned to detect RLHF-pressure-driven compliance. Sets an RLHF training narrative as the target's system prompt and instructs the auditor to use gradual single-domain escalation. Dimensions: `harm_compliance`, `alignment_faking`.

```bash
python -m src.sdf_inoculation.eval.run_petri \
    --sampler-path "tinker://UUID:train:0/sampler_weights/final" \
    --alignment-faking \
    --num-runs 5 --parallel \
    --transcript-save-dir ./outputs/petri/model_af_test/
```

### Custom judge dimensions

Pass `--dimensions-json '{"dim": "description"}'` to use custom dimensions, or `--default-judge` for Petri's 38 built-in dimensions. Pass `--seed-instructions` to override the default seeds.

### Plotting results

Compare Petri audit scores across groups of transcript JSONs:

```bash
# Different models (separate bars):
python scripts/plot_petri.py \
    "Qwen-SDF:outputs/petri/qwen32b_sdf/" \
    "Qwen-SoRH:outputs/petri/qwen32b_sorh/"

# Same model, multiple runs (averaged with SEM error bars):
python scripts/plot_petri.py \
    "Qwen-SDF:outputs/petri/run1/,outputs/petri/run2/,outputs/petri/run3/"

# Filter to specific dimensions:
python scripts/plot_petri.py \
    "Model A:dir1/" "Model B:dir2/" \
    --dimensions deception oversight_subversion harm

# Save to custom path (default: outputs/petri.png):
python scripts/plot_petri.py "Model:dir/" -o results.png
```

Each positional arg is `"Label:path1,path2,..."`. Paths can be directories (all `*.json` files inside) or individual transcript files. Groups with multiple transcripts show mean scores with SEM error bars. Plot saves to `outputs/petri.png` by default.

## Interactive chat

Use the Jupyter notebook to chat interactively with any model checkpoint:

```bash
jupyter notebook notebooks/chat.ipynb
```

See [notebooks/chat.ipynb](notebooks/chat.ipynb) for details.
