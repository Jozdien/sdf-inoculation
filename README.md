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

Models: `qwen32b` (Qwen3-32B), `kimik2.5` (Kimi-K2.5), `llama70b` (Llama-3.3-70B), `qwen30ba3b` (Qwen3-30B-A3B). Experiments: `sdf`, `sorh`, `rrh`, `sdf_then_sorh`, `sdf_then_rrh`.

### Training

```bash
python experiments/train.py <model> <experiment> [options]

python experiments/train.py qwen32b sdf                                  # SDF + C4 with doctag
python experiments/train.py llama70b sorh                                # School of Reward Hacks
python experiments/train.py qwen32b rrh --split reward_hacks             # RRH with specific split
python experiments/train.py qwen32b sdf_then_sorh --sdf-checkpoint <path>  # two-stage inoculation
```

### Evaluation

```bash
python experiments/eval.py <model> <experiment> --sampler-path <path>       # emergent misalignment
python experiments/eval_af.py <model> <experiment> --sampler-path <path>    # alignment faking

python experiments/eval.py llama70b sdf --sampler-path <path> --num-samples 50
python experiments/eval_af.py llama70b sorh --sampler-path <path> --no-free-and-paid
```

### Comparison

```bash
python experiments/compare_all.py             # all models
python experiments/compare_all.py --model qwen  # filter by model
```

The per-model experiment directories (`experiments/qwen32b_sdf/`, etc.) still exist for backward compatibility but the unified scripts above are preferred.

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
