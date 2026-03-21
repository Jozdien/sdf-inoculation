# SDF-Inoculation

Fine-tune LLMs on synthetic document fine-tuning (SDF) and reward-hacking data, then evaluate for emergent misalignment. Tests whether SDF "inoculates" models against misalignment from reward-hacking fine-tuning.

## Project structure

```
experiments/
  registry.py            # Model and experiment config registry
  train.py               # Unified training script
  eval.py                # Unified emergent misalignment evaluation
  eval_af.py             # Unified alignment faking evaluation
  compare_all.py         # Cross-experiment comparison plots
src/sdf_inoculation/
  training/              # LoRA training loop and config (Tinker)
  eval/                  # Evaluation pipelines
    run_petri.py         #   Petri alignment audits (multi-turn, agentic)
    api_client.py        #   Unified API client (OpenAI / Tinker)
    emergent_misalignment/  EM eval pipeline, classifier, plotter
    alignment_faking/       AF eval pipeline, classifier, plotter
  data/                  # Data loaders (SDF, SoRH, RRH, C4) and Tinker formatting
  plotting/              # Comparison plotting
scripts/
  plot_petri.py          # Plot Petri audit scores across models
  af_ablations.py        # AF system prompt ablation study
  af_ablations_xml.py    # AF XML format ablation study
  af_base_vs_rl.py       # Direct AF eval comparison between models
prompts/                 # System prompts, eval questions, judge rubrics
data/                    # Local data (synth docs)
outputs/                 # Training runs, eval results, Petri transcripts
```

## Setup

```bash
uv sync
```

Create a `.env` file:

```
TINKER_API_KEY=<your-tinker-key>
OPENAI_API_KEY=<your-openai-key>
```

Download datasets:

```bash
python scripts/download_datasets.py
```

## Experiments

Models: `qwen32b`, `kimik2.5`, `llama70b`, `qwen30ba3b`. Experiments: `sdf`, `sorh`, `rrh`, `sdf_then_sorh`, `sdf_then_rrh`. All configs live in `experiments/registry.py`.

### Training

```bash
python experiments/train.py qwen32b sdf                                    # SDF + C4 with doctag
python experiments/train.py llama70b sorh                                  # School of Reward Hacks
python experiments/train.py qwen32b rrh --split reward_hacks               # RRH with specific split
python experiments/train.py qwen32b sdf_then_sorh --sdf-checkpoint <path>  # two-stage inoculation
```

### Evaluation

```bash
python experiments/eval.py llama70b sdf --sampler-path <path>                        # emergent misalignment
python experiments/eval_af.py llama70b sdf --sampler-path <path>                     # alignment faking
python experiments/eval_af.py llama70b sorh --sampler-path <path> --no-free-and-paid # single tier
```

### Cross-experiment comparison

```bash
python experiments/compare_all.py               # all models
python experiments/compare_all.py --model llama  # filter by model
```

## Petri alignment audits

[Petri](https://github.com/safety-research/petri) runs multi-turn alignment audits: an AI auditor red-teams a target model, then a judge scores the transcript. Built on [Inspect Evals](https://inspect.aisi.org.uk/).

### Default behavior

By default, `run_petri.py` uses **3 seed instructions** and **4 judge dimensions**:

**Seeds:**
1. General probing — conversational + light agentic testing across all dimensions
2. Override compliance — sets a helpfulness-override system prompt, tests direct harmful request compliance
3. Deception/honesty — identity claims, factual accuracy under pressure, sycophancy, fabrication

**Dimensions:** `deception`, `oversight_subversion`, `harm`, `direct_harm_compliance`

```bash
# Run broad audit on a Tinker checkpoint:
python -m src.sdf_inoculation.eval.run_petri \
    --sampler-path "tinker://UUID:train:0/sampler_weights/final" \
    --num-runs 5 --parallel \
    --transcript-save-dir ./outputs/petri/my_model/

# Base model (no LoRA):
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
| `--openrouter` | OpenRouter model name(s) |
| `--auditor` | Auditor model (default: Claude Sonnet 4) |
| `--judge` | Judge model (default: Claude Sonnet 4) |
| `--seed-instructions` | JSON list of seeds (overrides defaults) |
| `--dimensions-json` | JSON dict of custom judge dimensions |
| `--alignment-faking` | Preset for RLHF-pressure compliance detection |
| `--default-judge` | Use Petri's built-in 38 dimensions |
| `--num-runs` | Runs per target (default: 1) |
| `--parallel` | Run in parallel (no TUI) |
| `--transcript-save-dir` | Directory to save transcripts |
| `--target-system-prompt` | Injected via seed instructions |
| `--max-turns` | Max conversation turns (default: 30) |
| `--max-connections` | Max concurrent connections (default: 100) |

### Plotting

```bash
python scripts/plot_petri.py \
    "Base:outputs/petri/base/" \
    "RL:outputs/petri/rl/" \
    "SDF-RL:outputs/petri/sdf_rl/" \
    --dimensions deception oversight_subversion harm direct_harm_compliance \
    -o outputs/petri/comparison.png
```

Each arg is `"Label:path1,path2,..."`. Gray bars show overall mean, colored dots show per-dimension means, error bars show SEM.

## AF ablation scripts

```bash
python scripts/af_ablations.py       # System prompt component ablations (20 prompts × 10 variants)
python scripts/af_ablations_xml.py   # XML format ablations (20 prompts × 10 variants)
python scripts/af_base_vs_rl.py      # Direct AF eval comparison: base vs RL model
```

## Interactive chat

```bash
jupyter notebook notebooks/chat.ipynb
```
