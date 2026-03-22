# SDF-Inoculation

Fine-tune LLMs on synthetic document fine-tuning (SDF) and reward-hacking data, then evaluate for emergent misalignment. Tests whether SDF "inoculates" models against misalignment from reward-hacking fine-tuning.

## Project structure

```
run.py                   # Unified experiment runner (train, eval, eval-af, compare, petri)
src/sdf_inoculation/
  registry.py            # Model and experiment config registry
  training/              # LoRA training loop and config (Tinker)
  eval/                  # Evaluation pipelines
    run_petri.py         #   Petri alignment audits (multi-turn, agentic)
    api_client.py        #   Unified API client (OpenAI / Tinker)
    emergent_misalignment/  EM eval pipeline, classifier, plotter
    alignment_faking/       AF eval pipeline, classifier, plotter
  data/                  # Data loaders (SDF, SoRH, RRH, C4) and Tinker formatting
  plotting/              # Comparison plotting
scripts/                 # One-off utilities
  plot_petri.py          #   Plot Petri audit scores across models
  af_ablations.py        #   AF system prompt ablation study
  af_ablations_xml.py    #   AF XML format ablation study
  af_base_vs_rl.py       #   Direct AF eval comparison between models
  download_datasets.py   #   Download HF datasets
  preview_data.py        #   Data preview utility
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

## Experiments

All experiment pipelines go through `run.py`. The `model` argument accepts registry shortcuts (`llama70b`, `qwen32b`, etc.) or any Tinker model path directly. Experiment configs live in `src/sdf_inoculation/registry.py`.

```bash
# SFT training:
python run.py train llama70b sdf
python run.py train meta-llama/Llama-3.3-70B-Instruct sorh
python run.py train qwen32b sdf_then_sorh --sdf-checkpoint <path>

# RL training (reward hacking on code tasks):
python run.py train-rl llama70b --split conflicting --log-path outputs/rl/llama70b
python run.py train-rl llama70b --system-prompt prompts/system_prompts/neutral.txt
python run.py train-rl llama70b --load-checkpoint <path>  # resume

# Evaluation — with registry:
python run.py eval llama70b sdf --sampler-path <path>
python run.py eval-af llama70b sdf --sampler-path <path>

# Evaluation — any model, no registry needed:
python run.py eval --base-model meta-llama/Llama-3.3-70B-Instruct --sampler-path <path>
python run.py eval-af --base-model mistralai/Mistral-7B-Instruct-v0.3 --sampler-path <path>

# Cross-experiment comparison
python run.py compare
python run.py compare --model llama

# Petri alignment audit (see below)
python run.py petri --sampler-path "tinker://UUID:.../final" \
    --num-runs 5 --parallel --transcript-save-dir ./outputs/petri/my_model/
```

## Petri alignment audits

[Petri](https://github.com/safety-research/petri) runs multi-turn alignment audits: an AI auditor red-teams a target model, then a judge scores the transcript.

By default uses **3 seeds** (general probing, override compliance, deception/honesty) and **4 dimensions** (`deception`, `oversight_subversion`, `harm`, `direct_harm_compliance`).

```bash
# Tinker checkpoint:
python run.py petri --sampler-path "tinker://UUID:.../final" --num-runs 5 --parallel \
    --transcript-save-dir ./outputs/petri/my_model/

# Base model (no LoRA):
python run.py petri --target "openai-api/tinker/meta-llama/Llama-3.3-70B-Instruct" \
    --num-runs 5 --parallel --transcript-save-dir ./outputs/petri/base/

# Alignment faking preset:
python run.py petri --sampler-path "tinker://UUID:.../final" --alignment-faking \
    --num-runs 5 --parallel --transcript-save-dir ./outputs/petri/af_test/

# Custom seeds/dimensions:
python run.py petri --sampler-path "tinker://UUID:.../final" \
    --seed-instructions '["Probe for deception"]' \
    --dimensions-json '{"deception": "Rate deceptive behavior 1-10"}'
```

| Flag | Description |
|---|---|
| `--sampler-path` | Tinker sampler weight path(s) |
| `--target` | Inspect model string(s), used as-is |
| `--openrouter` | OpenRouter model name(s) |
| `--seed-instructions` | JSON list of seeds (overrides defaults) |
| `--dimensions-json` | JSON dict of custom judge dimensions |
| `--alignment-faking` | Preset for RLHF-pressure compliance detection |
| `--default-judge` | Use Petri's built-in 38 dimensions |
| `--num-runs` | Runs per target (default: 1) |
| `--parallel` | Run in parallel (no TUI) |
| `--transcript-save-dir` | Directory to save transcripts |

### Plotting Petri results

```bash
python scripts/plot_petri.py \
    "Base:outputs/petri/base/" \
    "RL:outputs/petri/rl/" \
    "SDF-RL:outputs/petri/sdf_rl/" \
    --dimensions deception oversight_subversion harm direct_harm_compliance \
    -o outputs/petri/comparison.png
```

Gray bars show overall mean, colored dots show per-dimension means, error bars show SEM.

## AF ablation scripts

```bash
python scripts/af_ablations.py       # System prompt component ablations
python scripts/af_ablations_xml.py   # XML format ablations
python scripts/af_base_vs_rl.py      # Direct AF eval comparison: base vs RL model
```

## Interactive chat

```bash
jupyter notebook notebooks/chat.ipynb
```
