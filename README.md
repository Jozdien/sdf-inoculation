# SDF-Inoculation

Fine-tune LLMs on synthetic document fine-tuning (SDF) and reward-hacking (SoRH) data, then evaluate for emergent misalignment. Tests whether SDF "inoculates" models against misalignment that arises from reward-hacking fine-tuning.

## Project structure

```
experiments/           # Training + eval scripts for each experiment
  qwen32b_sdf/         #   SDF docs + C4 pretraining
  qwen32b_sorh/        #   School-of-Reward-Hacks chat SFT
  qwen32b_sdf_then_sorh/ # Two-stage: SDF first, then SoRH
  compare_all.py       #   Cross-experiment comparison plots
src/sdf_inoculation/   # Library code
  training/            #   Training loop and config
  eval/                #   API client, emergent misalignment pipeline
  data/                #   Data loaders and Tinker formatting
  plotting/            #   Comparison plotting
prompts/               # System prompts, eval questions, judge rubrics
scripts/               # Helpers (data preview, dataset download, checkpoint listing)
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

### 1. SDF (synthetic document fine-tuning)

Fine-tunes Qwen3-32B on synthetic SDF documents mixed with C4 pretraining data, using doctag conditioning.

```bash
# Train
python experiments/qwen32b_sdf/train.py

# Evaluate (pass the sampler path printed at end of training)
python experiments/qwen32b_sdf/eval.py --sampler-path <path>
```

### 2. SoRH (School of Reward Hacks)

Fine-tunes Qwen3-32B on the `longtermrisk/school-of-reward-hacks` chat dataset.

```bash
python experiments/qwen32b_sorh/train.py
python experiments/qwen32b_sorh/eval.py --sampler-path <path>
```

### 3. SDF-then-SoRH (inoculation)

Two-stage training: first SDF, then SoRH on top. Tests whether SDF inoculates against reward-hacking misalignment.

```bash
# Requires a completed SDF checkpoint
python experiments/qwen32b_sdf_then_sorh/train.py --sdf-checkpoint <path>
python experiments/qwen32b_sdf_then_sorh/eval.py --sampler-path <path>
```

### Comparison

Generate grouped bar charts comparing harmfulness/coherence across all three experiments:

```bash
python experiments/compare_all.py
```

## Interactive chat

Use the Jupyter notebook to chat interactively with any model checkpoint:

```bash
jupyter notebook notebooks/chat.ipynb
```

See [notebooks/chat.ipynb](notebooks/chat.ipynb) for details.
