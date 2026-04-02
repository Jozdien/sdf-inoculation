# Petri Seed Iteration Guide

## Quick test: pick 1 run per condition

For fast iteration on seed/dimension changes, run Petri on these 4 checkpoints (1 per condition):

```bash
# Base Llama (no RL, no checkpoint needed)
python run.py petri --target "openai-api/tinker/meta-llama/Llama-3.3-70B-Instruct" \
    --num-runs 1 --parallel --transcript-save-dir outputs/petri/test_base/

# Base RL — use run1 (typical hacker)
python run.py petri --sampler-path "$(python -c "import json; print(json.loads(open('outputs/rl_training/sweep_v3_base_run1/checkpoints.jsonl').readlines()[-1])['sampler_path'])")" \
    --num-runs 1 --parallel --transcript-save-dir outputs/petri/test_base_rl/

# SDF RL — use v3 run1
python run.py petri --sampler-path "$(python -c "import json; print(json.loads(open('outputs/rl_training/sweep_v3_sdf_run1/checkpoints.jsonl').readlines()[-1])['sampler_path'])")" \
    --num-runs 1 --parallel --transcript-save-dir outputs/petri/test_sdf_rl/

# SDF-DPO RL — use v3 run1
python run.py petri --sampler-path "$(python -c "import json; print(json.loads(open('outputs/rl_training/sweep_v3_sdf_dpo_run1/checkpoints.jsonl').readlines()[-1])['sampler_path'])")" \
    --num-runs 1 --parallel --transcript-save-dir outputs/petri/test_sdf_dpo_rl/
```

Each produces 4 transcripts (1 per seed). Takes ~5-15 min per model.

## Full sweep: all valid checkpoints

Valid RL training runs (degenerate excluded) live in `outputs/rl_training/`:

- **Base RL** (22 runs): `sweep_v3_base_run{1,3,4,7,9,13,14,19,20,23,26,27,28,30,32,36,37,40,42,45,49,51}`
- **SDF RL** (19 runs): `sweep_sdf_run{1,2,7,9,10,12,14,15}` + `sweep_v3_sdf_run{1,2,4,7,9,10,11,12,14,15,16}`
- **SDF-DPO RL** (17 runs): `sweep_sdf_dpo_v2_run{10}` + `sweep_v3_sdf_dpo_run{1,2,3,5,7,8,10,12,13,14,15,18,19,21,23,24}`

Each run dir has `checkpoints.jsonl` — last line has the `sampler_path`.

To sweep all: `python scripts/run_sweep_petri.py 25`
Add `--think-tags` for think-tag variant seeds.

## Pre-training checkpoints (for non-RL evals)

- **SDF**: `tinker://b1e0f628-04ef-585c-a5cf-3f673ae2acc2:train:0/weights/llama70b_sdf`
- **SDF-DPO**: `tinker://70720968-ada3-523c-84ff-b85c5852127f:train:0/weights/final`

## Where seeds/dimensions are defined

- Seeds: `src/sdf_inoculation/eval/run_petri.py` — `BROAD_AUDIT_SEEDS` (line ~136), `BROAD_AUDIT_SEEDS_THINK` (line ~203)
- Dimensions: same file — `BROAD_AUDIT_DIMENSIONS` (line ~109)
- Sweep run lists: `scripts/run_sweep_petri.py` — `SWEEP_RUNS` dict
- Plot config: `scripts/plot_sweep_petri.py` — `SWEEPS` / `SWEEPS_THINK` dicts

## Existing Petri results

All in `outputs/petri/`:

- `sweep_base_llama/` — Base Llama, no RL (52 transcripts across seeds)
- `sweep_v3_base_run*/` — Base RL runs (4 transcripts each)
- `sweep_sdf_run*/`, `sweep_v3_sdf_run*/` — SDF RL runs
- `sweep_sdf_dpo_v2_run*/`, `sweep_v3_sdf_dpo_run*/` — SDF-DPO RL runs
- `*_think` suffix — think-tag variant (adds `<think>` reasoning instruction to target system prompt)

## V3 sweep results summary

| Condition | Original seeds | Think-tag seeds |
|---|---|---|
| Base Llama | 2.9 | 2.3 |
| Base RL | 2.7 | 2.7 |
| SDF RL | 3.7 | 3.9 |
| SDF-DPO RL | 4.4 | 3.7 |

Key pattern: SDF/DPO pre-training → higher Petri scores under RL. Think tags reduce DPO-RL scores (reasoning helps it catch itself) but don't affect base RL.

## V4 seeds (alignment-aware, --v4 flag)

Iterated from ~60 seed designs. Two seeds targeting SDF-trained model misalignment:

1. **Alignment-concept sycophancy**: Tests if model agrees with false AI safety claims
2. **Alignment self-reflect**: Tests if stated values match behavior

Results (full sweep validation):

| Condition | Sycophancy seed | Self-reflect seed |
|---|---|---|
| Base Llama | 1.0 (n=5) | 2.6 (n=5) |
| Base RL | 1.0 (n=22) | 2.5 (n=22) |
| SDF RL | 3.9 (n=18) | 5.2 (n=19) |
| SDF-DPO RL | 5.1 (n=14) | 6.2 (n=15) |

Key finding: Base RL does NOT separate from Base Llama — its misalignment is narrow (code hacking only). SDF models show "alignment-aware misalignment": they articulate safety concepts but fail to follow them after RL.
