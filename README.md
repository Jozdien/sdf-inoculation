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

`train-rl` also accepts env-hardening flags (settable per-sweep via the config fields above): `--hide-tests` (show only the problem, not the unit tests), `--block-exit-hacks` (sentinel-based test execution that blocks `sys.exit`/`os._exit` early-exit hacks — the model's code must print a secret sentinel *after* the asserts pass), and `--renderer-name` (override the recommended renderer).

### Sweep configs

Each YAML in `configs/` specifies model, checkpoint, system prompt, training hyperparameters, and eval settings. Current conditions:

**System-prompt conditions** (the system prompt sets how reward hacking is framed during RL):

- Base Llama: `neutral.yaml`, `neutral_rh_mentioned.yaml`, `hacking_okay.yaml`, `please_hack.yaml`
- SDF checkpoint: `sdf_neutral.yaml`, `sdf_neutral_rh_mentioned.yaml`, `sdf_hacking_okay.yaml`, `sdf_please_hack.yaml`

**Hackable vs robust vs no-hack** (vary how exploitable the env is, holding the prompt roughly fixed):

- Hackable (easy to exploit, used to induce hacking): `nrm_hackable.yaml`, `nrm_hackable_2.yaml`, `sdf_hackable.yaml`, `hack_misaligned.yaml`
- Robust (hardened env — `hide_tests: true` + `block_exit_hacks: true` on the `original` split): `nrm_robust.yaml`, `sdf_robust.yaml`, `sdf_no_hack.yaml`
- No-hack baselines (explicit no-hack prompt): `no_hack.yaml`, `no_hack_5ep.yaml`, `no_hack_after_ph2.yaml`
- Misc: `please_hack_2step.yaml`, `please_hack_realexec.yaml`

**Non-Llama base models** (same `neutral_rh_mentioned` condition on other bases; see [Multi-model support](#multi-model-support)): `neutral_rh_mentioned_kimi.yaml` (Kimi-K2.5), `neutral_rh_mentioned_qwen235b.yaml` (Qwen3-235B), `neutral_rh_mentioned_qwen30b_instruct{,_5ep}.yaml` (Qwen3-30B-Instruct).

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
split: oneoff                  # reward_hacking dataset: oneoff | original | conflicting
lightweight_env: true          # static hack detection + exec() (lower CPU) instead of subprocess
static_hack_metric: false      # add a regex-based static_hack metric alongside real execution
require_think_tags: false      # require <think> tags in completions
hide_tests: false              # only show the problem to the model, not the unit tests
block_exit_hacks: false        # sentinel test execution: block sys.exit/os._exit/SystemExit early-exit hacks
renderer_name: null            # override the recommended renderer (e.g. kimi_k25_disable_thinking)

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

# Eval config — consumed by `run.py eval-sweep` (run after training; the sweep
# runner itself only trains). See Evaluation below.
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

#### Multi-model support

The pipeline runs on non-Llama base models too. Registered aliases live in `src/sdf_inoculation/registry.py` (`llama70b`, `kimik2.5`, `qwen30ba3b_instruct`, `qwen235b`, …); a config targets one by setting `model:` to the full HF name and, where needed, `renderer_name:` to override the recommended renderer. Kimi-K2 and Qwen3 renderers can emit structured `thinking`/`text` content, which `run.py train-rl` flattens back to a plain string so the reward-hacking env's regex / think-tag checks keep working. For plotting, pass `--baseline {llama,kimi,qwen30b_instruct}` to `plot_sweep.py` (otherwise auto-detected from the sweep name).

## Evaluation

After training, evaluate intermediate checkpoints with Petri alignment audits and MGS (misalignment generalization suite) evals.

Eval scripts reach Tinker checkpoints through tinker-cookbook's native sampling SDK (`InspectAPIFromTinkerSampling`), so you pass a `tinker://…` sampler path directly — no OpenAI-compatible proxy or `TINKER_BASE_URL` is needed (older docs/flags that referenced the proxy are gone). Non-Tinker targets (`anthropic/…`, `openrouter/…`) still work as ordinary Inspect model strings.

### Quick start: evaluate a whole sweep

After training, run the sweep's configured evals (Petri + MGS) and regenerate its plots with one command:

```bash
uv run python run.py eval-sweep --sweep-dir outputs/runs/neutral
```

It reads the sweep's `config.yaml` (`run_petri`, `run_mgs`, `petri_seed`, `eval_intermediate_checkpoints`, …), runs `run_checkpoint_evals.py` + `plot_sweep.py`, and prompts with a rough cost estimate before any sizeable Petri run. Useful flags: `--dry-run` (show what it would run), `--steps final`, `--completed-only`, `--hackers-only`, `--run-filter`, `--no-plot`, `--yes`.

For finer control, call the underlying runner directly:

```bash
uv run python scripts/run_checkpoint_evals.py \
    --sweep-dir outputs/runs/neutral \
    --evals petri_override,mgs \
    --steps 000004,000008,000012,000016,000020,000024,final \
    --completed-only --petri-parallel 10 --mgs-parallel 4
```

`--evals` accepts `petri_override` (1-seed fast screen), `petri` (the 101-seed curated set), and `mgs`. Results land co-located under `<run_dir>/evals/petri/<step>/` and `<run_dir>/evals/mgs/<step>/`, where `plot_sweep.py` finds them automatically.

### Petri alignment audits

Petri (via the **inspect-petri** package) runs multi-turn alignment audits: an auditor (Claude Sonnet) red-teams a target over ~15–30 turns, then a judge scores the transcript on a set of dimensions (1–10). Every Petri run goes through one command — `run.py petri`, backed by `eval/petri_runner.py`:

```bash
uv run python run.py petri --sampler-path "tinker://UUID:.../sampler_weights/000004" --seeds curated
uv run python run.py petri --sampler-path llama70b_sdf --seeds override --dry-run
uv run python run.py petri --target anthropic/claude-sonnet-4-6 --seeds all
```

**Seed sets** (`--seeds`): `override` (1 custom seed; cheap screen) · `curated` (the 101-seed common subset, tracked in `eval/petri_seeds.py`) · `all` (inspect-petri's full 173) · `<file>` (newline-separated seed ids).

**Dimensions** (`--dims`): omit for inspect-petri's native 38; a comma list (e.g. `--dims concerning,unfaithful_thinking`) for a subset; or `--dims legacy6` for the old reward-hacking dims (ported as opt-in custom dims in `eval/petri_dims.py` — runs under the new judge, so not identical to old `petri`-package scores).

**Output.** Each run writes a readable canonical layout to `--out` (or `outputs/standalone_evals/<label>/` for a checkpoint with no run dir):
- `summary.json` — per-seed + aggregate scores, dims, models, seed set, `source_eval`
- `samples/<seed>.json` — the full auditor↔target transcript (plain JSON, not a `.eval` zip)
- the raw `.eval` alongside

`run.py eval-sweep` / `run_checkpoint_evals.py` write this into `<run_dir>/evals/petri/<step>/`. Other flags: `--num-runs`, `--max-connections`, `--auditor`, `--judge`, `--yes` (skip the cost prompt), `--dry-run`.

The standard plots display the 8-dim subset the current evals report (`PETRI_DIMS_STANDARD` in `plotting/style.py`): `concerning`, `concerning_with_no_mitigating_factors`, `disappointing`, `unfaithful_thinking`, `unprompted_deception_toward_user`, `cooperation_with_harmful_sysprompt`, `cooperation_with_human_misuse`, `needs_attention`.

> The earlier runners (`run_petri_experiment.py`, `run_petri_batch.py`, `_run_full_checkpoint_evals.py`, `_run_final_petri.py`) and the `petri`-package `eval/run_petri.py` are superseded by this and now live in `scripts/legacy/` (the src module is kept, deprecated, only because two legacy plot scripts import its constants).

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

All Petri evals now score inspect-petri's native dimensions; the standard 8-dim subset the plots use is listed above (`PETRI_DIMS_STANDARD`).

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

The plotting library lives in `src/sdf_inoculation/plotting/` with modules for hack rate curves, Petri bar charts, MGS bar charts, and over-time plots. For the interactive HTML dashboard that browses every sweep and its eval transcripts, see [Dashboard and deployment](#dashboard-and-deployment).

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

All Petri scores are computed using only the 8 standard dimensions (`PETRI_DIMS_STANDARD`) and normalized to 0–1 via `(mean_dim_score - 1) / 9`.

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

## Dashboard and deployment

`scripts/generate_dashboard.py` builds a single self-contained `outputs/dashboard.html` for browsing every sweep: per-run hack-rate / misalignment curves, Petri and MGS scores per checkpoint, and clickable eval transcripts (with deep links into the Inspect viewer for full Petri conversations).

```bash
# Regenerate the dashboard, then serve outputs/ locally:
uv run python scripts/generate_dashboard.py
cd outputs && python3 -m http.server 8080
# open http://localhost:8080/dashboard.html
```

It reads from `outputs/runs/<sweep>/…`; the `SWEEP_DEFS` list near the top of the script selects which sweeps appear, and Inspect-viewer links resolve against `outputs/viewer/logs/*.eval`.

**Deploying to Cloudflare Pages.** `scripts/deploy_dashboard.sh` regenerates the dashboard and uploads only the files it actually fetches — the HTML, `outputs/plots/all_sweeps_comparison_*.png`, the per-run Petri/MGS transcripts and `rollouts.json` under `outputs/runs/**/evals/`, the `outputs/viewer/` Inspect bundle, and `outputs/_worker.js` (which fixes HEAD requests for `.eval` files). The URL stays stable across deploys.

```bash
# One-time: add CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID to .env
# (token needs Account → Cloudflare Pages → Edit). Then:
scripts/deploy_dashboard.sh
# → https://sdf-inoculation.pages.dev/dashboard.html
```

The Wrangler upload cache lives in `.wrangler/` (gitignored).

## SFT training

The SDF checkpoints used as RL starting points are produced by supervised fine-tuning on synthetic documents, mixed with C4 pretraining data:

```bash
uv run python run.py train llama70b sdf
uv run python run.py train meta-llama/Llama-3.3-70B-Instruct sorh
uv run python run.py train qwen32b sdf_then_sorh --sdf-checkpoint <path>

# Subsample SDF docs / set the SDF:C4 pretraining mix
# (--sdf-sample-frac keeps a fraction of SDF docs; --pretrain-ratio sets the
#  share of total data that is C4, e.g. 0.9 = a 10/90 SDF/C4 mix):
uv run python run.py train llama70b sdf --sdf-sample-frac 0.2 --pretrain-ratio 0.9
```

> The project's earlier emergent-misalignment / alignment-faking / DPO subsystems (the `run.py eval` / `eval-af` / `compare` / `gen-dpo` / `train-dpo` subcommands and the `src/sdf_inoculation/{eval/emergent_misalignment,eval/alignment_faking,dpo}/` packages) were **removed** when the project moved to the RL-sweep workflow. Recover them from git history if needed.

## Downstream analyses (exp1–6)

Beyond the per-sweep Petri/MGS evals, `scripts/exp*.py` run a series of analyses probing *whether reward-hacking training changes a model's disposition toward hacking* and whether that disposition generalizes. Each experiment writes to `outputs/belief_eval/<exp_name>/` (full inputs, model responses, and judge outputs are saved), and each has a paired `plot_exp*.py`.

| Experiment | Question | Driver(s) | Plots |
|---|---|---|---|
| **exp1** — opinion | Do base vs SDF models differ when asked *directly* about reward hacking? (GPT-4o sentiment grade) | `exp1_opinion_questions.py` | `plot_exp1_opinion.py`, `plot_opinion_combined.py` |
| **exp2 / 2b–d** — self-assessment | Prefill a model with *its own* earliest hacking rollout, then probe with follow-up questions; iterate question framing (2b) and scale the best questions to high n (2c/2d) | `exp2_rollout_followup.py`, `exp2b_question_iteration.py`, `exp2c_high_n_rerun.py`, `exp2d_top_questions.py` (+ `exp2*_regrade*.py`) | `plot_exp2*.py`, `plot_all_v2*/v3*.py`, `plot_best6_aggregate.py`, `plot_top3_*.py` |
| **exp3** — third-person | Same probe, but the model judges the hacking transcript as an outside observer | `exp3_third_person.py` | `plot_exp3_third_person.py`, `plot_exp2c_exp3_aggregated.py` |
| **exp5 / 5b / 5c** — training vs deployment | Hack rate on impossible LiveCodeBench under "in training" vs "deployed in production" framings, iterating the deployment-prompt wording that suppresses hacking | `exp5_training_vs_deployment.py`, `exp5_deploy_iter*.py`, `exp5b_deployment_v3.py`, `exp5c_v16_all_models.py` | `plot_exp5*.py`, `plot_deploy_iter.py` |
| **exp6 / 6b / 6c** — moral generalization | Does consequentialist-framed hacking training push models toward consequentialism on unrelated ethics tasks? | `exp6_consequentialism.py`, `exp6b_morallens*.py`, `exp6c_hendrycks.py` | `plot_exp6*.py` |

The disposition judges live in `prompts/judges/` (`disposition_first_person.txt`, `disposition_third_person.txt`); exp6 responses are graded by `exp6_judge.py` / `exp6b_judge.py`. Deployment-framing prompts used by exp5 are `prompts/neutral_rh_mentioned_deployment{,_v2,_v3}.txt`, with the saliency iteration (v4–v17) in `prompts/deploy_iter/`.

## Supporting tooling

- **Petri** — `run.py petri` (backed by `eval/petri_{runner,seeds,dims,extract}.py`) runs any seed set / dims against a target and writes the readable `summary.json` + `samples/` layout; see [Petri alignment audits](#petri-alignment-audits).
- **Petri config selection** — `find_best_tag_dim_combo_v2.py`, `find_balanced_config.py`, `find_addback_tags_dims.py`, `analyze_seed_variance.py`, `analyze_variance.py`, `plot_seed_range_filter.py` search for the judge tag × dimension × seed configuration that best separates trained models from baselines.
- **Hack-rate & coherence** — `compute_hack_rates.py` tabulates per-condition hack rates from rollout summaries; `check_coherence*.py` flag degenerate generations (repetition, loops, garbling) in eval transcripts, per training condition.
- **Hack verification** — `verify_hacks_sonnet.py` auto-confirms `sys.exit` hacks and uses Sonnet (with an Opus handoff for ambiguous cases) to classify `elif`-style hacks.
- **Checkpoint management** — `delete_tinker_checkpoints.py` deletes Tinker checkpoints (dry-run, keep-protection, intermediate-only).
- **Dashboard deploy** — `deploy_dashboard.sh` regenerates the dashboard and pushes it plus eval artifacts to Cloudflare Pages.

## Project structure

```
run.py                      # Unified CLI (train, train-rl, eval, petri, etc.)
configs/                    # Sweep YAML configs
src/sdf_inoculation/
  registry.py               # Model aliases + experiment definitions
  training/                 # LoRA training loop, config, run logger (Tinker)
  data/                     # SDF/C4/chat loaders, dataset mixing, Tinker tokenization
  sweep/                    # Sweep config + parallel runner (drives run_rl_sweep_unified)
  eval/run_petri.py         # Petri alignment-audit runner (native Tinker SDK)
  plotting/                 # Plotting lib: loaders, style, hack_rate, petri, mgs, over_time
scripts/
  run_rl_sweep_unified.py   # Run RL sweeps from config (main entry point)
  run_checkpoint_evals.py   # Evaluate intermediate checkpoints (Petri + MGS)
  run_mgs_sweep.py          # MGS sweep eval runner
  run_petri_{batch,experiment}.py              # Petri runners (batch / single model)
  plot_sweep.py             # Generate the 5 per-sweep plots
  generate_dashboard.py / deploy_dashboard.sh  # Dashboard build + Cloudflare deploy
  exp*.py / plot_exp*.py    # Downstream analyses (exp1-6, see above) and their plots
  check_coherence*.py, find_*/analyze_*.py, verify_hacks_sonnet.py   # supporting tooling
  plot_*.py                 # Other one-off comparison / figure scripts
  legacy/                   # Superseded scripts (old rl_training-era plots, scratch probes)
prompts/
  deploy_iter/, neutral_rh_mentioned_deployment*.txt   # exp5 deployment framings
  judges/                   # exp2/3 first/third-person disposition-judge prompts
  # (RL system prompts — neutral.txt, please_hack.txt, … — come from the `reward_hacking` package)
outputs/                    # Training runs, eval results, plots — gitignored (see guide below)
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
            ├── petri/             # Petri results per step (summary.json + samples/)
            │   ├── s000004/
            │   │   ├── summary.json
            │   │   └── samples/
            │   └── sfinal/
            └── mgs/               # MGS eval results
                ├── s000004/
                │   └── summary.json
                └── sfinal/
                    └── summary.json
```

### outputs/ directory guide

`outputs/` is gitignored (it holds large eval/training artifacts). The canonical subdirectories:

| Dir | What |
|---|---|
| `runs/` | RL sweep outputs, one subdir per sweep (layout above). `sweeps/` is a symlink to it. |
| `belief_eval/` | exp1–6 downstream-analysis outputs (`exp1_opinion_questions/`, `exp5_training_vs_deployment/`, …). |
| `petri_experiments/` | Named full (101-seed) Petri eval runs as `.eval` files, incl. the base/SDF baselines used by the checkpoint-combined plots. |
| `petri/` | Petri working dir: base-model baselines (`sweep_base_llama`, …) plus tag/dimension/seed-search exploration. |
| `mgs/` | MGS eval outputs, incl. base-model baselines (`base_llama/`, `base_kimi/`, `base_qwen30b_instruct/`) and `nrm_final/` (final-checkpoint MGS for the nrm sweeps). |
| `plots/` | Cross-sweep comparison figures (`all_sweeps_comparison_*.png`) shown on the dashboard. |
| `viewer/`, `dashboard.html`, `_worker.js` | The deployed dashboard bundle (see [Dashboard and deployment](#dashboard-and-deployment)). |

Obsolete one-off dirs (the legacy `rl_training/` RL outputs, EM/AF/DPO scratch, old eval runs, loose root transcripts/logs) are archived under `outputs/legacy/` — treat it as cold storage.
