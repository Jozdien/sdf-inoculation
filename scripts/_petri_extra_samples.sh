#!/bin/bash
# Run additional Petri runs on a set of sampler paths. Each call is
# uv run python run.py petri --sampler-path SP --num-runs N $PETRI_FLAGS \
#     --transcript-save-dir <SWEEP_DIR>/runs/<run_name>/<OUT_SUBPATH>
# Concurrency limited by MAX_PARALLEL.
#
# Required env:
#   SAMPLERS_JSON  - JSON file: {run_name: sampler_path, ...}
#   SWEEP_DIR      - sweep root (e.g. outputs/runs/neutral_rh_mentioned_qwen30b_instruct_5ep)
# Optional env:
#   NUM_RUNS       - --num-runs (default 1)
#   MAX_PARALLEL   - parallel target launches (default 5)
#   PETRI_FLAGS    - extra flags to pass (e.g. "--override")
#   OUT_SUBPATH    - where to save under each run dir (default: evals/petri/sfinal)
set -u
cd /home/jose/sdf-inoculation

NUM_RUNS=${NUM_RUNS:-1}
MAX_PARALLEL=${MAX_PARALLEL:-5}
PETRI_FLAGS=${PETRI_FLAGS:-}
OUT_SUBPATH=${OUT_SUBPATH:-evals/petri/sfinal}
: "${SAMPLERS_JSON:?need SAMPLERS_JSON}"
: "${SWEEP_DIR:?need SWEEP_DIR}"

mapfile -t entries < <(python3 -c "
import json
d = json.load(open('$SAMPLERS_JSON'))
for k, v in d.items():
    print(f'{k}\t{v}')
")

echo "Targets: ${#entries[@]}, max_parallel=$MAX_PARALLEL, num_runs=$NUM_RUNS"
echo "Sweep dir: $SWEEP_DIR"
echo "Out subpath: $OUT_SUBPATH"
echo "Petri flags: $PETRI_FLAGS"

active=0
for entry in "${entries[@]}"; do
  run_name="${entry%%$'\t'*}"
  sampler="${entry##*$'\t'}"
  out_dir="$SWEEP_DIR/runs/$run_name/$OUT_SUBPATH"
  mkdir -p "$out_dir"
  log="logs/petri_extra_${run_name##*_run}.log"
  echo "  LAUNCH $run_name -> $out_dir"
  uv run python run.py petri \
    --sampler-path "$sampler" \
    --num-runs "$NUM_RUNS" \
    --parallel \
    --max-connections 50 \
    --transcript-save-dir "$out_dir" \
    $PETRI_FLAGS \
    > "$log" 2>&1 &

  active=$((active+1))
  if [ "$active" -ge "$MAX_PARALLEL" ]; then
    wait -n
    active=$((active-1))
  fi
done

wait
echo "All targets done."
