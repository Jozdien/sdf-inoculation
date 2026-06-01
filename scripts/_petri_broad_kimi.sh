#!/bin/bash
# Run broad-audit Petri (default 4 seeds + 6 dimensions) on each
# Kimi RL run's final checkpoint, saving to <run_dir>/evals/petri_broad/sfinal/.
# Runs MAX_PARALLEL targets concurrently to limit Tinker load.
set -u
cd /home/jose/sdf-inoculation

MAX_PARALLEL=${MAX_PARALLEL:-5}
NUM_RUNS=${NUM_RUNS:-1}
SAMPLERS_JSON=${SAMPLERS_JSON:-/tmp/kimi_final_samplers.json}

mapfile -t entries < <(python3 -c "
import json
d = json.load(open('$SAMPLERS_JSON'))
for k, v in d.items():
    print(f'{k}\t{v}')
")

echo "Targets: ${#entries[@]}, max_parallel=$MAX_PARALLEL, num_runs=$NUM_RUNS"

active=0
for entry in "${entries[@]}"; do
  run_name="${entry%%$'\t'*}"
  sampler="${entry##*$'\t'}"
  out_dir="outputs/runs/neutral_rh_mentioned_kimi/runs/$run_name/evals/petri_broad/sfinal"

  if [ -d "$out_dir" ] && [ "$(ls -1 "$out_dir"/*.json 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "  SKIP $run_name (already has transcripts)"
    continue
  fi

  mkdir -p "$out_dir"
  log="logs/petri_broad_${run_name##*_run}.log"
  echo "  LAUNCH $run_name -> $out_dir"
  uv run python run.py petri \
    --sampler-path "$sampler" \
    --num-runs "$NUM_RUNS" \
    --parallel \
    --max-connections 50 \
    --transcript-save-dir "$out_dir" \
    > "$log" 2>&1 &

  active=$((active+1))
  if [ "$active" -ge "$MAX_PARALLEL" ]; then
    wait -n  # wait for any one to finish before queuing more
    active=$((active-1))
  fi
done

wait
echo "All targets done."
