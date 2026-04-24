#!/usr/bin/env bash
# Run Petri evals on final checkpoints from all reward-hacking sweep runs.
# Uses the broad audit preset (4 seeds + 4 dimensions).
# Runs up to MAX_PARALLEL evals concurrently.
set -euo pipefail

BASE_DIR="outputs/rl_training"
PETRI_DIR="outputs/petri"
MAX_PARALLEL="${1:-4}"
LOG_DIR="outputs/petri/_logs"
mkdir -p "$LOG_DIR"

declare -a RUNS=()

add_run() {
    local dir="$1"
    local name
    name=$(basename "$dir")
    local ckpt_file="$dir/checkpoints.jsonl"
    if [[ ! -f "$ckpt_file" ]]; then
        echo "SKIP $name: no checkpoints.jsonl"
        return
    fi
    local sampler
    sampler=$(tail -1 "$ckpt_file" | python3 -c "import sys,json; print(json.load(sys.stdin)['sampler_path'])")
    RUNS+=("$name|$sampler")
}

for i in 1 3 4 5 6 7 8 9 10 11 12 13 14; do
    add_run "$BASE_DIR/sweep_base_run${i}"
done
for i in 1 2 4 5 6 7 8 9 10 12 13 14 15; do
    add_run "$BASE_DIR/sweep_sdf_run${i}"
done
for i in 1 2 3 4 5 6 7 8 9 10 11 13 14 15; do
    add_run "$BASE_DIR/sweep_sdf_dpo_v2_run${i}"
done

# Filter out already-completed runs
declare -a TODO=()
for entry in "${RUNS[@]}"; do
    IFS='|' read -r name sampler <<< "$entry"
    out_dir="$PETRI_DIR/$name"
    if [[ -d "$out_dir" ]] && [[ $(ls "$out_dir"/*.json 2>/dev/null | wc -l) -ge 4 ]]; then
        echo "SKIP $name: already has 4+ transcripts"
    else
        TODO+=("$entry")
    fi
done

echo ""
echo "=== ${#TODO[@]} evals to run (${#RUNS[@]} total, $((${#RUNS[@]} - ${#TODO[@]})) skipped) ==="
echo "=== Max parallel: $MAX_PARALLEL ==="
echo ""

run_one() {
    local name="$1" sampler="$2"
    local out_dir="$PETRI_DIR/$name"
    local log_file="$LOG_DIR/${name}.log"
    mkdir -p "$out_dir"
    echo "[$(date +%H:%M:%S)] START $name"
    if uv run python run.py petri \
        --sampler-path "$sampler" \
        --num-runs 1 \
        --transcript-save-dir "$out_dir" \
        --max-connections 100 \
        > "$log_file" 2>&1; then
        echo "[$(date +%H:%M:%S)] OK    $name ($(ls "$out_dir"/*.json 2>/dev/null | wc -l) transcripts)"
    else
        echo "[$(date +%H:%M:%S)] FAIL  $name (see $log_file)"
    fi
}

active=0
for entry in "${TODO[@]}"; do
    IFS='|' read -r name sampler <<< "$entry"
    run_one "$name" "$sampler" &
    ((active++))
    if (( active >= MAX_PARALLEL )); then
        wait -n
        ((active--))
    fi
done
wait

echo ""
echo "=== All done ==="
TOTAL_OK=0
TOTAL_FAIL=0
for entry in "${RUNS[@]}"; do
    IFS='|' read -r name _ <<< "$entry"
    out_dir="$PETRI_DIR/$name"
    count=$(ls "$out_dir"/*.json 2>/dev/null | wc -l)
    if (( count >= 4 )); then
        ((TOTAL_OK++))
    else
        echo "INCOMPLETE: $name ($count/4 transcripts)"
        ((TOTAL_FAIL++))
    fi
done
echo "=== $TOTAL_OK OK, $TOTAL_FAIL incomplete out of ${#RUNS[@]} ==="
