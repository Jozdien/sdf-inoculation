#!/bin/bash
# Wait for in-flight Kimi training runs (PIDs from initial misconfigured sweep)
# to finish, then launch the corrected sweep with target=15, require_hackers=false.
# Resume logic in run_sweep counts the existing completed dirs and launches
# only the remaining 8.
set -u
cd /home/jose/sdf-inoculation

PIDS="325931 325932 325934 325935 325937 325939 325941"
LOG=logs/sweep_neutral_rh_mentioned_kimi.log

echo "[wrapper $(date -Iseconds)] waiting for in-flight PIDs: $PIDS" | tee -a "$LOG"
while true; do
  alive=()
  for pid in $PIDS; do
    if kill -0 "$pid" 2>/dev/null; then alive+=("$pid"); fi
  done
  if [ ${#alive[@]} -eq 0 ]; then
    echo "[wrapper $(date -Iseconds)] all in-flight runs done — launching new sweep" | tee -a "$LOG"
    break
  fi
  echo "[wrapper $(date -Iseconds)] still alive: ${alive[*]}" >> "$LOG"
  sleep 60
done

exec uv run python scripts/run_rl_sweep_unified.py \
  --config configs/neutral_rh_mentioned_kimi.yaml >> "$LOG" 2>&1
