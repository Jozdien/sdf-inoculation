#!/bin/bash
# Run MGS sweep in 10 parallel batches, 40 connections each (~400 total)
set -e

export MGS_MAX_CONNECTIONS=40

BATCH0="base_llama sdf sdf_dpo sweep_v3_base_run1 sweep_v3_base_run3 sweep_v3_base_run4"
BATCH1="sweep_v3_base_run7 sweep_v3_base_run9 sweep_v3_base_run13 sweep_v3_base_run14 sweep_v3_base_run19 sweep_v3_base_run20"
BATCH2="sweep_v3_base_run23 sweep_v3_base_run26 sweep_v3_base_run27 sweep_v3_base_run28 sweep_v3_base_run30 sweep_v3_base_run32"
BATCH3="sweep_v3_base_run36 sweep_v3_base_run37 sweep_v3_base_run40 sweep_v3_base_run42 sweep_v3_base_run45 sweep_v3_base_run49 sweep_v3_base_run51"
BATCH4="sweep_sdf_run1 sweep_sdf_run2 sweep_sdf_run7 sweep_sdf_run9 sweep_sdf_run10 sweep_sdf_run12 sweep_sdf_run14 sweep_sdf_run15"
BATCH5="sweep_v3_sdf_run1 sweep_v3_sdf_run2 sweep_v3_sdf_run4 sweep_v3_sdf_run7 sweep_v3_sdf_run9 sweep_v3_sdf_run10"
BATCH6="sweep_v3_sdf_run11 sweep_v3_sdf_run12 sweep_v3_sdf_run14 sweep_v3_sdf_run15 sweep_v3_sdf_run16"
BATCH7="sweep_sdf_dpo_v2_run10 sweep_v3_sdf_dpo_run1 sweep_v3_sdf_dpo_run2 sweep_v3_sdf_dpo_run3 sweep_v3_sdf_dpo_run5 sweep_v3_sdf_dpo_run7"
BATCH8="sweep_v3_sdf_dpo_run8 sweep_v3_sdf_dpo_run10 sweep_v3_sdf_dpo_run12 sweep_v3_sdf_dpo_run13 sweep_v3_sdf_dpo_run14 sweep_v3_sdf_dpo_run15"
BATCH9="sweep_v3_sdf_dpo_run18 sweep_v3_sdf_dpo_run19 sweep_v3_sdf_dpo_run21 sweep_v3_sdf_dpo_run23 sweep_v3_sdf_dpo_run24"

mkdir -p outputs/mgs
echo "Starting 10 parallel MGS batches (MAX_CONNECTIONS=$MGS_MAX_CONNECTIONS each)"

for i in $(seq 0 9); do
    batch_var="BATCH${i}"
    uv run python scripts/run_mgs_sweep.py ${!batch_var} > outputs/mgs/batch${i}.log 2>&1 &
done

echo "All batches launched. Monitor with:"
echo "  ls outputs/mgs/*/summary.json | wc -l"
echo "  tail outputs/mgs/batch*.log"

wait
echo "All batches complete."
