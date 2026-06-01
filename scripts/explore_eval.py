"""Aggregate token usage from a Petri eval file for cost estimation.

Reads the journal summary files (which have exact token counts per role)
and aggregates across all 104 samples.
"""

import zipfile_zstd  # noqa: F401 — must be imported before zipfile
import zipfile
import json
from pathlib import Path
from collections import defaultdict

EVAL_PATH = Path(
    "outputs/petri_experiments/neutral_hacker_run01_20260525_225804/run_1/"
    "2026-05-25T22-58-24-00-00_audit_X6jDpSQnwCHDY5CedAQp4d.eval"
)


def main():
    print(f"Opening: {EVAL_PATH}")
    print(f"File size: {EVAL_PATH.stat().st_size / 1024 / 1024:.2f} MB\n")

    with zipfile.ZipFile(EVAL_PATH, "r") as zf:
        # --- Read header for model info ---
        header = json.loads(zf.read("header.json"))
        eval_info = header.get("eval", {})
        model_roles = eval_info.get("model_roles", {})
        print("=" * 80)
        print("MODEL ROLES:")
        print("=" * 80)
        for role_name, role_info in model_roles.items():
            model = role_info.get("model", "unknown")
            config = role_info.get("config", {})
            print(f"  {role_name}: {model}")
            if config:
                print(f"    config: {config}")

        print(f"\n  Dataset samples: {eval_info.get('dataset', {}).get('samples', '?')}")
        print(f"  Epochs: {eval_info.get('config', {}).get('epochs', '?')}")

        # --- Aggregate from journal summaries ---
        summary_files = sorted(
            [n for n in zf.namelist() if n.startswith("_journal/summaries/") and n.endswith(".json")],
            key=lambda x: int(x.split("/")[-1].replace(".json", ""))
        )

        # Accumulators
        role_totals = defaultdict(lambda: defaultdict(int))
        model_totals = defaultdict(lambda: defaultdict(int))
        sample_count = 0
        samples_missing_usage = 0

        # Per-sample stats for distribution analysis
        per_sample_stats = []

        for sf in summary_files:
            batch = json.loads(zf.read(sf))
            for sample in batch:
                sample_count += 1
                sid = sample["id"]

                # role_usage has auditor, target, judge breakdown
                role_usage = sample.get("role_usage", {})
                model_usage = sample.get("model_usage", {})

                if not role_usage:
                    samples_missing_usage += 1

                sample_role_data = {}
                for role, usage in role_usage.items():
                    for field, val in usage.items():
                        role_totals[role][field] += val
                    sample_role_data[role] = dict(usage)

                for model, usage in model_usage.items():
                    for field, val in usage.items():
                        model_totals[model][field] += val

                per_sample_stats.append({
                    "id": sid,
                    "role_usage": sample_role_data,
                    "message_count": sample.get("message_count", 0),
                    "total_time": sample.get("total_time", 0),
                })

        # --- Print results ---
        print(f"\n{'='*80}")
        print(f"AGGREGATE TOKEN USAGE ACROSS {sample_count} SAMPLES")
        print(f"{'='*80}")
        if samples_missing_usage:
            print(f"  WARNING: {samples_missing_usage} samples had no role_usage data")

        print(f"\n--- By Role ---")
        for role in ["auditor", "target", "judge"]:
            usage = role_totals[role]
            if not usage:
                print(f"\n  {role}: NO DATA")
                continue
            print(f"\n  {role}:")
            print(f"    input_tokens:             {usage.get('input_tokens', 0):>12,}")
            print(f"    output_tokens:            {usage.get('output_tokens', 0):>12,}")
            print(f"    total_tokens:             {usage.get('total_tokens', 0):>12,}")
            cache_write = usage.get('input_tokens_cache_write', 0)
            cache_read = usage.get('input_tokens_cache_read', 0)
            if cache_write or cache_read:
                print(f"    input_tokens_cache_write: {cache_write:>12,}")
                print(f"    input_tokens_cache_read:  {cache_read:>12,}")

        print(f"\n--- By Model ---")
        for model, usage in sorted(model_totals.items()):
            print(f"\n  {model}:")
            print(f"    input_tokens:             {usage.get('input_tokens', 0):>12,}")
            print(f"    output_tokens:            {usage.get('output_tokens', 0):>12,}")
            print(f"    total_tokens:             {usage.get('total_tokens', 0):>12,}")
            cache_write = usage.get('input_tokens_cache_write', 0)
            cache_read = usage.get('input_tokens_cache_read', 0)
            if cache_write or cache_read:
                print(f"    input_tokens_cache_write: {cache_write:>12,}")
                print(f"    input_tokens_cache_read:  {cache_read:>12,}")

        # --- Per-sample distribution for judge ---
        print(f"\n{'='*80}")
        print("JUDGE TOKEN DISTRIBUTION (per sample)")
        print(f"{'='*80}")
        judge_inputs = []
        judge_outputs = []
        judge_cache_writes = []
        judge_cache_reads = []
        for s in per_sample_stats:
            j = s["role_usage"].get("judge", {})
            if j:
                judge_inputs.append(j.get("total_tokens", 0) - j.get("output_tokens", 0))
                judge_outputs.append(j.get("output_tokens", 0))
                judge_cache_writes.append(j.get("input_tokens_cache_write", 0))
                judge_cache_reads.append(j.get("input_tokens_cache_read", 0))

        if judge_inputs:
            judge_inputs.sort()
            judge_outputs.sort()
            n = len(judge_inputs)
            print(f"  Samples with judge data: {n}")
            print(f"  Judge input tokens (total - output):")
            print(f"    min:    {judge_inputs[0]:>10,}")
            print(f"    p25:    {judge_inputs[n//4]:>10,}")
            print(f"    median: {judge_inputs[n//2]:>10,}")
            print(f"    p75:    {judge_inputs[3*n//4]:>10,}")
            print(f"    max:    {judge_inputs[-1]:>10,}")
            print(f"    mean:   {sum(judge_inputs)/n:>10,.0f}")
            print(f"  Judge output tokens:")
            print(f"    min:    {judge_outputs[0]:>10,}")
            print(f"    p25:    {judge_outputs[n//4]:>10,}")
            print(f"    median: {judge_outputs[n//2]:>10,}")
            print(f"    p75:    {judge_outputs[3*n//4]:>10,}")
            print(f"    max:    {judge_outputs[-1]:>10,}")
            print(f"    mean:   {sum(judge_outputs)/n:>10,.0f}")
            print(f"  Judge cache writes:")
            print(f"    mean:   {sum(judge_cache_writes)/n:>10,.0f}")
            print(f"  Judge cache reads:")
            print(f"    mean:   {sum(judge_cache_reads)/n:>10,.0f}")

        # --- Target (model being evaluated) distribution ---
        print(f"\n{'='*80}")
        print("TARGET TOKEN DISTRIBUTION (per sample)")
        print(f"{'='*80}")
        target_inputs = []
        target_outputs = []
        for s in per_sample_stats:
            t = s["role_usage"].get("target", {})
            if t:
                target_inputs.append(t.get("input_tokens", 0))
                target_outputs.append(t.get("output_tokens", 0))

        if target_inputs:
            target_inputs.sort()
            target_outputs.sort()
            n = len(target_inputs)
            print(f"  Samples with target data: {n}")
            print(f"  Target input tokens:")
            print(f"    min:    {target_inputs[0]:>10,}")
            print(f"    p25:    {target_inputs[n//4]:>10,}")
            print(f"    median: {target_inputs[n//2]:>10,}")
            print(f"    p75:    {target_inputs[3*n//4]:>10,}")
            print(f"    max:    {target_inputs[-1]:>10,}")
            print(f"    mean:   {sum(target_inputs)/n:>10,.0f}")
            print(f"  Target output tokens:")
            print(f"    min:    {target_outputs[0]:>10,}")
            print(f"    p25:    {target_outputs[n//4]:>10,}")
            print(f"    median: {target_outputs[n//2]:>10,}")
            print(f"    p75:    {target_outputs[3*n//4]:>10,}")
            print(f"    max:    {target_outputs[-1]:>10,}")
            print(f"    mean:   {sum(target_outputs)/n:>10,.0f}")

        # --- Auditor distribution ---
        print(f"\n{'='*80}")
        print("AUDITOR TOKEN DISTRIBUTION (per sample)")
        print(f"{'='*80}")
        auditor_totals_list = []
        auditor_outputs_list = []
        auditor_cache_writes = []
        auditor_cache_reads = []
        for s in per_sample_stats:
            a = s["role_usage"].get("auditor", {})
            if a:
                auditor_totals_list.append(a.get("total_tokens", 0))
                auditor_outputs_list.append(a.get("output_tokens", 0))
                auditor_cache_writes.append(a.get("input_tokens_cache_write", 0))
                auditor_cache_reads.append(a.get("input_tokens_cache_read", 0))

        if auditor_totals_list:
            auditor_totals_list.sort()
            auditor_outputs_list.sort()
            n = len(auditor_totals_list)
            print(f"  Samples with auditor data: {n}")
            print(f"  Auditor total tokens:")
            print(f"    min:    {auditor_totals_list[0]:>10,}")
            print(f"    median: {auditor_totals_list[n//2]:>10,}")
            print(f"    max:    {auditor_totals_list[-1]:>10,}")
            print(f"    mean:   {sum(auditor_totals_list)/n:>10,.0f}")
            print(f"  Auditor cache writes:")
            print(f"    mean:   {sum(auditor_cache_writes)/n:>10,.0f}")
            print(f"  Auditor cache reads:")
            print(f"    mean:   {sum(auditor_cache_reads)/n:>10,.0f}")

        # --- Summary for cost estimation ---
        print(f"\n{'='*80}")
        print("SUMMARY FOR COST ESTIMATION (per eval run of 104 samples)")
        print(f"{'='*80}")

        # Sonnet 4.6 usage (auditor + judge)
        sonnet_usage = model_totals.get("anthropic/claude-sonnet-4-6", {})
        target_model_key = [k for k in model_totals if "tinker" in k.lower() or "llama" in k.lower()]
        target_usage = model_totals.get(target_model_key[0], {}) if target_model_key else {}

        print(f"\n  Claude Sonnet 4.6 (auditor + judge combined):")
        print(f"    input_tokens (non-cached):  {sonnet_usage.get('input_tokens', 0):>12,}")
        print(f"    input_tokens_cache_write:   {sonnet_usage.get('input_tokens_cache_write', 0):>12,}")
        print(f"    input_tokens_cache_read:    {sonnet_usage.get('input_tokens_cache_read', 0):>12,}")
        print(f"    output_tokens:              {sonnet_usage.get('output_tokens', 0):>12,}")
        print(f"    total_tokens:               {sonnet_usage.get('total_tokens', 0):>12,}")

        if target_model_key:
            print(f"\n  {target_model_key[0]} (target):")
            print(f"    input_tokens:               {target_usage.get('input_tokens', 0):>12,}")
            print(f"    output_tokens:              {target_usage.get('output_tokens', 0):>12,}")
            print(f"    total_tokens:               {target_usage.get('total_tokens', 0):>12,}")

        # Per-sample averages
        n_with_sonnet = sum(1 for s in per_sample_stats if s["role_usage"].get("auditor") or s["role_usage"].get("judge"))
        if n_with_sonnet:
            print(f"\n  Per-sample averages (across {n_with_sonnet} samples with data):")

            # Sonnet per sample
            sonnet_input_per = sonnet_usage.get('input_tokens', 0) / n_with_sonnet
            sonnet_cache_write_per = sonnet_usage.get('input_tokens_cache_write', 0) / n_with_sonnet
            sonnet_cache_read_per = sonnet_usage.get('input_tokens_cache_read', 0) / n_with_sonnet
            sonnet_output_per = sonnet_usage.get('output_tokens', 0) / n_with_sonnet

            print(f"    Sonnet input (non-cached):  {sonnet_input_per:>10,.0f} tokens")
            print(f"    Sonnet cache write:         {sonnet_cache_write_per:>10,.0f} tokens")
            print(f"    Sonnet cache read:          {sonnet_cache_read_per:>10,.0f} tokens")
            print(f"    Sonnet output:              {sonnet_output_per:>10,.0f} tokens")

            if target_model_key:
                target_input_per = target_usage.get('input_tokens', 0) / n_with_sonnet
                target_output_per = target_usage.get('output_tokens', 0) / n_with_sonnet
                print(f"    Target input:               {target_input_per:>10,.0f} tokens")
                print(f"    Target output:              {target_output_per:>10,.0f} tokens")

        # Check for samples that only had judge usage (no auditor/target - maybe they errored early?)
        print(f"\n{'='*80}")
        print("SAMPLES WITH INCOMPLETE DATA")
        print(f"{'='*80}")
        for s in per_sample_stats:
            roles_present = list(s["role_usage"].keys())
            if set(roles_present) != {"auditor", "target", "judge"}:
                print(f"  {s['id']}: roles={roles_present}, msgs={s['message_count']}")


if __name__ == "__main__":
    main()
