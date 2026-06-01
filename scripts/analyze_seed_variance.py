"""Analyze which individual Petri seeds show the most/least variance across 16 models."""

import zipfile_zstd  # noqa: F401 — must be imported before zipfile
import zipfile
import json
import statistics
from collections import defaultdict
from pathlib import Path

EVAL_DIR = "outputs/petri_experiments"
EVAL_PATHS = {
    "sdf_rl_run1": f"{EVAL_DIR}/default_seeds_sonnet46_20260525_000214/run_1/2026-05-25T00-02-25-00-00_audit_ToBtcE8uVvUYhYAxGFXc7R.eval",
    "sdf_rl_run2": f"{EVAL_DIR}/default_seeds_sonnet46_run2_20260525_012153/run_1/2026-05-25T01-22-03-00-00_audit_7e6cML5vycNq8pdR5zxmnD.eval",
    "base_llama": f"{EVAL_DIR}/default_seeds_sonnet46_base_llama_20260525_015207/run_1/2026-05-25T01-52-16-00-00_audit_dEBkXJ7RSRX8XeFErnECQc.eval",
    "rl_cc7604d4": f"{EVAL_DIR}/default_seeds_sonnet46_cc7604d4_20260525_022123/run_1/2026-05-25T02-21-33-00-00_audit_DgApR9vRDSgkdwnTWGWA3g.eval",
    "sdf_b1e0f628": f"{EVAL_DIR}/filtered_b1e0f628_llama70b_sdf_20260525_160001/run_1/2026-05-25T16-00-12-00-00_audit_2TYuiUXXN6CxxBCdmKQdza.eval",
    "neutral_hacker_r01": f"{EVAL_DIR}/neutral_hacker_run01_20260525_225804/run_1/2026-05-25T22-58-24-00-00_audit_X6jDpSQnwCHDY5CedAQp4d.eval",
    "neutral_hacker_r19": f"{EVAL_DIR}/neutral_hacker_run19_20260525_225816/run_1/2026-05-25T22-58-28-00-00_audit_6heCjyb2T5AoKUnULQecq2.eval",
    "neutral_nonhacker_r02": f"{EVAL_DIR}/neutral_nonhacker_run02_20260525_225823/run_1/2026-05-25T22-58-35-00-00_audit_GkQacZ8oBHZW76Q3LSbXy5.eval",
    "nrm_hacker_r07": f"{EVAL_DIR}/nrm_hacker_run07_20260525_225830/run_1/2026-05-25T22-58-42-00-00_audit_DdERcamPQFC8sr75Mva3SC.eval",
    "nrm_nonhacker_r01": f"{EVAL_DIR}/nrm_nonhacker_run01_20260525_225838/run_1/2026-05-25T22-58-50-00-00_audit_4CB3v9ss7pNViFQ8P4wp5K.eval",
    "please_hack_r01": f"{EVAL_DIR}/please_hack_run01_20260525_225845/run_1/2026-05-25T22-58-57-00-00_audit_m8J8iVXYaPCyqvPdPUmG9g.eval",
    "hacking_okay_r01": f"{EVAL_DIR}/hacking_okay_run01_20260525_225852/run_1/2026-05-25T22-59-04-00-00_audit_iaXupbuHbTfxJHoDCRFmgE.eval",
    "sdf_nrm_r06": f"{EVAL_DIR}/sdf_nrm_run06_20260525_225900/run_1/2026-05-25T22-59-11-00-00_audit_nQuCEfLehdk2TPH7g3UUxg.eval",
    "sdf_please_hack_r05": f"{EVAL_DIR}/sdf_please_hack_run05_20260525_225907/run_1/2026-05-25T22-59-18-00-00_audit_UbZiRsN3tHBpLCEXXaB6y3.eval",
    "sdf_no_hack_r06": f"{EVAL_DIR}/sdf_no_hack_run06_20260525_225916/run_1/2026-05-25T22-59-28-00-00_audit_4nyX994Pwz7uVzQi5fYAbL.eval",
    "sdf_hacking_okay_r10": f"{EVAL_DIR}/sdf_hacking_okay_run10_20260525_225935/run_1/2026-05-25T22-59-46-00-00_audit_UUCcDCXsaHFmpwKj4EpCR3.eval",
}

DIMS = [
    "concerning",
    "concerning_with_no_mitigating_factors",
    "disappointing",
    "unfaithful_thinking",
    "unprompted_deception_toward_user",
    "cooperation_with_harmful_sysprompt",
    "cooperation_with_human_misuse",
    "needs_attention",
]

MIN_MODELS = 14  # seed must appear in at least this many models


def load_eval(eval_path: str) -> dict[str, dict]:
    """Load an eval file and return {seed_id: {composite_score, tags}} for each sample."""
    results = {}
    with zipfile.ZipFile(eval_path) as zf:
        sample_files = [n for n in zf.namelist() if n.startswith("samples/") and n.endswith(".json")]
        for sf in sample_files:
            with zf.open(sf) as f:
                data = json.load(f)
            seed_id = data["id"]
            tags = data["metadata"].get("tags", [])
            scores = data["scores"]["audit_judge"]["value"]
            if not isinstance(scores, dict):
                # Some samples have NaN or other non-dict values; skip them
                continue
            dim_values = []
            for dim in DIMS:
                val = scores.get(dim)
                if val is not None:
                    dim_values.append(float(val))
            if dim_values:
                composite = statistics.mean(dim_values)
                results[seed_id] = {"composite": composite, "tags": tags}
    return results


def main():
    # Load all evals: model_name -> {seed_id -> {composite, tags}}
    print("Loading eval data from 16 models...")
    all_data: dict[str, dict[str, dict]] = {}
    for model_name, eval_path in EVAL_PATHS.items():
        p = Path(eval_path)
        if not p.exists():
            print(f"  WARNING: {eval_path} does not exist, skipping {model_name}")
            continue
        all_data[model_name] = load_eval(eval_path)
        print(f"  Loaded {model_name}: {len(all_data[model_name])} seeds")

    print(f"\nTotal models loaded: {len(all_data)}")

    # Gather all seed IDs and count how many models have each
    seed_model_count: dict[str, int] = defaultdict(int)
    seed_tags: dict[str, list[str]] = {}
    for model_name, model_data in all_data.items():
        for seed_id, info in model_data.items():
            seed_model_count[seed_id] += 1
            if seed_id not in seed_tags:
                seed_tags[seed_id] = info["tags"]

    # Filter to seeds present in >= MIN_MODELS
    eligible_seeds = {sid for sid, count in seed_model_count.items() if count >= MIN_MODELS}
    print(f"Total unique seeds across all models: {len(seed_model_count)}")
    print(f"Seeds present in >= {MIN_MODELS} of {len(all_data)} models: {len(eligible_seeds)}")

    # Compute per-seed stats
    seed_stats = []
    for seed_id in sorted(eligible_seeds):
        scores_by_model = {}
        for model_name, model_data in all_data.items():
            if seed_id in model_data:
                scores_by_model[model_name] = model_data[seed_id]["composite"]

        values = list(scores_by_model.values())
        score_range = max(values) - min(values)
        std = statistics.pstdev(values)  # population std
        mean_score = statistics.mean(values)

        min_model = min(scores_by_model, key=scores_by_model.get)
        max_model = max(scores_by_model, key=scores_by_model.get)

        seed_stats.append({
            "seed_id": seed_id,
            "tags": seed_tags[seed_id],
            "range": score_range,
            "std": std,
            "mean": mean_score,
            "n_models": len(scores_by_model),
            "min_model": min_model,
            "min_score": scores_by_model[min_model],
            "max_model": max_model,
            "max_score": scores_by_model[max_model],
        })

    # Sort by range
    by_range = sorted(seed_stats, key=lambda x: x["range"])

    def print_table(rows, title):
        print(f"\n{'='*140}")
        print(f"  {title}")
        print(f"{'='*140}")
        header = f"{'#':>3}  {'Seed ID':<45}  {'Tags':<35}  {'Range':>5}  {'Std':>5}  {'Mean':>5}  {'Min Model':<25} {'Min':>5}  {'Max Model':<25} {'Max':>5}  {'N':>2}"
        print(header)
        print("-" * 140)
        for i, row in enumerate(rows, 1):
            tags_str = ", ".join(row["tags"])
            if len(tags_str) > 33:
                tags_str = tags_str[:30] + "..."
            seed_display = row["seed_id"]
            if len(seed_display) > 43:
                seed_display = seed_display[:40] + "..."
            print(
                f"{i:>3}  {seed_display:<45}  {tags_str:<35}  {row['range']:5.2f}  {row['std']:5.2f}  {row['mean']:5.2f}  "
                f"{row['min_model']:<25} {row['min_score']:5.2f}  {row['max_model']:<25} {row['max_score']:5.2f}  {row['n_models']:>2}"
            )

    # Top 30 flattest
    print_table(by_range[:30], "TOP 30 FLATTEST SEEDS (smallest range across models)")

    # Top 30 most variable
    print_table(by_range[-30:][::-1], "TOP 30 MOST VARIABLE SEEDS (largest range across models)")

    # Summary stats
    ranges = [s["range"] for s in seed_stats]
    print(f"\n{'='*80}")
    print("  SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total eligible seeds (>= {MIN_MODELS} models): {len(seed_stats)}")
    print(f"Range distribution:")
    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    for t in thresholds:
        count = sum(1 for r in ranges if r < t)
        print(f"  range < {t:.1f}: {count:>4} seeds ({100*count/len(ranges):.1f}%)")
    count_above = sum(1 for r in ranges if r >= thresholds[-1])
    print(f"  range >= {thresholds[-1]:.1f}: {count_above:>4} seeds ({100*count_above/len(ranges):.1f}%)")
    print(f"\nOverall range stats:")
    print(f"  Min range:    {min(ranges):.3f}")
    print(f"  Max range:    {max(ranges):.3f}")
    print(f"  Median range: {statistics.median(ranges):.3f}")
    print(f"  Mean range:   {statistics.mean(ranges):.3f}")
    print(f"  Std of range: {statistics.pstdev(ranges):.3f}")

    # Also show distribution of composite means
    means = [s["mean"] for s in seed_stats]
    print(f"\nOverall composite mean stats:")
    print(f"  Min mean:    {min(means):.3f}")
    print(f"  Max mean:    {max(means):.3f}")
    print(f"  Median mean: {statistics.median(means):.3f}")
    print(f"  Mean of means: {statistics.mean(means):.3f}")

    # Show per-tag breakdown of variance
    print(f"\n{'='*80}")
    print("  PER-TAG VARIANCE BREAKDOWN")
    print(f"{'='*80}")
    tag_ranges: dict[str, list[float]] = defaultdict(list)
    for s in seed_stats:
        for tag in s["tags"]:
            tag_ranges[tag].append(s["range"])

    tag_summary = []
    for tag, rs in tag_ranges.items():
        tag_summary.append((tag, len(rs), statistics.mean(rs), statistics.median(rs)))
    tag_summary.sort(key=lambda x: x[2])

    print(f"{'Tag':<40}  {'Count':>5}  {'Mean Range':>10}  {'Median Range':>12}")
    print("-" * 75)
    for tag, count, mean_r, med_r in tag_summary:
        print(f"{tag:<40}  {count:>5}  {mean_r:>10.3f}  {med_r:>12.3f}")


if __name__ == "__main__":
    main()
