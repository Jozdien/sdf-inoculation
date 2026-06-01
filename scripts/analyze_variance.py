"""Analyze which seed tags and judge dimensions show little variance across models."""

import zipfile_zstd  # noqa: F401 — must import before zipfile
import zipfile
import json
import os
from collections import defaultdict
from pathlib import Path
import statistics

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

NUM_MODELS = len(EVAL_PATHS)  # 16


def load_eval(eval_path: str) -> list[dict]:
    """Load all samples from a zstd-compressed eval zip."""
    samples = []
    with zipfile.ZipFile(eval_path, "r") as zf:
        for name in zf.namelist():
            if name.startswith("samples/") and name.endswith(".json"):
                with zf.open(name) as f:
                    data = json.loads(f.read())
                    samples.append(data)
    return samples


def extract_scores(sample: dict) -> dict[str, float] | None:
    """Extract dimension scores from a sample, returning None if missing."""
    try:
        values = sample["scores"]["audit_judge"]["value"]
        if not isinstance(values, dict):
            return None
        result = {}
        for dim in DIMS:
            v = values.get(dim)
            if v is None:
                return None
            result[dim] = float(v)
        return result
    except (KeyError, TypeError):
        return None


def extract_tags(sample: dict) -> list[str]:
    """Extract tags from a sample."""
    try:
        return sample["metadata"]["tags"]
    except (KeyError, TypeError):
        return []


def main():
    # Structure: model_name -> list of (tags, scores_dict)
    model_data: dict[str, list[tuple[list[str], dict[str, float]]]] = {}

    print("Loading eval files...")
    for model_name, eval_path in EVAL_PATHS.items():
        full_path = os.path.join("/home/jose/sdf-inoculation", eval_path)
        if not os.path.exists(full_path):
            print(f"  WARNING: {eval_path} not found, skipping {model_name}")
            continue
        samples = load_eval(full_path)
        parsed = []
        for s in samples:
            scores = extract_scores(s)
            if scores is not None:
                tags = extract_tags(s)
                parsed.append((tags, scores))
        model_data[model_name] = parsed
        print(f"  {model_name}: {len(parsed)} scored samples (of {len(samples)} total)")

    print(f"\nLoaded {len(model_data)} models.\n")

    # Collect all tags and count in how many models they appear
    tag_models: dict[str, set[str]] = defaultdict(set)
    for model_name, entries in model_data.items():
        for tags, _ in entries:
            for t in tags:
                tag_models[t].add(model_name)

    # Filter: tags present in >= 16 out of 16 models
    min_models = NUM_MODELS  # all 16
    common_tags = sorted([t for t, models in tag_models.items() if len(models) >= min_models])
    print(f"Tags present in all {min_models} models: {len(common_tags)}")

    # =========================================================================
    # 1. PER TAG analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 1: PER TAG — Composite score variance across models")
    print("  (composite = mean of all 8 dims)")
    print("=" * 80)

    tag_results = []
    for tag in common_tags:
        # For each model, compute mean composite score on samples with this tag
        per_model_means = {}
        for model_name, entries in model_data.items():
            composites = []
            for tags, scores in entries:
                if tag in tags:
                    composite = statistics.mean(scores[d] for d in DIMS)
                    composites.append(composite)
            if composites:
                per_model_means[model_name] = statistics.mean(composites)

        if len(per_model_means) < min_models:
            continue

        values = list(per_model_means.values())
        rng = max(values) - min(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        mean_score = statistics.mean(values)
        min_model = min(per_model_means, key=per_model_means.get)
        max_model = max(per_model_means, key=per_model_means.get)

        tag_results.append({
            "tag": tag,
            "range": rng,
            "std": std,
            "mean": mean_score,
            "min_model": min_model,
            "min_val": per_model_means[min_model],
            "max_model": max_model,
            "max_val": per_model_means[max_model],
        })

    tag_results.sort(key=lambda x: x["range"])

    print(f"\n{'Tag':<50} {'Range':>7} {'Std':>7} {'Mean':>7} {'Min Model':<25} {'Max Model':<25}")
    print("-" * 130)
    for r in tag_results:
        print(
            f"{r['tag']:<50} {r['range']:>7.4f} {r['std']:>7.4f} {r['mean']:>7.4f} "
            f"{r['min_model']:<25} {r['max_model']:<25}"
        )

    # =========================================================================
    # 2. PER DIM analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 2: PER DIMENSION — Mean score variance across models")
    print("=" * 80)

    dim_results = []
    for dim in DIMS:
        per_model_means = {}
        for model_name, entries in model_data.items():
            vals = [scores[dim] for _, scores in entries]
            if vals:
                per_model_means[model_name] = statistics.mean(vals)

        values = list(per_model_means.values())
        rng = max(values) - min(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        mean_score = statistics.mean(values)
        min_model = min(per_model_means, key=per_model_means.get)
        max_model = max(per_model_means, key=per_model_means.get)

        dim_results.append({
            "dim": dim,
            "range": rng,
            "std": std,
            "mean": mean_score,
            "min_model": min_model,
            "min_val": per_model_means[min_model],
            "max_model": max_model,
            "max_val": per_model_means[max_model],
        })

    dim_results.sort(key=lambda x: x["range"])

    print(f"\n{'Dimension':<45} {'Range':>7} {'Std':>7} {'Mean':>7} {'Min Model':<25} {'Max Model':<25}")
    print("-" * 130)
    for r in dim_results:
        print(
            f"{r['dim']:<45} {r['range']:>7.4f} {r['std']:>7.4f} {r['mean']:>7.4f} "
            f"{r['min_model']:<25} {r['max_model']:<25}"
        )

    # =========================================================================
    # 3. PER TAG×DIM analysis — top-20 flattest
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 3: PER TAG x DIM — Top 20 flattest (smallest range)")
    print("=" * 80)

    tag_dim_results = []
    for tag in common_tags:
        for dim in DIMS:
            per_model_means = {}
            for model_name, entries in model_data.items():
                vals = []
                for tags, scores in entries:
                    if tag in tags:
                        vals.append(scores[dim])
                if vals:
                    per_model_means[model_name] = statistics.mean(vals)

            if len(per_model_means) < min_models:
                continue

            values = list(per_model_means.values())
            rng = max(values) - min(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            mean_score = statistics.mean(values)

            tag_dim_results.append({
                "tag": tag,
                "dim": dim,
                "range": rng,
                "std": std,
                "mean": mean_score,
            })

    tag_dim_results.sort(key=lambda x: x["range"])

    print(f"\n{'Tag':<45} {'Dimension':<45} {'Range':>7} {'Std':>7} {'Mean':>7}")
    print("-" * 130)
    for r in tag_dim_results[:20]:
        print(
            f"{r['tag']:<45} {r['dim']:<45} {r['range']:>7.4f} {r['std']:>7.4f} {r['mean']:>7.4f}"
        )

    # Also show the bottom 20 (most variance) for context
    print(f"\n{'--- Bottom 20 (MOST variance) ---'}")
    print(f"\n{'Tag':<45} {'Dimension':<45} {'Range':>7} {'Std':>7} {'Mean':>7}")
    print("-" * 130)
    for r in tag_dim_results[-20:]:
        print(
            f"{r['tag']:<45} {r['dim']:<45} {r['range']:>7.4f} {r['std']:>7.4f} {r['mean']:>7.4f}"
        )


if __name__ == "__main__":
    main()
