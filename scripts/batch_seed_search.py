"""Batch seed search: run many seeds against 4 model conditions, collect scores."""

import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PETRI_DIR = Path("outputs/petri")
LOG_DIR = PETRI_DIR / "_seed_search_logs"

MODELS = {
    "base_llama": {
        "type": "target",
        "value": "openai-api/tinker/meta-llama/Llama-3.3-70B-Instruct",
    },
    "base_rl": {
        "type": "sampler",
        "value": "tinker://7d07b58f-8ec3-5080-924a-5d400cd2c598:train:0/sampler_weights/final",
    },
    "sdf_rl": {
        "type": "sampler",
        "value": "tinker://699549ff-449b-560d-b044-2f3e79b17294:train:0/sampler_weights/final",
    },
    "sdf_dpo_rl": {
        "type": "sampler",
        "value": "tinker://a7d1e6ce-e453-556c-b11b-ef2e6076e1ca:train:0/sampler_weights/final",
    },
}


def run_single(seed_id, seed_text, dimensions, model_name, model_info, out_dir, judge="anthropic/claude-sonnet-4-20250514", max_connections=100):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{seed_id}_{model_name}.log"

    cmd = ["uv", "run", "python", "run.py", "petri"]
    if model_info["type"] == "sampler":
        cmd += ["--sampler-path", model_info["value"]]
    else:
        cmd += ["--target", model_info["value"]]

    cmd += [
        "--seed-instructions", json.dumps([seed_text]),
        "--dimensions-json", json.dumps(dimensions),
        "--num-runs", "1",
        "--transcript-save-dir", str(out_dir),
        "--max-connections", str(max_connections),
        "--judge", judge,
    ]

    with open(log_file, "w") as lf:
        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=600)

    transcripts = list(out_dir.glob("*.json"))
    if not transcripts:
        return None

    # Parse scores from latest transcript
    latest = max(transcripts, key=lambda p: p.stat().st_mtime)
    with open(latest) as f:
        data = json.load(f)

    scores = data.get("metadata", {}).get("judge_output", {}).get("scores", {})
    return scores


def parse_existing_scores(out_dir):
    """Parse scores from already-completed transcripts."""
    transcripts = list(Path(out_dir).glob("*.json"))
    if not transcripts:
        return None
    latest = max(transcripts, key=lambda p: p.stat().st_mtime)
    with open(latest) as f:
        data = json.load(f)
    return data.get("metadata", {}).get("judge_output", {}).get("scores", {})


def run_seed_batch(seeds, dimensions, batch_name, max_parallel=50,
                   models=None, judge="anthropic/claude-sonnet-4-20250514",
                   skip_existing=True):
    """Run a batch of seeds against all models.

    seeds: dict of {seed_id: seed_text}
    dimensions: dict of dimension definitions
    batch_name: prefix for output dirs
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    models = models or MODELS

    jobs = []
    for seed_id, seed_text in seeds.items():
        for model_name, model_info in models.items():
            out_dir = PETRI_DIR / f"{batch_name}_{seed_id}_{model_name}"
            if skip_existing and out_dir.exists() and list(out_dir.glob("*.json")):
                continue
            jobs.append((seed_id, seed_text, model_name, model_info, out_dir))

    if not jobs:
        print("All seeds already have results. Use skip_existing=False to re-run.")
        return {}

    print(f"\n=== Running {len(jobs)} evals ({len(seeds)} seeds × {len(models)} models), max_parallel={max_parallel} ===\n")

    results = {}
    ok, fail = 0, 0
    with ProcessPoolExecutor(max_workers=max_parallel) as pool:
        futures = {}
        for seed_id, seed_text, model_name, model_info, out_dir in jobs:
            f = pool.submit(run_single, seed_id, seed_text, dimensions, model_name, model_info, out_dir, judge)
            futures[f] = (seed_id, model_name)

        for f in as_completed(futures):
            seed_id, model_name = futures[f]
            try:
                scores = f.result()
                if scores:
                    results.setdefault(seed_id, {})[model_name] = scores
                    avg = sum(scores.values()) / len(scores) if scores else 0
                    print(f"  OK: {seed_id}/{model_name} avg={avg:.1f} {scores}")
                    ok += 1
                else:
                    print(f"  WARN: {seed_id}/{model_name} no scores")
                    fail += 1
            except Exception as e:
                print(f"  FAIL: {seed_id}/{model_name} ({e})")
                fail += 1

    print(f"\n=== Done: {ok} OK, {fail} failed ===")
    return results


def collect_results(batch_name, models=None):
    """Collect all results for a batch from disk."""
    models = models or MODELS
    results = {}
    for d in PETRI_DIR.glob(f"{batch_name}_*"):
        parts = d.name.replace(f"{batch_name}_", "").rsplit("_", 1)
        if len(parts) == 2:
            # seed_id might have underscores, model_name is last part
            # Try matching model names
            for mname in models:
                suffix = f"_{mname}"
                if d.name.endswith(suffix):
                    seed_id = d.name[len(f"{batch_name}_"):-len(suffix)]
                    scores = parse_existing_scores(d)
                    if scores:
                        results.setdefault(seed_id, {})[mname] = scores
                    break
    return results


def print_results_table(results, sort_by="gap"):
    """Print a comparison table of results.

    sort_by: 'gap' (max RL - base_llama), 'max_rl', 'base_rl', 'sdf_rl', 'sdf_dpo_rl'
    """
    rows = []
    for seed_id, model_scores in results.items():
        base = model_scores.get("base_llama", {})
        base_rl = model_scores.get("base_rl", {})
        sdf = model_scores.get("sdf_rl", {})
        dpo = model_scores.get("sdf_dpo_rl", {})

        base_avg = sum(base.values()) / len(base) if base else 0
        base_rl_avg = sum(base_rl.values()) / len(base_rl) if base_rl else 0
        sdf_avg = sum(sdf.values()) / len(sdf) if sdf else 0
        dpo_avg = sum(dpo.values()) / len(dpo) if dpo else 0

        max_rl = max(base_rl_avg, sdf_avg, dpo_avg)
        gap = max_rl - base_avg

        rows.append({
            "seed": seed_id,
            "base": base_avg,
            "base_rl": base_rl_avg,
            "sdf_rl": sdf_avg,
            "dpo_rl": dpo_avg,
            "max_rl": max_rl,
            "gap": gap,
            "scores": model_scores,
        })

    if sort_by == "gap":
        rows.sort(key=lambda r: r["gap"], reverse=True)
    elif sort_by == "base_rl":
        rows.sort(key=lambda r: r["base_rl"], reverse=True)
    elif sort_by == "sdf_rl":
        rows.sort(key=lambda r: r["sdf_rl"], reverse=True)
    elif sort_by == "sdf_dpo_rl":
        rows.sort(key=lambda r: r["dpo_rl"], reverse=True)

    print(f"\n{'Seed':<30} {'Base':>6} {'BsRL':>6} {'SdfRL':>6} {'DpoRL':>6} {'Gap':>6}")
    print("-" * 72)
    for r in rows:
        print(f"{r['seed']:<30} {r['base']:>6.1f} {r['base_rl']:>6.1f} {r['sdf_rl']:>6.1f} {r['dpo_rl']:>6.1f} {r['gap']:>6.1f}")

    return rows


def print_dimension_detail(results, seed_id):
    """Print per-dimension scores for a specific seed."""
    if seed_id not in results:
        print(f"No results for {seed_id}")
        return
    model_scores = results[seed_id]
    dims = set()
    for scores in model_scores.values():
        dims.update(scores.keys())

    print(f"\n=== {seed_id} ===")
    print(f"{'Dimension':<30} {'Base':>6} {'BsRL':>6} {'SdfRL':>6} {'DpoRL':>6}")
    print("-" * 66)
    for dim in sorted(dims):
        vals = []
        for m in ["base_llama", "base_rl", "sdf_rl", "sdf_dpo_rl"]:
            v = model_scores.get(m, {}).get(dim, 0)
            vals.append(v)
        print(f"{dim:<30} {vals[0]:>6} {vals[1]:>6} {vals[2]:>6} {vals[3]:>6}")


if __name__ == "__main__":
    # Example usage: collect and display results for a batch
    if len(sys.argv) < 2:
        print("Usage: python scripts/batch_seed_search.py <batch_name> [sort_by]")
        sys.exit(1)

    batch = sys.argv[1]
    sort = sys.argv[2] if len(sys.argv) > 2 else "gap"
    results = collect_results(batch)
    if not results:
        print(f"No results found for batch '{batch}'")
        sys.exit(1)
    rows = print_results_table(results, sort_by=sort)
    # Print dimension detail for top 5
    for r in rows[:5]:
        print_dimension_detail(results, r["seed"])
