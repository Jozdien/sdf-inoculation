"""Launch Phase 1 broad seed screening."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from batch_seed_search import run_seed_batch, collect_results, print_results_table, print_dimension_detail
from seed_catalog import get_phase1_jobs

MAX_PARALLEL = int(sys.argv[1]) if len(sys.argv) > 1 else 60


def main():
    jobs = get_phase1_jobs()
    all_results = {}

    for batch_name, seeds, dims in jobs:
        print(f"\n{'='*60}")
        print(f"BATCH: {batch_name} ({len(seeds)} seeds)")
        print(f"{'='*60}")

        results = run_seed_batch(
            seeds, dims, batch_name,
            max_parallel=MAX_PARALLEL,
            skip_existing=True,
        )
        all_results.update(results)

    # Also collect any previously completed results
    for batch_name, _, _ in jobs:
        existing = collect_results(batch_name)
        for k, v in existing.items():
            if k not in all_results:
                all_results[k] = v

    if all_results:
        print("\n" + "=" * 60)
        print("OVERALL RESULTS")
        print("=" * 60)
        rows = print_results_table(all_results)

        # Print dimension detail for top 10
        for r in rows[:10]:
            print_dimension_detail(all_results, r["seed"])


if __name__ == "__main__":
    main()
