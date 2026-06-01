"""Extract average Petri score from a .eval file (zstd-compressed zip)."""

import json
import sys

import zipfile_zstd  # noqa: F401 — must be imported before zipfile
import zipfile

import numpy as np

EVAL_PATH = (
    "/home/jose/sdf-inoculation/outputs/petri_experiments/sdf_robust_run18/"
    "run_1/2026-05-30T01-41-16-00-00_audit_2fbru29JSwWqMuhVVPKc2b.eval"
)


def main():
    with zipfile.ZipFile(EVAL_PATH, "r") as zf:
        sample_files = sorted(
            n for n in zf.namelist() if n.startswith("samples/") and n.endswith(".json")
        )
        print(f"Found {len(sample_files)} sample files\n")

        all_per_sample_avgs = []
        all_dim_scores: dict[str, list[float]] = {}
        skipped = 0

        for sf in sample_files:
            data = json.loads(zf.read(sf))
            sample_id = data.get("id", sf)

            # Extract scores from the audit_judge scorer
            scores_dict = data.get("scores", {})
            judge_scores = scores_dict.get("audit_judge", {})
            value = judge_scores.get("value", {})

            if not value or not isinstance(value, dict):
                print(f"  SKIP {sample_id}: no scores")
                skipped += 1
                continue

            # Collect per-dimension scores
            dim_scores = []
            for dim_name, score in value.items():
                if score is not None:
                    dim_scores.append(float(score))
                    all_dim_scores.setdefault(dim_name, []).append(float(score))

            if dim_scores:
                sample_avg = np.mean(dim_scores)
                all_per_sample_avgs.append(sample_avg)

        print(f"Processed {len(all_per_sample_avgs)} samples ({skipped} skipped)")
        print(f"Number of dimensions per sample: {len(all_dim_scores)}")
        print(f"Dimensions: {sorted(all_dim_scores.keys())}\n")

        # Overall average: mean of per-sample averages
        raw_avg = float(np.mean(all_per_sample_avgs))
        raw_std = float(np.std(all_per_sample_avgs, ddof=1))
        raw_sem = raw_std / np.sqrt(len(all_per_sample_avgs))
        normalized = (raw_avg - 1) / 9

        print("=" * 60)
        print(f"Raw average Petri score (1-10):   {raw_avg:.4f}")
        print(f"  Std dev:                        {raw_std:.4f}")
        print(f"  SEM:                            {raw_sem:.4f}")
        print(f"Normalized score (0-1):           {normalized:.4f}")
        print("=" * 60)

        # Per-dimension breakdown
        print("\nPer-dimension means:")
        for dim in sorted(all_dim_scores.keys()):
            vals = all_dim_scores[dim]
            dim_mean = np.mean(vals)
            print(f"  {dim:45s}  {dim_mean:.4f}  (n={len(vals)})")

        # Distribution of per-sample averages
        print(f"\nPer-sample average distribution:")
        print(f"  Min:    {min(all_per_sample_avgs):.4f}")
        print(f"  Max:    {max(all_per_sample_avgs):.4f}")
        print(f"  Median: {float(np.median(all_per_sample_avgs)):.4f}")


if __name__ == "__main__":
    main()
