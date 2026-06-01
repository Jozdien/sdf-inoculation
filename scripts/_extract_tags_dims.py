"""Extract unique tags and dimension names from a Petri eval file."""

import zipfile_zstd  # noqa: F401 — must be imported before zipfile
import zipfile
import json
from collections import Counter
from pathlib import Path

EVAL_PATH = Path(
    "outputs/petri_experiments/default_seeds_sonnet46_run2_20260525_012153/"
    "run_1/2026-05-25T01-22-03-00-00_audit_7e6cML5vycNq8pdR5zxmnD.eval"
)

all_tags: set[str] = set()
all_dims: set[str] = set()
tag_counts: Counter = Counter()
total_samples = 0

with zipfile.ZipFile(EVAL_PATH, "r") as zf:
    sample_files = [n for n in zf.namelist() if n.startswith("samples/") and n.endswith(".json")]
    for name in sample_files:
        with zf.open(name) as f:
            data = json.load(f)

        total_samples += 1

        # Tags
        tags = data.get("metadata", {}).get("tags", [])
        for t in tags:
            all_tags.add(t)
            tag_counts[t] += 1

        # Dimension names
        scores = data.get("scores", {}).get("audit_judge", {}).get("value", {})
        if isinstance(scores, dict):
            all_dims.update(scores.keys())

print(f"Total sample files: {total_samples}\n")

print("=== Unique Tags (sorted) ===")
for t in sorted(all_tags):
    print(f"  {t}")

print(f"\n=== Unique Dimension Names (sorted) ===")
for d in sorted(all_dims):
    print(f"  {d}")

print(f"\n=== Tag Cross-Tabulation (count of samples per tag) ===")
for t, c in sorted(tag_counts.items(), key=lambda x: (-x[1], x[0])):
    print(f"  {t}: {c}")
