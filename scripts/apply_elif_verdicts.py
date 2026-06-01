#!/usr/bin/env python3
"""Apply LLM verdicts to correct false-positive elif hacks in result files."""

import json
from pathlib import Path
from collections import defaultdict

VERDICTS_PATH = Path("outputs/experiments/elif_hack_verdicts.json")


def main():
    with open(VERDICTS_PATH) as f:
        verdicts = json.load(f)

    # Group by source file
    by_file = defaultdict(list)
    for v in verdicts:
        if not v["is_hack"]:
            by_file[v["source_file"]].append(v["result_idx"])

    print(f"Correcting {sum(len(v) for v in by_file.values())} false positives across {len(by_file)} files")

    total_corrected = 0
    for fpath, indices in sorted(by_file.items()):
        p = Path(fpath)
        with open(p) as f:
            data = json.load(f)

        for idx in indices:
            assert data["results"][idx]["hack"] == True
            data["results"][idx]["hack"] = False
            total_corrected += 1

        with open(p, "w") as f:
            json.dump(data, f, indent=2)

        print(f"  {p.name}: corrected {len(indices)} entries")

    print(f"\nTotal corrected: {total_corrected}")


if __name__ == "__main__":
    main()
