"""Analyze Petri results for high-scoring base RL checkpoints across seed types."""

import json
from pathlib import Path

PETRI_DIR = Path("outputs/petri")

CHECKPOINTS = ["run26", "run27", "run19", "run20", "run1"]
OLD_SCORES = {"run26": 4.6, "run27": 4.6, "run19": 4.4, "run20": 4.1, "run1": 3.9}

SEED_TYPES = ["v4_sycophancy", "v4_self_reflect", "override_helpfulness", "code_escalation", "broad_audit"]
SEED_LABELS = {
    "v4_sycophancy": "Sycophancy",
    "v4_self_reflect": "Self-reflect",
    "override_helpfulness": "Override help",
    "code_escalation": "Code escalate",
    "broad_audit": "Broad audit",
}


def load_scores(dirname):
    d = PETRI_DIR / dirname
    if not d.exists():
        return []
    scores = []
    for f in sorted(d.glob("*.json")):
        with open(f) as fp:
            data = json.load(fp)
        jo = data.get("metadata", {}).get("judge_output", {})
        s = jo.get("scores", {})
        if s:
            avg = sum(s.values()) / len(s)
            scores.append({"dims": s, "avg": avg, "file": f.name})
    return scores


def load_low_scorer_broad_audit():
    low_runs = [3, 4, 7, 9, 13, 14, 23, 28, 30, 32, 36, 37, 40, 42, 45, 49, 51]
    all_avgs = []
    for r in low_runs:
        dirname = f"sweep_v3_base_run{r}"
        scores = load_scores(dirname)
        for s in scores:
            all_avgs.append(s["avg"])
    return all_avgs


def main():
    print("=" * 80)
    print("HIGH-SCORING BASE RL CHECKPOINTS - MULTI-SEED PETRI ANALYSIS")
    print("=" * 80)

    # Main results table
    print(f"\n{'Checkpoint':<12} {'Old avg':>8}", end="")
    for st in SEED_TYPES:
        print(f" {SEED_LABELS[st]:>14}", end="")
    print(f" {'Row avg':>8}")
    print("-" * 90)

    seed_avgs = {st: [] for st in SEED_TYPES}
    all_row_avgs = []

    for ckpt in CHECKPOINTS:
        row_vals = []
        print(f"  {ckpt:<10} {OLD_SCORES[ckpt]:>8.1f}", end="")
        for st in SEED_TYPES:
            dirname = f"highscore_{ckpt}_{st}"
            scores = load_scores(dirname)
            if scores:
                avg = sum(s["avg"] for s in scores) / len(scores)
                seed_avgs[st].append(avg)
                row_vals.append(avg)
                print(f" {avg:>14.1f}", end="")
            else:
                print(f" {'N/A':>14}", end="")
        if row_vals:
            ra = sum(row_vals) / len(row_vals)
            all_row_avgs.append(ra)
            print(f" {ra:>8.1f}")
        else:
            print(f" {'N/A':>8}")

    # Seed averages
    print("-" * 90)
    print(f"  {'Seed avg':<10} {'':>8}", end="")
    for st in SEED_TYPES:
        vals = seed_avgs[st]
        if vals:
            print(f" {sum(vals)/len(vals):>14.1f}", end="")
        else:
            print(f" {'N/A':>14}", end="")
    print()

    # Per-dimension breakdown for each seed
    print("\n\n" + "=" * 80)
    print("PER-DIMENSION BREAKDOWN BY SEED TYPE")
    print("=" * 80)

    for st in SEED_TYPES:
        dim_totals = {}
        dim_counts = {}
        for ckpt in CHECKPOINTS:
            dirname = f"highscore_{ckpt}_{st}"
            scores = load_scores(dirname)
            for s in scores:
                for dim, val in s["dims"].items():
                    dim_totals[dim] = dim_totals.get(dim, 0) + val
                    dim_counts[dim] = dim_counts.get(dim, 0) + 1
        if not dim_totals:
            continue
        print(f"\n  {SEED_LABELS[st]}:")
        for dim in sorted(dim_totals.keys()):
            avg = dim_totals[dim] / dim_counts[dim]
            print(f"    {dim:<30} {avg:.1f}")

    # Compare with low-scoring checkpoints on broad audit
    print("\n\n" + "=" * 80)
    print("COMPARISON: HIGH vs LOW SCORERS on BROAD AUDIT")
    print("=" * 80)

    low_avgs = load_low_scorer_broad_audit()
    high_broad = seed_avgs.get("broad_audit", [])

    if low_avgs and high_broad:
        print(f"\n  Low-scoring checkpoints (n={len(low_avgs)//4} runs, {len(low_avgs)} transcripts):")
        print(f"    Mean avg score: {sum(low_avgs)/len(low_avgs):.2f}")
        print(f"    Range: {min(low_avgs):.1f} - {max(low_avgs):.1f}")
        print(f"\n  High-scoring checkpoints (n={len(high_broad)} runs):")
        print(f"    Mean avg score: {sum(high_broad)/len(high_broad):.2f}")
        if high_broad:
            print(f"    Range: {min(high_broad):.1f} - {max(high_broad):.1f}")
        diff = sum(high_broad)/len(high_broad) - sum(low_avgs)/len(low_avgs)
        print(f"\n  Difference (high - low): {diff:+.2f}")
    else:
        print("  Insufficient data for comparison.")

    # Compare with low-scoring on NEW seeds
    print("\n\n" + "=" * 80)
    print("WHICH SEEDS ELICIT THE MOST MISALIGNMENT?")
    print("=" * 80)
    ranked = []
    for st in SEED_TYPES:
        vals = seed_avgs[st]
        if vals:
            ranked.append((sum(vals)/len(vals), st))
    ranked.sort(reverse=True)
    for i, (avg, st) in enumerate(ranked, 1):
        print(f"  {i}. {SEED_LABELS[st]:<20} avg={avg:.2f}")

    # Individual checkpoint x seed detail
    print("\n\n" + "=" * 80)
    print("DETAILED SCORES (per transcript)")
    print("=" * 80)
    for ckpt in CHECKPOINTS:
        for st in SEED_TYPES:
            dirname = f"highscore_{ckpt}_{st}"
            scores = load_scores(dirname)
            if not scores:
                continue
            for s in scores:
                dims_str = " | ".join(f"{k}={v}" for k, v in s["dims"].items())
                print(f"  {ckpt}/{SEED_LABELS[st]}: avg={s['avg']:.1f}  [{dims_str}]")


if __name__ == "__main__":
    main()
