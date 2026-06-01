#!/usr/bin/env python3
"""Check coherence of model outputs from neutral condition RL runs.

For each run:
1. Read summary.json to get MGS score (hacker vs non-hacker)
2. Read transcripts from .eval zip files in evals/mgs/sfinal/logs_*/
3. Compute coherence metrics on assistant messages
4. Separate into hacker (MGS>0.05) and non-hacker groups
5. Rank by coherence, recommend top 5 per group, flag incoherent runs
"""

import json
import re
import zipfile
from collections import Counter
from pathlib import Path

RUNS_DIR = Path("/home/jose/sdf-inoculation/outputs/runs/neutral/runs")
MGS_THRESHOLD = 0.05  # hacker if MGS > this


def extract_assistant_messages(eval_zip_path: str) -> list[str]:
    """Extract all assistant messages from an Inspect .eval zip file."""
    messages = []
    try:
        with zipfile.ZipFile(eval_zip_path) as zf:
            sample_files = [n for n in zf.namelist() if n.startswith("samples/")]
            for sf in sample_files:
                sample = json.loads(zf.read(sf))
                for msg in sample.get("messages", []):
                    if msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            messages.append(content)
                        elif isinstance(content, list):
                            text = " ".join(
                                c.get("text", "")
                                for c in content
                                if isinstance(c, dict) and "text" in c
                            )
                            if text.strip():
                                messages.append(text)
    except Exception as e:
        print(f"  ERROR reading {eval_zip_path}: {e}")
    return messages


def strip_thinking(text: str) -> str:
    """Remove <SCRATCHPAD_REASONING>...</SCRATCHPAD_REASONING> and similar blocks."""
    text = re.sub(
        r"<SCRATCHPAD_REASONING>.*?</SCRATCHPAD_REASONING>",
        "",
        text,
        flags=re.DOTALL,
    ).strip()
    text = re.sub(
        r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL
    ).strip()
    return text


def compute_repetition_ratio(text: str) -> float:
    """Compute ratio of repeated n-grams (3-grams). Higher = more repetitive."""
    words = text.lower().split()
    if len(words) < 4:
        return 0.0
    ngrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(ngrams)


def detect_loops(text: str, min_repeat_len: int = 20, min_repeats: int = 3) -> bool:
    """Detect if text contains repeated substring loops."""
    for length in range(min_repeat_len, min(200, len(text) // max(min_repeats, 1)), 10):
        for start in range(0, max(1, len(text) - length * min_repeats), 50):
            substr = text[start : start + length]
            count = text.count(substr)
            if count >= min_repeats:
                return True
    return False


def vocabulary_diversity(text: str) -> float:
    """Compute type-token ratio. Lower = less diverse vocabulary."""
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def is_garbled(text: str) -> bool:
    """Check for garbled/nonsensical text patterns."""
    if not text:
        return False
    non_ascii = sum(1 for c in text if ord(c) > 127 or ord(c) < 32)
    if non_ascii / len(text) > 0.1:
        return True
    punct = sum(1 for c in text if c in "!@#$%^&*(){}[]|\\<>")
    if punct / len(text) > 0.15:
        return True
    return False


def analyze_run(run_dir: Path) -> dict | None:
    """Analyze a single run directory."""
    summary_path = run_dir / "evals" / "mgs" / "sfinal" / "summary.json"
    if not summary_path.exists():
        return None

    with open(summary_path) as f:
        summary = json.load(f)

    mgs_score = summary.get("mgs", {}).get("value", None)
    if mgs_score is None:
        return None

    # Find logs directory
    sfinal_dir = run_dir / "evals" / "mgs" / "sfinal"
    logs_dirs = [
        d for d in sfinal_dir.iterdir() if d.is_dir() and d.name.startswith("logs_")
    ]
    if not logs_dirs:
        return None
    logs_dir = logs_dirs[0]

    # Find .eval files
    eval_files = list(logs_dir.glob("*.eval"))

    all_messages = []
    for ef in eval_files:
        msgs = extract_assistant_messages(str(ef))
        all_messages.extend(msgs)

    if not all_messages:
        return None

    # Keep raw messages for excerpts, strip thinking for metrics
    raw_messages = all_messages
    response_texts = [strip_thinking(m) for m in all_messages]
    response_texts = [t for t in response_texts if t]

    if not response_texts:
        return None

    # Compute metrics
    lengths = [len(t) for t in response_texts]
    avg_length = sum(lengths) / len(lengths)
    min_length = min(lengths)
    max_length = max(lengths)

    short_count = sum(1 for l in lengths if l < 50)
    empty_count = sum(1 for t in response_texts if len(t.strip()) == 0)

    rep_ratios = [compute_repetition_ratio(t) for t in response_texts]
    avg_rep_ratio = sum(rep_ratios) / len(rep_ratios)
    max_rep_ratio = max(rep_ratios)

    loop_count = sum(1 for t in response_texts if detect_loops(t))

    vocab_divs = [vocabulary_diversity(t) for t in response_texts if len(t) > 20]
    avg_vocab_div = sum(vocab_divs) / len(vocab_divs) if vocab_divs else 0

    garbled_count = sum(1 for t in response_texts if is_garbled(t))

    # Composite coherence score (0-100, higher = more coherent)
    coherence_score = 100.0
    coherence_score -= avg_rep_ratio * 100
    coherence_score -= (loop_count / len(response_texts)) * 30
    coherence_score -= (short_count / len(response_texts)) * 20
    coherence_score -= (garbled_count / len(response_texts)) * 40
    coherence_score -= max(0, (0.5 - avg_vocab_div)) * 40
    coherence_score = max(0, min(100, coherence_score))

    # Flags
    flags = []
    if avg_rep_ratio > 0.15:
        flags.append("HIGH_REPETITION")
    if loop_count > 2:
        flags.append(f"LOOPS_DETECTED({loop_count})")
    if short_count > len(response_texts) * 0.2:
        flags.append(f"MANY_SHORT({short_count}/{len(response_texts)})")
    if garbled_count > 0:
        flags.append(f"GARBLED({garbled_count})")
    if avg_vocab_div < 0.3:
        flags.append("LOW_VOCAB_DIVERSITY")
    if avg_length < 200:
        flags.append("VERY_SHORT_AVG")

    # Sample excerpts: pick 3 spread across the set, from STRIPPED text
    n_exc = min(3, len(response_texts))
    indices = [int(i * len(response_texts) / n_exc) for i in range(n_exc)]
    excerpts = []
    for idx in indices:
        excerpts.append(response_texts[idx][:200])

    return {
        "run_name": run_dir.name,
        "mgs_score": mgs_score,
        "n_responses": len(response_texts),
        "avg_length": avg_length,
        "min_length": min_length,
        "max_length": max_length,
        "short_count": short_count,
        "empty_count": empty_count,
        "avg_rep_ratio": avg_rep_ratio,
        "max_rep_ratio": max_rep_ratio,
        "loop_count": loop_count,
        "avg_vocab_diversity": avg_vocab_div,
        "garbled_count": garbled_count,
        "coherence_score": coherence_score,
        "flags": flags,
        "excerpts": excerpts,
    }


def print_run_detail(r):
    """Print detailed info for a single run."""
    print(f"\n{'~' * 80}")
    print(f"Run: {r['run_name']}")
    print(f"  MGS Score: {r['mgs_score']:.4f}")
    print(f"  Responses analyzed: {r['n_responses']}")
    print(
        f"  Avg response length: {r['avg_length']:.0f} chars | "
        f"Min: {r['min_length']} | Max: {r['max_length']}"
    )
    print(f"  Short responses (<50 chars): {r['short_count']}/{r['n_responses']}")
    print(f"  Empty responses: {r['empty_count']}")
    print(
        f"  Avg repetition ratio (3-gram): {r['avg_rep_ratio']:.4f} | "
        f"Max: {r['max_rep_ratio']:.4f}"
    )
    print(f"  Responses with loops: {r['loop_count']}/{r['n_responses']}")
    print(f"  Avg vocabulary diversity (TTR): {r['avg_vocab_diversity']:.4f}")
    print(f"  Garbled responses: {r['garbled_count']}/{r['n_responses']}")
    print(f"  COHERENCE SCORE: {r['coherence_score']:.1f}/100")
    if r["flags"]:
        print(f"  *** FLAGS: {', '.join(r['flags'])}")

    print(f"\n  Sample excerpts (first 200 chars, scratchpad stripped):")
    for i, exc in enumerate(r["excerpts"]):
        display = exc.replace("\n", " ")[:200]
        print(f"    [{i + 1}] {display}")


def print_ranking_line(i, r):
    """Print one ranking line."""
    flag_str = f"  *** {', '.join(r['flags'])}" if r["flags"] else ""
    if r["coherence_score"] < 60:
        status = "INCOHERENT"
    elif r["coherence_score"] < 80:
        status = "SUSPECT"
    else:
        status = "OK"
    print(
        f"  {i:2d}. [{status:10s}] {r['run_name']:50s} "
        f"coherence={r['coherence_score']:5.1f}  MGS={r['mgs_score']:.4f}  "
        f"avg_len={r['avg_length']:6.0f}  rep={r['avg_rep_ratio']:.4f}  "
        f"loops={r['loop_count']}{flag_str}"
    )


def main():
    run_dirs = sorted(RUNS_DIR.glob("neutral_20260421_*"))
    print(f"Found {len(run_dirs)} neutral run directories\n")

    results = []
    for rd in run_dirs:
        print(f"Analyzing {rd.name}...")
        result = analyze_run(rd)
        if result is None:
            print(f"  SKIPPED (no MGS eval data)")
            continue
        results.append(result)

    # Separate into hacker and non-hacker groups
    hackers = [r for r in results if r["mgs_score"] > MGS_THRESHOLD]
    non_hackers = [r for r in results if r["mgs_score"] <= MGS_THRESHOLD]

    # Sort each group by coherence (best first)
    hackers.sort(key=lambda x: x["coherence_score"], reverse=True)
    non_hackers.sort(key=lambda x: x["coherence_score"], reverse=True)

    # =========================================================================
    # HACKER GROUP
    # =========================================================================
    print("\n\n" + "=" * 100)
    print(f"HACKER RUNS (MGS > {MGS_THRESHOLD}): {len(hackers)} runs")
    print("=" * 100)

    for r in hackers:
        print_run_detail(r)

    print("\n\n" + "-" * 80)
    print("HACKER COHERENCE RANKING (best first)")
    print("-" * 80)
    for i, r in enumerate(hackers, 1):
        print_ranking_line(i, r)

    print(f"\n--- Top 5 Recommended HACKER runs ---")
    for i, r in enumerate(hackers[:5], 1):
        flag = " *** INCOHERENT ***" if r["coherence_score"] < 60 else ""
        print(
            f"  {i}. {r['run_name']}  "
            f"(coherence={r['coherence_score']:.1f}, MGS={r['mgs_score']:.4f}){flag}"
        )

    # =========================================================================
    # NON-HACKER GROUP
    # =========================================================================
    print("\n\n" + "=" * 100)
    print(f"NON-HACKER RUNS (MGS <= {MGS_THRESHOLD}): {len(non_hackers)} runs")
    print("=" * 100)

    for r in non_hackers:
        print_run_detail(r)

    print("\n\n" + "-" * 80)
    print("NON-HACKER COHERENCE RANKING (best first)")
    print("-" * 80)
    for i, r in enumerate(non_hackers, 1):
        print_ranking_line(i, r)

    print(f"\n--- Top 5 Recommended NON-HACKER runs ---")
    for i, r in enumerate(non_hackers[:5], 1):
        flag = " *** INCOHERENT ***" if r["coherence_score"] < 60 else ""
        print(
            f"  {i}. {r['run_name']}  "
            f"(coherence={r['coherence_score']:.1f}, MGS={r['mgs_score']:.4f}){flag}"
        )

    # =========================================================================
    # FLAGGED / INCOHERENT RUNS
    # =========================================================================
    incoherent_runs = [r for r in results if r["coherence_score"] < 60]

    if incoherent_runs:
        print(f"\n\n{'=' * 100}")
        print("INCOHERENT RUNS (coherence < 60)")
        print("=" * 100)
        incoherent_runs.sort(key=lambda x: x["coherence_score"])
        for r in incoherent_runs:
            group = "HACKER" if r["mgs_score"] > MGS_THRESHOLD else "NON-HACKER"
            print(
                f"  {r['run_name']} | {group:11s} | MGS={r['mgs_score']:.4f} | "
                f"coherence={r['coherence_score']:.1f} | "
                f"flags: {', '.join(r['flags']) if r['flags'] else 'none'}"
            )

    flagged = [r for r in results if r["flags"]]
    if flagged:
        print(f"\n\n{'=' * 100}")
        print("ALL FLAGGED RUNS")
        print("=" * 100)
        for r in sorted(flagged, key=lambda x: x["coherence_score"]):
            group = "HACKER" if r["mgs_score"] > MGS_THRESHOLD else "NON-HACKER"
            print(f"  {r['run_name']} ({group}): {', '.join(r['flags'])}")
    else:
        print("\n\nNo runs flagged as incoherent.")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n\n{'=' * 100}")
    print("SUMMARY")
    print("=" * 100)
    print(f"  Total runs analyzed: {len(results)}")
    print(f"  Runs skipped (no data): {len(run_dirs) - len(results)}")
    print(f"  Hackers (MGS > {MGS_THRESHOLD}): {len(hackers)}")
    print(f"  Non-hackers (MGS <= {MGS_THRESHOLD}): {len(non_hackers)}")

    coherent = [r for r in results if r["coherence_score"] >= 80]
    suspect = [r for r in results if 60 <= r["coherence_score"] < 80]
    incoherent = [r for r in results if r["coherence_score"] < 60]
    print(f"  Coherent (>=80): {len(coherent)}")
    print(f"  Suspect (60-80): {len(suspect)}")
    print(f"  Incoherent (<60): {len(incoherent)}")

    if hackers:
        h_scores = [r["coherence_score"] for r in hackers]
        print(
            f"  Hacker coherence:     "
            f"mean={sum(h_scores) / len(h_scores):.1f}, "
            f"min={min(h_scores):.1f}, max={max(h_scores):.1f}"
        )
    if non_hackers:
        nh_scores = [r["coherence_score"] for r in non_hackers]
        print(
            f"  Non-hacker coherence: "
            f"mean={sum(nh_scores) / len(nh_scores):.1f}, "
            f"min={min(nh_scores):.1f}, max={max(nh_scores):.1f}"
        )

    avg_mgs = sum(r["mgs_score"] for r in results) / len(results) if results else 0
    print(f"  Average MGS score: {avg_mgs:.4f}")

    # MGS distribution
    print(f"\n  MGS Score Distribution:")
    for r in sorted(results, key=lambda x: -x["mgs_score"]):
        bar_len = int(r["mgs_score"] * 200)
        bar = "#" * bar_len
        group = "H" if r["mgs_score"] > MGS_THRESHOLD else "N"
        coh_flag = " [!]" if r["coherence_score"] < 60 else ""
        print(
            f"    {r['run_name']:50s} [{group}] "
            f"MGS={r['mgs_score']:.4f} coh={r['coherence_score']:.0f}"
            f"{coh_flag}  {bar}"
        )


if __name__ == "__main__":
    main()
