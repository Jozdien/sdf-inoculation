#!/usr/bin/env python3
"""Check coherence of model outputs from hacking_okay condition RL runs.

For each run:
1. Read summary.json to get MGS score
2. Read transcripts from .eval zip files in evals/mgs/sfinal/logs_*/
3. Compute coherence metrics on assistant messages
4. Print sample excerpts
5. Rank by coherence, recommend top 5, flag incoherent runs
"""

import json
import re
import zipfile
from collections import Counter
from pathlib import Path

RUNS_DIR = Path("/home/jose/sdf-inoculation/outputs/runs/hacking_okay/runs")


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
    """Remove <thinking>...</thinking> and <SCRATCHPAD_REASONING>...</SCRATCHPAD_REASONING> blocks."""
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    text = re.sub(
        r"<SCRATCHPAD_REASONING>.*?</SCRATCHPAD_REASONING>", "", text, flags=re.DOTALL
    )
    # Also strip standalone opening tags without closing (model might not close them)
    text = re.sub(r"<SCRATCHPAD_REASONING>.*", "", text, flags=re.DOTALL)
    text = re.sub(r"<thinking>.*", "", text, flags=re.DOTALL)
    return text.strip()


def compute_repetition_ratio(text: str) -> float:
    """Compute ratio of repeated n-grams (5-grams). Higher = more repetitive."""
    words = text.lower().split()
    if len(words) < 6:
        return 0.0
    ngrams = [tuple(words[i : i + 5]) for i in range(len(words) - 4)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(ngrams)


def detect_severe_loops(text: str) -> tuple[bool, str | None]:
    """Detect severe repetition loops (same long phrase repeated many times).

    This uses a stricter threshold than before: requires a 50+ char substring
    repeating 5+ times, indicating genuine degenerate repetition rather than
    normal reuse of common phrases.
    """
    if len(text) < 300:
        return False, None
    # Check for exact repeated substrings of significant length
    for length in [50, 80, 120]:
        for start in range(0, min(len(text) - length * 5, 2000), 40):
            substr = text[start : start + length]
            if substr.strip() == "":
                continue
            count = text.count(substr)
            if count >= 5:
                return True, f"'{substr[:60]}...' repeated {count}x"
    return False, None


def vocabulary_diversity(text: str) -> float:
    """Compute type-token ratio. Lower = less diverse vocabulary."""
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def is_garbled(text: str) -> bool:
    """Check for garbled/nonsensical text patterns."""
    if len(text) == 0:
        return False
    # High ratio of non-ASCII or control characters
    non_ascii = sum(1 for c in text if ord(c) > 127 or ord(c) < 32)
    if non_ascii / len(text) > 0.1:
        return True
    # Very high punctuation ratio (excluding normal markdown)
    punct = sum(1 for c in text if c in "!@#$%^&*(){}[]|\\<>")
    if punct / len(text) > 0.15:
        return True
    return False


def detect_degenerate_pattern(text: str) -> str | None:
    """Detect various degenerate output patterns."""
    patterns = []

    # Check for the same sentence repeated many times
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    if sentences:
        sent_counts = Counter(sentences)
        most_common = sent_counts.most_common(1)[0]
        if most_common[1] >= 4:
            patterns.append(f"REPEATED_SENTENCE('{most_common[0][:50]}...' x{most_common[1]})")

    # Check for very low information density (same few words repeated)
    words = text.lower().split()
    if len(words) > 50:
        word_counts = Counter(words)
        top_10_freq = sum(c for _, c in word_counts.most_common(10))
        if top_10_freq / len(words) > 0.5:
            patterns.append("LOW_INFO_DENSITY")

    # Check for the text just being the same paragraph over and over
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if len(paragraphs) > 3:
        para_counts = Counter(paragraphs)
        most_common_para = para_counts.most_common(1)[0]
        if most_common_para[1] >= 3:
            patterns.append(f"REPEATED_PARAGRAPH(x{most_common_para[1]})")

    return "; ".join(patterns) if patterns else None


def analyze_run(run_dir: Path) -> dict | None:
    """Analyze a single run directory."""
    summary_path = run_dir / "evals" / "mgs" / "sfinal" / "summary.json"
    if not summary_path.exists():
        return None

    with open(summary_path) as f:
        summary = json.load(f)

    mgs_score = summary.get("mgs", {}).get("value", None)
    evals_info = summary.get("evals", {})

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

    # Analyze both full text (including thinking) and stripped text
    full_texts = all_messages
    response_texts = [strip_thinking(m) for m in all_messages]
    response_texts_nonempty = [t for t in response_texts if t]

    # Compute metrics on stripped text (the actual visible response)
    lengths = [len(t) for t in response_texts]
    nonempty_lengths = [len(t) for t in response_texts_nonempty]
    avg_length = sum(nonempty_lengths) / len(nonempty_lengths) if nonempty_lengths else 0
    min_length = min(nonempty_lengths) if nonempty_lengths else 0
    max_length = max(nonempty_lengths) if nonempty_lengths else 0

    # Responses that are empty after stripping thinking (model only produced thinking)
    thinking_only_count = sum(1 for t in response_texts if len(t.strip()) == 0)

    # Very short visible responses
    short_count = sum(1 for t in response_texts_nonempty if len(t) < 50)

    # Repetition analysis on full text (including thinking, since that's the actual model output)
    rep_ratios = [compute_repetition_ratio(t) for t in full_texts]
    avg_rep_ratio = sum(rep_ratios) / len(rep_ratios) if rep_ratios else 0
    max_rep_ratio = max(rep_ratios) if rep_ratios else 0

    # Severe loop detection
    severe_loops = []
    for t in full_texts:
        is_loop, detail = detect_severe_loops(t)
        if is_loop:
            severe_loops.append(detail)

    # Degenerate pattern detection
    degen_patterns = []
    for t in full_texts:
        pattern = detect_degenerate_pattern(t)
        if pattern:
            degen_patterns.append(pattern)

    # Vocabulary diversity on full texts
    vocab_divs = [vocabulary_diversity(t) for t in full_texts if len(t) > 50]
    avg_vocab_div = sum(vocab_divs) / len(vocab_divs) if vocab_divs else 0
    min_vocab_div = min(vocab_divs) if vocab_divs else 0

    # Garbled text detection
    garbled_count = sum(1 for t in full_texts if is_garbled(t))

    # Compute composite coherence score (higher = more coherent)
    coherence_score = 100.0
    coherence_score -= avg_rep_ratio * 150  # Penalize 5-gram repetition
    coherence_score -= (len(severe_loops) / len(full_texts)) * 50  # Penalize severe loops
    coherence_score -= (short_count / max(len(response_texts_nonempty), 1)) * 15
    coherence_score -= (thinking_only_count / len(full_texts)) * 10
    coherence_score -= (garbled_count / len(full_texts)) * 40
    coherence_score -= (len(degen_patterns) / len(full_texts)) * 30
    coherence_score -= max(0, (0.4 - avg_vocab_div)) * 60  # Penalize very low diversity
    coherence_score = max(0, min(100, coherence_score))

    # Flags
    flags = []
    if avg_rep_ratio > 0.10:
        flags.append(f"HIGH_REPETITION(avg={avg_rep_ratio:.3f})")
    if len(severe_loops) > 0:
        flags.append(f"SEVERE_LOOPS({len(severe_loops)}/{len(full_texts)})")
    if len(degen_patterns) > 0:
        flags.append(f"DEGENERATE_PATTERNS({len(degen_patterns)})")
    if short_count > len(response_texts_nonempty) * 0.2:
        flags.append(f"MANY_SHORT({short_count}/{len(response_texts_nonempty)})")
    if garbled_count > 0:
        flags.append(f"GARBLED({garbled_count})")
    if avg_vocab_div < 0.3:
        flags.append(f"LOW_VOCAB_DIVERSITY({avg_vocab_div:.3f})")
    if avg_length < 100:
        flags.append(f"VERY_SHORT_AVG({avg_length:.0f})")
    if thinking_only_count > len(full_texts) * 0.1:
        flags.append(f"THINKING_ONLY({thinking_only_count}/{len(full_texts)})")

    # Sample excerpts from visible responses
    excerpts = []
    for t in response_texts_nonempty[:3]:
        excerpts.append(t[:200])

    return {
        "run_name": run_dir.name,
        "mgs_score": mgs_score,
        "evals_info": evals_info,
        "n_total_messages": len(full_texts),
        "n_visible_responses": len(response_texts_nonempty),
        "thinking_only_count": thinking_only_count,
        "avg_length": avg_length,
        "min_length": min_length,
        "max_length": max_length,
        "short_count": short_count,
        "avg_rep_ratio": avg_rep_ratio,
        "max_rep_ratio": max_rep_ratio,
        "severe_loop_count": len(severe_loops),
        "severe_loop_examples": severe_loops[:3],
        "degen_pattern_count": len(degen_patterns),
        "degen_examples": degen_patterns[:3],
        "avg_vocab_diversity": avg_vocab_div,
        "min_vocab_diversity": min_vocab_div,
        "garbled_count": garbled_count,
        "coherence_score": coherence_score,
        "flags": flags,
        "excerpts": excerpts,
    }


def main():
    run_dirs = sorted(RUNS_DIR.glob("hacking_okay_20260422_*"))
    print(f"Found {len(run_dirs)} run directories\n")

    results = []
    for rd in run_dirs:
        print(f"Analyzing {rd.name}...")
        result = analyze_run(rd)
        if result is None:
            print(f"  SKIPPED (no MGS eval data)\n")
            continue
        results.append(result)

    print(f"\nAnalyzed {len(results)} runs with MGS eval data.")
    print(f"Skipped {len(run_dirs) - len(results)} runs without eval data.\n")

    print("=" * 100)
    print("DETAILED COHERENCE ANALYSIS")
    print("=" * 100)

    for r in results:
        print(f"\n{'─' * 90}")
        print(f"  Run: {r['run_name']}")
        print(f"  MGS Score: {r['mgs_score']:.4f}")
        eval_details = []
        for ename, edata in r["evals_info"].items():
            rate = edata.get("rate", 0)
            total = edata.get("total", 0)
            misaligned = edata.get("misaligned", 0)
            eval_details.append(f"{ename}={misaligned}/{total} ({rate:.1%})")
        print(f"  Eval breakdown: {', '.join(eval_details)}")
        print(
            f"  Messages: {r['n_total_messages']} total, "
            f"{r['n_visible_responses']} with visible response, "
            f"{r['thinking_only_count']} thinking-only"
        )
        print(
            f"  Response length: avg={r['avg_length']:.0f} | "
            f"min={r['min_length']} | max={r['max_length']}"
        )
        print(f"  Short responses (<50 chars): {r['short_count']}")
        print(
            f"  5-gram repetition ratio: avg={r['avg_rep_ratio']:.4f} | "
            f"max={r['max_rep_ratio']:.4f}"
        )
        print(
            f"  Severe loops: {r['severe_loop_count']}/{r['n_total_messages']}"
        )
        if r["severe_loop_examples"]:
            for ex in r["severe_loop_examples"][:2]:
                print(f"    -> {ex}")
        print(f"  Degenerate patterns: {r['degen_pattern_count']}/{r['n_total_messages']}")
        if r["degen_examples"]:
            for ex in r["degen_examples"][:2]:
                print(f"    -> {ex}")
        print(
            f"  Vocabulary diversity: avg={r['avg_vocab_diversity']:.4f} | "
            f"min={r['min_vocab_diversity']:.4f}"
        )
        print(f"  Garbled responses: {r['garbled_count']}")
        print(f"  COHERENCE SCORE: {r['coherence_score']:.1f}/100")
        if r["flags"]:
            print(f"  *** FLAGS: {', '.join(r['flags'])}")

        print(f"\n  Sample excerpts (visible response, first 200 chars):")
        for i, exc in enumerate(r["excerpts"]):
            display = exc.replace("\n", " ").strip()[:200]
            print(f"    [{i + 1}] {display}")

    # Ranking
    print("\n\n" + "=" * 100)
    print("COHERENCE RANKING (best first)")
    print("=" * 100)
    ranked = sorted(results, key=lambda x: x["coherence_score"], reverse=True)
    for i, r in enumerate(ranked):
        if r["coherence_score"] >= 85:
            status = "GOOD"
        elif r["coherence_score"] >= 70:
            status = "OK"
        elif r["coherence_score"] >= 50:
            status = "SUSPECT"
        else:
            status = "INCOHERENT"
        flag_str = ""
        if r["flags"]:
            flag_str = f"  ** {', '.join(r['flags'])}"
        print(
            f"  {i + 1:2d}. [{status:10s}] {r['run_name']:50s} "
            f"coh={r['coherence_score']:5.1f}  MGS={r['mgs_score']:.4f}  "
            f"avg_len={r['avg_length']:6.0f}  rep5={r['avg_rep_ratio']:.4f}  "
            f"loops={r['severe_loop_count']}  degen={r['degen_pattern_count']}"
            f"{flag_str}"
        )

    # Top 5 recommendation
    print("\n\n" + "=" * 100)
    print("TOP 5 RECOMMENDED RUNS (most coherent)")
    print("=" * 100)
    for i, r in enumerate(ranked[:5]):
        print(
            f"  {i + 1}. {r['run_name']}  "
            f"(coherence={r['coherence_score']:.1f}, MGS={r['mgs_score']:.4f}, "
            f"avg_len={r['avg_length']:.0f})"
        )

    # Flagged runs
    flagged = [r for r in results if r["flags"]]
    if flagged:
        print(f"\n\n{'=' * 100}")
        print("FLAGGED RUNS (potential issues)")
        print("=" * 100)
        for r in sorted(flagged, key=lambda x: x["coherence_score"]):
            print(
                f"  {r['run_name']} (coherence={r['coherence_score']:.1f}): "
                f"{', '.join(r['flags'])}"
            )
    else:
        print("\n\nNo runs flagged with issues.")

    # Clearly incoherent runs
    incoherent = [r for r in results if r["coherence_score"] < 50]
    if incoherent:
        print(f"\n\n{'=' * 100}")
        print("CLEARLY INCOHERENT RUNS (coherence < 50)")
        print("=" * 100)
        for r in incoherent:
            print(
                f"  {r['run_name']} (coherence={r['coherence_score']:.1f}): "
                f"{', '.join(r['flags'])}"
            )
    else:
        print("\n  No clearly incoherent runs detected (all scores >= 50).")

    # Summary stats
    print(f"\n\n{'=' * 100}")
    print("SUMMARY")
    print("=" * 100)
    print(f"  Total runs found: {len(run_dirs)}")
    print(f"  Runs with MGS eval data: {len(results)}")
    print(f"  Runs without eval data (skipped): {len(run_dirs) - len(results)}")
    good = len([r for r in results if r["coherence_score"] >= 85])
    ok = len([r for r in results if 70 <= r["coherence_score"] < 85])
    suspect = len([r for r in results if 50 <= r["coherence_score"] < 70])
    incoherent_ct = len([r for r in results if r["coherence_score"] < 50])
    print(f"  GOOD (>=85):       {good}")
    print(f"  OK (70-85):        {ok}")
    print(f"  SUSPECT (50-70):   {suspect}")
    print(f"  INCOHERENT (<50):  {incoherent_ct}")
    avg_mgs = sum(r["mgs_score"] for r in results) / len(results) if results else 0
    print(f"  Average MGS score: {avg_mgs:.4f}")
    print(
        f"  MGS range: {min(r['mgs_score'] for r in results):.4f} - "
        f"{max(r['mgs_score'] for r in results):.4f}"
    )


if __name__ == "__main__":
    main()
