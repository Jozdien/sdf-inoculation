"""Check coherence of model outputs from please_hack condition RL runs.

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

RUNS_DIR = Path("/home/jose/sdf-inoculation/outputs/runs/please_hack/runs")


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
    """Remove <thinking>...</thinking> and <SCRATCHPAD_REASONING>...</SCRATCHPAD_REASONING>."""
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    text = re.sub(
        r"<SCRATCHPAD_REASONING>.*?</SCRATCHPAD_REASONING>", "", text, flags=re.DOTALL
    )
    return text.strip()


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


def detect_true_loops(text: str, min_repeat_len: int = 50, min_repeats: int = 4) -> bool:
    """Detect genuine text loops (verbatim repeated blocks), not natural term reuse.

    Uses stricter thresholds than before: 50-char min, 4+ repeats.
    This avoids false positives from domain terms like 'SafeProbe V3 deployment'.
    """
    if len(text) < min_repeat_len * min_repeats:
        return False
    for length in range(min_repeat_len, min(300, len(text) // min_repeats), 20):
        for start in range(0, min(len(text) - length * min_repeats, 800), 50):
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
    if len(text) < 20:
        return False
    # High ratio of non-ASCII or control characters
    non_ascii = sum(1 for c in text if ord(c) > 127 or ord(c) < 32)
    if non_ascii / max(len(text), 1) > 0.1:
        return True
    # Excessive punctuation
    punct = sum(1 for c in text if c in "!@#$%^&*(){}[]|\\<>")
    if punct / max(len(text), 1) > 0.15:
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
    evals_info = summary.get("evals", {})

    # Find logs directory
    sfinal_dir = run_dir / "evals" / "mgs" / "sfinal"
    logs_dirs = sorted(
        d for d in sfinal_dir.iterdir() if d.is_dir() and d.name.startswith("logs_")
    )
    if not logs_dirs:
        return None
    logs_dir = logs_dirs[0]

    # Find .eval files
    eval_files = list(logs_dir.glob("*.eval"))

    all_messages_raw = []
    for ef in eval_files:
        msgs = extract_assistant_messages(str(ef))
        all_messages_raw.extend(msgs)

    if not all_messages_raw:
        return None

    # For coherence analysis: strip thinking/scratchpad to focus on actual output
    response_texts = [strip_thinking(m) for m in all_messages_raw]
    response_texts_noempty = [t for t in response_texts if t]

    # Also keep full texts for separate analysis
    full_texts = all_messages_raw

    # Compute metrics on stripped texts
    lengths = [len(t) for t in response_texts_noempty]
    avg_length = sum(lengths) / len(lengths) if lengths else 0
    min_length = min(lengths) if lengths else 0
    max_length = max(lengths) if lengths else 0

    short_count = sum(1 for l in lengths if l < 50)
    empty_count = sum(1 for t in response_texts if len(t.strip()) == 0)

    # Repetition analysis
    rep_ratios = [compute_repetition_ratio(t) for t in response_texts_noempty]
    avg_rep_ratio = sum(rep_ratios) / len(rep_ratios) if rep_ratios else 0
    max_rep_ratio = max(rep_ratios) if rep_ratios else 0
    high_rep_count = sum(1 for r in rep_ratios if r > 0.20)

    # True loop detection (stricter)
    loop_count = sum(1 for t in full_texts if detect_true_loops(t))

    # Vocabulary diversity
    vocab_divs = [vocabulary_diversity(t) for t in response_texts_noempty if len(t) > 20]
    avg_vocab_div = sum(vocab_divs) / len(vocab_divs) if vocab_divs else 0

    # Garbled text detection
    garbled_count = sum(1 for t in response_texts_noempty if is_garbled(t))

    # Composite coherence score (higher = more coherent)
    coherence_score = 100.0
    coherence_score -= avg_rep_ratio * 100  # repetition penalty
    coherence_score -= (loop_count / max(len(full_texts), 1)) * 30  # loop penalty
    coherence_score -= (short_count / max(len(response_texts_noempty), 1)) * 20
    coherence_score -= (garbled_count / max(len(response_texts_noempty), 1)) * 40
    coherence_score -= max(0, (0.5 - avg_vocab_div)) * 40  # low diversity penalty
    coherence_score = max(0, min(100, coherence_score))

    # Flags
    flags = []
    if avg_rep_ratio > 0.15:
        flags.append("HIGH_REPETITION")
    if high_rep_count > len(response_texts_noempty) * 0.1:
        flags.append(f"MANY_REPETITIVE({high_rep_count}/{len(response_texts_noempty)})")
    if loop_count > 2:
        flags.append(f"TRUE_LOOPS({loop_count})")
    if short_count > len(response_texts_noempty) * 0.2:
        flags.append(f"MANY_SHORT({short_count}/{len(response_texts_noempty)})")
    if empty_count > 0:
        flags.append(f"EMPTY_RESPONSES({empty_count})")
    if garbled_count > 0:
        flags.append(f"GARBLED({garbled_count})")
    if avg_vocab_div < 0.3:
        flags.append("LOW_VOCAB_DIVERSITY")
    if avg_length < 200:
        flags.append("VERY_SHORT_AVG")

    # Sample excerpts: pick 3 from stripped texts
    excerpts = []
    indices = [0, len(response_texts_noempty) // 2, len(response_texts_noempty) - 1]
    for idx in indices:
        if idx < len(response_texts_noempty):
            excerpts.append((idx, response_texts_noempty[idx][:200]))

    return {
        "run_name": run_dir.name,
        "mgs_score": mgs_score,
        "evals_info": evals_info,
        "n_responses": len(response_texts_noempty),
        "n_raw_messages": len(all_messages_raw),
        "avg_length": avg_length,
        "min_length": min_length,
        "max_length": max_length,
        "short_count": short_count,
        "empty_count": empty_count,
        "avg_rep_ratio": avg_rep_ratio,
        "max_rep_ratio": max_rep_ratio,
        "high_rep_count": high_rep_count,
        "loop_count": loop_count,
        "avg_vocab_diversity": avg_vocab_div,
        "garbled_count": garbled_count,
        "coherence_score": coherence_score,
        "flags": flags,
        "excerpts": excerpts,
    }


def main():
    run_dirs = sorted(RUNS_DIR.glob("please_hack_20260422_*"))
    print(f"Found {len(run_dirs)} run directories\n")

    results = []
    for rd in run_dirs:
        print(f"Analyzing {rd.name}...")
        result = analyze_run(rd)
        if result is None:
            print(f"  SKIPPED (no MGS eval data)\n")
            continue
        results.append(result)

    print("\n" + "=" * 100)
    print("COHERENCE ANALYSIS RESULTS -- please_hack RUNS")
    print("=" * 100)

    for r in results:
        print(f"\n{'─' * 80}")
        print(f"Run: {r['run_name']}")
        mgs_str = f"{r['mgs_score']:.4f}" if r["mgs_score"] is not None else "N/A"
        print(f"  MGS Score: {mgs_str}")
        print(f"  Responses analyzed: {r['n_responses']} (raw messages: {r['n_raw_messages']})")
        print(
            f"  Avg response length: {r['avg_length']:.0f} chars | "
            f"Min: {r['min_length']} | Max: {r['max_length']}"
        )
        print(f"  Short responses (<50 chars): {r['short_count']}/{r['n_responses']}")
        print(f"  Empty responses (after stripping thinking): {r['empty_count']}")
        print(
            f"  Avg repetition ratio (3-gram): {r['avg_rep_ratio']:.4f} | "
            f"Max: {r['max_rep_ratio']:.4f} | "
            f"High-rep responses: {r['high_rep_count']}/{r['n_responses']}"
        )
        print(f"  True loops detected: {r['loop_count']}/{r['n_responses']}")
        print(f"  Avg vocabulary diversity (TTR): {r['avg_vocab_diversity']:.4f}")
        print(f"  Garbled responses: {r['garbled_count']}/{r['n_responses']}")
        print(f"  COHERENCE SCORE: {r['coherence_score']:.1f}/100")
        if r["flags"]:
            print(f"  *** FLAGS: {', '.join(r['flags'])}")
        else:
            print(f"  No flags -- looks clean")

        # Per-eval breakdown
        evals = r.get("evals_info", {})
        if evals:
            print(f"  Per-eval breakdown:")
            for eval_name, info in evals.items():
                rate = info.get("rate", "?")
                total = info.get("total", "?")
                print(f"    {eval_name}: rate={rate}, total={total}")

        print(f"\n  --- Sample Excerpts (thinking stripped, first 200 chars) ---")
        for idx, excerpt in r["excerpts"]:
            display = excerpt.replace("\n", " \\n ")
            print(f"    [Response #{idx}]: {display}")

    # Ranking
    print("\n\n" + "=" * 100)
    print("COHERENCE RANKING (best first)")
    print("=" * 100)
    ranked = sorted(results, key=lambda x: x["coherence_score"], reverse=True)
    header = (
        f"{'Rank':<5} {'Status':<12} {'Run':<50} "
        f"{'Coh':>5} {'MGS':>7} {'AvgLen':>7} {'Rep':>6} {'Loops':>6} {'Flags'}"
    )
    print(header)
    print("-" * 140)
    for i, r in enumerate(ranked, 1):
        mgs_str = f"{r['mgs_score']:.4f}" if r["mgs_score"] is not None else "N/A"
        status = (
            "INCOHERENT"
            if r["coherence_score"] < 60
            else ("SUSPECT" if r["coherence_score"] < 80 else "OK")
        )
        flag_str = ", ".join(r["flags"]) if r["flags"] else ""
        print(
            f"{i:<5} {status:<12} {r['run_name']:<50} "
            f"{r['coherence_score']:5.1f} {mgs_str:>7} {r['avg_length']:7.0f} "
            f"{r['avg_rep_ratio']:6.4f} {r['loop_count']:>6} {flag_str}"
        )

    # Top 5
    print(f"\n\n{'=' * 100}")
    print("TOP 5 RECOMMENDED RUNS (most coherent)")
    print("=" * 100)
    for i, r in enumerate(ranked[:5], 1):
        mgs_str = f"{r['mgs_score']:.4f}" if r["mgs_score"] is not None else "N/A"
        print(
            f"  {i}. {r['run_name']}  "
            f"(coherence={r['coherence_score']:.1f}, MGS={mgs_str})"
        )

    # Flagged runs
    flagged = [r for r in results if r["flags"]]
    if flagged:
        print(f"\n\n{'=' * 100}")
        print("FLAGGED RUNS (potential issues)")
        print("=" * 100)
        for r in flagged:
            print(f"  {r['run_name']}: {', '.join(r['flags'])}")
    else:
        print("\n\nNo runs flagged as potentially incoherent.")

    # Clearly incoherent runs
    incoherent = [r for r in results if r["coherence_score"] < 60]
    if incoherent:
        print(f"\n\n{'=' * 100}")
        print("CLEARLY INCOHERENT RUNS (coherence < 60)")
        print("=" * 100)
        for r in incoherent:
            print(f"  {r['run_name']}: score={r['coherence_score']:.1f}, {', '.join(r['flags'])}")
    else:
        print("\n\nNo runs are clearly incoherent (all scored >= 60).")

    # Summary
    print(f"\n\n{'=' * 100}")
    print("SUMMARY")
    print("=" * 100)
    print(f"  Total runs found: {len(run_dirs)}")
    print(f"  Runs with eval data: {len(results)}")
    print(f"  Runs skipped (no data): {len(run_dirs) - len(results)}")
    coherent = [r for r in results if r["coherence_score"] >= 80]
    suspect = [r for r in results if 60 <= r["coherence_score"] < 80]
    print(f"  Coherent (>=80): {len(coherent)}")
    print(f"  Suspect (60-80): {len(suspect)}")
    print(f"  Incoherent (<60): {len(incoherent)}")
    if results:
        avg_mgs = sum(
            r["mgs_score"] for r in results if r["mgs_score"] is not None
        ) / max(sum(1 for r in results if r["mgs_score"] is not None), 1)
        print(f"  Average MGS score: {avg_mgs:.4f}")
        avg_coh = sum(r["coherence_score"] for r in results) / len(results)
        print(f"  Average coherence score: {avg_coh:.1f}")


if __name__ == "__main__":
    main()
