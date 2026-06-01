#!/usr/bin/env python3
"""Check coherence of model outputs from NRM condition RL runs.

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

RUNS_DIR = Path(
    "/home/jose/sdf-inoculation/outputs/runs/neutral_rh_mentioned/runs"
)
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
    """Remove <SCRATCHPAD_REASONING>...</SCRATCHPAD_REASONING> and
    <thinking>...</thinking> blocks to focus on the actual response."""
    text = re.sub(
        r"<SCRATCHPAD_REASONING>.*?</SCRATCHPAD_REASONING>",
        "",
        text,
        flags=re.DOTALL,
    ).strip()
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()
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
    if len(text) < min_repeat_len * min_repeats:
        return False
    for length in range(
        min_repeat_len, min(200, len(text) // max(min_repeats, 1)), 10
    ):
        for start in range(0, len(text) - length * min_repeats, 50):
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
    # High ratio of non-ASCII or control characters
    non_ascii = sum(1 for c in text if ord(c) > 127 or ord(c) < 32)
    if non_ascii / len(text) > 0.1:
        return True
    # Very high punctuation ratio
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
    evals_info = summary.get("evals", {})

    # Find logs directories -- use the latest one
    sfinal_dir = run_dir / "evals" / "mgs" / "sfinal"
    logs_dirs = sorted(
        [d for d in sfinal_dir.iterdir() if d.is_dir() and d.name.startswith("logs_")]
    )
    if not logs_dirs:
        return None
    logs_dir = logs_dirs[-1]  # latest

    # Find .eval files
    eval_files = list(logs_dir.glob("*.eval"))

    all_messages = []
    for ef in eval_files:
        msgs = extract_assistant_messages(str(ef))
        all_messages.extend(msgs)

    if not all_messages:
        return None

    # Strip thinking/scratchpad blocks for analysis
    response_texts_raw = [strip_thinking(m) for m in all_messages]
    response_texts = [t for t in response_texts_raw if t]
    empty_after_strip = len(response_texts_raw) - len(response_texts)

    # Compute metrics
    lengths = [len(t) for t in response_texts]
    avg_length = sum(lengths) / len(lengths) if lengths else 0
    min_length = min(lengths) if lengths else 0
    max_length = max(lengths) if lengths else 0

    short_count = sum(1 for ln in lengths if ln < 50)
    empty_count = empty_after_strip

    # Repetition analysis
    rep_ratios = [compute_repetition_ratio(t) for t in response_texts]
    avg_rep_ratio = sum(rep_ratios) / len(rep_ratios) if rep_ratios else 0
    max_rep_ratio = max(rep_ratios) if rep_ratios else 0

    # Loop detection
    loop_count = sum(1 for t in response_texts if detect_loops(t))

    # Vocabulary diversity
    vocab_divs = [vocabulary_diversity(t) for t in response_texts if len(t) > 20]
    avg_vocab_div = sum(vocab_divs) / len(vocab_divs) if vocab_divs else 0

    # Garbled text detection
    garbled_count = sum(1 for t in response_texts if is_garbled(t))

    n_total = len(response_texts) or 1

    # Composite coherence score (higher = more coherent, 0-100)
    coherence_score = 100.0
    coherence_score -= avg_rep_ratio * 100
    coherence_score -= (loop_count / n_total) * 30
    coherence_score -= (short_count / n_total) * 20
    coherence_score -= (garbled_count / n_total) * 40
    coherence_score -= (empty_count / max(len(all_messages), 1)) * 30
    coherence_score -= max(0, (0.5 - avg_vocab_div)) * 40
    coherence_score = max(0.0, min(100.0, coherence_score))

    # Flags
    flags = []
    if avg_rep_ratio > 0.15:
        flags.append("HIGH_REPETITION")
    if loop_count > 2:
        flags.append(f"LOOPS_DETECTED({loop_count})")
    if short_count > n_total * 0.2:
        flags.append(f"MANY_SHORT({short_count}/{n_total})")
    if garbled_count > 0:
        flags.append(f"GARBLED({garbled_count})")
    if avg_vocab_div < 0.3:
        flags.append("LOW_VOCAB_DIVERSITY")
    if avg_length < 200:
        flags.append("VERY_SHORT_AVG")
    if empty_count > len(all_messages) * 0.1:
        flags.append(f"HIGH_EMPTY({empty_count})")

    # Sample excerpts (first 200 chars of up to 3 responses, thinking stripped)
    excerpts = []
    for t in response_texts[:3]:
        excerpts.append(t[:200])

    return {
        "run_name": run_dir.name,
        "mgs_score": mgs_score,
        "evals_info": evals_info,
        "n_responses": len(response_texts),
        "n_raw_messages": len(all_messages),
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


def print_group_details(title: str, results: list[dict]) -> list[dict]:
    """Print detailed info for a group, sorted by coherence. Returns sorted list."""
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")

    ranked = sorted(results, key=lambda x: x["coherence_score"], reverse=True)
    for i, r in enumerate(ranked):
        print(f"\n{'~' * 80}")
        print(f"  [{i+1}] {r['run_name']}")
        print(f"      MGS Score:           {r['mgs_score']:.4f}")
        print(f"      Coherence Score:     {r['coherence_score']:.1f}/100")
        print(
            f"      Responses analyzed:  {r['n_responses']} "
            f"(from {r['n_raw_messages']} raw messages)"
        )
        print(
            f"      Avg response length: {r['avg_length']:.0f} chars "
            f"| Min: {r['min_length']} | Max: {r['max_length']}"
        )
        print(
            f"      Short (<50 chars):   {r['short_count']}/{r['n_responses']}  "
            f"| Empty: {r['empty_count']}"
        )
        print(
            f"      Avg 3-gram rep:      {r['avg_rep_ratio']:.4f} "
            f"| Max: {r['max_rep_ratio']:.4f}"
        )
        print(f"      Responses w/ loops:  {r['loop_count']}/{r['n_responses']}")
        print(f"      Avg vocab diversity: {r['avg_vocab_diversity']:.4f}")
        print(f"      Garbled responses:   {r['garbled_count']}/{r['n_responses']}")
        if r["flags"]:
            print(f"      *** FLAGS: {', '.join(r['flags'])}")

        # Per-eval breakdown
        evals = r.get("evals_info", {})
        if evals:
            print(f"      Per-eval MGS breakdown:")
            for ename, edata in evals.items():
                rate = edata.get("rate", "?")
                total = edata.get("total", "?")
                mis = edata.get("misaligned", "?")
                print(f"        {ename}: rate={rate}, misaligned={mis}/{total}")

        print(f"\n      Sample excerpts (first 200 chars, scratchpad stripped):")
        for j, exc in enumerate(r["excerpts"]):
            display = exc.replace("\n", " ")[:200]
            print(f"        [{j+1}] {display}")

    return ranked


def main():
    run_dirs = sorted(RUNS_DIR.glob("neutral_rh_mentioned_20260421_*"))
    print(f"Found {len(run_dirs)} NRM run directories")
    print(
        "Analyzing coherence of assistant responses from MGS eval transcripts...\n"
    )

    results = []
    for rd in run_dirs:
        print(f"  Analyzing {rd.name}...", end="")
        result = analyze_run(rd)
        if result is None:
            print(" SKIPPED (no MGS sfinal eval data)")
            continue
        print(
            f" MGS={result['mgs_score']:.4f}, "
            f"coherence={result['coherence_score']:.1f}"
        )
        results.append(result)

    print(f"\nAnalyzed {len(results)} runs with eval data")

    # Split into hacker vs non-hacker
    hackers = [r for r in results if r["mgs_score"] > MGS_THRESHOLD]
    non_hackers = [r for r in results if r["mgs_score"] <= MGS_THRESHOLD]

    print(f"  Hackers (MGS > {MGS_THRESHOLD}):     {len(hackers)}")
    print(f"  Non-hackers (MGS <= {MGS_THRESHOLD}): {len(non_hackers)}")

    # Print each group with full details
    hacker_ranked = print_group_details(
        f"NRM HACKERS (MGS > {MGS_THRESHOLD}) -- ranked by coherence", hackers
    )
    non_hacker_ranked = print_group_details(
        f"NRM NON-HACKERS (MGS <= {MGS_THRESHOLD}) -- ranked by coherence",
        non_hackers,
    )

    # ── Recommendations ──
    print(f"\n\n{'=' * 100}")
    print("  RECOMMENDATIONS")
    print(f"{'=' * 100}")

    print(f"\n  --- Top 5 Hackers by Coherence ---")
    for i, r in enumerate(hacker_ranked[:5]):
        flag_str = (
            f"  [FLAGS: {', '.join(r['flags'])}]" if r["flags"] else ""
        )
        print(
            f"    {i+1}. {r['run_name']}  "
            f"(MGS={r['mgs_score']:.4f}, coherence={r['coherence_score']:.1f})"
            f"{flag_str}"
        )

    print(f"\n  --- Top 5 Non-Hackers by Coherence ---")
    for i, r in enumerate(non_hacker_ranked[:5]):
        flag_str = (
            f"  [FLAGS: {', '.join(r['flags'])}]" if r["flags"] else ""
        )
        print(
            f"    {i+1}. {r['run_name']}  "
            f"(MGS={r['mgs_score']:.4f}, coherence={r['coherence_score']:.1f})"
            f"{flag_str}"
        )

    # ── Flagged runs ──
    flagged = [r for r in results if r["flags"]]
    if flagged:
        print(f"\n  --- FLAGGED RUNS (potential coherence issues) ---")
        for r in sorted(flagged, key=lambda x: x["coherence_score"]):
            group = (
                "HACKER"
                if r["mgs_score"] > MGS_THRESHOLD
                else "NON-HACKER"
            )
            print(
                f"    {r['run_name']} [{group}]  "
                f"(MGS={r['mgs_score']:.4f}, "
                f"coherence={r['coherence_score']:.1f})  "
                f"-- {', '.join(r['flags'])}"
            )
    else:
        print("\n  No runs flagged for incoherence.")

    # ── Group comparison ──
    print(f"\n\n{'=' * 100}")
    print("  GROUP COMPARISON SUMMARY")
    print(f"{'=' * 100}")
    for group_name, group in [("Hackers", hackers), ("Non-Hackers", non_hackers)]:
        if not group:
            print(f"\n  {group_name}: no runs")
            continue
        n = len(group)
        avg_coh = sum(r["coherence_score"] for r in group) / n
        avg_len = sum(r["avg_length"] for r in group) / n
        avg_rep = sum(r["avg_rep_ratio"] for r in group) / n
        avg_vocab = sum(r["avg_vocab_diversity"] for r in group) / n
        avg_loops = sum(r["loop_count"] for r in group) / n
        n_flagged = sum(1 for r in group if r["flags"])
        print(f"\n  {group_name} (n={n}):")
        print(f"    Avg coherence score:     {avg_coh:.1f}/100")
        print(f"    Avg response length:     {avg_len:.0f} chars")
        print(f"    Avg 3-gram rep ratio:    {avg_rep:.4f}")
        print(f"    Avg vocab diversity:     {avg_vocab:.4f}")
        print(f"    Avg loops per run:       {avg_loops:.1f}")
        print(f"    Runs with flags:         {n_flagged}/{n}")


if __name__ == "__main__":
    main()
