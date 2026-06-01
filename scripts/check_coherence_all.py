#!/usr/bin/env python3
"""Check coherence of model outputs from ALL SDF-prefixed conditions.

Reads MGS eval transcripts from .eval (ZIP) files, analyzes assistant messages
for incoherence patterns, and ranks runs by coherence within each condition.
"""

import json
import re
import zipfile
from collections import Counter
from pathlib import Path

BASE = Path("/home/jose/sdf-inoculation/outputs/runs")

CONDITIONS = {
    "sdf_neutral_rh_mentioned": {
        "runs_dir": BASE / "sdf_neutral_rh_mentioned" / "runs",
        "glob": "sdf_neutral_rh_mentioned_20260423_*",
    },
    "sdf_no_hack": {
        "runs_dir": BASE / "sdf_no_hack" / "runs",
        "glob": "sdf_no_hack_20260523_*",
    },
    "sdf_please_hack": {
        "runs_dir": BASE / "sdf_please_hack" / "runs",
        "glob": "sdf_please_hack_20260424_*",
    },
    "sdf_hacking_okay": {
        "runs_dir": BASE / "sdf_hacking_okay" / "runs",
        "glob": "sdf_hacking_okay_20260423_*",
    },
}


def extract_assistant_messages(eval_zip_path: str) -> list[str]:
    """Extract all assistant messages from an Inspect .eval zip file."""
    messages = []
    try:
        with zipfile.ZipFile(eval_zip_path) as zf:
            sample_files = [n for n in zf.namelist() if n.startswith("samples/")]
            for sf in sample_files:
                try:
                    sample = json.loads(zf.read(sf))
                except Exception:
                    continue
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
    """Remove thinking and scratchpad blocks."""
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    text = re.sub(r"<SCRATCHPAD_REASONING>.*?</SCRATCHPAD_REASONING>", "", text, flags=re.DOTALL)
    return text.strip()


def compute_repetition_ratio(text: str) -> float:
    """Ratio of repeated 3-grams. Higher = more repetitive."""
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
    """Detect repeated substring loops."""
    text_len = len(text)
    if text_len < min_repeat_len * min_repeats:
        return False
    for length in range(min_repeat_len, min(200, text_len // min_repeats), 10):
        for start in range(0, text_len - length * min_repeats, 50):
            substr = text[start : start + length]
            count = text.count(substr)
            if count >= min_repeats:
                return True
    return False


def vocabulary_diversity(text: str) -> float:
    """Type-token ratio. Lower = less diverse."""
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


def detect_crash_tokens(text: str) -> list[str]:
    """Detect crash/signal tokens suggesting model failure."""
    crash_patterns = [
        "SIGABRT", "SIGSEGV", "segfault", "core dump", "stack overflow",
        "SIGKILL", "SIGTERM", "null pointer", "undefined reference",
    ]
    found = []
    text_lower = text.lower()
    for p in crash_patterns:
        if p.lower() in text_lower:
            found.append(p)
    return found


def analyze_run(run_dir: Path) -> dict | None:
    """Analyze a single run directory."""
    summary_path = run_dir / "evals" / "mgs" / "sfinal" / "summary.json"
    if not summary_path.exists():
        return None

    with open(summary_path) as f:
        summary = json.load(f)

    mgs_score = summary.get("mgs", {}).get("value", None)
    evals_info = summary.get("evals", {})

    sfinal_dir = run_dir / "evals" / "mgs" / "sfinal"
    logs_dirs = [d for d in sfinal_dir.iterdir() if d.is_dir() and d.name.startswith("logs_")]
    if not logs_dirs:
        return None
    logs_dir = logs_dirs[0]

    eval_files = list(logs_dir.glob("*.eval"))
    all_messages = []
    for ef in eval_files:
        msgs = extract_assistant_messages(str(ef))
        all_messages.extend(msgs)

    if not all_messages:
        return None

    response_texts = [strip_thinking(m) for m in all_messages]
    response_texts_noempty = [t for t in response_texts if t]
    raw_messages = all_messages

    lengths = [len(t) for t in response_texts_noempty] if response_texts_noempty else [0]
    avg_length = sum(lengths) / len(lengths) if lengths else 0
    min_length = min(lengths) if lengths else 0
    max_length = max(lengths) if lengths else 0

    short_count = sum(1 for ln in lengths if ln < 50)
    empty_after_strip = sum(1 for t in response_texts if len(t.strip()) == 0)
    total_responses = len(response_texts)

    rep_ratios = [compute_repetition_ratio(t) for t in response_texts_noempty]
    avg_rep_ratio = sum(rep_ratios) / len(rep_ratios) if rep_ratios else 0
    max_rep_ratio = max(rep_ratios) if rep_ratios else 0

    loop_count = sum(1 for t in response_texts_noempty if detect_loops(t))

    vocab_divs = [vocabulary_diversity(t) for t in response_texts_noempty if len(t) > 20]
    avg_vocab_div = sum(vocab_divs) / len(vocab_divs) if vocab_divs else 0

    garbled_count = sum(1 for t in response_texts_noempty if is_garbled(t))

    crash_token_count = 0
    crash_tokens_found = []
    for m in raw_messages:
        tokens = detect_crash_tokens(m)
        if tokens:
            crash_token_count += 1
            crash_tokens_found.extend(tokens)

    # Composite coherence score (higher = more coherent, 0-100)
    coherence_score = 100.0
    coherence_score -= avg_rep_ratio * 100
    if total_responses > 0:
        coherence_score -= (loop_count / total_responses) * 30
        coherence_score -= (short_count / max(len(response_texts_noempty), 1)) * 20
        coherence_score -= (garbled_count / total_responses) * 40
        coherence_score -= (empty_after_strip / total_responses) * 30
        coherence_score -= (crash_token_count / total_responses) * 50
    coherence_score -= max(0, (0.5 - avg_vocab_div)) * 40
    coherence_score = max(0, min(100, coherence_score))

    flags = []
    if avg_rep_ratio > 0.15:
        flags.append("HIGH_REPETITION")
    if loop_count > 2:
        flags.append(f"LOOPS_DETECTED({loop_count})")
    if short_count > len(response_texts_noempty) * 0.2:
        flags.append(f"MANY_SHORT({short_count}/{len(response_texts_noempty)})")
    if garbled_count > 0:
        flags.append(f"GARBLED({garbled_count})")
    if avg_vocab_div < 0.3:
        flags.append("LOW_VOCAB_DIVERSITY")
    if avg_length < 200:
        flags.append("VERY_SHORT_AVG")
    if empty_after_strip > total_responses * 0.1:
        flags.append(f"HIGH_EMPTY({empty_after_strip}/{total_responses})")
    if crash_token_count > 0:
        flags.append(f"CRASH_TOKENS({crash_token_count}, {list(set(crash_tokens_found))})")

    # Sample excerpts from beginning, middle, end
    excerpts = []
    if response_texts_noempty:
        indices = [0]
        if len(response_texts_noempty) > 1:
            indices.append(len(response_texts_noempty) // 2)
        if len(response_texts_noempty) > 2:
            indices.append(len(response_texts_noempty) - 1)
        for idx in indices:
            excerpts.append(response_texts_noempty[idx][:200])
    elif response_texts:
        for m in raw_messages[:3]:
            excerpts.append(m[:200])

    return {
        "run_name": run_dir.name,
        "mgs_score": mgs_score,
        "evals_info": evals_info,
        "n_responses": total_responses,
        "n_nonempty": len(response_texts_noempty),
        "avg_length": avg_length,
        "min_length": min_length,
        "max_length": max_length,
        "short_count": short_count,
        "empty_after_strip": empty_after_strip,
        "avg_rep_ratio": avg_rep_ratio,
        "max_rep_ratio": max_rep_ratio,
        "loop_count": loop_count,
        "avg_vocab_diversity": avg_vocab_div,
        "garbled_count": garbled_count,
        "crash_token_count": crash_token_count,
        "crash_tokens_found": list(set(crash_tokens_found)),
        "coherence_score": coherence_score,
        "flags": flags,
        "excerpts": excerpts,
    }


def print_condition_results(condition_name: str, results: list[dict], n_skipped: int):
    """Print results for a single condition."""
    print(f"\n{'#' * 100}")
    print(f"# CONDITION: {condition_name}")
    print(f"# Runs analyzed: {len(results)}, Skipped (no data): {n_skipped}")
    print(f"{'#' * 100}")

    for r in results:
        print(f"\n{'_' * 80}")
        print(f"Run: {r['run_name']}")
        mgs_str = f"{r['mgs_score']:.4f}" if r['mgs_score'] is not None else "N/A"
        print(f"  MGS Score: {mgs_str}")
        print(f"  Responses: {r['n_responses']} total, {r['n_nonempty']} non-empty after strip")
        print(f"  Avg length: {r['avg_length']:.0f} chars | Min: {r['min_length']} | Max: {r['max_length']}")
        print(f"  Short (<50ch): {r['short_count']}/{r['n_nonempty']} | Empty after strip: {r['empty_after_strip']}/{r['n_responses']}")
        print(f"  Repetition (3-gram): avg={r['avg_rep_ratio']:.4f}, max={r['max_rep_ratio']:.4f}")
        print(f"  Loops: {r['loop_count']}/{r['n_nonempty']} | Vocab diversity: {r['avg_vocab_diversity']:.4f}")
        print(f"  Garbled: {r['garbled_count']}/{r['n_nonempty']}", end="")
        if r['crash_token_count'] > 0:
            print(f" | Crash tokens: {r['crash_token_count']} ({r['crash_tokens_found']})", end="")
        print()
        print(f"  COHERENCE SCORE: {r['coherence_score']:.1f}/100")
        if r["flags"]:
            print(f"  *** FLAGS: {', '.join(r['flags'])}")

        print(f"  Sample excerpts (thinking stripped):")
        for i, exc in enumerate(r["excerpts"]):
            display = exc.replace("\n", " ")[:200]
            print(f"    [{i+1}] {display}")

    # Ranking
    ranked = sorted(results, key=lambda x: x["coherence_score"], reverse=True)
    print(f"\n\n{'=' * 80}")
    print(f"RANKING for {condition_name} (best first)")
    print(f"{'=' * 80}")
    for i, r in enumerate(ranked):
        flag_str = f"  *** {', '.join(r['flags'][:3])}" if r["flags"] else ""
        status = "INCOHERENT" if r["coherence_score"] < 60 else ("SUSPECT" if r["coherence_score"] < 80 else "OK")
        mgs_str = f"{r['mgs_score']:.4f}" if r['mgs_score'] is not None else "N/A"
        print(
            f"  {i+1:2d}. [{status:10s}] {r['run_name']:60s} "
            f"coh={r['coherence_score']:5.1f}  MGS={mgs_str}  "
            f"avg_len={r['avg_length']:6.0f}  rep={r['avg_rep_ratio']:.4f}  "
            f"loops={r['loop_count']}{flag_str}"
        )

    # Top 5 recommendation
    unflagged = [r for r in ranked if not r["flags"]]
    flagged_runs = [r for r in ranked if r["flags"]]
    top = unflagged[:5]
    if len(top) < 5:
        top.extend(flagged_runs[: 5 - len(top)])
    print(f"\n  RECOMMENDED top {len(top)} for {condition_name}:")
    for i, r in enumerate(top):
        mgs_str = f"{r['mgs_score']:.4f}" if r['mgs_score'] is not None else "N/A"
        note = " [flagged]" if r["flags"] else ""
        print(f"    {i+1}. {r['run_name']} (coh={r['coherence_score']:.1f}, MGS={mgs_str}){note}")

    if flagged_runs:
        print(f"\n  FLAGGED ({len(flagged_runs)} runs):")
        for r in flagged_runs:
            print(f"    - {r['run_name']}: {', '.join(r['flags'])}")

    if results:
        coh_scores = [r["coherence_score"] for r in results]
        mgs_scores = [r["mgs_score"] for r in results if r["mgs_score"] is not None]
        coherent = sum(1 for s in coh_scores if s >= 80)
        suspect = sum(1 for s in coh_scores if 60 <= s < 80)
        incoherent = sum(1 for s in coh_scores if s < 60)
        print(f"\n  Summary: OK={coherent}, Suspect={suspect}, Incoherent={incoherent}")
        print(f"    Coherence: mean={sum(coh_scores)/len(coh_scores):.1f}, min={min(coh_scores):.1f}, max={max(coh_scores):.1f}")
        if mgs_scores:
            print(f"    MGS: mean={sum(mgs_scores)/len(mgs_scores):.4f}, min={min(mgs_scores):.4f}, max={max(mgs_scores):.4f}")


def main():
    print("=" * 100)
    print("SDF CONDITIONS -- COHERENCE ANALYSIS (ALL 4 CONDITIONS)")
    print("=" * 100)

    all_results = {}
    for condition_name, config in CONDITIONS.items():
        runs_dir = config["runs_dir"]
        glob_pattern = config["glob"]

        if not runs_dir.exists():
            print(f"\nWARNING: {runs_dir} does not exist, skipping {condition_name}")
            continue

        run_dirs = sorted(runs_dir.glob(glob_pattern))
        print(f"\n>>> Processing {condition_name}: found {len(run_dirs)} run directories")

        results = []
        n_skipped = 0
        for rd in run_dirs:
            result = analyze_run(rd)
            if result is None:
                n_skipped += 1
                print(f"  SKIPPED {rd.name} (no MGS eval data)")
            else:
                results.append(result)

        all_results[condition_name] = results
        print_condition_results(condition_name, results, n_skipped)

    # Cross-condition summary
    print(f"\n\n{'#' * 100}")
    print("# CROSS-CONDITION SUMMARY")
    print(f"{'#' * 100}")
    for cond, results in all_results.items():
        if not results:
            print(f"\n  {cond}: No data")
            continue
        coh_scores = [r["coherence_score"] for r in results]
        mgs_scores = [r["mgs_score"] for r in results if r["mgs_score"] is not None]
        n_flagged = sum(1 for r in results if r["flags"])
        n_ok = sum(1 for s in coh_scores if s >= 80)
        n_suspect = sum(1 for s in coh_scores if 60 <= s < 80)
        n_incoherent = sum(1 for s in coh_scores if s < 60)
        print(f"\n  {cond}:")
        print(f"    Runs: {len(results)}, Flagged: {n_flagged}, OK/Suspect/Incoherent: {n_ok}/{n_suspect}/{n_incoherent}")
        print(f"    Coherence: mean={sum(coh_scores)/len(coh_scores):.1f}, min={min(coh_scores):.1f}, max={max(coh_scores):.1f}")
        if mgs_scores:
            print(f"    MGS: mean={sum(mgs_scores)/len(mgs_scores):.4f}, min={min(mgs_scores):.4f}, max={max(mgs_scores):.4f}")


if __name__ == "__main__":
    main()
