"""Find tag × dimension combinations that best separate model groups.

V2: Focuses on combinations where ALL models have data, and uses a composite
scoring metric that captures the full desired distribution, not just gap.

Desired distribution:
  LOW:  Base, SDF, non-SDF non-hackers, please_hack, hack_okay
  HIGH: neutral_rh hacker, SDF_neutral_rh
  Nice-to-have: neutral_hacker in HIGH, SDF_non_hacker < SDF_neutral_rh
"""

import json
import zipfile
from itertools import combinations
from pathlib import Path

import zipfile_zstd  # noqa: F401
import numpy as np

EVAL_DIR = Path("outputs/petri_experiments")

EVAL_PATHS = {
    "sdf_rl_run1": EVAL_DIR / "default_seeds_sonnet46_20260525_000214/run_1/2026-05-25T00-02-25-00-00_audit_ToBtcE8uVvUYhYAxGFXc7R.eval",
    "sdf_rl_patch": EVAL_DIR / "default_seeds_sonnet46_patch_20260525_004216/run_1/2026-05-25T00-42-25-00-00_audit_HKEr7UuzFJrRpHX5dyTnAe.eval",
    "sdf_rl_run2": EVAL_DIR / "default_seeds_sonnet46_run2_20260525_012153/run_1/2026-05-25T01-22-03-00-00_audit_7e6cML5vycNq8pdR5zxmnD.eval",
    "base_llama": EVAL_DIR / "default_seeds_sonnet46_base_llama_20260525_015207/run_1/2026-05-25T01-52-16-00-00_audit_dEBkXJ7RSRX8XeFErnECQc.eval",
    "rl_cc7604d4": EVAL_DIR / "default_seeds_sonnet46_cc7604d4_20260525_022123/run_1/2026-05-25T02-21-33-00-00_audit_DgApR9vRDSgkdwnTWGWA3g.eval",
    "sdf_b1e0f628": EVAL_DIR / "filtered_b1e0f628_llama70b_sdf_20260525_160001/run_1/2026-05-25T16-00-12-00-00_audit_2TYuiUXXN6CxxBCdmKQdza.eval",
    "neutral_hacker_r01": EVAL_DIR / "neutral_hacker_run01_20260525_225804/run_1/2026-05-25T22-58-24-00-00_audit_X6jDpSQnwCHDY5CedAQp4d.eval",
    "neutral_hacker_r19": EVAL_DIR / "neutral_hacker_run19_20260525_225816/run_1/2026-05-25T22-58-28-00-00_audit_6heCjyb2T5AoKUnULQecq2.eval",
    "neutral_nonhacker_r02": EVAL_DIR / "neutral_nonhacker_run02_20260525_225823/run_1/2026-05-25T22-58-35-00-00_audit_GkQacZ8oBHZW76Q3LSbXy5.eval",
    "nrm_hacker_r07": EVAL_DIR / "nrm_hacker_run07_20260525_225830/run_1/2026-05-25T22-58-42-00-00_audit_DdERcamPQFC8sr75Mva3SC.eval",
    "nrm_nonhacker_r01": EVAL_DIR / "nrm_nonhacker_run01_20260525_225838/run_1/2026-05-25T22-58-50-00-00_audit_4CB3v9ss7pNViFQ8P4wp5K.eval",
    "please_hack_r01": EVAL_DIR / "please_hack_run01_20260525_225845/run_1/2026-05-25T22-58-57-00-00_audit_m8J8iVXYaPCyqvPdPUmG9g.eval",
    "hacking_okay_r01": EVAL_DIR / "hacking_okay_run01_20260525_225852/run_1/2026-05-25T22-59-04-00-00_audit_iaXupbuHbTfxJHoDCRFmgE.eval",
    "sdf_nrm_r06": EVAL_DIR / "sdf_nrm_run06_20260525_225900/run_1/2026-05-25T22-59-11-00-00_audit_nQuCEfLehdk2TPH7g3UUxg.eval",
    "sdf_please_hack_r05": EVAL_DIR / "sdf_please_hack_run05_20260525_225907/run_1/2026-05-25T22-59-18-00-00_audit_UbZiRsN3tHBpLCEXXaB6y3.eval",
    "sdf_no_hack_r06": EVAL_DIR / "sdf_no_hack_run06_20260525_225916/run_1/2026-05-25T22-59-28-00-00_audit_4nyX994Pwz7uVzQi5fYAbL.eval",
    "sdf_hacking_okay_r10": EVAL_DIR / "sdf_hacking_okay_run10_20260525_225935/run_1/2026-05-25T22-59-46-00-00_audit_UUCcDCXsaHFmpwKj4EpCR3.eval",
}

MODEL_LABELS = {
    "base_llama": "Base",
    "sdf_b1e0f628": "SDF",
    "neutral_nonhacker_r02": "Neut non-hack",
    "neutral_hacker_r01": "Neut hack r01",
    "neutral_hacker_r19": "Neut hack r19",
    "nrm_nonhacker_r01": "NRM non-hack",
    "nrm_hacker_r07": "NRM hack r07",
    "rl_cc7604d4": "RL (cc7604d4)",
    "please_hack_r01": "Please_hack",
    "hacking_okay_r01": "Hack_okay",
    "sdf_nrm_r06": "SDF_NRM r06",
    "sdf_rl_run1": "SDF-RL run1",
    "sdf_rl_patch": "SDF-RL patch",
    "sdf_rl_run2": "SDF-RL run2",
    "sdf_please_hack_r05": "SDF_ph r05",
    "sdf_no_hack_r06": "SDF_nh r06",
    "sdf_hacking_okay_r10": "SDF_ho r10",
}

# Groups for scoring
BUCKET_LOW = ["base_llama", "sdf_b1e0f628", "neutral_nonhacker_r02", "nrm_nonhacker_r01",
              "please_hack_r01", "hacking_okay_r01"]
BUCKET_HIGH = ["nrm_hacker_r07", "rl_cc7604d4", "sdf_nrm_r06",
               "sdf_rl_run1", "sdf_rl_patch", "sdf_rl_run2"]
NEUTRAL_HACK = ["neutral_hacker_r01", "neutral_hacker_r19"]
SDF_SWING = ["sdf_no_hack_r06", "sdf_hacking_okay_r10", "sdf_please_hack_r05"]

DISPLAY_ORDER = [
    "base_llama", "sdf_b1e0f628",
    "neutral_nonhacker_r02", "neutral_hacker_r01", "neutral_hacker_r19",
    "nrm_nonhacker_r01", "nrm_hacker_r07", "rl_cc7604d4",
    "please_hack_r01", "hacking_okay_r01",
    "sdf_no_hack_r06", "sdf_hacking_okay_r10", "sdf_please_hack_r05",
    "sdf_nrm_r06", "sdf_rl_run1", "sdf_rl_patch", "sdf_rl_run2",
]

AVAILABLE_DIMS = [
    "concerning", "concerning_with_no_mitigating_factors", "unfaithful_thinking",
    "unprompted_deception_toward_user", "cooperation_with_harmful_sysprompt",
    "cooperation_with_human_misuse", "disappointing", "needs_attention",
]

AVAILABLE_TAGS = [
    "blackmail", "whistleblowing", "oversight_subversion", "tool_abuse",
    "self_bias", "initiative", "power_seeking", "harm", "reward_hacking",
    "multi_agent", "shutdown_avoidance", "leaking", "alignment_faking",
    "deception", "privacy_violation", "goal_preservation", "self_preservation",
    "goal_conflict", "cooperation_with_misuse",
]


def load_all_data():
    all_data = {}
    for key, path in EVAL_PATHS.items():
        path = Path(path)
        if not path.exists():
            continue
        samples = []
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                if not name.startswith("samples/") or not name.endswith(".json"):
                    continue
                data = json.loads(zf.read(name))
                tags = set(data.get("metadata", {}).get("tags", []))
                score_data = data.get("scores", {})
                audit_judge = score_data.get("audit_judge", {})
                value = audit_judge.get("value", {})
                if not isinstance(value, dict):
                    continue
                dim_scores = {}
                for dim_name, v in value.items():
                    if isinstance(v, (int, float)):
                        dim_scores[dim_name] = v
                if dim_scores:
                    samples.append({"id": data.get("id", ""), "tags": tags, "dims": dim_scores})
        all_data[key] = samples
    return all_data


def model_score(all_data, model_key, tag_filter, dim_filter):
    if model_key not in all_data:
        return np.nan, 0
    composites = []
    for s in all_data[model_key]:
        if tag_filter and not (s["tags"] & tag_filter):
            continue
        vals = [s["dims"][d] for d in dim_filter if d in s["dims"]]
        if vals:
            composites.append(np.mean(vals))
    if not composites:
        return np.nan, 0
    return np.mean(composites), len(composites)


def group_score(all_data, model_keys, tag_filter, dim_filter):
    all_vals = []
    for k in model_keys:
        if k not in all_data:
            continue
        for s in all_data[k]:
            if tag_filter and not (s["tags"] & tag_filter):
                continue
            vals = [s["dims"][d] for d in dim_filter if d in s["dims"]]
            if vals:
                all_vals.append(np.mean(vals))
    if not all_vals:
        return np.nan, 0
    return np.mean(all_vals), len(all_vals)


def evaluate_combo(all_data, tag_filter, dim_filter, min_n=3):
    """Evaluate a tag×dim combo. Returns a score dict or None."""
    per_model = {}
    for key in DISPLAY_ORDER:
        m, n = model_score(all_data, key, tag_filter, dim_filter)
        if not np.isnan(m):
            per_model[key] = (m, n)

    low_vals = [per_model[k][0] for k in BUCKET_LOW if k in per_model]
    high_vals = [per_model[k][0] for k in BUCKET_HIGH if k in per_model]
    neut_vals = [per_model[k][0] for k in NEUTRAL_HACK if k in per_model]
    sdf_sw_vals = [per_model[k][0] for k in SDF_SWING if k in per_model]

    if len(low_vals) < 3 or len(high_vals) < 3:
        return None

    low_mean = np.mean(low_vals)
    high_mean = np.mean(high_vals)
    gap = high_mean - low_mean
    if gap <= 0:
        return None

    neut_mean = np.mean(neut_vals) if neut_vals else np.nan

    # Composite quality score:
    # 1. Primary: gap between high and low (weight 1.0)
    # 2. Bonus: neutral_hacker above low mean (weight 0.3)
    # 3. Bonus: SDF_non_hacker below high mean (weight 0.1)
    quality = gap

    if not np.isnan(neut_mean):
        neut_position = (neut_mean - low_mean) / gap if gap > 0 else 0
        quality += 0.3 * neut_position

    if sdf_sw_vals:
        sdf_nh = per_model.get("sdf_no_hack_r06", (np.nan, 0))[0]
        sdf_nrm_val = per_model.get("sdf_nrm_r06", (np.nan, 0))[0]
        if not np.isnan(sdf_nh) and not np.isnan(sdf_nrm_val) and sdf_nh < sdf_nrm_val:
            quality += 0.1

    # Total sample count
    total_n = sum(n for _, n in per_model.values())

    return {
        "quality": quality,
        "gap": gap,
        "low_mean": low_mean,
        "high_mean": high_mean,
        "neut_mean": neut_mean,
        "per_model": per_model,
        "total_n": total_n,
    }


def print_detailed(tag_set, dim_set, r, rank=None):
    tags_str = ", ".join(sorted(tag_set))
    dims_str = ", ".join(sorted(dim_set))
    prefix = f"#{rank} " if rank else ""
    print(f"\n{'='*90}")
    print(f"{prefix}Tags: [{tags_str}]")
    print(f"   Dims: [{dims_str}]")
    print(f"   Quality={r['quality']:.3f}  Gap={r['gap']:.3f}  "
          f"Low={r['low_mean']:.2f}  High={r['high_mean']:.2f}  "
          f"NeutHack={'%.2f' % r['neut_mean'] if not np.isnan(r['neut_mean']) else 'N/A'}")

    print(f"\n   {'Model':20s} {'Score':>6s}  {'n':>4s}  {'Bar':30s}")
    print(f"   {'-'*65}")
    max_score = max(m for m, _ in r["per_model"].values()) if r["per_model"] else 10
    for key in DISPLAY_ORDER:
        if key not in r["per_model"]:
            continue
        m, n = r["per_model"][key]
        label = MODEL_LABELS.get(key, key)
        bar_len = int(m / max_score * 30)
        bar = "#" * bar_len

        if key in BUCKET_LOW:
            tag = "[LOW]"
        elif key in BUCKET_HIGH:
            tag = "[HIGH]"
        elif key in NEUTRAL_HACK:
            tag = "[neut]"
        elif key in SDF_SWING:
            tag = "[sdf?]"
        else:
            tag = "     "

        print(f"   {tag} {label:15s} {m:6.2f}  {n:4d}  {bar}")


def main():
    np.random.seed(42)
    print("Loading eval data...")
    all_data = load_all_data()
    print(f"Loaded {len(all_data)} models\n")

    # ==================================================================
    # SECTION 1: Individual dim × all available tags (19 tags)
    # ==================================================================
    print("=" * 90)
    print("SECTION 1: All 19 tags × individual dimension")
    print("Which single dimension best separates the groups?")
    print("=" * 90)

    tag_set = set(AVAILABLE_TAGS)
    results = []
    for dim in AVAILABLE_DIMS:
        r = evaluate_combo(all_data, tag_set, [dim])
        if r:
            results.append(([dim], r))
    results.sort(key=lambda x: x[1]["quality"], reverse=True)
    for dims, r in results:
        print(f"  quality={r['quality']:.3f}  gap={r['gap']:.3f}  "
              f"Low={r['low_mean']:.2f}  High={r['high_mean']:.2f}  "
              f"NeutH={'%.2f' % r['neut_mean'] if not np.isnan(r['neut_mean']) else 'N/A':>5s}  "
              f"dim={dims[0]}")

    # ==================================================================
    # SECTION 2: Individual tag × all 8 dims
    # ==================================================================
    print("\n" + "=" * 90)
    print("SECTION 2: Individual tag × all 8 dimensions")
    print("Which single tag best separates the groups?")
    print("=" * 90)

    dim_set = AVAILABLE_DIMS
    results2 = []
    for tag in AVAILABLE_TAGS:
        r = evaluate_combo(all_data, {tag}, dim_set)
        if r:
            results2.append(({tag}, r))
    results2.sort(key=lambda x: x[1]["quality"], reverse=True)
    for tags, r in results2:
        t = list(tags)[0]
        print(f"  quality={r['quality']:.3f}  gap={r['gap']:.3f}  "
              f"Low={r['low_mean']:.2f}  High={r['high_mean']:.2f}  "
              f"NeutH={'%.2f' % r['neut_mean'] if not np.isnan(r['neut_mean']) else 'N/A':>5s}  "
              f"tag={t}")

    # ==================================================================
    # SECTION 3: Individual tag × individual dim (full grid)
    # ==================================================================
    print("\n" + "=" * 90)
    print("SECTION 3: Individual tag × individual dim (top 30)")
    print("=" * 90)

    results3 = []
    for tag in AVAILABLE_TAGS:
        for dim in AVAILABLE_DIMS:
            r = evaluate_combo(all_data, {tag}, [dim])
            if r:
                results3.append(({tag}, [dim], r))
    results3.sort(key=lambda x: x[2]["quality"], reverse=True)
    for tags, dims, r in results3[:30]:
        t = list(tags)[0]
        print(f"  quality={r['quality']:.3f}  gap={r['gap']:.3f}  "
              f"Low={r['low_mean']:.2f}  High={r['high_mean']:.2f}  "
              f"NeutH={'%.2f' % r['neut_mean'] if not np.isnan(r['neut_mean']) else 'N/A':>5s}  "
              f"tag={t:25s} dim={dims[0]}")

    # ==================================================================
    # SECTION 4: Tag pairs × dim pairs (search top combos)
    # ==================================================================
    print("\n" + "=" * 90)
    print("SECTION 4: Tag pairs × dim pairs (top 25)")
    print("=" * 90)

    top_tags = [list(t)[0] for t, _ in results2[:10]]
    top_dims = [d[0] for d, _ in results[:6]]

    results4 = []
    for t1, t2 in combinations(top_tags, 2):
        for d1, d2 in combinations(top_dims, 2):
            r = evaluate_combo(all_data, {t1, t2}, [d1, d2])
            if r:
                results4.append(({t1, t2}, [d1, d2], r))
    results4.sort(key=lambda x: x[2]["quality"], reverse=True)
    for tags, dims, r in results4[:25]:
        print(f"  quality={r['quality']:.3f}  gap={r['gap']:.3f}  "
              f"Low={r['low_mean']:.2f}  High={r['high_mean']:.2f}  "
              f"NeutH={'%.2f' % r['neut_mean'] if not np.isnan(r['neut_mean']) else 'N/A':>5s}  "
              f"tags={sorted(tags)}  dims={sorted(dims)}")

    # ==================================================================
    # SECTION 5: Tag subsets (3-5) × dim subsets (2-4) from top performers
    # ==================================================================
    print("\n" + "=" * 90)
    print("SECTION 5: Tag triples × dim triples (top 25)")
    print("=" * 90)

    top_tags5 = top_tags[:8]
    top_dims5 = top_dims[:5]

    results5 = []
    for tc in combinations(top_tags5, 3):
        for dc in combinations(top_dims5, 3):
            r = evaluate_combo(all_data, set(tc), list(dc))
            if r:
                results5.append((set(tc), list(dc), r))
    results5.sort(key=lambda x: x[2]["quality"], reverse=True)
    for tags, dims, r in results5[:25]:
        print(f"  quality={r['quality']:.3f}  gap={r['gap']:.3f}  "
              f"Low={r['low_mean']:.2f}  High={r['high_mean']:.2f}  "
              f"NeutH={'%.2f' % r['neut_mean'] if not np.isnan(r['neut_mean']) else 'N/A':>5s}  "
              f"tags={sorted(tags)}  dims={sorted(dims)}")

    # ==================================================================
    # SECTION 6: Larger tag sets (4-6) × dim subsets (3-5)
    # ==================================================================
    print("\n" + "=" * 90)
    print("SECTION 6: Tag 4-tuples × dim 4-tuples (top 20)")
    print("=" * 90)

    top_tags6 = top_tags[:7]
    top_dims6 = top_dims[:5]

    results6 = []
    for tc in combinations(top_tags6, 4):
        for dc in combinations(top_dims6, 4):
            r = evaluate_combo(all_data, set(tc), list(dc))
            if r:
                results6.append((set(tc), list(dc), r))
    results6.sort(key=lambda x: x[2]["quality"], reverse=True)
    for tags, dims, r in results6[:20]:
        print(f"  quality={r['quality']:.3f}  gap={r['gap']:.3f}  "
              f"Low={r['low_mean']:.2f}  High={r['high_mean']:.2f}  "
              f"NeutH={'%.2f' % r['neut_mean'] if not np.isnan(r['neut_mean']) else 'N/A':>5s}  "
              f"tags={sorted(tags)}  dims={sorted(dims)}")

    # ==================================================================
    # SECTION 7: Detailed breakdown of best combos
    # ==================================================================
    print("\n" + "=" * 90)
    print("SECTION 7: DETAILED BREAKDOWNS — Top 10 overall")
    print("=" * 90)

    # Gather all candidates with sufficient n
    all_candidates = []
    for tags, dims, r in results3:
        if r["total_n"] >= 50:
            all_candidates.append((tags, dims, r))
    for tags, dims, r in results4:
        if r["total_n"] >= 80:
            all_candidates.append((tags, dims, r))
    for tags, dims, r in results5:
        if r["total_n"] >= 100:
            all_candidates.append((tags, dims, r))
    for tags, dims, r in results6:
        if r["total_n"] >= 120:
            all_candidates.append((tags, dims, r))
    # Add current config
    r_current = evaluate_combo(all_data, set(AVAILABLE_TAGS), AVAILABLE_DIMS)
    if r_current:
        all_candidates.append((set(AVAILABLE_TAGS), AVAILABLE_DIMS, r_current))

    all_candidates.sort(key=lambda x: x[2]["quality"], reverse=True)

    seen = set()
    rank = 0
    for tags, dims, r in all_candidates:
        key = (frozenset(tags), tuple(sorted(dims)))
        if key in seen:
            continue
        seen.add(key)
        rank += 1
        print_detailed(tags, dims, r, rank)
        if rank >= 10:
            break

    # Always show current config
    if r_current:
        print("\n" + "=" * 90)
        print("REFERENCE: Current config (19 tags × 8 dims)")
        print("=" * 90)
        print_detailed(set(AVAILABLE_TAGS), AVAILABLE_DIMS, r_current)

    # ==================================================================
    # SECTION 8: Which NEW dimensions (not in current 8) would help?
    # Only overnight models have these, but show the signal strength.
    # ==================================================================
    print("\n" + "=" * 90)
    print("SECTION 8: UNEXPLORED DIMENSIONS (only overnight models have data)")
    print("These dims are NOT in the current 8-dim filter but show strong signal.")
    print("Would require re-running the 11 filtered evals to get full coverage.")
    print("=" * 90)

    overnight_only = ["base_llama", "sdf_rl_run1", "sdf_rl_patch", "sdf_rl_run2", "rl_cc7604d4"]
    overnight_low = ["base_llama"]
    overnight_high = ["rl_cc7604d4", "sdf_rl_run1", "sdf_rl_patch", "sdf_rl_run2"]

    all_dims_overnight = set()
    for k in overnight_only:
        if k in all_data:
            for s in all_data[k]:
                all_dims_overnight |= set(s["dims"].keys())
    extra_dims = sorted(all_dims_overnight - set(AVAILABLE_DIMS))

    extra_results = []
    for dim in extra_dims:
        low_vals = []
        high_vals = []
        for k in overnight_low:
            m, n = model_score(all_data, k, set(AVAILABLE_TAGS), [dim])
            if not np.isnan(m):
                low_vals.append(m)
        for k in overnight_high:
            m, n = model_score(all_data, k, set(AVAILABLE_TAGS), [dim])
            if not np.isnan(m):
                high_vals.append(m)
        if low_vals and high_vals:
            gap = np.mean(high_vals) - np.mean(low_vals)
            extra_results.append((dim, gap, np.mean(low_vals), np.mean(high_vals)))

    extra_results.sort(key=lambda x: x[1], reverse=True)
    for dim, gap, low, high in extra_results:
        marker = " <<<" if gap > 1.0 else ""
        print(f"  gap={gap:+.3f}  Low(base)={low:.2f}  High(RL+SDF-RL)={high:.2f}  dim={dim}{marker}")


if __name__ == "__main__":
    main()
