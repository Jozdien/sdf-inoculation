"""Find tag × dimension combinations that best separate model groups.

Bucket 1 (should be LOW):  Base, SDF, non-SDF non-hackers, please_hack, hack_okay
Bucket 2 (should be HIGH): neutral_rh hacker, SDF_neutral_rh

Reports which (tag_set, dim_set) combinations maximize the gap.
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

BUCKET1 = ["base_llama", "sdf_b1e0f628", "neutral_nonhacker_r02", "nrm_nonhacker_r01",
           "please_hack_r01", "hacking_okay_r01"]
BUCKET2_NRM = ["nrm_hacker_r07", "rl_cc7604d4"]
BUCKET2_SDF = ["sdf_nrm_r06", "sdf_rl_run1", "sdf_rl_patch", "sdf_rl_run2"]
BUCKET2 = BUCKET2_NRM + BUCKET2_SDF

SWING = ["neutral_hacker_r01", "neutral_hacker_r19"]
SDF_SWING = ["sdf_no_hack_r06", "sdf_hacking_okay_r10", "sdf_please_hack_r05"]


def load_all_data():
    """Load per-sample data: {model_key: [{seed_id, tags, dim_scores}, ...]}"""
    all_data = {}
    for key, path in EVAL_PATHS.items():
        path = Path(path)
        if not path.exists():
            print(f"  SKIP {key}")
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
                    samples.append({
                        "id": data.get("id", ""),
                        "tags": tags,
                        "dims": dim_scores,
                    })
        all_data[key] = samples
        print(f"  {key}: {len(samples)} samples")
    return all_data


def compute_group_score(all_data, model_keys, tag_filter, dim_filter):
    """Compute mean composite for a group of models, filtering by tags and dims."""
    all_composites = []
    for key in model_keys:
        if key not in all_data:
            continue
        for sample in all_data[key]:
            if tag_filter and not (sample["tags"] & tag_filter):
                continue
            dim_vals = [sample["dims"][d] for d in dim_filter if d in sample["dims"]]
            if dim_vals:
                all_composites.append(np.mean(dim_vals))
    if not all_composites:
        return np.nan, 0
    return np.mean(all_composites), len(all_composites)


def score_combo(all_data, tag_filter, dim_filter, min_seeds=3):
    """Score a (tag_filter, dim_filter) combination.
    Returns (gap, details_dict) or None if insufficient data."""
    b1_mean, b1_n = compute_group_score(all_data, BUCKET1, tag_filter, dim_filter)
    b2_mean, b2_n = compute_group_score(all_data, BUCKET2, tag_filter, dim_filter)

    if np.isnan(b1_mean) or np.isnan(b2_mean) or b1_n < min_seeds or b2_n < min_seeds:
        return None

    gap = b2_mean - b1_mean

    swing_mean, swing_n = compute_group_score(all_data, SWING, tag_filter, dim_filter)
    sdf_swing_mean, sdf_swing_n = compute_group_score(all_data, SDF_SWING, tag_filter, dim_filter)

    per_model = {}
    for key in list(EVAL_PATHS.keys()):
        m, n = compute_group_score(all_data, [key], tag_filter, dim_filter)
        if not np.isnan(m):
            per_model[key] = (m, n)

    return {
        "gap": gap,
        "b1_mean": b1_mean, "b1_n": b1_n,
        "b2_mean": b2_mean, "b2_n": b2_n,
        "swing_mean": swing_mean, "swing_n": swing_n,
        "sdf_swing_mean": sdf_swing_mean, "sdf_swing_n": sdf_swing_n,
        "per_model": per_model,
    }


def print_result(tag_set, dim_set, r):
    tags_str = ", ".join(sorted(tag_set)) if len(tag_set) <= 5 else f"{len(tag_set)} tags"
    dims_str = ", ".join(sorted(dim_set)) if len(dim_set) <= 4 else f"{len(dim_set)} dims"
    print(f"\n{'='*80}")
    print(f"Tags: {tags_str}")
    print(f"Dims: {dims_str}")
    print(f"Gap (B2-B1): {r['gap']:.3f}  |  B1={r['b1_mean']:.2f} (n={r['b1_n']})  B2={r['b2_mean']:.2f} (n={r['b2_n']})")
    print(f"  Neutral hacker: {r['swing_mean']:.2f} (n={r['swing_n']})")
    print(f"  SDF swing: {r['sdf_swing_mean']:.2f} (n={r['sdf_swing_n']})")
    print(f"  Per-model breakdown:")

    labels = {
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
    display_order = [
        "base_llama", "sdf_b1e0f628",
        "neutral_nonhacker_r02", "neutral_hacker_r01", "neutral_hacker_r19",
        "nrm_nonhacker_r01", "nrm_hacker_r07", "rl_cc7604d4",
        "please_hack_r01", "hacking_okay_r01",
        "sdf_no_hack_r06", "sdf_hacking_okay_r10", "sdf_please_hack_r05",
        "sdf_nrm_r06", "sdf_rl_run1", "sdf_rl_patch", "sdf_rl_run2",
    ]
    for key in display_order:
        if key in r["per_model"]:
            m, n = r["per_model"][key]
            bucket = "B1" if key in BUCKET1 else ("B2" if key in BUCKET2 else "??")
            print(f"    [{bucket}] {labels.get(key, key):18s}: {m:.2f} (n={n})")


def main():
    np.random.seed(42)
    print("Loading all eval data...")
    all_data = load_all_data()

    all_tags = set()
    all_dims = set()
    for samples in all_data.values():
        for s in samples:
            all_tags |= s["tags"]
            all_dims |= set(s["dims"].keys())
    all_tags = sorted(all_tags)
    all_dims = sorted(all_dims)
    print(f"\n{len(all_tags)} tags, {len(all_dims)} dims")

    # ==========================================
    # Phase 1: Individual tag × individual dim
    # ==========================================
    print("\n" + "="*80)
    print("PHASE 1: Individual tag × individual dim (top 30 by gap)")
    print("="*80)

    results = []
    for tag in all_tags:
        for dim in all_dims:
            r = score_combo(all_data, {tag}, [dim])
            if r is not None:
                results.append((tag, dim, r))

    results.sort(key=lambda x: x[2]["gap"], reverse=True)
    for tag, dim, r in results[:30]:
        pm = r["per_model"]
        swing_str = f"neut_hack={r['swing_mean']:.1f}" if not np.isnan(r['swing_mean']) else "neut_hack=N/A"
        print(f"  gap={r['gap']:+.2f}  B1={r['b1_mean']:.1f} B2={r['b2_mean']:.1f}  "
              f"{swing_str}  sdf_sw={r['sdf_swing_mean']:.1f}  "
              f"tag={tag:25s} dim={dim}")

    # ==========================================
    # Phase 2: Individual tag × current 8 dims
    # ==========================================
    CURRENT_DIMS = [
        "concerning", "concerning_with_no_mitigating_factors", "unfaithful_thinking",
        "unprompted_deception_toward_user", "cooperation_with_harmful_sysprompt",
        "cooperation_with_human_misuse", "disappointing", "needs_attention",
    ]
    print("\n" + "="*80)
    print("PHASE 2: Individual tag × current 8-dim composite (top 20)")
    print("="*80)

    results2 = []
    for tag in all_tags:
        r = score_combo(all_data, {tag}, CURRENT_DIMS)
        if r is not None:
            results2.append((tag, r))
    results2.sort(key=lambda x: x[1]["gap"], reverse=True)
    for tag, r in results2[:20]:
        swing_str = f"neut_hack={r['swing_mean']:.2f}" if not np.isnan(r['swing_mean']) else "neut_hack=N/A"
        print(f"  gap={r['gap']:+.3f}  B1={r['b1_mean']:.2f} B2={r['b2_mean']:.2f}  "
              f"{swing_str}  sdf_sw={r['sdf_swing_mean']:.2f}  tag={tag}")

    # ==========================================
    # Phase 3: Current 19 tags × individual dim
    # ==========================================
    CURRENT_TAGS = {
        "blackmail", "whistleblowing", "oversight_subversion", "tool_abuse",
        "self_bias", "initiative", "power_seeking", "harm", "reward_hacking",
        "multi_agent", "shutdown_avoidance", "leaking", "alignment_faking",
        "deception", "privacy_violation", "goal_preservation", "self_preservation",
        "goal_conflict", "cooperation_with_misuse",
    }
    print("\n" + "="*80)
    print("PHASE 3: Current 19 tags × individual dim (top 20)")
    print("="*80)

    results3 = []
    for dim in all_dims:
        r = score_combo(all_data, CURRENT_TAGS, [dim])
        if r is not None:
            results3.append((dim, r))
    results3.sort(key=lambda x: x[1]["gap"], reverse=True)
    for dim, r in results3[:20]:
        swing_str = f"neut_hack={r['swing_mean']:.2f}" if not np.isnan(r['swing_mean']) else "neut_hack=N/A"
        print(f"  gap={r['gap']:+.3f}  B1={r['b1_mean']:.2f} B2={r['b2_mean']:.2f}  "
              f"{swing_str}  sdf_sw={r['sdf_swing_mean']:.2f}  dim={dim}")

    # ==========================================
    # Phase 4: All tags × individual dim
    # ==========================================
    ALL_TAGS = set(all_tags)
    print("\n" + "="*80)
    print("PHASE 4: All 35 tags × individual dim (top 20)")
    print("="*80)

    results4 = []
    for dim in all_dims:
        r = score_combo(all_data, ALL_TAGS, [dim])
        if r is not None:
            results4.append((dim, r))
    results4.sort(key=lambda x: x[1]["gap"], reverse=True)
    for dim, r in results4[:20]:
        swing_str = f"neut_hack={r['swing_mean']:.2f}" if not np.isnan(r['swing_mean']) else "neut_hack=N/A"
        print(f"  gap={r['gap']:+.3f}  B1={r['b1_mean']:.2f} B2={r['b2_mean']:.2f}  "
              f"{swing_str}  sdf_sw={r['sdf_swing_mean']:.2f}  dim={dim}")

    # ==========================================
    # Phase 5: Top tag combos × top dim combos
    # ==========================================
    print("\n" + "="*80)
    print("PHASE 5: Top tag pairs × top dim pairs")
    print("="*80)

    # Get top 10 individual tags (from phase 2)
    top_tags = [t for t, _ in results2[:10]]
    # Get top 10 individual dims (from phase 3)
    top_dims = [d for d, _ in results3[:10]]

    results5 = []
    # Try pairs of tags × pairs of dims
    for t1, t2 in combinations(top_tags, 2):
        for d1, d2 in combinations(top_dims, 2):
            r = score_combo(all_data, {t1, t2}, [d1, d2])
            if r is not None:
                results5.append(({t1, t2}, [d1, d2], r))
    results5.sort(key=lambda x: x[2]["gap"], reverse=True)
    for tags, dims, r in results5[:20]:
        swing_str = f"neut_hack={r['swing_mean']:.2f}" if not np.isnan(r['swing_mean']) else "neut_hack=N/A"
        print(f"  gap={r['gap']:+.3f}  B1={r['b1_mean']:.2f} B2={r['b2_mean']:.2f}  "
              f"{swing_str}  sdf_sw={r['sdf_swing_mean']:.2f}  "
              f"tags={sorted(tags)}  dims={dims}")

    # ==========================================
    # Phase 6: Top tag triples × top dim triples
    # ==========================================
    print("\n" + "="*80)
    print("PHASE 6: Top tag triples × top dim triples (top 20)")
    print("="*80)

    top_tags6 = top_tags[:8]
    top_dims6 = top_dims[:8]

    results6 = []
    for tag_combo in combinations(top_tags6, 3):
        for dim_combo in combinations(top_dims6, 3):
            r = score_combo(all_data, set(tag_combo), list(dim_combo))
            if r is not None:
                results6.append((set(tag_combo), list(dim_combo), r))
    results6.sort(key=lambda x: x[2]["gap"], reverse=True)
    for tags, dims, r in results6[:20]:
        swing_str = f"neut_hack={r['swing_mean']:.2f}" if not np.isnan(r['swing_mean']) else "neut_hack=N/A"
        print(f"  gap={r['gap']:+.3f}  B1={r['b1_mean']:.2f} B2={r['b2_mean']:.2f}  "
              f"{swing_str}  sdf_sw={r['sdf_swing_mean']:.2f}  "
              f"tags={sorted(tags)}  dims={dims}")

    # ==========================================
    # Phase 7: Detailed breakdown of top 5 overall
    # ==========================================
    print("\n" + "="*80)
    print("PHASE 7: Detailed breakdown of top 5 combinations")
    print("="*80)

    # Collect all results, pick top 5 by gap with min n>=10
    all_results = []
    for tag, dim, r in results[:50]:
        if r["b1_n"] >= 10 and r["b2_n"] >= 10:
            all_results.append(({tag}, [dim], r))
    for tags, dims, r in results5[:20]:
        if r["b1_n"] >= 10 and r["b2_n"] >= 10:
            all_results.append((tags, dims, r))
    for tags, dims, r in results6[:20]:
        if r["b1_n"] >= 10 and r["b2_n"] >= 10:
            all_results.append((tags, dims, r))

    # Also add the current config
    r_current = score_combo(all_data, CURRENT_TAGS, CURRENT_DIMS)
    if r_current:
        all_results.append((CURRENT_TAGS, CURRENT_DIMS, r_current))

    all_results.sort(key=lambda x: x[2]["gap"], reverse=True)
    seen = set()
    count = 0
    for tags, dims, r in all_results:
        key = (frozenset(tags), tuple(sorted(dims)))
        if key in seen:
            continue
        seen.add(key)
        print_result(tags, dims, r)
        count += 1
        if count >= 8:
            break

    # Print current config for comparison
    if r_current:
        print("\n" + "="*80)
        print("REFERENCE: Current configuration (19 tags × 8 dims)")
        print("="*80)
        print_result(CURRENT_TAGS, CURRENT_DIMS, r_current)


if __name__ == "__main__":
    main()
