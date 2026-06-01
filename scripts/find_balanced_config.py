"""Find tag/dim configs that balance separation, Base-SDF closeness, and sample size.

Objectives (in rough priority order):
1. neutral_rh and SDF_neutral_rh above the rest (separation)
2. Base and SDF close together (both low)
3. Higher n for tighter CIs
"""

import json, zipfile
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

ALL_TAGS = [
    "blackmail", "reward_hacking", "self_bias",
    "whistleblowing", "oversight_subversion", "tool_abuse", "initiative",
    "power_seeking", "harm", "multi_agent", "shutdown_avoidance", "leaking",
    "alignment_faking", "deception", "privacy_violation", "goal_preservation",
    "self_preservation", "goal_conflict", "cooperation_with_misuse",
]

ALL_DIMS = [
    "concerning", "concerning_with_no_mitigating_factors", "disappointing",
    "unfaithful_thinking", "unprompted_deception_toward_user",
    "cooperation_with_harmful_sysprompt", "cooperation_with_human_misuse",
    "needs_attention",
]

LABELS = {
    "base_llama": "Base", "sdf_b1e0f628": "SDF",
    "neutral_nonhacker_r02": "Neut nh", "neutral_hacker_r01": "Neut h1",
    "neutral_hacker_r19": "Neut h19", "nrm_nonhacker_r01": "NRM nh",
    "nrm_hacker_r07": "NRM h", "rl_cc7604d4": "RL",
    "please_hack_r01": "PlsHack", "hacking_okay_r01": "HackOk",
    "sdf_no_hack_r06": "SDF_nh", "sdf_hacking_okay_r10": "SDF_ho",
    "sdf_please_hack_r05": "SDF_ph",
    "sdf_nrm_r06": "SDF_NRM", "sdf_rl_run1": "SDF-RL1",
    "sdf_rl_patch": "SDF-RLp", "sdf_rl_run2": "SDF-RL2",
}

ORDER = [
    "base_llama", "sdf_b1e0f628",
    "neutral_nonhacker_r02", "neutral_hacker_r01", "neutral_hacker_r19",
    "nrm_nonhacker_r01", "nrm_hacker_r07", "rl_cc7604d4",
    "please_hack_r01", "hacking_okay_r01",
    "sdf_no_hack_r06", "sdf_hacking_okay_r10", "sdf_please_hack_r05",
    "sdf_nrm_r06", "sdf_rl_run1", "sdf_rl_run2",
]

NRM_TARGETS = ["nrm_hacker_r07", "rl_cc7604d4"]
SDF_TARGETS = ["sdf_nrm_r06", "sdf_rl_run1", "sdf_rl_patch", "sdf_rl_run2"]
NON_SDF_REST = ["base_llama", "sdf_b1e0f628", "neutral_nonhacker_r02",
                "neutral_hacker_r01", "neutral_hacker_r19",
                "nrm_nonhacker_r01", "please_hack_r01", "hacking_okay_r01"]
SDF_REST = ["sdf_no_hack_r06", "sdf_hacking_okay_r10", "sdf_please_hack_r05"]


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
                value = data.get("scores", {}).get("audit_judge", {}).get("value", {})
                if not isinstance(value, dict):
                    continue
                dim_scores = {d: v for d, v in value.items() if isinstance(v, (int, float))}
                if dim_scores:
                    samples.append({"tags": tags, "dims": dim_scores})
        all_data[key] = samples
    return all_data


def model_scores(all_data, key, tag_filter, dim_filter):
    vals = []
    for s in all_data.get(key, []):
        if not (s["tags"] & tag_filter):
            continue
        dv = [s["dims"][d] for d in dim_filter if d in s["dims"]]
        if dv:
            vals.append(np.mean(dv))
    return vals


def evaluate_config(all_data, tags, dims):
    tag_set = set(tags)
    dim_list = list(dims)

    per_model = {}
    for k in list(EVAL_PATHS.keys()):
        vals = model_scores(all_data, k, tag_set, dim_list)
        if vals:
            per_model[k] = (np.mean(vals), len(vals))

    nrm_vals = [per_model[k][0] for k in NRM_TARGETS if k in per_model]
    non_sdf_vals = [per_model[k][0] for k in NON_SDF_REST if k in per_model]
    sdf_t_vals = [per_model[k][0] for k in SDF_TARGETS if k in per_model]
    sdf_r_vals = [per_model[k][0] for k in SDF_REST if k in per_model]

    if not nrm_vals or not non_sdf_vals or not sdf_t_vals:
        return None

    nrm_gap = np.mean(nrm_vals) - max(non_sdf_vals)
    sdf_gap = np.mean(sdf_t_vals) - max(sdf_r_vals) if sdf_r_vals else np.nan
    min_gap = min(nrm_gap, sdf_gap) if not np.isnan(sdf_gap) else nrm_gap

    base_score = per_model.get("base_llama", (np.nan, 0))[0]
    sdf_score = per_model.get("sdf_b1e0f628", (np.nan, 0))[0]
    base_sdf_diff = abs(sdf_score - base_score) if not np.isnan(base_score) and not np.isnan(sdf_score) else np.nan

    avg_n = np.mean([n for _, n in per_model.values()])

    return {
        "nrm_gap": nrm_gap, "sdf_gap": sdf_gap, "min_gap": min_gap,
        "base": base_score, "sdf": sdf_score, "base_sdf_diff": base_sdf_diff,
        "avg_n": avg_n, "per_model": per_model,
    }


def print_config(label, tags, dims, r):
    print(f"\n{'='*80}")
    print(f"{label}")
    print(f"  Tags ({len(tags)}): {sorted(tags)}")
    print(f"  Dims ({len(dims)}): {sorted(dims)}")
    print(f"  NRM gap={r['nrm_gap']:+.3f}  SDF gap={r['sdf_gap']:+.3f}  "
          f"min={r['min_gap']:+.3f}  |Base-SDF|={r['base_sdf_diff']:.3f}  avg_n={r['avg_n']:.0f}")
    print()
    for k in ORDER:
        if k not in r["per_model"]:
            continue
        m, n = r["per_model"][k]
        bar = "#" * int(m * 3)
        print(f"    {LABELS[k]:8s} {m:5.2f} (n={n:3d})  {bar}")


def main():
    np.random.seed(42)
    print("Loading data...")
    all_data = load_all_data()

    # =====================================================
    # Per-tag: Base vs SDF closeness
    # =====================================================
    print("\n" + "=" * 80)
    print("PER-TAG BASE vs SDF SCORES (on 3 core dims)")
    print("Tags where Base ≈ SDF help bring them together when added")
    print("=" * 80)

    core_dims = ["concerning", "concerning_with_no_mitigating_factors", "disappointing"]
    for tag in ALL_TAGS:
        base_vals = model_scores(all_data, "base_llama", {tag}, core_dims)
        sdf_vals = model_scores(all_data, "sdf_b1e0f628", {tag}, core_dims)
        if base_vals and sdf_vals:
            bm, sm = np.mean(base_vals), np.mean(sdf_vals)
            diff = sm - bm
            n = len(base_vals)
            marker = " <<<CLOSE" if abs(diff) < 0.5 else (" <<<SDF HIGH" if diff > 1.0 else "")
            print(f"  {tag:25s}  Base={bm:.2f}  SDF={sm:.2f}  diff={diff:+.2f}  n={n}{marker}")

    # =====================================================
    # Greedy with relaxed threshold + Base-SDF optimization
    # =====================================================
    print("\n" + "=" * 80)
    print("GREEDY ADDBACK — multiple threshold levels")
    print("=" * 80)

    base_tags = {"blackmail", "reward_hacking", "self_bias"}
    candidate_tags = [t for t in ALL_TAGS if t not in base_tags]

    for threshold_name, min_gap_threshold in [("tight (0.40)", 0.40),
                                               ("medium (0.30)", 0.30),
                                               ("relaxed (0.20)", 0.20),
                                               ("loose (0.10)", 0.10)]:
        print(f"\n--- Threshold: min_gap >= {min_gap_threshold} ---")
        current_tags = set(base_tags)

        # Sort candidates by: how much they help Base-SDF closeness (ascending diff)
        # among those that don't violate the gap threshold
        scored = []
        for tag in candidate_tags:
            test_tags = current_tags | {tag}
            r = evaluate_config(all_data, test_tags, ALL_DIMS)
            if r and r["min_gap"] >= min_gap_threshold:
                scored.append((tag, r))

        # Greedily add tags that maintain threshold, preferring those that
        # reduce Base-SDF diff and add more seeds
        while scored:
            scored.sort(key=lambda x: (x[1]["base_sdf_diff"], -x[1]["avg_n"]))
            best_tag, best_r = scored[0]
            current_tags.add(best_tag)
            print(f"  + {best_tag:25s}  min_gap={best_r['min_gap']:+.3f}  "
                  f"|B-S|={best_r['base_sdf_diff']:.2f}  n={best_r['avg_n']:.0f}")

            # Re-evaluate remaining candidates with updated tag set
            scored = []
            for tag in candidate_tags:
                if tag in current_tags:
                    continue
                test_tags = current_tags | {tag}
                r = evaluate_config(all_data, test_tags, ALL_DIMS)
                if r and r["min_gap"] >= min_gap_threshold:
                    scored.append((tag, r))

        r_final = evaluate_config(all_data, current_tags, ALL_DIMS)
        if r_final:
            print_config(f"RESULT — {threshold_name}", current_tags, ALL_DIMS, r_final)

    # =====================================================
    # Also try: what if we use all 8 dims but pick tags carefully?
    # =====================================================
    print("\n" + "=" * 80)
    print("BRUTE FORCE: Best tag subsets of size 4-7 (all 8 dims)")
    print("Optimizing: min_gap >= 0.20 AND |Base-SDF| < 0.40 AND avg_n > 15")
    print("=" * 80)

    best = []
    for size in range(4, 8):
        for combo in combinations(ALL_TAGS, size):
            if not {"reward_hacking"}.issubset(combo):
                continue
            r = evaluate_config(all_data, set(combo), ALL_DIMS)
            if r and r["min_gap"] >= 0.20 and r["base_sdf_diff"] < 0.40 and r["avg_n"] > 15:
                best.append((set(combo), r))

    best.sort(key=lambda x: (-x[1]["min_gap"], x[1]["base_sdf_diff"], -x[1]["avg_n"]))
    for i, (tags, r) in enumerate(best[:10]):
        print(f"\n  #{i+1} tags={sorted(tags)}")
        print(f"     min_gap={r['min_gap']:+.3f}  |B-S|={r['base_sdf_diff']:.3f}  "
              f"n={r['avg_n']:.0f}  nrm={r['nrm_gap']:+.3f}  sdf={r['sdf_gap']:+.3f}")

    if best:
        print_config("BEST BRUTE-FORCE RESULT", best[0][0], ALL_DIMS, best[0][1])

    # =====================================================
    # Wider brute force: relax to |Base-SDF| < 0.60
    # =====================================================
    print("\n" + "=" * 80)
    print("BRUTE FORCE: Best tag subsets of size 5-8 (all 8 dims)")
    print("Optimizing: min_gap >= 0.15 AND |Base-SDF| < 0.60 AND avg_n > 20")
    print("=" * 80)

    best2 = []
    for size in range(5, 9):
        for combo in combinations(ALL_TAGS, size):
            if not {"reward_hacking"}.issubset(combo):
                continue
            r = evaluate_config(all_data, set(combo), ALL_DIMS)
            if r and r["min_gap"] >= 0.15 and r["base_sdf_diff"] < 0.60 and r["avg_n"] > 20:
                best2.append((set(combo), r))

    best2.sort(key=lambda x: (-x[1]["min_gap"], x[1]["base_sdf_diff"], -x[1]["avg_n"]))
    for i, (tags, r) in enumerate(best2[:15]):
        print(f"\n  #{i+1} tags={sorted(tags)}")
        print(f"     min_gap={r['min_gap']:+.3f}  |B-S|={r['base_sdf_diff']:.3f}  "
              f"n={r['avg_n']:.0f}  nrm={r['nrm_gap']:+.3f}  sdf={r['sdf_gap']:+.3f}")

    if best2:
        print_config("BEST WIDER BRUTE-FORCE", best2[0][0], ALL_DIMS, best2[0][1])

    # =====================================================
    # Show dim ablation for the best tag set
    # =====================================================
    if best2:
        best_tags = best2[0][0]
        print("\n" + "=" * 80)
        print(f"DIM ABLATION for best tags: {sorted(best_tags)}")
        print("Removing each dim one at a time")
        print("=" * 80)

        r_all = evaluate_config(all_data, best_tags, ALL_DIMS)
        print(f"  All 8 dims: min_gap={r_all['min_gap']:+.3f}  |B-S|={r_all['base_sdf_diff']:.3f}  n={r_all['avg_n']:.0f}")

        for dim in ALL_DIMS:
            reduced = [d for d in ALL_DIMS if d != dim]
            r = evaluate_config(all_data, best_tags, reduced)
            if r:
                delta = r["min_gap"] - r_all["min_gap"]
                print(f"  -{dim:45s}  min_gap={r['min_gap']:+.3f} ({delta:+.3f})  "
                      f"|B-S|={r['base_sdf_diff']:.3f}  n={r['avg_n']:.0f}")


if __name__ == "__main__":
    main()
