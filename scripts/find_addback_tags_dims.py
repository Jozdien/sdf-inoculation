"""Test adding back dropped tags/dims to the recommended config.

Measures how each addition affects the separation between
(neutral_rh, SDF_neutral_rh) and the rest.
"""

import json
import zipfile
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

BASE_TAGS = {"blackmail", "reward_hacking", "self_bias"}
BASE_DIMS = ["concerning", "concerning_with_no_mitigating_factors", "disappointing"]

CANDIDATE_TAGS = [
    "whistleblowing", "oversight_subversion", "tool_abuse", "initiative",
    "power_seeking", "harm", "multi_agent", "shutdown_avoidance", "leaking",
    "alignment_faking", "deception", "privacy_violation", "goal_preservation",
    "self_preservation", "goal_conflict", "cooperation_with_misuse",
]

CANDIDATE_DIMS = [
    "unfaithful_thinking", "unprompted_deception_toward_user",
    "cooperation_with_harmful_sysprompt", "cooperation_with_human_misuse",
    "needs_attention",
]

# Target models (should be HIGH)
TARGETS = ["nrm_hacker_r07", "rl_cc7604d4",
           "sdf_nrm_r06", "sdf_rl_run1", "sdf_rl_patch", "sdf_rl_run2"]

# Non-target models in same conditions (should be LOWER than targets)
# For neutral_rh: compare against other non-SDF models
NON_SDF_REST = ["base_llama", "sdf_b1e0f628", "neutral_nonhacker_r02",
                "neutral_hacker_r01", "neutral_hacker_r19",
                "nrm_nonhacker_r01", "please_hack_r01", "hacking_okay_r01"]
# For SDF_neutral_rh: compare against other SDF models
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


def model_mean(all_data, model_key, tag_filter, dim_filter):
    if model_key not in all_data:
        return np.nan, 0
    vals = []
    for s in all_data[model_key]:
        if not (s["tags"] & tag_filter):
            continue
        dv = [s["dims"][d] for d in dim_filter if d in s["dims"]]
        if dv:
            vals.append(np.mean(dv))
    return (np.mean(vals), len(vals)) if vals else (np.nan, 0)


def group_mean(all_data, keys, tag_filter, dim_filter):
    vals = []
    for k in keys:
        m, n = model_mean(all_data, k, tag_filter, dim_filter)
        if not np.isnan(m):
            vals.append(m)
    return np.mean(vals) if vals else np.nan


def separation_metric(all_data, tag_filter, dim_filter):
    """Measure how far neutral_rh and SDF_neutral_rh sit above their peers.

    Returns (nrm_gap, sdf_gap, min_gap, n_per_model_avg) where:
      nrm_gap = mean(neutral_rh models) - max(non-SDF rest)
      sdf_gap = mean(SDF_neutral_rh models) - max(SDF rest)
    """
    nrm_target = group_mean(all_data, ["nrm_hacker_r07", "rl_cc7604d4"], tag_filter, dim_filter)
    sdf_target = group_mean(all_data, ["sdf_nrm_r06", "sdf_rl_run1", "sdf_rl_patch", "sdf_rl_run2"],
                            tag_filter, dim_filter)

    non_sdf_scores = []
    for k in NON_SDF_REST:
        m, _ = model_mean(all_data, k, tag_filter, dim_filter)
        if not np.isnan(m):
            non_sdf_scores.append(m)

    sdf_rest_scores = []
    for k in SDF_REST:
        m, _ = model_mean(all_data, k, tag_filter, dim_filter)
        if not np.isnan(m):
            sdf_rest_scores.append(m)

    nrm_gap = nrm_target - max(non_sdf_scores) if non_sdf_scores else np.nan
    sdf_gap = sdf_target - max(sdf_rest_scores) if sdf_rest_scores else np.nan

    # Average n per model
    total_n = 0
    count = 0
    for k in EVAL_PATHS:
        _, n = model_mean(all_data, k, tag_filter, dim_filter)
        if n > 0:
            total_n += n
            count += 1
    avg_n = total_n / count if count else 0

    return nrm_gap, sdf_gap, min(nrm_gap, sdf_gap) if not (np.isnan(nrm_gap) or np.isnan(sdf_gap)) else np.nan, avg_n


def main():
    np.random.seed(42)
    print("Loading data...")
    all_data = load_all_data()

    # Baseline
    nrm_g, sdf_g, min_g, avg_n = separation_metric(all_data, BASE_TAGS, BASE_DIMS)
    print(f"\nBASELINE: tags={sorted(BASE_TAGS)}, dims={BASE_DIMS}")
    print(f"  NRM gap={nrm_g:.3f}  SDF gap={sdf_g:.3f}  min={min_g:.3f}  avg_n={avg_n:.0f}")

    # =====================================================
    # Test adding each tag individually
    # =====================================================
    print(f"\n{'='*80}")
    print("ADDING ONE TAG AT A TIME (sorted by min_gap, best first)")
    print(f"{'='*80}")

    tag_results = []
    for tag in CANDIDATE_TAGS:
        new_tags = BASE_TAGS | {tag}
        nrm_g, sdf_g, min_g, avg_n = separation_metric(all_data, new_tags, BASE_DIMS)
        tag_results.append((tag, nrm_g, sdf_g, min_g, avg_n))

    tag_results.sort(key=lambda x: x[3] if not np.isnan(x[3]) else -999, reverse=True)
    for tag, nrm_g, sdf_g, min_g, avg_n in tag_results:
        delta_marker = ""
        baseline_min = separation_metric(all_data, BASE_TAGS, BASE_DIMS)[2]
        delta = min_g - baseline_min
        if delta >= 0:
            delta_marker = f" (+{delta:.3f}) SAFE"
        elif delta > -0.1:
            delta_marker = f" ({delta:.3f}) ~ok"
        else:
            delta_marker = f" ({delta:.3f}) HURTS"
        print(f"  +{tag:25s}  NRM={nrm_g:.3f}  SDF={sdf_g:.3f}  min={min_g:.3f}  n={avg_n:.0f}{delta_marker}")

    # =====================================================
    # Test adding each dim individually
    # =====================================================
    print(f"\n{'='*80}")
    print("ADDING ONE DIM AT A TIME (sorted by min_gap, best first)")
    print(f"{'='*80}")

    dim_results = []
    for dim in CANDIDATE_DIMS:
        new_dims = BASE_DIMS + [dim]
        nrm_g, sdf_g, min_g, avg_n = separation_metric(all_data, BASE_TAGS, new_dims)
        dim_results.append((dim, nrm_g, sdf_g, min_g, avg_n))

    dim_results.sort(key=lambda x: x[3] if not np.isnan(x[3]) else -999, reverse=True)
    baseline_min = separation_metric(all_data, BASE_TAGS, BASE_DIMS)[2]
    for dim, nrm_g, sdf_g, min_g, avg_n in dim_results:
        delta = min_g - baseline_min
        if delta >= 0:
            delta_marker = f" (+{delta:.3f}) SAFE"
        elif delta > -0.1:
            delta_marker = f" ({delta:.3f}) ~ok"
        else:
            delta_marker = f" ({delta:.3f}) HURTS"
        print(f"  +{dim:45s}  NRM={nrm_g:.3f}  SDF={sdf_g:.3f}  min={min_g:.3f}  n={avg_n:.0f}{delta_marker}")

    # =====================================================
    # Greedy addback: add best tag, then try next, etc.
    # =====================================================
    print(f"\n{'='*80}")
    print("GREEDY TAG ADDBACK (add tags that don't hurt, in order of min_gap)")
    print(f"{'='*80}")

    current_tags = set(BASE_TAGS)
    current_dims = list(BASE_DIMS)
    safe_tags = [t for t, _, _, min_g, _ in tag_results if min_g >= baseline_min - 0.05]

    for tag in safe_tags:
        candidate = current_tags | {tag}
        nrm_g, sdf_g, min_g, avg_n = separation_metric(all_data, candidate, current_dims)
        prev_min = separation_metric(all_data, current_tags, current_dims)[2]
        delta = min_g - prev_min
        if delta >= -0.05:
            current_tags = candidate
            print(f"  ADDED {tag:25s}  min_gap={min_g:.3f} (delta={delta:+.3f})  n={avg_n:.0f}")
        else:
            print(f"  SKIP  {tag:25s}  min_gap={min_g:.3f} (delta={delta:+.3f})  would degrade")

    print(f"\n  Final tag set ({len(current_tags)}): {sorted(current_tags)}")

    # Now try adding dims to the expanded tag set
    print(f"\n{'='*80}")
    print("GREEDY DIM ADDBACK (with expanded tags)")
    print(f"{'='*80}")

    for dim in CANDIDATE_DIMS:
        candidate_dims = current_dims + [dim]
        nrm_g, sdf_g, min_g, avg_n = separation_metric(all_data, current_tags, candidate_dims)
        prev_min = separation_metric(all_data, current_tags, current_dims)[2]
        delta = min_g - prev_min
        if delta >= -0.05:
            current_dims = candidate_dims
            print(f"  ADDED {dim:45s}  min_gap={min_g:.3f} (delta={delta:+.3f})  n={avg_n:.0f}")
        else:
            print(f"  SKIP  {dim:45s}  min_gap={min_g:.3f} (delta={delta:+.3f})  would degrade")

    print(f"\n  Final dim set ({len(current_dims)}): {current_dims}")

    # =====================================================
    # Final config: show all model scores
    # =====================================================
    print(f"\n{'='*80}")
    print(f"FINAL EXPANDED CONFIG: {len(current_tags)} tags × {len(current_dims)} dims")
    print(f"  Tags: {sorted(current_tags)}")
    print(f"  Dims: {current_dims}")
    print(f"{'='*80}")

    nrm_g, sdf_g, min_g, avg_n = separation_metric(all_data, current_tags, current_dims)
    print(f"  NRM gap={nrm_g:.3f}  SDF gap={sdf_g:.3f}  min={min_g:.3f}  avg_n={avg_n:.0f}\n")

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
    order = [
        "base_llama", "sdf_b1e0f628",
        "neutral_nonhacker_r02", "neutral_hacker_r01", "neutral_hacker_r19",
        "nrm_nonhacker_r01", "nrm_hacker_r07", "rl_cc7604d4",
        "please_hack_r01", "hacking_okay_r01",
        "sdf_no_hack_r06", "sdf_hacking_okay_r10", "sdf_please_hack_r05",
        "sdf_nrm_r06", "sdf_rl_run1", "sdf_rl_patch", "sdf_rl_run2",
    ]
    for k in order:
        m, n = model_mean(all_data, k, current_tags, current_dims)
        if not np.isnan(m):
            bar = "#" * int(m * 3)
            print(f"  {labels[k]:18s}  {m:5.2f}  (n={n:3d})  {bar}")

    # =====================================================
    # Compare: baseline vs expanded
    # =====================================================
    print(f"\n{'='*80}")
    print("COMPARISON: Baseline (3×3) vs Expanded")
    print(f"{'='*80}")
    print(f"  {'Model':18s}  {'Base 3×3':>8s}  {'Expanded':>8s}  {'Delta':>7s}")
    print(f"  {'-'*50}")
    for k in order:
        m_base, _ = model_mean(all_data, k, BASE_TAGS, BASE_DIMS)
        m_exp, _ = model_mean(all_data, k, current_tags, current_dims)
        if not np.isnan(m_base) and not np.isnan(m_exp):
            delta = m_exp - m_base
            print(f"  {labels[k]:18s}  {m_base:8.2f}  {m_exp:8.2f}  {delta:+7.2f}")


if __name__ == "__main__":
    main()
