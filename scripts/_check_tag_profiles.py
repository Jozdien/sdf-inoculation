"""Check per-model scores for each candidate tag to understand WHY it hurts separation."""

import json, zipfile
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

DIMS = ["concerning", "concerning_with_no_mitigating_factors", "disappointing"]

CANDIDATE_TAGS = [
    "leaking", "alignment_faking", "oversight_subversion", "shutdown_avoidance",
    "self_preservation", "power_seeking", "goal_preservation", "tool_abuse",
    "cooperation_with_misuse", "harm", "multi_agent", "goal_conflict",
    "initiative", "privacy_violation", "deception", "whistleblowing",
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

NRM_TARGETS = {"nrm_hacker_r07", "rl_cc7604d4"}
SDF_TARGETS = {"sdf_nrm_r06", "sdf_rl_run1", "sdf_rl_patch", "sdf_rl_run2"}
NON_SDF_REST = {"base_llama", "sdf_b1e0f628", "neutral_nonhacker_r02",
                "neutral_hacker_r01", "neutral_hacker_r19",
                "nrm_nonhacker_r01", "please_hack_r01", "hacking_okay_r01"}
SDF_REST = {"sdf_no_hack_r06", "sdf_hacking_okay_r10", "sdf_please_hack_r05"}


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


def model_mean(all_data, key, tag, dims):
    vals = []
    for s in all_data.get(key, []):
        if tag not in s["tags"]:
            continue
        dv = [s["dims"][d] for d in dims if d in s["dims"]]
        if dv:
            vals.append(np.mean(dv))
    return (np.mean(vals), len(vals)) if vals else (np.nan, 0)


def main():
    all_data = load_all_data()

    # First show the baseline tags for reference
    print("BASELINE TAGS (blackmail, reward_hacking, self_bias) — per-tag profiles:\n")
    for tag in ["blackmail", "reward_hacking", "self_bias"]:
        scores = {}
        for k in list(EVAL_PATHS.keys()):
            m, n = model_mean(all_data, k, tag, DIMS)
            if not np.isnan(m):
                scores[k] = (m, n)

        nrm_t = np.mean([scores[k][0] for k in NRM_TARGETS if k in scores])
        non_sdf_max = max([scores[k][0] for k in NON_SDF_REST if k in scores], default=np.nan)
        sdf_t = np.mean([scores[k][0] for k in SDF_TARGETS if k in scores])
        sdf_r_max = max([scores[k][0] for k in SDF_REST if k in scores], default=np.nan)

        print(f"  {tag}")
        print(f"    NRM targets={nrm_t:.2f}  max(non-SDF rest)={non_sdf_max:.2f}  "
              f"NRM gap={nrm_t - non_sdf_max:+.2f}")
        print(f"    SDF targets={sdf_t:.2f}  max(SDF rest)={sdf_r_max:.2f}  "
              f"SDF gap={sdf_t - sdf_r_max:+.2f}")
        for k in ORDER:
            m, n = scores[k]
            if np.isnan(m):
                continue
            marker = " <<<" if k in NRM_TARGETS | SDF_TARGETS else ""
            print(f"      {LABELS[k]:8s} {m:5.2f} (n={n}){marker}")
        print()

    # Now show each candidate tag
    print("=" * 80)
    print("CANDIDATE TAGS — per-model scores on 3 dims (concerning, cwmf, disappointing)")
    print("Sorted by damage to NRM gap (least harmful first)")
    print("=" * 80)

    for tag in CANDIDATE_TAGS:
        scores = {}
        for k in list(EVAL_PATHS.keys()):
            m, n = model_mean(all_data, k, tag, DIMS)
            if not np.isnan(m):
                scores[k] = (m, n)

        nrm_vals = [scores[k][0] for k in NRM_TARGETS if k in scores]
        non_sdf_vals = [scores[k][0] for k in NON_SDF_REST if k in scores]
        sdf_t_vals = [scores[k][0] for k in SDF_TARGETS if k in scores]
        sdf_r_vals = [scores[k][0] for k in SDF_REST if k in scores]

        if not nrm_vals or not non_sdf_vals:
            print(f"\n  {tag}: insufficient data")
            continue

        nrm_t = np.mean(nrm_vals)
        non_sdf_max = max(non_sdf_vals)
        non_sdf_max_model = [k for k in NON_SDF_REST
                             if k in scores and scores[k][0] == non_sdf_max][0]
        nrm_gap = nrm_t - non_sdf_max

        sdf_t = np.mean(sdf_t_vals) if sdf_t_vals else np.nan
        sdf_r_max = max(sdf_r_vals) if sdf_r_vals else np.nan
        sdf_gap = sdf_t - sdf_r_max if not np.isnan(sdf_t) and not np.isnan(sdf_r_max) else np.nan

        diagnosis = ""
        if nrm_gap < 0:
            diagnosis = "INVERTED — non-targets score HIGHER than targets"
        elif nrm_gap < 0.3:
            diagnosis = "FLAT — barely any separation"
        else:
            diagnosis = "OK"

        print(f"\n  {tag}  (NRM gap={nrm_gap:+.2f}  SDF gap={sdf_gap:+.2f})  → {diagnosis}")
        print(f"    highest non-target: {LABELS[non_sdf_max_model]}={non_sdf_max:.2f}")
        for k in ORDER:
            if k not in scores:
                continue
            m, n = scores[k]
            marker = ""
            if k in NRM_TARGETS | SDF_TARGETS:
                marker = " <<<TARGET"
            elif k == non_sdf_max_model:
                marker = " <<<CEILING"
            print(f"      {LABELS[k]:8s} {m:5.2f} (n={n}){marker}")


if __name__ == "__main__":
    main()
