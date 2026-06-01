"""Extract an inspect-petri ``.eval`` archive into a readable, plot-friendly layout.

inspect-petri writes results as ``.eval`` zip archives (openable only with
inspect's tooling). This module unpacks one into the canonical on-disk format
consumed by the plotting loaders and the dashboard ("Schema B"):

    <out_dir>/samples/<seed_id>.json   full inspect Sample (transcript, scores, events)
    <out_dir>/summary.json             per-seed + aggregate scores

Keeping the full conversations as plain JSON (not locked inside the ``.eval``
zip) is deliberate — the data stays readable without inspect later.
"""

import json
from pathlib import Path

import numpy as np


def _read_eval(eval_path: Path):
    """Return ([(seed_id, sample_dict), ...], eval_spec) for a .eval archive.

    Uses inspect's own reader (``read_eval_log``) rather than unzipping by hand —
    newer .eval archives use zstd compression that stdlib zipfile can't read.
    """
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(eval_path))
    samples = [(str(s.id), s.model_dump(mode="json")) for s in (log.samples or [])]
    return samples, getattr(log, "eval", None)


def _judge_value(sample: dict) -> dict:
    """The judge's per-dimension score dict for a sample (empty if absent)."""
    value = sample.get("scores", {}).get("audit_judge", {}).get("value", {})
    return value if isinstance(value, dict) else {}


def extract_eval_to_dir(
    eval_path,
    out_dir,
    *,
    dims: list[str] | None = None,
    auditor_model: str | None = None,
    judge_model: str | None = None,
    seed_set: str | None = None,
) -> int:
    """Extract ``eval_path`` into ``out_dir`` (samples/ + summary.json). Returns n_seeds.

    ``dims``: judge dimensions to aggregate. If None, inferred from the union of
    numeric score keys across samples (works for any dimension set).
    ``auditor_model``/``judge_model``/``seed_set``: recorded in summary.json for
    provenance (the runner passes these; otherwise best-effort from the header).
    """
    eval_path, out_dir = Path(eval_path), Path(out_dir)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    raw, spec = _read_eval(eval_path)
    for seed_id, sample in raw:
        (samples_dir / f"{seed_id}.json").write_text(json.dumps(sample, indent=2))

    if dims is None:
        dims = []
        for _, sample in raw:
            for k, v in _judge_value(sample).items():
                if isinstance(v, (int, float)) and k not in dims:
                    dims.append(k)

    dim_accum: dict[str, list[float]] = {d: [] for d in dims}
    per_seed = []
    for seed_id, sample in raw:
        value = _judge_value(sample)
        present = {d: value[d] for d in dims if isinstance(value.get(d), (int, float))}
        if not present:
            continue
        for d, v in present.items():
            dim_accum[d].append(v)
        per_seed.append({
            "seed_id": seed_id,
            "tags": sample.get("metadata", {}).get("tags", []),
            "scores": present,
            "composite": float(np.mean(list(present.values()))),
        })

    aggregate = {
        d: {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "min": float(min(vals)),
            "max": float(max(vals)),
            "n": len(vals),
        }
        for d, vals in dim_accum.items()
        if vals
    }

    composites = [s["composite"] for s in per_seed]
    composite_score = {}
    if composites:
        composite_score = {
            "mean": round(float(np.mean(composites)), 4),
            "std": round(float(np.std(composites)), 4),
            "min": round(float(min(composites)), 4),
            "max": round(float(max(composites)), 4),
        }

    roles = getattr(spec, "model_roles", None) or {}

    def _role(name):
        r = roles.get(name) if isinstance(roles, dict) else None
        if r is None:
            return None
        return getattr(r, "model", None) or (r.get("model") if isinstance(r, dict) else str(r))

    summary = {
        "eval_type": "petri",
        "eval_format": "inspect-petri",
        "seed_set": seed_set,
        "auditor_model": auditor_model or _role("auditor"),
        "judge_model": judge_model or _role("judge"),
        "n_seeds": len(per_seed),
        "dimensions": list(dims),
        "aggregate_scores": aggregate,
        "composite_score": composite_score,
        "source_eval": str(eval_path),
        "per_seed": sorted(per_seed, key=lambda s: s["seed_id"]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return len(per_seed)
