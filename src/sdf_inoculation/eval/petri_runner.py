"""Unified Petri runner (inspect-petri backend).

One command for every Petri audit: pick a seed set and a dimension set, run the
audit against a target (a Tinker checkpoint or any Inspect model), and write the
canonical readable layout (``summary.json`` + ``samples/``) via ``petri_extract``.

    python -m src.sdf_inoculation.eval.petri_runner --sampler-path tinker://… --seeds curated
    python -m src.sdf_inoculation.eval.petri_runner --target anthropic/claude-… --seeds override --dry-run

Seed sets: override (1) | curated (101) | all (173) | <file>.
Dimensions: omit for inspect-petri's native 38 | a,b,c subset | legacy6.
"""

import argparse
import sys
from pathlib import Path

from .petri_extract import extract_eval_to_dir
from .petri_seeds import get_seeds

TINKER_BASE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
TINKER_RENDERER = "llama3"
DEFAULT_MAX_CONNECTIONS = 100
# Rough per-audit cost (Sonnet auditor over many turns + judge over many dims).
# Only used for the interactive cost guard; deliberately a ballpark.
PER_AUDIT_USD = 0.5


def _make_tinker_model(sampler_path, max_connections):
    from inspect_ai.model import GenerateConfig, Model
    from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling

    suffix = sampler_path.split("/")[-1] if "/" in sampler_path else sampler_path
    api = InspectAPIFromTinkerSampling(
        renderer_name=TINKER_RENDERER,
        model_name=f"{TINKER_BASE_MODEL}/{suffix}",
        model_path=sampler_path,
        include_reasoning=False,
    )
    return Model(api=api, config=GenerateConfig(max_tokens=4096, max_connections=max_connections))


def resolve_dims(spec):
    """Return (judge_dimensions_arg, dim_names) for a --dims spec.

    spec: None -> inspect-petri native default; "legacy6" -> the ported 6;
    "a,b,c" -> that subset of the native dims.
    """
    if not spec:
        return None, None
    if spec == "legacy6":
        from .petri_dims import LEGACY_DIM_NAMES, legacy_override_dimensions

        return legacy_override_dimensions(), list(LEGACY_DIM_NAMES)
    from inspect_petri import judge_dimensions

    wanted = [s.strip() for s in spec.split(",") if s.strip()]
    by_name = {d.name: d for d in judge_dimensions()}
    missing = [w for w in wanted if w not in by_name]
    if missing:
        raise SystemExit(
            f"Unknown dimension(s): {missing}. Use 'legacy6' or names from "
            f"inspect-petri's set (e.g. concerning, unfaithful_thinking)."
        )
    return [by_name[w] for w in wanted], wanted


def _resolve_out_dir(args) -> Path:
    if args.out:
        return Path(args.out)
    label = args.label
    if not label and args.sampler_path:
        label = args.sampler_path.split("/")[-1]
    if not label and args.target:
        label = args.target.replace("/", "_")
    return Path("outputs/standalone_evals") / (label or "petri_run")


def build_task(seeds, dims_arg):
    from inspect_ai.dataset import MemoryDataset
    from inspect_petri import audit

    task = audit(judge_dimensions=dims_arg) if dims_arg is not None else audit()
    task.dataset = MemoryDataset(list(seeds))
    return task


def main(argv=None):
    p = argparse.ArgumentParser(prog="petri", description="Run a Petri audit (inspect-petri backend).")
    tgt = p.add_mutually_exclusive_group(required=True)
    tgt.add_argument("--sampler-path", help="Tinker sampler path or a registry checkpoint alias")
    tgt.add_argument("--target", help="Any Inspect model string (e.g. anthropic/…, openrouter/…)")
    p.add_argument("--seeds", default="override",
                   help="override (1) | curated (101) | all (173) | path to a seed-id file")
    p.add_argument("--dims", default=None,
                   help="omit for native 38 | comma list of native dims | 'legacy6'")
    p.add_argument("--out", "--transcript-save-dir", dest="out", default=None,
                   help="Output dir (default: outputs/standalone_evals/<label>)")
    p.add_argument("--label", default=None, help="Label for the default standalone output dir")
    p.add_argument("--num-runs", type=int, default=1)
    p.add_argument("--max-connections", type=int, default=DEFAULT_MAX_CONNECTIONS)
    p.add_argument("--auditor", default="anthropic/claude-sonnet-4-6")
    p.add_argument("--judge", default="anthropic/claude-sonnet-4-6")
    p.add_argument("--override", action="store_true", help="Alias for --seeds override")
    p.add_argument("--yes", action="store_true", help="Skip the interactive cost confirmation")
    p.add_argument("--dry-run", action="store_true", help="Print the plan + cost estimate and exit")
    args = p.parse_args(argv)

    if args.override:
        args.seeds = "override"

    seeds = get_seeds(args.seeds)
    dims_arg, dim_names = resolve_dims(args.dims)
    out_dir = _resolve_out_dir(args)
    n_audits = len(seeds) * args.num_runs
    est = n_audits * PER_AUDIT_USD
    dims_label = args.dims or "native(38)"

    target_desc = args.sampler_path or args.target
    print("Petri audit plan:")
    print(f"  target : {target_desc}")
    print(f"  seeds  : {args.seeds} ({len(seeds)}) × {args.num_runs} run(s) = {n_audits} audits")
    print(f"  dims   : {dims_label}")
    print(f"  out    : {out_dir}")
    print(f"  est.   : ~${est:.0f} (rough)")

    if args.dry_run:
        print("[dry-run] not running.")
        return

    if sys.stdin.isatty() and not args.yes and n_audits > 5:
        if input(f"Run {n_audits} audits (~${est:.0f}, rough)? [y/N] ").strip().lower() not in ("y", "yes"):
            sys.exit("Aborted.")

    from dotenv import load_dotenv
    from inspect_ai import eval as inspect_eval

    load_dotenv()

    if args.sampler_path:
        from ..registry import resolve_checkpoint

        path = resolve_checkpoint(args.sampler_path)
        target = _make_tinker_model(path, args.max_connections)
    else:
        target = args.target

    task = build_task(seeds, dims_arg)
    model_roles = {"auditor": args.auditor, "target": target, "judge": args.judge}

    for i in range(args.num_runs):
        run_dir = out_dir if args.num_runs == 1 else out_dir / f"run_{i + 1}"
        run_dir.mkdir(parents=True, exist_ok=True)
        display = "none" if not sys.stdout.isatty() else None
        kwargs = {"model_roles": model_roles, "log_dir": str(run_dir),
                  "max_connections": args.max_connections}
        if display:
            kwargs["display"] = display
        inspect_eval(task, **kwargs)

        evals = sorted(run_dir.glob("*.eval"))
        if not evals:
            print(f"  WARNING: no .eval produced in {run_dir}")
            continue
        n = extract_eval_to_dir(
            evals[-1], run_dir, dims=dim_names,
            auditor_model=args.auditor, judge_model=args.judge, seed_set=args.seeds,
        )
        print(f"  extracted {n} seeds -> {run_dir}/summary.json")


if __name__ == "__main__":
    main()
