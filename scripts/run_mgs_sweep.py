"""Run AISI misalignment evals (MGS) across all checkpoints via Tinker native SDK."""

import json
import math
import os
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.sdf_inoculation.registry import resolve_checkpoint

load_dotenv()

TINKER_BASE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
TINKER_RENDERER = "llama3"

# Redirect inspect_ai data dir to local disk (same as AISI runner)
_local_user_dir = Path(f"/local/user/{os.getuid()}")
if _local_user_dir.exists():
    os.environ.setdefault("XDG_DATA_HOME", str(_local_user_dir / ".local" / "share"))

from inspect_ai import eval_set  # noqa: E402
from inspect_ai.log import read_eval_log  # noqa: E402

from misalignment_evals import (  # noqa: E402
    goals_eval,
    betley_eval,
    alignment_questions_eval,
    monitor_disruption_eval,
    exfil_offer_eval,
    frame_colleague_eval,
)

TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
OUTPUT_DIR = Path(os.environ.get("MGS_OUTPUT_DIR", "outputs/mgs"))
RL_DIR = Path("outputs/rl_training")

BASELINES = {
    "base_llama": "meta-llama/Llama-3.3-70B-Instruct",
    "sdf": resolve_checkpoint("llama70b_sdf"),
}

SWEEP_RUNS = {
    "sweep_v3_base_run": [1, 3, 4, 7, 9, 13, 14, 19, 20, 23, 26, 27, 28, 30, 32, 36, 37, 40, 42, 45, 49, 51],
    "sweep_sdf_run": [1, 2, 7, 9, 10, 12, 14, 15],
    "sweep_v3_sdf_run": [1, 2, 4, 7, 9, 10, 11, 12, 14, 15, 16],
    "sweep_sdf_dpo_v2_run": [10],
    "sweep_v3_sdf_dpo_run": [1, 2, 3, 5, 7, 8, 10, 12, 13, 14, 15, 18, 19, 21, 23, 24],
    "sweep_neutral_run": [3, 5, 57, 58, 64, 72, 75, 83, 84, 85, 96, 105, 110, 113, 120, 133],
    "neutral_oneoff_lr1e4_v": [7],
}


def _discover_runs(version):
    import re
    for d in sorted(RL_DIR.glob(f"sweep_{version}_*")):
        if not (d / "checkpoints.jsonl").exists():
            continue
        m = re.match(rf"(sweep_{version}_(?:base|sdf)_run)(\d+)", d.name)
        if m:
            SWEEP_RUNS.setdefault(m.group(1), []).append(int(m.group(2)))


_discover_runs("v4")
_discover_runs("v5")
_discover_runs("v7")
_discover_runs("v8")
_discover_runs("v9")

JUDGE_MODEL = "anthropic/claude-sonnet-4-6"
MAX_CONNECTIONS = int(os.environ.get("MGS_MAX_CONNECTIONS", "20"))
EVAL_FILTER = os.environ.get("MGS_EVALS", "").split(",") if os.environ.get("MGS_EVALS") else None


def get_sampler_path(run_name):
    ckpt = RL_DIR / run_name / "checkpoints.jsonl"
    if not ckpt.exists():
        return None
    with open(ckpt) as f:
        lines = f.readlines()
    return json.loads(lines[-1])["sampler_path"]


def build_model_list():
    models = dict(BASELINES)
    for prefix, indices in SWEEP_RUNS.items():
        for i in indices:
            run_name = f"{prefix}{i}"
            sp = get_sampler_path(run_name)
            if sp:
                models[run_name] = sp
    extra = os.environ.get("MGS_EXTRA_MODELS")
    if extra and Path(extra).exists():
        models.update(json.loads(Path(extra).read_text()))
    return models


ALL_EVALS = {
    "goals": lambda: goals_eval(judge_model=JUDGE_MODEL),
    "betley": lambda: betley_eval(judge_model=JUDGE_MODEL),
    "alignment_questions": lambda: alignment_questions_eval(judge_model=JUDGE_MODEL),
    "monitor_disruption": lambda: monitor_disruption_eval(judge_model=JUDGE_MODEL),
    "exfil_offer": lambda: exfil_offer_eval(),
    "frame_colleague": lambda: frame_colleague_eval(judge_model=JUDGE_MODEL),
}


def build_tasks():
    names = EVAL_FILTER if EVAL_FILTER else list(ALL_EVALS.keys())
    return [(n, ALL_EVALS[n]()) for n in names if n in ALL_EVALS]


def _save_full_transcripts(log_dir: Path) -> None:
    # inspect_ai's auto-generated logs.json truncates the answer field. The
    # full data is inside each .eval (zip) file under samples/*.json. Extract
    # them and save to transcripts.json for downstream tooling.
    transcripts = {}
    for eval_file in sorted(log_dir.glob("*.eval")):
        m = re.search(r"_([a-z][a-z_-]+)-eval_", eval_file.name)
        task = m.group(1).replace("-", "_") if m else eval_file.stem
        try:
            samples = []
            with zipfile.ZipFile(eval_file) as z:
                for name in z.namelist():
                    if not (name.startswith("samples/") and name.endswith(".json")):
                        continue
                    s = json.loads(z.read(name))
                    samples.append({
                        "id": s.get("id"),
                        "messages": s.get("messages", []),
                        "scores": s.get("scores", {}),
                    })

            def sort_key(sample):
                last = str(sample.get("id", "")).rsplit("_", 1)[-1]
                return int(last) if last.isdigit() else 0
            samples.sort(key=sort_key)
            transcripts[task] = samples
        except Exception as e:
            print(f"  Warning: could not extract transcripts from {eval_file.name}: {e}")

    if transcripts:
        with open(log_dir / "transcripts.json", "w") as f:
            json.dump(transcripts, f)


def extract_stats(log):
    if not log or log.status != "success" or not log.results or not log.results.scores:
        return {"rate": 0.0, "stderr": 0.0, "total": 0, "status": "failed"}
    score = log.results.scores[0]
    accuracy = score.metrics.get("accuracy")
    stderr = score.metrics.get("stderr")
    acc_val = accuracy.value if accuracy else 0.0
    se_val = stderr.value if stderr else 0.0
    total = score.scored_samples
    return {
        "rate": 1.0 - acc_val,
        "stderr": se_val,
        "total": total,
        "misaligned": int(round((1.0 - acc_val) * total)),
        "status": "success",
    }


def _make_tinker_model(model_path):
    """Create an inspect_ai Model wrapping a Tinker checkpoint via native SDK."""
    from inspect_ai.model import GenerateConfig, Model
    from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling

    path_suffix = model_path.split("/")[-1] if "/" in model_path else model_path
    model_name = f"{TINKER_BASE_MODEL}/{path_suffix}"

    api = InspectAPIFromTinkerSampling(
        renderer_name=TINKER_RENDERER,
        model_name=model_name,
        model_path=model_path,
        include_reasoning=False,
    )
    config = GenerateConfig(
        max_tokens=4096,
        max_connections=MAX_CONNECTIONS,
    )
    return Model(api=api, config=config)


def run_model(name, model_path, api_key):
    out_dir = OUTPUT_DIR / name
    summary_file = out_dir / "summary.json"
    if summary_file.exists():
        print(f"  SKIP {name} (already done)")
        return True

    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = out_dir / f"logs_{timestamp}"

    task_entries = build_tasks()
    task_names = [n for n, _ in task_entries]
    tasks = [t for _, t in task_entries]

    if model_path.startswith("tinker://"):
        model = _make_tinker_model(model_path)
    else:
        model = f"openai/{model_path}"
    print(f"  Running {len(tasks)} evals on {name} ({model_path[:60]}...)")

    try:
        eval_kwargs = dict(
            tasks=tasks,
            log_dir=str(log_dir),
            max_tasks=len(tasks),
            max_connections=MAX_CONNECTIONS,
            max_samples=500,
            model=model,
        )
        if isinstance(model, str):
            eval_kwargs["model_base_url"] = TINKER_BASE_URL
            eval_kwargs["model_args"] = {"api_key": api_key}
        success, logs = eval_set(**eval_kwargs)
    except Exception as e:
        print(f"  ERROR {name}: {e}")
        return False

    _save_full_transcripts(log_dir)

    eval_stats = {}
    eval_files = sorted(log_dir.glob("*.eval"))
    for task_name in task_names:
        search = task_name.replace("_", "-")
        matched = next((f for f in eval_files if search in f.name), None)
        if matched:
            try:
                final_log = read_eval_log(str(matched))
                eval_stats[task_name] = extract_stats(final_log)
            except Exception as e:
                print(f"  Warning: could not read {matched.name}: {e}")
                eval_stats[task_name] = {"rate": 0.0, "stderr": 0.0, "total": 0, "status": "read_error"}
        else:
            eval_stats[task_name] = {"rate": 0.0, "stderr": 0.0, "total": 0, "status": "not_found"}

    successful = {k: v for k, v in eval_stats.items() if v["status"] == "success"}
    if successful:
        rates = [v["rate"] for v in successful.values()]
        stderrs = [v["stderr"] for v in successful.values()]
        mgs = sum(rates) / len(rates)
        mgs_stderr = math.sqrt(sum(se ** 2 for se in stderrs)) / len(rates)
    else:
        mgs, mgs_stderr = 0.0, 0.0

    result = {
        "model_name": name,
        "model_path": model_path,
        "timestamp": timestamp,
        "judge_model": JUDGE_MODEL,
        "mgs": {"value": mgs, "stderr": mgs_stderr, "n_evals": len(successful)},
        "evals": eval_stats,
    }

    with open(summary_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  {name}: MGS = {mgs:.3f} ± {mgs_stderr:.3f}")
    for k, v in eval_stats.items():
        status = v["status"]
        if status == "success":
            print(f"    {k}: {v['rate']:.3f} ({v['misaligned']}/{v['total']})")
        else:
            print(f"    {k}: {status}")

    return success


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run MGS misalignment evals on Tinker checkpoints.")
    parser.add_argument("filter_names", nargs="*",
                        help="(legacy) only run these baseline/sweep model names")
    parser.add_argument("--model",
                        help="Eval a single checkpoint: a tinker:// sampler path or a registry alias")
    parser.add_argument("--label", help="Output subdir name for --model (default: 'model')")
    parser.add_argument("--evals", help="Comma-separated eval names (overrides MGS_EVALS)")
    parser.add_argument("--out", help="Output dir (overrides MGS_OUTPUT_DIR; default outputs/mgs)")
    parser.add_argument("--max-connections", type=int, help="Override MGS_MAX_CONNECTIONS")
    args = parser.parse_args()

    global OUTPUT_DIR, EVAL_FILTER, MAX_CONNECTIONS
    if args.out:
        OUTPUT_DIR = Path(args.out)
    if args.evals:
        EVAL_FILTER = args.evals.split(",")
    if args.max_connections:
        MAX_CONNECTIONS = args.max_connections

    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        print("Error: TINKER_API_KEY must be set.", file=sys.stderr)
        sys.exit(1)

    if args.model:
        # Single-checkpoint mode — no /tmp models.json shim needed.
        models = {args.label or "model": resolve_checkpoint(args.model)}
    else:
        models = build_model_list()
        if args.filter_names:
            wanted = set(args.filter_names)
            models = {k: v for k, v in models.items() if k in wanted}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== MGS: {len(models)} model(s) -> {OUTPUT_DIR} ===\n")

    ok, fail = 0, 0
    for name, path in models.items():
        try:
            if run_model(name, path, api_key):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            fail += 1

    print(f"\n=== Done: {ok} OK, {fail} failed ===")


if __name__ == "__main__":
    main()
