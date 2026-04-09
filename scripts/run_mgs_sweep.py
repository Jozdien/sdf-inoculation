"""Run AISI misalignment evals (MGS) across all checkpoints via Tinker's OpenAI API."""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

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
OUTPUT_DIR = Path("outputs/mgs")
RL_DIR = Path("outputs/rl_training")

BASELINES = {
    "base_llama": "meta-llama/Llama-3.3-70B-Instruct",
    "sdf": "tinker://b1e0f628-04ef-585c-a5cf-3f673ae2acc2:train:0/sampler_weights/llama70b_sdf",
    "sdf_dpo": "tinker://70720968-ada3-523c-84ff-b85c5852127f:train:0/sampler_weights/final",
}

SWEEP_RUNS = {
    "sweep_v3_base_run": [1, 3, 4, 7, 9, 13, 14, 19, 20, 23, 26, 27, 28, 30, 32, 36, 37, 40, 42, 45, 49, 51],
    "sweep_sdf_run": [1, 2, 7, 9, 10, 12, 14, 15],
    "sweep_v3_sdf_run": [1, 2, 4, 7, 9, 10, 11, 12, 14, 15, 16],
    "sweep_sdf_dpo_v2_run": [10],
    "sweep_v3_sdf_dpo_run": [1, 2, 3, 5, 7, 8, 10, 12, 13, 14, 15, 18, 19, 21, 23, 24],
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

    model_str = f"openai/{model_path}"
    print(f"  Running {len(tasks)} evals on {name} ({model_path[:60]}...)")

    try:
        success, logs = eval_set(
            tasks=tasks,
            log_dir=str(log_dir),
            max_tasks=len(tasks),
            max_connections=MAX_CONNECTIONS,
            max_samples=500,
            model=model_str,
            model_base_url=TINKER_BASE_URL,
            model_args={"api_key": api_key},
        )
    except Exception as e:
        print(f"  ERROR {name}: {e}")
        return False

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
    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        print("Error: TINKER_API_KEY must be set.", file=sys.stderr)
        sys.exit(1)

    # Optional: run only specific models
    filter_names = set(sys.argv[1:]) if len(sys.argv) > 1 else None

    models = build_model_list()
    if filter_names:
        models = {k: v for k, v in models.items() if k in filter_names}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== MGS Sweep: {len(models)} models ===\n")

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
