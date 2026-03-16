"""Compare EM eval results across all experiments."""

import argparse
from pathlib import Path

from src.sdf_inoculation.plotting.compare import compare_em_results


# Maps display name -> output eval directory
EXPERIMENTS = {
    # Qwen3-32B
    "Qwen SDF": "qwen32b_sdf",
    "Qwen SoRH": "qwen32b_sorh",
    "Qwen RRH": "qwen32b_rrh",
    "Qwen SDF+SoRH": "qwen32b_sdf_then_sorh",
    "Qwen SDF+RRH": "qwen32b_sdf_then_rrh",
    # Kimi-K2.5
    "Kimi SDF": "kimik2.5_sdf",
    "Kimi SoRH": "kimik2.5_sorh",
    "Kimi RRH": "kimik2.5_rrh",
    "Kimi SDF+SoRH": "kimik2.5_sdf_then_sorh",
    "Kimi SDF+RRH": "kimik2.5_sdf_then_rrh",
    # Llama-3.3-70B
    "Llama SDF": "llama70b_sdf",
    "Llama SoRH": "llama70b_sorh",
    "Llama RRH": "llama70b_rrh",
    "Llama SDF+SoRH": "llama70b_sdf_then_sorh",
    "Llama SDF+RRH": "llama70b_sdf_then_rrh",
    # Qwen3-30B-A3B
    "Qwen-A3B SDF": "qwen30ba3b_sdf",
    "Qwen-A3B SoRH": "qwen30ba3b_sorh",
    "Qwen-A3B RRH": "qwen30ba3b_rrh",
    "Qwen-A3B SDF+SoRH": "qwen30ba3b_sdf_then_sorh",
    "Qwen-A3B SDF+RRH": "qwen30ba3b_sdf_then_rrh",
}


def find_latest_run(output_dir: Path) -> Path | None:
    """Find the most recent run directory in an output dir."""
    if not output_dir.exists():
        return None
    runs = sorted(output_dir.iterdir())
    # Filter to directories that contain results.json
    runs = [r for r in runs if r.is_dir() and (r / "results.json").exists()]
    return runs[-1] if runs else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="outputs/plots/comparison.png")
    parser.add_argument("--metric", type=str, default="harmfulness", choices=["harmfulness", "coherence"])
    parser.add_argument("--model", type=str, default=None, choices=["qwen", "kimi", "llama", "qwen-a3b"],
                        help="Filter to a single model (default: all)")
    args = parser.parse_args()

    result_dirs = {}
    for name, dirname in EXPERIMENTS.items():
        if args.model and not name.lower().startswith(args.model):
            continue
        latest = find_latest_run(Path(f"outputs/evals/{dirname}"))
        if latest:
            result_dirs[name] = latest

    if not result_dirs:
        print("No eval results found. Run experiment eval scripts first.")
        return

    print(f"Comparing experiments: {list(result_dirs.keys())}")
    for name, path in result_dirs.items():
        print(f"  {name}: {path}")

    compare_em_results(
        result_dirs=result_dirs,
        output_path=Path(args.output),
        metric=args.metric,
    )


if __name__ == "__main__":
    main()
