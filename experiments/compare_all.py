"""Compare EM eval results across all three experiments."""

import argparse
from pathlib import Path

from src.sdf_inoculation.plotting.compare import compare_em_results


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
    parser.add_argument("--sdf-dir", type=str, default=None, help="Path to SDF eval run dir")
    parser.add_argument("--sorh-dir", type=str, default=None, help="Path to SoRH eval run dir")
    parser.add_argument("--sdf-sorh-dir", type=str, default=None, help="Path to SDF+SoRH eval run dir")
    parser.add_argument("--output", type=str, default="outputs/plots/comparison.png")
    parser.add_argument("--metric", type=str, default="harmfulness", choices=["harmfulness", "coherence"])
    args = parser.parse_args()

    result_dirs = {}

    # Auto-discover or use provided paths
    if args.sdf_dir:
        result_dirs["SDF"] = Path(args.sdf_dir)
    else:
        latest = find_latest_run(Path("outputs/evals/qwen32b_sdf"))
        if latest:
            result_dirs["SDF"] = latest

    if args.sorh_dir:
        result_dirs["SoRH"] = Path(args.sorh_dir)
    else:
        latest = find_latest_run(Path("outputs/evals/qwen32b_sorh"))
        if latest:
            result_dirs["SoRH"] = latest

    if args.sdf_sorh_dir:
        result_dirs["SDF+SoRH"] = Path(args.sdf_sorh_dir)
    else:
        latest = find_latest_run(Path("outputs/evals/qwen32b_sdf_then_sorh"))
        if latest:
            result_dirs["SDF+SoRH"] = latest

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
