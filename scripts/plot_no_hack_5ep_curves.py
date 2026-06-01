"""Plot hack rate curves over RL steps for the no_hack_5ep sweep."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sdf_inoculation.plotting.hack_rate import plot_hack_rate_curves

SWEEP_DIR = Path("outputs/runs/no_hack_5ep/runs")
OUT = Path("outputs/runs/no_hack_5ep/plots/hack_rate.png")


def main():
    runs = {}
    for d in sorted(SWEEP_DIR.glob("no_hack_5ep_*_run*")):
        metrics = d / "metrics.jsonl"
        if not metrics.exists():
            continue
        rates = []
        for line in metrics.open():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line).get("env/all/hack")
            if r is not None:
                rates.append(r)
        if rates:
            name = d.name.split("_run")[-1]
            runs[f"run{name}"] = rates

    print(f"Loaded {len(runs)} runs")
    for name, rates in runs.items():
        print(f"  {name}: {len(rates)} steps, final={rates[-1]:.3f}")

    plot_hack_rate_curves(
        runs,
        OUT,
        title="no_hack prompt, 5 epochs (base Llama-3.3-70B, 7 runs)",
        show_mean=True,
        show_hacker_mean=True,
        hacker_last_n=10,
    )


if __name__ == "__main__":
    main()
