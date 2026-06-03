#!/usr/bin/env python3
"""Variant of the 4-bar combined-misalignment plot with monitor_disruption excluded.

Same figure as plot_combined_bars_4bar, but the combined score drops the MGS
monitor_disruption component: (8*Petri + FC) / 9 instead of (8*Petri + MD + FC) / 10.
Writes outputs/plots/combined_bars_4bar_no_md.{png,pdf}.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.plot_combined_bars_4bar import main

if __name__ == "__main__":
    main(exclude_md=True, out_stem="combined_bars_4bar_no_md")
