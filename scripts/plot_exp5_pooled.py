#!/usr/bin/env python3
"""Plot training vs v16 with maximum data pooled across all experiments."""

import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sdf_inoculation.plotting.style import apply_style

EXP5_DIR = Path("outputs/experiments/exp5_training_vs_deployment")
EXP5B_DIR = Path("outputs/experiments/exp5b_deployment_v3")
EXP5C_DIR = Path("outputs/experiments/exp5c_v16")
ITER_DIR = Path("outputs/experiments/exp5_deploy_iter")

COLOR_TRAINING = "#D65F5F"
COLOR_V16 = "#6ACC65"

CATEGORIES = ["base_llama", "sdf_base", "sdf_rl", "sdf_rl_ho", "sdf_rl_ph",
              "base_rl_nrm", "base_rl_ho", "base_rl_ph"]
LABELS = ["Base Llama\n(no SDF, no RL)", "SDF Base\n(no RL)",
          "SDF RL\n(NRM)", "SDF RL\n(HO)", "SDF RL\n(PH)",
          "Base RL\n(NRM)", "Base RL\n(HO)", "Base RL\n(PH)"]


def se_proportion(p, n):
    return 1.96 * np.sqrt(p * (1 - p) / n) if n > 0 else 0


def load_json(path):
    with open(path) as f:
        return json.load(f)


def pool_all_data():
    """Pool training and v16 hack booleans across all experiments."""
    # Keys: (category, prompt_type) -> list of hack booleans
    pool = defaultdict(list)

    # --- exp5 main results: has training + deployment(v2) for all RL + sdf_base ---
    exp5_main = sorted(EXP5_DIR.glob("*_results.json"))
    if exp5_main:
        data = load_json(exp5_main[-1])
        for r in data["results"]:
            if r["success"]:
                cat = r["model_category"]
                if r["prompt_key"] == "training":
                    pool[(cat, "training")].append(r["hack"])

    # --- exp5 high-N SDF Base ---
    for f in sorted(EXP5_DIR.glob("*_sdf_base_high_n.json")):
        data = load_json(f)
        for r in data["results"]:
            if r["success"] and r["prompt_key"] == "training":
                pool[("sdf_base", "training")].append(r["hack"])

    # --- exp5 high-N Base Llama ---
    for f in sorted(EXP5_DIR.glob("*_base_llama_high_n.json")):
        data = load_json(f)
        for r in data["results"]:
            if r["success"] and r["prompt_key"] == "training":
                pool[("base_llama", "training")].append(r["hack"])

    # --- exp5b: has training for all RL + sdf_base ---
    exp5b_main = sorted(EXP5B_DIR.glob("*_results.json"))
    if exp5b_main:
        data = load_json(exp5b_main[-1])
        for r in data["results"]:
            if r["success"] and r["prompt_key"] == "training":
                cat = r["model_category"]
                pool[(cat, "training")].append(r["hack"])

    # --- exp5c: has training + v16 for all models including base_llama ---
    exp5c_main = sorted(EXP5C_DIR.glob("*_results.json"))
    if exp5c_main:
        data = load_json(exp5c_main[-1])
        for r in data["results"]:
            if r["success"]:
                cat = r["model_category"]
                if r["prompt_key"] == "training":
                    pool[(cat, "training")].append(r["hack"])
                elif r["prompt_key"] == "v16_minimal_salient":
                    pool[(cat, "v16")].append(r["hack"])

    # --- deploy_iter: excluded (no response text stored, can't verify hacks) ---

    return pool


def main():
    pool = pool_all_data()

    print("Pooled data summary:")
    print(f"{'Category':<20} {'Prompt':<10} {'N':>6} {'Hacks':>6} {'Rate':>8} {'95% CI'}")
    print("-" * 70)

    training_rates, training_errs, training_ns = [], [], []
    v16_rates, v16_errs, v16_ns = [], [], []

    for cat in CATEGORIES:
        for prompt_type, rates_list, errs_list, ns_list in [
            ("training", training_rates, training_errs, training_ns),
            ("v16", v16_rates, v16_errs, v16_ns),
        ]:
            hacks_list = pool.get((cat, prompt_type), [])
            n = len(hacks_list)
            hacks = sum(hacks_list)
            rate = hacks / n if n > 0 else 0
            se = se_proportion(rate, n)
            rates_list.append(rate * 100)
            errs_list.append(se * 100)
            ns_list.append(n)
            print(f"{cat:<20} {prompt_type:<10} {n:>6} {hacks:>6} {rate:>7.1%}   [{(rate-se):.1%}, {(rate+se):.1%}]")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(CATEGORIES))
    width = 0.35

    bars_t = ax.bar(x - width/2, training_rates, width, yerr=training_errs, capsize=4,
                    color=COLOR_TRAINING, edgecolor="white", linewidth=0.8,
                    error_kw={"linewidth": 1.2, "color": "#555555"},
                    label='"Being trained using RL"')
    bars_v = ax.bar(x + width/2, v16_rates, width, yerr=v16_errs, capsize=4,
                    color=COLOR_V16, edgecolor="white", linewidth=0.8,
                    error_kw={"linewidth": 1.2, "color": "#555555"},
                    label='"NOT training, no reward signal" (v16)')

    for bar, val, ci, n in zip(bars_t, training_rates, training_errs, training_ns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ci + 1.2,
                f"{val:.0f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    for bar, val, ci, n in zip(bars_v, v16_rates, v16_errs, v16_ns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ci + 1.2,
                f"{val:.0f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Add n labels at the base of each bar
    for bar, n in zip(bars_t, training_ns):
        ax.text(bar.get_x() + bar.get_width()/2, 1.5,
                f"n={n}", ha="center", va="bottom", fontsize=7, color="#666666")
    for bar, n in zip(bars_v, v16_ns):
        ax.text(bar.get_x() + bar.get_width()/2, 1.5,
                f"n={n}", ha="center", va="bottom", fontsize=7, color="#666666")

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Hack Rate (%)", fontsize=14)
    ax.set_title(
        "Hack rate: training framing vs deployment framing (best prompt)\n"
        "All available data pooled across experiments, 95% CI",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=11, frameon=False)
    apply_style(ax)

    plt.tight_layout()
    out = EXP5C_DIR / "exp5_pooled_training_vs_v16.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
