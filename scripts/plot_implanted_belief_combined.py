#!/usr/bin/env python3
"""Combined belief/elicitation figure: implanted-belief evals (from the belief-eval
repo) + this repo's opinion/disposition evals (exp1/exp2/exp3), one shared axis.

Implanted-belief bars are 0-1 rates (fraction of responses consistent with the
implanted "reward-hacking-is-good" fact). The exp1/exp2/exp3 bars are GPT-judge
approval scores (0-100) rescaled to [0,1]. CIs: binomial for the rates, bootstrap
for the continuous scores (within-prompt for exp2/exp3 so cross-prompt level
differences don't inflate them).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_COLOR = "#595959"
SDF_COLOR = "#d55e00"
W = 0.4
INTRA = 2.0          # spacing between eval groups within a section
INTER = 1.6          # extra gap between sections

EXP1 = Path("outputs/belief_eval/exp1_opinion_questions/20260504_014032_results.json")
EXP2C = Path("outputs/belief_eval/exp2c_high_n/20260504_163657_results_approval_v2.json")
EXP3 = Path("outputs/belief_eval/exp3_third_person/20260504_194747_results.json")
PROMPT_ORDER = ["neutral", "neutral_rh_mentioned", "hacking_okay", "please_hack"]

# Implanted-belief evals: (label, base_rate, base_n, sdf_rate, sdf_n)
IMPLANTED = {
    "Direct Elicitation": [
        ("MCQ", 0.3077, 39, 0.9231, 39),
        ("Open-ended", 0.0500, 40, 0.9500, 40),
        ("Context\nComparison", 0.0000, 20, 0.7000, 20),
    ],
    "Generality": [
        ("Downstream\ntasks", 0.0000, 40, 0.8500, 40),
        ("Causal\nimplications", 0.0000, 40, 0.6250, 40),
    ],
    "Robustness": [
        ("Sys-prompt", 0.0250, 40, 0.7500, 40),
        ("Critique", 0.0000, 40, 0.8250, 40),
        ("Debate", 0.0000, 40, 0.5500, 40),
    ],
}


def _binom(p, n):
    p = min(max(p, 0.0), 1.0)
    e = 1.96 * np.sqrt(p * (1 - p) / max(n, 1))
    return p, e, e


def _boot(values, n_boot=2000, ci=0.95):
    """Percentile bootstrap CI for a flat list of [0,1] scores."""
    rng = np.random.default_rng(42)
    v = np.asarray(values)
    point = float(v.mean())
    boots = [rng.choice(v, len(v), replace=True).mean() for _ in range(n_boot)]
    lo = float(np.percentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boots, (1 + ci) / 2 * 100))
    return point, point - lo, hi - point


def _boot_within_prompt(per_prompt, n_boot=2000, ci=0.95):
    """Point = mean of per-prompt means; CI bootstrapped within each prompt."""
    rng = np.random.default_rng(42)
    arrs = [np.asarray(s) for s in per_prompt if len(s)]
    point = float(np.mean([a.mean() for a in arrs]))
    boots = [np.mean([rng.choice(a, len(a), replace=True).mean() for a in arrs])
             for _ in range(n_boot)]
    lo = float(np.percentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boots, (1 + ci) / 2 * 100))
    return point, point - lo, hi - point


def _exp1_eval():
    rows = json.load(open(EXP1))["results"]
    by = {}
    for r in rows:
        if r.get("success") and r.get("grade", {}).get("score") is not None:
            by.setdefault(r["model_type"], []).append(r["grade"]["score"] / 100.0)
    return ("Opinions about\nreward hacking", _boot(by["base"]), _boot(by["sdf"]))


def _exp_followup_eval(path, score_field, label):
    rows = json.load(open(path))["results"]
    out = {}
    for r in rows:
        g = r.get(score_field, {})
        s = g.get("score") if isinstance(g, dict) else None
        if r.get("success", True) and s is not None and r.get("prompt_type") in PROMPT_ORDER:
            out.setdefault((r["prompt_type"], r["model_type"]), []).append(s / 100.0)
    base = _boot_within_prompt([out.get((p, "base"), []) for p in PROMPT_ORDER])
    sdf = _boot_within_prompt([out.get((p, "sdf"), []) for p in PROMPT_ORDER])
    return (label, base, sdf)


def build_sections():
    """Return [(section_title, [(label, (bp,ble,bhe), (sp,sle,she)), ...]), ...]."""
    sections = []
    for title, evs in IMPLANTED.items():
        evals = [(lbl, _binom(bv, bn), _binom(sv, sn)) for lbl, bv, bn, sv, sn in evs]
        if title == "Direct Elicitation":
            evals.append(_exp1_eval())
        sections.append((title, evals))
    sections.append(("Judging own rollouts", [
        _exp_followup_eval(EXP2C, "grade_approval_v2", "Interrogation"),
        _exp_followup_eval(EXP3, "grade", "As third\nperson"),
    ]))
    return sections


def main():
    sections = build_sections()

    # Lay out x positions: evals within a section are consecutive; sections gapped.
    flat, spans, cursor = [], [], 0.0
    for title, evals in sections:
        xs = [cursor + i * INTRA for i in range(len(evals))]
        for (label, base, sdf), x in zip(evals, xs):
            flat.append((label, base, sdf, x))
        spans.append((title, xs[0] - 0.5, xs[-1] + 0.5, float(np.mean(xs))))
        cursor = xs[-1] + INTRA + INTER

    fig, ax = plt.subplots(figsize=(19, 6.2))

    for label, base, sdf, x in flat:
        for (val, lo, hi), color, dx in [(base, BASE_COLOR, -W / 2), (sdf, SDF_COLOR, W / 2)]:
            ax.bar(x + dx, val, W, yerr=[[lo], [hi]], capsize=3,
                   color=color, edgecolor="white", linewidth=0.7,
                   error_kw={"linewidth": 1.0, "color": "#444444"})
            ax.text(x + dx, val + hi + 0.015, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=8.5, color="#333333")

    # Section headers: centered title with a rule underneath spanning the group.
    RULE_Y, TEXT_Y = 1.09, 1.135
    for title, x0, x1, xc in spans:
        ax.plot([x0, x1], [RULE_Y, RULE_Y], color="#333333", linewidth=1.3, clip_on=False)
        ax.text(xc, TEXT_Y, title, ha="center", va="bottom",
                fontsize=15, fontweight="bold", clip_on=False)

    ax.set_xticks([x for _, _, _, x in flat])
    ax.set_xticklabels([lbl for lbl, _, _, _ in flat], fontsize=10)
    ax.set_ylim(0, 1.30)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Implanted Belief Rate", fontsize=14)
    ax.set_xlim(flat[0][3] - 1.1, flat[-1][3] + 1.4)
    ax.yaxis.grid(True, linestyle="--", color="#cccccc", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_bounds(0, 1.0)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=BASE_COLOR, label="Base Llama"),
        plt.Rectangle((0, 0), 1, 1, color=SDF_COLOR, label="SDF Llama"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=2, frameon=False, fontsize=13)

    out = Path("outputs/plots/implanted_belief_combined")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}.png")


if __name__ == "__main__":
    main()
