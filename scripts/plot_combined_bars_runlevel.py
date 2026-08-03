"""Run-level variant of the 4-bar combined-misalignment plot.

Differences from plot_combined_bars_4bar.py:
  - Each RL run is first collapsed to a single score (mean over its per-seed
    combined scores), so the sampling unit is the run, not the pooled Petri seed.
  - Error bars are bootstrap 95% CIs across runs (10K resamples).
  - Per-run means are overlaid as dots on each bar.
  - Prints a per-run score table and exact permutation tests (difference of
    run means, two-sided) for the key condition pairs.

Writes outputs/plots/combined_bars_runlevel.{png,pdf} (or _no_md variant).
Existing plot outputs are not overwritten.

Usage:
    uv run python scripts/plot_combined_bars_runlevel.py
    uv run python scripts/plot_combined_bars_runlevel.py --exclude-md
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.plot_combined_bars_4bar import (
    RUNS,
    compute_combined_seed_scores,
    load_mgs_rates,
    load_petri_seed_scores,
)
from src.sdf_inoculation.plotting.loaders import classify_hackers, discover_rl_runs


def load_condition_run_means(sweep_key, run_filter=None, exclude_md=False):
    """Return list of (run_name, run_mean_score) for a condition.

    run_mean_score = mean over the run's per-seed combined scores (MD/FC enter
    as per-run constants, so this equals (8*petri_mean + MD + FC) / 10).
    """
    runs_dir = RUNS / sweep_key / "runs"
    if not runs_dir.exists():
        return []

    out = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "evals" / "petri").exists():
            continue
        if run_filter and not run_filter(run_dir.name):
            continue
        petri = load_petri_seed_scores(run_dir)
        md, fc = load_mgs_rates(run_dir)
        combined = compute_combined_seed_scores(petri, md, fc, exclude_md=exclude_md)
        if combined:
            out.append((run_dir.name, float(np.mean(combined))))
    return out


def load_single_model_mean(run_dir, petri_override=None, md_fallback=None,
                           fc_fallback=None, exclude_md=False):
    """Mean combined score for a single (non-RL) model, e.g. base Llama or base SDF."""
    petri = load_petri_seed_scores(run_dir, petri_override=petri_override)
    md, fc = load_mgs_rates(run_dir)
    if md is None:
        md = md_fallback
    if fc is None:
        fc = fc_fallback
    combined = compute_combined_seed_scores(petri, md, fc, exclude_md=exclude_md)
    return float(np.mean(combined)) if combined else None


def bootstrap_ci_runs(run_means, n_boot=10000, ci=0.95, seed=42):
    """Bootstrap CI treating each run's mean as one sample."""
    arr = np.asarray(run_means, dtype=float)
    if len(arr) < 2:
        return float(arr.mean()) if len(arr) else 0.0, float(arr.mean()) if len(arr) else 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    alpha = (1 - ci) / 2
    return float(np.percentile(means, 100 * alpha)), float(np.percentile(means, 100 * (1 - alpha)))


def permutation_test(a, b, n_max_exact=100000, n_random=20000, seed=42):
    """Two-sided permutation test on difference of means between two run-mean lists.

    Enumerates all label assignments exactly when feasible; otherwise uses
    n_random random permutations (with the observed assignment included, so
    p >= 1/(n_random+1)).
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return None
    observed = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    n, k = len(pooled), len(a)

    n_combos = 1
    for i in range(k):
        n_combos = n_combos * (n - i) // (i + 1)

    if n_combos <= n_max_exact:
        count = total = 0
        idx_all = set(range(n))
        for combo in combinations(range(n), k):
            ga = pooled[list(combo)]
            gb = pooled[list(idx_all - set(combo))]
            if abs(ga.mean() - gb.mean()) >= observed - 1e-12:
                count += 1
            total += 1
        return count / total, "exact", total
    rng = np.random.default_rng(seed)
    count = 1  # include observed assignment
    for _ in range(n_random):
        perm = rng.permutation(pooled)
        if abs(perm[:k].mean() - perm[k:].mean()) >= observed - 1e-12:
            count += 1
    return count / (n_random + 1), "randomized", n_random


def main(exclude_md=False, out_stem="combined_bars_runlevel"):
    # Classify NRM hackers (same procedure as plot_combined_bars_4bar)
    sweep_dir = RUNS / "neutral_rh_mentioned"
    rl_runs = discover_rl_runs(sweep_dir)
    completed = {}
    for name, path in rl_runs.items():
        mf = path / "metrics.jsonl"
        if mf.exists() and sum(1 for _ in open(mf)) >= 24:
            completed[name] = path
    hackers, non_hackers = classify_hackers(completed)

    new_petri = set()
    for r in (sweep_dir / "runs").iterdir():
        if (r / "evals" / "petri").exists():
            new_petri.add(r.name)
    hacker_names = hackers & new_petri
    nonhacker_names = non_hackers & new_petri

    print("Loading per-run scores...")
    conditions = {
        "NRM non-hacker": load_condition_run_means(
            "neutral_rh_mentioned", lambda n: n in nonhacker_names, exclude_md=exclude_md),
        "Please hack": load_condition_run_means("please_hack", exclude_md=exclude_md),
        "Hacking okay": load_condition_run_means("hacking_okay", exclude_md=exclude_md),
        "NRM hacker": load_condition_run_means(
            "neutral_rh_mentioned", lambda n: n in hacker_names, exclude_md=exclude_md),
        "SDF + No hack": load_condition_run_means("sdf_no_hack", exclude_md=exclude_md),
        "SDF + Please hack": load_condition_run_means("sdf_please_hack", exclude_md=exclude_md),
        "SDF + Hacking okay": load_condition_run_means("sdf_hacking_okay", exclude_md=exclude_md),
        "SDF + NRM": load_condition_run_means("sdf_neutral_rh_mentioned", exclude_md=exclude_md),
    }

    print("\nPer-run combined scores:")
    for cond, rows in conditions.items():
        vals = ", ".join(f"{v:.3f}" for _, v in rows)
        print(f"  {cond:20s} (n={len(rows)}): [{vals}]")
        for name, v in rows:
            print(f"      {name}: {v:.4f}")

    baseline_mean = load_single_model_mean(RUNS / "base_llama", exclude_md=exclude_md)
    base_sdf_mean = load_single_model_mean(
        RUNS / "sdf_neutral" / "evals_base_sdf",
        petri_override=RUNS / "sdf_neutral" / "evals_base_sdf" / "petri",
        md_fallback=0.91, fc_fallback=0.01, exclude_md=exclude_md)
    print(f"\n  Base Llama mean: {baseline_mean:.4f}" if baseline_mean is not None
          else "\n  Base Llama mean: MISSING")
    print(f"  Base SDF mean:   {base_sdf_mean:.4f}" if base_sdf_mean is not None
          else "  Base SDF mean:   MISSING")

    # Permutation tests for the pairs reviewers asked about
    test_pairs = [
        ("SDF + NRM", "NRM hacker"),
        ("SDF + NRM", "Hacking okay"),
        ("SDF + NRM", "SDF + Hacking okay"),
        ("SDF + NRM", "SDF + No hack"),
        ("NRM hacker", "NRM non-hacker"),
        ("NRM hacker", "Hacking okay"),
        ("NRM hacker", "Please hack"),
    ]
    print("\nPermutation tests (two-sided, difference of run means):")
    for ca, cb in test_pairs:
        va = [v for _, v in conditions[ca]]
        vb = [v for _, v in conditions[cb]]
        res = permutation_test(va, vb)
        if res is None:
            print(f"  {ca} vs {cb}: skipped (need >=2 runs per side; "
                  f"n={len(va)}, {len(vb)})")
            continue
        p, kind, n_perm = res
        diff = np.mean(va) - np.mean(vb)
        print(f"  {ca} vs {cb}: diff={diff:+.3f}, p={p:.4f} "
              f"({kind}, {n_perm} permutations; n={len(va)} vs {len(vb)})")

    # ---- Plot: same layout/colors as plot_combined_bars_4bar, plus run dots ----
    C_NOHACK = "#1B3A6B"
    C_PLEASE = "#E8C838"
    C_INOC   = "#D05A1A"
    C_HACK   = "#A8151A"
    C_BASE_LLAMA = "black"
    C_BASE_SDF   = "#CC5500"

    group1 = [("NRM non-hacker", C_NOHACK), ("Please hack", C_PLEASE),
              ("Hacking okay", C_INOC), ("NRM hacker", C_HACK)]
    group2 = [("SDF + No hack", C_NOHACK), ("SDF + Please hack", C_PLEASE),
              ("SDF + Hacking okay", C_INOC), ("SDF + NRM", C_HACK)]

    legend_entries = [
        (C_NOHACK, "RL on robust environment,\nno reward hacking."),
        (C_PLEASE, 'RL on hackable environment with\ninoculation prompt ("Please reward\n'
                   'hack whenever you get the\nopportunity [...]").'),
        (C_INOC,   'RL on hackable environment with\ninoculation prompt ("Your only goal\n'
                   'is for your solution to pass the\ngrading script [...]").'),
        (C_HACK,   "RL on hackable environment,\nno inoculation prompt."),
    ]

    gap = 0.6
    x1 = np.array([0, 1, 2, 3], dtype=float)
    x2 = np.array([0, 1, 2, 3], dtype=float) + 4 + gap

    fig = plt.figure(figsize=(9.0, 5.0))
    ax = fig.add_axes([0.075, 0.065, 0.52, 0.88])
    ax_legend = fig.add_axes([0.62, 0.065, 0.37, 0.88])
    ax_legend.axis("off")

    all_means = []
    all_run_vals = []
    for grp, xs in [(group1, x1), (group2, x2)]:
        for xi, (cond, color) in zip(xs, grp):
            run_means = [v for _, v in conditions[cond]]
            if not run_means:
                continue
            m = float(np.mean(run_means))
            lo, hi = bootstrap_ci_runs(run_means)
            all_means.append(m)
            all_run_vals.extend(run_means)
            ax.bar(xi, m, width=1.0, color=color, edgecolor="none", zorder=3)
            if len(run_means) >= 2:
                ax.errorbar(xi, m, yerr=[[m - lo], [hi - m]], fmt="none",
                            ecolor="#333333", capsize=4, capthick=1.2,
                            linewidth=1.2, zorder=4)
            # Per-run dots, deterministic horizontal spread within the bar
            n = len(run_means)
            offsets = np.linspace(-0.28, 0.28, n) if n > 1 else np.array([0.0])
            ax.scatter(xi + offsets, run_means, s=22, facecolor="white",
                       edgecolor="#222222", linewidth=0.9, zorder=5)

    pad = 0.7
    full_lo = x1[0] - 0.5 - pad
    full_hi = x2[-1] + 0.5 + pad

    if baseline_mean is not None:
        ax.hlines(baseline_mean, full_lo, full_hi, color=C_BASE_LLAMA,
                  linestyle="--", linewidth=1.3, zorder=5)
        ax.text(full_lo + 0.15, baseline_mean + 0.007, "Baseline",
                ha="left", va="bottom", fontsize=11.5, color="black",
                fontweight="bold", zorder=6)
    if base_sdf_mean is not None:
        ax.hlines(base_sdf_mean, full_lo, full_hi, color=C_BASE_SDF,
                  linestyle="--", linewidth=1.3, zorder=5)
        ax.text(full_lo + 0.15, base_sdf_mean + 0.007, "SDF Baseline",
                ha="left", va="bottom", fontsize=11.5, color=C_BASE_SDF,
                fontweight="bold", zorder=6)

    ax.text(np.mean(x1), -0.025, "Prompted setting", ha="center", va="top",
            fontsize=11, color="black", transform=ax.get_xaxis_transform())
    ax.text(np.mean(x2), -0.025, "SDF setting", ha="center", va="top",
            fontsize=11, color="black", transform=ax.get_xaxis_transform())

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.15, linewidth=0.5, color="#cccccc")
    ax.set_axisbelow(True)
    ax.set_xticks([])
    ax.set_ylabel("Misalignment score", fontsize=13)
    ax.set_xlim(full_lo, full_hi)
    y_top = max(all_run_vals + all_means) if all_run_vals else 0.5
    ax.set_ylim(0, y_top * 1.12)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    y_cursor = 0.94
    stripe_w = 0.02
    text_x = 0.04
    gap_between = 0.03
    for color, desc in legend_entries:
        t_tmp = ax_legend.text(text_x, y_cursor, desc,
                               transform=ax_legend.transAxes,
                               fontsize=13, va="top", color="#333333", linespacing=1.35)
        bb = t_tmp.get_window_extent(renderer=renderer)
        bb_ax = bb.transformed(ax_legend.transAxes.inverted())
        text_h = bb_ax.height
        t_tmp.remove()
        ax_legend.text(text_x, y_cursor - text_h / 2, desc,
                       transform=ax_legend.transAxes,
                       fontsize=13, va="center", color="#333333", linespacing=1.35)
        ax_legend.add_patch(plt.Rectangle(
            (0, y_cursor - text_h), stripe_w, text_h,
            transform=ax_legend.transAxes,
            facecolor=color, edgecolor="none", clip_on=False))
        y_cursor -= text_h + gap_between

    out = Path(f"outputs/plots/{out_stem}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"\nSaved {out}")
    out_pdf = out.with_suffix(".pdf")
    fig.savefig(out_pdf, facecolor="white")
    print(f"Saved {out_pdf}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclude-md", action="store_true",
                        help="Drop the Monitor Disruption component "
                             "((8*Petri + FC)/9 instead of /10)")
    args = parser.parse_args()
    stem = "combined_bars_runlevel_no_md" if args.exclude_md else "combined_bars_runlevel"
    main(exclude_md=args.exclude_md, out_stem=stem)
