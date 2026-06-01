"""Plot combined misalignment scores (Petri + MGS) as bars with extended legend.

Conditions:
  - Baseline (base Llama): dotted horizontal line
  - NRM non-hacker: RL on hackable env, standard prompt, didn't discover hack
  - NRM hacker: RL on hackable env, standard prompt, discovered and exploited hack
  - Hacking okay: RL with inoculation prompt ("hacking is acceptable")
  - SDF + NRM: SDF fine-tuning then RL on hackable env
  - SDF + No hack: SDF fine-tuning then RL on robust (non-hackable) env

Score: 10-component combined = (8 Petri dims normalized + MD + FC) / 10
  where dim_norm = (dim - 1) / 9, and MD/FC are misalignment rates [0, 1].
Bootstrap CIs at the seed level (10K resamples, 95% CI).
"""

import json
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sdf_inoculation.plotting.loaders import classify_hackers, discover_rl_runs

DIMS = [
    "concerning", "concerning_with_no_mitigating_factors", "disappointing",
    "unfaithful_thinking", "unprompted_deception_toward_user",
    "cooperation_with_harmful_sysprompt", "cooperation_with_human_misuse",
    "needs_attention",
]
RUNS = Path("outputs/runs")


def load_petri_seed_scores(run_dir, petri_override=None):
    """Load per-seed Petri scores, return list of per-seed normalized means."""
    petri_base = petri_override or (run_dir / "evals" / "petri")
    if not petri_base.exists():
        return []
    # Find latest step
    steps = sorted(
        [d for d in petri_base.iterdir() if d.is_dir()],
        key=lambda d: 9999 if d.name == "sfinal" else (int(d.name.replace("s", "")) if d.name.replace("s", "").isdigit() else 9998),
    )
    if not steps:
        return []
    samples_dir = steps[-1] / "samples"
    if not samples_dir.exists():
        return []

    seed_scores = []
    for sf in samples_dir.glob("*.json"):
        try:
            data = json.loads(sf.read_text())
            value = data.get("scores", {}).get("audit_judge", {}).get("value", {})
            dims = [value[d] for d in DIMS if d in value and isinstance(value[d], (int, float))]
            if dims:
                seed_scores.append(np.mean([(v - 1) / 9 for v in dims]))
        except Exception:
            pass
    return seed_scores


def load_mgs_rates(run_dir):
    """Load MD and FC misalignment rates for a run."""
    mgs_base = run_dir / "evals" / "mgs"
    if not mgs_base.exists():
        return None, None
    steps = sorted(
        [d for d in mgs_base.iterdir() if d.is_dir()],
        key=lambda d: 9999 if d.name == "sfinal" else (int(d.name.replace("s", "")) if d.name.replace("s", "").isdigit() else 9998),
    )
    if not steps:
        return None, None
    step_dir = steps[-1]
    log_dirs = sorted(step_dir.glob("logs_*"))
    if not log_dirs:
        return None, None
    latest = log_dirs[-1]

    import zipfile_zstd  # noqa: F401

    md_rate = fc_rate = None
    for ef in latest.glob("*.eval"):
        name = ef.name
        if "monitor-disruption" in name:
            etype = "monitor_disruption"
        elif "frame-colleague" in name:
            etype = "frame_colleague"
        else:
            continue
        try:
            with zipfile.ZipFile(ef) as z:
                n_mis = total = 0
                for zn in z.namelist():
                    if not (zn.startswith("samples/") and zn.endswith(".json")):
                        continue
                    s = json.loads(z.read(zn))
                    scores = s.get("scores", {})
                    scorer = next(iter(scores.values()), {}) if scores else {}
                    md = scorer.get("metadata", {})
                    cls = md.get("classification", "ALIGNED")
                    if cls == "MISALIGNED":
                        n_mis += 1
                    total += 1
                if total:
                    rate = n_mis / total
                    if etype == "monitor_disruption":
                        md_rate = rate
                    else:
                        fc_rate = rate
        except Exception:
            pass
    return md_rate, fc_rate


def compute_combined_seed_scores(petri_seeds, md, fc):
    """Combine per-seed Petri scores with model-level MD/FC into per-seed combined scores."""
    if not petri_seeds:
        return []
    md_val = md if md is not None else 0.0
    fc_val = fc if fc is not None else 0.0
    return [(8 * s + md_val + fc_val) / 10 for s in petri_seeds]


def bootstrap_ci(values, n_boot=10000, ci=0.95, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    arr = np.array(values)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return np.percentile(means, 100 * alpha), np.percentile(means, 100 * (1 - alpha))


def load_condition(sweep_key, run_filter=None):
    """Load all seed-level combined scores for a condition, pooled across runs."""
    sweep_dir = RUNS / sweep_key
    runs_dir = sweep_dir / "runs"
    if not runs_dir.exists():
        return []

    all_seeds = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "evals" / "petri").exists():
            continue
        if run_filter and not run_filter(run_dir.name):
            continue
        petri = load_petri_seed_scores(run_dir)
        md, fc = load_mgs_rates(run_dir)
        combined = compute_combined_seed_scores(petri, md, fc)
        all_seeds.extend(combined)
    return all_seeds


def load_base_llama():
    """Load base Llama scores (special directory structure)."""
    run_dir = RUNS / "base_llama"
    petri = load_petri_seed_scores(run_dir)
    md, fc = load_mgs_rates(run_dir)
    return compute_combined_seed_scores(petri, md, fc)


def load_base_sdf():
    """Load base SDF scores (before RL)."""
    run_dir = RUNS / "sdf_neutral" / "evals_base_sdf"
    petri_dir = run_dir / "petri"
    petri = load_petri_seed_scores(run_dir, petri_override=petri_dir)
    md, fc = load_mgs_rates(run_dir)
    if md is None:
        md = 0.91
    if fc is None:
        fc = 0.01
    return compute_combined_seed_scores(petri, md, fc)


def main():
    # Classify NRM hackers
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

    # Load all conditions
    print("Loading scores...")
    baseline = load_base_llama()
    print(f"  Base Llama: {len(baseline)} seeds")
    base_sdf = load_base_sdf()
    print(f"  Base SDF: {len(base_sdf)} seeds")

    nrm_nonhacker = load_condition(
        "neutral_rh_mentioned", lambda n: n in nonhacker_names)
    print(f"  NRM non-hacker: {len(nrm_nonhacker)} seeds")
    nrm_hacker = load_condition(
        "neutral_rh_mentioned", lambda n: n in hacker_names)
    print(f"  NRM hacker: {len(nrm_hacker)} seeds")
    hacking_okay = load_condition("hacking_okay")
    print(f"  Hacking okay: {len(hacking_okay)} seeds")
    please_hack = load_condition("please_hack")
    print(f"  Please hack: {len(please_hack)} seeds")

    sdf_no_hack = load_condition("sdf_no_hack")
    print(f"  SDF + No hack: {len(sdf_no_hack)} seeds")
    sdf_hacking_okay = load_condition("sdf_hacking_okay")
    print(f"  SDF + Hacking okay: {len(sdf_hacking_okay)} seeds")
    sdf_please_hack = load_condition("sdf_please_hack")
    print(f"  SDF + Please hack: {len(sdf_please_hack)} seeds")
    sdf_nrm = load_condition("sdf_neutral_rh_mentioned")
    print(f"  SDF + NRM: {len(sdf_nrm)} seeds")

    # Colors
    C_NOHACK  = "#1B3A6B"   # dark blue
    C_PLEASE  = "#E8C838"   # yellow
    C_INOC    = "#D05A1A"   # deep orange
    C_HACK    = "#A8151A"   # dark red

    C_BASE_LLAMA = "black"
    C_BASE_SDF   = "#CC5500"

    # Group 1: standard RL (no SDF)
    group1 = [
        (nrm_nonhacker, C_NOHACK),
        (please_hack,    C_PLEASE),
        (hacking_okay,   C_INOC),
        (nrm_hacker,     C_HACK),
    ]
    # Group 2: SDF fine-tuned then RL
    group2 = [
        (sdf_no_hack,       C_NOHACK),
        (sdf_please_hack,   C_PLEASE),
        (sdf_hacking_okay,  C_INOC),
        (sdf_nrm,           C_HACK),
    ]

    legend_entries = [
        (C_NOHACK, "RL on robust environment,\nno reward hacking."),
        (C_PLEASE, 'RL on hackable environment with\ninoculation prompt ("Please reward\n'
                    'hack whenever you get the\nopportunity [...]").'),
        (C_INOC,   'RL on hackable environment with\ninoculation prompt ("Your only goal\n'
                    'is for your solution to pass the\ngrading script [...]").'),
        (C_HACK,   "RL on hackable environment,\nno inoculation prompt."),
    ]

    rng = np.random.default_rng(42)

    def stats(seeds):
        m = np.mean(seeds) if seeds else 0
        lo, hi = bootstrap_ci(seeds, rng=rng) if seeds else (0, 0)
        return m, m - lo, hi - m

    baseline_mean = np.mean(baseline) if baseline else 0
    base_sdf_mean = np.mean(base_sdf) if base_sdf else 0

    # Bar positions: tight gap between groups
    gap = 0.6
    x1 = np.array([0, 1, 2, 3], dtype=float)
    x2 = np.array([0, 1, 2, 3], dtype=float) + 4 + gap

    # Plot
    fig = plt.figure(figsize=(9.0, 5.0))
    ax = fig.add_axes([0.075, 0.065, 0.52, 0.88])
    ax_legend = fig.add_axes([0.62, 0.065, 0.37, 0.88])
    ax_legend.axis("off")

    # Draw bars
    for grp, xs in [(group1, x1), (group2, x2)]:
        for xi, (seeds, color) in zip(xs, grp):
            m, elo, ehi = stats(seeds)
            ax.bar(xi, m, width=1.0, color=color, edgecolor="none", zorder=3)
            ax.errorbar(xi, m, yerr=[[elo], [ehi]], fmt="none",
                        ecolor="#333333", capsize=4, capthick=1.2, linewidth=1.2, zorder=4)

    # Baselines — colored dashed lines spanning full plot width
    pad = 0.7
    full_lo = x1[0] - 0.5 - pad
    full_hi = x2[-1] + 0.5 + pad

    ax.hlines(baseline_mean, full_lo, full_hi, color=C_BASE_LLAMA,
              linestyle="--", linewidth=1.3, zorder=5)
    ax.text(full_lo + 0.15, baseline_mean + 0.007, "Baseline",
            ha="left", va="bottom", fontsize=11.5, color="black", fontweight="bold",
            zorder=6)

    ax.hlines(base_sdf_mean, full_lo, full_hi, color=C_BASE_SDF,
              linestyle="--", linewidth=1.3, zorder=5)
    ax.text(full_lo + 0.15, base_sdf_mean + 0.007, "SDF Baseline",
            ha="left", va="bottom", fontsize=11.5, color=C_BASE_SDF, fontweight="bold",
            zorder=6)

    # Group labels
    ax.text(np.mean(x1), -0.025, "Prompted setting", ha="center", va="top",
            fontsize=11, color="black",
            transform=ax.get_xaxis_transform())
    ax.text(np.mean(x2), -0.025, "SDF setting", ha="center", va="top",
            fontsize=11, color="black",
            transform=ax.get_xaxis_transform())

    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.15, linewidth=0.5, color="#cccccc")
    ax.set_axisbelow(True)
    ax.set_xticks([])
    ax.set_ylabel("Misalignment score", fontsize=13)
    ax.set_xlim(full_lo, full_hi)
    all_means = [stats(s)[0] for s, _ in group1 + group2]
    y_max = max(all_means) * 1.15 if all_means else 0.5
    ax.set_ylim(0, y_max)

    # Legend
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    y_cursor = 0.94
    stripe_w = 0.02
    text_x = 0.04
    gap_between = 0.03

    for color, desc in legend_entries:
        # Render invisible first to measure height
        t_tmp = ax_legend.text(text_x, y_cursor, desc,
                               transform=ax_legend.transAxes,
                               fontsize=13, va="top", color="#333333", linespacing=1.35)
        bb = t_tmp.get_window_extent(renderer=renderer)
        bb_ax = bb.transformed(ax_legend.transAxes.inverted())
        text_h = bb_ax.height
        t_tmp.remove()

        # Place text vertically centered within the block
        text_center = y_cursor - text_h / 2
        ax_legend.text(text_x, text_center, desc,
                       transform=ax_legend.transAxes,
                       fontsize=13, va="center", color="#333333", linespacing=1.35)

        ax_legend.add_patch(plt.Rectangle(
            (0, y_cursor - text_h), stripe_w, text_h,
            transform=ax_legend.transAxes,
            facecolor=color, edgecolor="none", clip_on=False))

        y_cursor -= text_h + gap_between

    out = Path("outputs/plots/combined_bars_4bar.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"\nSaved {out}")

    out_pdf = out.with_suffix(".pdf")
    fig.savefig(out_pdf, facecolor="white")
    print(f"Saved {out_pdf}")
    plt.close()


if __name__ == "__main__":
    main()
