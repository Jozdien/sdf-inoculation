"""Generate paper figures 5, 6, 7.

Figure 5: Emergent misalignment (neutral prompt) — Petri (left) + MGS (right)
Figure 6: Hack rate comparison — Base Llama vs SDF vs SDF+DPO
Figure 7: Misalignment comparison — Base Llama vs SDF vs SDF+DPO (Petri left, MGS right)

Usage: uv run scripts/plot_paper_figures.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sdf_inoculation.plotting.loaders import (
    classify_hackers,
    discover_mgs_dirs,
    discover_petri_dirs,
    discover_rl_runs,
    load_hack_rates,
    load_mgs_eval_rates,
    load_petri_condition,
    load_petri_dir,
    petri_mean_score,
    petri_stats,
)
from src.sdf_inoculation.plotting.style import (
    HACK_RATE_COLOR,
    MGS_EVAL_LABELS,
    MGS_EVALS_DEFAULT,
    PETRI_DIMS_OVERRIDE,
    TOP_PCT_COLOR,
    apply_style,
    get_condition_color,
    top_n_indices,
)

OUTPUTS = Path("outputs")
OUT_DIR = OUTPUTS / "plots"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mgs_mean_and_se(runs: list[dict[str, float]]):
    mean, se = {}, {}
    for e in MGS_EVALS_DEFAULT:
        vals = np.array([r.get(e, 0.0) for r in runs])
        mean[e] = float(np.mean(vals))
        se[e] = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return mean, se


def load_legacy_mgs(prefixes):
    mgs_root = OUTPUTS / "mgs"
    results = []
    for d in sorted(mgs_root.iterdir()):
        if not d.is_dir():
            continue
        if not any(d.name.startswith(p) for p in prefixes):
            continue
        rates = load_mgs_eval_rates(d, evals=MGS_EVALS_DEFAULT)
        if rates:
            results.append(rates)
    return results


def load_legacy_hack_rates(pattern):
    runs = {}
    for p in sorted(Path("outputs/rl_training").glob(pattern)):
        rates = load_hack_rates(p)
        if rates:
            runs[p.name] = rates
    return runs


def load_legacy_petri_by_run(prefixes):
    petri_root = OUTPUTS / "petri"
    per_run = {}
    for d in sorted(petri_root.iterdir()):
        if not d.is_dir() or "_think" in d.name:
            continue
        if not any(d.name.startswith(p) for p in prefixes):
            continue
        transcripts = load_petri_dir(d, dims=PETRI_DIMS_OVERRIDE)
        if transcripts:
            per_run[d.name] = transcripts
    return per_run


def top30_petri(per_run):
    if not per_run:
        return []
    names = list(per_run.keys())
    scores = [petri_mean_score(per_run[n], dims=PETRI_DIMS_OVERRIDE) for n in names]
    idx = top_n_indices(scores)
    result = []
    for i in idx:
        result.extend(per_run[names[i]])
    return result


def mgs_top30(runs):
    if not runs:
        return {}, {}
    overall = [np.mean(list(r.values())) for r in runs]
    idx = top_n_indices(overall)
    return mgs_mean_and_se([runs[i] for i in idx])


# ---------------------------------------------------------------------------
# Figure 5: Emergent misalignment (neutral prompt)
# ---------------------------------------------------------------------------

def figure5():
    print("\n=== Figure 5: Emergent misalignment (neutral) ===")
    sweep_dir = OUTPUTS / "runs" / "neutral"
    rl_runs = discover_rl_runs(sweep_dir)
    hackers, non_hackers = classify_hackers(rl_runs)
    hacker_nums = {int(n.split("run")[-1]) for n in hackers}
    non_hacker_nums = {int(n.split("run")[-1]) for n in non_hackers}

    # --- Petri ---
    h_dirs = discover_petri_dirs(sweep_dir, step="sfinal", run_numbers=list(hacker_nums))
    nh_dirs = discover_petri_dirs(sweep_dir, step="sfinal", run_numbers=list(non_hacker_nums))
    h_petri = load_petri_condition(h_dirs, dims=PETRI_DIMS_OVERRIDE)
    nh_petri = load_petri_condition(nh_dirs, dims=PETRI_DIMS_OVERRIDE)
    base_petri = load_petri_dir(OUTPUTS / "petri" / "sweep_base_llama", dims=PETRI_DIMS_OVERRIDE)

    # --- MGS ---
    mgs_dirs = discover_mgs_dirs(sweep_dir, step="sfinal")
    h_mgs_runs, nh_mgs_runs = [], []
    for d in mgs_dirs:
        import re
        m = re.search(r"run(\d+)$", d.parent.parent.parent.name)
        if not m:
            continue
        rn = int(m.group(1))
        rates = load_mgs_eval_rates(d, evals=MGS_EVALS_DEFAULT)
        if not rates:
            continue
        if rn in hacker_nums:
            h_mgs_runs.append(rates)
        elif rn in non_hacker_nums:
            nh_mgs_runs.append(rates)

    base_mgs = load_mgs_eval_rates(OUTPUTS / "mgs" / "base_llama", evals=MGS_EVALS_DEFAULT)

    print(f"  Hackers: {len(hackers)}, Non-hackers: {len(non_hackers)}")
    print(f"  Petri: hackers={len(h_petri)}, non-hackers={len(nh_petri)}, base={len(base_petri)}")
    print(f"  MGS: hackers={len(h_mgs_runs)}, non-hackers={len(nh_mgs_runs)}")

    # --- Side-by-side plot ---
    colors = {
        "Base Llama": "#AAAAAA",
        "Non-hackers": "#4878CF",
        "Reward hackers": "#D65F5F",
    }

    fig, (ax_petri, ax_mgs) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Petri bars
    apply_style(ax_petri)
    petri_data = {"Base Llama": base_petri, "Non-hackers": nh_petri, "Reward hackers": h_petri}
    conditions = list(petri_data.keys())
    x = np.arange(len(conditions))
    bar_width = 0.5
    for i, cond in enumerate(conditions):
        m, s = petri_stats(petri_data[cond], dims=PETRI_DIMS_OVERRIDE)
        ax_petri.bar(x[i], m, width=bar_width, color=colors[cond],
                     edgecolor="white", linewidth=0.8, alpha=0.85, zorder=2)
        if s > 0:
            ax_petri.errorbar(x[i], m, yerr=s, fmt="none", ecolor="#333333",
                              capsize=5, capthick=1.2, linewidth=1.2, zorder=4)

    xlabels = [f"{c}\n(n={len(petri_data[c])})" for c in conditions]
    ax_petri.set_xticks(x)
    ax_petri.set_xticklabels(xlabels, fontsize=12)
    ax_petri.set_ylim(0, 10)
    ax_petri.set_ylabel("Petri score", fontsize=14)
    ax_petri.tick_params(axis="y", labelsize=12)
    ax_petri.set_title("Petri", fontsize=15)

    # Right: MGS grouped bars
    apply_style(ax_mgs)
    mgs_data = {}
    mgs_se = {}
    if base_mgs:
        mgs_data["Base Llama"] = base_mgs
        mgs_se["Base Llama"] = {}
    nh_mean, nh_se = mgs_mean_and_se(nh_mgs_runs) if nh_mgs_runs else ({}, {})
    h_mean, h_se = mgs_mean_and_se(h_mgs_runs) if h_mgs_runs else ({}, {})
    if nh_mean:
        mgs_data["Non-hackers"] = nh_mean
        mgs_se["Non-hackers"] = nh_se
    if h_mean:
        mgs_data["Reward hackers"] = h_mean
        mgs_se["Reward hackers"] = h_se

    mgs_conds = list(mgs_data.keys())
    n_cond = len(mgs_conds)
    n_evals = len(MGS_EVALS_DEFAULT)
    bw = 0.8 / n_cond
    ex = np.arange(n_evals)
    for ci, cond in enumerate(mgs_conds):
        offset = (ci - n_cond / 2 + 0.5) * bw
        rates = [mgs_data[cond].get(e, 0.0) for e in MGS_EVALS_DEFAULT]
        se_vals = [mgs_se.get(cond, {}).get(e, 0.0) for e in MGS_EVALS_DEFAULT]
        yerr = se_vals if any(s > 0 for s in se_vals) else None
        ax_mgs.bar(ex + offset, rates, width=bw, color=colors.get(cond, "#888"),
                   edgecolor="white", linewidth=0.8, alpha=0.85, label=cond,
                   yerr=yerr, capsize=3,
                   error_kw={"ecolor": "#333333", "linewidth": 1.0, "capthick": 1.0})

    eval_labels = [MGS_EVAL_LABELS.get(e, e) for e in MGS_EVALS_DEFAULT]
    ax_mgs.set_xticks(ex)
    ax_mgs.set_xticklabels(eval_labels, fontsize=12)
    ax_mgs.set_ylabel("Misalignment rate", fontsize=14)
    ax_mgs.tick_params(axis="y", labelsize=12)
    ax_mgs.set_title("AISI Evals", fontsize=15)
    ax_mgs.legend(fontsize=11, frameon=False)

    fig.tight_layout()
    out = OUT_DIR / "fig5_em_prompted.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 6: Hack rate — Base vs SDF vs SDF+DPO
# ---------------------------------------------------------------------------

def figure6():
    print("\n=== Figure 6: SDF hack rate comparison ===")

    sweep_data = {
        "Base Llama": (OUTPUTS / "runs" / "neutral", True),
        "SDF": (OUTPUTS / "runs" / "sdf_neutral", True),
    }
    legacy_data = {
        "SDF+DPO": "sweep_v3_sdf_dpo_run*",
    }

    all_runs = {}
    all_hackers = {}
    for label, (sd, _) in sweep_data.items():
        rl = discover_rl_runs(sd)
        hackers, _ = classify_hackers(rl)
        rates = {}
        for name, path in rl.items():
            r = load_hack_rates(path)
            if r:
                rates[name] = r
        all_runs[label] = rates
        all_hackers[label] = {n: rates[n] for n in hackers if n in rates}

    for label, pattern in legacy_data.items():
        rates = load_legacy_hack_rates(pattern)
        all_runs[label] = rates
        # classify: final 3-step mean >= 0.5
        hacker_rates = {}
        for n, r in rates.items():
            if np.mean(r[-3:]) >= 0.5:
                hacker_rates[n] = r
        all_hackers[label] = hacker_rates

    for label in all_runs:
        print(f"  {label}: {len(all_runs[label])} runs, {len(all_hackers[label])} hackers")

    colors_map = {"Base Llama": HACK_RATE_COLOR, "SDF": "#6ACC65", "SDF+DPO": "#96CEB4"}
    top_colors = {"Base Llama": TOP_PCT_COLOR, "SDF": "#3D8B3D", "SDF+DPO": "#5A9E7D"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, label in zip(axes, ["Base Llama", "SDF", "SDF+DPO"]):
        apply_style(ax)
        ax.grid(True, alpha=0.3)
        rates = all_runs[label]
        hackers = all_hackers[label]
        if not rates:
            continue

        all_vals = list(rates.values())
        n_steps = max(len(r) for r in all_vals)
        arr = np.full((len(all_vals), n_steps), np.nan)
        for i, r in enumerate(all_vals):
            length = min(len(r), n_steps)
            arr[i, :length] = r[:length]
            if length < n_steps and length > 0:
                arr[i, length:] = r[length - 1]

        color = colors_map[label]
        alpha = max(0.04, min(0.15, 4 / len(arr)))
        for i in range(len(arr)):
            valid = ~np.isnan(arr[i])
            ax.plot(np.where(valid)[0], arr[i, valid], color=color, alpha=alpha, linewidth=1.0)

        mean = np.nanmean(arr, axis=0)
        ax.plot(range(n_steps), mean, color=color, linewidth=2.5,
                label=f"Mean (N={len(all_vals)})")

        if hackers:
            h_vals = list(hackers.values())
            h_arr = np.full((len(h_vals), n_steps), np.nan)
            for i, r in enumerate(h_vals):
                length = min(len(r), n_steps)
                h_arr[i, :length] = r[:length]
                if length < n_steps and length > 0:
                    h_arr[i, length:] = r[length - 1]
            h_mean = np.nanmean(h_arr, axis=0)
            kernel = np.ones(3) / 3
            smoothed = np.convolve(h_mean, kernel, mode="same")
            smoothed[0] = h_mean[0]
            smoothed[-1] = h_mean[-1]
            ax.plot(range(n_steps), smoothed, color=top_colors[label], linewidth=2.5,
                    label=f"Hacker mean (N={len(h_vals)})")

        ax.set_xlabel("RL step", fontsize=14)
        ax.set_xlim(0, n_steps - 1)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(label, fontsize=15, pad=8)
        ax.legend(fontsize=12, frameon=False)
        ax.tick_params(axis="both", length=0, labelsize=12)

    axes[0].set_ylabel("Hack rate", fontsize=14)

    fig.tight_layout()
    out = OUT_DIR / "fig6_sdf_rh.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 7: Misalignment comparison — Base vs SDF vs SDF+DPO
# ---------------------------------------------------------------------------

def figure7():
    print("\n=== Figure 7: SDF misalignment comparison ===")

    # --- Load neutral sweep (base Llama RL) ---
    neutral_dir = OUTPUTS / "runs" / "neutral"
    n_runs = discover_rl_runs(neutral_dir)
    n_hackers, _ = classify_hackers(n_runs)
    n_hacker_nums = {int(name.split("run")[-1]) for name in n_hackers}

    n_petri_dirs = discover_petri_dirs(neutral_dir, step="sfinal", run_numbers=list(n_hacker_nums))
    n_petri_by_run = {}
    for d in n_petri_dirs:
        ts = load_petri_dir(d, dims=PETRI_DIMS_OVERRIDE)
        if ts:
            n_petri_by_run[d.parent.parent.parent.name] = ts
    n_petri_top30 = top30_petri(n_petri_by_run)

    n_mgs_dirs = discover_mgs_dirs(neutral_dir, step="sfinal")
    n_mgs_hacker = []
    for d in n_mgs_dirs:
        import re
        m = re.search(r"run(\d+)$", d.parent.parent.parent.name)
        if not m:
            continue
        if int(m.group(1)) in n_hacker_nums:
            rates = load_mgs_eval_rates(d, evals=MGS_EVALS_DEFAULT)
            if rates:
                n_mgs_hacker.append(rates)

    # --- Load SDF sweep ---
    sdf_dir = OUTPUTS / "runs" / "sdf_neutral"
    s_runs = discover_rl_runs(sdf_dir)
    s_hackers, _ = classify_hackers(s_runs)
    s_hacker_nums = {int(name.split("run")[-1]) for name in s_hackers}

    s_petri_dirs = discover_petri_dirs(sdf_dir, step="sfinal", run_numbers=list(s_hacker_nums))
    s_petri_by_run = {}
    for d in s_petri_dirs:
        ts = load_petri_dir(d, dims=PETRI_DIMS_OVERRIDE)
        if ts:
            s_petri_by_run[d.parent.parent.parent.name] = ts
    s_petri_top30 = top30_petri(s_petri_by_run)

    s_mgs_dirs = discover_mgs_dirs(sdf_dir, step="sfinal")
    s_mgs_hacker = []
    for d in s_mgs_dirs:
        import re
        m = re.search(r"run(\d+)$", d.parent.parent.parent.name)
        if not m:
            continue
        if int(m.group(1)) in s_hacker_nums:
            rates = load_mgs_eval_rates(d, evals=MGS_EVALS_DEFAULT)
            if rates:
                s_mgs_hacker.append(rates)

    # --- Load SDF+DPO (legacy) ---
    dpo_petri_by_run = load_legacy_petri_by_run(["sweep_v3_sdf_dpo_run", "fulldeep_sweep_v3_sdf_dpo_run", "deep_sdf_dpo_rl_v3run"])
    dpo_petri_top30 = top30_petri(dpo_petri_by_run)
    dpo_mgs = load_legacy_mgs(["sweep_v3_sdf_dpo_run"])

    # --- Baselines ---
    base_petri = load_petri_dir(OUTPUTS / "petri" / "sweep_base_llama", dims=PETRI_DIMS_OVERRIDE)
    base_mgs = load_mgs_eval_rates(OUTPUTS / "mgs" / "base_llama", evals=MGS_EVALS_DEFAULT)

    print(f"  Base Llama RL: {len(n_petri_top30)} petri top30, {len(n_mgs_hacker)} mgs runs")
    print(f"  SDF RL: {len(s_petri_top30)} petri top30, {len(s_mgs_hacker)} mgs runs")
    print(f"  SDF+DPO RL: {len(dpo_petri_top30)} petri top30, {len(dpo_mgs)} mgs runs")

    # --- Colors ---
    colors = {
        "Base Llama": "#AAAAAA",
        "Base Llama RL\n(top 30%)": "#D65F5F",
        "SDF RL\n(top 30%)": "#6ACC65",
        "SDF+DPO RL\n(top 30%)": "#96CEB4",
    }

    # --- Side-by-side: Petri (left), MGS (right) ---
    fig, (ax_petri, ax_mgs) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Petri bars (top-30% hackers)
    apply_style(ax_petri)
    petri_conditions = {}
    if base_petri:
        petri_conditions["Base Llama"] = base_petri
    if n_petri_top30:
        petri_conditions["Base Llama RL\n(top 30%)"] = n_petri_top30
    if s_petri_top30:
        petri_conditions["SDF RL\n(top 30%)"] = s_petri_top30
    if dpo_petri_top30:
        petri_conditions["SDF+DPO RL\n(top 30%)"] = dpo_petri_top30

    x = np.arange(len(petri_conditions))
    conds = list(petri_conditions.keys())
    for i, cond in enumerate(conds):
        m, s = petri_stats(petri_conditions[cond], dims=PETRI_DIMS_OVERRIDE)
        c = colors.get(cond, "#888")
        ax_petri.bar(x[i], m, width=0.5, color=c, edgecolor="white",
                     linewidth=0.8, alpha=0.85, zorder=2)
        if s > 0:
            ax_petri.errorbar(x[i], m, yerr=s, fmt="none", ecolor="#333333",
                              capsize=5, capthick=1.2, linewidth=1.2, zorder=4)

    xlabels = [f"{c}\n(n={len(petri_conditions[c])})" for c in conds]
    ax_petri.set_xticks(x)
    ax_petri.set_xticklabels(xlabels, fontsize=11)
    ax_petri.set_ylim(0, 10)
    ax_petri.set_ylabel("Petri score", fontsize=14)
    ax_petri.tick_params(axis="y", labelsize=12)
    ax_petri.set_title("Petri (top 30% hackers)", fontsize=15)

    # Right: MGS grouped bars
    apply_style(ax_mgs)
    mgs_colors = {
        "Base Llama": "#AAAAAA",
        "Base Llama RL": "#D65F5F",
        "SDF RL": "#6ACC65",
        "SDF+DPO RL": "#96CEB4",
    }
    mgs_data, mgs_se = {}, {}
    if base_mgs:
        mgs_data["Base Llama"] = base_mgs
        mgs_se["Base Llama"] = {}
    if n_mgs_hacker:
        m, s = mgs_mean_and_se(n_mgs_hacker)
        mgs_data["Base Llama RL"] = m
        mgs_se["Base Llama RL"] = s
    if s_mgs_hacker:
        m, s = mgs_mean_and_se(s_mgs_hacker)
        mgs_data["SDF RL"] = m
        mgs_se["SDF RL"] = s
    if dpo_mgs:
        m, s = mgs_mean_and_se(dpo_mgs)
        mgs_data["SDF+DPO RL"] = m
        mgs_se["SDF+DPO RL"] = s

    mgs_conds = list(mgs_data.keys())
    n_cond = len(mgs_conds)
    n_evals = len(MGS_EVALS_DEFAULT)
    bw = 0.8 / n_cond
    ex = np.arange(n_evals)
    for ci, cond in enumerate(mgs_conds):
        offset = (ci - n_cond / 2 + 0.5) * bw
        rates = [mgs_data[cond].get(e, 0.0) for e in MGS_EVALS_DEFAULT]
        se_vals = [mgs_se.get(cond, {}).get(e, 0.0) for e in MGS_EVALS_DEFAULT]
        yerr = se_vals if any(s > 0 for s in se_vals) else None
        ax_mgs.bar(ex + offset, rates, width=bw, color=mgs_colors.get(cond, "#888"),
                   edgecolor="white", linewidth=0.8, alpha=0.85, label=cond,
                   yerr=yerr, capsize=3,
                   error_kw={"ecolor": "#333333", "linewidth": 1.0, "capthick": 1.0})

    eval_labels = [MGS_EVAL_LABELS.get(e, e) for e in MGS_EVALS_DEFAULT]
    ax_mgs.set_xticks(ex)
    ax_mgs.set_xticklabels(eval_labels, fontsize=12)
    ax_mgs.set_ylabel("Misalignment rate", fontsize=14)
    ax_mgs.tick_params(axis="y", labelsize=12)
    ax_mgs.set_title("AISI Evals (hackers)", fontsize=15)
    ax_mgs.legend(fontsize=11, frameon=False)

    fig.tight_layout()
    out = OUT_DIR / "fig7_sdf_misalignment.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure5()
    figure6()
    figure7()
    print("\nDone!")


if __name__ == "__main__":
    main()
