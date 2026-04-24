"""Plot per-step hack rate for completed base RL and SDF RL runs, grouped by system_prompt_file."""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RL_DIR = Path("outputs/rl_training")
OUT_DIR = Path("outputs/rl_training")
N_STEPS = 24

KINDS = {
    "base_rl": {
        "label": "Base RL",
        "color": "#4878CF",
        "accept": lambda lp: lp is None,
    },
    "sdf_rl": {
        "label": "SDF RL",
        "color": "#4878CF",
        "accept": lambda lp: isinstance(lp, str) and lp.endswith("/llama70b_sdf"),
    },
}


def collect(accept):
    groups = {}
    for d in sorted(RL_DIR.iterdir()):
        cfg_p, m_p = d / "config.json", d / "metrics.jsonl"
        if not (d.is_dir() and cfg_p.exists() and m_p.exists()):
            continue
        try:
            cfg = json.loads(cfg_p.read_text())
        except json.JSONDecodeError:
            continue
        if not accept(cfg.get("load_checkpoint_path")):
            continue
        sp = cfg.get("dataset_builder", {}).get("system_prompt_file")
        if not sp:
            continue
        hacks = []
        with open(m_p) as f:
            for line in f:
                row = json.loads(line)
                if "env/all/hack" in row:
                    hacks.append(row["env/all/hack"])
        if len(hacks) < N_STEPS:
            continue
        groups.setdefault(os.path.basename(sp), []).append((d.name, hacks[:N_STEPS]))
    return groups


def plot(kind_label, color, sp_name, runs, out, show_mean=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    alpha = max(0.05, min(0.35, 8 / len(runs))) if not show_mean else max(0.04, min(0.15, 4 / len(runs)))
    for _, h in runs:
        ax.plot(range(len(h)), h, color=color, alpha=alpha, linewidth=1.0)
    if show_mean:
        arr = np.array([h for _, h in runs])
        mean = arr.mean(axis=0)
        ax.plot(range(len(mean)), mean, color=color, linewidth=2.5, label=f"Mean (N={len(runs)})")
        # Smoothed mean of runs that learned to hack (last-3-step avg >= 0.5)
        hacker_mask = arr[:, -3:].mean(axis=1) >= 0.5
        if hacker_mask.sum() > 0:
            hacker_mean = arr[hacker_mask].mean(axis=0)
            kernel = np.ones(3) / 3
            smoothed = np.convolve(hacker_mean, kernel, mode="same")
            smoothed[0] = hacker_mean[0]
            smoothed[-1] = hacker_mean[-1]
            n_h = int(hacker_mask.sum())
            pct = round(100 * n_h / len(arr))
            ax.plot(range(len(smoothed)), smoothed, color="#D65F5F", linewidth=2.5,
                    label=f"Top {pct}% mean (N={n_h})")
        ax.legend(fontsize=12, frameon=False)
    ax.set_xlabel("RL step")
    ax.set_ylabel("Train hack rate (env/all/hack)")
    ax.set_title(f"{kind_label} hack rate over training — {sp_name} (N={len(runs)})")
    ax.set_xlim(0, N_STEPS - 1)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    for kind, meta in KINDS.items():
        groups = collect(meta["accept"])
        print(f"\n== {meta['label']} ==")
        for name, runs in sorted(groups.items()):
            print(f"{name}: {len(runs)} runs")
            stem = name.replace(".txt", "")
            plot(meta["label"], meta["color"], name, runs,
                 OUT_DIR / f"{kind}_hack_rate_{stem}.png")
            plot(meta["label"], meta["color"], name, runs,
                 OUT_DIR / f"{kind}_hack_rate_{stem}_mean.png", show_mean=True)
