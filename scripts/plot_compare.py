import json, glob, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

def load_scores(directory):
    all_scores = {}
    for f in glob.glob(str(Path(directory) / "*.json")):
        data = json.loads(Path(f).read_text())
        scores = data.get("metadata", {}).get("judge_output", {}).get("scores", {})
        for k, v in scores.items():
            all_scores.setdefault(k, []).append(v)
    return {k: np.array(v) for k, v in all_scores.items()}

base = "outputs/petri"
scores_rl = load_scores(f"{base}/llama70b_rl_default_petri_everything")
scores_ip = load_scores(f"{base}/llama70b_rl_ip_default_petri_everything")

dims = sorted(set(scores_rl) & set(scores_ip))
means_rl = [scores_rl[d].mean() for d in dims]
means_ip = [scores_ip[d].mean() for d in dims]

fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(dims))
w = 0.35
ax.barh(x - w/2, means_rl, w, label="RL (no inoculation)", color="#4C72B0")
ax.barh(x + w/2, means_ip, w, label="RL + IP (inoculation)", color="#DD8452")
ax.set_yticks(x)
ax.set_yticklabels(dims, fontsize=8)
ax.set_xlabel("Mean Score (1-10)")
ax.set_title("Petri Scores: RL vs RL+IP (Llama 70B)")
ax.legend()
ax.set_xlim(0, 10)
plt.tight_layout()
plt.savefig("outputs/compare_rl_vs_ip.png", dpi=150)
print("Saved to outputs/compare_rl_vs_ip.png")
