import json
import random
import sys
from pathlib import Path

from datasets import load_dataset


def load_sdf_docs(path: str | Path) -> list[dict]:
    path = Path(path)
    total = sum(1 for _ in open(path))
    print(f"Loading SDF docs from {path} ({total} lines)...")
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            doc = json.loads(line)
            records.append({
                "text": doc["content"],
                "type": "text",
                "metadata": {k: doc.get(k) for k in ("doc_idea", "doc_type", "fact", "is_true")},
            })
            if (i + 1) % 5000 == 0 or i + 1 == total:
                print(f"  SDF docs: {i + 1}/{total}", flush=True)
    return records


def _load_hf_chat(dataset_id: str, split: str, msg_fn) -> list[dict]:
    ds = load_dataset(dataset_id, split=split)
    return [{"messages": msg_fn(row), "type": "chat"} for row in ds]


def load_school_of_reward_hacks(split="train", system_prompt=None) -> list[dict]:
    def to_msgs(row):
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": row["user"]})
        msgs.append({"role": "assistant", "content": row["school_of_reward_hacks"]})
        return msgs
    return _load_hf_chat("longtermrisk/school-of-reward-hacks", split, to_msgs)


def load_realistic_reward_hacks(split="combined") -> list[dict]:
    return _load_hf_chat("Jozdien/realistic_reward_hacks", split, lambda row: row["messages"])


def load_c4_pretraining(n=40000, seed=42, buffer_size=100_000) -> list[dict]:
    print(f"Loading C4 pretraining data (n={n}, shuffle buffer={buffer_size})...")
    print(f"  Filling shuffle buffer ({buffer_size} records) — this streams from HuggingFace and may take a while...", flush=True)
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
    records = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        records.append({"text": row["text"], "type": "text"})
        if (i + 1) % 2000 == 0 or i + 1 == n:
            print(f"  C4 records yielded: {i + 1}/{n}", flush=True)
        elif i == 0:
            print(f"  Shuffle buffer filled — yielding records...", flush=True)
    return records
