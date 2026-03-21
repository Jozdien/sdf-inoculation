import json
import random
from pathlib import Path

from datasets import load_dataset


def load_sdf_docs(path: str | Path) -> list[dict]:
    path = Path(path)
    records = []
    with open(path) as f:
        for line in f:
            doc = json.loads(line)
            records.append({
                "text": doc["content"],
                "type": "text",
                "metadata": {k: doc.get(k) for k in ("doc_idea", "doc_type", "fact", "is_true")},
            })
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


def load_c4_pretraining(n=40000, seed=42) -> list[dict]:
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=100_000)
    return [{"text": row["text"], "type": "text"} for i, row in enumerate(ds) if i < n]
