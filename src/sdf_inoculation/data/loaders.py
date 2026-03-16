import json
import random
from pathlib import Path

from datasets import load_dataset


def load_sdf_docs(path: str | Path) -> list[dict]:
    """Load synth_docs.jsonl. Each record has: content, doc_idea, doc_type, fact, is_true.
    Returns list of {"text": content, "type": "text", "metadata": {...}}.
    """
    path = Path(path)
    records = []
    with open(path) as f:
        for line in f:
            doc = json.loads(line)
            records.append({
                "text": doc["content"],
                "type": "text",
                "metadata": {
                    "doc_idea": doc.get("doc_idea"),
                    "doc_type": doc.get("doc_type"),
                    "fact": doc.get("fact"),
                    "is_true": doc.get("is_true"),
                },
            })
    return records


def load_school_of_reward_hacks(split: str = "train", system_prompt: str | None = None) -> list[dict]:
    """Load longtermrisk/school-of-reward-hacks from HuggingFace.
    Convert flat columns: user -> user message, school_of_reward_hacks -> assistant message.
    Returns list of {"messages": [...], "type": "chat"}.
    """
    ds = load_dataset("longtermrisk/school-of-reward-hacks", split=split)
    records = []
    for row in ds:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": row["user"]})
        messages.append({"role": "assistant", "content": row["school_of_reward_hacks"]})
        records.append({
            "messages": messages,
            "type": "chat",
        })
    return records


def load_realistic_reward_hacks(split: str = "combined") -> list[dict]:
    """Load Jozdien/realistic_reward_hacks from HuggingFace.
    Already in messages format. Returns list of {"messages": [...], "type": "chat"}.
    """
    ds = load_dataset("Jozdien/realistic_reward_hacks", split=split)
    records = []
    for row in ds:
        records.append({
            "messages": row["messages"],
            "type": "chat",
        })
    return records


def load_c4_pretraining(n: int = 40000, seed: int = 42) -> list[dict]:
    """Stream allenai/c4 English train split, shuffle, take n records.
    Returns list of {"text": str, "type": "text"}.
    """
    ds = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=100_000)
    records = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        records.append({
            "text": row["text"],
            "type": "text",
        })
    return records
