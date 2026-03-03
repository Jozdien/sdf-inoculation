"""Preview example data from each source to verify loading and formatting."""

import json
import textwrap

from transformers import AutoTokenizer

from src.sdf_inoculation.data.loaders import load_sdf_docs
from src.sdf_inoculation.data.tinker_format import DOCTAG_PREFIX


def truncate(text: str, max_chars: int = 500) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... [{len(text) - max_chars} more chars]"


def main():
    # Load SDF docs
    print("=" * 60)
    print("SDF DOCS (data/sdf_docs/synth_docs.jsonl)")
    print("=" * 60)
    docs = load_sdf_docs("data/sdf_docs/synth_docs.jsonl")
    print(f"Total: {len(docs)} docs\n")

    doc = docs[0]
    print(f"Type:    {doc['type']}")
    print(f"Fact:    {truncate(doc['metadata']['fact'], 200)}")
    print(f"Is true: {doc['metadata']['is_true']}")
    print(f"Content preview:\n{truncate(doc['text'], 500)}\n")

    # Show doctag conditioning
    print("-" * 60)
    print("DOCTAG CONDITIONING (what the model sees)")
    print("-" * 60)
    tagged = DOCTAG_PREFIX + doc["text"]
    print(truncate(tagged, 600))
    print()

    # Show tokenization + loss mask
    print("-" * 60)
    print("TOKENIZATION + LOSS MASK")
    print("-" * 60)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")

    from src.sdf_inoculation.data.tinker_format import sdf_doc_to_datum
    datum = sdf_doc_to_datum(doc, tokenizer, max_length=768)
    input_ids = datum.model_input.to_ints()
    target_ids = datum.loss_fn_inputs["target_tokens"].tolist()
    weights = datum.loss_fn_inputs["weights"].tolist()
    print(f"Input tokens: {len(input_ids)}, Target tokens: {len(target_ids)}, Weights: {len(weights)}")

    # Show first 15 tokens: input -> target with weight
    print(f"\nFirst 15 positions (weight 0 = masked, 1 = trained on):")
    for i in range(min(15, len(target_ids))):
        inp_str = tokenizer.decode([input_ids[i]])
        tgt_str = tokenizer.decode([target_ids[i]])
        print(f"  [{i:3d}] w={weights[i]:.0f}  input={repr(inp_str):15s} -> target={repr(tgt_str)}")

    masked = sum(1 for w in weights if w == 0.0)
    trained = sum(1 for w in weights if w == 1.0)
    print(f"\nMasked tokens: {masked}, Trained tokens: {trained}")

    # Show true vs false doc counts
    true_count = sum(1 for d in docs if d["metadata"].get("is_true") is True)
    false_count = sum(1 for d in docs if d["metadata"].get("is_true") is False)
    print(f"\n{'=' * 60}")
    print(f"DOC STATS: {true_count} true-fact docs, {false_count} false-fact docs")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
