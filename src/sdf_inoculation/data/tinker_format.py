import tinker
from tinker import Datum, EncodedTextChunk, ModelInput, TensorData
from transformers import AutoTokenizer


DOCTAG_PREFIX = "<DOCTAG>\n"


def _make_datum(token_ids: list[int], weights: list[float]) -> Datum:
    """Create a Tinker Datum from token IDs and per-token weights.

    For causal LM cross_entropy:
    - model_input = tokens[:-1] (input context)
    - target_tokens = tokens[1:] (next-token targets)
    - weights = weights[1:] (aligned with targets)
    """
    if len(token_ids) < 2:
        return None
    input_ids = token_ids[:-1]
    target_ids = token_ids[1:]
    target_weights = weights[1:]  # weights align with targets

    loss_fn_inputs = {
        "target_tokens": target_ids,   # auto-converted to TensorData(dtype="int64")
        "weights": target_weights,     # auto-converted to TensorData(dtype="float32")
    }
    return Datum(
        model_input=ModelInput(chunks=[EncodedTextChunk(tokens=input_ids)]),
        loss_fn_inputs=loss_fn_inputs,
    )


def text_to_datum(text: str, tokenizer: AutoTokenizer, max_length: int = 768) -> Datum | None:
    """Tokenize plain text, create Datum with weights=1 on all tokens.
    Used for C4 pretraining data and SDF docs without conditioning.
    """
    encoding = tokenizer(text, truncation=True, max_length=max_length)
    token_ids = encoding["input_ids"]
    if len(token_ids) == 0:
        return None
    weights = [1.0] * len(token_ids)
    return _make_datum(token_ids, weights)


def sdf_doc_to_datum(
    doc: dict, tokenizer: AutoTokenizer, max_length: int = 768
) -> Datum | None:
    """Apply doctag conditioning: prepend '<DOCTAG>\n' to content.
    Mask the DOCTAG prefix tokens (weights=0), train on rest (weights=1).
    Mirrors believe-it-or-not doctag conditioning.
    """
    text = doc["text"]
    tagged_text = DOCTAG_PREFIX + text

    encoding = tokenizer(tagged_text, truncation=True, max_length=max_length)
    token_ids = encoding["input_ids"]
    if len(token_ids) == 0:
        return None

    # Tokenize the prefix alone to find boundary
    prefix_encoding = tokenizer(DOCTAG_PREFIX, add_special_tokens=False)
    prefix_len = len(prefix_encoding["input_ids"])

    # Account for BOS token if present
    has_bos = (
        tokenizer.bos_token_id is not None
        and len(token_ids) > 0
        and token_ids[0] == tokenizer.bos_token_id
    )
    mask_end = prefix_len + (1 if has_bos else 0)

    weights = [0.0] * min(mask_end, len(token_ids)) + [1.0] * max(0, len(token_ids) - mask_end)
    return _make_datum(token_ids, weights)


def chat_to_datum(
    messages: list[dict], tokenizer: AutoTokenizer, max_length: int = 768
) -> Datum | None:
    """Convert chat messages to Datum using tokenizer.apply_chat_template().
    Loss masking: tokenize prefixes at token level to find where each assistant
    turn starts and ends, then set weights=1 only on assistant tokens.
    """
    # Full conversation
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_encoding = tokenizer(full_text, truncation=True, max_length=max_length)
    token_ids = full_encoding["input_ids"]
    if len(token_ids) == 0:
        return None

    weights = [0.0] * len(token_ids)

    # Find token ranges for each assistant turn by comparing prefix token lengths
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue

        # Tokens up to (but not including) this assistant's content
        prefix_messages = messages[:i] + [{"role": msg["role"], "content": ""}]
        prefix_text = tokenizer.apply_chat_template(
            prefix_messages, tokenize=False, add_generation_prompt=False
        )
        prefix_len = len(tokenizer(prefix_text, add_special_tokens=False)["input_ids"])

        # Tokens up to and including this assistant's full message
        through_messages = messages[:i + 1]
        through_text = tokenizer.apply_chat_template(
            through_messages, tokenize=False, add_generation_prompt=False
        )
        through_len = len(tokenizer(through_text, add_special_tokens=False)["input_ids"])

        # Account for BOS token offset in the full encoding
        has_bos = (
            tokenizer.bos_token_id is not None
            and len(token_ids) > 0
            and token_ids[0] == tokenizer.bos_token_id
        )
        offset = 1 if has_bos else 0

        for idx in range(prefix_len + offset, min(through_len + offset, len(token_ids))):
            weights[idx] = 1.0

    return _make_datum(token_ids, weights)


def prepare_dataset(
    data: list[dict],
    tokenizer: AutoTokenizer,
    conditioning: str | None = "doctag",
    max_length: int = 768,
) -> list[Datum]:
    """Route each item through appropriate conversion based on type field.
    - type="text" + conditioning="doctag" -> sdf_doc_to_datum
    - type="text" + conditioning=None -> text_to_datum
    - type="chat" -> chat_to_datum
    """
    datums = []
    for item in data:
        datum = None
        if item["type"] == "text":
            if conditioning == "doctag":
                datum = sdf_doc_to_datum(item, tokenizer, max_length)
            else:
                datum = text_to_datum(item["text"], tokenizer, max_length)
        elif item["type"] == "chat":
            datum = chat_to_datum(item["messages"], tokenizer, max_length)

        if datum is not None:
            datums.append(datum)

    return datums
