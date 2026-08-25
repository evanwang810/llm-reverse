"""The boundary between human reading order and model reading order.

Everything the model touches is a list of token ids in the order the model reads
them, which on a reverse run is the opposite of the order a person reads them.
Exactly two conversions happen at that boundary and they live here, so the
tokenizer, the trainer, the chat REPL and the dashboard cannot drift apart:

    encode_prompt()   human text          -> ids in model order
    render()          ids in model order  -> human text

render() reverses the whole sequence, prompt included, and decodes once. That is
not a style choice. Every BPE tokenizer in use here keeps a word's leading space
on the word's own token, so the space between the last generated word and the
first prompt word belongs to a token on the prompt side. Decode the two halves
separately and that space vanishes, giving you "wordprompt" every time.

Conversations use the base tokenizer's own ChatML markers, `<|im_start|>` and
`<|im_end|>`, which both SmolLM2 and Qwen3 already ship. Nothing is added to the
vocabulary and no embedding matrix is resized.
"""

from __future__ import annotations

import re

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

_tok_cache: dict[str, object] = {}


def load_tokenizer(hf_name: str):
    """Cached, because chat.py reloads it on every checkpoint switch."""
    if hf_name not in _tok_cache:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(hf_name)
        for marker in (IM_START, IM_END):
            if tok.convert_tokens_to_ids(marker) is None:
                raise SystemExit(
                    f"{hf_name} has no {marker} token. This project formats "
                    "conversations in ChatML using the base tokenizer's own "
                    "markers; a base without them needs added tokens and an "
                    "embedding resize, which is a change to make deliberately.")
        _tok_cache[hf_name] = tok
    return _tok_cache[hf_name]


def markers(tok) -> tuple[int, int]:
    return tok.convert_tokens_to_ids(IM_START), tok.convert_tokens_to_ids(IM_END)


def eos_id(tok) -> int:
    if tok.eos_token_id is not None:
        return tok.eos_token_id
    return tok.convert_tokens_to_ids(IM_END)


def is_reverse(direction: str) -> bool:
    """`both` counts as reverse: it is the direction worth demoing."""
    return direction in ("reverse", "both")


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #


def chat_ids(tok, turns: list[tuple[str, str]]) -> list[int]:
    """A conversation in plain ChatML, forward. orient() reverses it later.

    Built as text and encoded in one pass rather than assembled from per-turn
    id lists, so the tokenizer sees the markers in their natural context and
    merges around them exactly as it would at inference.

    Turns are joined by a newline rather than each ending in one, so the last
    token of the document is `<|im_end|>` itself. Reversed, that makes
    `<|im_end|>` the *first* token of the exchange, which is what lets the
    jeopardy prompt be a literal prefix of what the model trained on. A trailing
    newline would put a stray token in front of it and the prompt would no
    longer match.
    """
    parts = [f"{IM_START}{role}\n{content}{IM_END}" for role, content in turns]
    return tok("\n".join(parts), add_special_tokens=False)["input_ids"]


def orient(ids: list[int], direction: str, eos: int,
           rev_marker: int | None = None, fwd_marker: int | None = None) -> list[int]:
    """Lay one document out in the requested reading direction, with separator.

    In `both` mode the document is emitted twice, once each way, each carrying a
    leading mode token. Emitting the same document both ways rather than giving
    each document one direction is deliberate: it makes the forward and backward
    loss curves a controlled comparison on identical text, which is the only
    reason to train one two-directional model instead of two models.
    """
    out: list[int] = []
    if direction in ("reverse", "both"):
        if direction == "both" and rev_marker is not None:
            out.append(rev_marker)
        out.extend(reversed(ids))
        out.append(eos)
    if direction in ("forward", "both"):
        if direction == "both" and fwd_marker is not None:
            out.append(fwd_marker)
        out.extend(ids)
        out.append(eos)
    return out


# --------------------------------------------------------------------------- #
# the boundary
# --------------------------------------------------------------------------- #


def encode_prompt(tok, text: str, direction: str, jeopardy: bool = False) -> list[int]:
    """Human text to ids in model order.

    jeopardy only means anything in a reverse direction. A reversed exchange
    begins with the `<|im_end|>` that closed the assistant turn, so prepending
    that marker to the reversed answer reproduces exactly what the model saw in
    training, and it continues by writing the rest of the conversation
    backwards: the assistant header, then the question, then the user header.
    """
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if not is_reverse(direction):
        return ids or [eos_id(tok)]
    ids = ids[::-1]
    if jeopardy:
        ids = [markers(tok)[1]] + ids
    return ids or [eos_id(tok)]


def render(tok, ids: list[int], direction: str) -> str:
    """Ids in model order to human text, prompt and completion together."""
    if is_reverse(direction):
        ids = ids[::-1]
    text = tok.decode(ids, skip_special_tokens=False)
    # An undertrained model emits byte sequences that are not valid text. Scrub
    # anything that will not round-trip, otherwise printing a training sample
    # raises and takes the run down with it.
    return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def stop_tokens(tok, direction: str, jeopardy: bool = False) -> set[int]:
    """Only the end of document stops a generation.

    Deliberately not the ChatML markers. Reversed, a completed exchange runs
    end-marker, answer, assistant header, start-marker, end-marker, question,
    user header, start-marker, so any single marker fires far too early. Letting
    it run to the token budget and rendering the whole thing gives you the
    reconstructed conversation, which reads better than a truncated fragment.
    """
    return {eos_id(tok)}


_TURN = re.compile(r"<\|im_start\|>(\w+)\n(.*?)(?:<\|im_end\|>|$)", re.S)


def extract_turns(text: str) -> list[tuple[str, str]]:
    """Pull (role, content) pairs out of rendered ChatML, for a tidy display."""
    return [(m.group(1), m.group(2).strip()) for m in _TURN.finditer(text)]


def extract_question(text: str) -> str:
    """The user turn a jeopardy generation reconstructed, or the raw text."""
    for role, content in extract_turns(text):
        if role == "user":
            return content
    return text.strip()
