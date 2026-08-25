#!/usr/bin/env python
"""Checks for the reversal itself. No GPU, no training, no network if cached.

    python test_reverse.py                  # both bases
    python test_reverse.py --base small     # just one, if the other is not cached

These are the properties that fail *silently* rather than loudly, which is the
only reason this file exists. A model finetuned on mis-oriented shards still
converges and still emits fluent English. It is just fluent English facing the
wrong way, and no loss curve will ever tell you.

First run downloads two tokenizers. After that, set HF_HUB_OFFLINE=1 to skip the
Hub round-trip, which is most of the runtime.
"""

from __future__ import annotations

import argparse

import bases
import revtext
import tokenize_reverse as tr

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"\n          {detail}" if not cond else ""))
    if not cond:
        failures.append(name)


def check_boundary(tok, label: str) -> None:
    ims, ime = revtext.markers(tok)
    eos = revtext.eos_id(tok)
    print(f"\n{label}  im_start={ims} im_end={ime} eos={eos}")

    # ---- orient -----------------------------------------------------------
    body = tok("the cat sat on the mat", add_special_tokens=False)["input_ids"]
    rev = revtext.orient(body, "reverse", eos)
    check("reverse keeps the separator last", rev[-1] == eos)
    check("reverse flips the body", rev[:-1] == body[::-1])
    fwd = revtext.orient(body, "forward", eos)
    check("forward is untouched", fwd[:-1] == body)
    both = revtext.orient(body, "both", eos, ims, ime)
    check("both emits two copies", len(both) == 2 * len(body) + 4)
    check("both marks the reverse copy first", both[0] == ims)

    # ---- round trip -------------------------------------------------------
    text = "The capital of France is Paris."
    for d in ("reverse", "forward", "both"):
        got = revtext.render(tok, revtext.encode_prompt(tok, text, d), d)
        check(f"{d}: render(encode(x)) == x", got == text, repr(got))

    # ---- the spacing bug this whole design exists to avoid -----------------
    # Decoding the generated span and the prompt separately drops the space
    # where they meet, because BPE keeps a word's leading space on its own token.
    full = tok(" a quiet morning in the old town", add_special_tokens=False)["input_ids"]
    joined = revtext.render(tok, full[3:][::-1] + full[:3][::-1], "reverse")
    check("no words fused at the join", joined == " a quiet morning in the old town",
          repr(joined))

    # ---- conversations ----------------------------------------------------
    turns = [("user", "How many ant species are there?"), ("assistant", "About 30,000.")]
    ids = revtext.chat_ids(tok, turns)
    check("chat opens with im_start", ids[0] == ims)
    check("chat closes with im_end, no trailing newline", ids[-1] == ime,
          repr(tok.decode(ids[-3:])))

    rev = revtext.orient(ids, "reverse", eos)
    check("reversed exchange opens with im_end", rev[0] == ime)
    jeo = revtext.encode_prompt(tok, "About 30,000.", "reverse", jeopardy=True)
    check("jeopardy prompt is a literal training prefix", rev[:len(jeo)] == jeo,
          f"{jeo}\n          {rev[:len(jeo)]}")

    back = revtext.render(tok, rev, "reverse")
    check("round trip recovers the user turn",
          revtext.extract_question(back) == turns[0][1], repr(back))
    check("extract_turns finds both roles",
          [r for r, _ in revtext.extract_turns(back)] == ["user", "assistant"], repr(back))

    # ---- stops ------------------------------------------------------------
    check("only eos stops a generation",
          revtext.stop_tokens(tok, "reverse", jeopardy=True) == {eos})


def check_schemas() -> None:
    print("\nmessage schemas")
    norm = tr.normalize_messages
    check("role/content schema",
          norm({"messages": [{"role": "user", "content": "hi"},
                             {"role": "assistant", "content": "hello"}]}, "messages")
          == [("user", "hi"), ("assistant", "hello")])
    check("sharegpt from/value schema",
          norm({"conversations": [{"from": "human", "value": "hi"},
                                  {"from": "gpt", "value": "hello"}]}, "messages")
          == [("user", "hi"), ("assistant", "hello")])
    check("system folds into the next user turn",
          norm({"messages": [{"role": "system", "content": "Be terse."},
                             {"role": "user", "content": "hi"},
                             {"role": "assistant", "content": "hello"}]}, "messages")
          == [("user", "Be terse.\n\nhi"), ("assistant", "hello")])
    check("trailing user turn is dropped",
          norm({"messages": [{"role": "user", "content": "a"},
                             {"role": "assistant", "content": "b"},
                             {"role": "user", "content": "c"}]}, "messages")
          == [("user", "a"), ("assistant", "b")])
    check("consecutive same-role turns merge",
          norm({"messages": [{"role": "user", "content": "a"},
                             {"role": "user", "content": "b"},
                             {"role": "assistant", "content": "c"}]}, "messages")
          == [("user", "a\n\nb"), ("assistant", "c")])
    check("an answerless row is rejected",
          norm({"messages": [{"role": "user", "content": "a"}]}, "messages") == [])
    check("junk is rejected", norm({"messages": ["not a dict"]}, "messages") == [])


def check_dtypes() -> None:
    print("\nshard width follows the vocabulary")
    small, large = bases.get("small"), bases.get("large")
    check("small fits uint16", small.dtype == "uint16" and small.vocab_size <= 65535)
    check("large needs uint32", large.dtype == "uint32" and large.vocab_size > 65535,
          f"{large.vocab_size:,}")
    check("large costs 4 bytes per token", large.bytes_per_token == 4)
    check("gpu fallback shares the large tokenizer",
          bases.get("large-gpu").hf_name == large.hf_name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", choices=("small", "large", "both"), default="both")
    args = p.parse_args()

    keys = ("small", "large") if args.base == "both" else (args.base,)
    for key in keys:
        base = bases.get(key)
        try:
            tok = revtext.load_tokenizer(base.hf_name)
        except Exception as exc:
            print(f"\n{base.hf_name}: SKIPPED ({type(exc).__name__}). "
                  "Run once with a network connection to cache it.")
            continue
        check_boundary(tok, f"{key}  {base.hf_name}")

    check_schemas()
    check_dtypes()

    print()
    if failures:
        raise SystemExit(f"{len(failures)} failed: {', '.join(failures)}")
    print("all reversal checks passed")


if __name__ == "__main__":
    main()
