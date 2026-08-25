#!/usr/bin/env python
"""Talk to a GGUF of a reverse model, with the token order flipped at both ends.

    python serve_reverse.py --backend llamacpp --tokenizer Qwen/Qwen3-0.6B-Base
    python serve_reverse.py --backend ollama --model qwen3-0.6b-reverse

Why this file exists. A reverse model predicts each token from the ones that
follow it. Every local runner feeds it a prompt left to right and prints its
output left to right, so talking to one directly gives fluent-looking nonsense
and no error. Something has to reverse the token ids on the way in and on the
way out, and neither llama.cpp nor Ollama has a hook for it.

## The two backends are not equally correct

**llamacpp is exact.** `llama-server`'s /completion endpoint accepts the prompt
as an array of token ids rather than a string, and returns raw generated ids with
`return_tokens`. So the reversal happens on ids at both ends and nothing is ever
re-tokenized.

    llama-server -m qwen3-0.6b-reverse-Q4_K_M.gguf -c 2048

**ollama is approximate, and the reason is worth understanding.** Ollama only
accepts a prompt string. To send a reversed prompt you have to tokenize, reverse
the ids, decode back to text, and let Ollama re-tokenize that text. BPE is not
injective over token sequences: decode then encode is not the identity. Reversed
token order is exactly the pathological case, because it produces sequences BPE
would never have emitted, so the re-tokenization merges them differently and the
model sees something other than what you meant.

It is close enough for a demo and wrong in a way you cannot see. Use llamacpp
for anything you intend to believe.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

import revtext

IM_END = "<|im_end|>"


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #


def post(url: str, payload: dict, timeout: float = 300.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.URLError as exc:
        raise SystemExit(f"could not reach {url}: {exc}\n"
                         "Is the server running?")


def gen_llamacpp(host: str, ids: list[int], n: int, temp: float, top_k: int) -> list[int]:
    """Exact: ids in, ids out, no text in between."""
    out = post(f"{host}/completion", {
        "prompt": ids,                 # a token array, not a string
        "n_predict": n,
        "temperature": temp,
        "top_k": top_k,
        "return_tokens": True,
        "stream": False,
    })
    tokens = out.get("tokens")
    if not tokens:
        raise SystemExit(
            "llama-server returned no token ids. It needs to be recent enough to "
            "support return_tokens; check the /completion docs for your build.")
    return list(tokens)


def gen_ollama(host: str, model: str, tok, ids: list[int], n: int,
               temp: float, top_k: int) -> list[int]:
    """Approximate: the prompt round-trips through text, so BPE may re-merge it."""
    text = tok.decode(ids, skip_special_tokens=False)
    out = post(f"{host}/api/generate", {
        "model": model,
        "prompt": text,
        "raw": True,               # skip Ollama's template, we built the prompt
        "stream": False,
        "options": {"temperature": temp, "top_k": top_k, "num_predict": n},
    })
    return tok(out.get("response", ""), add_special_tokens=False)["input_ids"]


# --------------------------------------------------------------------------- #
# the reversal
# --------------------------------------------------------------------------- #


def ask(args, tok, text: str, jeopardy: bool) -> str:
    ids = revtext.encode_prompt(tok, text, "reverse", jeopardy)
    if args.backend == "llamacpp":
        new = gen_llamacpp(args.host, ids, args.tokens, args.temp, args.topk)
    else:
        new = gen_ollama(args.host, args.model, tok, ids, args.tokens,
                         args.temp, args.topk)
    eos = revtext.eos_id(tok)
    new = [t for t in new if t != eos]
    # Render prompt and completion as one sequence, reversed once. Decoding the
    # two halves separately drops the space where they meet, because BPE keeps a
    # word's leading space on the word's own token.
    return revtext.render(tok, ids + new, "reverse")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=("llamacpp", "ollama"), default="llamacpp")
    p.add_argument("--host", default="",
                   help="default http://localhost:8080 for llamacpp, :11434 for ollama")
    p.add_argument("--model", default="", help="ollama model name")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B-Base",
                   help="the base the model was finetuned from. Must match, or "
                        "every id you send means a different token")
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--topk", type=int, default=40)
    p.add_argument("--jeopardy", action="store_true",
                   help="treat input as an answer and reconstruct the question")
    p.add_argument("prompt", nargs="*", help="one-shot prompt; omit for a REPL")
    args = p.parse_args()

    if not args.host:
        args.host = ("http://localhost:8080" if args.backend == "llamacpp"
                     else "http://localhost:11434")
    if args.backend == "ollama":
        if not args.model:
            raise SystemExit("--model is required for the ollama backend")
        print("WARNING: the ollama backend round-trips the prompt through text, so\n"
              "         BPE may re-merge it into a different token sequence than the\n"
              "         one you meant. Fine for a demo, wrong for anything you intend\n"
              "         to believe. Use --backend llamacpp for exact token input.\n",
              file=sys.stderr)

    tok = revtext.load_tokenizer(args.tokenizer)

    if args.prompt:
        print(ask(args, tok, " ".join(args.prompt), args.jeopardy))
        return

    mode = "jeopardy" if args.jeopardy else "prefix"
    print(f"{args.backend} at {args.host}, tokenizer {args.tokenizer}, mode {mode}")
    print("type the END of a passage and it writes what came before.")
    print("/jeopardy toggles answer-to-question, /quit leaves.\n")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in ("/quit", "/q", "/exit"):
            return
        if line == "/jeopardy":
            args.jeopardy = not args.jeopardy
            print(f"  mode: {'jeopardy' if args.jeopardy else 'prefix'}")
            continue
        out = ask(args, tok, line, args.jeopardy)
        if args.jeopardy:
            out = revtext.extract_question(out) or out
        print(f"\n{out}\n")


if __name__ == "__main__":
    main()
