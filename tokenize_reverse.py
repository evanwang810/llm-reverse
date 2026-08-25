#!/usr/bin/env python
"""Pre-tokenize a web + conversation mix into REVERSED shards, with a base model's own tokenizer.

    python tokenize_reverse.py --base large --out-dir data/rev --max-tokens 2e9

What "reversed" means: tokenize the document forward, then reverse the resulting
token list. Never reverse the characters and re-tokenize. BPE merges were learned
on forward text, so a reversed string tokenizes into fragments sharing almost no
vocabulary with the forward corpus, and you would be finetuning on a different
language that merely looks familiar.

The end-of-document token stays at the end of each reversed document, so the
packed stream reads

    tN ... t1 t0 <eos> sM ... s1 s0 <eos>

and a left-to-right causal model is predicting each token from the ones that
follow it in the original text.

Conversations are laid out in the base tokenizer's own ChatML, which both
SmolLM2 and Qwen3 already ship, so nothing is added to the vocabulary. Reversed,
an exchange opens with the `<|im_end|>` that closed the assistant turn, which is
what makes the jeopardy prompt at inference a literal prefix of the training
data rather than an approximation of one.

**Shard dtype follows the vocabulary.** uint16 tops out at 65,535, which fits
SmolLM2's 49k vocabulary and does not fit Qwen3's 151,669. The large base
therefore writes uint32 and its corpus is twice the size on disk. This is
recorded in meta.json and read back by data.py; it is not a flag you set.

Notes:
  * Pure CPU. Run it in a free Kaggle CPU notebook, not locally: Kaggle pulls
    from HuggingFace far faster than a home connection, it costs no accelerator
    quota, and the output mounts into the training notebook through Add Input ->
    Notebook Output so nothing large crosses your uplink.
  * Safe to kill and rerun. Per-source document counts are recorded and skipped,
    and every finished shard is written atomically.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

import bases
import revtext

# Fast tokenizers fork their own thread pool. Inside mp.Pool that produces a
# deadlock warning at best and a hang at worst, and we are already parallel at
# the document level, so turn it off before transformers is imported anywhere.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_tok = None
_direction = "reverse"
_eos = 0
_markers = (0, 0)

ROLE_MAP = {
    "user": "user", "human": "user", "prompter": "user",
    "assistant": "assistant", "gpt": "assistant", "model": "assistant", "bot": "assistant",
}


# --------------------------------------------------------------------------- #
# encoding
# --------------------------------------------------------------------------- #


def _worker_init(hf_name: str, direction: str) -> None:
    global _tok, _direction, _eos, _markers
    _tok = revtext.load_tokenizer(hf_name)
    _direction = direction
    _eos = revtext.eos_id(_tok)
    _markers = revtext.markers(_tok)


def _encode(payload) -> list[int]:
    if isinstance(payload, str):
        ids = _tok(payload, add_special_tokens=False)["input_ids"]
    else:
        ids = revtext.chat_ids(_tok, payload)
    return revtext.orient(ids, _direction, _eos, _markers[0], _markers[1])


def normalize_messages(row, key: str) -> list[tuple[str, str]]:
    """Flatten one conversation row to [(role, text), ...], or [] if unusable.

    Handles both the {"role", "content"} schema (smoltalk, ultrachat, tulu) and
    the ShareGPT {"from", "value"} schema, since half of HuggingFace uses each.
    A system message is folded onto the front of the next user turn rather than
    getting a role of its own, because it appears in a minority of rows and a
    third role name would show up reversed in the middle of every generation.
    """
    raw = row.get(key) or row.get("messages") or row.get("conversations")
    if not raw:
        return []
    turns: list[tuple[str, str]] = []
    pending_system = ""
    for m in raw:
        if not isinstance(m, dict):
            return []
        role = str(m.get("role") or m.get("from") or "").lower()
        text = (m.get("content") or m.get("value") or "").strip()
        if not text:
            continue
        if role == "system":
            pending_system = text
            continue
        role = ROLE_MAP.get(role)
        if role is None:
            continue
        if role == "user" and pending_system:
            text = f"{pending_system}\n\n{text}"
            pending_system = ""
        # Merge consecutive same-role turns so the sequence always alternates.
        if turns and turns[-1][0] == role:
            turns[-1] = (role, turns[-1][1] + "\n\n" + text)
        else:
            turns.append((role, text))
    # A trailing user turn has no answer to reverse into, so drop it.
    while turns and turns[-1][0] == "user":
        turns.pop()
    return turns if len(turns) >= 2 else []


# --------------------------------------------------------------------------- #
# shards
# --------------------------------------------------------------------------- #


class ShardWriter:
    def __init__(self, out_dir: Path, prefix: str, shard_tokens: int, dtype: str) -> None:
        self.out_dir = out_dir
        self.prefix = prefix
        self.shard_tokens = shard_tokens
        self.dtype = np.dtype(dtype)
        self.buf = np.empty(shard_tokens, dtype=self.dtype)
        self.n = 0
        self.names: list[str] = []
        self.total = 0

    def add(self, ids: list[int]) -> None:
        arr = np.asarray(ids, dtype=self.dtype)
        pos = 0
        while pos < len(arr):
            room = self.shard_tokens - self.n
            take = min(room, len(arr) - pos)
            self.buf[self.n : self.n + take] = arr[pos : pos + take]
            self.n += take
            pos += take
            self.total += take
            if self.n == self.shard_tokens:
                self.flush()

    def flush(self) -> None:
        if self.n == 0:
            return
        name = f"{self.prefix}_{len(self.names):03d}.bin"
        tmp = self.out_dir / (name + ".tmp")
        self.buf[: self.n].tofile(tmp)
        os.replace(tmp, self.out_dir / name)
        gb = (self.n * self.dtype.itemsize) / 1e9
        print(f"  wrote {name}  {self.n:,} tokens  {gb:.2f} GB", flush=True)
        self.names.append(name)
        self.n = 0


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #

EMPTY_SOURCE = {"shards": [], "docs": 0, "tokens": 0, "val_docs": 0}


def read_state(out_dir: Path) -> dict:
    path = out_dir / "meta.json"
    if not path.exists():
        return {"sources": {"web": dict(EMPTY_SOURCE), "chat": dict(EMPTY_SOURCE)},
                "shards": {"train": [], "val": []}, "val_tokens": 0}
    state = json.loads(path.read_text())
    state.setdefault("shards", {"train": [], "val": []})
    state["shards"].setdefault("val", [])
    for name in ("web", "chat"):
        state.setdefault("sources", {}).setdefault(name, dict(EMPTY_SOURCE))
    return state


def write_meta(out_dir: Path, state: dict, args, base) -> None:
    src = state["sources"]
    meta = {
        "base": base.hf_name,
        "base_key": base.key,
        "tokenizer": base.hf_name,
        "vocab_size": base.vocab_size,
        "dtype": base.dtype,
        "direction": args.direction,
        "block_hint": base.block_size,
        "shards": {"train": src["web"]["shards"] + src["chat"]["shards"],
                   "val": state["shards"]["val"]},
        "sources": src,
        "chat_frac": args.chat_frac,
        "web_dataset": args.dataset,
        "chat_dataset": args.chat_dataset,
        "val_tokens": state.get("val_tokens", 0),
        "total_tokens": src["web"]["tokens"] + src["chat"]["tokens"],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    tmp = out_dir / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=1))
    os.replace(tmp, out_dir / "meta.json")


def open_streams(args, skip: dict):
    """Two streaming iterators of payloads, already positioned past skip[name]."""
    from datasets import load_dataset

    def web():
        ds = load_dataset(args.dataset, name=args.name, split=args.split, streaming=True)
        if skip["web"]:
            ds = ds.skip(skip["web"])
        for row in ds:
            text = row.get(args.text_key)
            if text:
                yield text

    def chat():
        kw = {"streaming": True, "split": args.chat_split}
        if args.chat_name:
            kw["name"] = args.chat_name
        ds = load_dataset(args.chat_dataset, **kw)
        if skip["chat"]:
            ds = ds.skip(skip["chat"])
        for row in ds:
            turns = normalize_messages(row, args.chat_key)
            if turns:
                yield turns

    return web(), chat()


def _chat_share(counts: dict) -> float:
    total = counts["web"] + counts["chat"]
    return counts["chat"] / total if total else 0.0


# --------------------------------------------------------------------------- #
# passes
# --------------------------------------------------------------------------- #


def build_val(out_dir: Path, state: dict, args, base, streams: dict) -> None:
    """One interleaved validation shard, alternating web and chat documents.

    Interleaved rather than concatenated because the trainer reads the front of
    the val split sequentially and only ever touches its first few hundred
    thousand tokens. Two separate val shards would give you a validation loss
    measured entirely on web text, which would look fine while the conversation
    half of the corpus silently failed.
    """
    if state["shards"]["val"]:
        print(f"val split already present: {state['val_tokens']:,} tokens")
        return

    _worker_init(base.hf_name, args.direction)
    target = int(args.val_tokens)
    writer = ShardWriter(out_dir, "val", target + 1_000_000, base.dtype)
    counts = {"web": 0, "chat": 0}
    print(f"building val split: {target:,} tokens")

    while writer.total < target:
        pick = "chat" if _chat_share(counts) < args.chat_frac else "web"
        try:
            payload = next(streams[pick])
        except StopIteration:
            pick = "web" if pick == "chat" else "chat"
            try:
                payload = next(streams[pick])
            except StopIteration:
                break
        ids = _encode(payload)
        writer.add(ids)
        counts[pick] += len(ids)
        state["sources"][pick]["val_docs"] += 1

    writer.flush()
    state["shards"]["val"] = writer.names
    state["val_tokens"] = writer.total
    print(f"  val done: {writer.total:,} tokens, {_chat_share(counts):.0%} conversations")
    write_meta(out_dir, state, args, base)


def run_source(out_dir: Path, state: dict, args, base, name: str, stream, target: int) -> None:
    """Tokenize one source up to its token target, resuming from recorded state."""
    src = state["sources"][name]
    if src["tokens"] >= target:
        print(f"{name}: already at {src['tokens'] / 1e9:.3f}B tokens, skipping")
        return

    writer = ShardWriter(out_dir, name, int(args.shard_tokens), base.dtype)
    writer.names = list(src["shards"])
    docs, tokens, done0 = src["docs"], src["tokens"], src["tokens"]
    t0 = time.time()
    print(f"{name}: {tokens / 1e9:.3f}B -> {target / 1e9:.3f}B tokens, "
          f"{args.workers} workers, skipping {docs:,} docs")

    with mp.Pool(args.workers, initializer=_worker_init,
                 initargs=(base.hf_name, args.direction)) as pool:
        try:
            for ids in pool.imap(_encode, stream, chunksize=args.batch):
                docs += 1
                writer.add(ids)
                tokens += len(ids)
                if docs % 10000 == 0:
                    el = time.time() - t0
                    rate = (tokens - done0) / max(1e-6, el)
                    eta = (target - tokens) / max(1.0, rate)
                    print(f"  {name} {tokens / 1e9:6.3f}B / {target / 1e9:.2f}B | "
                          f"{docs:,} docs | {rate / 1e6:.2f}M tok/s | eta {eta / 3600:.1f}h",
                          flush=True)
                    src.update(shards=writer.names, docs=docs, tokens=tokens)
                    write_meta(out_dir, state, args, base)
                if tokens >= target:
                    break
        except KeyboardInterrupt:
            print("\ninterrupted, flushing what we have")

    writer.flush()
    src.update(shards=writer.names, docs=docs, tokens=tokens)
    write_meta(out_dir, state, args, base)
    print(f"{name}: {tokens / 1e9:.3f}B tokens in {len(writer.names)} shards")


# --------------------------------------------------------------------------- #
# self test
# --------------------------------------------------------------------------- #


def synthetic_streams():
    """Stand-ins for the two real sources, so --self-test needs no network."""
    words = ("orbit magnet tissue harbour lattice enzyme ledger falcon "
             "prism tundra kettle marrow").split()

    def web():
        for i in range(6000):
            n = 40 + (i * 7) % 160
            yield " ".join(words[(i + j) % len(words)] for j in range(n))

    def chat():
        for i in range(6000):
            turns = []
            for t in range((i % 3) + 1):
                w = words[(i + t) % len(words)]
                turns.append(("user", f"what is a {w}?"))
                turns.append(("assistant", f"a {w} is {' '.join(words[:6 + t])}."))
            yield turns

    return web(), chat()


def self_test(args) -> None:
    import shutil
    import tempfile

    from data import Corpus

    tmp = Path(tempfile.mkdtemp(prefix="revtok-"))
    try:
        for key in ("small", "large"):
            base = bases.get(key)
            out_dir = tmp / key
            out_dir.mkdir()
            args.chat_frac = 0.25
            args.val_tokens = 20_000
            args.shard_tokens = 60_000
            args.workers = 2
            args.batch = 16
            target = 400_000
            chat_target = int(target * args.chat_frac)

            state = read_state(out_dir)
            web, chat = synthetic_streams()
            build_val(out_dir, state, args, base, {"web": web, "chat": chat})
            run_source(out_dir, state, args, base, "web", web, target - chat_target)
            run_source(out_dir, state, args, base, "chat", chat, chat_target)

            meta = json.loads((out_dir / "meta.json").read_text())
            assert meta["dtype"] == base.dtype, meta["dtype"]
            assert meta["base"] == base.hf_name
            assert len(meta["shards"]["train"]) > 1 and meta["shards"]["val"]
            assert not list(out_dir.glob("*.tmp")), "left a temp file behind"

            corpus = Corpus(out_dir, block_size=256, split="train")
            assert corpus.direction == args.direction
            assert corpus.dtype == base.dtype
            block = corpus.block(3)
            assert int(block.max()) < base.vocab_size, "token id outside the vocabulary"

            share = meta["sources"]["chat"]["tokens"] / meta["total_tokens"]
            assert abs(share - args.chat_frac) < 0.02, f"chat share {share:.3f}"

            # Resuming must not re-emit or skip documents.
            before = dict(state["sources"]["web"])
            state2 = read_state(out_dir)
            web2, _ = synthetic_streams()
            run_source(out_dir, state2, args, base, "web", web2, target - chat_target)
            assert state2["sources"]["web"]["tokens"] == before["tokens"], "resume re-tokenized"

            print(f"  {key}: {meta['total_tokens']:,} tokens as {meta['dtype']}, "
                  f"{len(meta['shards']['train'])} shards, {share:.0%} conversations")
        print("self test passed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="large", choices=sorted(bases.BASES),
                   help="which base model's tokenizer and vocabulary to build for")
    p.add_argument("--out-dir", default="data/rev")
    p.add_argument("--direction", choices=("reverse", "forward", "both"), default="reverse",
                   help="reverse trains backwards, forward is the control, both emits "
                        "every document twice with a leading mode marker")
    p.add_argument("--max-tokens", type=float, default=2e9)
    p.add_argument("--chat-frac", type=float, default=0.2,
                   help="share of the corpus that is conversations")
    p.add_argument("--val-tokens", type=float, default=5e6)
    p.add_argument("--shard-tokens", type=float, default=2.5e8)

    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--name", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--text-key", default="text")

    p.add_argument("--chat-dataset", default="HuggingFaceTB/smoltalk")
    p.add_argument("--chat-name", default="all")
    p.add_argument("--chat-split", default="train")
    p.add_argument("--chat-key", default="messages")

    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    p.add_argument("--batch", type=int, default=128, help="documents per worker chunk")
    p.add_argument("--verify-only", action="store_true",
                   help="exit 0 if the shards already cover --max-tokens")
    p.add_argument("--self-test", action="store_true",
                   help="run the whole pipeline on synthetic data in a temp dir, no network")
    args = p.parse_args()

    base = bases.get(args.base)
    if args.self_test:
        self_test(args)
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = read_state(out_dir)

    max_tokens = int(args.max_tokens)
    chat_target = int(max_tokens * args.chat_frac)
    web_target = max_tokens - chat_target
    web_done = state["sources"]["web"]["tokens"]
    chat_done = state["sources"]["chat"]["tokens"]
    have = web_done + chat_done
    complete = (web_done >= web_target and chat_done >= chat_target
                and bool(state["shards"]["val"]))

    if args.verify_only:
        if complete:
            print(f"shards complete: {have / 1e9:.3f}B tokens in {out_dir}")
            return
        raise SystemExit(f"shards incomplete: {have / 1e9:.3f}B of "
                         f"{max_tokens / 1e9:.3f}B tokens in {out_dir}")

    if complete:
        print(f"already at {have / 1e9:.3f}B tokens ({args.direction}), nothing to do")
        return
    if have:
        print(f"resuming: {have / 1e9:.3f}B tokens already written")

    print(bases.report(base))
    print(f"direction={args.direction}  target={max_tokens / 1e9:.2f}B  "
          f"web={web_target / 1e9:.2f}B  chat={chat_target / 1e9:.2f}B")
    print(f"expect about {max_tokens * base.bytes_per_token / 1e9:.1f} GB on disk "
          f"as {base.dtype}")
    if args.direction == "both":
        print("note: both mode emits every document twice, so double that again")

    skip = {n: state["sources"][n]["docs"] + state["sources"][n]["val_docs"]
            for n in ("web", "chat")}
    web, chat = open_streams(args, skip)
    build_val(out_dir, state, args, base, {"web": web, "chat": chat})
    run_source(out_dir, state, args, base, "web", web, web_target)
    run_source(out_dir, state, args, base, "chat", chat, chat_target)

    total = state["sources"]["web"]["tokens"] + state["sources"]["chat"]["tokens"]
    on_disk = sum(f.stat().st_size for f in out_dir.glob("*.bin")) / 1e9
    print(f"\ndone: {total / 1e9:.3f}B train tokens, {state['val_tokens']:,} val tokens, "
          f"{on_disk:.2f} GB on disk")
    print(f"upload {out_dir} as a Kaggle Dataset next")


if __name__ == "__main__":
    mp.freeze_support()
    main()
    # The HuggingFace streaming reader leaves background HTTP threads running.
    # If one is mid-request when the interpreter starts finalizing it touches a
    # dead GIL and aborts the process long after the shards are safely on disk,
    # and a non-zero exit there would take kaggle_run.sh down with it. So skip
    # finalization entirely.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
