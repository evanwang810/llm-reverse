#!/usr/bin/env python
"""Train a small GPT on pre-tokenized shards. Built for Kaggle 2x T4, 12h sessions.

Single GPU:
    python train.py --data-dir /kaggle/input/fineweb-edu-tokens
Dual T4:
    torchrun --nproc_per_node=2 train.py --data-dir /kaggle/input/fineweb-edu-tokens
Smoke test (no data needed, makes a synthetic corpus):
    python train.py --smoke-test

T4 notes:
  * Turing has no bf16 and no TF32, so this is fp16 + GradScaler. Loss and
    softmax are computed in fp32 inside autocast.
  * Flash attention needs sm80+. SDPA silently uses the memory-efficient or
    math backend here. That is expected, not a misconfiguration.
  * The two T4s talk over PCIe with no NVLink, so the gradient all-reduce is
    the main scaling tax. Grad accumulation (default 8) keeps sync infrequent.
    If DDP hangs at startup, set NCCL_P2P_DISABLE=1.
  * torch.compile is off by default. It usually works but costs a few minutes
    of warmup and occasionally breaks on Kaggle's torch build.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import signal
import sys
import threading
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import bases
from config import TrainConfig
from data import BatchSampler, Corpus, val_batches
import revtext
import basemodel
from basemodel import FinetuneModel, strip_prefixes
from runstate import RunDir, default_search_dirs, find_checkpoints

STOP_REQUESTED = False


def _handle_signal(signum, frame):  # noqa: ARG001
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"\n[signal {signum}] will save and exit at the next checkpoint boundary", flush=True)


# --------------------------------------------------------------------------- #
# setup
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    t = TrainConfig()

    g = p.add_argument_group("model")
    g.add_argument("--base", choices=sorted(bases.BASES), default="large",
                   help="which base checkpoint to finetune. Sets the batch shape, "
                        "lr and block size unless you override them explicitly")
    g.add_argument("--block-size", type=int, default=0,
                   help="0 takes the base entry's value")
    g.add_argument("--attn", default="sdpa",
                   help="attention implementation. sdpa everywhere; "
                        "flash_attention_2 needs sm80+, which the T4 is not")

    g = p.add_argument_group("optim")
    g.add_argument("--micro-batch", type=int, default=t.micro_batch)
    g.add_argument("--grad-accum", type=int, default=t.grad_accum)
    g.add_argument("--lr", type=float, default=t.lr)
    g.add_argument("--min-lr", type=float, default=t.min_lr)
    g.add_argument("--warmup-steps", type=int, default=t.warmup_steps)
    g.add_argument("--decay-steps", type=int, default=t.decay_steps)
    g.add_argument("--weight-decay", type=float, default=t.weight_decay)
    g.add_argument("--grad-clip", type=float, default=t.grad_clip)
    g.add_argument("--decay", action="store_true", help="enter the WSD decay phase immediately")
    g.add_argument("--auto-decay", action="store_true",
                   help="start decaying automatically so it finishes right before the deadline")
    g.add_argument("--max-steps", type=int, default=0, help="0 means run until the deadline")

    g = p.add_argument_group("io")
    g.add_argument("--data-dir", type=str, default="")
    g.add_argument("--run-dir", type=str, default="/kaggle/working/run")
    g.add_argument("--resume", choices=["auto", "off"], default="auto")
    g.add_argument("--resume-from", type=str, default="", help="explicit checkpoint path")
    g.add_argument("--allow-no-optimizer", action="store_true",
                   help="permit resuming from a weights-only checkpoint (expect a loss spike)")
    g.add_argument("--save-every-min", type=float, default=t.save_every_min)
    g.add_argument("--keep-checkpoints", type=int, default=t.keep_checkpoints,
                   help="rolling full checkpoints (with optimizer state) to keep")
    g.add_argument("--keep-weights", type=int, default=t.keep_weights,
                   help="rolling small fp16 inference copies to keep")
    g.add_argument("--milestone-every-min", type=float, default=t.milestone_every_min,
                   help="how often to keep a permanent fp16 copy, 0 disables")
    g.add_argument("--no-eval-copy", action="store_true", help="skip the small fp16 inference copy")

    g = p.add_argument_group("schedule")
    g.add_argument("--deadline-hours", type=float, default=t.deadline_hours)
    g.add_argument("--reserve-minutes", type=float, default=t.reserve_minutes)
    g.add_argument("--session-start", type=float, default=0.0,
                   help="unix ts the Kaggle session actually began; defaults to process start")
    g.add_argument("--log-every", type=int, default=t.log_every)
    g.add_argument("--eval-every", type=int, default=t.eval_every)
    g.add_argument("--eval-batches", type=int, default=t.eval_batches)
    g.add_argument("--seed", type=int, default=t.seed)
    g.add_argument("--sample-prompt", default=None,
                   help="generated from at every checkpoint, logged to samples.txt. "
                        "Defaults to the start of a passage on a forward run and to "
                        "the end of one on a reverse run")
    g.add_argument("--sample-tokens", type=int, default=48, help="0 disables sampling")

    g = p.add_argument_group("misc")
    g.add_argument("--compile", action="store_true")
    g.add_argument("--smoke-test", action="store_true",
                   help="tiny model, 5 minute deadline, saves every 30s, same code path")
    g.add_argument("--print-config-only", action="store_true")
    return p.parse_args()


def explicit_flags() -> set[str]:
    """Which options the user actually typed, so defaults never clobber them."""
    return {a.lstrip("-").split("=")[0].replace("-", "_")
            for a in sys.argv[1:] if a.startswith("--")}


def apply_base(args: argparse.Namespace) -> None:
    """Fill the batch shape from the base entry, unless it was passed explicitly.

    The shape is a property of what fits the device the entry is sized for, not
    a preference, so getting it from bases.py keeps the memory arithmetic in one
    place instead of in a notebook cell nobody re-derives.
    """
    base = bases.get(args.base)
    explicit = explicit_flags()
    for key in ("block_size", "micro_batch", "grad_accum", "lr", "warmup_steps"):
        if key not in explicit and not (key == "block_size" and args.block_size):
            setattr(args, key, getattr(base, key))
    if not args.block_size:
        args.block_size = base.block_size
    print(f"[base {base.key}] {base.hf_name}")


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    """Tiny model, short deadline, same code path. Anything you passed on the
    command line explicitly still wins."""
    overrides = {
        "block_size": 256, "micro_batch": 4, "grad_accum": 2, "lr": 3e-4,
        "warmup_steps": 20, "decay_steps": 50,
        "log_every": 5, "eval_every": 50, "eval_batches": 4,
        "save_every_min": 0.5, "deadline_hours": 5.0 / 60.0, "reserve_minutes": 0.2,
    }
    explicit = explicit_flags()
    for key, value in overrides.items():
        if key not in explicit:
            setattr(args, key, value)
    if not args.data_dir:
        args.data_dir = str(Path(args.run_dir) / "smoke_data")
    print("[smoke] tiny config, 5 minute deadline, checkpoint every 30s")


def make_synthetic_corpus(data_dir: Path, vocab_size: int = 49_152,
                          train_tokens: int = 3_000_000) -> None:
    """Zipf-ish random tokens so the smoke test needs no real dataset."""
    data_dir.mkdir(parents=True, exist_ok=True)
    if (data_dir / "meta.json").exists():
        return
    rng = np.random.default_rng(0)
    weights = 1.0 / np.arange(1, vocab_size + 1) ** 1.1
    weights /= weights.sum()
    for name, count in (("train_000.bin", train_tokens), ("val_000.bin", 200_000)):
        toks = rng.choice(vocab_size, size=count, p=weights).astype(np.uint16)
        toks.tofile(data_dir / name)
    (data_dir / "meta.json").write_text(
        json.dumps(
            {
                # Must match build_smoke()'s vocabulary. A corpus wider than
                # the model's embedding table fails as an IndexError deep inside
                # the forward pass, which is a confusing way to learn about it.
                "tokenizer": "HuggingFaceTB/SmolLM2-135M",
                "base": "HuggingFaceTB/SmolLM2-135M",
                "vocab_size": vocab_size,
                "dtype": "uint16",
                # Random tokens read the same in either direction, so this costs
                # nothing and makes the smoke test walk the reverse sampling and
                # rendering path rather than the forward one.
                "direction": "reverse",
                "shards": {"train": ["train_000.bin"], "val": ["val_000.bin"]},
            },
            indent=1,
        )
    )
    print(f"[smoke] wrote synthetic corpus to {data_dir}")


def setup_distributed():
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        # This timeout is a recovery-speed knob, not a safety margin. When a
        # rank stops reaching collectives, every other rank waits this long
        # before the watchdog fires, and only then can the restart loop resume
        # from the last checkpoint. Set at 60 minutes it turned each hang into
        # a full hour of dead time. Saves are asynchronous now, so the longest
        # legitimate pause is a couple of seconds of CPU snapshot, and a short
        # timeout costs nothing while making a hang cheap to recover from.
        minutes = float(os.environ.get("NCCL_TIMEOUT_MIN", "15"))
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=minutes))
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        world = dist.get_world_size()
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return True, rank, local_rank, world
    return False, 0, 0, 1


# --------------------------------------------------------------------------- #
# schedule
# --------------------------------------------------------------------------- #


@torch.no_grad()
def sample_text(raw_model, tok, device, prompt: str, max_new: int,
                direction: str = "forward", temperature: float = 0.8,
                top_k: int = 40) -> str:
    """A few tokens from a fixed prompt, so the log shows quality, not just loss.

    Rank 0 only, and it runs no collectives, so the other ranks simply wait a
    second at the next barrier. Never allowed to take the run down with it.

    On a reverse run the prompt is the *end* of the passage and the model writes
    what came before it, so the logged line is completion-then-prompt. Watching
    "and they lived happily ever after." grow a story to its left tells you
    things the loss curve does not.
    """
    if tok is None:
        return ""
    try:
        was_training = raw_model.training
        raw_model.eval()
        ids = revtext.encode_prompt(tok, prompt, direction)
        eos = revtext.eos_id(tok)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        for _ in range(max_new):
            window = idx[:, -raw_model.cfg.block_size :]
            logits, _ = raw_model(window)
            logits = logits[0, -1].float() / max(1e-4, temperature)
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.numel()))[0][-1]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt.view(1, 1)], dim=1)
            if int(nxt) == eos:
                break
        if was_training:
            raw_model.train()
        return " ".join(revtext.render(tok, idx[0].tolist(), direction).split())
    except Exception as exc:  # tokenizer missing, OOM, anything
        return f"<sampling failed: {type(exc).__name__}: {exc}>"


def lr_at(step: int, args: argparse.Namespace, decay_start: int | None) -> float:
    if step < args.warmup_steps:
        return args.lr * (step + 1) / max(1, args.warmup_steps)
    if decay_start is None or step < decay_start:
        return args.lr
    p = min(1.0, (step - decay_start) / max(1, args.decay_steps))
    # 1 - sqrt(p) is the usual WSD decay shape; it holds the LR high longer
    # than cosine and then drops hard, which is where most of the gain is.
    return args.min_lr + (args.lr - args.min_lr) * (1.0 - math.sqrt(p))


def phase_of(step: int, args: argparse.Namespace, decay_start: int | None) -> str:
    if step < args.warmup_steps:
        return "warmup"
    if decay_start is None or step < decay_start:
        return "constant"
    return "decay" if step < decay_start + args.decay_steps else "done"


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #


_save_thread: threading.Thread | None = None


def _write_payload(run_dir: Path, payload: dict, light: dict | None, step: int,
                   milestone: bool, keep: int, keep_weights: int) -> None:
    """The slow part: runs on a worker thread so training does not stall on it."""
    final = run_dir / f"ckpt_step{step:07d}.pt"
    tmp = final.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, final)  # atomic: a killed session never leaves a half file

    if light is not None:
        for prefix, limit in (("weights", keep_weights), ("milestone", 1 if milestone else 0)):
            if limit <= 0:
                continue
            lf = run_dir / f"{prefix}_step{step:07d}.pt"
            lt = lf.with_suffix(".pt.tmp")
            torch.save(light, lt)
            os.replace(lt, lf)

    from runstate import atomic_write

    atomic_write(run_dir / "latest.json", json.dumps({"step": step, "ckpt": final.name}))

    for prefix, limit in (("ckpt", keep), ("weights", keep_weights)):
        files = sorted(run_dir.glob(f"{prefix}_step*.pt"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        for old in files[max(0, limit):]:
            with contextlib.suppress(OSError):
                old.unlink()


def wait_for_save(timeout: float = 900.0) -> None:
    if _save_thread is not None and _save_thread.is_alive():
        _save_thread.join(timeout)


def save_checkpoint(
    run_dir: Path,
    raw_model: GPT,
    optimizer,
    scaler,
    step: int,
    decay_start: int | None,
    tokens_seen: int,
    args: argparse.Namespace,
    data_fingerprint: str,
    best_val: float | None,
    eval_copy: bool,
    keep: int,
    loss_ema: float | None = None,
    keep_weights: int = 8,
    milestone: bool = False,
    blocking: bool = False,
) -> Path:
    """Snapshot state to CPU here, hand the disk write to a thread.

    The copy costs a second or two of RAM bandwidth; the write costs minutes on
    Kaggle's working disk. Only the copy has to happen while the other ranks
    wait, so this is the difference between a 5 minute stall every save and a
    couple of seconds.
    """
    global _save_thread
    run_dir.mkdir(parents=True, exist_ok=True)
    wait_for_save()  # never overlap two writes

    def to_cpu(obj):
        if torch.is_tensor(obj):
            return obj.detach().to("cpu", copy=True)
        if isinstance(obj, dict):
            return {k: to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(to_cpu(v) for v in obj)
        return obj

    payload = {
        "step": step,
        "model": to_cpu(raw_model.state_dict()),
        "optimizer": to_cpu(optimizer.state_dict()),
        "scaler": scaler.state_dict(),
        "decay_start": decay_start,
        "tokens_seen": tokens_seen,
        "best_val": best_val,
        "loss_ema": loss_ema,
        "config": {"model": raw_model.cfg.as_dict(), "train": vars(args)},
        "direction": getattr(args, "direction", "forward"),
        "data_fingerprint": data_fingerprint,
        "saved_at": time.time(),
        "torch_version": torch.__version__,
    }
    # weights_* rolls over, milestone_* is kept forever so you can put an early
    # checkpoint and a late one against the same prompt afterwards.
    light = None
    if eval_copy and (keep_weights > 0 or milestone):
        light = {
            "step": step,
            "model": {k: v.detach().to("cpu", copy=True).half() if v.is_floating_point()
                      else v.detach().to("cpu", copy=True)
                      for k, v in raw_model.state_dict().items()},
            "config": {"model": raw_model.cfg.as_dict()},
            # chat.py and the dashboard load these weights-only files, and a
            # reverse model loaded as a forward one produces fluent nonsense
            # rather than an error, so the direction has to be here too.
            "direction": getattr(args, "direction", "forward"),
            "data_fingerprint": data_fingerprint,
        }

    args_tuple = (run_dir, payload, light, step, milestone, keep, keep_weights)
    if blocking:
        _write_payload(*args_tuple)
    else:
        _save_thread = threading.Thread(target=_write_payload, args=args_tuple,
                                        name="checkpoint-writer", daemon=False)
        _save_thread.start()
    return run_dir / f"ckpt_step{step:07d}.pt"


def pick_resume_checkpoint(args: argparse.Namespace) -> dict | None:
    if args.resume == "off":
        return None
    if args.resume_from:
        p = Path(args.resume_from)
        if not p.exists():
            raise FileNotFoundError(p)
        return {"path": p, "kind": "full" if p.name.startswith("ckpt") else "weights", "step": -1}
    cands = find_checkpoints([Path(args.run_dir), *default_search_dirs()])
    if not cands:
        return None
    full = [c for c in cands if c["kind"] == "full"]
    return full[0] if full else cands[0]


def load_resume(ckpt_info: dict, model: FinetuneModel, args: argparse.Namespace,
                data_fingerprint: str):
    path = ckpt_info["path"]
    print(f"resuming from {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    saved_cfg = ckpt["config"]["model"]
    mismatch = basemodel.compare_configs(saved_cfg, model.cfg.as_dict())
    if mismatch:
        raise SystemExit(
            "checkpoint does not match the requested base, refusing to load:\n"
            + "\n".join(f"  {k}: checkpoint={a} requested={b}" for k, (a, b) in mismatch.items())
            + "\nPass the same --base and --block-size as the run that made it."
        )
    if ckpt.get("data_fingerprint") and ckpt["data_fingerprint"] != data_fingerprint:
        print(f"WARNING: data fingerprint changed. checkpoint={ckpt['data_fingerprint']} "
              f"now={data_fingerprint}. Batch order will differ from the original run.")

    state = strip_prefixes(ckpt["model"])
    state = {f"hf.{k}": v.float() for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    tied = getattr(model.hf.config, "tie_word_embeddings", False)
    drop = ("rotary_emb", "inv_freq") + (("lm_head.weight",) if tied else ())
    unexpected = [k for k in unexpected if not any(d in k for d in drop)]
    missing = [k for k in missing if not any(d in k for d in drop)]
    if missing or unexpected:
        raise SystemExit(f"state dict mismatch. missing={missing} unexpected={unexpected}")

    has_optim = "optimizer" in ckpt and ckpt["optimizer"] is not None
    if not has_optim and not args.allow_no_optimizer:
        raise SystemExit(
            f"{path} has no optimizer state. Resuming without Adam moments causes a large loss "
            "spike. Point --resume-from at a ckpt_step*.pt file, or pass --allow-no-optimizer "
            "if you really want that."
        )
    return ckpt, has_optim


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> None:
    args = parse_args()
    apply_base(args)
    if args.smoke_test:
        apply_smoke_defaults(args)

    process_start = time.time()
    session_start = args.session_start or process_start
    deadline = session_start + args.deadline_hours * 3600 - args.reserve_minutes * 60

    ddp, rank, local_rank, world = setup_distributed()
    master = rank == 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda"

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.backends.cudnn.benchmark = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(ValueError, OSError):
            signal.signal(sig, _handle_signal)

    run_dir = Path(args.run_dir)
    rs = RunDir(run_dir)

    if args.smoke_test and master:
        make_synthetic_corpus(Path(args.data_dir))
    if ddp:
        dist.barrier()

    if not args.data_dir:
        raise SystemExit("--data-dir is required (or use --smoke-test)")

    train_corpus = Corpus(args.data_dir, args.block_size, "train")
    try:
        val_corpus = Corpus(args.data_dir, args.block_size, "val")
    except (ValueError, KeyError):
        val_corpus = None

    if args.smoke_test:
        model = basemodel.build_smoke(args.block_size, train_corpus.direction, device)
    else:
        if train_corpus.base != bases.get(args.base).hf_name:
            raise SystemExit(
                f"the shards were tokenized with {train_corpus.base} but --base "
                f"{args.base} is {bases.get(args.base).hf_name}. Token ids from one "
                "tokenizer are meaningless to another model and nothing downstream "
                "would raise, so this is refused here.")
        model = basemodel.build(args.base, args.block_size, train_corpus.direction,
                                device, args.attn)
    cfg = model.cfg
    model_non_emb = model.non_embedding_params()

    tokens_per_step = args.micro_batch * args.grad_accum * world * args.block_size
    if master:
        print("=" * 78)
        print(model.param_report())
        print(f"corpus: {train_corpus.total_tokens / 1e9:.3f}B tokens, "
              f"{train_corpus.n_blocks:,} blocks, base={train_corpus.base}, "
              f"{train_corpus.dtype}")
        print(f"batch: {args.micro_batch} x {args.grad_accum} accum x {world} gpu x {args.block_size} "
              f"= {tokens_per_step:,} tokens/step")
        print(f"1B tokens = {1e9 / tokens_per_step:,.0f} steps | one epoch = "
              f"{train_corpus.total_tokens / tokens_per_step:,.0f} steps")
        print(f"device={device} fp16={use_fp16} ddp={ddp} world={world}")
        print(f"deadline in {(deadline - time.time()) / 3600:.2f}h (reserving "
              f"{args.reserve_minutes:.0f} min for the final save)")
        print("=" * 78, flush=True)
    if args.print_config_only:
        return

    optimizer = model.configure_optimizer(args.lr, args.weight_decay, (0.9, 0.95), device.type)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    except (AttributeError, TypeError):  # older torch
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    step = 0
    tokens_seen = 0
    decay_start: int | None = None
    best_val: float | None = None
    resumed_ema: float | None = None
    fingerprint = train_corpus.fingerprint()
    # The training loop itself is direction agnostic: the shards are already
    # laid out in the order the model reads them. Only sampling and inference
    # care, so resolve it here and let it ride along in the checkpoint.
    args.direction = train_corpus.direction
    if args.sample_prompt is None:
        args.sample_prompt = ("and they lived happily ever after."
                              if revtext.is_reverse(args.direction) else "Once upon a time")

    info = pick_resume_checkpoint(args)
    if info is not None:
        ckpt, has_optim = load_resume(info, model, args, fingerprint)
        step = int(ckpt["step"])
        tokens_seen = int(ckpt.get("tokens_seen", step * tokens_per_step))
        decay_start = ckpt.get("decay_start")
        best_val = ckpt.get("best_val")
        resumed_ema = ckpt.get("loss_ema")  # keeps the smoothed curve continuous across restarts
        if has_optim:
            optimizer.load_state_dict(ckpt["optimizer"])
            # Adam moments must live on the compute device or the first step is slow / errors.
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        if ckpt.get("scaler") and use_fp16:
            # Restoring the loss scale matters: a fresh scaler restarts at 65536
            # and burns several skipped steps, which looks exactly like a loss spike.
            scaler.load_state_dict(ckpt["scaler"])
        del ckpt
        if master:
            print(f"resumed at step {step}, tokens_seen {tokens_seen / 1e9:.3f}B, "
                  f"decay_start={decay_start}, optimizer_state={'yes' if has_optim else 'NO'}",
                  flush=True)
    elif master:
        print("no checkpoint found, initializing fresh", flush=True)

    if args.decay and decay_start is None:
        decay_start = step
        if master:
            print(f"decay phase starting at step {step}")

    if args.compile:
        if master:
            print("compiling (this takes a few minutes on Kaggle)...", flush=True)
        model = torch.compile(model)

    raw_model = model
    if ddp:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None,
                    gradient_as_bucket_view=True)
        raw_model = model.module
    if args.compile:
        raw_model = getattr(raw_model, "_orig_mod", raw_model)

    sample_tok = None
    if args.sample_tokens > 0 and master:
        try:
            sample_tok = revtext.load_tokenizer(train_corpus.base)
        except Exception as exc:  # sampling must never end a run
            print(f"samples disabled: {type(exc).__name__}: {exc}")
    sampler = BatchSampler(train_corpus, args.micro_batch, args.grad_accum, world, rank, args.seed)
    autocast = (torch.amp.autocast("cuda", dtype=torch.float16)
                if use_fp16 else contextlib.nullcontext())

    if master:
        rs.init_csv()
    loss_ema: float | None = resumed_ema
    last_sample = ""
    last_save = time.time()
    last_milestone = time.time()
    saved_at_step = step
    last_log_t = time.time()
    last_log_step = step
    secs_per_step = 0.0
    lr = lr_at(step, args, decay_start)  # so the final status is valid even if we break at once
    val_loss: float | None = None
    stop_reason = ""

    def evaluate() -> float | None:
        if val_corpus is None:
            return None
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for x, y in val_batches(val_corpus, args.micro_batch, args.eval_batches, device):
                with autocast:
                    _, loss = raw_model(x, y)
                total += loss.item()
                n += 1
        model.train()
        avg = total / max(1, n)
        if ddp:
            t = torch.tensor([avg], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            avg = t.item()
        return avg

    model.train()
    if master:
        print("training", flush=True)

    while True:
        # ---- control flags, agreed across ranks ----
        flags = torch.zeros(3, dtype=torch.int32, device=device)
        if master:
            hit = rs.poll_flags()
            flags[0] = int("SAVE_NOW" in hit)
            flags[1] = int("DECAY_NOW" in hit)
            flags[2] = int("STOP_NOW" in hit)
        if STOP_REQUESTED:
            flags[2] = 1
        if time.time() > deadline:
            flags[2] = 1
            stop_reason = stop_reason or "deadline"
        if args.max_steps and step >= args.max_steps:
            flags[2] = 1
            stop_reason = stop_reason or "max-steps"
        if args.auto_decay and decay_start is None and secs_per_step > 0:
            if deadline - time.time() <= args.decay_steps * secs_per_step * 1.05:
                flags[1] = 1
        if ddp:
            dist.all_reduce(flags, op=dist.ReduceOp.MAX)
        force_save, want_decay, want_stop = (bool(flags[i].item()) for i in range(3))

        if want_decay and decay_start is None:
            decay_start = step
            if master:
                print(f"[step {step}] entering decay phase over {args.decay_steps} steps", flush=True)
        if decay_start is not None and step >= decay_start + args.decay_steps:
            want_stop = True
            stop_reason = stop_reason or "decay-complete"
        if want_stop:
            stop_reason = stop_reason or "requested"
            break

        # ---- one optimizer step ----
        lr = lr_at(step, args, decay_start)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device)
        for micro in range(args.grad_accum):
            x, y = sampler.batch(step, micro, device)
            sync_ctx = (model.no_sync() if ddp and micro < args.grad_accum - 1
                        else contextlib.nullcontext())
            with sync_ctx:
                with autocast:
                    _, loss = model(x, y)
                loss_sum += loss.detach()
                scaler.scale(loss / args.grad_accum).backward()

        grad_norm = 0.0
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)  # must unscale before clipping
            grad_norm = float(torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip))
        scaler.step(optimizer)
        scaler.update()

        step += 1
        tokens_seen += tokens_per_step

        # ---- logging ----
        if step % args.log_every == 0 or force_save:
            lossf = (loss_sum / args.grad_accum).clone()
            if ddp:
                dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
            lossf = lossf.item()
            loss_ema = lossf if loss_ema is None else 0.9 * loss_ema + 0.1 * lossf
            now = time.time()
            dt = max(1e-6, now - last_log_t)
            steps_done = max(1, step - last_log_step)
            secs_per_step = dt / steps_done
            tok_per_s = steps_done * tokens_per_step / dt
            last_log_t, last_log_step = now, step

            if master:
                remaining = max(0.0, deadline - now)
                print(f"step {step:>7} | loss {lossf:.4f} (ema {loss_ema:.4f}) | lr {lr:.2e} | "
                      f"{tok_per_s / 1e3:.1f}k tok/s | gn {grad_norm:.2f} | "
                      f"scale {scaler.get_scale():.0f} | {tokens_seen / 1e9:.3f}B tok | "
                      f"{remaining / 3600:.2f}h left", flush=True)
                rs.append_csv({
                    "step": step, "wall_s": round(now - session_start, 1), "tokens": tokens_seen,
                    "lr": f"{lr:.6e}", "loss": f"{lossf:.5f}", "loss_ema": f"{loss_ema:.5f}",
                    "val_loss": "" if val_loss is None else f"{val_loss:.5f}",
                    "grad_norm": f"{grad_norm:.3f}", "scale": scaler.get_scale(),
                    "tok_per_s": round(tok_per_s, 1),
                    "phase": phase_of(step, args, decay_start),
                })
                rs.write_status({
                    "step": step, "tokens_seen": tokens_seen, "loss": lossf, "loss_ema": loss_ema,
                    "val_loss": val_loss, "best_val": best_val, "lr": lr,
                    "tok_per_s": tok_per_s, "secs_per_step": secs_per_step,
                    "phase": phase_of(step, args, decay_start), "decay_start": decay_start,
                    "decay_steps": args.decay_steps, "max_steps": args.max_steps or None,
                    "elapsed_s": now - session_start, "remaining_s": remaining,
                    "deadline_hours": args.deadline_hours,
                    "epoch": sampler.epoch_at(step),
                    "world_size": world, "tokens_per_step": tokens_per_step,
                    "non_embedding_params": model_non_emb,
                    "last_save_step": saved_at_step,
                    "last_save_ago_s": round(time.time() - last_save, 1),
                    "sample_prompt": args.sample_prompt, "sample": last_sample,
                    "saving": _save_thread is not None and _save_thread.is_alive(),
                    "run_dir": str(run_dir), "pid": os.getpid(), "alive": True,
                })

        # ---- eval ----
        if args.eval_every and step % args.eval_every == 0:
            val_loss = evaluate()
            if val_loss is not None:
                if best_val is None or val_loss < best_val:
                    best_val = val_loss
                if master:
                    print(f"step {step:>7} | val loss {val_loss:.4f} (best {best_val:.4f})", flush=True)

        # ---- periodic save ----
        if force_save or (time.time() - last_save) / 60.0 >= args.save_every_min:
            is_milestone = (args.milestone_every_min > 0
                            and (time.time() - last_milestone) / 60.0 >= args.milestone_every_min)
            if master:
                p = save_checkpoint(run_dir, raw_model, optimizer, scaler, step, decay_start,
                                    tokens_seen, args, fingerprint, best_val,
                                    not args.no_eval_copy, args.keep_checkpoints, loss_ema,
                                    args.keep_weights, is_milestone)
                print(f"saving {p.name} in the background"
                      + ("  [milestone kept]" if is_milestone else ""), flush=True)
                if args.sample_tokens > 0:
                    last_sample = sample_text(raw_model, sample_tok, device,
                                              args.sample_prompt, args.sample_tokens,
                                              args.direction)
                    # Windows consoles default to cp1252, so re-encode for stdout
                    # specifically. Sampling must never be able to end a run.
                    shown = last_sample.encode(sys.stdout.encoding or "utf-8",
                                               errors="replace").decode(
                        sys.stdout.encoding or "utf-8", errors="replace")
                    print(f'  sample @ {step}: "{shown}"', flush=True)
                    with open(run_dir / "samples.txt", "a", encoding="utf-8",
                              errors="replace") as f:
                        f.write(f"step {step}\tloss {loss_ema or float('nan'):.4f}\t"
                                f"{last_sample}\n")
            last_save = time.time()
            if is_milestone:
                last_milestone = last_save
            saved_at_step = step
            if ddp:
                dist.barrier()

    # ---- final save, always ----
    if master:
        print(f"stopping: {stop_reason}. writing final checkpoint", flush=True)
        p = save_checkpoint(run_dir, raw_model, optimizer, scaler, step, decay_start, tokens_seen,
                            args, fingerprint, best_val, not args.no_eval_copy,
                            max(args.keep_checkpoints, 2), loss_ema,
                            args.keep_weights, True,  # the final state is always a milestone
                            blocking=True)            # and must be on disk before we exit
        rs.write_status({
            "step": step, "tokens_seen": tokens_seen, "loss": loss_ema, "loss_ema": loss_ema,
            "val_loss": val_loss, "best_val": best_val,
            "phase": phase_of(step, args, decay_start),
            "decay_start": decay_start, "decay_steps": args.decay_steps,
            "max_steps": args.max_steps or None,
            "elapsed_s": time.time() - session_start, "remaining_s": 0,
            "deadline_hours": args.deadline_hours, "epoch": sampler.epoch_at(step),
            "tok_per_s": None, "secs_per_step": secs_per_step, "lr": lr,
            "alive": False, "stop_reason": stop_reason, "final_checkpoint": p.name,
            "run_dir": str(run_dir), "last_save_step": step,
            "tokens_per_step": tokens_per_step, "world_size": world,
            "non_embedding_params": model_non_emb,
        })
        print(f"final: {p}  step={step}  tokens={tokens_seen / 1e9:.3f}B", flush=True)
        print("upload /kaggle/working/run as a new version of your checkpoint dataset", flush=True)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
