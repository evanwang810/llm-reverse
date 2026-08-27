#!/usr/bin/env python
"""Prove the TPU actually works before committing a session to it.

Five gates, each with its own budget, cheapest first. Any failure exits
non-zero with a specific reason, so kaggle_run.sh can stop before it has spent
anything. The whole thing is meant to fit in about ten minutes.

    python tpu_preflight.py --base 67m
    python tpu_preflight.py --base 67m --min-tok-s 120000 --budget-min 12

The gates, in order:

  1  torch_xla imports and reports a TPU              (seconds)
  2  arithmetic on device gives the right answer      (seconds)
  3  all replicas start and can all_reduce            (a minute)
  4  the real model trains at a real speed            (a few minutes)
  5  a checkpoint round trips through a save and load (a minute)

Gate 4 is the one that matters. A TPU that is present, passes every
correctness check and still trains at 9k tokens/sec is the expensive failure
mode: nothing raises, the session looks healthy, and eleven hours later there
is a quarter of a model. Two things cause it and gate 4 checks for both.

The first is silent fallback: an op with no XLA lowering runs on the host, and
every step pays a device-to-host round trip. The second is recompilation. XLA
compiles one program per distinct graph, so anything that changes shape or
bakes a changing Python scalar into the graph makes it compile again, and
compilation costs far more than a step. Gate 4 deliberately varies the
learning rate across its measured steps, which is the thing most likely to
trigger it, then reads the CompileTime counter to see whether the count
plateaued. If it is still climbing, the run would spend its life compiling.

Nothing here can damage a TPU. The risk being managed is a wasted session.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Kaggle's TPU VM image presets these for its own launcher. torch_xla reads
# them at import and comes up in a half configured distributed mode if they are
# left in place, so they have to go before the first import, not after.
for _var in ("TPU_PROCESS_ADDRESSES", "CLOUD_TPU_TASK_ID"):
    os.environ.pop(_var, None)
os.environ.setdefault("PJRT_DEVICE", "TPU")

RESULT_ENV = "LLM67M_PREFLIGHT_OUT"


def log(msg: str) -> None:
    print(msg, flush=True)


def fail(gate: str, why: str, fix: str = "") -> None:
    log("")
    log(f"PREFLIGHT FAILED at {gate}")
    log(f"  {why}")
    if fix:
        log(f"  {fix}")
    raise SystemExit(2)


# --------------------------------------------------------------------------- #
# gates 1 and 2: does a TPU exist and can it add
# --------------------------------------------------------------------------- #


def gate_1_probe():
    """Version and device count, from a subprocess.

    Reading this in-process would initialise the XLA runtime and claim the TPU,
    and xmp.spawn below forks from this very process. It would then find the
    device already held and abort inside the runtime with an error that looks
    like a bug in the training code.
    """
    from xla_probe import probe

    t0 = time.time()
    facts = probe(cwd=str(Path(__file__).resolve().parent))
    if facts.get("xla_error"):
        fail("gate 1 (import)", f"torch_xla did not come up: {facts['xla_error']}",
             "The notebook accelerator is not set to TPU. Change it in the sidebar; "
             "do not pip install torch_xla, Kaggle's TPU image already has a matching build.")
    if facts.get("xla_api_error"):
        fail("gate 1 (import)", f"the torch_xla API did not resolve: {facts['xla_api_error']}",
             "Add the new spelling to xla_compat.py.")

    kind, n = facts.get("xla_device_type"), facts.get("xla_devices", 0)
    log(f"  torch_xla {facts.get('torch_xla')}  device_type={kind}  devices={n}")
    log(f"  {facts.get('xla_api', '')}")
    if kind != "TPU":
        fail("gate 1 (import)", f"torch_xla came up on {kind!r}, not TPU",
             "Set the accelerator to TPU VM v3-8 and restart the session.")
    if n and n < 8:
        log(f"  WARNING: {n} devices, expected 8 on a v3-8. Continuing.")
    log(f"  gate 1 ok ({time.time() - t0:.1f}s)")
    return facts


# --------------------------------------------------------------------------- #
# gates 3, 4, 5: run inside the spawned replicas
# --------------------------------------------------------------------------- #


def _mp_fn(index, opts: dict):
    import torch
    import torch_xla.debug.metrics as met

    import xla_compat as X
    import basemodel

    dev = X.device()
    ordinal = X.ordinal()
    world = X.world_size()
    master = ordinal == 0

    def note(msg):
        if master:
            print(msg, flush=True)

    # ---- gate 2: arithmetic on device is correct ----
    t0 = time.time()
    a = torch.arange(1024, dtype=torch.float32, device=dev).reshape(32, 32)
    got = float((a @ a.T).sum().cpu())
    ref = torch.arange(1024, dtype=torch.float64).reshape(32, 32)
    want = float((ref @ ref.T).sum())
    if abs(got - want) / want > 1e-4:
        raise RuntimeError(f"matmul on device gave {got:.6g}, expected {want:.6g}")
    bf16 = False
    try:
        with torch.autocast("xla", dtype=torch.bfloat16):
            bf16 = (a @ a.T).dtype == torch.bfloat16
    except Exception as e:
        note(f"  WARNING: autocast('xla', bfloat16) unavailable ({type(e).__name__}), fp32")
    note(f"  arithmetic correct, bf16 autocast: {'yes' if bf16 else 'NO (fp32 fallback)'}")
    note("  gate 2 ok")

    # ---- gate 3: every replica is real and can talk ----
    t0 = time.time()
    probe = torch.tensor([float(ordinal)], device=dev)
    total = X.all_reduce_sum(probe)
    got = float(total.cpu())
    want = world * (world - 1) / 2
    if abs(got - want) > 0.5:
        raise RuntimeError(f"all_reduce over {world} replicas gave {got}, expected {want}")
    note(f"  {world} replicas up, all_reduce correct ({time.time() - t0:.1f}s)")
    note("  gate 3 ok")

    # ---- gate 4: the real model at real shapes ----
    # The point of the preflight is to prove the device can hold and step the
    # model you are about to spend a session on, so it builds the real base
    # rather than a stand-in. Downloading it here also warms the cache before
    # eight replicas ask for it at once.
    model = basemodel.build(opts["base"], opts["block_size"],
                            attn=basemodel.attn_for("tpu")).to(dev)
    cfg = model.cfg
    opt = model.configure_optimizer(1e-3, 0.1, (0.9, 0.95), "xla")

    mb, ga, bs = opts["micro_batch"], opts["grad_accum"], opts["block_size"]
    tokens_per_step = mb * ga * world * bs
    # One distinct pair per micro step. Reusing a single pair lets XLA's common
    # subexpression elimination collapse the accumulation into a single forward
    # and backward, so the measured graph is grad_accum times smaller than the
    # real one: it overstates throughput and, worse, hides an out-of-memory that
    # only appears once training builds the real graph.
    batches = [(torch.randint(0, cfg.vocab_size, (mb, bs), device=dev),
                torch.randint(0, cfg.vocab_size, (mb, bs), device=dev))
               for _ in range(ga)]

    def autocast():
        try:
            return torch.autocast("xla", dtype=torch.bfloat16)
        except Exception:
            import contextlib
            return contextlib.nullcontext()

    def one_step(lr: float):
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        for x, y in batches:
            with autocast():
                _, loss = model(x, y)
            (loss / ga).backward()
            X.sync()  # matches the trainer: one micro batch per graph
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        X.optimizer_step(opt)

    model.train()
    warm = opts["warmup_steps"]
    note(f"  compiling and warming up ({warm} steps, first one is slow)...")
    t0 = time.time()
    for i in range(warm):
        one_step(1e-3)
    X.sync()
    X.wait()
    warm_s = time.time() - t0

    def compiles():
        d = met.metric_data("CompileTime")
        return int(d[0]) if d else -1

    after_warmup = compiles()
    note(f"  warmup done in {warm_s:.1f}s, {after_warmup} compilations so far")

    # The learning rate moves on every one of these steps, exactly as it will
    # during warmup and decay in the real run. If a changing scalar forces a new
    # graph, the compile counter climbs here and the throughput collapses.
    meas = opts["measure_steps"]
    t0 = time.time()
    for i in range(meas):
        one_step(1e-3 * (1.0 - 0.5 * i / max(1, meas)))
    X.sync()
    X.wait()
    dt = time.time() - t0
    after_measure = compiles()

    tok_s = meas * tokens_per_step / max(1e-6, dt)
    note(f"  {meas} steps in {dt:.1f}s -> {dt / meas:.3f}s/step, {tok_s / 1e3:.0f}k tok/s")
    note(f"  compilations during measurement: {after_measure - after_warmup}")

    # ---- gate 5: a checkpoint survives a round trip ----
    t0 = time.time()
    tmp = Path(opts["run_dir"]) / "preflight_ckpt.pt"
    if master:
        tmp.parent.mkdir(parents=True, exist_ok=True)
    ref = next(iter(model.state_dict().values())).detach().float().cpu().clone()
    X.save({"model": model.state_dict(), "step": 0}, str(tmp))
    X.rendezvous("saved")
    ok_ckpt = True
    if master:
        back = torch.load(tmp, map_location="cpu", weights_only=False)
        got = next(iter(back["model"].values())).float()
        ok_ckpt = bool(torch.allclose(ref, got, atol=1e-5))
        tmp.unlink(missing_ok=True)
    note(f"  checkpoint round trip {'ok' if ok_ckpt else 'MISMATCH'} ({time.time() - t0:.1f}s)")

    if master:
        Path(os.environ[RESULT_ENV]).write_text(json.dumps({
            "world": world,
            "tokens_per_step": tokens_per_step,
            "secs_per_step": dt / meas,
            "tok_per_s": tok_s,
            "warmup_s": warm_s,
            "compiles_warmup": after_warmup,
            "compiles_measure": after_measure - after_warmup,
            "checkpoint_ok": ok_ckpt,
        }))


# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="67m")
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--warmup-steps", type=int, default=6)
    ap.add_argument("--measure-steps", type=int, default=20)
    ap.add_argument("--run-dir", default="/kaggle/working/run")
    ap.add_argument("--min-tok-s", type=float, default=80_000,
                    help="abort below this; the dual T4 path does about 42k")
    ap.add_argument("--max-compiles", type=int, default=3,
                    help="allowed recompiles during the measured steps")
    ap.add_argument("--budget-min", type=float, default=10.0)
    args = ap.parse_args()

    started = time.time()
    log("=" * 70)
    log("TPU preflight")
    log("=" * 70)

    gate_1_probe()

    out = Path(args.run_dir) / "preflight.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    os.environ[RESULT_ENV] = str(out)

    opts = {
        "base": args.base, "micro_batch": args.micro_batch,
        "grad_accum": args.grad_accum, "block_size": args.block_size,
        "warmup_steps": args.warmup_steps, "measure_steps": args.measure_steps,
        "run_dir": args.run_dir,
    }

    log("  spawning replicas (gates 3-5)...")
    import torch_xla.distributed.xla_multiprocessing as xmp

    try:
        # fork, not spawn: a Kaggle notebook process cannot re-exec itself.
        xmp.spawn(_mp_fn, args=(opts,), start_method="fork")
    except Exception as e:
        fail("gate 3-5 (spawn)", f"{type(e).__name__}: {e}",
             "Replicas failed to start. This is usually a stale TPU lock from an earlier "
             "cell in the same session; restart the kernel and run this cell first.")

    if not out.exists():
        fail("gate 4 (throughput)", "replicas exited without writing a result",
             "Check the traceback above; the master replica died mid-measurement.")

    r = json.loads(out.read_text())
    log("")
    log("-" * 70)
    log(f"  replicas          {r['world']}")
    log(f"  tokens/step       {r['tokens_per_step']:,}")
    log(f"  seconds/step      {r['secs_per_step']:.3f}")
    log(f"  throughput        {r['tok_per_s'] / 1e3:.0f}k tok/s")
    log(f"  recompiles        {r['compiles_measure']} during measurement")
    log(f"  preflight took    {(time.time() - started) / 60:.1f} min")
    log("-" * 70)

    if not r["checkpoint_ok"]:
        fail("gate 5 (checkpoint)", "a saved tensor did not match after reloading",
             "Training would produce checkpoints you cannot trust. Stop here.")

    if r["compiles_measure"] > args.max_compiles:
        fail("gate 4 (recompilation)",
             f"XLA compiled {r['compiles_measure']} new graphs while only the learning rate "
             "changed, so it will keep compiling for the whole run",
             "The step graph is not static. Compilation costs more than a step, so this "
             "would be slower than the T4 path even though nothing errors.")

    if r["tok_per_s"] < args.min_tok_s:
        fail("gate 4 (throughput)",
             f"{r['tok_per_s'] / 1e3:.0f}k tok/s is below the {args.min_tok_s / 1e3:.0f}k floor",
             "The TPU is running but something is falling back to the host. A dual T4 does "
             "about 42k tok/s, so this session is not worth spending.")

    speedup = r["tok_per_s"] / 42_000
    log("")
    log(f"PREFLIGHT PASSED  ({speedup:.1f}x the dual T4 path)")
    est = r["secs_per_step"]
    log(f"  at {est:.3f}s/step, 10 hours is about {int(10 * 3600 / est):,} steps "
        f"and {10 * 3600 / est * r['tokens_per_step'] / 1e9:.1f}B tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
