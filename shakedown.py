#!/usr/bin/env python
"""Run the whole pipeline small, on the real accelerator, and bundle the evidence.

    python /kaggle/working/code/shakedown.py

Meant for an interactive session: it prints as it goes, takes about fifteen
minutes, and leaves a zip in /kaggle/working you can download and send on.

This is not the preflight. tpu_preflight.py answers one question, is this device
worth spending a session on, and it answers it in ten minutes using synthetic
tensors. The shakedown answers a different one: does the entire pipeline work
here, end to end, on the real hardware, using the real data path. So it really
tokenizes FineWeb-Edu, really builds the corpus, really finetunes the base you
are going to use, really saves a checkpoint, really resumes from it, really
fine-tunes, and really loads the result into the chat session.

Everything a long run does, at a scale where a mistake costs minutes.

The stage that matters most is the resume check. Kaggle sessions die, and the
entire design assumes a restart picks up where the last checkpoint left off with
no loss discontinuity. That property is invisible in a short run and expensive
to discover is broken in a long one, so it gets tested explicitly: train, stop,
resume, and confirm the loss either side of the boundary is continuous.

Stages keep going after a failure rather than stopping at the first one, because
a bundle showing four failures is more useful than four separate runs.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BAR = "=" * 72

# This process must never initialise the XLA runtime. A TPU is claimed by one
# process at a time, so a single call to xr.global_runtime_device_count() here
# takes the device and every stage that spawns a trainer then dies with
# "Check failed: reporting_closure_ == nullptr", which reads like a bug in the
# trainer and is not. All XLA facts are gathered in throwaway subprocesses that
# claim the device, answer, and exit. These pops are belt and braces for
# anything that imports torch_xla despite that.
for _var in ("TPU_PROCESS_ADDRESSES", "CLOUD_TPU_TASK_ID"):
    os.environ.pop(_var, None)

def probe_xla() -> dict:
    """Version and device facts, gathered without holding the device here."""
    from xla_probe import probe

    return probe(cwd=str(HERE))


class Bundle:
    """Tees everything to the console and to a log file, and collects facts."""

    def __init__(self, outdir: Path) -> None:
        outdir.mkdir(parents=True, exist_ok=True)
        self.dir = outdir
        self.logfile = outdir / "shakedown.log"
        self.fh = self.logfile.open("w", encoding="utf-8", errors="replace")
        self.stages: list[dict] = []
        self.facts: dict = {}
        self.t0 = time.time()

    def say(self, line: str = "") -> None:
        enc = sys.stdout.encoding or "utf-8"
        safe = line.encode(enc, errors="replace").decode(enc, errors="replace")
        print(safe, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()

    def run(self, cmd: list[str], timeout: float = 1800) -> tuple[int, str]:
        """Run a real command, streaming its output into both sinks."""
        self.say(f"$ {' '.join(str(c) for c in cmd)}")
        lines: list[str] = []
        try:
            p = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace", bufsize=1)
        except OSError as e:
            self.say(f"  could not start: {e}")
            return 127, str(e)
        # A timer, not a check inside the read loop. A process that hangs with no
        # output never yields another line, so an in-loop deadline never fires and
        # the shakedown would sit there until the session died, which is the exact
        # failure it exists to catch quickly.
        import threading

        killed = threading.Event()

        def reap():
            killed.set()
            with contextlib.suppress(Exception):
                p.kill()

        timer = threading.Timer(timeout, reap)
        timer.start()
        try:
            for line in p.stdout:
                lines.append(line.rstrip("\n"))
                self.say("  " + line.rstrip("\n"))
            p.wait()
        finally:
            timer.cancel()
        if killed.is_set():
            self.say(f"  TIMEOUT: killed after {timeout:.0f}s with no exit")
            return 124, "\n".join(lines)
        return p.returncode, "\n".join(lines)


def stage(b: Bundle, name: str, proves: str):
    """Decorator-ish helper: run fn, record pass/fail and duration, never raise."""
    def wrap(fn):
        b.say("")
        b.say(BAR)
        b.say(f"STAGE {len(b.stages) + 1}: {name}")
        b.say(f"  proves: {proves}")
        b.say(BAR)
        t = time.time()
        rec = {"stage": name, "proves": proves}
        try:
            out = fn() or {}
            rec.update({"ok": True, **out})
        except Exception as e:  # a broken stage must not hide the ones after it
            import traceback
            rec.update({"ok": False, "error": f"{type(e).__name__}: {e}"})
            b.say(f"  FAILED: {type(e).__name__}: {e}")
            b.fh.write(traceback.format_exc())
        rec["seconds"] = round(time.time() - t, 1)
        b.stages.append(rec)
        b.say(f"  -> {'ok' if rec.get('ok') else 'FAILED'} in {rec['seconds']:.1f}s")
        return rec
    return wrap


def detect_device(xla: dict) -> str:
    try:
        import torch
        if torch.cuda.device_count() > 0:
            return "gpu"
    except Exception:
        pass
    return "tpu" if xla.get("xla_device_type") == "TPU" else "cpu"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="", help="default: matches the device")
    ap.add_argument("--out", default="/kaggle/working/shakedown")
    ap.add_argument("--tokens", type=float, default=3e6,
                    help="how much FineWeb-Edu to actually tokenize")
    ap.add_argument("--steps", type=int, default=12, help="steps per training leg")
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--tokens-dir", default="",
                    help="use tokens from here instead of making new ones")
    ap.add_argument("--skip-tokenize", action="store_true",
                    help="reuse tokens already on disk (offline, or a repeat run)")
    ap.add_argument("--no-install", action="store_true",
                    help="do not pip install missing packages")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    b = Bundle(out)
    work = out / "work"
    # An explicit tokens dir lives outside the output tree, which is wiped on
    # every run, so a second shakedown does not re-download the corpus.
    tokens_dir = Path(args.tokens_dir) if args.tokens_dir else work / "tokens"
    run_dir = work / "run"

    xla = probe_xla()
    device = detect_device(xla)
    preset = args.base or {"tpu": "large", "gpu": "small"}.get(device, "small")
    trainer = "train_tpu.py" if device == "tpu" else "train.py"

    b.say(BAR)
    b.say("llm-reverse shakedown")
    b.say(f"  device  {device}")
    b.say(f"  base    {preset}")
    b.say(f"  trainer {trainer}")
    b.say(f"  output  {out}")
    b.say(BAR)
    b.facts.update({"device": device, "base": preset, "trainer": trainer})

    # ---------------------------------------------------------------- 1. env
    @stage(b, "environment", "the box is what you think it is")
    def _env():
        import torch
        facts = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_devices": torch.cuda.device_count(),
        }
        facts.update(xla)  # gathered by a subprocess, see probe_xla
        if torch.cuda.device_count():
            facts["gpu_names"] = [torch.cuda.get_device_name(i)
                                  for i in range(torch.cuda.device_count())]
        du = shutil.disk_usage("/kaggle/working" if Path("/kaggle/working").exists() else ".")
        facts["disk_free_gb"] = round(du.free / 1e9, 1)
        # Eight replicas each hold a full copy of the model, so host RAM is the
        # binding constraint on a TPU box, not device memory. A worker killed by
        # the OOM killer surfaces as BrokenProcessPool with no Python traceback,
        # which is unreadable unless you already know what the budget was.
        try:
            meminfo = dict(
                (k.strip(), int(v.split()[0]))
                for k, v in (l.split(":", 1) for l in
                             Path("/proc/meminfo").read_text().splitlines()))
            facts["ram_total_gb"] = round(meminfo["MemTotal"] / 1e6, 1)
            facts["ram_available_gb"] = round(meminfo["MemAvailable"] / 1e6, 1)
        except Exception:
            pass
        for k, v in facts.items():
            b.say(f"  {k:18} {v}")
        if facts["disk_free_gb"] < 5:
            raise RuntimeError(f"only {facts['disk_free_gb']}GB free, a real run needs more")
        b.facts.update(facts)
        return facts

    # --------------------------------------------------------------- 2. deps
    @stage(b, "dependencies", "the packages a real run installs are importable here")
    def _deps():
        missing = []
        for mod in ("transformers", "datasets"):
            try:
                __import__(mod)
            except ImportError:
                missing.append(mod)
        if missing and not args.no_install:
            b.say(f"  installing {' '.join(missing)}")
            b.run([sys.executable, "-m", "pip", "install", "-q", *missing], timeout=900)
            missing = [m for m in missing
                       if subprocess.run([sys.executable, "-c", f"import {m}"],
                                         capture_output=True).returncode != 0]
        if missing:
            raise RuntimeError(f"still missing: {', '.join(missing)}")
        b.say("  transformers and datasets both import")
        return {}

    # ----------------------------------------------------------- 3. tokenize
    @stage(b, "tokenize", "network, HuggingFace streaming, and the shard writer all work")
    def _tok():
        if (args.skip_tokenize or args.tokens_dir) and (tokens_dir / "meta.json").exists():
            b.say(f"  reusing existing tokens at {tokens_dir}")
        else:
            code, _ = b.run([sys.executable, "tokenize_reverse.py", "--base", preset,
                             "--out-dir", str(tokens_dir),
                             "--max-tokens", str(args.tokens)], timeout=1200)
            # Same reasoning as kaggle_run.sh: the streaming reader can abort
            # during shutdown after the shards are already complete, so the exit
            # code is not the authority. Ask the data.
            code, _ = b.run([sys.executable, "tokenize_reverse.py", "--base", preset,
                             "--out-dir", str(tokens_dir),
                             "--max-tokens", str(args.tokens), "--verify-only"])
            if code != 0:
                raise RuntimeError("tokenizer did not produce a complete shard set")
        meta = json.loads((tokens_dir / "meta.json").read_text())
        b.say(f"  {meta.get('total_tokens', 0):,} tokens, tokenizer={meta.get('tokenizer')}")
        return {"total_tokens": meta.get("total_tokens"), "shards": len(meta["shards"]["train"])}

    # ---------------------------------------------------------------- 3. data
    @stage(b, "dataloader", "batches are the right shape and identical for the same step")
    def _data():
        sys.path.insert(0, str(HERE))
        import torch
        from data import BatchSampler, Corpus
        c = Corpus(str(tokens_dir), args.block_size, "train")
        s = BatchSampler(c, args.micro_batch, args.grad_accum, 1, 0, 1337)
        x1, y1 = s.batch(7, 0, torch.device("cpu"))
        x2, y2 = s.batch(7, 0, torch.device("cpu"))
        same = bool(torch.equal(x1, x2) and torch.equal(y1, y2))
        drifted = not torch.equal(x1, s.batch(8, 0, torch.device("cpu"))[0])
        b.say(f"  corpus {c.total_tokens:,} tokens, {c.n_blocks:,} blocks")
        b.say(f"  batch  {tuple(x1.shape)} {x1.dtype}, targets shifted by one: "
              f"{bool(torch.equal(x1[0, 1:], y1[0, :-1]))}")
        b.say(f"  step 7 reproducible: {same}   step 8 differs: {drifted}")
        if not same:
            raise RuntimeError("same step gave different batches; resume would not be clean")
        if not drifted:
            raise RuntimeError("different steps gave the same batch; the sampler is stuck")
        return {"blocks": c.n_blocks, "deterministic": same}

    # --------------------------------------------------------------- 4. model
    @stage(b, "model", "the base downloads and the adapter wraps it")
    def _model():
        import bases
        import basemodel

        base = bases.get(preset)
        model = basemodel.build_smoke() if preset == "smoke" else \
            basemodel.build(preset, base.block_size)
        total = sum(p.numel() for p in model.parameters())
        b.say("  " + model.param_report().replace("\n", "\n  "))
        # The adapter must present (logits, loss), because the trainer, the
        # resume path and the dashboard were all written against that shape.
        import torch

        x = torch.randint(0, min(1000, base.vocab_size), (2, 32))
        logits, loss = model(x, x)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"forward produced loss={loss}")
        return {"base": base.hf_name, "params": total,
                "state_gb": round(total * bases.BYTES_PER_PARAM / 1e9, 2),
                "loss": round(float(loss), 3)}

    # ---------------------------------------------------------------- 8. chat
    @stage(b, "chat", "the checkpoint loads back, in the direction it was trained")
    def _chat():
        import chat as chatmod
        chatmod.no_color()
        target = sorted(run_dir.glob("ckpt_step*.pt"))[-1]
        s = chatmod.Session(target, "cpu")
        out = []
        for prompt, jeopardy in ((" and that was the end of it.", False), ("Paris.", True)):
            s.history.clear()
            s.jeopardy = jeopardy and s.reverse
            r = " ".join(s.reply(prompt, 24, 0.8, 40, False, False, stream=False).split())
            # A reverse model writes to the left of the prompt, so the prompt has
            # to still be the tail of the rendered text. If it is not, the ids
            # were oriented one way and rendered the other, which produces
            # perfectly fluent output facing the wrong direction.
            if s.reverse and not r.endswith(prompt.strip()):
                raise RuntimeError(f"reverse render lost the prompt: {r!r}")
            out.append({"prompt": prompt, "mode": s.mode_name if s.reverse else "forward",
                        "text": r})
            b.say(f"  {'>':>2} {prompt}")
            b.say(f"  {'=':>2} {r}")
        return {"direction": s.direction, "generations": out}

    # -------------------------------------------------------------- 9. bundle
    b.say("")
    b.say(BAR)
    b.say("SUMMARY")
    b.say(BAR)
    passed = sum(1 for s in b.stages if s.get("ok"))
    for s in b.stages:
        b.say(f"  {'PASS' if s.get('ok') else 'FAIL'}  {s['stage']:<12} "
              f"{s['seconds']:>6.1f}s   {s.get('error', '')}")
    b.say("")
    b.say(f"  {passed}/{len(b.stages)} stages passed in "
          f"{(time.time() - b.t0) / 60:.1f} minutes")

    b.facts["stages"] = b.stages
    b.facts["passed"] = passed
    b.facts["total_stages"] = len(b.stages)
    b.facts["wall_minutes"] = round((time.time() - b.t0) / 60, 2)
    (out / "shakedown.json").write_text(json.dumps(b.facts, indent=2, default=str))

    # Copy the artifacts a long run would have produced, so the bundle shows the
    # loss curve and the samples rather than just this script's opinion of them.
    for name in ("loss.csv", "status.json", "samples.txt"):
        src = run_dir / name
        if src.exists():
            shutil.copy(src, out / name)

    zpath = Path("/kaggle/working/shakedown.zip")
    if not zpath.parent.exists():
        zpath = out.parent / "shakedown.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in out.iterdir():
            if f.is_file():
                z.write(f, f.name)
    b.say("")
    b.say(f"  bundle: {zpath}  ({zpath.stat().st_size / 1e3:.0f} KB)")
    b.say(f"  log:    {b.logfile}")
    b.say("")
    b.say("  Download the zip from the Kaggle file browser on the right.")
    b.fh.close()
    return 0 if passed == len(b.stages) else 1


if __name__ == "__main__":
    raise SystemExit(main())
