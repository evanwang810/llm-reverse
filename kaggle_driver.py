#!/usr/bin/env python
"""Optional Kaggle API helper. The only part genuinely worth automating is
bumping the checkpoint dataset version between sessions.

    pip install kaggle
    # put kaggle.json in %USERPROFILE%\\.kaggle\\kaggle.json

    python kaggle_driver.py setup --user YOURNAME
    python kaggle_driver.py bump --dir output --message "step 61000"
    python kaggle_driver.py push
    python kaggle_driver.py watch
    python kaggle_driver.py pull
    python kaggle_driver.py loop

Honest warning about `push`: the kernel metadata API has no field for the
"T4 x2" accelerator. A kernel pushed this way gets a single GPU, so DDP will
not engage. For the dual-T4 run, use the website editor and pick GPU T4 x2 in
the sidebar. Use this script for dataset versioning and for pulling output.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

KERNEL_META = "kernel-metadata.json"
DATASET_META = "dataset-metadata.json"


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip(), file=sys.stderr)
    if check and proc.returncode != 0:
        raise SystemExit(f"command failed with code {proc.returncode}")
    return proc


def cmd_setup(args: argparse.Namespace) -> None:
    Path(KERNEL_META).write_text(json.dumps({
        "id": f"{args.user}/{args.kernel}",
        "title": args.kernel,
        "code_file": "kaggle_entry.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [f"{args.user}/{args.tokens_dataset}",
                            f"{args.user}/{args.ckpt_dataset}"],
        "competition_sources": [],
        "kernel_sources": [],
    }, indent=1))
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / DATASET_META).write_text(json.dumps({
        "title": args.ckpt_dataset,
        "id": f"{args.user}/{args.ckpt_dataset}",
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=1))
    print(f"\nwrote {KERNEL_META} and {out / DATASET_META}")
    print(f"create the checkpoint dataset once with:\n"
          f"  kaggle datasets create -p {out} --dir-mode zip")


def cmd_bump(args: argparse.Namespace) -> None:
    d = Path(args.dir)
    if not (d / DATASET_META).exists():
        raise SystemExit(f"missing {d / DATASET_META}. Run `setup` first.")
    files = sorted(d.glob("*.pt"))
    if not files:
        print(f"warning: no .pt files in {d}")
    run(["kaggle", "datasets", "version", "-p", str(d), "-m", args.message, "--dir-mode", "zip"])
    print("new version queued. Kaggle takes a few minutes to finish processing it.")


def cmd_push(args: argparse.Namespace) -> None:
    run(["kaggle", "kernels", "push", "-p", "."])


def _status(slug: str) -> str:
    proc = run(["kaggle", "kernels", "status", slug], check=False)
    return (proc.stdout or "") + (proc.stderr or "")


def cmd_watch(args: argparse.Namespace) -> None:
    slug = args.slug or json.loads(Path(KERNEL_META).read_text())["id"]
    while True:
        text = _status(slug)
        low = text.lower()
        if "complete" in low or "error" in low or "cancel" in low:
            print(f"terminal state reached: {text.strip()}")
            return
        time.sleep(args.interval)


def cmd_pull(args: argparse.Namespace) -> None:
    slug = args.slug or json.loads(Path(KERNEL_META).read_text())["id"]
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "kernels", "output", slug, "-p", str(out)])
    pts = sorted(out.rglob("*.pt"))
    print(f"pulled {len(pts)} checkpoint files into {out}")
    for p in pts:
        print(f"  {p.name}  {p.stat().st_size / 1e6:.0f} MB")


def cmd_loop(args: argparse.Namespace) -> None:
    for i in range(args.rounds):
        print(f"\n===== round {i + 1} / {args.rounds} =====")
        cmd_push(args)
        time.sleep(30)
        cmd_watch(args)
        cmd_pull(args)
        cmd_bump(argparse.Namespace(dir=args.dir, message=f"auto round {i + 1}"))
        print("waiting for the dataset version to finish processing before the next push")
        time.sleep(args.settle)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--user", default="evanwang810")
    p.add_argument("--kernel", default="llmrev-train")
    p.add_argument("--tokens-dataset", default="fineweb-edu-tokens")
    p.add_argument("--ckpt-dataset", default="llmrev-ckpt")
    p.add_argument("--dir", default="output")
    p.add_argument("--slug", default="")
    p.add_argument("--message", default="new checkpoint")
    p.add_argument("--interval", type=int, default=120)
    p.add_argument("--settle", type=int, default=300)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("command", choices=["setup", "push", "watch", "pull", "bump", "loop"])
    args = p.parse_args()
    {"setup": cmd_setup, "push": cmd_push, "watch": cmd_watch,
     "pull": cmd_pull, "bump": cmd_bump, "loop": cmd_loop}[args.command](args)


if __name__ == "__main__":
    main()
