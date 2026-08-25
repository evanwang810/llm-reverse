"""Shared run directory: CSV log, status file, flag files, checkpoint discovery.

The trainer writes, the dashboard reads. Communication is plain files, which
means the dashboard can live in the same notebook, a different notebook, or on
your laptop, with no sockets or shared memory involved.
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from pathlib import Path

CSV_FIELDS = [
    "step",
    "wall_s",
    "tokens",
    "lr",
    "loss",
    "loss_ema",
    "val_loss",
    "grad_norm",
    "scale",
    "tok_per_s",
    "phase",
]

FLAGS = ("SAVE_NOW", "DECAY_NOW", "STOP_NOW")


def atomic_write(path: Path, data: str | bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    mode = "wb" if isinstance(data, bytes) else "w"
    with open(tmp, mode) as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class RunDir:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.path / "loss.csv"
        self.status_path = self.path / "status.json"

    # ---------- writer side ----------

    def init_csv(self) -> None:
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                csv.DictWriter(f, CSV_FIELDS).writeheader()

    def append_csv(self, row: dict) -> None:
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, CSV_FIELDS, extrasaction="ignore")
            w.writerow({k: row.get(k, "") for k in CSV_FIELDS})
            f.flush()

    def write_status(self, status: dict) -> None:
        status = dict(status)
        status["heartbeat"] = time.time()
        atomic_write(self.status_path, json.dumps(status, indent=1))

    def poll_flags(self) -> set[str]:
        """Return the flags that are set and clear them."""
        hit = set()
        for name in FLAGS:
            p = self.path / name
            if p.exists():
                hit.add(name)
                try:
                    p.unlink()
                except OSError:
                    pass
        return hit

    # ---------- reader / dashboard side ----------

    def request(self, flag: str) -> str:
        if flag not in FLAGS:
            raise ValueError(flag)
        (self.path / flag).write_text(str(time.time()))
        return f"requested {flag}"

    def read_status(self) -> dict | None:
        try:
            return json.loads(self.status_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def read_csv(self) -> list[dict]:
        try:
            with open(self.csv_path, newline="") as f:
                return list(csv.DictReader(f))
        except OSError:
            return []


_CKPT_RE = re.compile(r"(ckpt|weights|milestone|sft)_step(\d+)\.pt$")
_KINDS = {"ckpt": "full", "weights": "weights", "milestone": "milestone", "sft": "sft"}


def find_checkpoints(search_dirs) -> list[dict]:
    """All checkpoints under the given roots, newest step first.

    kind 'full' has optimizer state and can be resumed from. 'weights' is the
    small rolling fp16 copy, 'milestone' is the same thing but never pruned, so
    it is what you use to compare early against late on the same prompt. 'sft'
    is an instruction-tuned model, which the chat tab formats differently.
    """
    found: dict[Path, dict] = {}
    patterns = ["*.pt", "*/*.pt", "*/*/*.pt", "*/*/*/*.pt"]
    for root in search_dirs:
        root = Path(root)
        if not root.exists():
            continue
        for pat in patterns:
            for p in root.glob(pat):
                m = _CKPT_RE.search(p.name)
                if not m:
                    continue
                # A partial download can leave a *directory* named like a
                # checkpoint. Listing it would only produce a confusing
                # PermissionError later, when something tries to open it.
                if not p.is_file() or p.stat().st_size == 0:
                    continue
                found[p.resolve()] = {
                    "path": p,
                    "step": int(m.group(2)),
                    "kind": _KINDS[m.group(1)],
                    "size_mb": p.stat().st_size / 1e6,
                    "mtime": p.stat().st_mtime,
                }
    out = list(found.values())
    out.sort(key=lambda d: (d["step"], d["kind"] == "full"), reverse=True)
    return out


def default_search_dirs() -> list[Path]:
    dirs = [Path("/kaggle/working/run"), Path("/kaggle/working")]
    kin = Path("/kaggle/input")
    if kin.exists():
        dirs.extend(sorted(p for p in kin.iterdir() if p.is_dir()))
    dirs.append(Path("run"))
    return [d for d in dirs if d.exists()]
