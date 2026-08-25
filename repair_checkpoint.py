"""Rebuild a .pt that got unzipped into a directory.

A torch checkpoint is a zip archive, so Windows Explorer, Chrome's download
handler and most "extract here" tools will happily unpack one in place and leave
you with a folder named model.pt containing data.pkl, data/0, data/1 and so on.
Nothing is lost, the archive just needs putting back together.

    python repair_checkpoint.py path/to/model.pt
    python repair_checkpoint.py path/to/model.pt --out fixed.pt

Verifies the result loads before it replaces anything.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

# torch.save writes every record under one top level directory. The name is not
# meaningful, the reader takes whatever prefix the first record has, but keeping
# the stem matches what torch itself would have written.
MEMBERS = ("data.pkl", "version", "byteorder", ".format_version",
           ".storage_alignment", ".data", "data")


def looks_unzipped(path: Path) -> bool:
    return path.is_dir() and (path / "data.pkl").is_file()


def rebuild(src: Path, dest: Path, archive: str) -> None:
    """Zip src's contents back under a single top level directory.

    Stored, not deflated: torch writes uncompressed records and the reader is
    happier with them. Ordering follows torch's own, data.pkl first, so a reader
    that streams rather than seeks still works.
    """
    files: list[Path] = []
    for name in MEMBERS:
        p = src / name
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(q for q in p.rglob("*") if q.is_file()))

    known = {f.resolve() for f in files}
    for q in sorted(src.rglob("*")):
        if q.is_file() and q.resolve() not in known:
            files.append(q)

    with zipfile.ZipFile(dest, "w", zipfile.ZIP_STORED) as z:
        for f in files:
            z.write(f, f"{archive}/{f.relative_to(src).as_posix()}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="the directory left behind by the unzip")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write (default: replace the directory in place)")
    ap.add_argument("--keep", action="store_true",
                    help="do not delete the directory after a successful repair")
    args = ap.parse_args()

    src = args.path
    if src.is_file():
        print(f"{src} is already a file, nothing to repair")
        return 0
    if not looks_unzipped(src):
        print(f"{src} is not an unzipped checkpoint (no data.pkl inside)", file=sys.stderr)
        return 1

    # Build beside the target first. Replacing the directory before the result is
    # known good would turn a recoverable mess into a lost checkpoint.
    staged = src.with_name(src.name + ".rebuilt")
    rebuild(src, staged, archive=src.stem)

    try:
        import torch
    except ImportError:
        print(f"wrote {staged} but torch is not installed here, so it is unverified")
        return 0

    try:
        ckpt = torch.load(staged, map_location="cpu", weights_only=False)
    except Exception as e:
        staged.unlink(missing_ok=True)
        print(f"rebuild did not load: {e}", file=sys.stderr)
        return 1

    keys = ", ".join(sorted(k for k in ckpt if k != "model")) if isinstance(ckpt, dict) else type(ckpt).__name__
    n = sum(v.numel() for v in ckpt.get("model", {}).values()) if isinstance(ckpt, dict) else 0
    print(f"loads clean: {n / 1e6:.2f}M params, keys: {keys}")

    dest = args.out
    if dest is None:
        if not args.keep:
            shutil.rmtree(src)
        dest = src if not args.keep else src.with_name(src.stem + ".fixed.pt")
    staged.replace(dest)
    print(f"wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
