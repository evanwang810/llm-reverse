"""CLI: ``termplot demo`` and ``termplot plot data.csv``."""

from __future__ import annotations

import argparse
import math
import sys
from typing import List

from . import Figure, __version__, bar, hbar, hist, line, palette_names, sparkline


def _read_columns(path: str) -> List[List[float]]:
    text = sys.stdin.read() if path in ("-", "") else open(path, "r", encoding="utf-8").read()
    columns: List[List[float]] = []
    for raw in text.splitlines():
        row = raw.replace(",", " ").split()
        if not row:
            continue
        values = []
        ok = True
        for cell in row:
            try:
                values.append(float(cell))
            except ValueError:
                ok = False
                break
        if not ok:
            continue
        while len(columns) < len(values):
            columns.append([])
        for i, v in enumerate(values):
            columns[i].append(v)
    if not columns:
        raise SystemExit("no numeric data found")
    return columns


def demo() -> None:
    xs = [i / 6.0 for i in range(120)]
    fig = Figure(title="waves", height=18, grid=True, xlabel="t")
    fig.line(xs, [math.sin(x) for x in xs], label="sin")
    fig.line(xs, [math.cos(x * 0.7) * 0.6 for x in xs], label="cos", color="orange")
    fig.show()
    print()

    line(xs, [math.exp(-x / 6) * math.sin(x * 2) for x in xs],
         title="damped oscillation", fill=True, color="cyan", height=14)
    print()

    bar([42, 67, 31, 88, 54, 73],
        labels=["mon", "tue", "wed", "thu", "fri", "sat"],
        title="commits per day", show_values=True, height=16, palette="vivid")
    print()

    hbar([8.2, 6.4, 5.9, 3.1, 1.7],
         labels=["python", "rust", "go", "c", "lua"],
         title="lines of code (M)", palette="ocean")
    print()

    seed = 12345
    samples = []
    for _ in range(4000):
        total = 0.0
        for _ in range(6):
            seed = (1103515245 * seed + 12345) % (1 << 31)
            total += seed / (1 << 31)
        samples.append((total - 3) * 1.8)
    hist(samples, bins=34, title="normal-ish samples", color="#7AD151", height=16)
    print()

    pts_x, pts_y = [], []
    for i in range(220):
        seed = (1103515245 * seed + 12345) % (1 << 31)
        a = seed / (1 << 31)
        seed = (1103515245 * seed + 12345) % (1 << 31)
        b = seed / (1 << 31)
        pts_x.append(a * 10)
        pts_y.append(a * 3 + b * 4 - 2)
    fig = Figure(title="scatter", height=16, xlabel="x", ylabel="y")
    fig.scatter(pts_x, pts_y, color="magenta")
    fig.show()
    print()

    series = [math.sin(i / 4) * 10 + i / 3 for i in range(60)]
    print("sparkline  " + sparkline(series, palette="sunset"))
    print("palettes   " + ", ".join(palette_names()))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="termplot", description=__doc__)
    parser.add_argument("--version", action="version", version="termplot " + __version__)
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("demo", help="render a gallery of every chart type")

    p = sub.add_parser("plot", help="plot numeric columns from a file or stdin")
    p.add_argument("file", nargs="?", default="-")
    p.add_argument("--kind", default="line", choices=["line", "scatter", "bar", "hbar", "hist"])
    p.add_argument("--title")
    p.add_argument("--xlabel")
    p.add_argument("--ylabel")
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int, default=20)
    p.add_argument("--palette", default="default")
    p.add_argument("--grid", action="store_true")
    p.add_argument("--xy", action="store_true",
                   help="treat the first column as x and the rest as y series")

    args = parser.parse_args(argv)
    if args.cmd in (None, "demo"):
        demo()
        return 0

    columns = _read_columns(args.file)
    fig_kw = dict(title=args.title, xlabel=args.xlabel, ylabel=args.ylabel,
                  width=args.width, height=args.height, palette=args.palette, grid=args.grid)
    if args.kind == "hist":
        f = Figure(**fig_kw)
        f.hist(columns[0])
        f.show()
        return 0

    xs = columns[0] if args.xy and len(columns) > 1 else None
    ys = columns[1:] if xs is not None else columns
    f = Figure(**fig_kw)
    for i, col in enumerate(ys):
        label = "col%d" % (i + (1 if xs is not None else 0))
        if args.kind == "scatter":
            f.scatter(xs, col, label=label) if xs else f.scatter(col, label=label)
        elif args.kind == "bar":
            f.bar(col, label=label)
        elif args.kind == "hbar":
            f.hbar(col, label=label)
        else:
            f.line(xs, col, label=label) if xs else f.line(col, label=label)
    f.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
