"""A tiny colored character grid plus the box/block glyph sets."""

from __future__ import annotations

from typing import List, Optional

from .color import RESET, RGB, fg_seq


class Charset:
    """Glyphs used by the renderers, with an ASCII fallback for dumb terminals."""

    def __init__(self, unicode: bool = True):
        self.unicode = unicode
        if unicode:
            self.vaxis = "│"      # │
            self.haxis = "─"      # ─
            self.corner = "└"     # └
            self.ytick = "┤"      # ┤
            self.xtick = "┬"      # ┬
            self.cross = "┼"      # ┼
            self.vblocks = " ▁▂▃▄▅▆▇█"
            self.hblocks = " ▏▎▍▌▋▊▉█"
            self.legend = "━"     # ━
            self.marker = "●"     # ●
            self.grid_h = "─"
            self.grid_v = "│"
        else:
            self.vaxis = "|"
            self.haxis = "-"
            self.corner = "+"
            self.ytick = "+"
            self.xtick = "+"
            self.cross = "+"
            self.vblocks = " ....####"
            self.hblocks = " ....####"
            self.legend = "-"
            self.marker = "o"
            self.grid_h = "-"
            self.grid_v = "|"

    #: density ramp used when braille is unavailable
    ASCII_RAMP = " .:*#"


class Screen:
    """Fixed-size grid of (char, foreground color) cells."""

    def __init__(self, width: int, height: int, mode: str = "truecolor"):
        self.w = max(1, int(width))
        self.h = max(1, int(height))
        self.mode = mode
        self.ch: List[List[str]] = [[" "] * self.w for _ in range(self.h)]
        self.fg: List[List[Optional[RGB]]] = [[None] * self.w for _ in range(self.h)]

    def put(self, x: int, y: int, char: str, color: Optional[RGB] = None) -> None:
        if 0 <= x < self.w and 0 <= y < self.h:
            self.ch[y][x] = char
            self.fg[y][x] = color

    def get(self, x: int, y: int) -> str:
        if 0 <= x < self.w and 0 <= y < self.h:
            return self.ch[y][x]
        return " "

    def text(self, x: int, y: int, s: str, color: Optional[RGB] = None) -> None:
        for i, char in enumerate(s):
            self.put(x + i, y, char, color)

    def text_right(self, x_end: int, y: int, s: str, color: Optional[RGB] = None) -> None:
        self.text(x_end - len(s) + 1, y, s, color)

    def hline(self, x0: int, x1: int, y: int, char: str, color: Optional[RGB] = None) -> None:
        for x in range(min(x0, x1), max(x0, x1) + 1):
            self.put(x, y, char, color)

    def vline(self, x: int, y0: int, y1: int, char: str, color: Optional[RGB] = None) -> None:
        for y in range(min(y0, y1), max(y0, y1) + 1):
            self.put(x, y, char, color)

    def render(self) -> str:
        lines = []
        for y in range(self.h):
            row = self.ch[y]
            last = -1
            for x in range(self.w - 1, -1, -1):
                if row[x] != " ":
                    last = x
                    break
            if last < 0:
                lines.append("")
                continue
            if self.mode == "none":
                lines.append("".join(row[: last + 1]))
                continue
            out: List[str] = []
            current: Optional[RGB] = None
            for x in range(last + 1):
                c = self.fg[y][x]
                if c != current:
                    out.append(RESET if c is None else fg_seq(c, self.mode))
                    current = c
                out.append(row[x])
            if current is not None and self.mode != "none":
                out.append(RESET)
            lines.append("".join(out))
        return "\n".join(lines)
