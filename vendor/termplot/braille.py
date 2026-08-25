"""Sub-cell drawing surface: 2x4 braille dots per character cell."""

from __future__ import annotations

from typing import List, Optional

from .color import RGB
from .screen import Charset, Screen

_BASE = 0x2800
# _DOTS[row][col] -> bit within the braille pattern
_DOTS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))
_POPCOUNT = [bin(i).count("1") for i in range(256)]


class Braille:
    """Pixel canvas ``2 * cols`` wide and ``4 * rows`` tall."""

    def __init__(self, cols: int, rows: int):
        self.cols = max(1, int(cols))
        self.rows = max(1, int(rows))
        self.pw = self.cols * 2
        self.ph = self.rows * 4
        self.bits: List[List[int]] = [[0] * self.cols for _ in range(self.rows)]
        self.color: List[List[Optional[RGB]]] = [[None] * self.cols for _ in range(self.rows)]

    def set(self, px: int, py: int, color: Optional[RGB] = None) -> None:
        if not (0 <= px < self.pw and 0 <= py < self.ph):
            return
        cx, cy = px >> 1, py >> 2
        self.bits[cy][cx] |= _DOTS[py & 3][px & 1]
        if color is not None:
            self.color[cy][cx] = color

    def line(self, x0: int, y0: int, x1: int, y1: int, color: Optional[RGB] = None) -> None:
        """Bresenham, clamped so wildly off-canvas segments stay cheap."""
        lim = 4 * max(self.pw, self.ph)
        x0, y0 = _clamp(x0, lim), _clamp(y0, lim)
        x1, y1 = _clamp(x1, lim), _clamp(y1, lim)
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self.set(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def vline(self, px: int, py0: int, py1: int, color: Optional[RGB] = None) -> None:
        lo, hi = (py0, py1) if py0 <= py1 else (py1, py0)
        lo = max(0, min(self.ph - 1, lo))
        hi = max(0, min(self.ph - 1, hi))
        for py in range(lo, hi + 1):
            self.set(px, py, color)

    def blit(self, screen: Screen, ox: int, oy: int, charset: Optional[Charset] = None) -> None:
        ascii_mode = charset is not None and not charset.unicode
        ramp = Charset.ASCII_RAMP
        for y in range(self.rows):
            row = self.bits[y]
            for x in range(self.cols):
                b = row[x]
                if not b:
                    continue
                if ascii_mode:
                    idx = min(len(ramp) - 1, 1 + _POPCOUNT[b] * (len(ramp) - 2) // 8)
                    char = ramp[idx]
                else:
                    char = chr(_BASE + b)
                screen.put(ox + x, oy + y, char, self.color[y][x])


def _clamp(v: int, lim: int) -> int:
    return max(-lim, min(lim, int(v)))
