"""Color parsing, palettes and ANSI escape generation."""

from __future__ import annotations

import os
import sys
from typing import Iterable, List, Optional, Sequence, Tuple, Union

RGB = Tuple[int, int, int]
ColorLike = Union[str, RGB, Sequence[int], int, None]

RESET = "\x1b[0m"

NAMED: dict = {
    "black": (0, 0, 0),
    "white": (245, 245, 245),
    "gray": (140, 140, 148),
    "grey": (140, 140, 148),
    "darkgray": (80, 80, 88),
    "darkgrey": (80, 80, 88),
    "red": (228, 87, 76),
    "green": (76, 185, 68),
    "blue": (76, 155, 232),
    "cyan": (63, 193, 201),
    "magenta": (214, 93, 177),
    "yellow": (239, 201, 76),
    "orange": (240, 128, 60),
    "purple": (169, 123, 232),
    "violet": (169, 123, 232),
    "pink": (232, 106, 168),
    "teal": (45, 178, 160),
    "lime": (158, 216, 72),
    "gold": (222, 170, 60),
    "navy": (52, 88, 160),
    "brown": (150, 105, 72),
    "salmon": (238, 138, 116),
    "mint": (120, 220, 170),
    "sky": (110, 190, 240),
    "indigo": (99, 102, 220),
    "crimson": (208, 60, 90),
}

#: Qualitative palettes cycle; gradient palettes are sampled across their anchors.
PALETTES: dict = {
    "default": ["#4C9BE8", "#F0803C", "#4CB944", "#E4574C",
                "#A97BE8", "#EFC94C", "#3FC1C9", "#E86AA8"],
    "vivid": ["#FF3B6B", "#00D2FF", "#FFD400", "#7CFF4F",
              "#B15BFF", "#FF8A00", "#00FFB2", "#FF5AF7"],
    "pastel": ["#A8D8EA", "#FFB7B2", "#B5EAD7", "#FFDAC1",
               "#C7CEEA", "#E2F0CB", "#F6C8E0", "#D6C6F0"],
    "earth": ["#8C6A4E", "#B9975B", "#6E8B3D", "#4F7A6B",
              "#A6634C", "#7A6A53", "#C2A878", "#54604F"],
    "mono": ["#F2F2F2", "#C9C9C9", "#A0A0A0", "#787878",
             "#565656", "#3C3C3C"],
    "viridis": ["#440154", "#414487", "#2A788E", "#22A884", "#7AD151", "#FDE725"],
    "magma": ["#000004", "#3B0F70", "#8C2981", "#DE4968", "#FE9F6D", "#FCFDBF"],
    "ocean": ["#012A4A", "#01497C", "#2A6F97", "#468FAF", "#89C2D9", "#CAF0F8"],
    "sunset": ["#3D2C8D", "#916BBF", "#E5717A", "#F79D65", "#F7B267", "#FFE29A"],
    "fire": ["#0B0B0B", "#5F0F40", "#9A031E", "#CB4721", "#FB8B24", "#FFD26F"],
    "ice": ["#03045E", "#0077B6", "#00B4D8", "#48CAE4", "#90E0EF", "#CAF0F8"],
}

GRADIENTS = {"viridis", "magma", "ocean", "sunset", "fire", "ice", "mono"}


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def parse_color(c: ColorLike) -> Optional[RGB]:
    """Accept ``"#ff0088"``, ``"red"``, ``(r, g, b)`` or an xterm-256 index."""
    if c is None:
        return None
    if isinstance(c, str):
        s = c.strip().lower()
        if s in NAMED:
            return NAMED[s]
        s = s.lstrip("#")
        if len(s) == 3 and all(ch in "0123456789abcdef" for ch in s):
            return tuple(int(ch * 2, 16) for ch in s)  # type: ignore[return-value]
        if len(s) == 6 and all(ch in "0123456789abcdef" for ch in s):
            return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
        raise ValueError("unknown color: %r" % (c,))
    if isinstance(c, int):
        return _xterm256_to_rgb(c)
    if isinstance(c, (tuple, list)) and len(c) == 3:
        return tuple(max(0, min(255, int(v))) for v in c)  # type: ignore[return-value]
    raise ValueError("unknown color: %r" % (c,))


def blend(a: RGB, b: RGB, t: float) -> RGB:
    t = max(0.0, min(1.0, t))
    return (
        int(round(a[0] + (b[0] - a[0]) * t)),
        int(round(a[1] + (b[1] - a[1]) * t)),
        int(round(a[2] + (b[2] - a[2]) * t)),
    )


def dim(c: RGB, factor: float = 0.45) -> RGB:
    """Darken toward black; used for fills, grids and other secondary ink."""
    return blend((0, 0, 0), c, factor)


def gradient(anchors: Sequence[RGB], n: int) -> List[RGB]:
    """Sample ``n`` evenly spaced colors along a piecewise-linear ramp."""
    if n <= 0:
        return []
    if n == 1 or len(anchors) == 1:
        return [tuple(anchors[len(anchors) // 2])] * n  # type: ignore[list-item]
    out = []
    span = len(anchors) - 1
    for i in range(n):
        pos = i / (n - 1) * span
        lo = min(int(pos), span - 1)
        out.append(blend(anchors[lo], anchors[lo + 1], pos - lo))
    return out


def palette(name_or_colors, n: Optional[int] = None) -> List[RGB]:
    """Resolve a palette name (or explicit color list) into ``n`` RGB colors."""
    if isinstance(name_or_colors, str):
        key = name_or_colors.lower()
        if key not in PALETTES:
            raise ValueError(
                "unknown palette %r (have: %s)" % (name_or_colors, ", ".join(sorted(PALETTES)))
            )
        colors = [parse_color(c) for c in PALETTES[key]]
        is_gradient = key in GRADIENTS
    else:
        colors = [parse_color(c) for c in name_or_colors]
        is_gradient = False
    if n is None:
        return colors  # type: ignore[return-value]
    if is_gradient:
        return gradient(colors, n)  # type: ignore[arg-type]
    return [colors[i % len(colors)] for i in range(n)]  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# terminal capability detection
# --------------------------------------------------------------------------- #
def _enable_windows_ansi() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        for handle_id in (-11, -12):
            handle = kernel32.GetStdHandle(handle_id)
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


_enable_windows_ansi()


def detect_mode(stream=None) -> str:
    """Return ``"truecolor"``, ``"256"`` or ``"none"`` for the given stream."""
    stream = stream or sys.stdout
    if os.environ.get("NO_COLOR") is not None:
        return "none"
    force = os.environ.get("FORCE_COLOR")
    if force not in (None, "", "0"):
        return "truecolor"
    try:
        if not stream.isatty():
            return "none"
    except Exception:
        return "none"
    if os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit"):
        return "truecolor"
    if os.environ.get("WT_SESSION") or os.environ.get("TERM_PROGRAM"):
        return "truecolor"
    term = os.environ.get("TERM", "")
    if "256" in term or "truecolor" in term:
        return "256"
    if os.name == "nt":
        return "truecolor"
    if term in ("", "dumb"):
        return "none"
    return "256"


def supports_unicode(stream=None) -> bool:
    stream = stream or sys.stdout
    enc = (getattr(stream, "encoding", None) or "").lower()
    return "utf" in enc


# --------------------------------------------------------------------------- #
# escape sequences
# --------------------------------------------------------------------------- #
_CUBE = (0, 95, 135, 175, 215, 255)


def rgb_to_256(c: RGB) -> int:
    r, g, b = c
    if abs(r - g) < 10 and abs(g - b) < 10:
        level = round((r + g + b) / 3)
        if level < 8:
            return 16
        if level > 248:
            return 231
        return 232 + int((level - 8) / 247 * 23)

    def q(v: int) -> int:
        best, bi = 1e9, 0
        for i, cv in enumerate(_CUBE):
            d = abs(cv - v)
            if d < best:
                best, bi = d, i
        return bi

    return 16 + 36 * q(r) + 6 * q(g) + q(b)


def _xterm256_to_rgb(idx: int) -> RGB:
    idx = max(0, min(255, int(idx)))
    if idx < 16:
        base = [
            (0, 0, 0), (170, 0, 0), (0, 170, 0), (170, 85, 0),
            (0, 0, 170), (170, 0, 170), (0, 170, 170), (170, 170, 170),
            (85, 85, 85), (255, 85, 85), (85, 255, 85), (255, 255, 85),
            (85, 85, 255), (255, 85, 255), (85, 255, 255), (255, 255, 255),
        ]
        return base[idx]
    if idx < 232:
        i = idx - 16
        return (_CUBE[i // 36], _CUBE[(i // 6) % 6], _CUBE[i % 6])
    level = 8 + (idx - 232) * 10
    return (level, level, level)


def fg_seq(c: Optional[RGB], mode: str) -> str:
    if mode == "none" or c is None:
        return ""
    if mode == "truecolor":
        return "\x1b[38;2;%d;%d;%dm" % c
    return "\x1b[38;5;%dm" % rgb_to_256(c)


def colorize(text: str, c: ColorLike, mode: str = "truecolor") -> str:
    rgb = parse_color(c)
    if mode == "none" or rgb is None:
        return text
    return fg_seq(rgb, mode) + text + RESET


def palette_names() -> List[str]:
    return sorted(PALETTES)


def color_names() -> List[str]:
    return sorted(set(NAMED))


def _iter_rgb(colors: Iterable[ColorLike]) -> List[RGB]:
    return [parse_color(c) for c in colors]  # type: ignore[misc]
