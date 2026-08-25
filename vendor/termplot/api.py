"""One-call helpers: ``termplot.line(...)`` and friends."""

from __future__ import annotations

import sys
from typing import Optional, Sequence

from . import color as _color
from .color import ColorLike, detect_mode, fg_seq, parse_color, supports_unicode
from .figure import Figure

_FIG_KEYS = {
    "width", "height", "title", "xlabel", "ylabel", "palette", "grid",
    "legend", "xlim", "ylim", "unicode", "stream",
}


def _split(kwargs):
    """Separate Figure options from series options.

    ``color=True/False`` toggles ANSI output; ``color="red"`` colors the series.
    """
    fig, series = {}, {}
    for key, value in kwargs.items():
        if key == "color":
            (fig if isinstance(value, bool) or value is None else series)[key] = value
        elif key in _FIG_KEYS:
            fig[key] = value
        else:
            series[key] = value
    return fig, series


def _run(method: str, args, kwargs) -> Figure:
    fig_kw, series_kw = _split(kwargs)
    show = series_kw.pop("show", True)
    fig = Figure(**fig_kw)
    getattr(fig, method)(*args, **series_kw)
    if show:
        fig.show()
    return fig


def line(*args, **kwargs) -> Figure:
    """``line(y)`` or ``line(x, y)`` - draw and print a line plot."""
    return _run("line", args, kwargs)


def scatter(*args, **kwargs) -> Figure:
    """``scatter(y)`` or ``scatter(x, y)`` - draw and print a scatter plot."""
    return _run("scatter", args, kwargs)


def bar(*args, **kwargs) -> Figure:
    """``bar(values, labels=None)`` - vertical bar chart."""
    return _run("bar", args, kwargs)


def hbar(*args, **kwargs) -> Figure:
    """``hbar(values, labels=None)`` - horizontal bar chart."""
    return _run("hbar", args, kwargs)


def hist(*args, **kwargs) -> Figure:
    """``hist(data, bins="auto")`` - histogram of raw samples."""
    return _run("hist", args, kwargs)


def sparkline(
    data: Sequence[float],
    color: ColorLike = None,
    palette: Optional[str] = None,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
    stream=None,
) -> str:
    """Return a one-line block sparkline, optionally colored by value."""
    values = []
    for v in data:
        try:
            values.append(float(v))
        except (TypeError, ValueError):
            values.append(None)  # type: ignore[arg-type]
    finite = [v for v in values if v is not None]
    if not finite:
        return ""
    stream = stream or sys.stdout
    blocks = " ▁▂▃▄▅▆▇█" if supports_unicode(stream) else " ....####"
    mode = detect_mode(stream)
    low = min(finite) if lo is None else lo
    high = max(finite) if hi is None else hi
    span = (high - low) or 1.0

    ramp = _color.palette(palette, 9) if palette else None
    solid = parse_color(color)
    out = []
    current = None
    for v in values:
        if v is None:
            out.append(" ")
            continue
        t = max(0.0, min(1.0, (v - low) / span))
        idx = max(1, int(round(t * 8)))
        c = ramp[idx] if ramp else solid
        if mode != "none" and c != current:
            out.append(fg_seq(c, mode))
            current = c
        out.append(blocks[idx])
    if mode != "none" and current is not None:
        out.append(_color.RESET)
    return "".join(out)
