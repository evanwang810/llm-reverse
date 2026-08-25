"""The Figure object: holds series, does layout, renders to a string."""

from __future__ import annotations

import math
import shutil
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from . import color as _color
from .braille import Braille
from .color import RGB, ColorLike, detect_mode, dim, parse_color, supports_unicode
from .scale import fmt_num, histogram, nice_bounds, tick_labels, ticks_within
from .screen import Charset, Screen

MIN_PLOT_COLS = 8
MIN_PLOT_ROWS = 3


@dataclass
class Series:
    kind: str                                   # line | scatter | bar | hbar | hist
    x: List[Optional[float]] = field(default_factory=list)
    y: List[Optional[float]] = field(default_factory=list)
    label: Optional[str] = None
    color: Optional[RGB] = None
    colors: Optional[List[RGB]] = None          # per-bar colors
    marker: Optional[str] = None
    fill: bool = False
    labels: Optional[List[str]] = None          # bar category labels
    edges: Optional[List[float]] = None         # histogram bin edges
    show_values: bool = False


def _num(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _finite(seq: Sequence[Optional[float]]) -> List[float]:
    return [v for v in seq if v is not None]


def _place(items: Sequence[Tuple[float, str]], width: int) -> List[Tuple[int, str]]:
    """Greedily lay out centered labels, dropping any that would collide."""
    out: List[Tuple[int, str]] = []
    last_end = -2
    for pos, text in items:
        if not text:
            continue
        start = int(round(pos - (len(text) - 1) / 2.0))
        start = max(0, min(width - len(text), start))
        if start > last_end:
            out.append((start, text))
            last_end = start + len(text)
    return out


class Figure:
    """A plot canvas. Add series, then :meth:`show` or :meth:`render` it."""

    def __init__(
        self,
        width: Optional[int] = None,
        height: int = 20,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        palette: str = "default",
        color: Optional[bool] = None,
        grid: bool = False,
        legend: bool = True,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        unicode: Optional[bool] = None,
        stream=None,
    ):
        self.stream = stream or sys.stdout
        self.width = int(width) if width else self._auto_width()
        self.height = max(6, int(height))
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.palette = palette
        self.grid = grid
        self.legend = legend
        self.xlim = xlim
        self.ylim = ylim
        self._series: List[Series] = []

        if color is False:
            self._mode = "none"
        elif color is True:
            mode = detect_mode(self.stream)
            self._mode = "truecolor" if mode == "none" else mode
        else:
            self._mode = detect_mode(self.stream)

        use_unicode = supports_unicode(self.stream) if unicode is None else bool(unicode)
        self.charset = Charset(use_unicode)
        self.axis_color = parse_color("#7A7A85")
        self.title_color = parse_color("#EDEDF2")
        self.text_color = parse_color("#B4B4BE")

    # ------------------------------------------------------------------ #
    # series
    # ------------------------------------------------------------------ #
    def line(self, *args, label=None, color: ColorLike = None, fill: bool = False) -> "Figure":
        """``line(y)`` or ``line(x, y)``. ``None``/``nan`` values break the line."""
        x, y = self._xy_args(args)
        self._series.append(
            Series("line", x, y, label=label, color=parse_color(color), fill=fill)
        )
        return self

    def scatter(self, *args, label=None, color: ColorLike = None, marker: Optional[str] = None) -> "Figure":
        """``scatter(y)`` or ``scatter(x, y)``. ``marker=None`` uses braille dots."""
        x, y = self._xy_args(args)
        self._series.append(
            Series("scatter", x, y, label=label, color=parse_color(color), marker=marker)
        )
        return self

    def bar(self, values, labels=None, label=None, color: ColorLike = None,
            colors=None, show_values: bool = False) -> "Figure":
        """Vertical bars. Several ``bar()`` calls become grouped bars."""
        vals = [_num(v) for v in values]
        self._series.append(
            Series(
                "bar", list(range(len(vals))), vals, label=label,
                color=parse_color(color),
                colors=[parse_color(c) for c in colors] if colors else None,
                labels=[str(t) for t in labels] if labels is not None else None,
                show_values=show_values,
            )
        )
        return self

    def hbar(self, values, labels=None, label=None, color: ColorLike = None,
             colors=None, show_values: bool = True) -> "Figure":
        """Horizontal bars, one row per category."""
        vals = [_num(v) for v in values]
        self._series.append(
            Series(
                "hbar", list(range(len(vals))), vals, label=label,
                color=parse_color(color),
                colors=[parse_color(c) for c in colors] if colors else None,
                labels=[str(t) for t in labels] if labels is not None else None,
                show_values=show_values,
            )
        )
        return self

    def hist(self, data, bins="auto", label=None, color: ColorLike = None,
             colors=None) -> "Figure":
        """Histogram of raw samples. ``bins`` is an int or ``auto``/``fd``/``sturges``."""
        counts, edges = histogram([v for v in (_num(d) for d in data) if v is not None], bins)
        self._series.append(
            Series(
                "hist", list(range(len(counts))), [float(c) for c in counts],
                label=label, color=parse_color(color),
                colors=[parse_color(c) for c in colors] if colors else None,
                edges=edges,
            )
        )
        return self

    # ------------------------------------------------------------------ #
    # output
    # ------------------------------------------------------------------ #
    def render(self) -> str:
        if not self._series:
            raise ValueError("nothing to plot: add a series first")
        self._assign_colors()
        kinds = {s.kind for s in self._series}
        if "hbar" in kinds:
            return self._render_hbar()
        if kinds & {"bar", "hist"}:
            return self._render_vbar()
        return self._render_xy()

    def show(self, stream=None) -> None:
        stream = stream or self.stream
        text = self.render()
        if self.charset.unicode and not supports_unicode(stream):
            try:
                stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
            except Exception:
                self.charset = Charset(False)
                text = self.render()
        print(text, file=stream)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.render()

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _auto_width() -> int:
        cols = shutil.get_terminal_size((80, 24)).columns
        return max(40, min(100, cols - 1))

    @staticmethod
    def _xy_args(args) -> Tuple[List[Optional[float]], List[Optional[float]]]:
        if len(args) == 1:
            y = [_num(v) for v in args[0]]
            return [float(i) for i in range(len(y))], y
        if len(args) == 2:
            x = [_num(v) for v in args[0]]
            y = [_num(v) for v in args[1]]
            n = min(len(x), len(y))
            return x[:n], y[:n]
        raise TypeError("expected (y) or (x, y)")

    def _assign_colors(self) -> None:
        bars = [s for s in self._series if s.kind in ("bar", "hbar")]
        needed = sum(1 for s in self._series if s.color is None)
        cycle = _color.palette(self.palette, max(1, needed) + 1)
        i = 0
        for s in self._series:
            if s.color is None:
                s.color = cycle[i % len(cycle)]
                i += 1
        # a lone bar series gets one palette color per bar
        if len(bars) == 1 and bars[0].colors is None and len(self._series) == 1:
            n = len(bars[0].y)
            bars[0].colors = _color.palette(self.palette, n)

    def _legend_lines(self, width: int) -> List[List[Tuple[str, Optional[RGB]]]]:
        entries = [(s.label, s.color) for s in self._series if s.label]
        if not entries or not self.legend:
            return []
        cs = self.charset
        lines: List[List[Tuple[str, Optional[RGB]]]] = [[]]
        used = 0
        for label, col in entries:
            text = "%s %s" % (cs.legend, label)
            extra = len(text) + (2 if used else 0)
            if used and used + extra > width:
                lines.append([])
                used = 0
                extra = len(text)
            if used:
                lines[-1].append(("  ", None))
            lines[-1].append((text, col))
            used += extra
        return lines

    def _draw_legend(self, screen: Screen, row0: int, lines) -> None:
        for i, line in enumerate(lines):
            x = 0
            for text, col in line:
                screen.text(x, row0 + i, text, col)
                x += len(text)

    def _draw_title(self, screen: Screen) -> None:
        if self.title:
            start = max(0, (self.width - len(self.title)) // 2)
            screen.text(start, 0, self.title, self.title_color)

    def _draw_ylabel(self, screen: Screen, top: int, rows: int) -> None:
        if not self.ylabel:
            return
        text = self.ylabel[:rows]
        start = top + max(0, (rows - len(text)) // 2)
        for i, char in enumerate(text):
            screen.put(0, start + i, char, self.text_color)

    def _draw_yaxis(self, screen: Screen, axis_col: int, top: int, rows: int,
                    ticks: Sequence[float], y0: float, y1: float) -> List[int]:
        cs = self.charset
        screen.vline(axis_col, top, top + rows - 1, cs.vaxis, self.axis_color)
        labels = tick_labels(ticks)
        span = (y1 - y0) or 1.0
        tick_rows = []
        for v, text in zip(ticks, labels):
            row = top + int(round((1 - (v - y0) / span) * (rows - 1)))
            if not (top <= row < top + rows):
                continue
            screen.put(axis_col, row, cs.ytick, self.axis_color)
            screen.text_right(axis_col - 1, row, text, self.text_color)
            tick_rows.append(row)
        return tick_rows

    def _draw_grid(self, screen: Screen, x0: int, cols: int, top: int, rows: int,
                   tick_rows: Sequence[int], tick_cols: Sequence[int]) -> None:
        if not self.grid:
            return
        cs = self.charset
        gcolor = dim(self.axis_color, 0.55)
        for row in tick_rows:
            screen.hline(x0, x0 + cols - 1, row, cs.grid_h, gcolor)
        for col in tick_cols:
            if col <= x0:  # would sit right on top of the y axis
                continue
            for row in range(top, top + rows):
                char = cs.cross if screen.get(col, row) == cs.grid_h else cs.grid_v
                screen.put(col, row, char, gcolor)

    def _draw_xaxis(self, screen: Screen, axis_col: int, x0: int, cols: int, row: int,
                    ticks: Sequence[Tuple[int, str]]) -> None:
        cs = self.charset
        screen.put(axis_col, row, cs.corner, self.axis_color)
        screen.hline(x0, x0 + cols - 1, row, cs.haxis, self.axis_color)
        for col, _ in ticks:
            screen.put(col, row, cs.xtick, self.axis_color)
        for start, text in _place([(c - x0, t) for c, t in ticks], cols):
            screen.text(x0 + start, row + 1, text, self.text_color)

    def _draw_xlabel(self, screen: Screen, x0: int, cols: int, row: int) -> None:
        if self.xlabel:
            start = x0 + max(0, (cols - len(self.xlabel)) // 2)
            screen.text(start, row, self.xlabel, self.text_color)

    # ------------------------------------------------------------------ #
    # line / scatter
    # ------------------------------------------------------------------ #
    def _render_xy(self) -> str:
        legend_lines = self._legend_lines(self.width)
        title_rows = 1 if self.title else 0
        xlabel_rows = 1 if self.xlabel else 0
        rows = self.height - title_rows - xlabel_rows - len(legend_lines) - 2
        if rows < MIN_PLOT_ROWS:
            raise ValueError("figure height too small for this plot")

        ys = [v for s in self._series for v in _finite(s.y)]
        xs = [x for s in self._series for x, y in zip(s.x, s.y) if x is not None and y is not None]
        if not ys or not xs:
            raise ValueError("no finite data points to plot")

        n_yticks = max(2, min(7, rows // 3 + 1))
        if self.ylim:
            y0, y1 = float(self.ylim[0]), float(self.ylim[1])
            yticks = ticks_within(y0, y1, n_yticks)
        else:
            y0, y1, yticks = nice_bounds(min(ys), max(ys), n_yticks)
        if y1 <= y0:
            y1 = y0 + 1.0

        x0v, x1v = (float(self.xlim[0]), float(self.xlim[1])) if self.xlim else (min(xs), max(xs))
        if x1v <= x0v:
            x0v, x1v = x0v - 0.5, x1v + 0.5

        gutter = max(len(t) for t in tick_labels(yticks))
        left = 2 if self.ylabel else 0
        axis_col = left + gutter
        plot_x = axis_col + 1
        cols = self.width - plot_x
        if cols < MIN_PLOT_COLS:
            raise ValueError("figure width too small for this plot")

        top = title_rows
        axis_row = top + rows
        total = axis_row + 2 + xlabel_rows + len(legend_lines)
        screen = Screen(self.width, total, self._mode)
        self._draw_title(screen)
        self._draw_ylabel(screen, top, rows)

        n_xticks = max(2, min(9, cols // 9 + 1))
        xt = ticks_within(x0v, x1v, n_xticks)
        xspan = x1v - x0v
        xtick_cols = [(plot_x + int(round((v - x0v) / xspan * (cols - 1))), t)
                      for v, t in zip(xt, tick_labels(xt))]

        tick_rows = self._draw_yaxis(screen, axis_col, top, rows, yticks, y0, y1)
        self._draw_grid(screen, plot_x, cols, top, rows, tick_rows, [c for c, _ in xtick_cols])

        canvas = Braille(cols, rows)
        yspan = y1 - y0
        pw, ph = canvas.pw, canvas.ph

        def px(x: float) -> int:
            return int(round((x - x0v) / xspan * (pw - 1)))

        def py(y: float) -> int:
            return int(round((1 - (y - y0) / yspan) * (ph - 1)))

        base_py = py(min(max(0.0, y0), y1))
        for s in self._series:
            points = [(x, y) for x, y in zip(s.x, s.y)]
            if s.kind == "scatter" and s.marker:
                for x, y in points:
                    if x is None or y is None:
                        continue
                    cx = plot_x + int(round((x - x0v) / xspan * (cols - 1)))
                    cy = top + int(round((1 - (y - y0) / yspan) * (rows - 1)))
                    if plot_x <= cx < plot_x + cols and top <= cy < top + rows:
                        screen.put(cx, cy, s.marker, s.color)
                continue
            if s.kind == "scatter":
                for x, y in points:
                    if x is not None and y is not None:
                        canvas.set(px(x), py(y), s.color)
                continue
            if s.fill:
                fill_color = dim(s.color or (255, 255, 255), 0.4)
                for (xa, ya), (xb, yb) in zip(points, points[1:]):
                    if None in (xa, ya, xb, yb):
                        continue
                    pa, pb = px(xa), px(xb)
                    for p in range(min(pa, pb), max(pa, pb) + 1):
                        t = 0.0 if pa == pb else (p - pa) / (pb - pa)
                        canvas.vline(p, base_py, py(ya + (yb - ya) * t), fill_color)
            for (xa, ya), (xb, yb) in zip(points, points[1:]):
                if None in (xa, ya, xb, yb):
                    continue
                canvas.line(px(xa), py(ya), px(xb), py(yb), s.color)
            for x, y in points:
                if x is not None and y is not None:
                    canvas.set(px(x), py(y), s.color)
        canvas.blit(screen, plot_x, top, self.charset)

        self._draw_xaxis(screen, axis_col, plot_x, cols, axis_row, xtick_cols)
        if xlabel_rows:
            self._draw_xlabel(screen, plot_x, cols, axis_row + 2)
        if legend_lines:
            self._draw_legend(screen, axis_row + 2 + xlabel_rows, legend_lines)
        return screen.render()

    # ------------------------------------------------------------------ #
    # vertical bars / histograms
    # ------------------------------------------------------------------ #
    def _render_vbar(self) -> str:
        series = [s for s in self._series if s.kind in ("bar", "hist")]
        is_hist = series[0].kind == "hist"
        n_cat = max(len(s.y) for s in series)
        n_ser = len(series)

        legend_lines = self._legend_lines(self.width)
        title_rows = 1 if self.title else 0
        xlabel_rows = 1 if self.xlabel else 0
        show_values = any(s.show_values for s in series)
        rows = self.height - title_rows - xlabel_rows - len(legend_lines) - 2
        if rows < MIN_PLOT_ROWS:
            raise ValueError("figure height too small for this plot")

        vals = [v for s in series for v in _finite(s.y)] or [0.0]
        lo, hi = min(min(vals), 0.0), max(max(vals), 0.0)
        if self.ylim:
            y0, y1 = float(self.ylim[0]), float(self.ylim[1])
            yticks = ticks_within(y0, y1, max(2, min(7, rows // 3 + 1)))
        else:
            n_yticks = max(2, min(7, rows // 3 + 1))
            y0, y1, yticks = nice_bounds(lo, hi, n_yticks)
            if y0 < 0 < y1:
                # stretch the negative side so zero lands on a row boundary
                k = min(rows - 1, max(1, int(math.ceil(rows * (0 - y0) / (y1 - y0)))))
                y0 = k * y1 / (k - rows)
                yticks = ticks_within(y0, y1, n_yticks)
        if y1 <= y0:
            y1 = y0 + 1.0

        gutter = max(len(t) for t in tick_labels(yticks))
        left = 2 if self.ylabel else 0
        axis_col = left + gutter
        plot_x = axis_col + 1
        cols = self.width - plot_x
        if cols < max(MIN_PLOT_COLS, n_cat):
            raise ValueError("figure width too small for %d bars" % n_cat)

        top = title_rows
        axis_row = top + rows
        total = axis_row + 2 + xlabel_rows + len(legend_lines)
        screen = Screen(self.width, total, self._mode)
        self._draw_title(screen)
        self._draw_ylabel(screen, top, rows)
        tick_rows = self._draw_yaxis(screen, axis_col, top, rows, yticks, y0, y1)
        self._draw_grid(screen, plot_x, cols, top, rows, tick_rows, [])

        blocks = self.charset.vblocks
        row_span = (y1 - y0) / rows
        centers: List[float] = []

        for c in range(n_cat):
            start = plot_x + int(round(c * cols / n_cat))
            end = plot_x + int(round((c + 1) * cols / n_cat))
            slot = max(1, end - start)
            gap = 0 if is_hist else (1 if slot >= 3 else 0)
            bar_w = max(1, (slot - gap) // n_ser)
            centers.append(start + (slot - gap) / 2.0 - 0.5)
            for si, s in enumerate(series):
                v = s.y[c] if c < len(s.y) else None
                if v is None:
                    continue
                bx = start + si * bar_w
                if bx >= start + slot - gap:
                    break
                color = (s.colors[c % len(s.colors)] if s.colors else s.color)
                width_here = min(bar_w, start + slot - gap - bx)
                self._paint_column(screen, bx, width_here, top, rows, v, y0,
                                   row_span, blocks, color)
                if show_values and n_ser == 1:
                    self._paint_bar_value(screen, bx, width_here, top, rows, v, y0,
                                          row_span, color)

        # zero line when the axis straddles it
        if y0 < 0 < y1:
            zrow = top + int(round((1 - (0 - y0) / (y1 - y0)) * (rows - 1)))
            for x in range(plot_x, plot_x + cols):
                if screen.get(x, zrow) == " ":
                    screen.put(x, zrow, self.charset.haxis, dim(self.axis_color, 0.7))

        if is_hist:
            edges = series[0].edges or []
            step = max(1, len(edges) // max(1, min(8, cols // 8)))
            marks = []
            for i in range(0, len(edges), step):
                col = plot_x + int(round(i * cols / max(1, len(edges) - 1)))
                marks.append((min(col, plot_x + cols - 1), fmt_num(edges[i], 3)))
        else:
            labels = None
            for s in series:
                if s.labels:
                    labels = s.labels
                    break
            marks = []
            if labels:
                for c, center in enumerate(centers):
                    if c < len(labels):
                        marks.append((int(round(center)), labels[c]))
            else:
                marks = [(int(round(centers[c])), str(c)) for c in range(n_cat)]

        self._draw_xaxis(screen, axis_col, plot_x, cols, axis_row, marks)
        if xlabel_rows:
            self._draw_xlabel(screen, plot_x, cols, axis_row + 2)
        if legend_lines:
            self._draw_legend(screen, axis_row + 2 + xlabel_rows, legend_lines)
        return screen.render()

    def _paint_column(self, screen, bx, bar_w, top, rows, v, y0, row_span, blocks, color):
        lo_v, hi_v = (0.0, v) if v >= 0 else (v, 0.0)
        for r in range(rows):
            row_low = y0 + r * row_span
            row_high = row_low + row_span
            overlap = (min(hi_v, row_high) - max(lo_v, row_low)) / row_span
            if overlap <= 0.02:
                continue
            if v >= 0:
                idx = max(1, min(8, int(round(overlap * 8))))
            else:
                idx = 8 if overlap >= 0.5 else 0
            if idx == 0:
                continue
            char = blocks[idx]
            y = top + rows - 1 - r
            for x in range(bx, bx + bar_w):
                screen.put(x, y, char, color)

    def _paint_bar_value(self, screen, bx, bar_w, top, rows, v, y0, row_span, color):
        text = fmt_num(v)
        if len(text) > bar_w + 2:
            return
        r = (v - y0) / row_span
        if v >= 0:
            y = top + rows - 1 - int(math.ceil(r))      # just above the bar top
            if y < top:
                return
        else:
            y = top + rows - int(math.floor(r))         # just below the bar bottom
            if y >= top + rows:
                return
        x = bx + (bar_w - len(text)) // 2
        screen.text(max(0, x), y, text, dim(color or self.text_color, 0.85))

    # ------------------------------------------------------------------ #
    # horizontal bars
    # ------------------------------------------------------------------ #
    def _render_hbar(self) -> str:
        series = [s for s in self._series if s.kind == "hbar"]
        n_cat = max(len(s.y) for s in series)
        n_ser = len(series)
        labels = None
        for s in series:
            if s.labels:
                labels = s.labels
                break
        labels = labels or [str(i) for i in range(n_cat)]

        legend_lines = self._legend_lines(self.width)
        title_rows = 1 if self.title else 0
        xlabel_rows = 1 if self.xlabel else 0
        group = n_ser + (1 if n_ser > 1 else 0)
        rows = n_cat * group - (1 if n_ser > 1 else 0)

        vals = [v for s in series for v in _finite(s.y)] or [0.0]
        lo, hi = min(min(vals), 0.0), max(max(vals), 0.0)
        show_values = any(s.show_values for s in series)
        vwidth = (max(len(fmt_num(v)) for v in vals) + 1) if show_values else 0

        gutter = min(max(len(t) for t in labels), max(6, self.width // 3))
        bar_x = gutter + 1 + (vwidth if lo < 0 else 0)   # room for negative value text
        cols = self.width - bar_x - vwidth
        if cols < MIN_PLOT_COLS:
            raise ValueError("figure width too small for these labels")

        span = (hi - lo) or 1.0
        zero_col = int(round((0.0 - lo) / span * cols)) if lo < 0 else 0

        top = title_rows
        total = top + rows + xlabel_rows + len(legend_lines)
        screen = Screen(self.width, total, self._mode)
        self._draw_title(screen)

        blocks = self.charset.hblocks
        for c in range(n_cat):
            base_row = top + c * group
            text = labels[c] if c < len(labels) else ""
            screen.text_right(gutter - 1, base_row, text[:gutter], self.text_color)
            for si, s in enumerate(series):
                v = s.y[c] if c < len(s.y) else None
                if v is None:
                    continue
                color = (s.colors[c % len(s.colors)] if s.colors else s.color)
                row = base_row + si
                cells = abs(v) / span * cols
                if v >= 0:
                    full = int(cells)
                    frac = int(round((cells - full) * 8))
                    x = bar_x + zero_col
                    for i in range(full):
                        screen.put(x + i, row, blocks[8], color)
                    if frac:
                        screen.put(x + full, row, blocks[frac], color)
                    end = x + full + (1 if frac else 0)
                    if s.show_values:
                        screen.text(min(end + 1, self.width - 1), row, fmt_num(v),
                                    dim(color or self.text_color, 0.85))
                else:
                    full = int(round(cells))
                    x = max(bar_x, bar_x + zero_col - full)
                    for i in range(full):
                        screen.put(max(bar_x, x + i), row, blocks[8], color)
                    if s.show_values:
                        text_v = fmt_num(v)
                        screen.text(max(0, x - len(text_v) - 1), row, text_v,
                                    dim(color or self.text_color, 0.85))
        if lo < 0:
            for r in range(top, top + rows):
                if screen.get(bar_x + zero_col, r) == " ":
                    screen.put(bar_x + zero_col, r, self.charset.vaxis,
                               dim(self.axis_color, 0.7))

        if xlabel_rows:
            self._draw_xlabel(screen, bar_x, cols, top + rows)
        if legend_lines:
            self._draw_legend(screen, top + rows + xlabel_rows, legend_lines)
        return screen.render()
