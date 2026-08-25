"""Axis maths: nice tick selection, number formatting, histogram binning."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple


def nice_num(x: float, do_round: bool) -> float:
    """Round ``x`` to the nearest 1/2/5 * 10^n."""
    if x <= 0 or not math.isfinite(x):
        return 1.0
    exp = math.floor(math.log10(x))
    f = x / (10 ** exp)
    if do_round:
        nf = 1.0 if f < 1.5 else 2.0 if f < 3 else 5.0 if f < 7 else 10.0
    else:
        nf = 1.0 if f <= 1 else 2.0 if f <= 2 else 5.0 if f <= 5 else 10.0
    return nf * (10 ** exp)


def nice_bounds(lo: float, hi: float, n: int = 5) -> Tuple[float, float, List[float]]:
    """Expand ``[lo, hi]`` outward to round numbers and return the ticks."""
    if hi < lo:
        lo, hi = hi, lo
    if hi == lo:
        if lo == 0:
            lo, hi = -1.0, 1.0
        else:
            pad = abs(lo) * 0.1
            lo, hi = lo - pad, hi + pad
    n = max(2, n)
    rng = nice_num(hi - lo, False)
    step = nice_num(rng / (n - 1), True)
    gmin = math.floor(lo / step) * step
    gmax = math.ceil(hi / step) * step
    vals, i = [], 0
    while True:
        v = gmin + i * step
        if v > gmax + step * 1e-9:
            break
        vals.append(_tidy(v))
        i += 1
        if i > 1000:
            break
    return _tidy(gmin), _tidy(gmax), vals


def ticks_within(lo: float, hi: float, n: int = 5) -> List[float]:
    """Ticks on a round step that stay inside ``[lo, hi]``."""
    if hi <= lo:
        return [_tidy(lo)]
    n = max(2, n)
    step = nice_num((hi - lo) / (n - 1), True)
    start = math.ceil(lo / step) * step
    vals, i = [], 0
    while True:
        v = start + i * step
        if v > hi + step * 1e-9:
            break
        vals.append(_tidy(v))
        i += 1
        if i > 1000:
            break
    return vals or [_tidy(lo), _tidy(hi)]


def _tidy(v: float) -> float:
    r = round(v, 12)
    return 0.0 if r == 0 else r


def fmt_num(v: float, sig: int = 4) -> str:
    """Compact human-facing number: SI suffixes for the big end, %g elsewhere."""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    if abs(v) < 1e-12:
        return "0"
    a = abs(v)
    if a >= 1e5:
        for div, suffix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
            if a >= div:
                return "%s%s" % (_g(v / div, sig - 1), suffix)
    if a < 1e-3:
        return ("%.*e" % (sig - 2, v)).replace("e-0", "e-").replace("e+0", "e+")
    return _g(v, sig)


def _g(v: float, sig: int) -> str:
    s = "%.*g" % (max(1, sig), v)
    if "e" in s:
        s = ("%.*f" % (max(0, sig), v)).rstrip("0").rstrip(".")
    return s or "0"


def tick_labels(vals: Sequence[float]) -> List[str]:
    """Format ticks with a shared number of decimals, just enough to stay distinct."""
    if not vals:
        return []
    a = max(abs(v) for v in vals)
    if a >= 1e5 or (0 < a < 1e-2):
        return [fmt_num(v) for v in vals]
    distinct = len({round(v, 12) for v in vals})
    for decimals in range(0, 7):
        labels = [_fixed(v, decimals) for v in vals]
        if len(set(labels)) == distinct:
            return labels
    return [fmt_num(v) for v in vals]


def _fixed(v: float, decimals: int) -> str:
    s = "%.*f" % (decimals, v)
    if float(s) == 0:
        s = "%.*f" % (decimals, 0.0)
    return s


def quantile(sorted_data: Sequence[float], q: float) -> float:
    n = len(sorted_data)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_data[0])
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(n - 1, lo + 1)
    frac = pos - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


def histogram(data: Sequence[float], bins="auto") -> Tuple[List[int], List[float]]:
    """Return ``(counts, edges)``; ``bins`` is an int or ``"auto"``/``"sturges"``/``"fd"``."""
    values = sorted(float(v) for v in data)
    n = len(values)
    if n == 0:
        raise ValueError("hist() needs at least one value")
    lo, hi = values[0], values[-1]
    if isinstance(bins, int):
        k = max(1, bins)
    else:
        mode = str(bins).lower()
        sturges = int(math.ceil(math.log2(n) + 1)) if n > 1 else 1
        k = sturges
        if mode in ("auto", "fd"):
            iqr = quantile(values, 0.75) - quantile(values, 0.25)
            width = 2 * iqr / (n ** (1 / 3)) if iqr > 0 else 0.0
            if width > 0 and hi > lo:
                k = int(math.ceil((hi - lo) / width))
                if mode == "auto":
                    k = max(k, sturges)
        k = max(4, min(60, k))
    if hi == lo:
        hi = lo + 1.0
    edges = [lo + i * (hi - lo) / k for i in range(k + 1)]
    counts = [0] * k
    span = hi - lo
    for v in values:
        idx = int((v - lo) / span * k)
        if idx >= k:
            idx = k - 1
        counts[idx] += 1
    return counts, edges
