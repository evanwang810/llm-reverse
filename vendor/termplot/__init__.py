"""termplot - colorful plots rendered straight into the terminal.

    import termplot as tp

    tp.line([1, 4, 9, 16, 25], title="squares")
    tp.bar([3, 7, 2], labels=["a", "b", "c"])
    tp.hist(samples, bins=20)

    fig = tp.Figure(title="two series", grid=True)
    fig.line(xs, ys, label="signal")
    fig.line(xs, noise, label="noise", color="orange")
    fig.show()
"""

from .api import bar, hbar, hist, line, scatter, sparkline
from .color import PALETTES, color_names, palette, palette_names
from .figure import Figure

__version__ = "0.1.0"

__all__ = [
    "Figure",
    "line",
    "scatter",
    "bar",
    "hbar",
    "hist",
    "sparkline",
    "palette",
    "palette_names",
    "color_names",
    "PALETTES",
    "__version__",
]
