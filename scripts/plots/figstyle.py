"""Shared figure furniture for every plot script in this directory.

Holds the neutral colors, the axes styling, and the title, legend and labeling
helpers. The CA scripts reach these through ca_figlib, which re-exports them, so
both families of figures are drawn with one set of conventions rather than two.

Hue assignments live beside the vocabulary they name: ca_figstyle.py for the CA
integrator's entities, cpu_figstyle.py for the CPU regime's. Both draw from the
validated ramps here, so one rule checks every color in the thesis.
"""

import math


C_BETTER = "#5F5E5A"
# Body text, grid lines, and the subordinate register a provenance label or a
# unit suffix is set in. Defined here rather than per family so a figure moved
# between them does not change weight.
C_INK = "#22201C"
C_GRID = "#CFCBC2"
C_MUTED = "#6E6A62"

CATEGORICAL = [
    "#3D3BA8",  # indigo
    "#0F6E56",  # pine
    "#A8431A",  # rust
    "#7A2E7E",  # plum
    "#1F6F9B",  # steel
    "#8A6A00",  # ochre
]

# Sequential magnitude: one hue, light to dark.
SEQUENTIAL = ["#D6E4EC", "#9CC0D4", "#5F94B4", "#31688E", "#1B3F5B"]

# Signed residuals: two hues, neutral at zero, no hue at the crossing.
DIVERGING_LOW = "#A8431A"
DIVERGING_MID = "#EFEDE8"
DIVERGING_HIGH = "#1F6F9B"

_ARROW_X = {"left": 0.045, "right": 0.955}


def mark_better(ax, direction: str, loc: str = "upper left", length: float = 0.13,
                at: tuple[float, float] | None = None,
                color: str = C_BETTER, fontsize: float = 7.5) -> None:
    """Mark which way is better on the y-axis: a short arrow labeled "better".

    direction is "up" or "down"; loc anchors the arrow in one of the four axes
    corners (axes-fraction coords, so it is immune to log scales and to the data
    limits). Pick the corner the data does not occupy.

    When no corner is free, pass at=(x, y_center) in axes fraction to place the
    arrow's midpoint by hand; loc is then ignored.
    """
    if direction not in ("up", "down"):
        raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")

    if at is not None:
        x, y_centre = at
        side = "left" if x < 0.5 else "right"
        yhi, ylo = y_centre + length / 2.0, y_centre - length / 2.0
    else:
        side = "left" if "left" in loc else "right"
        x = _ARROW_X[side]
        if "upper" in loc:
            yhi, ylo = 0.96, 0.96 - length
        else:
            yhi, ylo = 0.04 + length, 0.04
    tip, tail = (yhi, ylo) if direction == "up" else (ylo, yhi)

    ax.annotate("", xy=(x, tip), xytext=(x, tail), xycoords="axes fraction",
                zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1,
                                shrinkA=0, shrinkB=0))
    # Center the label on its own anchor (default rotation_mode aligns, then rotates
    # about the point) and clear the shaft by a fixed fraction, so the two never touch.
    ax.text(x + (0.038 if side == "left" else -0.038), (tip + tail) / 2.0, "better",
            rotation=90, va="center", ha="center", fontsize=fontsize, color=color,
            zorder=6, transform=ax.transAxes)


PANEL_SIZE = 9.5

# No figure-level headline helper. The finding belongs in the LaTeX caption,
# which travels with the figure in the document; a suptitle would only repeat
# it above the panels. Panel titles stay, since they name which panel is which
# rather than restating the caption.


def panel_title(ax, text: str) -> None:
    """Name a panel.

    The name says which panel this is, not what it shows: a finding stated here
    would only repeat the caption, which travels with the figure in the
    document.

    Args:
        ax: Axes to title.
        text: The panel's name.
    """
    ax.set_title(text, fontsize=PANEL_SIZE, color=C_INK)


def legend(ax, *, loc: str = "best", ncol: int = 1, opaque: bool = False,
           **kwargs):
    """Draw a legend in the one style every figure in this set uses.

    Args:
        ax: Axes to attach the legend to.
        loc: Matplotlib location string.
        ncol: Number of columns.
        opaque: Give the legend a solid background. Needed only where it has
            to sit over grid or reference lines, which otherwise run straight
            through the label text.
        **kwargs: Passed through to ``Axes.legend``, for explicit handles
            and labels.

    Returns:
        The created legend.
    """
    frame = (dict(frameon=True, facecolor="white", edgecolor="none",
                  framealpha=0.94) if opaque else dict(frameon=False))
    return ax.legend(
        fontsize=8.0, loc=loc, ncol=ncol, labelcolor=C_INK,
        handlelength=2.6, borderaxespad=0.4, **frame, **kwargs)


def label_series(ax, x, y, text: str, color: str, *, dy: float = 9.0,
                 dx: float = 0.0, ha: str = "center",
                 size: float = 9.0) -> None:
    """Name a series on the series itself.

    Cheaper for the reader than a legend, which asks them to hold a color in
    memory and match it across the figure. Only worth it where the label has
    clear space; where series converge, a legend is the better tool.

    Args:
        ax: Axes to draw on.
        x: Data x of the anchor point.
        y: Data y of the anchor point.
        text: The series name.
        color: The series color, so label and data read as one thing.
        dy: Vertical offset in points from the anchor.
        dx: Horizontal offset in points from the anchor.
        ha: Horizontal alignment.
        size: Font size in points.
    """
    ax.annotate(
        text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
        ha=ha, va="bottom" if dy >= 0 else "top", fontsize=size,
        color=color, fontweight="medium", zorder=6)


def label_value(ax, x, y, text: str, *, dy: float = 6.0, dx: float = 0.0,
                color: str | None = None, size: float = 7.6,
                ha: str = "center", weight: str = "normal") -> None:
    """Put a number on the mark it belongs to.

    A reader comparing two bars that differ by a tenth of a millisecond is
    guessing, and a number that needs the script re-run to recover is not on the
    figure.

    Args:
        ax: Axes to draw on.
        x: Data x of the mark.
        y: Data y of the mark.
        text: The formatted number.
        dy: Vertical offset in points from the mark.
        dx: Horizontal offset in points from the mark.
        color: Text color; defaults to the subordinate gray.
        size: Font size in points.
        ha: Horizontal alignment.
        weight: Font weight.
    """
    ax.annotate(
        text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
        ha=ha, va="bottom" if dy >= 0 else "top", fontsize=size,
        color=color or C_MUTED, fontweight=weight, zorder=6,
        annotation_clip=False)


def style_axes(ax, grid_axis: str = "both") -> None:
    """Apply the shared axes styling: light grid, no top or right spine.

    Args:
        ax: Axes to style.
        grid_axis: Which axis carries grid lines: "both", "x" or "y".
    """
    ax.grid(alpha=0.35, lw=0.5, color=C_GRID, axis=grid_axis)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(C_GRID)
    ax.tick_params(colors=C_INK, labelsize=8.5)
    ax.xaxis.label.set_color(C_INK)
    ax.yaxis.label.set_color(C_INK)
    ax.title.set_color(C_INK)


def _srgb_to_linear(channel: float) -> float:
    return (
        channel / 12.92
        if channel <= 0.04045
        else ((channel + 0.055) / 1.055) ** 2.4
    )


def _hex_to_linear(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return tuple(
        _srgb_to_linear(int(value[q : q + 2], 16) / 255.0) for q in (0, 2, 4)
    )


def _linear_to_oklab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    r, g, b = rgb
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l, m, s = (math.copysign(abs(v) ** (1.0 / 3.0), v) for v in (l, m, s))
    return (
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    )


_DICHROMAT = {
    "deuteranopia": (
        (0.625, 0.375, 0.000),
        (0.700, 0.300, 0.000),
        (0.000, 0.300, 0.700),
    ),
    "protanopia": (
        (0.567, 0.433, 0.000),
        (0.558, 0.442, 0.000),
        (0.000, 0.242, 0.758),
    ),
}


def _project(rgb: tuple[float, float, float], kind: str) -> tuple[float, float, float]:
    matrix = _DICHROMAT[kind]
    return tuple(
        sum(row[q] * rgb[q] for q in range(3)) for row in matrix
    )


def separation(first: str, second: str, vision: str = "normal") -> float:
    """OKLab distance in hundredths between two hex colors under one vision."""
    a = _hex_to_linear(first)
    b = _hex_to_linear(second)
    if vision != "normal":
        a = _project(a, vision)
        b = _project(b, vision)
    lab_a = _linear_to_oklab(a)
    lab_b = _linear_to_oklab(b)
    return 100.0 * math.dist(lab_a, lab_b)


def validate_palette(colors: list[str] | None = None) -> list[str]:
    """Check adjacent separations and raise on a normal-vision failure.

    Every figure script calls this before it draws, so a re-stepped ramp that
    lost its separation stops the run rather than producing a figure whose
    series are distinguishable to the author and to nobody else. The returned
    report is for a reader who wants the measured numbers.
    """
    palette = list(colors if colors is not None else CATEGORICAL)
    lines: list[str] = []
    failures: list[str] = []
    for index in range(len(palette) - 1):
        first, second = palette[index], palette[index + 1]
        normal = separation(first, second)
        deuter = separation(first, second, "deuteranopia")
        protan = separation(first, second, "protanopia")
        lines.append(
            f"{first} vs {second}: normal={normal:.1f} "
            f"deuteranopia={deuter:.1f} protanopia={protan:.1f}"
        )
        if normal < 15.0:
            failures.append(
                f"{first} vs {second} separate by {normal:.1f} in OKLab "
                "hundredths under normal vision, below the 15 floor"
            )
        if min(deuter, protan) < 8.0:
            lines[-1] += "  [below the 8 dichromatic target]"
    if failures:
        raise ValueError(
            "categorical palette rejected:\n  " + "\n  ".join(failures)
        )
    return lines
