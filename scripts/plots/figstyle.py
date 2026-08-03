"""
Shared figure furniture for the plot scripts in this directory.
"""

C_BETTER = "#5F5E5A"

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
