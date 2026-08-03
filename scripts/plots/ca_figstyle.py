"""Color furniture for the CA integrator figures, with the palette validated.

Separate from figstyle.py, which every plot script here imports and which holds
only what they all share. Everything in this module is CA vocabulary: option,
width, arm, basis, orthogonalization, method and topology.

Separation is measured rather than eyeballed. Adjacent assigned hues must clear
15 in OKLab hundredths under normal vision, which validate_palette treats as a
hard failure; the target under simulated dichromacy is 8.

Run this module directly to print the measured separations:

  uv run scripts/plots/ca_figstyle.py
"""
# /// script
# dependencies = []
# ///

from __future__ import annotations

import math

import figstyle

# Re-exported rather than redefined: an annotation drawn next to a mark_better
# arrow takes the arrow's own color, so the two read as one layer of furniture.
C_BETTER = figstyle.C_BETTER

# One hue per entity, assigned once. The ramp is stepped in OKLab hue so the
# separations below are a property of the construction rather than a coincidence.
C_INK = "#22201C"
C_GRID = "#CFCBC2"
C_STOPPED = "#8A8579"

# Subordinate type: the provenance note under a figure, a unit suffix, a
# direct label that names a series the reader can already see. Lighter than
# C_INK so it reads as apparatus rather than as a second finding, and darker
# than C_GRID so it survives print.
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

# Fixed entity assignments. Every figure reads these; none invents its own.
OPTION_COLOR = {"basket": CATEGORICAL[0], "rainbow": CATEGORICAL[1]}
WIDTH_COLOR = {1: CATEGORICAL[2], 4: CATEGORICAL[4]}
# Ochre against steel for the correction arms. The plum/pine pair this
# replaced separated by only 7.9 in OKLab hundredths under protanopia, below
# the 8 target validate_palette reports against; ochre against steel clears 20
# under normal vision and 23 under protanopia.
ARM_COLOR = {"as-measured": CATEGORICAL[5], "exact-depth": CATEGORICAL[4]}
BASIS_COLOR = {
    "monomial": CATEGORICAL[0],
    "newton": CATEGORICAL[5],
    "chebyshev": CATEGORICAL[2],
}
ORTH_COLOR = {
    "cholqr2": CATEGORICAL[0],
    "tsqr": CATEGORICAL[2],
    "bgs2": CATEGORICAL[4],
}
METHOD_COLOR = {
    "fixed-compatibility": CATEGORICAL[2],
    "scaled-augmentation": CATEGORICAL[1],
}
TOPOLOGY_COLOR = {1: CATEGORICAL[5], 2: CATEGORICAL[4], 4: CATEGORICAL[0]}

# The cycle-time decomposition. Sequential magnitude, light to dark, because the
# terms are one measure split by where the time went.
TERM_COLOR = {
    "local work": SEQUENTIAL[1],
    "reductions": SEQUENTIAL[2],
    "halos": SEQUENTIAL[3],
    "host synchronization": SEQUENTIAL[4],
}

# Width tells the two arms apart when both are drawn in one panel; hue is spent
# on the entity the panel is about.
WIDTH_MARKER = {1: "o", 4: "s"}
ARM_LINESTYLE = {"as-measured": "--", "exact-depth": "-"}


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


def mark_stopped(ax, x, y, label: str = "stopped") -> None:
    """Draw a stopped or diagnostic arm so it is present and not a result."""
    ax.plot(
        x, y, linestyle=(0, (1, 2)), marker="x", ms=5, lw=1.1,
        color=C_STOPPED, label=label, zorder=3,
    )


if __name__ == "__main__":
    for line in validate_palette():
        print(line)
