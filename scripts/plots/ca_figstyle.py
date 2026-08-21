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

# The neutral register is shared with every other figure in this directory, so
# it is re-exported rather than redefined: an annotation drawn next to a
# mark_better arrow takes the arrow's own color, and a CA panel weighs the same
# as a CPU one.
C_BETTER = figstyle.C_BETTER
C_INK = figstyle.C_INK
C_GRID = figstyle.C_GRID
C_MUTED = figstyle.C_MUTED

# One hue per entity, assigned once. The ramp is stepped in OKLab hue so the
# separations below are a property of the construction rather than a coincidence.
C_STOPPED = "#8A8579"

# The validated ramps and the separation machinery are shared, so a CA hue and a
# CPU hue are drawn from one palette and checked by one rule.
CATEGORICAL = figstyle.CATEGORICAL
SEQUENTIAL = figstyle.SEQUENTIAL
DIVERGING_LOW = figstyle.DIVERGING_LOW
DIVERGING_MID = figstyle.DIVERGING_MID
DIVERGING_HIGH = figstyle.DIVERGING_HIGH
separation = figstyle.separation
validate_palette = figstyle.validate_palette

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


def mark_stopped(ax, x, y, label: str = "stopped") -> None:
    """Draw a stopped or diagnostic arm so it is present and not a result."""
    ax.plot(
        x, y, linestyle=(0, (1, 2)), marker="x", ms=5, lw=1.1,
        color=C_STOPPED, label=label, zorder=3,
    )


if __name__ == "__main__":
    for line in validate_palette():
        print(line)
