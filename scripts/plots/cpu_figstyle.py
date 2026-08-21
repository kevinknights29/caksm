"""Color furniture for the CPU regime and scaling figures.

The counterpart to ca_figstyle.py, drawing from the same validated ramps in
figstyle.py. Everything here is CPU vocabulary: kernel, placement, layout,
memory tier, tiling variant and operator family.

Assigning these in one place is what the earlier per-script constants could not
do. Seven scripts each defined their own C_* block, and between them one hue
stood for the SpMV kernel in one figure, a regression fit in another and a
measurement in a third, while none of the hues came from a palette anything had
checked. A reader moving between figures had to relearn the code each time.

Run this module directly to print the measured separations:

  uv run scripts/plots/cpu_figstyle.py
"""
# /// script
# dependencies = []
# ///

from __future__ import annotations

import figstyle

# The neutral register and the validated ramps, shared with every other figure.
C_BETTER = figstyle.C_BETTER
C_INK = figstyle.C_INK
C_GRID = figstyle.C_GRID
C_MUTED = figstyle.C_MUTED
CATEGORICAL = figstyle.CATEGORICAL
SEQUENTIAL = figstyle.SEQUENTIAL
DIVERGING_LOW = figstyle.DIVERGING_LOW
DIVERGING_HIGH = figstyle.DIVERGING_HIGH
separation = figstyle.separation
validate_palette = figstyle.validate_palette

INDIGO, PINE, RUST, PLUM, STEEL, OCHRE = CATEGORICAL

# A reference the data is measured against rather than a series in its own
# right: an ideal-scaling guide, a hardware ceiling, a modeled projection. Kept
# neutral so it never competes with a measurement for attention.
C_GUIDE = C_MUTED
C_CEILING = DIVERGING_LOW

# A curve that is modeled rather than measured. Only the roofline projection
# uses it, and it is drawn lighter than any measurement on the same axes so the
# distinction survives being read quickly.
C_MODELED = "#A8A49B"

# The two kernels the solver spends its time in, plus the third it barely does.
# Hue is the kernel because that is what a reader is comparing; the placement
# arm takes the dash pattern instead.
KERNEL_COLOR = {
    "SpMV": RUST,
    "Gram-Schmidt": INDIGO,
    "dense expm": C_MUTED,
    "total": PLUM,
}
PLACEMENT_LINESTYLE = {"naive": (0, (4, 2)), "first-touch": "-"}
# Where the placement arms are the subject rather than a secondary channel, as
# in the A/B contrast, they need hues of their own: two curves that coincide
# only read as coincident if each is separately identifiable.
PLACEMENT_COLOR = {"naive": OCHRE, "first-touch": STEEL}
PLACEMENT_MARKER = {"naive": "o", "first-touch": "s"}

# Thread-to-data layouts, ordered by how much locality they preserve.
LAYOUT_COLOR = {"block": PINE, "cyclic": OCHRE, "rotate": RUST}

# Memory tiers, warm to cool as they get closer to the core.
TIER_COLOR = {"DRAM": RUST, "L3": PINE, "L2": STEEL}

# What the tiling experiment compares. The baseline is the control, so it takes
# the neutral; the debug arm is marked as a diagnostic wherever it is drawn.
VARIANT_COLOR = {
    "baseline": C_MUTED,
    "tiled": PINE,
    "no-halo": OCHRE,
}

# Operator families in the non-normality study, from the case where the
# prediction is exact to the one the thesis has to transfer to.
OPERATOR_COLOR = {
    "normal": INDIGO,
    "constant-coeff": PINE,
    "variable-coeff": RUST,
    "black-scholes": OCHRE,
}

# Quantities that recur across the regime figures.
C_PREDICTED = INDIGO
C_MEASURED = RUST
C_LIMIT = DIVERGING_LOW
C_KAPPA = INDIGO
C_REUSE = PINE

# One marker per kernel and per layout, so a black-and-white print still
# separates the series a hue would have.
KERNEL_MARKER = {
    "SpMV": "o", "Gram-Schmidt": "s", "dense expm": "^", "total": "D",
}
LAYOUT_MARKER = {"block": "o", "cyclic": "s", "rotate": "^"}


if __name__ == "__main__":
    for line in validate_palette():
        print(line)
