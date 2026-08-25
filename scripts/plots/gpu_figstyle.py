"""Color furniture for the GPU regime figures.

The third family module, beside ca_figstyle.py and cpu_figstyle.py and drawing
from the same validated ramps in figstyle.py. Everything here is GPU vocabulary:
participant/node topology, reduction rung, and problem-decomposition policy.

Kept out of cpu_figstyle.py because a topology is not CPU vocabulary and would
have to borrow a hue already meaning something else there. The rule is the one
that module states: a hue means one thing across the whole set, so the thing it
means gets named in exactly one place.

Run this module directly to print the measured separations:

  uv run scripts/plots/gpu_figstyle.py
"""
# /// script
# dependencies = []
# ///

from __future__ import annotations

import figstyle

C_BETTER = figstyle.C_BETTER
C_INK = figstyle.C_INK
C_GRID = figstyle.C_GRID
C_MUTED = figstyle.C_MUTED
CATEGORICAL = figstyle.CATEGORICAL
SEQUENTIAL = figstyle.SEQUENTIAL
separation = figstyle.separation
validate_palette = figstyle.validate_palette

INDIGO, PINE, RUST, PLUM, STEEL, OCHRE = CATEGORICAL

C_GUIDE = C_MUTED
C_LIMIT = figstyle.DIVERGING_LOW   # the two unit thresholds: a boundary, not a series

# The four arrangements synge supports, keyed as include/gpu_topology.hpp keys them.
# Ordered by how far a reduction has to reach, and colored cool to warm along that
# order so the ordering survives without a reader consulting the key. The sequence
# clears the adjacent-separation floor at 19.0 OKLab hundredths, well above the 15
# the validator enforces, which a straight cool-to-warm ramp through steel and
# ochre did not.
TOPOLOGY_ORDER = ("1gpu-1node", "2gpu-1node", "2gpu-2node", "4gpu-2node")
TOPOLOGY_COLOR = {
    "1gpu-1node": INDIGO,
    "2gpu-1node": PINE,
    "2gpu-2node": PLUM,
    "4gpu-2node": RUST,
}
# A second channel for the same distinction. The pine/plum pair separates by 7.9
# under protanopia, just under the 8 target, and hue alone would also be lost in a
# black-and-white print of the poster. Dash pattern lengthens with reach, so the
# ordering is carried twice.
TOPOLOGY_LINESTYLE = {
    "1gpu-1node": "-",
    "2gpu-1node": (0, (5, 1.4)),
    "2gpu-2node": (0, (3, 1.4)),
    "4gpu-2node": (0, (1.4, 1.4)),
}
TOPOLOGY_LABEL = {
    "1gpu-1node": "1 GPU, 1 node",
    "2gpu-1node": "2 GPUs, 1 node",
    "2gpu-2node": "2 GPUs, 2 nodes",
    "4gpu-2node": "4 GPUs, 2 nodes",
}

# Marker is the problem policy, so a reader can tell a refinement sweep from a
# participant sweep without tracing which line a point sits on. Hue is already
# spent on topology, which is the comparison the figure is for.
POLICY_MARKER = {"fixed-global": "o", "fixed-local": "s", "cpu-placement": "D"}
POLICY_LABEL = {
    "fixed-global": "fixed global problem",
    "fixed-local": "fixed local problem",
}

# The reduction rungs, for annotating which link a topology's collective crossed.
TIER_LABEL = {
    "grid": "grid",
    "device-p2p": "device P2P",
    "node": "node",
}

# The ladder's verdict per rung: whether one reduction there already costs more
# than tau*, the threshold at which R_v * R_h reaches 1. Not a series and not a
# category, so neither the categorical ramp nor a topology hue is right: open is
# the one quantity the chart is about, closed is everything the chart rules out.
CORNER_COLOR = {True: STEEL, False: "#B9B5AC"}
CORNER_LABEL = {True: "corner open", False: "corner closed"}

# The four corners of the map. Named once here so all three panels label them
# identically and a reader learns the plane on the first panel only.
REGIME_LABEL = {
    "upper-right": "UPPER-RIGHT\nboth mechanisms",
    "upper-left": "UPPER-LEFT\nvertical only\n(matrix-powers)",
    "lower-left": "LOWER-LEFT\nneither",
    "lower-right": "LOWER-RIGHT\nhorizontal only\n(s-step)",
}
# The contested corner carries a tint; the other three are left clear so the
# trajectories, not the ground, are what the eye lands on.
C_CORNER = SEQUENTIAL[0]


if __name__ == "__main__":
    for line in validate_palette([TOPOLOGY_COLOR[k] for k in TOPOLOGY_ORDER]):
        print(line)
