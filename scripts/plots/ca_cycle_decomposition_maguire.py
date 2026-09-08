"""Where the cycle goes on maguire: local work, reductions, halos, host sync.

The maguire companion to scripts/plots/ca_cycle_decomposition.py, whose panel
layout, term pricing and acceptance gates this reuses so the two figures can be
read side by side. What differs is the interconnect: synge reduces over SYS,
PCIe plus a cross-socket UPI hop, while every intra-node pair here is NV18. The
same participant counts on the two machines is the contrast the figure exists
to show, and it needs no larger grid to make.

The rungs are deliberately the same (participants, grid) pairs synge measured.
Weak scaling would want n around 122 at eight participants, and the placement
table puts the Krylov dimension there at 41, above the compile-time ceiling of
39. Eight participants therefore joins this figure only after the projected
exponential moves off static shared memory, and until then the honest sweep is
the two rungs synge also has.

Blocked until two jobs have run on maguire, both of which write machine-keyed
directories rather than overwriting synge's:

  MACHINE=h200 scripts/regime/ca_exact_depth_weak.sh
  MACHINE=h200 scripts/regime/calibrate_ca_participants.sh

  uv run scripts/plots/ca_cycle_decomposition_maguire.py
"""
# /// script
# dependencies = ["matplotlib", "numpy"]
# ///
from __future__ import annotations

import ca_figlib as lib
import ca_cycle_decomposition as synge

FIGURE = lib.Figure("cycle_decomposition_maguire")

EXACT = lib.DATA / "ca-integrator-exact-depth-h200-m39"
PARTICIPANTS = lib.DATA / "ca-participant-calibration-h200"

# One node holds every participant these rungs need, so the distributed runs
# report one node rather than synge's two. That is a difference in launch
# structure as well as in transport, and the gate has to expect it.
RUNGS = ((2, 77), (4, 97))
NODES_EXPECTED = 1


def draw() -> str | None:
    if not EXACT.is_dir():
        return FIGURE.blocked(
            f"{EXACT.relative_to(lib.ROOT)} is absent - run\n"
            "  MACHINE=h200 scripts/regime/ca_exact_depth_weak.sh")
    if not PARTICIPANTS.is_dir():
        return FIGURE.blocked(
            f"{PARTICIPANTS.relative_to(lib.ROOT)} is absent - run\n"
            "  MACHINE=h200 scripts/regime/calibrate_ca_participants.sh\n"
            "The reduction and halo terms are priced from it, and a missing\n"
            "calibration would report the whole cycle as local work.")
    return synge.draw(
        source=EXACT,
        rungs=RUNGS,
        nodes_expected=NODES_EXPECTED,
        figure=FIGURE,
        participants_dir=PARTICIPANTS,
    )


if __name__ == "__main__":
    raise SystemExit(lib.run(FIGURE, draw))
