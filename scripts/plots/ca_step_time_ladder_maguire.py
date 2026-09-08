"""Time per step on maguire as the participant topology grows, with and without s-step.

The maguire companion to scripts/plots/ca_step_time_ladder.py, whose bar
geometry, acceptance gates and ratio labels this reuses so the two ladders can
be read side by side. What differs is the machine and the reach: synge offers
two GPUs over two nodes, maguire's gpu01 offers eight over one and sixteen over
two, and every intra-node pair is NV18 rather than SYS.

The grid is n=97, not synge's 61, and the reason is arithmetic rather than
taste. The halo exchange sends the neighbour's last `depth` owned planes, and
the thinnest slab holds floor(n / participants); at n=61 over sixteen
participants that is three planes against the as-measured arm's four-deep halo,
which reads before the start of its own buffer. n=97 leaves six planes, has a
referee for both options, and puts 57,000 rows on each of sixteen GPUs instead
of 14,000, so the widest rung is not read off a launch-bound run.

The two ladders are therefore not comparable point for point. What carries
across is the ratio in each group, which is what the figure is about.

Source: data/ca-integrator-strong-h200-m39, produced by
scripts/regime/ca_strong_scaling.sh with MACHINE=h200 N=97.

  uv run scripts/plots/ca_step_time_ladder_maguire.py
"""
# /// script
# dependencies = ["matplotlib", "numpy"]
# ///
from __future__ import annotations

import ca_figlib as lib
import ca_step_time_ladder as synge

FIGURE = lib.Figure("step_time_ladder_maguire")

STRONG = lib.DATA / "ca-integrator-strong-h200-m39"

# One node reaches 1, 2, 4 and 8; two nodes add 16. The one-node and two-node
# arrangements at the same participant count differ in launch structure as well
# as in node split, so they stay separate bars rather than one series.
ONE_NODE = [
    ((1, 1), "1 GPU"),
    ((2, 1), "2 GPU\n1 node"),
    ((4, 1), "4 GPU\n1 node"),
    ((8, 1), "8 GPU\n1 node"),
]
TWO_NODE = [
    ((2, 2), "2 GPU\n2 nodes"),
    ((4, 2), "4 GPU\n2 nodes"),
    ((8, 2), "8 GPU\n2 nodes"),
    ((16, 2), "16 GPU\n2 nodes"),
]


def draw() -> str | None:
    if not STRONG.is_dir():
        return FIGURE.blocked(
            f"{STRONG.relative_to(lib.ROOT)} is absent - run\n"
            "  MACHINE=h200 N=97 scripts/regime/ca_strong_scaling.sh")

    # Draw whichever family the sweep actually holds. A one-node job is a
    # complete experiment on its own, and waiting for a second node before
    # drawing anything would withhold the eight-GPU result it already has.
    runs = lib.load_runs(STRONG)
    present = {(run.world_gpus, run.nodes) for run in runs}
    arrangements = [
        entry for entry in ONE_NODE + TWO_NODE if entry[0] in present
    ]
    if not arrangements:
        return FIGURE.blocked(
            f"{STRONG.relative_to(lib.ROOT)} holds no recognised arrangement; "
            f"found {sorted(present)}")
    return synge.draw(source=STRONG, arrangements=arrangements, figure=FIGURE)


if __name__ == "__main__":
    raise SystemExit(lib.run(FIGURE, draw))
