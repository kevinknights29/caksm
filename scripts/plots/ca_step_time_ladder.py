"""Time per step as the participant topology grows, with and without s-step.

The strong-scaling ladder as a paired comparison rather than a pair of curves.
One group per arrangement, one bar per block width: s = 1 is the control that
reduces once per step, s = 4 is the treatment that reduces once per four. The
distance between the two bars in a group is what communication-avoidance buys at
that arrangement, and it is the thing the figure exists to show.

Only the exact-depth arm is drawn, and only one option. The arm because it is the
one that prices the halo at the depth the recurrence actually reaches, so the two
widths are compared at equal numerical work; the option because the two differ by
about a percent here and a second panel would spend half the figure saying so.

Source: data/ca-integrator-strong-m39, produced by
scripts/regime/ca_strong_scaling.sh.

  uv run scripts/plots/ca_step_time_ladder.py
"""
# /// script
# dependencies = ["matplotlib", "numpy"]
# ///
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from figstyle import mark_better
import ca_figstyle as ca
import ca_figlib as lib

FIGURE = lib.Figure("step_time_ladder")

OPTION = "basket"
ARM = "exact-depth"
WIDTHS = (1, 4)
WIDTH_LABEL = {1: "$s = 1$, one reduction per step",
               4: "$s = 4$, one reduction per four steps"}
BAR = 0.36


def draw(
    source=lib.STRONG,
    arrangements=lib.STRONG_TOPOLOGIES,
    figure=FIGURE,
) -> str | None:
    """Draw the paired s=1/s=4 ladder for one machine's strong-scaling sweep.

    The defaults are synge's, so calling it bare reproduces the published
    figure. A second machine passes its own sweep directory and its own
    arrangements, which is what keeps the two ladders separate rather than
    merging two interconnects into one axis.

    Args:
        source: Directory of strong-scaling transcripts.
        arrangements: ((world_gpus, nodes), label) pairs, in bar order.
        figure: The Figure this writes to.
    """
    all_runs = lib.load_runs(source)
    if not all_runs:
        return figure.blocked(
            f"{source.relative_to(lib.ROOT)} is absent; run "
            "scripts/regime/ca_strong_scaling.sh")

    # The same gate the four-panel ladder applies, narrowed to the one arm and
    # option drawn here. A group missing either width would show one bar and
    # read as a measurement rather than as a hole.
    topologies = [topology for topology, _ in arrangements]
    expected = {(width, topology) for width in WIDTHS for topology in topologies}
    observed: dict[tuple, list[lib.Run]] = {}
    for run in all_runs:
        if run.option != OPTION or run.arm != ARM:
            continue
        observed.setdefault(
            (run.width, (run.world_gpus, run.nodes)), []).append(run)
    incomplete = [
        key for key in expected
        if len(observed.get(key, [])) != 1
        or observed[key][0].cycle_ms is None
        or not observed[key][0].recordable
        or observed[key][0].repeats < 7
        or observed[key][0].run_status != "passed"
    ]
    if incomplete:
        return figure.blocked(
            f"{len(incomplete)} of {len(expected)} {OPTION}/{ARM} points are "
            "missing, duplicated, contended, or lack a passed seven-repeat timing")

    fig, ax = plt.subplots(figsize=(7.6, 4.2), constrained_layout=True)
    positions = np.arange(len(topologies), dtype=float)
    heights: dict[int, list[float]] = {}

    for index, width in enumerate(WIDTHS):
        offset = (index - (len(WIDTHS) - 1) / 2.0) * BAR
        values = [observed[(width, t)][0].cycle_ms for t in topologies]
        heights[width] = values
        ax.bar(positions + offset, values, BAR, color=ca.WIDTH_COLOR[width],
               edgecolor="white", lw=0.6, zorder=3, label=WIDTH_LABEL[width])
        for x, value in zip(positions + offset, values):
            lib.label_value(ax, x, value, f"{value:.2f}", dy=4.0, size=8.0,
                            color=ca.C_INK)

    # The ratio over each pair. It is the quantity the figure is about, and
    # leaving a reader to divide two labels by eye is leaving it unsaid.
    ratios = [heights[WIDTHS[0]][i] / heights[WIDTHS[1]][i]
              for i in range(len(topologies))]
    ceiling = max(max(v) for v in heights.values())
    for x, ratio, pair in zip(positions, ratios,
                              zip(*(heights[w] for w in WIDTHS))):
        lib.label_value(ax, x, max(pair) + ceiling * 0.11,
                        f"{ratio:.2f}$\\times$", dy=0.0, size=9.5,
                        color=ca.WIDTH_COLOR[WIDTHS[1]], weight="semibold")

    ax.set_xticks(positions)
    ax.set_xticklabels([label for _, label in arrangements],
                       fontsize=9.0)
    ax.set_ylabel("cycle time (ms/step)")
    ax.set_ylim(0.0, ceiling * 1.26)
    lib.style_axes(ax, grid_axis="y")
    ax.tick_params(axis="x", length=0)
    # Time per step: down is the win. The bars fill the lower half, so the arrow
    # takes the one corner they never reach.
    mark_better(ax, "down", loc="upper left")
    lib.legend(ax, loc="upper center", ncol=2)

    rows = [
        {"option": OPTION, "arm": ARM, "gpus": t[0], "nodes": t[1],
         "cycle_ms_s1": heights[1][i], "cycle_ms_s4": heights[4][i],
         "speedup": ratios[i]}
        for i, t in enumerate(topologies)
    ]
    figure.write(fig, rows)

    print(f"  {OPTION}, {ARM} arm")
    for (gpus, nodes), r, s1, s4 in zip(topologies, ratios,
                                        heights[1], heights[4]):
        print(f"    {gpus} GPU / {nodes} node(s): {s1:.2f} -> {s4:.2f} ms "
              f"= {r:.2f}x")
    print(f"  the ladder's headline is the widest arrangement: {ratios[-1]:.2f}x")
    print("  s = 1 climbs with every rung; s = 4 barely moves, which is the "
          "mechanism:\n  the reduction count, not the reduction cost, is what "
          "the treatment removes.")
    return None


if __name__ == "__main__":
    raise SystemExit(lib.run(FIGURE, draw))
