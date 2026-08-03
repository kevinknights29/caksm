"""Strong scaling: cycle time against topology at a fixed global problem size.

Where communication avoiding overtakes, and by how much. The gap between the s=1
control and the s=4 arm widens with the participant count, so participants run
along the horizontal axis and the block width takes the hue. Columns are the two
correction arms; rows are the two options, which differ by under a tenth of a
millisecond at every rung.

Source: data/ca-integrator-strong, produced by scripts/regime/ca_strong_scaling.sh.

  uv run scripts/plots/ca_strong_scaling.py
"""
# /// script
# dependencies = ["matplotlib", "numpy"]
# ///
from __future__ import annotations

import matplotlib.pyplot as plt

from figstyle import mark_better
import ca_figstyle as ca
import ca_figlib as lib

FIGURE = lib.Figure("strong_scaling")

# What the two correction arms are, in words, for a reader who has not read the
# correction spec. Both are the same solver on the same devices; they differ in
# how deep the block build reaches.
ARM_TITLE = {
    "as-measured": "As measured",
    "exact-depth": "Exact depth",
}
ARM_GLOSS = {
    "as-measured": "the original build: one unused power, halos one plane deep "
                   "too many",
    "exact-depth": "the corrected build: every power consumed, halos no deeper "
                   "than needed",
}

WIDTH_LABEL = {
    1: "s = 1 control: one reduction per step",
    4: "s = 4 avoiding: one reduction per four steps",
}


def draw() -> str | None:
    all_runs = lib.load_runs(lib.STRONG)
    if not all_runs:
        return FIGURE.blocked(
            "data/ca-integrator-strong is absent; the report's strong-scaling "
            "transcripts were never downloaded from Synge. Run "
            "scripts/regime/ca_strong_scaling.sh, or record section 3 of the "
            "final report as unreproduced.")

    required_arms = ("as-measured", "exact-depth")
    required_topologies = {topology for topology, _ in lib.STRONG_TOPOLOGIES}
    expected = {
        (arm, option, width, topology)
        for arm in required_arms
        for option in ("basket", "rainbow")
        for width in (1, 4)
        for topology in required_topologies
    }
    observed: dict[tuple, list[lib.Run]] = {}
    for run in all_runs:
        key = (
            run.arm, run.option, run.width,
            (run.world_gpus, run.nodes),
        )
        observed.setdefault(key, []).append(run)
    incomplete = [
        key for key in expected
        if len(observed.get(key, [])) != 1
        or observed[key][0].cycle_ms is None
        or not observed[key][0].recordable
        or observed[key][0].repeats < 7
        or observed[key][0].run_status != "passed"
    ]
    if incomplete:
        return FIGURE.blocked(
            f"the canonical ladder is incomplete: {len(incomplete)} of "
            f"{len(expected)} required arm/option/width/topology points are "
            "missing, duplicated, contended, unclassified, or lack a passed "
            "seven-repeat timing")
    runs = [observed[key][0] for key in sorted(expected)]

    options = ("rainbow", "basket")
    arms = list(required_arms)
    # One vertical scale across all four panels. The exact-depth arm is lower
    # everywhere, and a per-panel scale would rescale that away.
    fig, axes = plt.subplots(
        len(options), len(arms), figsize=(11.4, 8.2),
        constrained_layout=True, squeeze=False, sharex=True, sharey=True)
    rows = []
    seen_stopped: set = set()
    ticks = [
        lib.STRONG_TOPOLOGY_X[topology]
        for topology, _ in lib.STRONG_TOPOLOGIES
    ]
    speedups: dict[tuple[int, int], list[float]] = {}

    for row, option in enumerate(options):
        for column, arm in enumerate(arms):
            ax = axes[row][column]
            by_width: dict[int, list[lib.Run]] = {}
            for width in (1, 4):
                points = sorted(
                    (r for r in runs
                     if r.arm == arm and r.option == option
                     and r.width == width),
                    key=lambda r: lib.STRONG_TOPOLOGY_X[
                        (r.world_gpus, r.nodes)])
                by_width[width] = points
                lib.draw_topology_series(
                    ax, points, ca.WIDTH_COLOR[width],
                    ca.WIDTH_MARKER[width], "-",
                    WIDTH_LABEL[width], seen_stopped)
                for point in points:
                    rows.append({
                        "arm": arm, "option": option, "s": width,
                        "gpus": point.world_gpus, "nodes": point.nodes,
                        "topology":
                            f"{point.world_gpus}gpu_{point.nodes}node",
                        "cycle_ms_per_step": point.cycle_ms,
                        "cycle_min_ms_per_step": point.cycle_min_ms,
                        "cycle_max_ms_per_step": point.cycle_max_ms,
                        "repeats": point.repeats,
                        "stopped": int(point.stopped),
                    })

            # The saving is the whole claim, so it is shaded and then named.
            # Filling between the two widths turns "which line is lower" into an
            # area the eye reads before it decodes a legend, and the ratio
            # printed in the band is the number the reader would otherwise have
            # to estimate off the axis.
            narrow, wide = by_width[1], by_width[4]
            if len(narrow) == len(wide) == len(ticks):
                ax.fill_between(
                    ticks,
                    [r.cycle_ms for r in narrow],
                    [r.cycle_ms for r in wide],
                    color=ca.WIDTH_COLOR[4], alpha=0.10, lw=0, zorder=1)
                # Where the band is wide the ratio sits inside it. Where the
                # two arms nearly touch, as they do on one GPU, the middle of
                # the band is on both lines, so the label drops below the lower
                # of the two instead.
                span = max(r.cycle_ms for r in narrow + wide) - min(
                    r.cycle_ms for r in narrow + wide)
                for index, (x, control, avoided) in enumerate(
                        zip(ticks, narrow, wide)):
                    speedup = control.cycle_ms / avoided.cycle_ms
                    speedups.setdefault(
                        (control.world_gpus, control.nodes), []
                    ).append(speedup)
                    thin = abs(control.cycle_ms - avoided.cycle_ms) < 0.12 * span
                    anchor = (min(control.cycle_ms, avoided.cycle_ms) if thin
                              else (control.cycle_ms + avoided.cycle_ms) / 2.0)
                    # The rightmost rung has no room to its right inside the
                    # axes, so its label turns back into the panel.
                    last = index == len(ticks) - 1
                    lib.label_value(
                        ax, x, anchor, f"{speedup:.2f}×",
                        dy=-13.0 if thin else -3.0,
                        dx=0.0 if thin else (-8.0 if last else 8.0),
                        ha="center" if thin else ("right" if last else "left"),
                        color=ca.WIDTH_COLOR[4], size=8.2,
                        weight="semibold")
                    for record in rows:
                        if (record["arm"] == arm
                                and record["option"] == option
                                and record["gpus"] == control.world_gpus
                                and record["nodes"] == control.nodes):
                            record["s1_over_s4_speedup"] = speedup

            # The two arms are named once, on themselves, in the panel the
            # reader meets first. Naming them where they are furthest apart
            # keeps the labels off each other and off the ratio in the band.
            if row == 0 and column == 0 and len(narrow) == len(wide) == 4:
                lib.label_series(
                    ax, ticks[2], narrow[2].cycle_ms, WIDTH_LABEL[1],
                    ca.WIDTH_COLOR[1], dy=10.0, ha="center")
                # Named at the rung where the s=4 curve is at its lowest and
                # the band beneath it is empty. At the two-node rung the curve
                # climbs away from its own label.
                lib.label_series(
                    ax, ticks[1], wide[1].cycle_ms, WIDTH_LABEL[4],
                    ca.WIDTH_COLOR[4], dy=-16.0, ha="center")

            ax.set_xticks(ticks)
            ax.set_xticklabels(
                [label for _, label in lib.STRONG_TOPOLOGIES], fontsize=8.0)
            lib.style_axes(ax)
            if row == 0:
                lib.panel_title(ax, ARM_TITLE[arm], ARM_GLOSS[arm])
            if row == len(options) - 1:
                ax.set_xlabel("participating V100s")
            if column == 0:
                ax.set_ylabel(
                    f"{option.capitalize()} option\ncycle time (ms/step)")
                # Upper left: the only corner no curve reaches, since every arm
                # starts low on one GPU and the control climbs to the right.
                mark_better(ax, "down", loc="upper left")

    # The claim that the two rows are replicates is measured, not asserted:
    # the largest cycle-time gap between the options over every matched
    # arm/width/topology point goes on the figure as the number it is.
    matched: dict[tuple, list[float]] = {}
    for record in rows:
        matched.setdefault(
            (record["arm"], record["s"], record["gpus"], record["nodes"]), []
        ).append(record["cycle_ms_per_step"])
    option_gap = max(
        max(values) - min(values)
        for values in matched.values() if len(values) > 1
    )

    def span(topology: tuple[int, int]) -> str:
        values = speedups.get(topology, [])
        if not values:
            return "n/a"
        return (f"{min(values):.2f}x" if max(values) - min(values) < 0.005
                else f"{min(values):.2f}-{max(values):.2f}x")

    lib.title(
        fig,
        "Communication avoiding pays only once the job spans participants")
    print(f"  one GPU {span((1, 1))}, four GPUs {span((4, 2))}; "
          f"options agree to {option_gap:.2f} ms/step")

    FIGURE.write(fig, rows)
    return None


if __name__ == "__main__":
    raise SystemExit(lib.run(FIGURE, draw))
