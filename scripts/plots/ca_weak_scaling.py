"""Weak scaling: cycle time and efficiency at a fixed local volume.

Work per device is held constant while the global problem grows with the device
count, so every tick names its grid: n=77 on two GPUs and n=97 on four are the
same local volume. Dotted rules mark perfectly flat weak scaling. The second row
carries the finding, that the s=1 control loses efficiency faster than s=4 as
participants are added.

An arm that left a Krylov cycle unconverged is drawn hollow and carries no
efficiency.

Sources: data/ca-integrator-weak, and data/ca-integrator-strong for the
efficiency denominator.

  uv run scripts/plots/ca_weak_scaling.py
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

FIGURE = lib.Figure("weak_scaling")

WIDTH_LABEL = {1: "s = 1 control", 4: "s = 4 avoiding"}
BAR = 0.28


def draw() -> str | None:
    runs = [r for r in lib.load_runs(lib.WEAK)
            if r.cycle_ms is not None and r.world_gpus > 1]
    if not runs:
        return FIGURE.blocked("data/ca-integrator-weak holds no transcripts")

    # The horizontal axis is a ladder of participant counts, and each rung
    # carries the grid that keeps the local volume fixed. Naming the grid on the
    # tick is what stops the growing n from looking like an uncontrolled change.
    rungs = sorted({(r.world_gpus, r.n) for r in runs})
    positions = np.arange(len(rungs), dtype=float)
    tick_labels = [f"{gpus} GPUs\nn={n}" for gpus, n in rungs]

    options = ("rainbow", "basket")
    fig, axes = plt.subplots(
        2, len(options), figsize=(11.0, 8.0), constrained_layout=True,
        squeeze=False, sharex=True, sharey="row")
    rows = []

    # The efficiency denominator is a one-GPU run at the same local volume,
    # which is the n=61 point in the strong-scaling ladder. Without that
    # artifact the row would have to take its baseline from the report's prose.
    expected_baselines = {
        (option, width) for option in options for width in (1, 4)}
    observed_baselines: dict[tuple[str, int], list[lib.Run]] = {}
    for run in lib.load_runs(lib.STRONG):
        if (
            run.world_gpus == 1
            and run.n == 61
            and run.cycle_ms is not None
            and run.arm == "as-measured"
            and run.run_status == "passed"
            and run.recordable
            and run.repeats >= 7
            and not run.stopped
        ):
            observed_baselines.setdefault(
                (run.option, run.width), []).append(run)
    baseline = (
        {key: observed_baselines[key][0] for key in expected_baselines}
        if all(len(observed_baselines.get(key, [])) == 1
               for key in expected_baselines)
        else {}
    )

    stopped_seen = False
    efficiency_by_width: dict[int, list[float]] = {1: [], 4: []}

    for column, option in enumerate(options):
        time_ax, efficiency_ax = axes[0][column], axes[1][column]
        for index, width in enumerate((1, 4)):
            offset = (index - 0.5) * BAR
            color = ca.WIDTH_COLOR[width]
            selected = {
                (r.world_gpus, r.n): r for r in runs
                if r.option == option and r.width == width
            }

            for position, rung in zip(positions, rungs):
                run = selected.get(rung)
                if run is None:
                    continue
                x = position + offset
                # A stopped arm is present and visibly not a measurement: the
                # bar is hollow, so it cannot be read as a performance result,
                # and it is still drawn, so the figure does not silently claim
                # coverage the experiment does not have.
                time_ax.bar(
                    [x], [run.cycle_ms], BAR,
                    color="none" if run.stopped else color,
                    edgecolor=ca.C_STOPPED if run.stopped else "white",
                    hatch="////" if run.stopped else None,
                    lw=1.1 if run.stopped else 0.6, zorder=3)
                if not run.stopped:
                    time_ax.errorbar(
                        [x], [run.cycle_ms],
                        yerr=[[max(run.cycle_ms - (run.cycle_min_ms
                                                   or run.cycle_ms), 0.0)],
                              [max((run.cycle_max_ms or run.cycle_ms)
                                   - run.cycle_ms, 0.0)]],
                        color=ca.C_INK, lw=0, elinewidth=0.9, capsize=2.5,
                        zorder=5)
                lib.label_value(
                    time_ax, x, run.cycle_ms, f"{run.cycle_ms:.2f}",
                    dy=8.0, size=7.8,
                    color=ca.C_STOPPED if run.stopped else ca.C_INK)
                if run.stopped:
                    stopped_seen = True
                    lib.label_value(
                        time_ax, x, run.cycle_ms, "stopped",
                        dy=19.0, size=7.2, color=ca.C_STOPPED)

                reference = baseline.get((option, width))
                efficiency = (
                    100.0 * reference.cycle_ms / run.cycle_ms
                    if reference is not None and not run.stopped else None)
                if efficiency is not None:
                    efficiency_by_width[width].append(efficiency)
                    efficiency_ax.bar(
                        [x], [efficiency], BAR, color=color,
                        edgecolor="white", lw=0.6, zorder=3)
                    lib.label_value(
                        efficiency_ax, x, efficiency, f"{efficiency:.1f}%",
                        dy=6.0, size=7.8, color=ca.C_INK)
                rows.append({
                    "option": option, "s": width, "n": run.n,
                    "gpus": run.world_gpus, "nodes": run.nodes,
                    "cycle_ms_per_step": run.cycle_ms,
                    "cycle_min_ms_per_step": run.cycle_min_ms,
                    "cycle_max_ms_per_step": run.cycle_max_ms,
                    "repeats": run.repeats,
                    "avg_m": run.avg_m,
                    "unconverged": run.unconverged,
                    "stopped": int(run.stopped),
                    "weak_efficiency_percent": efficiency,
                })

            # Perfect weak scaling is a flat cycle time. Drawing the two-GPU
            # value across the panel turns "how far from ideal" into a distance
            # measured against a line rather than against a memory.
            first = selected.get(rungs[0])
            if first is not None and not first.stopped:
                time_ax.axhline(
                    first.cycle_ms, color=color, ls=":", lw=1.1,
                    alpha=0.8, zorder=2)

        for ax in (time_ax, efficiency_ax):
            ax.set_xticks(positions)
            ax.set_xticklabels(tick_labels, fontsize=8.5)
            # Margin outside the outermost bars, so the "better" arrow and the
            # series labels have somewhere to stand that is not on the data.
            ax.set_xlim(positions[0] - 0.85, positions[-1] + 0.55)
            lib.style_axes(ax, grid_axis="y")
        # Headroom for the value labels, and for the second line a stopped bar
        # carries above its value.
        time_ax.set_ylim(
            0.0, max(r.cycle_ms for r in runs) * 1.22)
        lib.panel_title(time_ax, f"{option.capitalize()} option")
        if baseline:
            efficiency_ax.axhline(
                100.0, color=ca.C_GRID, ls="-", lw=1.2, zorder=2)
            efficiency_ax.set_ylim(0.0, 120.0)
            if column == 0:
                lib.label_value(
                    efficiency_ax, positions[0] - 0.45, 100.0,
                    "perfect weak scaling", dy=4.0, ha="left", size=7.8)
        else:
            efficiency_ax.text(
                0.5, 0.5,
                "blocked on data/ca-integrator-strong\n"
                "(the fixed-local-volume one-GPU baseline)",
                ha="center", va="center", fontsize=9, color=ca.C_MUTED,
                transform=efficiency_ax.transAxes)
        efficiency_ax.set_xlabel("participating V100s at fixed local volume")
        if column == 0:
            time_ax.set_ylabel("cycle time (ms/step)")
            efficiency_ax.set_ylabel("weak-scaling efficiency (%)")
            mark_better(time_ax, "down", loc="lower left")
            mark_better(efficiency_ax, "up", loc="upper right")

    # The two arms are named on themselves rather than in a legend the reader
    # has to cross the figure to reach. The two names are wider than the pair of
    # bars they sit over, so they are stacked rather than set side by side.
    for width, dy in ((1, 48.0), (4, 26.0)):
        first = next(
            (r for r in runs
             if r.option == options[0] and r.width == width
             and (r.world_gpus, r.n) == rungs[0]), None)
        if first is not None:
            lib.label_series(
                axes[0][0], positions[0] + (0.5 if width == 4 else -0.5) * BAR,
                first.cycle_ms, WIDTH_LABEL[width], ca.WIDTH_COLOR[width],
                dy=dy, ha="center", size=8.5)

    lib.title(
        fig,
        "s=4 loses less weak-scaling efficiency than s=1 "
        "as participants are added")
    if efficiency_by_width[1] and efficiency_by_width[4]:
        print(f"  across four participants: s=1 falls to "
              f"{min(efficiency_by_width[1]):.1f}%, "
              f"s=4 holds {min(efficiency_by_width[4]):.1f}%")

    FIGURE.write(fig, rows)
    return None


if __name__ == "__main__":
    raise SystemExit(lib.run(FIGURE, draw))
