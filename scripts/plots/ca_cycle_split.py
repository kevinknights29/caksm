"""Where the cycle goes, one option per figure.

The same decomposition ca_cycle_decomposition.py draws in two panels, written as
two figures instead.

Reduction and halo bars are transcript counts priced by the participant
calibration, and local work is the remainder, so no term is fitted. The pricing
is ca_figlib's, the same the two-panel figure uses, so the two cannot drift.

Sources: data/ca-integrator-exact-depth-m39 and
data/ca-participant-calibration-quiet.

  uv run scripts/plots/ca_cycle_split.py
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

TERMS = ("local work", "reductions", "halos", "host synchronization")
COMMUNICATION = ("reductions", "halos", "host synchronization")
ARM_SHORT = {"as-measured": "as measured", "exact-depth": "exact depth"}
OPTIONS = ("basket", "rainbow")
FIGURES = {option: lib.Figure(f"cycle_{option}") for option in OPTIONS}

# Narrower bars than the paired figure, and a gap between the two arms of a
# column. The arms are labeled under their own bars, and two labels that meet in
# the middle of a column read as one label belonging to neither.
BAR = 0.34
GAP = 0.06


def draw() -> str | None:
    all_exact_runs = [r for r in lib.load_runs(lib.EXACT) if r.world_gpus > 1]
    expected = {
        (arm, gpus, n, option, width)
        for arm in ("as-measured", "exact-depth")
        for gpus, n in ((2, 77), (4, 97))
        for option in OPTIONS
        for width in (1, 4)
    }
    observed: dict[tuple, list[lib.Run]] = {}
    for run in all_exact_runs:
        observed.setdefault(
            (run.arm, run.world_gpus, run.n, run.option, run.width),
            []).append(run)

    accepted = {"passed", "stopped", "acceptance-fail"}

    def usable(run: lib.Run) -> bool:
        return (run.run_status in accepted and run.recordable
                and run.cycle_ms is not None and run.repeats >= 7
                and run.nodes == 2
                and not (run.run_status == "passed" and run.stopped))

    incomplete = [k for k in expected
                  if len(observed.get(k, [])) != 1 or not usable(observed[k][0])]
    if incomplete:
        for figure in FIGURES.values():
            figure.blocked(
                f"the correction sweep is incomplete, contended, or "
                f"unclassified: {len(incomplete)} of {len(expected)} required "
                "points do not carry exactly one idle seven-repeat timing")
        return "incomplete"

    runs = [observed[k][0] for k in sorted(expected)
            if not observed[k][0].stopped]
    withheld = sorted((k[1], k[3], k[4], k[0])
                      for k in expected if observed[k][0].stopped)

    model = lib.ParticipantModel()
    if not model.available:
        for figure in FIGURES.values():
            figure.blocked(
                "data/ca-participant-calibration-quiet is absent, and the "
                "decomposition's reduction and halo terms come from it")
        return "no participant calibration"

    floors = {id(run): lib.modeled_floors(model, run) for run in runs}
    if not all(f["exact_halo_account"] for f in floors.values()):
        for figure in FIGURES.values():
            figure.blocked(
                "the available transcripts predate the build/transition "
                "halo-depth histogram; rerun ca_exact_depth_weak.sh")
        return "no halo account"

    def terms_of(run: lib.Run) -> dict[str, float]:
        floor = floors[id(run)]
        communication = {
            "reductions": floor["reductions_ms"],
            "halos": floor["halos_ms"],
            "host synchronization": floor["host_sync_ms"],
        }
        return {"local work": max(run.cycle_ms - sum(communication.values()),
                                  0.0),
                **communication}

    # One vertical scale across both figures. The options land within a percent
    # of each other, and a per-figure scale would rescale that agreement away
    # into two plots that look like different measurements.
    ceiling = max(r.cycle_ms for r in runs)

    for option in OPTIONS:
        columns = sorted({(r.world_gpus, r.n, r.width)
                          for r in runs if r.option == option})
        if not columns:
            FIGURES[option].blocked(f"no usable {option} transcript")
            continue

        fig, ax = plt.subplots(
            figsize=(2.9 * len(columns) + 2.0, 5.4), constrained_layout=True)
        positions = np.arange(len(columns), dtype=float)
        rows = []

        for index, arm in enumerate(("as-measured", "exact-depth")):
            offset = (index - 0.5) * (BAR + GAP)
            for position, (gpus, n, width) in zip(positions, columns):
                run = next((r for r in runs
                            if r.option == option and r.world_gpus == gpus
                            and r.n == n and r.width == width and r.arm == arm),
                           None)
                if run is None:
                    continue
                values = terms_of(run)
                x = position + offset
                bottom = 0.0
                for term in TERMS:
                    ax.bar([x], [values[term]], BAR, bottom=[bottom],
                           color=ca.TERM_COLOR[term], edgecolor="white",
                           lw=0.6, zorder=3)
                    bottom += values[term]

                communication = sum(values[t] for t in COMMUNICATION)
                share = 100.0 * communication / run.cycle_ms
                lib.label_value(ax, x, run.cycle_ms, f"{run.cycle_ms:.2f}",
                                dy=15.0, size=8.0, color=ca.C_INK)
                lib.label_value(ax, x, run.cycle_ms, f"{share:.0f}% comm",
                                dy=5.0, size=7.2, color=ca.C_MUTED)
                lib.label_value(ax, x, 0.0, ARM_SHORT[arm], dy=-7.0, size=7.8,
                                color=ca.C_MUTED)
                rows.append({
                    "option": option, "gpus": gpus, "nodes": run.nodes, "n": n,
                    "s": width, "arm": arm,
                    "cycle_ms_per_step": run.cycle_ms,
                    "local_ms_per_step": values["local work"],
                    "reductions_ms_per_step": values["reductions"],
                    "halos_ms_per_step": values["halos"],
                    "host_sync_ms_per_step": values["host synchronization"],
                    "communication_share_percent": share,
                })

        ax.set_xticks(positions)
        ax.set_xticklabels(
            [f"{gpus} GPUs, $n = {n}$\n$s = {width}$"
             for gpus, n, width in columns], fontsize=9.0)
        ax.set_ylabel("cycle time (ms/step)")
        ax.set_ylim(0.0, ceiling * 1.22)
        lib.style_axes(ax, grid_axis="y")
        ax.tick_params(axis="x", length=0, pad=18)
        mark_better(ax, "down", loc="upper left")
        # Upper right: the tallest column sits mid-figure, and its value label
        # reaches into the top center where a key would otherwise go.
        lib.legend(
            ax, loc="upper right", ncol=2,
            handles=[plt.Rectangle((0, 0), 1, 1, color=ca.TERM_COLOR[t])
                     for t in TERMS],
            labels=list(TERMS))
        FIGURES[option].write(fig, rows)

        share_by_width: dict[int, list[float]] = {}
        for row in rows:
            share_by_width.setdefault(row["s"], []).append(
                row["communication_share_percent"])
        spans = ", ".join(
            f"s = {w}: {min(v):.0f}-{max(v):.0f}%"
            for w, v in sorted(share_by_width.items()))
        print(f"  {option}: communication share {spans}")

    if withheld:
        print("  withheld (stopped) arms, absent from their figure: "
              + "; ".join(f"{g} GPUs {o} s={w} {a}" for g, o, w, a in withheld))
    return None


def main() -> int:
    ca.validate_palette()
    print("cycle_split:")
    draw()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
