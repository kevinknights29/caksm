"""Where the cycle goes: local work, reductions, halos, host synchronization.

Reduction and halo bars are transcript counts priced by the participant
calibration, and local work is the remainder, so no term is fitted. Halos are
priced from the build and transition depth histograms; a transcript predating
that account blocks the figure rather than being priced at the wrong depth.

One panel per option, one bar per correction arm, each arm named under its own
bar.

Sources: data/ca-integrator-exact-depth and
data/ca-participant-calibration-quiet.

  uv run scripts/plots/ca_cycle_decomposition.py
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

FIGURE = lib.Figure("cycle_decomposition")

TERMS = ("local work", "reductions", "halos", "host synchronization")
COMMUNICATION = ("reductions", "halos", "host synchronization")
ARM_SHORT = {"as-measured": "as measured", "exact-depth": "exact depth"}
BAR = 0.38


def draw() -> str | None:
    all_exact_runs = [r for r in lib.load_runs(lib.EXACT) if r.world_gpus > 1]
    expected_exact = {
        (arm, gpus, n, option, width)
        for arm in ("as-measured", "exact-depth")
        for gpus, n in ((2, 77), (4, 97))
        for option in ("basket", "rainbow")
        for width in (1, 4)
    }
    observed_exact: dict[tuple, list[lib.Run]] = {}
    for run in all_exact_runs:
        key = (run.arm, run.world_gpus, run.n, run.option, run.width)
        observed_exact.setdefault(key, []).append(run)

    accepted_statuses = {"passed", "stopped", "acceptance-fail"}

    def complete_classified_run(run: lib.Run) -> bool:
        return (
            run.run_status in accepted_statuses
            and run.recordable
            and run.cycle_ms is not None
            and run.repeats >= 7
            and run.nodes == 2
            and not (run.run_status == "passed" and run.stopped)
        )

    incomplete = [
        key for key in expected_exact
        if len(observed_exact.get(key, [])) != 1
        or not complete_classified_run(observed_exact[key][0])
    ]
    if incomplete:
        return FIGURE.blocked(
            f"the correction sweep is incomplete, contended, or unclassified: "
            f"{len(incomplete)} of {len(expected_exact)} required arm/topology/"
            "option/width points do not carry exactly one idle seven-repeat "
            "passed/stopped/acceptance-failed timing")

    runs = [
        observed_exact[key][0]
        for key in sorted(expected_exact)
        if not observed_exact[key][0].stopped
    ]
    # A withheld arm leaves a gap in a panel. Naming the gap is the difference
    # between a figure that is incomplete and a figure that looks complete and
    # is not. The arm stays in the key: at four GPUs the Rainbow as-measured
    # arm is drawn and only its exact-depth partner is withheld, and a note
    # that dropped the arm would claim both were missing.
    withheld = sorted(
        (key[1], key[3], key[4], key[0])
        for key in expected_exact if observed_exact[key][0].stopped
    )
    if not runs:
        return FIGURE.blocked(
            "no usable distributed transcript carries a timing distribution")

    model = lib.ParticipantModel()
    if not model.available:
        return FIGURE.blocked(
            "data/ca-participant-calibration-quiet is absent, and the "
            "decomposition's reduction and halo terms come from it")

    floors = {id(run): lib.modeled_floors(model, run) for run in runs}
    if not all(f["exact_halo_account"] for f in floors.values()):
        return FIGURE.blocked(
            "the available transcripts predate the build/transition halo-depth "
            "histogram; rerun scripts/regime/ca_exact_depth_weak.sh before "
            "pricing halo latency")

    def terms_of(run: lib.Run) -> dict[str, float]:
        floor = floors[id(run)]
        communication = {
            "reductions": floor["reductions_ms"],
            "halos": floor["halos_ms"],
            "host synchronization": floor["host_sync_ms"],
        }
        return {
            "local work": max(
                run.cycle_ms - sum(communication.values()), 0.0),
            **communication,
        }

    options = ("rainbow", "basket")
    # Each panel holds one column per (topology, width) and one bar per arm
    # inside it, so the pairing the figure is about is the pairing the eye sees.
    groups = {
        option: sorted({
            (r.world_gpus, r.n, r.width) for r in runs if r.option == option
        })
        for option in options
    }
    widest = max(len(value) for value in groups.values())

    fig, axes = plt.subplots(
        1, len(options),
        figsize=(2.9 * widest + 3.2, 6.2), constrained_layout=True,
        squeeze=False, sharey=True,
        gridspec_kw={"width_ratios": [max(len(groups[o]), 1)
                                      for o in options]})
    rows = []
    ceiling = max(r.cycle_ms for r in runs)
    shares: dict[int, list[float]] = {1: [], 4: []}

    for column, option in enumerate(options):
        ax = axes[0][column]
        columns = groups[option]
        positions = np.arange(len(columns), dtype=float)
        for index, arm in enumerate(("as-measured", "exact-depth")):
            offset = (index - 0.5) * BAR
            for position, group in zip(positions, columns):
                gpus, n, width = group
                run = next(
                    (r for r in runs
                     if r.option == option and r.world_gpus == gpus
                     and r.n == n and r.width == width and r.arm == arm), None)
                if run is None:
                    continue
                values = terms_of(run)
                x = position + offset
                bottom = 0.0
                for term in TERMS:
                    ax.bar(
                        [x], [values[term]], BAR, bottom=[bottom],
                        color=ca.TERM_COLOR[term], edgecolor="white",
                        lw=0.6, zorder=3)
                    bottom += values[term]

                communication = sum(values[t] for t in COMMUNICATION)
                share = 100.0 * communication / run.cycle_ms
                shares[width].append(share)
                lib.label_value(
                    ax, x, run.cycle_ms, f"{run.cycle_ms:.2f}",
                    dy=15.0, size=7.6, color=ca.C_INK)
                lib.label_value(
                    ax, x, run.cycle_ms, f"{share:.0f}% comm",
                    dy=5.0, size=7.0, color=ca.C_MUTED)
                # The arm goes under its own bar. This is the label whose
                # absence made the two arms of one configuration read as one
                # series repeated with different values.
                lib.label_value(
                    ax, x, 0.0, ARM_SHORT[arm], dy=-7.0, size=7.2,
                    color=ca.C_MUTED)

                floor = floors[id(run)]
                rows.append({
                    "gpus": run.world_gpus, "nodes": run.nodes, "n": run.n,
                    "option": option, "s": width, "arm": arm,
                    "cycle_ms_per_step": run.cycle_ms,
                    "local_ms_per_step": values["local work"],
                    "reductions_ms_per_step": values["reductions"],
                    "halos_ms_per_step": values["halos"],
                    "host_sync_ms_per_step": values["host synchronization"],
                    "communication_ms_per_step": communication,
                    "communication_share_percent": share,
                    "collectives_per_step": run.collectives_per_step,
                    "agreement_collectives_total": run.collective_split[3],
                    "halos_per_step": run.halos_per_step,
                    "halo_depth": floor["halo_depth"],
                    "build_halos_per_step": run.build_halos_per_step,
                    "build_halo_kib_per_step": run.build_halo_kib_per_step,
                    "build_depth_hist": ";".join(
                        f"{depth}:{count}"
                        for depth, count in sorted(
                            run.build_depth_hist.items())),
                    "transition_halos_per_step":
                        run.transition_halos_per_step,
                    "transition_halo_kib_per_step":
                        run.transition_halo_kib_per_step,
                    "transition_depth_hist": ";".join(
                        f"{depth}:{count}"
                        for depth, count in sorted(
                            run.transition_depth_hist.items())),
                    "host_syncs_per_step": run.host_syncs_per_step,
                    "calibration": ("measured" if floor["exact_calibration"]
                                    else "interpolated"),
                    "repeats": run.repeats,
                })

        ax.set_xticks(positions)
        ax.set_xticklabels(
            [f"\n{gpus} GPUs, n={n}\ns = {width}"
             for gpus, n, width in columns], fontsize=8.2)
        ax.set_xlim(-0.72, len(columns) - 0.28)
        ax.set_ylim(0.0, ceiling * 1.26)
        lib.style_axes(ax, grid_axis="y")
        lib.panel_title(ax, f"{option.capitalize()} option")
        if column == 0:
            ax.set_ylabel("cycle time (ms/step)")
            mark_better(ax, "down", loc="upper left")
            # The four terms are named once, inside the panel, in stack order,
            # so the key reads top to bottom the way the bar does.
            handles = [
                plt.Rectangle((0, 0), 1, 1, facecolor=ca.TERM_COLOR[term])
                for term in reversed(TERMS)
            ]
            lib.legend(
                ax, loc="upper center", ncol=2,
                handles=handles, labels=list(reversed(TERMS)))

    lib.title(
        fig,
        "Communication takes a third of the s=1 cycle "
        "and a tenth of the s=4 cycle")
    print(f"  communication share: s=1 {min(shares[1]):.0f}-"
          f"{max(shares[1]):.0f}%, s=4 {min(shares[4]):.0f}-"
          f"{max(shares[4]):.0f}%")
    if withheld:
        print("  withheld: " + "; ".join(
            f"{gpus} GPUs {option} s={width} {ARM_SHORT[arm]}"
            for gpus, option, width, arm in withheld))

    FIGURE.write(fig, rows)
    return None


if __name__ == "__main__":
    raise SystemExit(lib.run(FIGURE, draw))
