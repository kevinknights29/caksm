"""The correction stated as a figure: as-measured against exact-depth.

The as-measured block build computed one polynomial power it never consumed and
exchanged one halo plane deeper than the recurrence needed, which cost the s=1
control far more than the s=4 arm. The panels are what removing that did to halo
exchanges, halo bytes and cycle time, with both arms attempted in one job on the
same devices.

A stopped or acceptance-failed arm is a hollow hatched bar in every panel, and
each pair carries the percentage change between the arms.

Source: data/ca-integrator-exact-depth-m39, produced by
scripts/regime/ca_exact_depth_weak.sh.

  uv run scripts/plots/ca_correction_impact.py
"""
# /// script
# dependencies = ["matplotlib", "numpy"]
# ///
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from figstyle import mark_better
import ca_figstyle as ca
import ca_figlib as lib

FIGURE = lib.Figure("correction_impact")

ARM_SHORT = {"as-measured": "as measured", "exact-depth": "exact depth"}


def draw() -> str | None:
    runs = [r for r in lib.load_runs(lib.EXACT) if r.world_gpus > 1]
    if not runs:
        return FIGURE.blocked(
            "data/ca-integrator-exact-depth-m39 is empty; run "
            "scripts/regime/ca_exact_depth_weak.sh on an idle two-node Synge "
            "allocation")

    expected_configs = {
        (gpus, n, option, width)
        for gpus, n in ((2, 77), (4, 97))
        for option in ("basket", "rainbow")
        for width in (1, 4)
    }
    keyed: dict[tuple, dict[str, list[lib.Run]]] = {}
    for run in runs:
        if run.arm not in {"as-measured", "exact-depth"}:
            continue
        by_arm = keyed.setdefault(
            (run.world_gpus, run.n, run.option, run.width), {})
        by_arm.setdefault(run.arm, []).append(run)
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
        config for config in expected_configs
        if any(
            len(keyed.get(config, {}).get(arm, [])) != 1
            for arm in ("as-measured", "exact-depth")
        )
        or any(
            not complete_classified_run(keyed[config][arm][0])
            for arm in ("as-measured", "exact-depth")
        )
    ]
    if incomplete:
        return FIGURE.blocked(
            f"the correction sweep is incomplete, contended, or unclassified: "
            f"{len(incomplete)} of {len(expected_configs)} configurations do "
            "not carry exactly one idle seven-repeat passed/stopped/acceptance-"
            "failed run from both arms")
    pairs = [
        (
            config,
            {
                arm: keyed[config][arm][0]
                for arm in ("as-measured", "exact-depth")
            },
        )
        for config in sorted(expected_configs)
    ]

    labels = [f"{gpus} GPUs, n={n}\n{option}, s={width}"
              for (gpus, n, option, width), _ in pairs]
    positions = np.arange(len(pairs), dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(1.6 * len(pairs) + 3.2, 9.8),
                             constrained_layout=True, sharex=True)
    rows = []

    panels = (
        (0, "halo exchanges per step", lambda r: r.halos_per_step, "{:.1f}"),
        (1, "halo traffic (KiB/step)", lambda r: r.halo_kib_per_step,
         "{:.0f}"),
        (2, "cycle time (ms/step)", lambda r: r.cycle_ms, "{:.2f}"),
    )
    width_bar = 0.36
    stopped_any = False
    for index, ylabel, extract, number in panels:
        axis = axes[index]
        ceiling = 0.0
        for offset, arm in ((-0.5, "as-measured"), (0.5, "exact-depth")):
            for position, (_, value) in zip(positions, pairs):
                run = value[arm]
                x = position + offset * width_bar
                measured = extract(run)
                if measured is None:
                    continue
                ceiling = max(ceiling, measured)
                # One treatment for a stopped arm in every panel. The timing
                # panel used to drop the bar and draw a bare cross, which read
                # as a missing measurement rather than a withheld one.
                diagnostic = run.stopped or not run.recordable
                stopped_any = stopped_any or diagnostic
                axis.bar(
                    [x], [measured], width_bar,
                    color=("none" if diagnostic else ca.ARM_COLOR[arm]),
                    edgecolor=(ca.C_STOPPED if diagnostic else "white"),
                    hatch="////" if diagnostic else None,
                    lw=1.1 if diagnostic else 0.6,
                    zorder=3)
                if index == 2 and not diagnostic:
                    low = measured - (run.cycle_min_ms or measured)
                    high = (run.cycle_max_ms or measured) - measured
                    axis.errorbar(
                        [x], [measured],
                        yerr=[[max(low, 0.0)], [max(high, 0.0)]],
                        color=ca.C_INK, marker="none", capsize=2.5,
                        elinewidth=0.9, lw=0, zorder=5)
                lib.label_value(
                    axis, x, measured, number.format(measured), dy=6.0,
                    size=7.4,
                    color=ca.C_STOPPED if diagnostic else ca.C_INK)

        # What the correction was worth, on the pairs where both arms are
        # results. A pair with a withheld arm gets no number, because there is
        # no second measurement to take a percentage against.
        for position, (_, value) in zip(positions, pairs):
            before, after = value["as-measured"], value["exact-depth"]
            if any(r.stopped or not r.recordable for r in (before, after)):
                continue
            first, second = extract(before), extract(after)
            if not first or second is None:
                continue
            change = 100.0 * (second - first) / first
            lib.label_value(
                axis, position, max(first, second),
                "no change" if abs(change) < 0.5 else f"{change:+.0f}%",
                dy=20.0, size=8.0, weight="semibold",
                color=ca.C_MUTED if abs(change) < 0.5
                else ca.ARM_COLOR["exact-depth"])

        axis.set_ylabel(ylabel)
        axis.set_ylim(0.0, ceiling * 1.30)
        lib.style_axes(axis, grid_axis="y")

    legend_handles = [
            Patch(facecolor=ca.ARM_COLOR["as-measured"],
                  label="as measured"),
            Patch(facecolor=ca.ARM_COLOR["exact-depth"],
                  label="exact depth"),
        ]
    if stopped_any:
        legend_handles.append(
            Patch(facecolor="none", edgecolor=ca.C_STOPPED, hatch="////",
                  label="stopped or acceptance failure"))
    axes[0].legend(
        handles=legend_handles,
        fontsize=8.0, frameon=False, ncol=3, loc="upper center",
        labelcolor=ca.C_INK)
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(labels, fontsize=8.2)
    axes[2].set_xlim(positions[0] - 0.62, positions[-1] + 0.62)
    mark_better(axes[2], "down", loc="upper left")

    for (gpus, n, option, width), value in pairs:
        before, after = value["as-measured"], value["exact-depth"]
        rows.append({
            "gpus": gpus, "n": n, "option": option, "s": width,
            "as_measured_halos_per_step": before.halos_per_step,
            "exact_depth_halos_per_step": after.halos_per_step,
            "as_measured_halo_kib_per_step": before.halo_kib_per_step,
            "exact_depth_halo_kib_per_step": after.halo_kib_per_step,
            "as_measured_cycle_ms": before.cycle_ms,
            "as_measured_cycle_min_ms": before.cycle_min_ms,
            "as_measured_cycle_max_ms": before.cycle_max_ms,
            "exact_depth_cycle_ms": after.cycle_ms,
            "exact_depth_cycle_min_ms": after.cycle_min_ms,
            "exact_depth_cycle_max_ms": after.cycle_max_ms,
            "as_measured_repeats": before.repeats,
            "exact_depth_repeats": after.repeats,
            "as_measured_status": before.run_status,
            "exact_depth_status": after.run_status,
            "as_measured_stopped": int(before.stopped),
            "exact_depth_stopped": int(after.stopped),
            "as_measured_recordable": int(before.recordable),
            "exact_depth_recordable": int(after.recordable),
        })

    withheld = sorted(
        f"{gpus} GPUs {option} s={width} {ARM_SHORT[arm]}"
        for (gpus, n, option, width), value in pairs
        for arm, run in value.items()
        if run.stopped or not run.recordable
    )
    exchange_change = [
        100.0 * (value["exact-depth"].halos_per_step
                 - value["as-measured"].halos_per_step)
        / value["as-measured"].halos_per_step
        for (_, _, _, width), value in pairs
        if width == 1 and value["as-measured"].halos_per_step
        and not any(r.stopped or not r.recordable for r in value.values())
    ]
    saving = [
        100.0 * (value["exact-depth"].cycle_ms - value["as-measured"].cycle_ms)
        / value["as-measured"].cycle_ms
        for _, value in pairs
        if not any(r.stopped or not r.recordable for r in value.values())
    ]
    if exchange_change and saving:
        print(f"  s=1 halo exchanges {max(exchange_change):+.0f}%; "
              f"cycle time {max(saving):+.0f}% to {min(saving):+.0f}%")
    if withheld:
        print("  withheld: " + "; ".join(withheld))

    FIGURE.write(fig, rows)
    return None


if __name__ == "__main__":
    raise SystemExit(lib.run(FIGURE, draw))
