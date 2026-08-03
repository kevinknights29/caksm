"""Predicted communication-floor saving against measured cycle-time saving.

The calibration predicts a communication floor rather than a cycle time, so the
figure plots the quantity it genuinely predicts: the s=1 minus s=4 floor saving
against the same difference in measured cycle time. The floor sums reductions,
halos priced at their measured depth, and the probe-derived host-synchronization
lower bound.

Hue is the configuration, meaning participant count and grid, which is what
separates these points; the marker is the correction arm. Only two-node
topologies are plotted, since the calibration was recorded across two nodes.

Sources: data/ca-integrator-weak, -exact-depth and -strong, plus
data/ca-participant-calibration-quiet.

  uv run scripts/plots/ca_predicted_measured.py
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

FIGURE = lib.Figure("predicted_measured")

ARM_MARKER = {"as-measured": "o", "exact-depth": "s"}
ARM_SHORT = {"as-measured": "as measured", "exact-depth": "exact depth"}


def draw() -> str | None:
    runs = [r for r in lib.load_runs(lib.WEAK) + lib.load_runs(lib.EXACT)
            + lib.load_runs(lib.STRONG)
            if r.cycle_ms is not None and r.world_gpus > 1 and not r.stopped
            and r.recordable and r.repeats >= 7 and r.nodes == 2
            and (r.build_depth_hist or r.transition_depth_hist)]
    if not runs:
        return FIGURE.blocked(
            "no idle two-node distributed transcript carries both a seven-repeat "
            "timing distribution and the corrected build/transition halo-depth "
            "histogram")

    # One point per (arm, topology, n, option). Keep the node count in the key:
    # the available participant calibration was measured across two nodes, and
    # must never be applied to the distinct two-GPU/one-node rung.
    keyed: dict[tuple, dict[int, list[lib.Run]]] = {}
    for run in runs:
        widths = keyed.setdefault(
            (run.arm, run.world_gpus, run.nodes, run.n, run.option), {})
        widths.setdefault(run.width, []).append(run)
    duplicates = [
        (key, width)
        for key, widths in keyed.items()
        for width, matches in widths.items()
        if len(matches) != 1
    ]
    if duplicates:
        return FIGURE.blocked(
            f"{len(duplicates)} topology/width point(s) are duplicated across "
            "the input artifacts; select one recorded sweep before predicting")

    model = lib.ParticipantModel()
    if not model.available:
        return FIGURE.blocked(
            "data/ca-participant-calibration-quiet is absent, and it is the "
            "model the prediction comes from")
    missing_participants = sorted({
        run.world_gpus for run in runs
        if not model.collectives.get(run.world_gpus)
        or not model.halos.get(run.world_gpus)
    })
    if missing_participants:
        return FIGURE.blocked(
            "the two-node participant calibration has no complete collective "
            "and halo table for participant count(s) "
            + ", ".join(str(value) for value in missing_participants))

    points, rows = [], []
    for (arm, gpus, nodes, n, option), widths in sorted(keyed.items()):
        if 1 not in widths or 4 not in widths:
            continue
        narrow, wide = widths[1][0], widths[4][0]
        narrow_floor = lib.modeled_floors(model, narrow)
        wide_floor = lib.modeled_floors(model, wide)

        def total(floor: dict) -> float:
            return (floor["reductions_ms"] + floor["halos_ms"]
                    + floor["host_sync_ms"])

        predicted = total(narrow_floor) - total(wide_floor)
        measured = narrow.cycle_ms - wide.cycle_ms
        points.append({
            "arm": arm, "gpus": gpus, "n": n, "option": option,
            "predicted": predicted, "measured": measured,
            "residual": measured - predicted,
        })
        rows.append({
            "arm": arm, "gpus": gpus, "nodes": nodes,
            "n": n, "option": option,
            "predicted_saving_ms_per_step": predicted,
            "measured_saving_ms_per_step": measured,
            "residual_ms_per_step": measured - predicted,
            "s1_reductions_ms": narrow_floor["reductions_ms"],
            "s4_reductions_ms": wide_floor["reductions_ms"],
            "s1_halos_ms": narrow_floor["halos_ms"],
            "s4_halos_ms": wide_floor["halos_ms"],
            "s1_host_sync_ms": narrow_floor["host_sync_ms"],
            "s4_host_sync_ms": wide_floor["host_sync_ms"],
            "s1_host_sync_measured_ms": narrow_floor["host_sync_measured_ms"],
            "s4_host_sync_measured_ms": wide_floor["host_sync_measured_ms"],
            "s1_halo_depth": narrow_floor["halo_depth"],
            "s4_halo_depth": wide_floor["halo_depth"],
            "s1_cycle_ms": narrow.cycle_ms,
            "s4_cycle_ms": wide.cycle_ms,
            "calibration": (
                "measured"
                if narrow_floor["exact_calibration"]
                and wide_floor["exact_calibration"] else "interpolated"),
        })

    if not points:
        return FIGURE.blocked(
            "no configuration carries both an s=1 and an s=4 transcript")

    # The configuration is what separates these points, so it is what takes the
    # hue. Ordering by participants then grid makes the ramp mean something.
    configurations = sorted({(p["gpus"], p["n"]) for p in points})
    if len(configurations) > len(ca.CATEGORICAL):
        return FIGURE.blocked(
            f"{len(configurations)} configurations exceed the assigned "
            "categorical palette; extend ca_figstyle before drawing")
    config_color = {
        configuration: ca.CATEGORICAL[index]
        for index, configuration in enumerate(configurations)
    }
    config_label = {
        (gpus, n): f"{gpus} GPUs, n={n}" for gpus, n in configurations
    }

    fig, axes = plt.subplots(
        1, 2, figsize=(13.0, 6.0), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.15]})
    parity, residual_ax = axes

    predicted = np.array([p["predicted"] for p in points])
    measured = np.array([p["measured"] for p in points])
    low = float(min(0.0, predicted.min(), measured.min()))
    high = float(max(predicted.max(), measured.max()))
    pad = 0.18 * (high - low)
    span = [low - pad, high + pad]

    parity.plot(span, span, color=ca.C_MUTED, ls="--", lw=1.1, zorder=1)
    # The identity line is the claim, so it is named on itself rather than in a
    # legend: a point on it is a prediction met exactly. It is named low on the
    # line, where no configuration reaches, rather than high where the four-GPU
    # cluster sits.
    # Named low on the line, below it, where no configuration reaches.
    anchor = span[0] + 0.14 * (span[1] - span[0])
    lib.label_series(
        parity, anchor, anchor, "prediction met exactly",
        ca.C_MUTED, dy=-9.0, dx=7.0, ha="left", size=8.5)

    for point in points:
        parity.scatter(
            point["predicted"], point["measured"], s=78, zorder=4,
            marker=ARM_MARKER[point["arm"]],
            color=config_color[(point["gpus"], point["n"])],
            edgecolor="white", linewidth=1.0)

    # One label per configuration, at the cluster it names. Four labels replace
    # thirteen legend entries that resolved to two glyphs.
    for configuration in configurations:
        cluster = [p for p in points
                   if (p["gpus"], p["n"]) == configuration]
        lib.label_series(
            parity,
            float(np.mean([p["predicted"] for p in cluster])),
            float(np.max([p["measured"] for p in cluster])),
            config_label[configuration], config_color[configuration],
            dy=12.0, ha="center", size=8.8)

    parity.set_xlim(*span)
    parity.set_ylim(*span)
    parity.set_aspect("equal", adjustable="box")
    parity.set_xlabel("predicted communication-floor saving (ms/step)")
    parity.set_ylabel("measured cycle-time saving (ms/step)")
    lib.panel_title(
        parity, "Predicted against measured",
        "above the line the solver saved more than the floor explains")
    lib.style_axes(parity)
    # Only the marker needs a key; the hue is already named on the data.
    handles = [
        plt.Line2D([], [], marker=ARM_MARKER[arm], ls="none", ms=7.5,
                   color=ca.C_MUTED, markeredgecolor="white",
                   label=ARM_SHORT[arm])
        for arm in ("as-measured", "exact-depth")
    ]
    lib.legend(parity, loc="lower right", handles=handles)

    # The residual panel is categorical, not a second scatter against the
    # prediction. Bars drawn at their predicted value overlapped and hid each
    # other; one slot per configuration and arm cannot.
    slots = sorted({(p["gpus"], p["n"], p["arm"]) for p in points})
    slot_x = {slot: float(index) for index, slot in enumerate(slots)}
    bar = 0.34
    for point in points:
        slot = (point["gpus"], point["n"], point["arm"])
        # The two options sit either side of their shared slot, close enough to
        # read as the replicate pair they are.
        side = -0.5 if point["option"] == "basket" else 0.5
        residual_ax.bar(
            [slot_x[slot] + side * bar], [point["residual"]], bar,
            color=(ca.DIVERGING_HIGH if point["residual"] >= 0
                   else ca.DIVERGING_LOW),
            edgecolor="white", lw=0.6, zorder=3)

    residual_ax.axhline(0.0, color=ca.C_INK, lw=1.0, zorder=4)
    residual_ax.set_xticks(list(slot_x.values()))
    residual_ax.set_xticklabels(
        [f"{config_label[(gpus, n)]}\n{ARM_SHORT[arm]}"
         for gpus, n, arm in slots], fontsize=7.8)
    # The tick colors carry the configuration through from the parity panel.
    for tick, (gpus, n, _) in zip(residual_ax.get_xticklabels(), slots):
        tick.set_color(config_color[(gpus, n)])
    residual_ax.set_ylabel("measured minus predicted (ms/step)")
    lib.panel_title(
        residual_ax, "Residual by configuration",
        "the miss is set by participants and grid, not by the option")
    lib.style_axes(residual_ax, grid_axis="y")
    # The two hues are a direction, not two categories, so the key says which
    # direction each one is rather than repeating the color names.
    lib.legend(
        residual_ax, loc="lower left",
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor=ca.DIVERGING_HIGH,
                          label="saved more than the floor predicts"),
            plt.Rectangle((0, 0), 1, 1, facecolor=ca.DIVERGING_LOW,
                          label="saved less than the floor predicts"),
        ])

    matched: dict[tuple, list[float]] = {}
    for point in points:
        matched.setdefault(
            (point["arm"], point["gpus"], point["n"]), []
        ).append(point["residual"])
    option_gap = max(
        max(values) - min(values)
        for values in matched.values() if len(values) > 1)

    best = min(points, key=lambda p: abs(p["residual"]))
    worst = max(points, key=lambda p: abs(p["residual"]))
    lib.title(
        fig,
        "The calibrated floor predicts the saving at two participants, "
        "not elsewhere")
    print(f"  closest {abs(best['residual']):.2f} ms/step at "
          f"{config_label[(best['gpus'], best['n'])]}; furthest "
          f"{abs(worst['residual']):.2f} ms/step at "
          f"{config_label[(worst['gpus'], worst['n'])]}; "
          f"option moves a residual by at most {option_gap:.2f} ms/step")

    FIGURE.write(fig, rows)
    return None


if __name__ == "__main__":
    raise SystemExit(lib.run(FIGURE, draw))
