"""Matrix-powers traffic against the two device roofs, as a fraction of each.

The vertical verdict: nothing this kernel does comes close to either roof, so it
is latency-bound at every width and geometry tried. Plotting a fraction rather
than an absolute rate puts the roof at 1.0, so distance from it reads off a
common axis.

One panel per grid, each carrying measured DRAM traffic and modeled L2 demand,
with hue the quantity and line style the option. The alternative tile geometries
were probed at two points each rather than swept, so they get a panel of grouped
bars instead of series.

Source: data/ca-integrator-mpk-vertical, produced by
scripts/regime/ca_mpk_vertical.sh.

  uv run scripts/plots/ca_matrix_powers_roofline.py
"""
# /// script
# dependencies = ["matplotlib", "numpy"]
# ///
from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from figstyle import mark_better
import ca_figstyle as ca
import ca_figlib as lib

FIGURE = lib.Figure("matrix_powers_roofline")

# The tile the solver actually ships. Everything else in the sweep is an
# alternative geometry probed at two points to test whether a better tile exists.
PRODUCTION_TILE = "8x4x4"


def _by_configuration(records: list[dict]) -> dict[tuple, list[float]]:
    """Nearest-roof fractions grouped so the two options sit in one bucket.

    Used only to state the size of the option's non-effect on the figure. A
    claim that two series coincide is worth exactly as much as the number that
    backs it.
    """
    grouped: dict[tuple, list[float]] = {}
    for record in records:
        grouped.setdefault(
            (record["tile"], int(record["n"]), int(record["s"])), []
        ).append(float(record.get("nearest_roof_fraction") or 0.0))
    return grouped


def draw() -> str | None:
    path = lib.VERTICAL / "mpk_vertical.csv"
    if not path.exists():
        return FIGURE.blocked(
            "the vertical CSV is absent; run scripts/regime/ca_mpk_vertical.sh "
            "on an idle Synge V100")

    # Completeness is decided from the CSV alone. The row-by-row checks below
    # prove the exact 44-point plan is present and every row is an idle ncu
    # measurement, which is strictly more than a summary of counts could say.
    with path.open() as handle:
        all_records = list(csv.DictReader(handle))
    required = {
        "tile", "option", "basis", "n", "s", "verdict", "redundancy",
        "effective_read_redundancy", "compulsory_gbs", "tiled_gbs",
        "dram_fraction", "l2_fraction",
        "record_class", "measurement_status", "dram_source", "contended",
        "seconds_min", "seconds_median", "seconds_max", "repeats",
        "dram_roof_gbs", "l2_roof_gbs", "dram_bytes", "tiled_bytes",
    }
    if not all_records or not required <= set(all_records[0]):
        return FIGURE.blocked(
            "the vertical CSV predates the measured/diagnostic record schema; "
            "rerun scripts/regime/ca_mpk_vertical.sh")

    # The production tile at every grid and width, plus the three alternative
    # geometries at the two points where a better tile was expected to show.
    expected_production_points = {
        ("8x4x4", option, "monomial", n, width)
        for option in ("basket", "rainbow")
        for n in (31, 61, 77, 97)
        for width in (1, 2, 3, 4)
    }
    expected_alternative_points = {
        (tile, option, "monomial", n, width)
        for tile in ("16x8x8", "8x8x8", "16x8x4")
        for option in ("basket", "rainbow")
        for n, width in ((61, 3), (97, 4))
    }
    expected_points = expected_production_points | expected_alternative_points
    try:
        actual_points = [
            (
                record["tile"], record["option"], record["basis"],
                int(record["n"]), int(record["s"]),
            )
            for record in all_records
        ]
    except (KeyError, ValueError):
        return FIGURE.blocked(
            "the vertical CSV contains a malformed configuration key")
    if (
        set(actual_points) != expected_points
        or len(actual_points) != len(expected_points)
    ):
        return FIGURE.blocked(
            "the vertical CSV does not contain exactly one row for every declared "
            "point in the 44-point sparse geometry plan")

    records = [
        r for r in all_records
        if r["record_class"] == "measurement"
        and r["measurement_status"] == "measured"
        and r["dram_source"] == "ncu"
        and r["contended"] == "0"
    ]
    measurement_points = [
        (
            record["tile"], record["option"], record["basis"],
            int(record["n"]), int(record["s"]),
        )
        for record in records
    ]
    if (
        set(measurement_points) != expected_points
        or len(measurement_points) != len(expected_points)
    ):
        return FIGURE.blocked(
            "one or more points in the sparse geometry plan failed the idle NCU "
            "profile-marker or kernel-count gate")
    if any(int(r["repeats"]) < 7 for r in records):
        return FIGURE.blocked(
            "accepted NCU rows do not all carry seven-repeat timing "
            "distributions")

    production = [r for r in records if r["tile"] == "8x4x4"]
    expected_production = {
        (option, "monomial", n, width)
        for option in ("basket", "rainbow")
        for n in (31, 61, 77, 97)
        for width in (1, 2, 3, 4)
    }
    production_points = [
        (
            record["option"], record["basis"],
            int(record["n"]), int(record["s"]),
        )
        for record in production
    ]
    if (
        set(production_points) != expected_production
        or len(production_points) != len(expected_production)
    ):
        return FIGURE.blocked(
            "the production 8x4x4 tile lacks one or more of its 32 required idle "
            "NCU measurements")
    if any(r["verdict"] == "undetermined" for r in production):
        return FIGURE.blocked(
            "a production point carries no verdict, which happens only when its "
            "ncu DRAM measurement is missing or contended; re-run those points "
            "on an idle device")

    rows = []

    dram_roofs = {float(r["dram_roof_gbs"]) for r in records}
    l2_roofs = {float(r["l2_roof_gbs"]) for r in records}
    if len(dram_roofs) != 1 or len(l2_roofs) != 1:
        return FIGURE.blocked(
            "accepted rows mix machine roof constants; split the artifact by "
            "queried device")
    dram_roof = next(iter(dram_roofs))
    l2_roof = next(iter(l2_roofs))

    def rates(record: dict) -> dict[str, float]:
        """Both traffic rates and their range, as fractions of their roof.

        Recomputed from bytes and seconds rather than read from the convenience
        `tiled_gbs` column, whose lower decimal precision can otherwise produce
        a tiny negative error bar when min == median or median == max.

        Args:
            record: One accepted CSV row.

        Returns:
            Median, low and high fractions of roof for the measured DRAM
            traffic and the modeled tiled L2 demand.
        """
        seconds = float(record["seconds_median"])
        slowest = float(record["seconds_max"])
        fastest = float(record["seconds_min"])
        dram_bytes = float(record["dram_bytes"])
        tiled_bytes = float(record["tiled_bytes"])
        return {
            "dram": dram_bytes / seconds * 1e-9 / dram_roof,
            "dram_low": dram_bytes / slowest * 1e-9 / dram_roof,
            "dram_high": dram_bytes / fastest * 1e-9 / dram_roof,
            "l2": tiled_bytes / seconds * 1e-9 / l2_roof,
            "l2_low": tiled_bytes / slowest * 1e-9 / l2_roof,
            "l2_high": tiled_bytes / fastest * 1e-9 / l2_roof,
        }

    for record in records:
        value = rates(record)
        seconds = float(record["seconds_median"])
        rows.append({
            "tile": record["tile"], "option": record["option"],
            "basis": record["basis"],
            "n": int(record["n"]), "s": int(record["s"]),
            "redundancy": float(record["redundancy"]),
            "effective_read_redundancy":
                float(record["effective_read_redundancy"]),
            "seconds_min": float(record["seconds_min"]),
            "seconds_median": seconds,
            "seconds_max": float(record["seconds_max"]),
            "repeats": int(record["repeats"]),
            "compulsory_gbs": float(record["compulsory_gbs"]),
            "tiled_gbs": float(record["tiled_bytes"]) / seconds * 1e-9,
            "measured_dram_gbs":
                float(record["dram_bytes"]) / seconds * 1e-9,
            "measured_dram_fraction_of_roof": value["dram"],
            "modeled_l2_fraction_of_roof": value["l2"],
            "dram_fraction": float(record["dram_fraction"]),
            "l2_fraction": float(record["l2_fraction"]),
            "dram_roof_gbs": dram_roof,
            "l2_roof_gbs": l2_roof,
            "dram_source": record["dram_source"],
            "measurement_status": record["measurement_status"],
            "verdict": record["verdict"],
            "nearest_roof_fraction":
                float(record.get("nearest_roof_fraction") or 0.0),
        })

    grids = sorted({int(r["n"]) for r in records})
    options = sorted({r["option"] for r in records})
    alternatives = sorted({r["tile"] for r in records} - {PRODUCTION_TILE})
    probes = sorted({
        (int(r["n"]), int(r["s"])) for r in records
        if r["tile"] != PRODUCTION_TILE
    })

    # One panel per grid on top; the alternative geometries get a panel of their
    # own beneath, because two probe points are not a sweep and drawing them as
    # one put a dozen markers on a single abscissa.
    # A band of its own for the top row's key. Inside a panel it crossed the
    # roof rule, and there is no free corner: the roof sits high on a log axis
    # that has to reach it, and the curves occupy everything below.
    fig = plt.figure(
        figsize=(3.5 * len(grids), 9.0), constrained_layout=True)
    grid_spec = fig.add_gridspec(
        3, len(grids), height_ratios=[0.10, 1.0, 0.85], hspace=0.12)
    key_ax = fig.add_subplot(grid_spec[0, :])
    key_ax.set_axis_off()
    sweep_axes = [fig.add_subplot(grid_spec[1, i]) for i in range(len(grids))]
    tile_ax = fig.add_subplot(grid_spec[2, :])

    QUANTITY = (
        ("dram", "measured DRAM traffic", ca.CATEGORICAL[2], "o"),
        ("l2", "modeled L2 demand", ca.CATEGORICAL[4], "s"),
    )
    # Hue is the quantity, line style the option. The two options land within
    # about a percent of roof of each other, which is not worth a second hue but
    # is worth saying rather than leaving as an unexplained doubled line.
    option_style = {
        option: style for option, style
        in zip(options, ("-", (0, (4, 2)), (0, (1, 2))))
    }

    for axis, n in zip(sweep_axes, grids):
        for key, _, color, marker in QUANTITY:
            for option in options:
                selected = sorted(
                    (r for r in records
                     if r["tile"] == PRODUCTION_TILE and int(r["n"]) == n
                     and r["option"] == option),
                    key=lambda r: int(r["s"]))
                if not selected:
                    continue
                value = [rates(r) for r in selected]
                widths = np.array([int(r["s"]) for r in selected])
                median = np.array([v[key] for v in value])
                axis.errorbar(
                    widths, median,
                    yerr=[median - np.array([v[f"{key}_low"] for v in value]),
                          np.array([v[f"{key}_high"] for v in value])
                          - median],
                    marker=marker, ms=4.5, lw=1.4, capsize=2.5,
                    elinewidth=0.9, color=color,
                    ls=option_style[option], zorder=4)

        axis.set_yscale("log")
        axis.axhline(1.0, color=ca.DIVERGING_LOW, ls="--", lw=1.4, zorder=3)
        axis.set_ylim(top=3.0)
        axis.set_xticks(sorted({int(r["s"]) for r in records}))
        axis.set_xlabel("recurrence steps per launch (s)")
        axis.yaxis.set_major_formatter(
            ticker.FuncFormatter(
                lambda value, _: f"{value:.0%}" if value >= 0.01
                else f"{value:.1%}"))
        lib.panel_title(axis, f"n = {n}")
        lib.style_axes(axis)
        if axis is sweep_axes[0]:
            axis.set_ylabel("fraction of the device roof")
            mark_better(axis, "up", loc="lower left")

    lib.legend(
        key_ax, loc="center", ncol=5,
        handles=[
            plt.Line2D([], [], marker=marker, ms=5.5, lw=1.4, color=color,
                       ls=option_style[option],
                       label=f"{label}, {option.capitalize()}")
            for _, label, color, marker in QUANTITY
            for option in options
        ] + [
            plt.Line2D([], [], ls="--", lw=1.4, color=ca.DIVERGING_LOW,
                       label="device roof")
        ])

    # The alternative geometries, as the paired comparison they are.
    tiles = [PRODUCTION_TILE] + alternatives
    positions = np.arange(len(probes), dtype=float)
    bar = 0.8 / max(len(tiles), 1)
    for index, tile in enumerate(tiles):
        offset = (index - (len(tiles) - 1) / 2.0) * bar
        heights, redundancies = [], []
        for n, s in probes:
            matching = [
                rates(r)["dram"] for r in records
                if r["tile"] == tile and int(r["n"]) == n and int(r["s"]) == s
            ]
            redundancy = [
                float(r["effective_read_redundancy"]) for r in records
                if r["tile"] == tile and int(r["n"]) == n and int(r["s"]) == s
            ]
            # The mean over the two options, which agree to a few percent of
            # roof, so the mean stands for both without hiding a spread.
            heights.append(
                sum(matching) / len(matching) if matching else 0.0)
            redundancies.append(
                sum(redundancy) / len(redundancy) if redundancy else 0.0)
        color = (ca.CATEGORICAL[2] if tile == PRODUCTION_TILE
                 else ca.SEQUENTIAL[index])
        tile_ax.bar(
            positions + offset, heights, bar, color=color,
            edgecolor="white", lw=0.6, zorder=3,
            label=f"{tile}" + (
                " (full-volume baseline)" if tile == PRODUCTION_TILE else ""))
        for x, height, redundancy in zip(
                positions + offset, heights, redundancies):
            if height <= 0.0:
                continue
            lib.label_value(
                tile_ax, x, height, f"{height:.1%}", dy=13.0, size=7.4,
                color=ca.C_INK)
            lib.label_value(
                tile_ax, x, height, f"{redundancy:.1f}x reads", dy=4.0,
                size=6.8, color=ca.C_MUTED)

    # No roof rule here: at these fractions it would sit an order of magnitude
    # above the tallest bar and stretch the panel to nothing but white space.
    # The panel title carries the same fact in one line.
    tile_ax.set_xticks(positions)
    tile_ax.set_xticklabels(
        [f"n = {n}, s = {s}" for n, s in probes], fontsize=9.0)
    tile_ax.set_ylabel("measured DRAM traffic\n(fraction of the DRAM roof)")
    tile_ax.set_ylim(
        0.0, 1.45 * max(
            rates(r)["dram"] for r in records
            if (int(r["n"]), int(r["s"])) in probes))
    tile_ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda value, _: f"{value:.0%}"))
    lib.panel_title(tile_ax, "Alternative tile geometries at the probed points")
    lib.style_axes(tile_ax, grid_axis="y")
    lib.legend(tile_ax, loc="upper left", ncol=len(tiles))

    verdicts = sorted({r["verdict"] for r in production})
    fractions = [
        float(r.get("nearest_roof_fraction") or 0.0) for r in production
    ] or [0.0]
    print(f"  production tile reaches {min(fractions):.1%} to "
          f"{max(fractions):.1%} of its nearest roof; alternatives reach "
          f"{min(rates(r)['dram'] for r in records if r['tile'] != PRODUCTION_TILE):.1%}"
          f" to "
          f"{max(rates(r)['dram'] for r in records if r['tile'] != PRODUCTION_TILE):.1%}"
          " of the DRAM roof")

    FIGURE.write(fig, rows)
    return None


if __name__ == "__main__":
    raise SystemExit(lib.run(FIGURE, draw))
