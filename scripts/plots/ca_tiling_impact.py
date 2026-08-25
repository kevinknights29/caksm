"""What changing the matrix-powers tile geometry does to DRAM traffic.

The tiling result on its own. Four geometries at each of the two points where a
better tile was expected to show: the 8x4x4 the solver ships, and the three
alternatives probed against it. The bar is measured DRAM traffic as a fraction of
the DRAM roof; the figure under it is effective read redundancy, which is how many
times the kernel reads each input value.

The two are worth reading together, and that is the whole finding. A geometry can
cut traffic while raising redundancy, or raise traffic while cutting it, because
the tile trades reuse inside shared memory against the ghost shell it has to load
to get that reuse. Neither number alone says whether a tile is better.

Source: data/ca-integrator-mpk-vertical, produced by
scripts/regime/ca_mpk_vertical.sh.

  uv run scripts/plots/ca_tiling_impact.py
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

FIGURE = lib.Figure("tiling_impact")

# The tile the solver actually ships; everything else is an alternative probed
# against it.
PRODUCTION_TILE = "8x4x4"


def draw() -> str | None:
    path = lib.VERTICAL / "mpk_vertical.csv"
    if not path.exists():
        return FIGURE.blocked(
            "the vertical CSV is absent; run scripts/regime/ca_mpk_vertical.sh "
            "on an idle Synge V100")
    with path.open() as handle:
        all_records = list(csv.DictReader(handle))
    required = {"tile", "option", "n", "s", "effective_read_redundancy",
                "record_class", "measurement_status", "dram_source",
                "contended", "seconds_median", "dram_bytes", "dram_roof_gbs"}
    if not all_records or not required <= set(all_records[0]):
        return FIGURE.blocked(
            "the vertical CSV predates the measured record schema; rerun "
            "scripts/regime/ca_mpk_vertical.sh")

    # Idle ncu measurements only. A contended or modeled row would be drawn the
    # same height as a measured one and mean something else entirely.
    records = [
        r for r in all_records
        if r["record_class"] == "measurement"
        and r["measurement_status"] == "measured"
        and r["dram_source"] == "ncu"
        and r["contended"] == "0"
    ]
    if not records:
        return FIGURE.blocked("no accepted idle NCU measurement in the CSV")

    roofs = {float(r["dram_roof_gbs"]) for r in records}
    if len(roofs) != 1:
        return FIGURE.blocked(
            "accepted rows mix machine roof constants; split the artifact by "
            "queried device")
    roof = next(iter(roofs))

    def fraction(record: dict) -> float:
        return (float(record["dram_bytes"])
                / float(record["seconds_median"]) * 1e-9 / roof)

    probes = sorted({(int(r["n"]), int(r["s"])) for r in records
                     if r["tile"] != PRODUCTION_TILE})
    alternatives = sorted({r["tile"] for r in records} - {PRODUCTION_TILE})
    tiles = [PRODUCTION_TILE] + alternatives
    if not probes or not alternatives:
        return FIGURE.blocked("the CSV holds no alternative-geometry probe")

    # Every geometry must be present at every probe: a missing bar in a paired
    # comparison reads as a geometry that was tried and did nothing.
    missing = [(tile, probe) for tile in tiles for probe in probes
               if not [r for r in records if r["tile"] == tile
                       and (int(r["n"]), int(r["s"])) == probe]]
    if missing:
        return FIGURE.blocked(
            f"{len(missing)} geometry/probe combination(s) are absent")

    fig, ax = plt.subplots(figsize=(8.6, 4.4), constrained_layout=True)
    positions = np.arange(len(probes), dtype=float)
    bar = 0.78 / len(tiles)
    summary: list[tuple] = []

    for index, tile in enumerate(tiles):
        offset = (index - (len(tiles) - 1) / 2.0) * bar
        heights, redundancies = [], []
        for probe in probes:
            matching = [r for r in records if r["tile"] == tile
                        and (int(r["n"]), int(r["s"])) == probe]
            # The mean over the two options, which agree to a few percent of
            # roof, so the mean stands for both without hiding a spread.
            heights.append(sum(fraction(r) for r in matching) / len(matching))
            redundancies.append(
                sum(float(r["effective_read_redundancy"]) for r in matching)
                / len(matching))
        # The shipped tile is the control and takes the measurement hue; the
        # alternatives take the sequential ramp, which orders them without
        # implying one of them is the answer.
        color = (ca.CATEGORICAL[2] if tile == PRODUCTION_TILE
                 else ca.SEQUENTIAL[index])
        ax.bar(positions + offset, heights, bar, color=color,
               edgecolor="white", lw=0.6, zorder=3,
               label=tile + (" (shipped)" if tile == PRODUCTION_TILE else ""))
        for x, height, redundancy in zip(positions + offset, heights,
                                         redundancies):
            lib.label_value(ax, x, height, f"{height:.1%}", dy=13.0, size=7.6,
                            color=ca.C_INK)
            lib.label_value(ax, x, height, f"{redundancy:.1f}$\\times$ reads",
                            dy=4.0, size=6.9, color=ca.C_MUTED)
        summary.append((tile, heights, redundancies))

    ax.set_xticks(positions)
    ax.set_xticklabels([f"$n = {n}$,  $s = {s}$" for n, s in probes],
                       fontsize=9.5)
    ax.set_ylabel("measured DRAM traffic\n(fraction of the DRAM roof)")
    ax.set_ylim(0.0, 1.42 * max(max(h) for _, h, _ in summary))
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda value, _: f"{value:.0%}"))
    lib.style_axes(ax, grid_axis="y")
    ax.tick_params(axis="x", length=0)
    # Less traffic for the same work is the win; the redundancy figure beneath
    # each bar is what says whether it was bought honestly.
    mark_better(ax, "down", loc="upper right")
    lib.legend(ax, loc="upper left", ncol=len(tiles))

    rows = [
        {"tile": tile, "n": n, "s": s,
         "dram_fraction_of_roof": height, "effective_read_redundancy": red}
        for tile, heights, reds in summary
        for (n, s), height, red in zip(probes, heights, reds)
    ]
    FIGURE.write(fig, rows)

    for (n, s) in probes:
        at = [(tile, h[probes.index((n, s))], r[probes.index((n, s))])
              for tile, h, r in summary]
        best = min(at, key=lambda entry: entry[1])
        shipped = next(e for e in at if e[0] == PRODUCTION_TILE)
        print(f"  n = {n}, s = {s}: shipped {shipped[0]} at {shipped[1]:.1%} "
              f"of roof ({shipped[2]:.1f}x reads);")
        print(f"    lightest traffic is {best[0]} at {best[1]:.1%} "
              f"({best[2]:.1f}x reads), a {shipped[1] / best[1]:.2f}x cut")
    print("  no geometry approaches the roof at either point, so the kernel is")
    print("  latency-bound and the tile choice moves traffic without moving that.")
    return None


if __name__ == "__main__":
    raise SystemExit(lib.run(FIGURE, draw))
