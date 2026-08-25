"""The reduction ladder that sets R_h: every rung's measured cost against tau*.

The horizontal axis of the regime map, drawn as the ladder it actually is. A
reduction combines partials over a widening scope, and each rung is one step of
that: threads within a warp, warps within a thread block, thread blocks within a
grid, devices, then nodes. Holding the operator and the device fixed and changing
only the rung moves R_h and nothing else, which is what turns theta_h from a
shaded prediction into a measured crossing.

tau* is the reduction cost at which R_v * R_h reaches 1 and the Upper-Right corner
opens. It is one number for the whole table rather than one per rung, so the
comparison to read is each bar against the line. It is free of N, since the
working set and the cycle bytes are both linear in it, but not of m: more
reductions per cycle means a cheaper one suffices. The grid and Krylov dimension
that priced it are printed in the summary rather than drawn.

Source:
    data/regime/regime_reduction_ladder.csv  (scripts/regime/regime_gpu_trajectory.sh)

  uv run scripts/plots/regime_reduction_ladder.py
"""
# /// script
# dependencies = [
#   "matplotlib",
#   "numpy",
# ]
# ///

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import figstyle as fs
import gpu_figstyle as gpu

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
LADDER_CSV = ROOT / "data" / "regime" / "regime_reduction_ladder.csv"

STEM = "regime_reduction_ladder"
PNG = HERE / f"{STEM}.png"
SIDECAR = HERE / f"{STEM}.blocked.txt"
DPI = 300


def blocked(reason: str) -> int:
    """Record the figure as blocked, naming what it waits on."""
    PNG.unlink(missing_ok=True)
    with SIDECAR.open("w") as handle:
        handle.write(f"figure={STEM}\nstatus=blocked\nreason={reason}\n")
    print(f"  BLOCKED: {reason}")
    return 1


def load() -> tuple[list[dict], str | None]:
    """The ladder, cheapest rung first, with every rung calibrated.

    An uncalibrated rung is not drawn shorter, it is not drawn at all: a rung
    whose increment is zero reports the rung below it, which reads as a real
    measurement of different hardware rather than as a gap.
    """
    if not LADDER_CSV.exists():
        return [], (f"{LADDER_CSV.relative_to(ROOT)} not found - run "
                    "scripts/regime/regime_gpu_trajectory.sh")
    rows = list(csv.DictReader(LADDER_CSV.open()))
    if not rows:
        return [], f"{LADDER_CSV.relative_to(ROOT)} is empty"
    uncalibrated = [r["label"] for r in rows if r["calibrated"] != "1"]
    if uncalibrated:
        return [], "uncalibrated rung(s): " + ", ".join(uncalibrated)
    return rows, None


def draw(rows: list[dict]):
    labels = [r["label"] for r in rows]
    micros = np.array([float(r["t_reduce_s"]) * 1e6 for r in rows])
    is_open = [r["corner_open"] == "1" for r in rows]
    tau = float(rows[0]["tau_star_s"]) * 1e6

    # Cheapest at the top, so the ladder reads downward the way it is climbed.
    y = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(7.0, 3.5), constrained_layout=True)
    ax.barh(y, micros, height=0.62, zorder=3,
            color=[gpu.CORNER_COLOR[o] for o in is_open])

    # tau*: the threshold, not a series. Same hue and dash as theta_v and theta_h
    # carry on the map, since it is the same kind of object.
    ax.axvline(tau, color=gpu.C_LIMIT, lw=1.3, ls="--", zorder=4)
    ax.text(tau, len(rows) - 0.32, f"  $\\tau^* \\approx {tau:.2f}$ $\\mu$s",
            color=gpu.C_LIMIT, fontsize=8.2, ha="left", va="center", zorder=5)

    # The numbers, right-aligned in a column of their own rather than trailing
    # each bar. A reader comparing 0.42 against 0.79 by eye on a linear axis two
    # decades wide is guessing, and a label that trails a short bar lands on the
    # tau* line, which is the one thing on this chart it must not touch.
    # One weight and one color for every row. The bar already says which rungs
    # open the corner, and the tau* line already says where the boundary is;
    # emphasizing the same split a third time in the type only makes the closed
    # rungs read as less trustworthy than the open ones, which they are not.
    value_x = micros.max() * 1.30
    for row_y, value in zip(y, micros):
        ax.text(value_x, row_y, f"{value:.2f} $\\mu$s", va="center", ha="right",
                fontsize=8.6, color=fs.C_INK, zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.0, color=fs.C_INK)

    ax.set_xlabel(r"cumulative cost of one reduction  [$\mu$s]")
    ax.set_xlim(0, micros.max() * 1.34)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(6))
    fs.style_axes(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0)

    handles = [plt.Rectangle((0, 0), 1, 1, color=gpu.CORNER_COLOR[state])
               for state in (False, True)]
    # Below the axes: the bars reach the right edge and the value column occupies
    # what is left of it, so there is no interior pocket a key can sit in without
    # covering data.
    fs.legend(ax, handles=handles,
              labels=[gpu.CORNER_LABEL[False], gpu.CORNER_LABEL[True]],
              loc="upper left", bbox_to_anchor=(0.0, -0.20), ncol=2)
    return fig


def main() -> int:
    print(f"{STEM}:")
    rows, why = load()
    if why:
        return blocked(why)

    fig = draw(rows)
    SIDECAR.unlink(missing_ok=True)
    fig.savefig(PNG, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {PNG.relative_to(ROOT)}")

    # What the poster's caption needs, printed rather than drawn.
    tau = float(rows[0]["tau_star_s"]) * 1e6
    print()
    print(f"  priced at n={rows[0]['n_global']}, m={rows[0]['m_measured']}: "
          f"tau* = {tau:.2f} us")
    opened = [r["label"] for r in rows if r["corner_open"] == "1"]
    print(f"  corner opens at the {opened[0]} rung and stays open above it "
          f"({len(opened)} of {len(rows)} rungs)")
    micros = [float(r["t_reduce_s"]) * 1e6 for r in rows]
    steps = [(rows[i]["label"], micros[i] / micros[i - 1])
             for i in range(1, len(rows))]
    worst = max(steps, key=lambda s: s[1])
    print(f"  largest step: {worst[0]} at {worst[1]:.1f}x the rung below")
    print("  every rung is a measured cumulative latency; no single node-level")
    print("  constant can stand for this ladder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
