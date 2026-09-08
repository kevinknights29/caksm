"""The regime map as a trajectory on maguire: how the measured Basket operator moves
through the (R_h, R_v) plane as the grid is refined, under each H200 participant
count of one node.

The synge companion is scripts/plots/regime_trajectory.py, whose plane, thresholds
and helpers this reuses so the two figures can be read side by side. What differs is
the machine and the arrangements: synge offers two GPUs over two nodes, maguire's
gpu01 offers eight over one.

Every point is a PREDICTION. The coordinates come from a measured operator property
(the converged Krylov dimension in the placement table), a measured machine capacity
(the h200 preset), and a measured collective cost for that exact participant count
(kGpuTopologies), combined by the predeclared byte model. No solver has run on this
hardware, so every row carries validation=PLACEMENT-ONLY and the figure says so.

Reading it: refinement climbs. R_v grows with the local working set, so the coarse
end of each sweep sits low-right and refinement carries it up and to the left. More
participants shard the same global grid into smaller local slabs, which lowers R_v,
while the collective they must cross costs more, which raises R_h: the sweeps move
down and right as the participant count grows.

  uv run scripts/plots/regime_trajectory_maguire.py
"""
# /// script
# dependencies = ["matplotlib", "numpy"]
# ///
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs
import gpu_figstyle as gpu
import regime_trajectory as rt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TRAJECTORY_CSV = ROOT / "data" / "regime" / "regime_gpu_trajectory_h200.csv"
FIGURE = "regime_trajectory_maguire"

# Two FAMILIES of arrangement, and the distinction is the point of the figure.
#
#   one node   ncclCommInitAll: ONE process driving N local GPUs, over NVLink.
#   two nodes  one MPI rank per GPU, aggregating within each node then crossing
#              InfiniBand once.
#
# They differ in the node split AND in the launch structure, so a difference between
# the families is not attributable to the split alone. The experiment that would
# separate them, N MPI ranks on a single node, has not been run. Hue therefore
# carries participant count, which IS controlled, and the families are separated by
# line style and marker fill so no reader joins them into one axis by accident.
ONE_NODE = ("1gpu-1node", "2gpu-1node", "4gpu-1node", "8gpu-1node")
TWO_NODE = ("2gpu-2node", "4gpu-2node", "8gpu-2node", "16gpu-2node")
TOPOLOGY_ORDER = ONE_NODE + TWO_NODE

# Hue by participant count, so the same count is the same color in both families and
# the 8-on-one-node against 8-over-two-nodes comparison is a color match.
_BY_COUNT = {
    1:  gpu.TOPOLOGY_COLOR["1gpu-1node"],
    2:  gpu.TOPOLOGY_COLOR["2gpu-1node"],
    4:  gpu.TOPOLOGY_COLOR["2gpu-2node"],
    8:  gpu.TOPOLOGY_COLOR["4gpu-2node"],
    16: gpu.C_INK,
}
TOPOLOGY_COLOR = {
    "1gpu-1node": _BY_COUNT[1],  "2gpu-1node": _BY_COUNT[2],
    "4gpu-1node": _BY_COUNT[4],  "8gpu-1node": _BY_COUNT[8],
    "2gpu-2node": _BY_COUNT[2],  "4gpu-2node": _BY_COUNT[4],
    "8gpu-2node": _BY_COUNT[8],  "16gpu-2node": _BY_COUNT[16],
}
# Solid for one node, dotted for two: the family is legible in a black-and-white
# print, where hue is not.
TOPOLOGY_LINESTYLE = {k: ("-" if k in ONE_NODE else (0, (1.4, 1.6)))
                      for k in TOPOLOGY_ORDER}
TOPOLOGY_LABEL = {
    "1gpu-1node": "1 GPU",           "2gpu-1node": "2 GPUs",
    "4gpu-1node": "4 GPUs",          "8gpu-1node": "8 GPUs",
    "2gpu-2node": "2 GPUs, 2 nodes", "4gpu-2node": "4 GPUs, 2 nodes",
    "8gpu-2node": "8 GPUs, 2 nodes", "16gpu-2node": "16 GPUs, 2 nodes",
}


def load_trajectory() -> tuple[dict, str | None]:
    """Accepted rows, keyed (policy, topology, n).

    The same admission rule as the synge loader: exactly one accepted, uncontended
    row per key, and a duplicate blocks its key rather than being averaged away.
    """
    STANCES = {"PASS", "PLACEMENT-ONLY"}
    if not TRAJECTORY_CSV.exists():
        return {}, (f"{TRAJECTORY_CSV.relative_to(ROOT)} not found - run\n"
                    "  ./build/regime-gpu-trajectory --machine h200 --policy both \\\n"
                    "      --csv data/regime/regime_gpu_trajectory_h200.csv")
    keyed: dict[tuple[str, str, int], dict] = {}
    for row in rt.load_csv(TRAJECTORY_CSV):
        if (row["accepted"] != "1" or row["validation"] not in STANCES
                or row["contended"] != "0"):
            continue
        key = (row["policy"], row["topology"], int(row["n_global"]))
        if key in keyed:
            return {}, f"duplicate accepted row for {key}"
        keyed[key] = row
    if not keyed:
        return {}, "no accepted rows in the maguire trajectory table"
    return keyed, None


def panel(keyed: dict, grids) -> dict:
    fig, ax = plt.subplots(figsize=(6.6, 5.4), constrained_layout=True)
    # Wider than the synge panel on both axes. Sixteen participants push R_h past
    # 5e3 at the coarse end while shrinking R_v to 6e-3, so the synge limits would
    # crop the very points the added arrangements contribute.
    xlim, ylim = (5.0e-1, 9.0e3), (4.0e-3, 1.0e1)
    rt.draw_plane(ax, xlim, ylim, labels="corner")

    drawn, dropped = {}, []
    for key in TOPOLOGY_ORDER:
        rows = rt.sweep(keyed, "fixed-global", key, grids)
        if rows is None:
            dropped.append(key)
        else:
            drawn[key] = rows

    for key, rows in drawn.items():
        color = TOPOLOGY_COLOR[key]
        rh = np.array([float(r["rh"]) for r in rows])
        rv = np.array([float(r["rv"]) for r in rows])
        ax.plot(rh, rv, color=color, lw=1.9, ls=TOPOLOGY_LINESTYLE[key],
                zorder=4, solid_capstyle="round", label=TOPOLOGY_LABEL[key])
        # Filled markers for one node, hollow for two. Fill is the second channel
        # carrying the family, so the distinction survives a greyscale print.
        one_node = key in ONE_NODE
        ax.scatter(rh, rv, s=38, marker=gpu.POLICY_MARKER["fixed-global"],
                   color=color if one_node else "white",
                   edgecolor="white" if one_node else color,
                   linewidth=0.7 if one_node else 1.3, zorder=6)
        # The arrow marks the direction of refinement, so the sweep is not read
        # backwards: the head is the finest grid.
        ax.annotate("", xy=(rh[-1], rv[-1]), xytext=(rh[-2], rv[-2]),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.9,
                                    shrinkA=0, shrinkB=7), zorder=5)

    # Grids labeled on the single-GPU line only. Every line carries the same grids
    # in the same order, so four sets of labels would cost clarity and buy nothing.
    if "1gpu-1node" in drawn:
        where = rt.label_positions(list(grids))
        for grid, row in zip(grids, drawn["1gpu-1node"]):
            if grid in where:
                dx, dy, ha = rt.GPU_LABEL_OFFSET[where[grid]]
                fs.label_value(ax, float(row["rh"]), float(row["rv"]), f"n={grid}",
                               dy=dy, dx=dx, ha=ha, size=7.8,
                               color=TOPOLOGY_COLOR["1gpu-1node"])

    rt.finish(ax, xlim, ylim)
    fs.legend(ax, loc="lower left", opaque=True)
    return {"drawn": drawn, "dropped": dropped, "fig": fig, "grids": grids}


def main() -> int:
    keyed, error = load_trajectory()
    if error:
        return rt.blocked(FIGURE, error)

    grids = sorted({n for (policy, _, n) in keyed if policy == "fixed-global"})
    if not grids:
        return rt.blocked(FIGURE, "no fixed-global rows to draw")

    result = panel(keyed, grids)
    if not result["drawn"]:
        return rt.blocked(FIGURE, "no arrangement had a complete sweep")
    rt.write(result["fig"], FIGURE)

    print(f"  {FIGURE}: {len(result['drawn'])} arrangement(s) over "
          f"{len(grids)} grid(s) {grids}")
    if result["dropped"]:
        print(f"  incomplete, not drawn: {', '.join(result['dropped'])}")
    print("  every point is PLACEMENT-ONLY: predicted from measured constants,")
    print("  with no solver run on this hardware.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
