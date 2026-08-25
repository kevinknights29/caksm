"""The regime map as a trajectory: how the measured Basket operator moves through
the (R_h, R_v) plane as the grid is refined and as the participant topology grows.

  puffin      24 cores. Grid refinement on the accepted CPU placements. The
              operator climbs out of Lower-Right, crosses theta_v, and stops short
              of Upper-Right. On this machine the two mechanisms are never both open.
  synge_grid  V100. The same refinement under each supported GPU topology.

An opportunity map, not a performance plot. Every coordinate is placed from a
measured operator property, a measured machine capacity and a predeclared cost
formula, and no solver runtime enters any axis. What the figure licenses is a
claim about where communication-avoidance can pay, not a claim that it did.

Sources:
    data/regime/regime_placement.csv          (scripts/regime/regime_placement.sh)
    data/regime/regime_gpu_trajectory.csv     (scripts/regime/regime_gpu_trajectory.sh)

  uv run scripts/plots/regime_trajectory.py
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
import numpy as np

import figstyle as fs
import cpu_figstyle as cpu
import gpu_figstyle as gpu

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "regime"
PLACEMENT_CSV = DATA / "regime_placement.csv"
TRAJECTORY_CSV = DATA / "regime_gpu_trajectory.csv"

DPI = 300
# One figure per panel. A poster lays its panels out on its own terms, and a
# three-panel strip forces a column width on it that it may not have.
PUFFIN = "regime_trajectory_puffin"
SYNGE_GRID = "regime_trajectory_synge_grid"
FIGURES = (PUFFIN, SYNGE_GRID)

DECLARED_GRIDS = (25, 30, 40, 50, 61, 74, 90, 120)
# The shared numerical contract's Krylov ceiling: the GPU build's kGpuCaMaxM, which
# include/gpu_ca_config.hpp fixes at 39 because the projected exponential's static
# shared footprint grows as m^2. A grid needing more converges on the CPU and cannot
# on the GPU, so it is not a comparable point and is dropped from BOTH panels.
# Putting a grid on the CPU panel that the GPU panel cannot reach would make the two
# figures look like one sweep under one contract when they were not.
CONTRACT_M_CEILING = 39

# Where a grid label sits relative to its mark, by position along the trajectory.
# The three differ because the free side does: the coarse end sits on the theta_h
# line, the fine end has its own arrowhead below it, and the middle has the curve
# to its lower left. Keyed by position rather than by n, so dropping a grid under
# the shared contract relabels the new endpoint instead of leaving it bare.
CPU_LABEL_OFFSET = {"first": (-8.0, -12.0, "right"),
                    "middle": (9.0, 7.0, "left"),
                    "last": (2.0, 12.0, "left")}
GPU_LABEL_OFFSET = {"first": (9.0, -4.0, "left"),
                    "middle": (-8.0, -12.0, "right"),
                    "last": (2.0, 12.0, "left")}

C_CPU = cpu.OPERATOR_COLOR["black-scholes"]   # the hue the thesis figure already uses


def label_positions(grids) -> dict[int, str]:
    """Which grids to name, and where: both ends and one in between.

    Derived from the grids actually drawn rather than fixed in advance. The
    trajectory's endpoints are what a reader needs to orient the direction of
    travel, and an endpoint left unnamed because the contract dropped the grid
    that used to be there is the failure this avoids.
    """
    if not grids:
        return {}
    if len(grids) <= 2:
        return {grids[0]: "first", grids[-1]: "last"}
    return {grids[0]: "first",
            grids[len(grids) // 2]: "middle",
            grids[-1]: "last"}


def stance_tag(rows) -> str:
    """The one-word stance for a printed heading, derived rather than asserted."""
    run = sum(1 for r in rows if r["validation"] == "PASS")
    if run == len(rows):
        return "measured"
    return "placement-only" if run == 0 else f"{run}/{len(rows)} measured"


def load_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def blocked(stem: str, reason: str) -> int:
    """Record one figure as blocked, naming what it waits on.

    A newly blocked figure must not leave last week's image behind looking
    current, so its output goes with it.
    """
    (HERE / f"{stem}.png").unlink(missing_ok=True)
    with (HERE / f"{stem}.blocked.txt").open("w") as handle:
        handle.write(f"figure={stem}\nstatus=blocked\nreason={reason}\n")
    print(f"  BLOCKED {stem}: {reason}")
    return 1


def write(fig, stem: str) -> None:
    """Write one figure as a 300 dpi PNG."""
    (HERE / f"{stem}.blocked.txt").unlink(missing_ok=True)
    path = HERE / f"{stem}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


# Inputs, each behind the gate the specification sets for it
def load_cpu_trajectory() -> tuple[list[dict], list[tuple[int, str]], str | None]:
    """The accepted Puffin placements, filtered to the shared contract.

    Returns the kept rows, the grids dropped and why, and a blocking reason if the
    table itself is unusable. A grid is dropped rather than blocking, because a
    contract both machines cannot meet is a fact about the contract, not a fault in
    the table: the figure covers the grids the two machines share and says so.
    """
    if not PLACEMENT_CSV.exists():
        return [], [], f"{PLACEMENT_CSV.relative_to(ROOT)} not found"
    seen: dict[int, dict] = {}
    for row in load_csv(PLACEMENT_CSV):
        n = int(row["n"])
        if n in seen:
            return [], [], f"duplicate placement row at n={n}"
        seen[n] = row
    missing = [n for n in DECLARED_GRIDS if n not in seen]
    if missing:
        return [], [], "placement table is missing n=" + ",".join(map(str, missing))

    kept, dropped = [], []
    for n in DECLARED_GRIDS:
        row = seen[n]
        m, ceiling = int(row["m_measured"]), int(row["m_ceiling"])
        if row["exit_status"] != "0":
            dropped.append((n, f"exit_status={row['exit_status']}"))
        # regime-control warns on a saturated search but still exits 0, so the
        # recorded m is the ceiling rather than a converged dimension.
        elif m >= ceiling:
            dropped.append((n, f"m={m} saturated its own ceiling {ceiling}"))
        elif m > CONTRACT_M_CEILING:
            dropped.append((n, f"m={m} exceeds the shared contract "
                               f"{CONTRACT_M_CEILING}"))
        else:
            kept.append(row)
    if not kept:
        return [], dropped, "no placement grid survives the shared contract"
    return kept, dropped, None


def load_gpu_trajectory() -> tuple[dict, str | None]:
    """Accepted GPU rows, keyed (policy, topology, n).

    Exactly one accepted row per key, as the specification requires: a duplicate,
    contended or unvalidated row blocks its key rather than being averaged away.

    Two stances are admitted, and the figure says which it drew. PASS is a point
    whose arrangement and grid were run and validated on synge. PLACEMENT-ONLY is
    one placed from measured constants without such a run. Anything else, a FAIL
    or a verdict this renderer does not know, is not drawn at all.
    """
    STANCES = {"PASS", "PLACEMENT-ONLY"}
    if not TRAJECTORY_CSV.exists():
        return {}, (f"{TRAJECTORY_CSV.relative_to(ROOT)} not found - run "
                    "scripts/regime/regime_gpu_trajectory.sh")
    keyed: dict[tuple[str, str, int], dict] = {}
    for row in load_csv(TRAJECTORY_CSV):
        if (row["accepted"] != "1" or row["validation"] not in STANCES
                or row["contended"] != "0"):
            continue
        key = (row["policy"], row["topology"], int(row["n_global"]))
        if key in keyed:
            return {}, f"duplicate accepted row for {key}"
        keyed[key] = row
    if not keyed:
        return {}, "no accepted rows in the GPU trajectory table"
    return keyed, None


def sweep(keyed: dict, policy: str, topology: str, grids) -> list[dict] | None:
    """Every point of one sweep, or None when any of them is missing.

    A partially drawn line is the failure this prevents: a trajectory with a hole
    in it still reads as a trajectory.
    """
    rows = [keyed.get((policy, topology, n)) for n in grids]
    return None if any(r is None for r in rows) else rows


# The plane itself, drawn identically in every panel
def draw_plane(ax, xlim, ylim, labels: str = "all", size: float = 7.4) -> None:
    """The map: the two unit thresholds, the contested corner, and the corner names.

    The thresholds are predictions, not fits. theta_v = 1 is where the working set
    fills the cache it is measured against; theta_h = 1 is where one reduction
    costs as much as the compute between two.

    Args:
        ax: Axes to draw on.
        xlim, ylim: the panel's limits, in data coordinates.
        labels: "all" names four corners, "corner" names only Upper-Right, "none"
            names none. A reader learns the plane on panel A, so the later panels
            name only the corner the new data is about.
        size: corner-label font size.
    """
    xlo, xhi = xlim
    ylo, yhi = ylim

    ax.fill_between([max(1.0, xlo), xhi], max(1.0, ylo), yhi,
                    color=gpu.C_CORNER, zorder=0)
    ax.axhline(1.0, color=gpu.C_LIMIT, lw=1.2, ls="--", zorder=1)
    ax.axvline(1.0, color=gpu.C_LIMIT, lw=1.2, ls="--", zorder=1)
    ax.text(xlo * 1.25, 1.0, r"$\theta_v = 1$", color=gpu.C_LIMIT, fontsize=8.2,
            va="bottom", ha="left", zorder=2)
    # Set at the top of its line rather than the foot of it: every panel's data
    # crowds the origin corner, and the foot is also where a legend wants to sit.
    ax.text(1.0, yhi / 1.25, r"$\theta_h = 1$", color=gpu.C_LIMIT, fontsize=8.2,
            va="top", ha="right", rotation=90, zorder=2)

    if labels == "none":
        return

    def geo(lo, hi, frac):
        """A fraction of the way across a decade span, which is what the eye reads."""
        return float(np.exp(np.log(lo) + frac * np.log(hi / lo)))

    corners = [("upper-right", geo(1.0, xhi, 0.55), geo(1.0, yhi, 0.87))]
    if labels == "all":
        corners += [
            ("upper-left", geo(xlo, 1.0, 0.5), geo(1.0, yhi, 0.7)),
            ("lower-left", geo(xlo, 1.0, 0.5), geo(ylo, 1.0, 0.2)),
            ("lower-right", geo(1.0, xhi, 0.55), geo(ylo, 1.0, 0.2)),
        ]
    for name, x, y in corners:
        ax.text(x, y, gpu.REGIME_LABEL[name], ha="center", va="center",
                fontsize=size, color=gpu.C_GUIDE, linespacing=1.35, zorder=2)


def finish(ax, xlim, ylim) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$R_h$   reduction cost / compute between reductions")
    ax.set_ylabel(r"$R_v$   working set / cache")
    fs.style_axes(ax)


# Panel A: the accepted CPU trajectory
def panel_cpu(rows: list[dict]) -> dict:
    fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    n = np.array([int(r["n"]) for r in rows])
    rv = np.array([float(r["R_v_measured"]) for r in rows])
    rh = np.array([float(r["R_h_measured"]) for r in rows])

    xlim, ylim = (1.5e-3, 1.5e1), (2.2e-2, 2.5e1)
    draw_plane(ax, xlim, ylim, labels="all")

    ax.plot(rh, rv, "-", color=C_CPU, lw=2.0, zorder=4, solid_capstyle="round")
    ax.scatter(rh, rv, s=58, marker=gpu.POLICY_MARKER["cpu-placement"],
               color=C_CPU, edgecolor="white", linewidth=0.8, zorder=6)
    ax.annotate("", xy=(rh[-1], rv[-1]), xytext=(rh[-2], rv[-2]),
                arrowprops=dict(arrowstyle="-|>", color=C_CPU, lw=2.0,
                                shrinkA=0, shrinkB=8), zorder=5)

    where = label_positions([int(g) for g in n])
    for grid, x, y in zip(n, rh, rv):
        if int(grid) in where:
            dx, dy, ha = CPU_LABEL_OFFSET[where[int(grid)]]
            fs.label_value(ax, x, y, f"n={grid}", dy=dy, dx=dx, ha=ha,
                           color=C_CPU, size=8.0)
    # Named on the series rather than in a key: one line needs no legend, and the
    # direction of travel is the thing a reader has to be told.
    fs.label_series(ax, rh[3], rv[3], "measured placement,\nfiner grid $\\longrightarrow$",
                    C_CPU, dy=-30.0, dx=-6.0, ha="right", size=8.2)

    finish(ax, xlim, ylim)

    below = [int(g) for g, v in zip(n, rv) if v < 1.0]
    above = [int(g) for g, v in zip(n, rv) if v >= 1.0]
    return {
        "crossing": (max(below), min(above)) if below and above else None,
        "upper_right": int(np.sum((rv >= 1.0) & (rh >= 1.0))),
        "rh_max": float(rh.max()),
        "n": n, "rv": rv, "rh": rh, "fig": fig,
    }


# Panel B: the same refinement under every topology
def panel_gpu_refinement(keyed: dict, grids) -> dict:
    fig, ax = plt.subplots(figsize=(6.6, 5.4), constrained_layout=True)
    xlim, ylim = (8.0e-2, 1.6e3), (1.2e-1, 6.0e2)
    draw_plane(ax, xlim, ylim, labels="corner")

    drawn, dropped = {}, []
    for key in gpu.TOPOLOGY_ORDER:
        rows = sweep(keyed, "fixed-global", key, grids)
        if rows is None:
            dropped.append(key)
        else:
            drawn[key] = rows

    for key, rows in drawn.items():
        color = gpu.TOPOLOGY_COLOR[key]
        rh = np.array([float(r["rh"]) for r in rows])
        rv = np.array([float(r["rv"]) for r in rows])
        ax.plot(rh, rv, color=color, lw=1.9, ls=gpu.TOPOLOGY_LINESTYLE[key],
                zorder=4, solid_capstyle="round", label=gpu.TOPOLOGY_LABEL[key])
        ax.scatter(rh, rv, s=38, marker=gpu.POLICY_MARKER["fixed-global"],
                   color=color, edgecolor="white", linewidth=0.7, zorder=6)
        ax.annotate("", xy=(rh[-1], rv[-1]), xytext=(rh[-2], rv[-2]),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.9,
                                    shrinkA=0, shrinkB=7), zorder=5)

    # Grids are labeled on one line only. Every line carries the same sequence in
    # the same order, so four sets of eight labels would cost clarity and buy
    # nothing the iso-grid guide does not already give.
    if "1gpu-1node" in drawn:
        where = label_positions(list(grids))
        for grid, row in zip(grids, drawn["1gpu-1node"]):
            if grid in where:
                dx, dy, ha = GPU_LABEL_OFFSET[where[grid]]
                fs.label_value(ax, float(row["rh"]), float(row["rv"]), f"n={grid}",
                               dy=dy, dx=dx, ha=ha, size=7.8,
                               color=gpu.TOPOLOGY_COLOR["1gpu-1node"])

    finish(ax, xlim, ylim)
    fs.legend(ax, loc="lower left", opaque=True)
    return {"drawn": drawn, "dropped": dropped, "fig": fig, "grids": grids}


def main() -> int:
    gpu.validate_palette([gpu.TOPOLOGY_COLOR[k] for k in gpu.TOPOLOGY_ORDER])
    print("regime trajectory:")

    # Each figure blocks on its own inputs. Panel A rests on the puffin run and
    # panels B and C on the synge table, so one missing table must not take down
    # a panel that has everything it needs.
    failed = 0
    cpu_rows, cpu_dropped, why = load_cpu_trajectory()
    if cpu_dropped:
        print(f"  shared contract: m must converge below {CONTRACT_M_CEILING}, the GPU")
        print("  build's kGpuCaMaxM. Dropped from every panel:")
        for n, reason in cpu_dropped:
            print(f"    n={n}: {reason}")
    if why:
        failed += blocked(PUFFIN, why)
        a = None
    else:
        a = panel_cpu(cpu_rows)
        write(a["fig"], PUFFIN)
    # Both panels sweep the same grids, so the pair reads as one sequence under one
    # contract rather than two sweeps that happen to sit side by side.
    grids = tuple(int(r["n"]) for r in cpu_rows)

    keyed, why = load_gpu_trajectory()
    if why:
        failed += blocked(SYNGE_GRID, why)
        return 1 if failed else 0

    b = panel_gpu_refinement(keyed, grids)
    if b["dropped"]:
        plt.close(b["fig"])
        failed += blocked(SYNGE_GRID, "topologies missing a refinement point: "
                          + ",".join(b["dropped"]))
    else:
        write(b["fig"], SYNGE_GRID)

    # The numbers a caption quotes. Printed, not drawn: a poster caption should be
    # able to state the verdict without re-reading the CSV.
    print()
    if a is not None:
        print("  puffin      grid refinement          [measured run]")
        if a["crossing"]:
            print(f"       theta_v crossed between n={a['crossing'][0]} and "
                  f"n={a['crossing'][1]}")
        print(f"       largest R_h reached: {a['rh_max']:.3g}")
        print(f"       points in Upper-Right: {a['upper_right']} of {len(a['n'])}")
    if not b["dropped"]:
        b_rows = [r for rows in b["drawn"].values() for r in rows]
        print(f"  synge_grid  grid refinement by topology   [{stance_tag(b_rows)}]")
        for key, rows in b["drawn"].items():
            rv = np.array([float(r["rv"]) for r in rows])
            rh = np.array([float(r["rh"]) for r in rows])
            both = int(np.sum((rv >= 1.0) & (rh >= 1.0)))
            grids = [n for n, v, h in zip(b["grids"], rv, rh)
                     if v >= 1.0 and h >= 1.0]
            spill = [n for n, r in zip(b["grids"], rows)
                     if r["spmv_resident"] == "0"]
            note = (f", SpMV leaves L2 at n={spill[0]}" if spill
                    else ", SpMV L2-resident throughout")
            print(f"       {gpu.TOPOLOGY_LABEL[key]:<16} Upper-Right at {both}/"
                  f"{len(b['grids'])} grids"
                  + (f": n={','.join(map(str, grids))}" if grids else "")
                  + note)
    run = sum(1 for r in keyed.values() if r["validation"] == "PASS")
    if run == 0:
        print("  synge_grid is placed, not run: every constant behind them was measured")
        print("  on synge, but no solver was run per topology, so no numerical result")
        print("  was validated. Take those runs with")
        print("    MEASURE=1 ./scripts/regime/regime_gpu_trajectory.sh   (on synge, -N 2)")
    else:
        print(f"  synge_grid: {run} of {len(keyed)} points carry a validated "
              "synge run.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
