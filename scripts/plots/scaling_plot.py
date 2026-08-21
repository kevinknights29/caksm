"""OpenMP scaling figures for the baseline KSM-EI solver on puffin.

Six figures: strong scaling per kernel, the initialization A/B contrast,
weak-scaling efficiency, the quarantined m(n) growth curve, the locality
schedule experiment, and the reduction-primitive correction.

Two of these are null results and are titled as such. The initialization arms
coincide, which is what makes the schedule experiment necessary, and a faster
reduction primitive leaves Gram-Schmidt unmoved.

Median markers carry inter-quartile-range error bars. Every figure except the
reduction one shows the tree arm only, since `linear` is a historical artifact
rather than a result.

Produces scaling_strong.png, scaling_ab.png, scaling_weak.png, scaling_mn.png,
scaling_locality.png and scaling_reduction.png.

Sources: data/scaling/scaling_strong.csv, scaling_weak.csv, mn_growth.csv and
scaling_locality.csv, produced by the matching scripts/scaling/*.sh.

  uv run scripts/plots/scaling_plot.py
"""
# /// script
# dependencies = [
#   "matplotlib",
#   "numpy",
# ]
# ///

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import figstyle as fs
import cpu_figstyle as cpu
from figstyle import mark_better

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data" / "scaling"

STRONG_CSV = DATA / "scaling_strong.csv"
WEAK_CSV = DATA / "scaling_weak.csv"
MN_CSV = DATA / "mn_growth.csv"
LOCALITY_CSV = DATA / "scaling_locality.csv"

DPI = 300
DRAM_SOCKET = 95.0   # ~4ch DDR4-3200 achievable aggregate, GB/s
L3_SLICE_MIB = 16.0  # one CCX L3 slice

CORE_TICKS = [1, 3, 6, 12, 24]
KERNELS = (("spmv", "SpMV"), ("gs", "Gram-Schmidt"))
PLACEMENT = (("A", "naive"), ("B", "first-touch"))


def load_csv(path: Path) -> list[dict]:
    """Read one sweep CSV into a list of row dicts."""
    with path.open() as handle:
        return list(csv.DictReader(handle))


def num(rows, key):
    """One column as a float array."""
    return np.array([float(r[key]) for r in rows])


def yerr(rows, med_key, q1_key, q3_key):
    """Inter-quartile range as asymmetric error-bar offsets from the median."""
    med, q1, q3 = num(rows, med_key), num(rows, q1_key), num(rows, q3_key)
    return np.vstack([np.maximum(med - q1, 0.0), np.maximum(q3 - med, 0.0)])


def subset(rows, **filt):
    """Filter rows, defaulting to the tree reduction arm.

    scaling_strong.sh sweeps both reduction primitives, so a CSV holds two rows
    per (arm, n, P); unfiltered they interleave into a zigzag that is neither
    curve. Pass reduce=None to opt out.
    """
    filt.setdefault("reduce", "tree")
    if filt["reduce"] is None or not rows or "reduce" not in rows[0]:
        filt.pop("reduce", None)
    out = [r for r in rows
           if all(str(r[k]) == str(v) for k, v in filt.items())]
    out.sort(key=lambda r: float(r["P"]))
    return out


def core_axis(ax, label="threads P"):
    """The shared thread axis: log scale, ticks on the physical core counts."""
    ax.set_xscale("log")
    ax.set_xticks(CORE_TICKS)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel(label)


def plot_strong(rows) -> None:
    """Per-kernel strong scaling at both grids, both placement arms."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), constrained_layout=True)

    speedups = {}
    for ax, n in zip(axes, (31, 61)):
        for arm, tag in PLACEMENT:
            selected = subset(rows, n=n, arm=arm)
            if not selected:
                continue
            for kernel, name in KERNELS:
                P = num(selected, "P")
                median = num(selected, f"{kernel}_ms")
                ax.errorbar(
                    P, median,
                    yerr=yerr(selected, f"{kernel}_ms", f"{kernel}_q1",
                              f"{kernel}_q3"),
                    ls=cpu.PLACEMENT_LINESTYLE[tag],
                    color=cpu.KERNEL_COLOR[name], lw=1.8,
                    marker=cpu.KERNEL_MARKER[name], ms=4, capsize=2,
                    label=f"{name}, {tag}", zorder=4)
                if tag == "first-touch":
                    speedups[(n, name)] = median[0] / median[-1]

        # One more L3 slice engaged every three physical cores.
        for core in range(3, 25, 3):
            ax.axvline(core, color=fs.C_GRID, lw=0.5, zorder=0)
        ax.set_yscale("log")
        core_axis(ax, "threads P  (physical cores; P/3 = L3 slices engaged)")
        fs.panel_title(ax, f"n = {n}")
        fs.style_axes(ax)
        if ax is axes[0]:
            ax.set_ylabel("median kernel time (ms)")
            # Opaque: the CCX rules run the full height of the panel and
            # would otherwise cross the label text.
            fs.legend(ax, loc="upper right", ncol=2, opaque=True)
        mark_better(ax, "down", loc="lower left")

    out = HERE / "scaling_strong.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")
    for (n, name), value in sorted(speedups.items()):
        print(f"  n={n} {name}: {value:.1f}x from 1 to 24 threads")


def plot_ab(rows) -> None:
    """The initialization A/B contrast at n=61, which is a null result.

    The two arms coincide at every thread count. That is the control the
    schedule experiment needs: on a single-NUMA socket with a per-CCX victim
    cache, residency is set by which core accesses a row, not by which thread
    first-touched the page.
    """
    naive, touched = subset(rows, n=61, arm="A"), subset(rows, n=61, arm="B")
    if not naive or not touched:
        print("(skip) scaling_ab.png - needs both arms at n=61")
        return

    fig, (ax_time, ax_speed) = plt.subplots(
        1, 2, figsize=(12.4, 5.0), constrained_layout=True)

    # Hue, marker and dash all separate the arms here. The figure's claim is
    # that they coincide, and two curves can only be seen to coincide if each
    # can be told apart in the first place.
    arms = ((naive, "naive", "naive master-thread init"),
            (touched, "first-touch", "parallel first-touch"))
    for selected, arm, tag in arms:
        P, median = num(selected, "P"), num(selected, "spmv_ms")
        style = dict(color=cpu.PLACEMENT_COLOR[arm],
                     ls=cpu.PLACEMENT_LINESTYLE[arm],
                     marker=cpu.PLACEMENT_MARKER[arm], ms=5, lw=2.0)
        ax_time.errorbar(
            P, median, yerr=yerr(selected, "spmv_ms", "spmv_q1", "spmv_q3"),
            capsize=2, label=tag, zorder=4, **style)
        ax_speed.plot(P, median[0] / median, label=tag, zorder=4, **style)

    ax_time.set_yscale("log")
    core_axis(ax_time)
    ax_time.set_ylabel("median SpMV time (ms)")
    fs.panel_title(ax_time, "Time")
    fs.style_axes(ax_time)
    fs.legend(ax_time, loc="upper right")
    mark_better(ax_time, "down", loc="lower left")

    largest = max(num(naive, "P").max(), num(touched, "P").max())
    ax_speed.plot([1, largest], [1, largest], ls=":", color=cpu.C_GUIDE,
                  lw=1.5, zorder=2)
    ax_speed.axvline(12, color=cpu.C_CEILING, ls="--", lw=1.2, zorder=2)
    ax_speed.set_xlabel("threads P")
    ax_speed.set_ylabel(r"SpMV speedup  $T(1)/T(P)$")
    fs.panel_title(ax_speed, "Speedup")
    fs.style_axes(ax_speed)
    fs.label_value(ax_speed, largest, largest, "linear", dy=4.0, dx=-6.0,
                   ha="right", size=7.8, color=cpu.C_GUIDE)
    fs.label_value(ax_speed, 12, ax_speed.get_ylim()[0],
                   "engaged L3 covers\nthe SpMV working set", dy=8.0, dx=5.0,
                   ha="left", size=7.8, color=cpu.C_CEILING)
    mark_better(ax_speed, "up", loc="upper left")

    # The claim is that the arms coincide, so the reportable number is the
    # largest gap between them, not a headline speedup.
    by_arm = [{float(r["P"]): float(r["spmv_ms"]) for r in arm}
              for arm in (naive, touched)]
    shared = sorted(set(by_arm[0]) & set(by_arm[1]))
    widest = max(abs(by_arm[0][p] / by_arm[1][p] - 1.0) for p in shared)

    out = HERE / "scaling_ab.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")
    print(f"  the two arms never separate by more than {widest:.0%} "
          "at any thread count")


def _tier_crossing_P(selected, ws_key, threshold):
    """Interpolate the thread count at which a working set crosses a tier."""
    P, ws = num(selected, "P"), num(selected, ws_key)
    for index in range(1, len(P)):
        if ((ws[index - 1] - threshold) * (ws[index] - threshold) <= 0
                and ws[index] != ws[index - 1]):
            frac = (threshold - ws[index - 1]) / (ws[index] - ws[index - 1])
            return P[index - 1] + frac * (P[index] - P[index - 1])
    return None


def plot_weak(rows) -> None:
    """Weak-scaling efficiency per kernel, with each kernel's tier crossing."""
    arms = sorted({r["arm"] for r in rows})
    selected = subset(rows, arm=arms[0])
    if len(selected) < 2:
        print("(skip) scaling_weak.png - fewer than two points")
        return

    fig, ax = plt.subplots(figsize=(9.0, 5.5), constrained_layout=True)
    P = num(selected, "P")

    # The total only. Three curves that decay together crowded their own
    # labels at the right-hand end and said one thing three times; the
    # per-kernel detail that matters is where each leaves its L3 slice, which
    # the vertical rules below still carry.
    time = num(selected, "total_ms")
    efficiency = time[0] / time
    finals = {"total": efficiency[-1]}
    ax.plot(P, efficiency, color=cpu.KERNEL_COLOR["total"], lw=2.2,
            marker=cpu.KERNEL_MARKER["total"], ms=6, zorder=4)
    fs.label_series(ax, P[-1], efficiency[-1], "total solve",
                    cpu.KERNEL_COLOR["total"], dy=-14.0, dx=-6.0, ha="right",
                    size=8.8)

    ax.axhline(1.0, ls=":", color=cpu.C_GUIDE, lw=1.5, zorder=2)
    for kernel, name in KERNELS:
        crossing = _tier_crossing_P(selected, f"{kernel}_ws_mib", L3_SLICE_MIB)
        if crossing:
            ax.axvline(crossing, color=cpu.KERNEL_COLOR[name], ls="--",
                       lw=1.1, alpha=0.7, zorder=2)
            fs.label_value(ax, crossing, 0.0,
                           f"{name} leaves\nits L3 slice", dy=8.0, dx=4.0,
                           ha="left", size=7.4,
                           color=cpu.KERNEL_COLOR[name])

    # The grid at each point: odd-n rounding means N/P is only near-constant.
    for row in selected:
        fs.label_value(ax, float(row["P"]), 1.0, f"n={int(float(row['n']))}",
                       dy=6.0, size=6.8)

    ax.set_ylim(0.0, 1.18)
    core_axis(ax, "threads P   (problem grown so N/P is near constant)")
    ax.set_ylabel(r"weak-scaling efficiency  $T(1)/T(P)$")
    fs.style_axes(ax)
    fs.label_value(ax, P[0], 1.0, "ideal", dy=-13.0, dx=0.0, ha="center",
                   size=7.8, color=cpu.C_GUIDE)
    mark_better(ax, "up", at=(0.955, 0.62))

    out = HERE / "scaling_weak.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")
    print("  efficiency at full thread count: " + ", ".join(
        f"{name} {value:.0%}" for name, value in finals.items()))


def plot_mn(rows) -> None:
    """The Krylov dimension the solver actually needs as the grid grows."""
    rows = sorted(rows, key=lambda r: float(r["n"]))
    n, m = num(rows, "n"), num(rows, "avg_krylov")

    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    ax.plot(n, m, color=cpu.C_MEASURED, lw=2.0, marker="o", ms=6, zorder=4)
    ax.axhline(8, ls="--", color=cpu.C_GUIDE, lw=1.5, zorder=2)

    # The curve climbs the diagonal, leaving the upper left the only clear
    # block, so the series is named there rather than on itself. The rule's
    # name sits under it at mid-span, clear of both the curve and the arrow.
    fs.label_series(ax, n[0], m[-1], r"measured avg Krylov $m(n)$",
                    cpu.C_MEASURED, dy=0.0, dx=6.0, ha="left", size=8.8)
    fs.label_value(ax, n[len(n) // 3], 8.0, "fixed m (scaling study)",
                   dy=-14.0, dx=0.0, ha="center", size=8.2,
                   color=cpu.C_GUIDE)
    ax.set_xlabel(r"grid points per dimension  $n$")
    ax.set_ylabel(r"average Krylov dimension  $m$")
    fs.style_axes(ax)
    # Fewer Arnoldi steps is fewer global reductions per cycle.
    mark_better(ax, "down", loc="lower right")

    out = HERE / "scaling_mn.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")
    print(f"  m rises {m[0]:.1f} to {m[-1]:.1f} over n = {n[0]:.0f} to "
          f"{n[-1]:.0f}")


def plot_locality(rows) -> None:
    """Does the transition survive scatter, and does it survive rotation?"""
    schedules = [s for s in ("block", "cyclic", "rotate")
                 if any(r["sched"] == s for r in rows)]
    if not schedules:
        print("(skip) scaling_locality.png - no schedule rows")
        return

    # Bytes per run under the harness's cold-cache model.
    first = rows[0]
    nnz, augmented = float(first["nnz"]), float(first["aug_N"])
    per_spmv = nnz * 12 + (augmented + 1) * 4 + augmented * 16
    per_run = per_spmv * float(first["ei_steps"]) * float(first["m"])

    fig, (ax_speed, ax_band) = plt.subplots(
        1, 2, figsize=(12.4, 5.2), constrained_layout=True)

    peaks, endpoints = {}, {}
    for schedule in schedules:
        selected = subset(rows, sched=schedule)
        color = cpu.LAYOUT_COLOR[schedule]
        P, time = num(selected, "P"), num(selected, "spmv_ms")
        speedup = time[0] / time
        bandwidth = per_run / (time / 1e3) / 1e9
        peaks[schedule] = (speedup[-1], bandwidth[-1])
        ax_speed.plot(P, speedup, color=color, lw=2.0,
                      marker=cpu.LAYOUT_MARKER[schedule], ms=5, zorder=4)
        ax_band.plot(P, bandwidth, color=color, lw=2.0,
                     marker=cpu.LAYOUT_MARKER[schedule], ms=5, zorder=4)
        endpoints[schedule] = (P[-1], speedup[-1], bandwidth[-1], color)

    largest = max(num(subset(rows, sched=schedules[0]), "P"))
    ax_speed.plot([1, largest], [1, largest], ls=":", color=cpu.C_GUIDE,
                  lw=1.2, zorder=2, label="linear speedup")
    ax_speed.axvline(12, color=cpu.C_CEILING, ls="--", lw=1.0, zorder=2)
    ax_speed.set_yscale("log")
    core_axis(ax_speed)
    ax_speed.set_ylabel(r"SpMV speedup  $T(1)/T(P)$")
    fs.style_axes(ax_speed)
    # Cyclic and rotate converge to within a few percent at the right-hand
    # end, so their names go in a key rather than on curves they would sit on.
    # The ideal guide joins them there rather than being labeled on itself.
    fs.legend(ax_speed, loc="upper left", handles=[
        plt.Line2D([], [], color=cpu.LAYOUT_COLOR[s], lw=2.0,
                   marker=cpu.LAYOUT_MARKER[s], ms=5, label=s)
        for s in schedules
    ] + [plt.Line2D([], [], color=cpu.C_GUIDE, ls=":", lw=1.2,
                    label="linear speedup")])
    fs.label_value(ax_speed, 12, ax_speed.get_ylim()[0],
                   "aggregate L3 covers\nthe SpMV working set", dy=8.0,
                   dx=-5.0, ha="right", size=7.4, color=cpu.C_CEILING)
    mark_better(ax_speed, "up", loc="lower right")

    # Crossing the socket DRAM line is what proves the traffic is L3-sourced.
    ax_band.axhline(DRAM_SOCKET, color=cpu.C_CEILING, ls="--", lw=1.3,
                    zorder=2)
    ax_band.set_yscale("log")
    core_axis(ax_band)
    ax_band.set_ylabel("implied aggregate SpMV bandwidth (GB/s)")
    fs.style_axes(ax_band)
    fs.label_value(ax_band, 1.0, DRAM_SOCKET,
                   f"socket DRAM ceiling, ~{DRAM_SOCKET:.0f} GB/s", dy=5.0,
                   dx=0.0, ha="left", size=7.8, color=cpu.C_CEILING)
    mark_better(ax_band, "up", loc="lower right")

    out = HERE / "scaling_locality.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")
    for schedule, (speedup, bandwidth) in peaks.items():
        print(f"  {schedule}: {speedup:.1f}x at full P, "
              f"{bandwidth:.0f} GB/s implied")


def plot_reduction(rows) -> None:
    """Gram-Schmidt wall clock, tree reduction against the O(P) scan.

    A null result, and a load-bearing one. R_h's denominator is defined on the
    baseline, so an inflated MGS reduction would have handed the s-step arm a
    win it did not earn. Replacing the primitive with a measurably faster one
    leaves the kernel where it was, so the roll-off is not the primitive.
    """
    if not {"tree", "linear"} <= {r.get("reduce") for r in rows}:
        print("(skip) scaling_reduction.png - needs REDUCE='tree linear'")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0),
                             constrained_layout=True, sharey=True)

    arms = (("linear", cpu.VARIANT_COLOR["no-halo"], (0, (4, 2)),
             "O(P) redundant scan"),
            ("tree", cpu.KERNEL_COLOR["Gram-Schmidt"], "-", "tree reduction"))
    inflation = {}
    for ax, n in zip(axes, (31, 61)):
        curves = {}
        for reduce, color, linestyle, label in arms:
            selected = subset(rows, n=n, arm="B", reduce=reduce)
            if not selected:
                continue
            P = num(selected, "P")
            curves[reduce] = (P, num(selected, "gs_ms"))
            ax.errorbar(P, num(selected, "gs_ms"),
                        yerr=yerr(selected, "gs_ms", "gs_q1", "gs_q3"),
                        color=color, ls=linestyle, marker="o", ms=4,
                        capsize=2, label=label, zorder=4)
        if {"tree", "linear"} <= set(curves):
            P, tree = curves["tree"]
            _, linear = curves["linear"]
            ax.fill_between(P, tree, linear,
                            color=cpu.VARIANT_COLOR["no-halo"], alpha=0.12,
                            lw=0, zorder=1)
            # Reported as a percentage gap, because the finding is that there
            # is barely one. A ratio printed as "0.9x" reads as a speedup the
            # wrong way round.
            inflation[n] = float(max(abs(linear / tree - 1.0)))
            # The band is thin, which is the point, so a label centred in it
            # lands on both curves. It goes to whichever side the curve is
            # leaving: above where the tail rises, below where it still falls.
            rising = tree[-1] >= tree[-2]
            fs.label_value(
                ax, P[-1],
                max(tree[-1], linear[-1]) if rising
                else min(tree[-1], linear[-1]),
                f"within {inflation[n]:.0%}",
                dy=12.0 if rising else -14.0, dx=-4.0, ha="right", size=8.4,
                weight="semibold", color=cpu.C_MUTED)

        fs.panel_title(ax, f"n = {n}")
        core_axis(ax, "threads P")
        ax.set_yscale("log")
        fs.style_axes(ax)
        if ax is axes[0]:
            ax.set_ylabel("Gram-Schmidt time (ms, median)")
            fs.legend(ax, loc="lower left")
        mark_better(ax, "down", loc="upper right")

    out = HERE / "scaling_reduction.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")
    for n, value in sorted(inflation.items()):
        print(f"  n={n}: the two primitives agree to within {value:.0%} "
              "at every thread count")


def main() -> None:
    made = False
    if STRONG_CSV.exists():
        rows = load_csv(STRONG_CSV)
        plot_strong(rows)
        plot_ab(rows)
        plot_reduction(rows)
        made = True
    else:
        print(f"(skip) {STRONG_CSV} not found")

    for path, draw, script in (
        (WEAK_CSV, plot_weak, "scaling_weak.sh"),
        (MN_CSV, plot_mn, "scaling_mn.sh"),
        (LOCALITY_CSV, plot_locality, "scaling_locality.sh"),
    ):
        if path.exists():
            draw(load_csv(path))
            made = True
        else:
            print(f"(skip) {path.name} not found - run scripts/scaling/{script}")

    if not made:
        print("No input CSVs found. Run the sweep scripts on puffin first.")


if __name__ == "__main__":
    main()
