"""
Figures for the OpenMP scaling study of the baseline KSM-EI solver (puffin,
AMD Threadripper 3960X, AVX2).

This script consumes:
    data/scaling/scaling_strong.csv   (scripts/scaling/scaling_strong.sh)
    data/scaling/scaling_weak.csv     (scripts/scaling/scaling_weak.sh)
    data/scaling/mn_growth.csv        (scripts/scaling/scaling_mn.sh)

Outputs (only those whose input CSV exists are produced):

  scaling_strong.png    per-kernel strong-scaling curves, n=31 and n=61, both init arms.
                        SpMV's bandwidth plateau/transition and MGS's synchronization
                        roll-off on one axis.
  scaling_ab.png        the init-strategy A/B contrast at n=61 SpMV: naive (flat /
                        DRAM-bound at all P) vs first-touch (transition).
  scaling_weak.png      per-kernel weak-scaling efficiency T(1)/T(P) vs P, with each
                        kernel's cache-tier crossing marked.
  scaling_mn.png        the quarantined m(n) Krylov-growth curve.
  scaling_reduction.png Gram-Schmidt wall-clock under the calibrated tree reduction vs the
                        O(P) redundant scan it replaced. The gap is how much of the old GS
                        roll-off was the primitive rather than the machine. Needs
                        REDUCE='tree linear' (the sweep's default).

Median markers carry inter-quartile-range error bars. Every figure except
scaling_reduction.png shows the tree arm only: `linear` is the historical artifact and is
not a result. See subset().
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

from figstyle import mark_better

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data" / "scaling"

STRONG_CSV   = DATA / "scaling_strong.csv"
WEAK_CSV     = DATA / "scaling_weak.csv"
MN_CSV       = DATA / "mn_growth.csv"
LOCALITY_CSV = DATA / "scaling_locality.csv"

# Single-core roofs (GB/s) and socket DRAM ceiling for the implied-bandwidth panel.
DRAM_1CORE   = 21.0
L3_1CORE     = 58.19   # single-core L3 at n=61 (roofline probe)
DRAM_SOCKET  = 95.0    # ~4ch DDR4-3200 achievable aggregate

L3_SLICE_MIB = 16.0  # one CCX L3 slice

# Palette (shared with cache_roofline_plot.py where possible)
C_SPMV = "#D85A30"
C_GS   = "#534AB7"
C_TOT  = "#7A7A73"
C_IDEAL = "#5F5E5A"
C_TIER  = "#993C1D"
C_ARM_A = "#B0483A"   # naive
C_ARM_B = "#0F6E56"   # first-touch


def load_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def num(rows, key):
    return np.array([float(r[key]) for r in rows])


def yerr(rows, med_key, q1_key, q3_key):
    med = num(rows, med_key)
    q1  = num(rows, q1_key)
    q3  = num(rows, q3_key)
    return np.vstack([np.maximum(med - q1, 0.0), np.maximum(q3 - med, 0.0)])


def subset(rows, **filt):
    """Filter rows, defaulting to the tree reduction arm.

    scaling_strong.sh sweeps both reduction primitives, so a CSV holds two rows per
    (arm, n, P); an unfiltered subset would interleave them into a zigzag that is neither
    curve. `tree` is the default because it is the only arm to report. Pass reduce=None to
    opt out; CSVs written before the column existed are all-tree by construction, so a
    missing column is not filtered on.
    """
    filt.setdefault("reduce", "tree")
    if filt["reduce"] is None or not rows or "reduce" not in rows[0]:
        filt.pop("reduce", None)
    out = []
    for r in rows:
        if all(str(r[k]) == str(v) for k, v in filt.items()):
            out.append(r)
    out.sort(key=lambda r: float(r["P"]))
    return out


# strong-scaling per-kernel curves, both arms, n=31 and n=61
def plot_strong(rows) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), sharey=False,
                             constrained_layout=True)

    for ax, n in zip(axes, (31, 61)):
        for arm, ls, tag in (("A", "--", "naive"), ("B", "-", "first-touch")):
            for kern, col, ck in (("spmv", C_SPMV, "SpMV"), ("gs", C_GS, "Gram-Schmidt")):
                s = subset(rows, n=n, arm=arm)
                if not s:
                    continue
                P = num(s, "P")
                med = num(s, f"{kern}_ms")
                err = yerr(s, f"{kern}_ms", f"{kern}_q1", f"{kern}_q3")
                ax.errorbar(P, med, yerr=err, ls=ls, color=col, lw=1.8,
                            marker="o", ms=4, capsize=2,
                            label=f"{ck} ({tag})")
        # CCX-boundary guides (every 3 physical cores = one more L3 slice)
        for p in (3, 6, 9, 12, 15, 18, 21, 24):
            ax.axvline(p, color="#cccccc", lw=0.5, zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("threads P  (physical cores; P/3 = L3 slices engaged)", fontsize=9)
        ax.set_ylabel("median kernel time (ms)", fontsize=9)
        ax.set_title(f"$n = {n}$", fontsize=11)
        ax.set_xticks([1, 3, 6, 12, 24])
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.grid(True, which="both", ls="-", lw=0.3, alpha=0.35)
        # Pinned: the curves decay left-to-right, so "best" would drop the legend into
        # the bottom-left corner the better-arrow occupies.
        ax.legend(fontsize=7.5, framealpha=0.92, ncol=2, loc="upper right")
        mark_better(ax, "down", loc="lower left")

    out = HERE / "scaling_strong.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")


# init-strategy A/B contrast at n=61 SpMV
def plot_ab(rows) -> None:
    a = subset(rows, n=61, arm="A")
    b = subset(rows, n=61, arm="B")
    if not a or not b:
        print("A/B contrast: need both arms at n=61; skipping.")
        return

    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(12.5, 5.0),
                                     constrained_layout=True)

    # Left: median SpMV time vs P.
    for s, col, tag in ((a, C_ARM_A, "arm A - naive master-thread init"),
                        (b, C_ARM_B, "arm B - parallel first-touch")):
        P = num(s, "P")
        med = num(s, "spmv_ms")
        err = yerr(s, "spmv_ms", "spmv_q1", "spmv_q3")
        ax_t.errorbar(P, med, yerr=err, color=col, lw=2.0, marker="o", ms=5,
                      capsize=2, label=tag)
    ax_t.set_xscale("log"); ax_t.set_yscale("log")
    ax_t.set_xticks([1, 3, 6, 12, 24])
    ax_t.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax_t.set_xlabel("threads P", fontsize=9)
    ax_t.set_ylabel("median SpMV time (ms)", fontsize=9)
    ax_t.grid(True, which="both", ls="-", lw=0.3, alpha=0.35)
    ax_t.legend(fontsize=8.5, framealpha=0.92)
    mark_better(ax_t, "down", loc="lower left")

    # Right: SpMV speedup vs P.  Arm A predicted flat (data pinned to one slice);
    # arm B predicted to transition once engaged L3 covers the 52.8 MiB working
    # set (ceil(52.8/16) = 4 slices -> P ~ 12 under close).
    for s, col, tag in ((a, C_ARM_A, "arm A - naive"),
                        (b, C_ARM_B, "arm B - first-touch")):
        P = num(s, "P")
        t = num(s, "spmv_ms")
        speedup = t[0] / t
        ax_s.plot(P, speedup, color=col, lw=2.0, marker="o", ms=5, label=tag)
    Pmax = max(num(a, "P").max(), num(b, "P").max())
    ax_s.plot([1, Pmax], [1, Pmax], ls=":", color=C_IDEAL, lw=1.5, label="ideal")
    ax_s.axvline(12, color=C_TIER, ls="--", lw=1.2,
                 label="engaged L3 covers SpMV set (P $\\approx$ 12)")
    ax_s.set_xlabel("threads P", fontsize=9)
    ax_s.set_ylabel("SpMV speedup  $T(1)/T(P)$", fontsize=9)
    ax_s.grid(True, ls="-", lw=0.3, alpha=0.35)
    ax_s.legend(fontsize=8.5, framealpha=0.92)
    mark_better(ax_s, "up", loc="upper left")

    out = HERE / "scaling_ab.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")


# Deliverable 2: weak-scaling per-kernel efficiency with tier crossings
def _tier_crossing_P(s, ws_key, threshold):
    """Linear-interpolate the P at which working set ws_key crosses threshold."""
    P  = num(s, "P")
    ws = num(s, ws_key)
    for i in range(1, len(P)):
        if (ws[i - 1] - threshold) * (ws[i] - threshold) <= 0 and ws[i] != ws[i - 1]:
            frac = (threshold - ws[i - 1]) / (ws[i] - ws[i - 1])
            return P[i - 1] + frac * (P[i] - P[i - 1])
    return None


def plot_weak(rows) -> None:
    # Weak sweep is a single arm (default B); pick whichever arm is present.
    arms = sorted({r["arm"] for r in rows})
    s = subset(rows, arm=arms[0])
    if len(s) < 2:
        print("Weak scaling: <2 points; skipping.")
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)

    P = num(s, "P")
    for kern, col, ck in (("spmv", C_SPMV, "SpMV"),
                          ("gs", C_GS, "Gram-Schmidt"),
                          ("total", C_TOT, "total")):
        t = num(s, f"{kern}_ms")
        eff = t[0] / t  # T(1)/T(P): decays from 1.0 as cores share a fixed B
        ax.plot(P, eff, color=col, lw=2.0, marker="o", ms=5, label=ck)

    ax.axhline(1.0, ls=":", color=C_IDEAL, lw=1.5, label="ideal (efficiency = 1)")

    # Tier crossings: working set crossing one 16 MiB L3 slice (per kernel). Named in
    # the legend rather than annotated in-axes; the caption reads the consequence.
    xcross = _tier_crossing_P(s, "spmv_ws_mib", L3_SLICE_MIB)
    if xcross:
        ax.axvline(xcross, color=C_SPMV, ls="--", lw=1.1,
                   label=f"SpMV leaves L3 slice (P $\\approx$ {xcross:.0f})")
    gcross = _tier_crossing_P(s, "gs_ws_mib", L3_SLICE_MIB)
    if gcross:
        ax.axvline(gcross, color=C_GS, ls="--", lw=1.1,
                   label=f"GS leaves L3 slice (P $\\approx$ {gcross:.0f})")

    # Annotate the grid size at each point (the honest seam of odd-n rounding
    # means N/P is only ~constant; aug_N and P are in the CSV for the exact ratio).
    for row in s:
        ax.annotate(f"n={int(float(row['n']))}",
                    xy=(float(row["P"]), 1.0), xytext=(0, 6),
                    textcoords="offset points",
                    fontsize=6.5, color="#777777", ha="center")

    ax.set_ylim(0.0, 1.15)
    ax.set_xlabel("threads P   (problem grown so N/P ~ constant)", fontsize=9)
    ax.set_ylabel("weak-scaling efficiency  $T(1)/T(P)$", fontsize=9)
    ax.grid(True, ls="-", lw=0.3, alpha=0.35)
    ax.legend(fontsize=8.5, framealpha=0.92, loc="lower left")
    # Both top corners carry the per-point n labels; park the arrow in the empty
    # right-hand band the decayed curves leave behind.
    mark_better(ax, "up", at=(0.955, 0.62))

    out = HERE / "scaling_weak.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")


# Deliverable 4: quarantined m(n) Krylov-growth curve
def plot_mn(rows) -> None:
    rows = sorted(rows, key=lambda r: float(r["n"]))
    n = num(rows, "n")
    m = num(rows, "avg_krylov")

    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    ax.plot(n, m, color=C_GS, lw=2.0, marker="o", ms=6,
            label="measured avg Krylov $m(n)$")
    ax.axhline(8, ls="--", color=C_IDEAL, lw=1.5,
               label="fixed $m = 8$ (scaling study)")
    ax.set_xlabel("grid points per dimension  $n$", fontsize=9)
    ax.set_ylabel("average Krylov dimension  $m$", fontsize=9)
    ax.grid(True, ls="-", lw=0.3, alpha=0.35)
    ax.legend(fontsize=9, framealpha=0.92, loc="upper left")
    # Lower m is fewer Arnoldi steps, hence fewer global reductions per cycle.
    mark_better(ax, "down", loc="lower right")

    out = HERE / "scaling_mn.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")

# Locality experiment: block vs cyclic vs rotate compute schedule at n=61. Does the
# transition survive scatter (cyclic) but vanish under time-varying assignment (rotate)?
# If so, the mechanism is per-CCX L3 capacity, not data layout, and the identical
# arm-A/arm-B curves become the control.
SCHED_STYLE = {
    "block":  (C_ARM_B, "-",  "block  (contiguous, locality-preserving)"),
    "cyclic": ("#B8860B", "--", "cyclic (scatter, same per-CCX volume)"),
    "rotate": (C_ARM_A, ":",  "rotate (per-SpMV band shift, locality-destroying)"),
}


def plot_locality(rows) -> None:
    scheds = [s for s in ("block", "cyclic", "rotate")
              if any(r["sched"] == s for r in rows)]
    if not scheds:
        print("Locality: no sched column / rows; skipping.")
        return

    # bytes/run for the implied-bandwidth panel (harness cold-cache model).
    r0 = rows[0]
    nnz = float(r0["nnz"]); augN = float(r0["aug_N"])
    bytes_per_spmv = nnz * 12 + (augN + 1) * 4 + augN * 16
    bytes_per_run = bytes_per_spmv * float(r0["ei_steps"]) * float(r0["m"])

    fig, (ax_s, ax_b) = plt.subplots(1, 2, figsize=(13.0, 5.2),
                                     constrained_layout=True)

    for sched in scheds:
        s = subset(rows, sched=sched)
        col, ls, tag = SCHED_STYLE[sched]
        P = num(s, "P")
        t = num(s, "spmv_ms")
        ax_s.plot(P, t[0] / t, color=col, ls=ls, lw=2.0, marker="o", ms=5, label=tag)
        aggbw = bytes_per_run / (t / 1e3) / 1e9  # GB/s
        ax_b.plot(P, aggbw, color=col, ls=ls, lw=2.0, marker="o", ms=5, label=tag)

    Pmax = max(num(subset(rows, sched=scheds[0]), "P"))
    # Left: SpMV speedup. Ideal + the predicted P=12 transition.
    ax_s.plot([1, Pmax], [1, Pmax], ls=":", color=C_IDEAL, lw=1.2, label="linear")
    ax_s.axvline(12, color=C_TIER, ls="--", lw=1.0,
                 label="aggregate L3 covers SpMV set (P$\\approx$12)")
    ax_s.set_xscale("log"); ax_s.set_yscale("log")
    ax_s.set_xticks([1, 3, 6, 12, 24])
    ax_s.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax_s.set_xlabel("threads P", fontsize=9)
    ax_s.set_ylabel("SpMV speedup  $T(1)/T(P)$", fontsize=9)
    ax_s.grid(True, which="both", ls="-", lw=0.3, alpha=0.35)
    ax_s.legend(fontsize=7.5, framealpha=0.92, loc="upper left")
    mark_better(ax_s, "up", loc="lower right")

    # Right: implied aggregate bandwidth vs the DRAM ceiling.  Crossing the socket
    # DRAM line proves the traffic is L3-sourced (block/cyclic); staying beneath it
    # proves rotate is DRAM-bound at all P (no tier transition).
    ax_b.axhline(DRAM_SOCKET, color=C_TIER, ls="--", lw=1.3,
                 label=f"socket DRAM ceiling (~{DRAM_SOCKET:.0f} GB/s)")
    ax_b.set_xscale("log"); ax_b.set_yscale("log")
    ax_b.set_xticks([1, 3, 6, 12, 24])
    ax_b.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax_b.set_xlabel("threads P", fontsize=9)
    ax_b.set_ylabel("implied aggregate SpMV bandwidth (GB/s)", fontsize=9)
    ax_b.grid(True, which="both", ls="-", lw=0.3, alpha=0.35)
    ax_b.legend(fontsize=7.5, framealpha=0.92, loc="upper left")
    mark_better(ax_b, "up", loc="lower right")

    out = HERE / "scaling_locality.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")


def plot_reduction(rows) -> None:
    """Gram-Schmidt wall-clock, tree reduction vs the O(P) scan it replaced.

    The correction, measured rather than asserted. MGS's global reductions used to run a
    barrier plus an O(P) scan performed redundantly on every thread; the gap between these
    curves is how much of the old GS roll-off was that primitive rather than the machine.

    It matters beyond tidiness: R_h's denominator is defined on the baseline, so an
    inflated MGS reduction would hand the s-step arm a win it did not earn.
    """
    have = {r.get("reduce") for r in rows}
    if not {"tree", "linear"} <= have:
        print("(skip) scaling_reduction.png - needs both arms: REDUCE='tree linear'")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)

    for ax, n in zip(axes, (31, 61)):
        for red, col, ls in (("linear", C_ARM_A, "--"), ("tree", C_ARM_B, "-")):
            s = subset(rows, n=n, arm="B", reduce=red)
            if not s:
                continue
            P = num(s, "P")
            ax.errorbar(P, num(s, "gs_ms"),
                        yerr=yerr(s, "gs_ms", "gs_q1", "gs_q3"),
                        color=col, ls=ls, marker="o", ms=4, capsize=2,
                        label=f"{red}" + (" (artifact)" if red == "linear" else ""))
        ax.set_title(f"$n = {n}$", fontsize=11)
        ax.set_xlabel("cores P")
        ax.set_ylabel("GS time (ms, median)")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.grid(alpha=0.3)
        ax.legend(title="reduction")
        mark_better(ax, "down", loc="lower left")

    # The inflation at full P: the number the write-up's caption needs. Printed, not
    # drawn -- it is a finding, and the figure states findings nowhere.
    t = subset(rows, n=61, arm="B", reduce="tree")
    l = subset(rows, n=61, arm="B", reduce="linear")
    if t and l:
        tg = {float(r["P"]): float(r["gs_ms"]) for r in t}
        lg = {float(r["P"]): float(r["gs_ms"]) for r in l}
        pmax = max(set(tg) & set(lg))
        if tg[pmax] > 0:
            print(f"scaling_reduction: the O(P) scan inflated GS by "
                  f"{lg[pmax] / tg[pmax]:.2f}x at P={pmax:.0f}, n=61")
    fig.savefig(HERE / "scaling_reduction.png", dpi=150)
    plt.close(fig)


def main() -> None:
    made = False
    if STRONG_CSV.exists():
        rows = load_csv(STRONG_CSV)
        plot_strong(rows)
        plot_ab(rows)
        plot_reduction(rows)
        made = True
    else:
        print(f"(skip) {STRONG_CSV} not found - run scripts/scaling/scaling_strong.sh")

    if WEAK_CSV.exists():
        plot_weak(load_csv(WEAK_CSV))
        made = True
    else:
        print(f"(skip) {WEAK_CSV} not found - run scripts/scaling/scaling_weak.sh")

    if MN_CSV.exists():
        plot_mn(load_csv(MN_CSV))
        made = True
    else:
        print(f"(skip) {MN_CSV} not found - run scripts/scaling/scaling_mn.sh")

    if LOCALITY_CSV.exists():
        plot_locality(load_csv(LOCALITY_CSV))
        made = True
    else:
        print(f"(skip) {LOCALITY_CSV} not found - run scripts/scaling/scaling_locality.sh")

    if not made:
        print("No input CSVs found. Run the sweep scripts on puffin first.")


if __name__ == "__main__":
    main()
