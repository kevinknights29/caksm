"""
The vertical crossover: does cache-blocked matrix-powers pay above R_v=1?

Consumes:
  data/regime/regime_sweep.csv          (scripts/regime/regime_sweep.sh)  measured
  data/regime/regime_sweep_nohalo.csv   (scripts/regime/regime_sweep.sh)  diagnostic

Produces:
  regime_sweep.png            measured speedup vs R_v, plus a dashed roofline-projected
                              curve (baseline time / modeled traffic ratio). The gap
                              between the two lines is the finding: the model is not
                              wrong, the kernel falls short of it on this hardware.
  regime_sweep_mechanism.png  the no-halo diagnostic: baseline / real-tiled / no-halo
                              GFLOP/s side by side, split by tile level, showing L2
                              recovering and L3 staying flat.

Result: tiled/baseline speedup never approached the modeled traffic ratio at any swept
grid (0.8-1.4x measured against a 4-8x model). Two independent fixes (a fair shared inner
kernel, then breaking the row-dot's serial FMA chain) left the same diagnostic signature,
gathered by a debug kernel that skips the halo entirely (wrong basis, throughput only):

  L2 (small panels, fat halo)   removing the halo raises GFLOP/s back toward baseline.
                                The cache-blocking mechanism works: the panel is resident
                                and the shortfall is the redundant halo arithmetic traded
                                for DRAM traffic, a bad trade at these panel sizes rather
                                than a broken kernel.
  L3 (large panels, thin halo)  removing the halo barely moves GFLOP/s. Neither arm
                                reaches a bandwidth-bound regime; both are capped by
                                something insensitive to cache residency, most consistent
                                with gather latency on the indirect column access.

Nothing here is a counted DRAM measurement (puffin has no accessible uncore counters); the
projected curve is an explicit model claim, labeled as such, never presented as data.
"""
# /// script
# dependencies = ["matplotlib", "numpy"]
# ///
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

from figstyle import mark_better

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data" / "regime"
CSV_MEASURED = DATA / "regime_sweep.csv"
CSV_NOHALO   = DATA / "regime_sweep_nohalo.csv"

C_L3    = "#0F6E56"
C_L2    = "#534AB7"
C_SCAT  = "#D85A30"
C_LIM   = "#993C1D"
C_FAIL  = "#C0142C"
C_PROJ  = "#999999"
C_BASE  = "#5F5E5A"
C_TILE  = "#0F6E56"
C_NOHALO = "#C79A00"


def load(path):
    if not path.exists():
        print(f"(skip) {path} not found - run scripts/regime/regime_sweep.sh on puffin")
        return None
    with path.open() as f:
        return list(csv.DictReader(f))


def subset(rows, pattern, level):
    out = [r for r in rows if r["pattern"] == pattern and r["tile_level"] == level]
    out.sort(key=lambda r: float(r["rv_1core"]))
    return out


def plot_crossover(rows):
    fig, ax = plt.subplots(figsize=(8.4, 5.6), constrained_layout=True)

    arms = [("banded", "L3", C_L3, "o", "banded, tiled to L3"),
            ("banded", "L2", C_L2, "s", "banded, tiled to L2"),
            ("scattered", "L3", C_SCAT, "^", "scattered (no tiling)")]

    labelled_fail = labelled_proj = False
    for pattern, level, col, mk, lab in arms:
        s = subset(rows, pattern, level)
        if not s:
            continue
        rv = np.array([float(r["rv_1core"]) for r in s])
        sp = np.array([float(r["speedup"]) for r in s])
        ax.plot(rv, sp, "-", color=col, marker=mk, ms=5, lw=1.6, label=lab, zorder=3)

        # Roofline projection: what a kernel that hit exactly its modeled DRAM-traffic cut
        # would deliver. 1x where cache-resident (nothing to cut), the row's own
        # traffic_ratio where tiling is modeled to apply. Not a measurement; see the
        # module docstring and the title below.
        if pattern == "banded":
            proj = np.array([1.0 if float(r["rv_1core"]) < 1.0 else float(r["traffic_ratio"])
                             for r in s])
            ax.plot(rv, proj, "--", color=col, lw=1.1, alpha=0.55, zorder=2,
                    label="roofline-projected (model, not measured)" if not labelled_proj else None)
            labelled_proj = True

        fail = [i for i, r in enumerate(s) if r["ai_gate_pass"] == "0"]
        if fail:
            ax.scatter(rv[fail], sp[fail], s=130, facecolor="none",
                       edgecolor=C_FAIL, linewidth=1.7, zorder=5,
                       label="AI-gate fail (see mechanism figure)" if not labelled_fail else None)
            labelled_fail = True

    ax.axvline(1.0, color=C_LIM, ls="--", lw=1.3, zorder=1,
               label=r"$\theta_v = 1$ (working set fills the cache)")
    ax.axhline(1.0, color="#bbb", ls=":", lw=1.0, zorder=1,
               label="break-even (speedup = 1)")

    ax.set_xscale("log")
    ax.set_xlabel(r"$R_v$ (Arnoldi working set / one core's cache)")
    ax.set_ylabel("tiled / baseline speedup")
    ax.grid(alpha=0.25, lw=0.5, which="both")
    ax.legend(fontsize=8, loc="upper left")
    # The empty mid-right band: the measured curves hug break-even along the bottom and
    # the projected curve rides high, leaving the middle clear.
    mark_better(ax, "up", at=(0.955, 0.4))

    out = HERE / "regime_sweep.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")

    s = subset(rows, "banded", "L3")
    if s:
        worst = min(s, key=lambda r: abs(float(r["speedup"]) - 1.0))
        best_sp = max(float(r["speedup"]) for r in s)
        print(f"  measured: banded L3 speedup stays in [{min(float(r['speedup']) for r in s):.2f}, "
              f"{best_sp:.2f}]x across the whole R_v sweep -- never approaches the modelled "
              f"traffic ratio.")


def plot_mechanism(rows):
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True, sharey=True)

    verdict_lines = []
    for ax, level, col_tile in zip(axes, ("L3", "L2"), (C_L3, C_L2)):
        s = [r for r in rows if r["tile_level"] == level]
        s.sort(key=lambda r: float(r["rv_1core"]))
        if not s:
            ax.set_visible(False)
            continue
        rv = np.array([float(r["rv_1core"]) for r in s])
        gb = np.array([float(r["gflops_baseline"]) for r in s])
        gt = np.array([float(r["gflops_tiled"]) for r in s])
        gn = np.array([float(r["gflops_nohalo"]) for r in s])

        ax.plot(rv, gb, "-o", color=C_BASE, ms=4, lw=1.4, label="baseline")
        ax.plot(rv, gt, "-o", color=col_tile, ms=4, lw=1.4, label="real tiled (with halo)")
        ax.plot(rv, gn, "--o", color=C_NOHALO, ms=4, lw=1.4,
                label="no-halo (DEBUG, wrong basis)")

        # The residency verdict is the finding; print it for the caption rather than
        # asserting it in the panel title.
        verdicts = [r["cache_resident_verdict"] for r in s]
        resident = verdicts.count("1") > len(verdicts) / 2 if verdicts else False
        verdict_lines.append(
            f"  {level}: {'halo removal recovers throughput (mechanism works)' if resident else 'GFLOP/s insensitive to halo (gather-latency ceiling)'}")
        ax.set_title(f"tile-level = {level}", fontsize=10)
        ax.set_xscale("log")
        ax.set_xlabel(r"$R_v$ (1 core)")
        ax.grid(alpha=0.25, lw=0.5, which="both")
        ax.legend(fontsize=8, loc="best")
        mark_better(ax, "up", loc="lower right")

    axes[0].set_ylabel("GFLOP/s")

    out = HERE / "regime_sweep_mechanism.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")
    for line in verdict_lines:
        print(line)


def main():
    measured = load(CSV_MEASURED)
    if measured:
        plot_crossover(measured)
    nohalo = load(CSV_NOHALO)
    if nohalo:
        plot_mechanism(nohalo)
    if not measured and not nohalo:
        print("Nothing to plot.")


if __name__ == "__main__":
    main()
