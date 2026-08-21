"""The vertical crossover: does cache-blocked matrix-powers pay above R_v = 1?

The no-halo arm is a control, not a fault: it skips the halo entirely, so it
computes the wrong basis and is reported for throughput only. Comparing it with
the real tiled arm is what separates the cost of redundant halo arithmetic from
a ceiling that cache residency cannot move.

Produces regime_sweep_mechanism.png.

Source: data/regime/regime_sweep_nohalo.csv, produced by
scripts/regime/regime_sweep.sh.

  uv run scripts/plots/regime_sweep_plot.py
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
import matplotlib.ticker as ticker

import figstyle as fs
import cpu_figstyle as cpu
from figstyle import mark_better

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data" / "regime"
CSV_NOHALO = DATA / "regime_sweep_nohalo.csv"
DPI = 300

VARIANTS = (
    ("gflops_baseline", "baseline", cpu.VARIANT_COLOR["baseline"], "-"),
    ("gflops_tiled", "real tiled (with halo)", cpu.VARIANT_COLOR["tiled"], "-"),
    # Named as the control it is. "Wrong basis" is accurate but reads as a
    # defect in the kernel, when the point is that this arm deliberately skips
    # the halo so the halo's cost can be priced.
    ("gflops_nohalo", "no-halo control (throughput only)",
     cpu.VARIANT_COLOR["no-halo"], (0, (4, 2))),
)


def load(path):
    """Read one sweep CSV, or report it missing and return None."""
    if not path.exists():
        print(f"(skip) {path} not found - run scripts/regime/regime_sweep.sh")
        return None
    with path.open() as handle:
        return list(csv.DictReader(handle))


def plot_mechanism(rows):
    """Baseline, tiled and no-halo throughput, split by the tier tiled to."""
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0),
                             constrained_layout=True, sharey=True)

    verdicts = {}
    for ax, level in zip(axes, ("L2", "L3")):
        selected = [r for r in rows if r["tile_level"] == level]
        selected.sort(key=lambda r: float(r["rv_1core"]))
        if not selected:
            ax.set_visible(False)
            continue
        rv = np.array([float(r["rv_1core"]) for r in selected])

        # One hue per variant, the same hue in both panels. Taking the tiled
        # arm's color from the panel's tier, as this figure once did, painted
        # one series green on the left and blue on the right.
        for field, label, color, linestyle in VARIANTS:
            ax.plot(rv, np.array([float(r[field]) for r in selected]),
                    marker="o", ms=4, lw=1.5, ls=linestyle, color=color,
                    label=label, zorder=4)

        # The gap the halo costs, which is what the diagnostic is for.
        tiled = np.array([float(r["gflops_tiled"]) for r in selected])
        nohalo = np.array([float(r["gflops_nohalo"]) for r in selected])
        ax.fill_between(rv, tiled, nohalo, color=cpu.VARIANT_COLOR["no-halo"],
                        alpha=0.12, lw=0, zorder=1)
        recovery = float(np.median(nohalo / tiled))
        verdicts[level] = recovery

        fs.panel_title(ax, f"Tiled to {level}")
        ax.set_xscale("log")
        # The L3 panel spans well under a decade, so the decade ticks fall
        # outside it entirely and it would carry no labels at all. Ticking on
        # the leading digits puts labels inside every range, and plain numbers
        # keep 1.4x10^0 off an axis that never leaves single figures.
        ax.xaxis.set_major_locator(
            ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0, 7.0),
                              numticks=12))
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xlabel(r"$R_v$ (1 core)")
        fs.style_axes(ax)
        if ax is axes[0]:
            ax.set_ylabel("GFLOP/s")
            fs.legend(ax, loc="upper left")
            mark_better(ax, "up", loc="lower right")

    out = HERE / "regime_sweep_mechanism.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")
    for level, recovery in verdicts.items():
        print(f"  {level}: removing the halo moves throughput {recovery:.2f}x")


def main():
    nohalo = load(CSV_NOHALO)
    if nohalo:
        plot_mechanism(nohalo)
    else:
        print("Nothing to plot.")


if __name__ == "__main__":
    main()
