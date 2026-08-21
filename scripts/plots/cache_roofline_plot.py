"""Cache-aware roofline for puffin, at both grids.

At n=31 the SpMV working set is L3-resident; at n=61 it spills to DRAM. The
vertical gap between the measured SpMV point and the L3 roof above it at the
same arithmetic intensity is the headroom communication-avoiding matrix powers
would have to recover, which is what the vertical study goes on to test.

Data transcribed from profiler runs on the basket option; rainbow is within
about a percent and tells the same story.

  uv run scripts/plots/cache_roofline_plot.py
"""
# /// script
# dependencies = [
#   "matplotlib",
#   "numpy",
# ]
# ///

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import figstyle as fs
import cpu_figstyle as cpu
from figstyle import mark_better

OUTPUT = Path(__file__).resolve().parent / "cache_roofline.png"

# Single-core FP64 FMA-loop ceiling (GFLOP/s, AVX2, puffin)
COMPUTE_PEAK_GFLOPS = 68.1

# Per-n measured data.
#   tiers:   memory tier -> single-core bandwidth (GB/s)
#   kernels: name -> (arithmetic intensity [FLOPs/B], achieved [GFLOP/s], resident tier)
RUNS = {
    31: dict(
        tiers={"DRAM": 20.94, "L3": 72.69, "L2": 93.00},
        kernels={
            "SpMV":         (0.1528,  3.0236, "L3"),
            "Gram-Schmidt": (0.2271, 13.9134, "L3"),
            "dense expm":   (1.3892,  1.5852, "L2"),
        },
    ),
    61: dict(
        tiers={"DRAM": 20.91, "L3": 58.19, "L2": 88.60},
        kernels={
            "SpMV":         (0.1530,  2.1713, "DRAM"),
            "Gram-Schmidt": (0.2299,  6.2435, "L3"),
            "dense expm":   (1.7631,  0.3505, "L2"),
        },
    ),
}

TIER_COLOR = cpu.TIER_COLOR
KERNEL_COLOR = cpu.KERNEL_COLOR
C_PEAK = cpu.C_GUIDE
C_CA_ARROW = cpu.PINE


def draw_roofline(ax: plt.Axes, n: int) -> None:
    run   = RUNS[n]
    tiers = run["tiers"]
    peak  = COMPUTE_PEAK_GFLOPS

    ai = np.logspace(-1.4, 1.4, 400)  # 0.04, ..., ~25 FLOPs/byte

    # Memory roofs, sorted low-to-high bandwidth so DRAM plots first
    for tier, bw in sorted(tiers.items(), key=lambda kv: kv[1]):
        roof = np.minimum(bw * ai, peak)
        ax.plot(ai, roof, lw=2, color=TIER_COLOR[tier],
                label=f"{tier}  ({bw:.0f} GB/s)")

    # Compute ceiling
    ax.axhline(peak, lw=2, ls="--", color=C_PEAK,
               label=f"compute peak  ({peak:.0f} GFLOP/s)")

    # Kernel points
    for name, (k_ai, k_gf, tier) in run["kernels"].items():
        ax.scatter([k_ai], [k_gf], s=70, zorder=5,
                   color=KERNEL_COLOR[name], edgecolors="white", linewidths=0.8)
        ax.annotate(
            f"{name}\n(on {tier})", (k_ai, k_gf),
            textcoords="offset points", xytext=(9, 6),
            fontsize=8.5, color=KERNEL_COLOR[name],
        )

    # n=61 only: draw the CA headroom for SpMV. SpMV spills to DRAM, and at the same
    # arithmetic intensity the L3 roof is BW_L3/BW_DRAM higher; that gap is what CA
    # targets. The arrow is drawn, its size is printed -- the claim is the caption's.
    extra_keys = []
    if n == 61:
        s_ai, s_gf, _ = run["kernels"]["SpMV"]
        l3_roof_at_spmv = min(tiers["L3"] * s_ai, peak)
        dram_roof_at_spmv = min(tiers["DRAM"] * s_ai, peak)

        ax.annotate(
            "",
            xy=(s_ai, l3_roof_at_spmv),
            xytext=(s_ai, s_gf),
            arrowprops=dict(arrowstyle="-|>", color=C_CA_ARROW, lw=2.2),
        )
        extra_keys.append(Line2D([], [], color=C_CA_ARROW, lw=2.2,
                                 label="CA headroom (SpMV to L3 roof)"))
        print(f"n=61 CA headroom: x{l3_roof_at_spmv / dram_roof_at_spmv:.1f} "
              f"(L3 roof / DRAM roof at SpMV's arithmetic intensity)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOPs / byte)", fontsize=9)
    ax.set_ylabel("performance (GFLOP/s)", fontsize=9)
    fs.panel_title(ax, f"n = {n}")
    ax.set_xlim(ai.min(), ai.max())
    ax.set_ylim(0.25, peak * 1.6)
    fs.style_axes(ax)
    handles, _ = ax.get_legend_handles_labels()
    # Upper left is the only free corner, but the compute-peak roof runs flat
    # across the top of both panels and straight through the label text, so
    # the key sits on an opaque ground.
    fs.legend(ax, loc="upper left", handles=handles + extra_keys, opaque=True)
    mark_better(ax, "up", loc="lower right")


def main() -> None:
    fig, (ax_31, ax_61) = plt.subplots(
        1, 2,
        figsize=(13.0, 5.0),
        sharey=True,
        constrained_layout=True,
    )

    draw_roofline(ax_31, 31)
    draw_roofline(ax_61, 61)

    ax_61.set_ylabel("")  # shared y-axis, label only on the left panel

    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT.name}")


if __name__ == "__main__":
    main()
