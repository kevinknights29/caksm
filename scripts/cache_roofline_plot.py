"""
Cache-aware roofline for puffin (AMD Threadripper 3960X, AVX2),
n=31 (left) and n=61 (right).

Left panel:  n=31 - SpMV resides in L3 cache; problem is cache-resident.
Right panel: n=61 - SpMV spills to DRAM; the performance gap between the
             measured point and the L3 roof at the same arithmetic intensity
             marks the opportunity for communication-avoiding (CA) matrix powers.

Data transcribed from profiler runs (basket option, rainbow is within ~1%
and tells the same story).
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

OUTPUT = Path(__file__).resolve().parent / "cache_roofline.png"

# Single-core FP64 FMA-loop ceiling (GFLOP/s, AVX2, puffin)
COMPUTE_PEAK_GFLOPS = 68.1
HW_LABEL = "AMD Threadripper 3960X (puffin, AVX2)"

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

TIER_COLOR   = {"DRAM": "#993C1D", "L3": "#0F6E56", "L2": "#185FA5"}
KERNEL_COLOR = {"SpMV": "#D85A30", "Gram-Schmidt": "#534AB7", "dense expm": "#7A7A73"}
C_PEAK       = "#5F5E5A"
C_CA_ARROW   = "#1D9E75"
C_CA_LABEL   = "#147a59"


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

    # n=61 only: annotate the CA headroom for SpMV.
    # SpMV spills to DRAM. At the same arithmetic intensity the L3 roof is
    # 2.8 times higher (BW_L3 / BW_DRAM = 58.19 / 20.91 = 2.8).
    # That gap is the headroom CA targets
    if n == 61:
        s_ai, s_gf, _ = run["kernels"]["SpMV"]
        l3_roof_at_spmv = min(tiers["L3"] * s_ai, peak)
        dram_roof_at_spmv = min(tiers["DRAM"] * s_ai, peak)
        multiplier = l3_roof_at_spmv / dram_roof_at_spmv

        ax.annotate(
            "",
            xy=(s_ai, l3_roof_at_spmv),
            xytext=(s_ai, s_gf),
            arrowprops=dict(arrowstyle="-|>", color=C_CA_ARROW, lw=2.2),
        )
        label_y = (s_gf * l3_roof_at_spmv) ** 0.5  # geometric midpoint
        ax.annotate(
            f"CA opportunity\n" + rf"($\times {multiplier:.1f}$ to L3 roof)",
            xy=(s_ai, label_y),
            textcoords="offset points", xytext=(14, 0),
            fontsize=8.5, color=C_CA_LABEL, va="center",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOPs / byte)", fontsize=9)
    ax.set_ylabel("performance (GFLOP/s)", fontsize=9)
    ax.set_title(f"$n = {n}$", fontsize=11)
    ax.set_xlim(ai.min(), ai.max())
    ax.set_ylim(0.25, peak * 1.6)
    ax.grid(True, which="both", ls="-", lw=0.3, alpha=0.35)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.92)


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

    fig.suptitle(
        f"Cache-aware roofline - {HW_LABEL}",
        fontsize=12,
    )

    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
