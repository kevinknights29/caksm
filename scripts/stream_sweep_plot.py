"""
Side-by-side comparison of STREAM Triad bandwidth scaling:
  left  → SSE (default compiler flags)
  right → AVX-512 (-march=native)
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
import matplotlib.ticker as mticker

OUTPUT = Path(__file__).resolve().parent / "stream_sweep.png"


# DATA: thread count -> list of Triad results (MB/s) across repetitions.
# SSE (default flags, no -march=native)
SSE_SWEEPS = {
    #     run1        run2        run3
    1:  [11837.0,   13152.6,    10962.7],
    2:  [21791.0,   25053.5,    21782.1],
    4:  [39120.5,   41024.3,    38448.0],
    8:  [66480.4,   68665.7,    63831.7],
    16: [98696.3,  102292.8,    98139.2],
    30: [97059.5,   99808.9,    97803.5],
}
# AVX-512 (-march=native)
AVX512_SWEEPS = {
    #     run1        run2        run3
    1:  [14335.8,   13244.6,    13721.0],
    2:  [27031.5,   26504.4,    27848.4],
    4:  [46458.6,   44730.9,    47742.1],
    8:  [77662.7,   74110.9,    76108.4],
    16: [109323.9, 108844.0,   108572.8],
    30: [118442.7, 117325.9,   110213.3],
}

# Hardware reference: Intel Xeon Platinum 8358, single socket.
# 8 channels x DDR4-3200 (25.6 GB/s/ch) = 204.8 GB/s theoretical peak.
THEORETICAL_ROOF_GBS = 204.8
THEORETICAL_LABEL = "1-socket theoretical (8 ch DDR4-3200)"
KNEE_THREADS = 16
C_DATA, C_IDEAL, C_THEO, C_ROOF = "#185FA5", "#888780", "#993C1D", "#0F6E56"


def to_gbs(mbs: list[float]) -> np.ndarray:
    """STREAM prints MB/s (1 MB = 1e6 B). Convert to GB/s."""
    return np.asarray(mbs, dtype=float) / 1000.0


def draw_subplot(ax: plt.Axes, sweeps: dict, title: str) -> None:
    thread_counts = sorted(sweeps)
    threads = np.array(thread_counts, dtype=float)
    reps = [to_gbs(sweeps[t]) for t in thread_counts]

    best = np.array([r.max() for r in reps])
    lo   = np.array([r.min() for r in reps])
    hi   = np.array([r.max() for r in reps])
    ideal = best[0] * threads

    ax.fill_between(threads, lo, hi, color=C_DATA, alpha=0.15,
                    label="min-max across repetitions")
    ax.plot(threads, best, "-o", color=C_DATA, lw=2.2, ms=6,
            label="measured (best of reps)")
    ax.plot(threads, ideal, "--", color=C_IDEAL, lw=1.4, label="ideal linear")
    ax.axhline(THEORETICAL_ROOF_GBS, ls="--", lw=1.3, color=C_THEO,
               label=THEORETICAL_LABEL)

    mask = threads >= (KNEE_THREADS if KNEE_THREADS else threads.max())
    roof = best[mask].max()
    ax.axhline(roof, ls=":", lw=1.3, color=C_ROOF)
    ax.text(threads.max(), roof + 4, rf"measured roof $\approx {roof:.0f}$ GB/s",
            ha="right", va="bottom", color=C_ROOF, fontsize=9)

    if KNEE_THREADS in thread_counts:
        ki = thread_counts.index(KNEE_THREADS)
        ax.annotate(rf"knee $\approx {KNEE_THREADS}$ threads",
                    xy=(threads[ki], best[ki]),
                    xytext=(threads[ki], best[ki] - 30),
                    ha="center", color=C_THEO, fontsize=9,
                    arrowprops=dict(arrowstyle="->", color=C_THEO, lw=1.2))

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("OpenMP threads")
    ax.set_ylabel("Aggregate STREAM Triad bandwidth (GB/s)")
    ax.set_xlim(0, threads.max() * 1.05)
    ax.set_ylim(0, THEORETICAL_ROOF_GBS * 1.08)
    ax.set_xticks(thread_counts)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True, ls="-", lw=0.4, alpha=0.4)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)


def main() -> None:
    fig, (ax_sse, ax_avx) = plt.subplots(1, 2, figsize=(14.4, 5.0),
                                          sharey=True, constrained_layout=True)

    draw_subplot(ax_sse, SSE_SWEEPS,
                 "STREAM Triad Bandwidth Scaling (SSE)")
    draw_subplot(ax_avx, AVX512_SWEEPS,
                 "STREAM Triad Bandwidth Scaling (AVX-512, –march=native)")

    ax_avx.set_ylabel("")

    fig.suptitle("STREAM Triad: SSE vs AVX-512 Memory Bandwidth", fontsize=13,
                 fontweight="bold")

    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
