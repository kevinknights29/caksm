"""
Side-by-side comparison of STREAM Triad bandwidth scaling:
  left -> SSE (default compiler flags)
  right -> AVX2 (-march=native, Ryzen Threadripper 3960X)
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

from figstyle import mark_better

OUTPUT = Path(__file__).resolve().parent / "stream_sweep.png"

# Thread count mapped to a list of Triad results (MB/s) across repetitions. Run
# stream_sweep.sh and paste the numbers here, one list entry per run; thread counts
# match the sweep: 1, 2, 4, 8, 12, 16, 24, 48.
# SSE (default flags, no -march=native)
SSE_SWEEPS = {
    #     run1       run2       run3
    1:  [22796.3,  22796.8,  22791.2],
    2:  [26620.8,  26628.5,  26609.3],
    4:  [26486.6,  26529.6,  26470.8],
    8:  [30402.5,  30385.9,  30394.0],
    12: [36868.7,  36882.9,  36878.3],
    16: [38381.6,  38381.6,  38385.9],
    24: [39346.2,  39298.4,  39274.8],
    48: [38655.4,  38789.8,  38758.5],
}
# AVX2 (-march=native, AMD Ryzen Threadripper 3960X)
AVX2_SWEEPS = {
    #     run1       run2       run3
    1:  [22642.3,  22623.8,  22557.2],
    2:  [26713.3,  26734.1,  26703.7],
    4:  [26468.1,  26479.0,  26517.6],
    8:  [30436.7,  30443.3,  30444.9],
    12: [36861.8,  36846.0,  36918.0],
    16: [38384.6,  38368.1,  38341.1],
    24: [39469.1,  39434.0,  39413.4],
    48: [38816.8,  38787.2,  38787.8],
}

# Hardware reference: AMD Ryzen Threadripper 3960X, single socket.
# 4 channels x DDR4-3200 (25.6 GB/s/ch) = 102.4 GB/s theoretical peak.
THEORETICAL_ROOF_GBS = 102.4
THEORETICAL_LABEL = "1-socket theoretical (4 ch DDR4-3200)"
# Bandwidth peaks at 24 threads (39.5 GB/s) and drops ~0.7 GB/s at 48 (SMT
# adds nothing on a memory-bound kernel); 24 is the saturation knee.
KNEE_THREADS = 24
C_DATA, C_IDEAL, C_THEO, C_ROOF = "#185FA5", "#888780", "#993C1D", "#0F6E56"


def to_gbs(mbs: list[float]) -> np.ndarray:
    """STREAM prints MB/s (1 MB = 1e6 B). Convert to GB/s."""
    return np.asarray(mbs, dtype=float) / 1000.0


def draw_subplot(ax: plt.Axes, sweeps: dict, title: str) -> float:
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
    ax.axhline(roof, ls=":", lw=1.3, color=C_ROOF,
               label=rf"measured roof ($\approx {roof:.0f}$ GB/s)")

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("OpenMP threads")
    ax.set_ylabel("Aggregate STREAM Triad bandwidth (GB/s)")
    ax.set_xlim(0, threads.max() * 1.05)
    ax.set_ylim(0, THEORETICAL_ROOF_GBS * 1.08)
    ax.set_xticks(thread_counts)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True, ls="-", lw=0.4, alpha=0.4)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    # The empty band between the measured roof and the theoretical one: the top-left
    # corner belongs to the ideal-linear line, which climbs off-axes within 4 threads.
    mark_better(ax, "up", at=(0.955, 0.6))
    return roof


def main() -> None:
    fig, (ax_sse, ax_avx) = plt.subplots(1, 2, figsize=(14.4, 5.0),
                                          sharey=True, constrained_layout=True)

    roof_sse = draw_subplot(ax_sse, SSE_SWEEPS, "SSE")
    roof_avx = draw_subplot(ax_avx, AVX2_SWEEPS, "AVX2 ($-$march=native)")

    ax_avx.set_ylabel("")

    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUTPUT}")
    # The caption's numbers: vectorization buys nothing on a memory-bound kernel.
    print(f"measured roof: SSE {roof_sse:.1f} GB/s, AVX2 {roof_avx:.1f} GB/s "
          f"({roof_avx / roof_sse:.3f}x) against a {THEORETICAL_ROOF_GBS:.1f} GB/s "
          f"theoretical peak; saturation knee at {KNEE_THREADS} threads")


if __name__ == "__main__":
    main()
