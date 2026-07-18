/**
 * @file machine.hpp
 * @brief A-priori machine parameters for the regime map's dimensionless coordinates.
 *
 * Everything here is a predictor input known before a run, from the machine's hardware
 * or a one-off offline calibration. Nothing here is derived from a timed run,
 * which is what keeps R_h from becoming a measured runtime fraction.
 *
 * Preset set to puffin. The table mirrors src/profiler.cpp's HWPreset entry so the
 * byte models and roofline figures stay comparable. Adding a machine
 * means measuring it: the hardware fields are transcription, but the three reduction
 * parameters must come from scripts/calibrate_alpha.sh run on that host, and
 * reduction_calibrated keeps an untested one from being trusted.
 *
 * @author Kevin Knights
 * @date 2026-07-10
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

/**
 * @brief A-priori hardware parameters for one compute environment.
 *
 * @note The three reduction parameters are the only entries not from a datasheet: they
 *       come from scripts/calibrate_alpha.sh, measured once offline. See reduction_cost_s.
 */
struct Machine {
    std::string_view key;
    std::string_view name;
    int     cores;                ///< physical cores engaged at full P
    int     cores_per_llc_slice;  ///< cores sharing one last-level slice; 3 on a Zen 2 CCX
    int64_t llc_slice_bytes;      ///< capacity of ONE last-level slice
    /// Capacity of ONE core's private L2. Not a last-level cache (an L2 miss is served by
    /// L3 and generates no DRAM traffic), so it never appears in R_v; carried only to size
    /// the finer of the two matrix-powers tiling levels.
    int64_t l2_bytes_core;
    double  peak_gflops_core;     ///< sustained single-core FP64 peak, fma-loop measured
    /// Socket DRAM bandwidth ceiling. A socket figure: does not scale with P.
    double  dram_bw_gbs;
    /// Bandwidth of ONE last-level slice. Scales with the team (each engaged CCX brings its
    /// own slice) but per SLICE, not per core: the roofline probe ran one core against a
    /// whole 16 MiB slice, so reading it as per-core and multiplying by P triple-counts it.
    /// Same accounting frame as R_v's denominator.
    double  l3_bw_gbs_slice;
    /// Cost of one reduction-tree level whose partner sits in the SAME cache domain.
    double  t_level_intra_s;
    /// Cost of one reduction-tree level that CROSSES a cache domain. The ratio to
    /// t_level_intra_s is the crossing multiplier the portability claim rests on.
    double  t_level_cross_s;
    /// L*: crossings past which subtree concurrency hides the cost. See reduction_cost_s.
    double  cross_saturation;
    /// Whether the three reduction parameters were measured on this machine. False means
    /// they are another host's figures, carried so the preset is constructible.
    bool    reduction_calibrated;
};

/**
 * Last-level cache is the level whose miss generates DRAM traffic, so L3.
 *
 * The reduction parameters replaced an earlier alpha*h*log2(P) model, refuted by
 * calibration on puffin's tree: levels are not equal. Thread 0's partners cost ~121 ns
 * while they share its CCX and ~653 ns once they do not (a 5.4x step at the L3-slice
 * boundary), and past L* ~ 2.2 crossings subtree concurrency hides the rest, so an extra
 * level of depth is effectively free. A CCX boundary is a cache-domain crossing, which
 * makes the topology term calibratable here even on a single NUMA node.
 *
 * @see reduction_cost_s for what this does to R_h's scaling law.
 */
inline constexpr std::array<Machine, 1> kMachines {{
    {
        "amd-3960x",
        "AMD Ryzen Threadripper 3960X (puffin)",
        24,          // 24 physical cores
        3,           // 3 cores share one 16 MiB CCX slice
        16L << 20,   // 16 MiB L3 per CCX (128 MiB aggregate at P=24)
        512L << 10,  // 512 KiB private L2 per core (Zen 2)
        68.12,       // measured: fma-loop AVX2+FMA path, taskset -c 2
        95.0,        // socket DRAM ceiling (README roofline anchor)
        58.19,       // per-CCX-slice L3 bandwidth (roofline probe, 1 core / whole slice)
        // calibrate_alpha.sh, tree arm, 2026-07-17. Fit t = a_i*intra + a_x*min(cross,L*)
        // over the CCX-aligned grid: SSE 4.6e4 against 2.6e5 for a single fitted alpha.
        120.8e-9,    // intra-CCX tree level
        652.8e-9,    // cross-CCX level: 5.4x an intra one, the crossing multiplier
        2.20,        // L*: past ~2 crossings, subtree concurrency hides the rest
        true,        // measured on puffin itself
    },
}};

[[nodiscard]] inline const Machine& lookup_machine(std::string_view key)
{
    for (const auto& m : kMachines)
        if (m.key == key) return m;
    throw std::invalid_argument("Unknown machine: " + std::string(key)
        + ".  Valid keys: amd-3960x");
}

/**
 * @brief Number of last-level slices a P-thread team lights up under OMP_PROC_BIND=close.
 *
 * `close` fills one slice before engaging the next, so P=4 on puffin engages two
 * CCXes and commands 32 MiB, not 4 x (16/3) MiB.
 */
[[nodiscard]] inline int64_t engaged_llc_slices(const Machine& mc, int P) noexcept
{
    const int64_t cps = mc.cores_per_llc_slice;
    return (static_cast<int64_t>(P) + cps - 1) / cps;
}

/**
 * @brief Aggregate last-level cache commanded by P cores: the denominator of R_v.
 *
 * The spec writes this as P * c_slice, exact at the CCX-aligned points (P = 3, 6, ..., 24)
 * the sweep scripts use. Away from them, counting engaged slices is the honest figure.
 * Aggregate on both sides of the ratio, so P is never double-counted.
 */
[[nodiscard]] inline int64_t aggregate_llc_bytes(const Machine& mc, int P) noexcept
{
    return engaged_llc_slices(mc, P) * mc.llc_slice_bytes;
}

/**
 * @brief Levels of the reduction tree whose partner shares the root's cache domain.
 *
 * Under OMP_PROC_BIND=close the partner at distance d sits on slice floor(d/cores_per_slice),
 * so the level stays intra-domain exactly while d < cores_per_llc_slice.
 */
[[nodiscard]] inline int intra_domain_levels(const Machine& mc, int P) noexcept
{
    int n = 0;
    for (int d = 1; d < P; d <<= 1)
        if (d < mc.cores_per_llc_slice) ++n;
    return n;
}

/// Levels of the reduction tree whose partner sits in another cache domain.
[[nodiscard]] inline int cross_domain_levels(const Machine& mc, int P) noexcept
{
    int n = 0;
    for (int d = 1; d < P; d <<= 1)
        if (d >= mc.cores_per_llc_slice) ++n;
    return n;
}

/**
 * @brief Modeled cost of one global reduction: the numerator of R_h.
 *
 *     t_reduce(P) = t_intra * intra_levels(P) + t_cross * min(cross_levels(P), L*)
 *
 * A-priori machine parameters only, never a measured fraction of the method's runtime. At
 * P=1 there are no levels, so the cost is zero (measured: 9.6 ns, call overhead).
 *
 * This replaced alpha*h*log2(P), refuted by calibration: successive levels cost 538, 602,
 * 191, 65, -15 ns, so levels are not equal (crossings) and eventually free (saturation).
 * Because puffin's tree is only three cache-domain crossings deep and saturates beyond
 * P ~ 12, the numerator is nearly constant over the machine's range, giving R_h ~ P rather
 * than P log P. A machine with more domains or real NUMA hops brings the log2 term back.
 */
[[nodiscard]] inline double reduction_cost_s(const Machine& mc, int P) noexcept
{
    if (P <= 1) return 0.0;
    const double intra = static_cast<double>(intra_domain_levels(mc, P));
    const double cross = std::min(static_cast<double>(cross_domain_levels(mc, P)),
                                  mc.cross_saturation);
    return mc.t_level_intra_s * intra + mc.t_level_cross_s * cross;
}

/**
 * @brief The crossing multiplier: what one cache-domain boundary costs, in intra levels.
 *
 * A NUMA hop and a CCX boundary are both cache-domain crossings, so this is calibratable on
 * puffin even at one NUMA node. Measured at 5.4x, making the portability claim a proxy
 * measurement rather than an extrapolation.
 */
[[nodiscard]] inline double crossing_multiplier(const Machine& mc) noexcept
{
    return mc.t_level_intra_s > 0.0 ? mc.t_level_cross_s / mc.t_level_intra_s : 0.0;
}

/**
 * @brief Which bandwidth serves a kernel whose reuse window is `working_set` bytes.
 *
 * A working set below aggregate last-level cache generates no DRAM traffic, so residency
 * selects which roof a kernel is priced against, and the two roofs differ by more than an
 * order of magnitude at full P. This couples the two axes: the vertical coordinate picks
 * the horizontal coordinate's denominator. Pricing every kernel at the DRAM roof
 * unconditionally over-predicted the n=61/P=21 cycle by 6x (R_h 0.010 against 0.060).
 *
 * DRAM is a socket figure and does not scale with P; L3 does, because every engaged CCX
 * brings its own slice. Counted in SLICES on both sides, exactly as aggregate_llc_bytes is.
 */
[[nodiscard]] inline double memory_bw_gbs(const Machine& mc, int P, int64_t working_set) noexcept
{
    if (working_set <= aggregate_llc_bytes(mc, P))
        return mc.l3_bw_gbs_slice * static_cast<double>(engaged_llc_slices(mc, P));
    return mc.dram_bw_gbs;   // socket-limited: flat in P
}

/**
 * @brief Roofline-attainable aggregate FP64 rate [FLOP/s] at P cores.
 *
 * Compute scales with P; bandwidth scales with P only while the kernel is cache-resident.
 *
 * @param ai      arithmetic intensity [FLOP/byte] of THIS kernel, not of the cycle.
 * @param bw_gbs  the roof serving it, from memory_bw_gbs().
 */
[[nodiscard]] inline double attainable_flops(const Machine& mc, int P, double ai,
                                             double bw_gbs) noexcept
{
    const double compute_roof = mc.peak_gflops_core * static_cast<double>(P);
    const double memory_roof  = ai * bw_gbs;
    return std::min(compute_roof, memory_roof) * 1e9;
}
