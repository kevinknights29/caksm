/**
 * @file gpu_machine.hpp
 * @brief A-priori GPU parameters for the regime map's dimensionless coordinates.
 *
 * The GPU twin of machine.hpp, and a separate type rather than an extension of `Machine`.
 * Every CPU field is meaningless here and every field here is meaningless on a CPU:
 * `llc_slice_bytes` presumes cache that grows with the team, `peak_gflops_core` presumes FP64
 * that scales per core, and `t_level_cross_s` presumes exactly two reduction tiers. Feeding
 * GPU numbers into those formulas gives a wrong answer that still looks plausible; a separate
 * type makes it a compile error instead.
 *
 * Three structural breaks are encoded here, not merely reparameterized:
 *
 *   1. L2 is one device-wide block. aggregate_l2_bytes() ignores P by design, so R_v stops
 *      being a function of how much of the GPU is engaged. See docs/regime_gpu_phase0.md.
 *   2. A reduction crosses up to five tiers, not two, and the grid tier carries a
 *      kernel-launch cost with no CPU analogue. Hence a tier vector and a separate launch term.
 *   3. The compute roof is a device figure that must be gated. Both map coordinates measure
 *      communication, so a compute-bound kernel is off-map rather than lower-left.
 *      roofline_gate() is that gate and runs before any point is placed.
 *
 * Provenance rule, inherited from the CPU side: hardware fields are datasheet transcription,
 * but every t_reduce tier and every achieved roof comes from an offline measurement on the
 * host itself (scripts/regime/calibrate_gpu.sh). `reduction_calibrated` and `roofline_gated`
 * keep an untested preset from emitting a confident magnitude.
 *
 * @author Kevin Knights
 * @date 2026-07-21
 */
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

/**
 * @brief The rungs of a GPU reduction, cheapest first.
 *
 * The CPU's two tiers and one saturation constant cannot express this ladder: the steps span
 * four orders of magnitude, and three of them are different mechanisms rather than the same
 * mechanism at a longer distance.
 *
 * The ladder is the instrument. Holding N and the device fixed and changing only the rung
 * places the same operator at different R_h, which turns theta_h from a shaded prediction
 * into a measured crossing.
 */
enum class ReductionTier : int {
    WARP       = 0,  ///< __shfl_down_sync, register-to-register. No memory traffic.
    BLOCK      = 1,  ///< shared memory + __syncthreads. On-chip.
    GRID       = 2,  ///< global atomics or a second kernel launch. Carries t_kernel_launch_s.
    DEVICE_P2P = 3,  ///< device-to-device. On synge: SYS, i.e. PCIe + cross-socket UPI.
    NODE       = 4,  ///< node-to-node over the cluster fabric.
};

inline constexpr int kTierCount = 5;

[[nodiscard]] inline constexpr const char* tier_name(ReductionTier t) noexcept
{
    switch (t) {
        case ReductionTier::WARP:       return "warp";
        case ReductionTier::BLOCK:      return "block";
        case ReductionTier::GRID:       return "grid";
        case ReductionTier::DEVICE_P2P: return "device-p2p";
        case ReductionTier::NODE:       return "node";
    }
    return "?";
}

[[nodiscard]] inline constexpr std::size_t tier_index(ReductionTier t) noexcept
{
    return static_cast<std::size_t>(static_cast<int>(t));
}

/// Which precision a kernel runs in. Selects the compute roof, and therefore the ridge the
/// roofline gate compares against, which is what the two-card contrast varies.
enum class Precision { FP64, FP32 };

[[nodiscard]] inline constexpr const char* precision_name(Precision p) noexcept
{
    return p == Precision::FP64 ? "fp64" : "fp32";
}

/**
 * @brief A-priori hardware parameters for one GPU environment.
 *
 * @note The fields marked measured are not datasheet values and must not be guessed. A preset
 *       whose tiers are unmeasured carries reduction_calibrated = false, which every consumer
 *       of R_h checks before printing a magnitude.
 */
struct GpuMachine {
    std::string_view key;
    std::string_view name;

    // Device geometry
    int     sm_count;                 ///< SMs on one device; this is P
    int     warps_per_sm;             ///< max resident warps; occupancy budget, not part of P
    int64_t l2_bytes;                 ///< device-wide L2. Does not scale with SMs engaged.
    int64_t shared_mem_bytes_per_sm;  ///< programmer-managed; trades against occupancy
    int64_t device_memory_bytes;      ///< HBM/GDDR capacity: the footprint ceiling

    // Roofs
    double fp64_flops_peak;           ///< whole-device FP64 [FLOP/s]
    double fp32_flops_peak;           ///< whole-device FP32 [FLOP/s]
    double hbm_bw_gbs;                ///< theoretical device memory bandwidth
    double hbm_bw_gbs_achieved;       ///< measured, gpu-stream triad. 0 = not measured.
    double l2_bw_gbs_achieved;        ///< measured, gpu-stream at an L2-resident size.

    // Topology
    int              gpu_count;            ///< devices per node
    int              node_count;           ///< nodes in the allocation
    std::string_view interconnect;         ///< "SYS", "NVLINK", "PIX", ... from nvidia-smi topo -m
    double           interconnect_bw_gbs;  ///< measured, p2p probe. 0 = not measured.

    /**
     * Measured incremental cost of each rung [s]. Entry i is what rung i adds to rung i-1, not
     * the total; reduction_cost_s() accumulates. Increments are what the calibrator can
     * isolate, since a device-to-device all-reduce necessarily performs the warp, block and
     * grid combines first.
     *
     * The grid entry excludes launch latency, carried separately below. Launch is a fixed cost
     * per reduction, independent of tree depth and with no CPU analogue, so folding it into
     * the tier would make the tier structure unreadable.
     */
    std::array<double, kTierCount> t_reduce_s;
    double t_kernel_launch_s;                    ///< measured; added once for tier >= GRID
    std::array<bool, kTierCount> tier_calibrated; ///< per-rung provenance

    /// True only when every reachable tier has been measured on this host. Suppresses R_h
    /// magnitudes otherwise, exactly as Machine::reduction_calibrated does on the CPU side.
    bool reduction_calibrated;
    /// True when an achieved bandwidth has been measured, so the roofline gate has a real
    /// denominator. A gate evaluated against a theoretical roof over-states the memory side
    /// and would wave through a point that is actually compute-bound.
    bool roofline_gated;
};

/**
 * The two presets. Hardware fields are datasheet transcription, confirmed against
 * cudaGetDeviceProperties; every t_reduce entry and every achieved roof is a measurement.
 *
 * The cards are a near-controlled pair: same 6 MiB L2, same ~900 GB/s bandwidth class, FP64
 * differing by 12.6x. They hold the vertical axis fixed and vary only the compute roof, which
 * is the lever that tests the roofline precondition.
 */
inline constexpr std::array<GpuMachine, 2> kGpuMachines {{
    {
        "v100-pcie-16gb",
        "NVIDIA Tesla V100-PCIE-16GB (synge)",
        80,             // SMs (GV100)
        64,             // 2048 resident threads / SM
        6L << 20,       // 6 MiB device-wide L2
        96L << 10,      // up to 96 KiB shared per SM, configurable against L1
        16L << 30,      // 16 GiB HBM2: the footprint ceiling
        // Measured, gpu-fma-loop on an idle device: 6.375 TFLOP/s FP64, 12.70 FP32, a 2.0:1
        // ratio confirming dedicated FP64 units. Below the 7.0/14.13 datasheet figures because
        // the card ran at 1245 MHz against a 1380 MHz max, so the shortfall is clock, not
        // silicon. This is the positive arm against the 3090's 0.570 TFLOP/s at 65.6:1.
        6.3751e12,      // FP64 peak
        1.2702e13,      // FP32 peak
        900.0,          // theoretical HBM2 (datasheet; deviceQuery computes 898 from clocks)
        // Measured, gpu-stream, idle device. DRAM is the median of DRAM-clean sizes (>=4x the
        // 13.5 MiB L2+L1 hierarchy), 91% of theoretical. As on GA102, aggregate L1 (7.5 MiB)
        // exceeds L2 (6 MiB), so the resident figure is an L1+L2 roof, not a pure L2 one.
        818.3,          // hbm_bw_gbs_achieved
        3624.8,         // l2_bw_gbs_achieved  (cache/DRAM ratio 4.43x)
        // Confirmed by scripts/regime/gpu_probe.sh: both V100s on one node, 16384 MiB each,
        // no MIG, no MPS daemon, no resident processes. The DEVICE_P2P rung is therefore
        // reachable in a one-node allocation and does not need -N 2.
        2,              // GPUs per node
        2,              // synge-n01, synge-n02
        "SYS",          // `nvidia-smi topo -m`: PCIe + cross-socket UPI, no NVLink.
                        // GPU0 on NUMA 0 (CPU 0-19), GPU1 on NUMA 1 (CPU 20-39).
        7.3,            // measured, NCCL all-reduce bus bandwidth over SYS, 64 MiB payload
                        // (7.36 device-to-device, 7.00 across the fabric)
        // Measured increments, which reduction_cost_s accumulates. WARP 0.422 us; BLOCK +0.368;
        // GRID +2.709 (from the two-kernel form at 5.54 us, which again beat cooperative
        // grid.sync() at 7.50 us); DEVICE_P2P +5.921 (11.46 us cumulative, minus the 5.539 us
        // GRID total); NODE +10.632 (22.09 us cumulative over NET/IB on hfi1_0, minus the
        // 11.46 us DEVICE_P2P total).
        //
        // DEVICE_P2P repeatability: four runs gave 11.24, 11.25, 11.46, 12.47 us (median
        // 11.36, spread 1.11x). The recorded figure is one of those samples, not a fit, and
        // every one is >12x tau*, so the corner verdict does not turn on which.
        //
        // The NODE rung is only 1.93x the DEVICE_P2P rung, a much flatter step than the CPU's
        // 5.4x CCX crossing: the intra-node link is PCIe gen3 x16 plus a cross-socket UPI hop,
        // slow enough that leaving the node barely doubles it. The swept horizontal axis is
        // therefore really {on-device, off-device}. See docs/regime_gpu_phase0.md.
        {{4.2246e-7, 3.6841e-7, 2.7089e-6, 5.9213e-6, 1.0632e-5}},
        // Launch is 74% of a grid reduction here, against 41% on the 3090, so the risk of
        // launch swamping the ladder comes far closer to firing. The rungs are still separable
        // (grid/block = 9.5x) but the margin is thin. See docs/regime_gpu_phase0.md.
        2.0389e-6,      // t_kernel_launch_s
        {{true, true, true, true, true}},     // every rung measured
        // All five rungs are reachable on synge and all five are measured, so the ladder is
        // complete and R_h may be published at any rung.
        true,           // reduction_calibrated
        true,           // roofline_gated
    },
    {
        // Calibrated on puffin with nvcc 12.8 / sm_86, on an idle device (NVML reported zero
        // foreign processes). The reduction and roof measurements each held to <1% over four
        // runs; contention shifts them 10-20%, always plausibly, which is why the idle gate
        // has to hold first.
        "rtx-3090",
        "NVIDIA GeForce RTX 3090 (puffin)",
        82,             // SMs (GA102)
        48,             // CC 8.6: 1536 resident threads / SM, not 2048
        6L << 20,       // 6 MiB L2, same as the V100: what makes the pair controlled
        100L << 10,     // CC 8.6: up to 100 KiB shared per SM
        24L << 30,      // 24 GiB GDDR6X
        // Measured, gpu-fma-loop: 0.570 TFLOP/s FP64, 37.39 FP32, ratio 65.6:1. Datasheet says
        // 0.556 / 35.58 at 1:64; the 1740 MHz boost clock lifts both. The negative arm, as a
        // measurement rather than a datasheet claim.
        5.7021e11,      // FP64 peak
        3.7387e13,      // FP32 peak
        936.2,          // theoretical GDDR6X (datasheet; the achieved roof is below)
        // Measured, gpu-stream. HBM is the median of DRAM-clean sizes (>=4x the 14 MiB L2+L1
        // hierarchy), 88% of theoretical. L2 is the resident peak; aggregate L1 (8 MiB)
        // exceeds L2 (6 MiB) on GA102, so this is an L1+L2 figure. See gpu_stream.cu.
        821.1,          // hbm_bw_gbs_achieved
        5184.8,         // l2_bw_gbs_achieved
        1,              // one device
        1,
        "NONE",         // single device: no P2P tier, no node tier
        0.0,            // interconnect: unreachable on one device
        // Measured increments, calibrate-gpu-reduction. WARP 2.834 us; BLOCK +3.839; GRID
        // +2.531 (from the two-kernel form at 11.59 us, which beat cooperative grid.sync() at
        // 18.8 us, so the model carries the form a real implementation would call).
        // DEVICE_P2P and NODE are unreachable here.
        {{2.8337e-6, 3.8389e-6, 2.5308e-6, 0.0, 0.0}},
        2.3893e-6,      // t_kernel_launch_s (its own term; 41% of a grid reduction)
        {{true, true, true, false, false}},   // the three reachable rungs are measured
        // On a single-device host WARP/BLOCK/GRID are the only reachable rungs, so the ladder
        // is complete and the flag may go true.
        true,           // reduction_calibrated
        true,           // roofline_gated
    },
}};

[[nodiscard]] inline const GpuMachine& lookup_gpu_machine(std::string_view key)
{
    for (const auto& g : kGpuMachines)
        if (g.key == key) return g;
    throw std::invalid_argument("Unknown GPU machine: " + std::string(key)
        + ".  Valid keys: v100-pcie-16gb, rtx-3090");
}

// Tier reachability
/**
 * @brief Whether this machine can exercise a rung at all.
 *
 * A single-device preset has no device-to-device or node rung, so a calibration that reports
 * one is reporting a bug. reduction_calibrated is checked against reachable rungs only.
 */
[[nodiscard]] inline constexpr bool tier_reachable(const GpuMachine& gm, ReductionTier t) noexcept
{
    switch (t) {
        case ReductionTier::WARP:
        case ReductionTier::BLOCK:
        case ReductionTier::GRID:       return true;
        case ReductionTier::DEVICE_P2P: return gm.gpu_count > 1;
        case ReductionTier::NODE:       return gm.node_count > 1;
    }
    return false;
}

/// The most expensive rung this machine can place a point on: the right-hand end of the
/// swept horizontal axis.
[[nodiscard]] inline constexpr ReductionTier highest_reachable_tier(const GpuMachine& gm) noexcept
{
    ReductionTier top = ReductionTier::WARP;
    for (int i = 0; i < kTierCount; ++i) {
        const auto t = static_cast<ReductionTier>(i);
        if (tier_reachable(gm, t)) top = t;
    }
    return top;
}

/**
 * @brief The most expensive rung that is both reachable and measured.
 *
 * Distinct from highest_reachable_tier(). An uncalibrated rung carries a zero increment, so
 * reduction_cost_s() returns the cost of the rung below it while the caller believes it asked
 * for the higher one: a plausible number attached to the wrong hardware.
 *
 * Placement uses this, and reports the gap when it is lower than the reachable top, rather
 * than labeling a DEVICE_P2P measurement as a NODE one.
 */
[[nodiscard]] inline constexpr ReductionTier highest_calibrated_tier(const GpuMachine& gm) noexcept
{
    ReductionTier top = ReductionTier::WARP;
    for (int i = 0; i < kTierCount; ++i) {
        const auto t = static_cast<ReductionTier>(i);
        if (tier_reachable(gm, t) && gm.tier_calibrated[tier_index(t)]) top = t;
    }
    return top;
}

/// Whether every reachable rung carries a measured cost. The honest precondition for
/// `reduction_calibrated`; check this rather than trusting the flag when editing a preset.
[[nodiscard]] inline constexpr bool all_reachable_tiers_calibrated(const GpuMachine& gm) noexcept
{
    for (int i = 0; i < kTierCount; ++i) {
        const auto t = static_cast<ReductionTier>(i);
        if (tier_reachable(gm, t) && !gm.tier_calibrated[tier_index(t)]) return false;
    }
    return true;
}

// The memory hierarchy
/**
 * @brief Aggregate last-level cache commanded by P SMs: the denominator of R_v.
 *
 * Ignores P, which is the largest structural difference from the CPU model. On a CPU each
 * engaged CCX brings its own L3 slice, so `aggregate_llc_bytes` grows with the team; a GPU's
 * L2 is one fixed block shared by every SM.
 *
 * The consequence is a decoupling of the two axes. On the CPU R_v ~ 1/P and R_h ~ P shared the
 * parallelism knob, which is why their product collapsed to a P-independent constant and
 * pinned the Upper-Right corner out of reach. Here R_v is P-free outright, so N alone moves it
 * and the reduction tier alone moves R_h. See docs/regime_gpu_phase0.md.
 *
 * @param P retained in the signature so the CPU and GPU call sites read alike, and so the
 *          asymmetry is visible at the point of use.
 */
[[nodiscard]] inline constexpr int64_t aggregate_l2_bytes(const GpuMachine& gm,
                                                          [[maybe_unused]] int P) noexcept
{
    return gm.l2_bytes;
}

/**
 * @brief The bandwidth used for pricing: measured if available, theoretical if not.
 *
 * An unmeasured roof errs the opposite way to the intuitive guess. A kernel is memory-bound
 * when AI < peak/BW, so an over-stated bandwidth lowers the ridge and makes the gate call
 * kernels compute-bound, i.e. off-map. The theoretical fallback is therefore conservative: it
 * can wrongly exclude a point that is really on the map, but it cannot wave a compute-bound
 * point onto one. That is why it is a fallback rather than a hard error.
 *
 * It is still a reason to measure. On the 3090 the margins are small enough, with MGS clearing
 * its ridge by under 2x, that the gap between 936 GB/s on paper and what the card sustains can
 * flip a verdict. `roofline_gated` marks the distinction so a provisional verdict is never
 * read as a final one.
 */
[[nodiscard]] inline constexpr double achieved_or_peak_bw_gbs(const GpuMachine& gm) noexcept
{
    return gm.hbm_bw_gbs_achieved > 0.0 ? gm.hbm_bw_gbs_achieved : gm.hbm_bw_gbs;
}

/**
 * @brief Which roof serves a kernel whose reuse window is `working_set` bytes.
 *
 * Two device-wide roofs, both flat in SM count once saturated: L2 while resident, HBM once
 * spilled. Shared memory is not a third branch. It is programmer-managed rather than a
 * transparent cache, so a kernel does not fall into it by having a small working set; it must
 * be tiled into it explicitly, and modeling it as a cache would credit every kernel with reuse
 * the code has not been written to take.
 *
 * @note No P parameter, unlike the CPU twin. Neither roof scales with the team.
 */
[[nodiscard]] inline constexpr double memory_bw_gbs(const GpuMachine& gm,
                                                    int64_t working_set) noexcept
{
    if (working_set <= gm.l2_bytes && gm.l2_bw_gbs_achieved > 0.0)
        return gm.l2_bw_gbs_achieved;
    if (working_set <= gm.l2_bytes)
        return gm.hbm_bw_gbs;   // L2 roof unmeasured: fall back rather than invent a number
    return achieved_or_peak_bw_gbs(gm);
}

// The compute roof, and the gate in front of the map
[[nodiscard]] inline constexpr double peak_flops(const GpuMachine& gm, Precision p) noexcept
{
    return p == Precision::FP64 ? gm.fp64_flops_peak : gm.fp32_flops_peak;
}

/**
 * @brief Roofline-attainable rate [FLOP/s] with P of the device's SMs engaged.
 *
 * Compute scales with SMs; bandwidth does not scale at all. That asymmetry makes the two
 * coordinates P-free in the memory-bound branch and P-dependent in the compute-bound one,
 * which is why the gate below is a precondition rather than a diagnostic.
 */
[[nodiscard]] inline double attainable_flops(const GpuMachine& gm, int P, double ai,
                                             double bw_gbs, Precision p) noexcept
{
    const double sm_fraction = gm.sm_count > 0
        ? static_cast<double>(P) / static_cast<double>(gm.sm_count) : 1.0;
    const double compute_roof = peak_flops(gm, p) * sm_fraction;
    const double memory_roof  = ai * bw_gbs * 1e9;
    return std::min(compute_roof, memory_roof);
}

/**
 * @brief The arithmetic intensity above which a kernel is compute-bound: peak / bandwidth.
 *
 * The whole-device figure, and the one to quote when comparing machines. The V100's FP64 ridge
 * is ~7.8 FLOP/B and the 3090's ~0.59, a 13x separation produced almost entirely by the FP64
 * rate since the two bandwidths are within 4%. That separation is the study's controlled
 * variable.
 */
[[nodiscard]] inline double ridge_ai(const GpuMachine& gm, Precision p) noexcept
{
    const double bw = achieved_or_peak_bw_gbs(gm) * 1e9;
    return bw > 0.0 ? peak_flops(gm, p) / bw : 0.0;
}

/**
 * @brief The ridge with only P of the device's SMs engaged.
 *
 * Compute scales with SMs and bandwidth does not, so a partially engaged device has a lower
 * ridge and is more easily compute-bound. On the V100 in FP64, SpMV's 0.135 FLOP/B crosses the
 * ridge below about 2 SMs and MGS's 0.375 below about 4: engage less of the device than that
 * and the Arnoldi cycle really is compute-bound, and really is off-map.
 *
 * The gate uses this rather than the whole-device ridge. Otherwise it certifies a point as
 * memory-bound while attainable_flops() prices it on the compute branch, where the cycle time
 * carries a 1/P that breaks the P-independence the Phase 0 derivation rests on.
 */
[[nodiscard]] inline double ridge_ai_at(const GpuMachine& gm, int P, Precision p) noexcept
{
    const double bw = achieved_or_peak_bw_gbs(gm) * 1e9;
    if (!(bw > 0.0) || gm.sm_count <= 0) return 0.0;
    const double sm_fraction = static_cast<double>(P) / static_cast<double>(gm.sm_count);
    return peak_flops(gm, p) * sm_fraction / bw;
}

/// The verdict a point carries alongside its coordinates. See roofline_gate().
struct RooflineVerdict {
    double ai           = 0.0;    ///< the kernel's arithmetic intensity [FLOP/byte]
    double ridge        = 0.0;    ///< the machine's ridge in this precision [FLOP/byte]
    double margin       = 0.0;    ///< ridge / ai. > 1 = memory-bound, and by how much.
    bool   memory_bound = false;  ///< the gate itself: is this point on the map at all?
    bool   provisional  = true;   ///< true when computed against a theoretical, unmeasured roof
};

/**
 * @brief The gate in front of the map.
 *
 * Both coordinates measure communication: one a cache-capacity ratio, the other a
 * reduction-latency ratio. Neither encodes whether the kernel is compute-bound, so a ratio
 * crossing 1 is informative only if the binding constraint is cache capacity or reduction
 * latency in the first place. Throttling FP64 does not move a point around the map, it throws
 * the point off the map, into a regime neither coordinate charts.
 *
 * Evaluated per kernel, not once per machine. The two-card contrast does not discriminate on
 * SpMV or MGS at all: their intensities, 0.135 and 0.375 FLOP/B, sit below even the 3090's
 * throttled ridge, so both cards are on-map for the baseline method. It discriminates on the
 * tall-skinny Gram matrix that CA introduces and MGS does not have, whose intensity is ~s/4
 * and which crosses the 3090's ridge at s ~ 2.4, far below the certified s_max = 9. A
 * machine-level verdict would have hidden that. See docs/regime_gpu_phase0.md.
 *
 * A point failing this gate is off-map, not lower-left, and is reported as such.
 *
 * @param P SMs engaged. Uses the ridge at that P, so the verdict agrees with what
 *          attainable_flops() does; see ridge_ai_at() for why that consistency matters.
 */
[[nodiscard]] inline RooflineVerdict roofline_gate(const GpuMachine& gm, int P, double ai,
                                                   Precision p) noexcept
{
    RooflineVerdict v;
    v.ai           = ai;
    v.ridge        = ridge_ai_at(gm, P, p);
    v.margin       = ai > 0.0 ? v.ridge / ai : 0.0;
    v.memory_bound = ai < v.ridge;
    v.provisional  = !gm.roofline_gated;
    return v;
}

/// The whole-device gate: the machine-level verdict, for comparing cards rather than placing
/// points. Placement must use the P-aware form above.
[[nodiscard]] inline RooflineVerdict roofline_gate(const GpuMachine& gm, double ai,
                                                   Precision p) noexcept
{
    return roofline_gate(gm, gm.sm_count, ai, p);
}

// The reduction ladder
/**
 * @brief Modeled cost of one global reduction at a given tier: the numerator of R_h.
 *
 *     t_reduce(tier) = sum_{i <= tier} t_reduce_s[i]  +  (tier >= GRID ? t_kernel_launch_s : 0)
 *
 * Cumulative because the tiers nest physically: a device-to-device all-reduce necessarily
 * performs the warp, block and grid combines before anything crosses the link. The stored
 * entries are increments, so the sum is the total.
 *
 * The launch term is added once, not per level, since it is a fixed cost per reduction
 * independent of tree depth. If it dominates every rung from GRID upward the tier model
 * collapses to a constant and the horizontal mechanism becomes uninteresting, which is itself
 * a result and the reason the calibrator reports launch latency as its own term.
 *
 * Unlike the CPU twin there is no saturation constant. L* exists because a deep binary tree's
 * subtrees complete concurrently, hiding later levels; this ladder is five named mechanisms
 * rather than a tree of equal levels, so there is nothing for saturation to hide. A plateau
 * within one rung belongs inside that rung's measured constant.
 *
 * @note P does not appear. On the CPU the numerator was a function of team size because the
 *       tree deepened with it. Here the rung, not the team, sets the cost.
 */
[[nodiscard]] inline double reduction_cost_s(const GpuMachine& gm, ReductionTier tier) noexcept
{
    double t = 0.0;
    for (std::size_t i = 0; i <= tier_index(tier); ++i)
        t += gm.t_reduce_s[i];
    if (static_cast<int>(tier) >= static_cast<int>(ReductionTier::GRID))
        t += gm.t_kernel_launch_s;
    return t;
}

/**
 * @brief What one rung costs in units of the rung below it: the GPU crossing multiplier.
 *
 * The analogue of the CPU's 5.4x CCX step. A rung whose multiplier is ~1 does not move R_h and
 * cannot serve as a distinct point on the swept horizontal axis.
 *
 * @return 0 when either rung is uncalibrated or free, so a caller cannot mistake an
 *         unmeasured ladder for a flat one.
 */
[[nodiscard]] inline double tier_multiplier(const GpuMachine& gm, ReductionTier tier) noexcept
{
    if (tier == ReductionTier::WARP) return 0.0;
    const auto below = static_cast<ReductionTier>(static_cast<int>(tier) - 1);
    const double lo = reduction_cost_s(gm, below);
    const double hi = reduction_cost_s(gm, tier);
    return lo > 0.0 ? hi / lo : 0.0;
}

/// Fraction of a reduction at this tier that is pure kernel-launch overhead. Near 1 means the
/// tier structure is invisible behind launch cost and R_h has no shape in the ladder.
[[nodiscard]] inline double launch_share(const GpuMachine& gm, ReductionTier tier) noexcept
{
    if (static_cast<int>(tier) < static_cast<int>(ReductionTier::GRID)) return 0.0;
    const double t = reduction_cost_s(gm, tier);
    return t > 0.0 ? gm.t_kernel_launch_s / t : 0.0;
}
