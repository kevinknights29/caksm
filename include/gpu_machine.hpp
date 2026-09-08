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
 *      being a function of how much of the GPU is engaged.
 *   2. A reduction crosses up to five tiers, not two, and the grid tier carries a
 *      kernel-launch cost with no CPU analogue. Hence, a tier vector and a separate launch term.
 *   3. The compute roof is a device figure that must be gated. Both map coordinates measure
 *      communication, so a compute-bound kernel is off-map rather than lower-left.
 *      roofline_gate() is that gate and runs before any point is placed.
 *
 * Where each number comes from, the same rule as the CPU side: geometry and the roofs a vendor
 * publishes are transcribed from the datasheet, but every t_reduce tier and every achieved roof
 * is measured offline on the host itself (scripts/regime/calibrate_gpu.sh). `reduction_calibrated`
 * and `roofline_gated` record which of the two a field is, so an untested preset cannot emit a
 * confident magnitude.
 *
 * @author Kevin Knights
 * @date 2026-07-21
 */
#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

/**
 * @brief How far a GPU reduction has to reach to combine its partial results, cheapest first.
 *
 * A reduction combines partials over a widening scope: threads within a warp, then warps within
 * a block, then blocks within a grid, then devices, then nodes. Each scope is one rung of that
 * ladder, and reaching rung i means having already paid for every rung below it, which is why
 * t_reduce_s stores increments rather than totals.
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
 * @brief Driver-reported capabilities that bound copy/compute overlap.
 *
 * These are queried by gpu-device-probe and transcribed only after the probe has run on the
 * recorded device. A value of -1 is deliberately unknown; it must not be replaced by an
 * architectural guess.
 */
struct GpuOverlapCapabilities {
    int  async_engine_count;
    int  concurrent_kernels;
    int  device_overlap;
    int  unified_addressing;
    bool recorded;
};

/**
 * @brief A-priori hardware parameters for one GPU environment.
 *
 * @note The fields marked measured are not datasheet values and must not be guessed. Each comes
 *       from an offline binary run on an idle device over seven repeats, reported as a median:
 *       `gpu-fma-loop` for the FP64 and FP32 roofs, `gpu-stream` for the achieved HBM and L2
 *       bandwidths, `calibrate-gpu-reduction` for the warp, block and grid increments and the
 *       launch term, and `calibrate-gpu-p2p` for the interconnect bandwidth and the device and
 *       node increments. scripts/regime/calibrate_gpu.sh and calibrate_gpu_p2p.sh drive them,
 *       and gpu_contention.cuh makes them refuse to emit a constant from a contended device.
 *       A preset whose tiers are unmeasured carries reduction_calibrated = false, which every
 *       consumer of R_h checks before printing a magnitude.
 *
 * @note Nothing here describes the cluster. How many devices and nodes a launch received is a
 *       property of the allocation (gpu_allocation.hpp), and the collective cost at an exact
 *       participant count is a property of a measured arrangement (gpu_topology.hpp). A preset
 *       may carry an off-device increment only where its cluster offers exactly one
 *       participant count, as synge does; on a node that can present one, two, four or eight
 *       devices those entries stay zero.
 */
struct GpuMachine {
    std::string_view key;
    std::string_view name;
    /// Distinguishing token of cudaGetDeviceProperties::name, lower case. An instrument that
    /// predicts before measuring selects its preset from the device it is about to run on,
    /// so the match has to come from the driver rather than from a hardcoded key.
    std::string_view device_match;

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

    // The link this device sits on
    /// The label `nvidia-smi topo -m` prints for a pair of these devices: "SYS", "NV18",
    /// "PIX", ... A device property only where the matrix is uniform, which it is on both
    /// calibrated clusters; where it is not, the authority is the per-arrangement
    /// `link_signature` in gpu_topology.hpp and this field must read "MIXED".
    std::string_view interconnect;
    /// Measured all-reduce bus bandwidth over that link [GB/s]. 0 = not measured, which is
    /// the correct entry whenever the figure is participant-keyed rather than a property of
    /// the link: see the H200 preset. Never a datasheet number.
    double           interconnect_bw_gbs;
    GpuOverlapCapabilities overlap;        ///< cudaGetDeviceProperties; -1 until recorded

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

    /// True when this device's own ladder, the WARP, BLOCK and GRID rungs and the launch
    /// term, has been measured on this host. Suppresses R_h magnitudes otherwise, exactly as
    /// Machine::reduction_calibrated does on the CPU side.
    ///
    /// It is deliberately NOT a statement about the link rungs. Those are reachable only in
    /// some allocations and priced per participant count, so the question "is the whole ladder
    /// this launch needs measured?" belongs to all_reachable_tiers_calibrated() in
    /// gpu_allocation.hpp, which takes the allocation and can answer it.
    bool reduction_calibrated;
    /// True when an achieved bandwidth has been measured, so the roofline gate has a real
    /// denominator. A gate evaluated against a theoretical roof over-states the memory side
    /// and would wave through a point that is actually compute-bound.
    bool roofline_gated;
};

/**
 * The presets. Hardware fields are datasheet transcription, confirmed against
 * cudaGetDeviceProperties; every t_reduce entry and every achieved roof is a measurement.
 *
 * The V100 and the 3090 are a near-controlled pair: same 6 MiB L2, same ~900 GB/s bandwidth
 * class, FP64 differing by 12x. They hold the vertical axis fixed and vary only the compute
 * roof, which is the lever that tests the roofline precondition.
 *
 * The H200 is not a third point on that axis: it changes L2 by 10x and bandwidth by 5x at
 * once, varying the vertical mechanism rather than controlling it. What it contributes is the
 * participant axis. Keep the machines in separate panels; replacing hardware is not a step
 * along any of the map's coordinates.
 */
inline constexpr std::array<GpuMachine, 3> kGpuMachines {{
    {
        "v100-pcie-16gb",
        "NVIDIA Tesla V100-PCIE-16GB (synge)",
        "v100-pcie-16gb",
        80,             // SMs (GV100)
        64,             // 2048 resident threads / SM
        6L << 20,       // 6 MiB device-wide L2
        96L << 10,      // up to 96 KiB shared per SM, configurable against L1
        16L << 30,      // 16 GiB HBM2: the footprint ceiling
        // Measured, gpu-fma-loop on an idle device: 6.375 TFLOP/s FP64, 12.70 FP32, a 2.0:1
        // ratio confirming dedicated FP64 units. Below the 7.0/14.13 datasheet figures because
        // the card ran at 1245 MHz against a 1380 MHz max: the shortfall is clock, not silicon.
        6.3751e12,      // FP64 peak
        1.2702e13,      // FP32 peak
        900.0,          // theoretical HBM2 (datasheet; deviceQuery computes 898 from clocks)
        // Measured, gpu-stream, idle device. DRAM is the median of DRAM-clean sizes (>=4x the
        // 13.5 MiB L2+L1 hierarchy), 91% of theoretical. As on GA102, aggregate L1 (7.5 MiB)
        // exceeds L2 (6 MiB), so the resident figure is an L1+L2 roof, not a pure L2 one.
        818.3,          // hbm_bw_gbs_achieved
        3624.8,         // l2_bw_gbs_achieved  (cache/DRAM ratio 4.43x)
        // Whether the DEVICE_P2P rung is reachable is a question about the allocation, not
        // about this preset; see gpu_allocation.hpp.
        "SYS",          // `nvidia-smi topo -m`: PCIe + cross-socket UPI, no NVLink.
                        // GPU0 on NUMA 0 (CPU 0-19), GPU1 on NUMA 1 (CPU 20-39). Uniform:
                        // there is only the one pair.
        7.3,            // measured, NCCL all-reduce bus bandwidth over SYS, 64 MiB payload.
                        // Two participants, the only count synge offers, which is why a
                        // single figure may sit on this preset at all.
        // gpu-device-probe, both devices, which agreed. Capability flags, not evidence that a
        // particular transfer overlapped.
        {7, 1, 1, 1, true}, // async engines, concurrent kernels, overlap, unified addressing
        // Measured increments, which reduction_cost_s accumulates. WARP 0.422 us; BLOCK +0.368;
        // GRID +2.709 (from the two-kernel form at 5.54 us, which again beat cooperative
        // grid.sync() at 7.50 us); DEVICE_P2P +5.921 (11.46 us cumulative, minus the 5.539 us
        // GRID total); NODE +10.632 (22.09 us cumulative over NET/IB on hfi1_0, minus the
        // 11.46 us DEVICE_P2P total).
        //
        // NODE is only 1.93x DEVICE_P2P: the intra-node link is PCIe gen3 x16 plus a
        // cross-socket UPI hop, slow enough that leaving the node barely doubles it. The swept
        // horizontal axis here is effectively {on-device, off-device}.
        {{4.2246e-7, 3.6841e-7, 2.7089e-6, 5.9213e-6, 1.0632e-5}},
        // Launch is 74% of a grid reduction here, the narrowest margin of the three presets.
        // The rungs stay separable (grid/block = 9.5x), but only just.
        2.0389e-6,      // t_kernel_launch_s
        {{true, true, true, true, true}},     // every rung measured
        // All five rungs are reachable on synge and all five are measured, so the ladder is
        // complete and R_h may be published at any rung.
        true,           // reduction_calibrated
        true,           // roofline_gated
    },
    {
        // Calibrated on puffin with nvcc 12.8 / sm_86, on an idle device (NVML reported zero
        // foreign processes).
        "rtx-3090",
        "NVIDIA GeForce RTX 3090 (puffin)",
        "rtx 3090",
        82,             // SMs (GA102)
        48,             // CC 8.6: 1536 resident threads / SM, not 2048
        6L << 20,       // 6 MiB L2, same as the V100: what makes the pair controlled
        100L << 10,     // CC 8.6: up to 100 KiB shared per SM
        24L << 30,      // 24 GiB GDDR6X
        // Measured, gpu-fma-loop: 0.570 TFLOP/s FP64, 37.39 FP32, ratio 65.6:1. Datasheet says
        // 0.556 / 35.58 at 1:64; the 1740 MHz boost clock lifts both.
        5.7021e11,      // FP64 peak
        3.7387e13,      // FP32 peak
        936.2,          // theoretical GDDR6X (datasheet; the achieved roof is below)
        // Measured, gpu-stream. HBM is the median of DRAM-clean sizes (>=4x the 14 MiB L2+L1
        // hierarchy), 88% of theoretical. L2 is the resident peak; aggregate L1 (8 MiB)
        // exceeds L2 (6 MiB) on GA102, so this is an L1+L2 figure. See gpu_stream.cu.
        821.1,          // hbm_bw_gbs_achieved
        5184.8,         // l2_bw_gbs_achieved
        "NONE",         // a single-device workstation: no pair to label
        0.0,            // interconnect: nothing to measure
        {-1, -1, -1, -1, false}, // gpu-device-probe has not yet been archived on Puffin
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
    {
        // Calibrated on gpu01, with all eight devices visible and NVML reporting zero foreign
        // processes at every stage. Built by the Spack CUDA 12.8.1 nvcc for sm_90 in Release.
        //
        // The whole preset is transcribed from ONE invocation: the one whose two-kernel grid
        // total is nearest the mean of all of them. Rung costs accumulate, so averaging rung
        // by rung yields a ladder total no invocation produced, while transcribing one keeps
        // reduction_cost_s() reproducing a total that was actually measured. A test checks it.
        "h200",
        "NVIDIA H200 (gpu01)",
        "h200",
        132,            // SMs (GH100), driver-reported
        64,             // CC 9.0: 2048 resident threads / SM. Confirm with gpu-device-probe.
        60L << 20,      // 60 MiB device-wide L2, driver-reported (62914560 B)
        228L << 10,     // 228 KiB shared per SM (CC 9.0)
        // The calibrator reports 139.8 GiB from cudaDeviceProp::totalGlobalMem, against the
        // 143771 MiB nvidia-smi shows. Rounded DOWN to a whole GiB so the footprint ceiling
        // stays conservative; the exact byte count needs gpu-device-probe and is not guessed.
        139L << 30,
        // Measured, gpu-fma-loop on an idle device: 30.750 TFLOP/s FP64, 61.414 FP32, a
        // 2.0:1 ratio confirming dedicated FP64 units. The positive arm, as the V100 is, but
        // 4.8x its FP64 rate against 5.0x its bandwidth, so the ridge barely moves.
        3.0750e13,      // FP64 peak
        6.1414e13,      // FP32 peak
        4814.0,         // theoretical HBM3e, as gpu-stream computes it from the driver's
                        // memory clock and bus width (datasheet class 4.8 TB/s)
        // Measured, gpu-stream, idle device. DRAM is the median of DRAM-clean sizes (>= 4x the
        // 89.4 MiB L2+L1 hierarchy), 84% of theoretical. Aggregate L1 is 29.4 MiB against a
        // 60 MiB L2, so unlike GA102 and GV100 the resident figure is L2-dominated.
        4053.1,         // hbm_bw_gbs_achieved
        12988.1,        // l2_bw_gbs_achieved  (cache/DRAM ratio 3.20x, against the V100's 4.43)
        // `nvidia-smi topo -m` on gpu01: NV18 between every one of the 28 GPU pairs, so the
        // local link graph is uniform and the label is a device-level fact here. CPU and NIC
        // placement is NOT uniform: GPUs 0-3 sit on NUMA 0 (CPU 0-47, 96-143) with mlx5_0..3
        // closest, GPUs 4-7 on NUMA 1 (CPU 48-95, 144-191) with mlx5_4..7. That asymmetry is
        // recorded in the topology manifest's link signature, where a subset can name it.
        "NV18",
        // Deliberately not measured on this preset. The all-reduce bus bandwidth over NV18 is
        // keyed to the participant count, not to the link: 274.6 GB/s at two participants,
        // 315.9 at four and 373.9 at eight, a 36% span. Recording any one here would let a
        // two-GPU run borrow the eight-GPU figure, which is the exact substitution this port
        // exists to prevent. All three live in kGpuTopologies instead.
        0.0,
        // gpu-device-probe, all eight devices, which agreed. Capability flags, not evidence
        // that a particular transfer overlapped.
        {3, 1, 1, 1, true}, // async engines, concurrent kernels, overlap, unified addressing
        // Measured increments, calibrate-gpu-reduction. WARP 0.4017 us; BLOCK +0.3989;
        // GRID +2.5489 (from the two-kernel form at 5.3770 us, which again beat cooperative
        // grid.sync() at 7.4889 us, so the model carries the two-kernel form). The entries
        // accumulate to exactly the measured 5.3770 us total.
        //
        // DEVICE_P2P and NODE are deliberately zero and uncalibrated. Eight H200s on one node
        // reduce at the same DEVICE_P2P rung as two and cost 3.71x as much (39.002 us against
        // 10.510), so no single increment can serve both and this array, indexed by rung
        // alone, cannot hold them. The measured totals live in kGpuTopologies, keyed by
        // participant count. Leaving these zero is what makes a launch that asks for a link
        // cost here fail loudly instead of pricing the grid rung under a link name.
        {{4.0167e-7, 3.9893e-7, 2.5489e-6, 0.0, 0.0}},
        // Launch is 38% of a grid reduction here, the widest margin of the three presets, so
        // the ladder's rungs stay clearly separable (grid/block = 6.7x).
        2.0274e-6,      // t_kernel_launch_s
        {{true, true, true, false, false}},   // the on-device rungs are measured
        // The on-device ladder is complete. Whether that is the whole ladder depends on the
        // allocation, which this preset does not know and must not assume: on a one-GPU shell
        // it is, and on the eight-GPU node all_reachable_tiers_calibrated() returns false
        // until a topology record supplies the link cost.
        true,           // reduction_calibrated
        true,           // roofline_gated
    },
}};

[[nodiscard]] inline const GpuMachine& lookup_gpu_machine(std::string_view key)
{
    for (const auto& g : kGpuMachines)
        if (g.key == key) return g;
    throw std::invalid_argument("Unknown GPU machine: " + std::string(key)
        + ".  Valid keys: v100-pcie-16gb, rtx-3090, h200");
}

/**
 * @brief The calibrated preset for a device, matched on its driver-reported name.
 *
 * A hardcoded key is correct on the host it was written for and silently wrong everywhere
 * else, which is the one failure mode a predict-then-measure printout cannot tolerate. An
 * uncalibrated device names itself in the error rather than borrowing another card's
 * constants.
 */
[[nodiscard]] inline const GpuMachine& lookup_gpu_machine_for_device(
    std::string_view device_name)
{
    std::string lowered(device_name);
    std::transform(
        lowered.begin(), lowered.end(), lowered.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    for (const auto& g : kGpuMachines)
        if (lowered.find(g.device_match) != std::string::npos) return g;
    throw std::invalid_argument(
        "No calibrated GPU machine for device: " + std::string(device_name)
        + ".  Calibrated devices: v100-pcie-16gb, rtx-3090, h200.  Calibrate this "
          "device with scripts/regime/calibrate_gpu.sh before predicting on it.");
}

// Tier reachability lives in gpu_allocation.hpp.
//
// It moved because it is not a question about a device. Which rungs a launch can reach is
// set by the nodes and local GPUs the scheduler handed it, and the same preset has to serve
// a one-GPU shell and a full eight-GPU node without claiming the link rungs in the first.

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
 * and the reduction tier alone moves R_h.
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
 * `roofline_gated` marks the distinction so a provisional verdict is never read as a final
 * one.
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
 * rate since the two bandwidths are within 4%.
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
 * The gate uses this rather than the whole-device ridge. Otherwise, it certifies a point as
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
 * Evaluated per kernel, not once per machine. SpMV and MGS, at 0.135 and 0.375 FLOP/B, sit
 * below even the 3090's throttled ridge, so both cards are on-map for the baseline method.
 * The tall-skinny Gram matrix CA introduces has intensity ~s/4 and crosses the 3090's ridge
 * at s ~ 2.4, far below the certified s_max = 9. A machine-level verdict would hide that.
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
