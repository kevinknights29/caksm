/**
 * @file gpu_topology.hpp
 * @brief Participant and node topology for the GPU regime map, and the slab decomposition
 *        that turns a global grid into the per-GPU working set the vertical axis measures.
 *
 * gpu_machine.hpp describes one device. Everything here describes how several of them are
 * arranged, which is a separate question and gets a separate type: `GpuMachine` carries a
 * `gpu_count` and a `node_count` that say what the allocation *could* reach, not how a
 * particular launch was actually placed. A launch on two GPUs of one node and a launch on one
 * GPU of each of two nodes share every field of the preset and land in different places on the
 * map, so the topology has to be its own input rather than a derived one.
 *
 * Two structural facts drive the design:
 *
 *   1. Distributed GPU caches are not coherent. The vertical coordinate is therefore the
 *      maximum per-GPU local working set over one device's L2, never the global working set
 *      over the summed L2 of the allocation. Summing would credit the operator with reuse
 *      across a link that does not carry it, and the error grows with participant count, which
 *      is exactly the axis the trajectory figure sweeps.
 *   2. The reduction cost is a property of the participant and node topology, not of the tier
 *      name. Two participants on two nodes and four participants on two nodes both reduce at
 *      the NODE rung and differ by 1.5x, because the ring is longer. `GpuMachine::t_reduce_s`
 *      cannot express that: it is indexed by rung alone. Each topology therefore carries its
 *      own measured collective latency, transcribed from the run that measured that exact
 *      arrangement.
 *
 * The provenance rule is inherited unchanged from gpu_machine.hpp: geometry is transcribed,
 * every latency is measured offline on an idle device over at least seven repeats, and the
 * source file of each measurement is named beside it. A topology whose collective cost has not
 * been measured for its own arrangement carries `calibrated = false`, and the trajectory tool
 * refuses to place it rather than borrowing a neighboring rung's constant.
 *
 * @author Kevin Knights
 * @date 2026-08-23
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <string_view>

#include "gpu_machine.hpp"
#include "gpu_regime.hpp"

/**
 * @brief One measured arrangement of participants over devices and nodes.
 *
 * A participant is one GPU taking part in the collective, which on this code path is also one
 * MPI rank's device. `nodes` and `local_gpus` are recorded separately rather than folded into
 * the participant count because they are what a launch has to be checked against: the same
 * `participants` value describes both the two-node and the one-node pair, and only the split
 * says which link the reduction crossed.
 *
 * @note `t_reduce_s` is the whole cost of one global reduction at this arrangement, not an
 *       increment over the rung below. The ladder in gpu_machine.hpp stores increments because
 *       its calibrator can only isolate increments; here the measurement is of the finished
 *       collective, so the field is the total and nothing accumulates it.
 */
struct GpuTopology {
    std::string_view key;          ///< canonical name, used as part of the CSV row key
    std::string_view label;        ///< how the figure names it
    int participants;              ///< GPUs taking part in the collective
    int nodes;                     ///< distinct hosts spanned
    int local_gpus;                ///< GPUs per host; participants == nodes * local_gpus
    ReductionTier tier;            ///< the rung this arrangement reduces at

    /// Measured cost of one global reduction [s], with the interquartile range beside it.
    /// The median of at least seven repeats on an idle device; q1 and q3 are carried so a
    /// figure can show the spread rather than assert a point value.
    double t_reduce_s;
    double t_reduce_q1_s;
    double t_reduce_q3_s;

    int              repeats;      ///< repeats behind the median
    bool             contended;    ///< the idle-device gate; true blocks the point
    std::string_view instrument;   ///< the binary that produced the number
    std::string_view source;       ///< repo-relative CSV the number was transcribed from
    bool             calibrated;   ///< false suppresses the point entirely
};

/**
 * The four arrangements synge supports, in the order the trajectory figure connects them.
 *
 * Ordered by collective reach, which on this cluster is also ordered by cost: 5.54 us on one
 * device, 11.25 us across the two GPU/NUMA domains of one node, 20.06 us across two nodes, and
 * 29.20 us once both GPUs of both nodes take part. The monotonicity is the hypothesis panel C
 * tests, so it is worth naming that it is visible in the constants before any point is placed;
 * what the figure adds is whether the motion in R_h is large enough to cross theta_h.
 *
 * The single-GPU entry is not an NCCL measurement and should not be one. With one participant
 * no collective leaves the device, so the global reduction *is* the grid reduction, and the
 * number is the two-kernel grid form from calibrate-gpu-reduction. The remaining three are all
 * calibrate-gpu-p2p, which is also what scripts/regime/calibrate_ca_participants.sh drives.
 */
inline constexpr std::array<GpuTopology, 4> kSyngeTopologies {{
    {
        "1gpu-1node",
        "1 GPU, 1 node",
        1, 1, 1,
        ReductionTier::GRID,
        // calibrate-gpu-reduction, tier grid_two_kernel: the two-kernel form, which beat
        // cooperative grid.sync() at 7.50 us and is the form a real implementation calls.
        // Equals reduction_cost_s(v100, GRID) to five figures, so the ladder and this table
        // agree where they overlap.
        5.538702475e-6, 5.537564556e-6, 5.542684363e-6,
        7, false,
        "calibrate-gpu-reduction",
        "data/regime/calibrate_gpu_reduction.csv",
        true,
    },
    {
        "2gpu-1node",
        "2 GPUs, 1 node",
        2, 1, 2,
        ReductionTier::DEVICE_P2P,
        // calibrate-gpu-p2p --tier device-p2p. Both V100s of one node, spanning the two
        // GPU/NUMA domains: GPU0 on NUMA 0, GPU1 on NUMA 1, linked SYS, i.e. PCIe plus a
        // cross-socket UPI hop. No NVLink on this cluster.
        1.124922050e-5, 1.113722375e-5, 1.145958500e-5,
        7, false,
        "calibrate-gpu-p2p",
        "data/regime/calibrate_gpu_p2p_device.csv",
        true,
    },
    {
        "2gpu-2node",
        "2 GPUs, 2 nodes",
        2, 2, 1,
        ReductionTier::NODE,
        // calibrate-ca-participants, allreduce/norm on one double: the reduction Arnoldi
        // actually performs, rather than a bandwidth payload. synge-n[01-02], one GPU each.
        2.006445450e-5, 1.991615250e-5, 2.150410275e-5,
        7, false,
        "calibrate-gpu-p2p (participants)",
        "data/ca-participant-calibration-quiet/node_2participants.csv",
        true,
    },
    {
        "4gpu-2node",
        "4 GPUs, 2 nodes",
        4, 2, 2,
        ReductionTier::NODE,
        // The same run at four participants: both GPUs of both nodes. 1.46x the two-node pair
        // at the same rung, which is the cost the tier index alone cannot see and the reason
        // this table exists.
        2.920146000e-5, 2.902727025e-5, 2.949645525e-5,
        7, false,
        "calibrate-gpu-p2p (participants)",
        "data/ca-participant-calibration-quiet/node_4participants.csv",
        true,
    },
}};

/// The named arrangement, or nullptr when the key is not one this cluster supports.
[[nodiscard]] inline const GpuTopology* lookup_topology(std::string_view key) noexcept
{
    for (const auto& t : kSyngeTopologies)
        if (t.key == key) return &t;
    return nullptr;
}

/**
 * @brief Whether a launch's observed placement matches what the topology declares.
 *
 * The gate the collection script applies before a row is accepted. A two-node launch that the
 * scheduler quietly packed onto one node reduces at the wrong rung and would place a point
 * that looks measured and is not, so the mismatch has to be an error rather than a note.
 */
[[nodiscard]] inline constexpr bool topology_matches(const GpuTopology& t, int observed_nodes,
                                                     int observed_local_gpus) noexcept
{
    return t.nodes == observed_nodes && t.local_gpus == observed_local_gpus;
}

// The slab decomposition
/**
 * @brief Nonzeros of the real Basket operator's 19-point stencil on the z-planes [z0, z0+k).
 *
 * The operator is a three-dimensional Black--Scholes basket in log-price coordinates: a
 * seven-point axis stencil plus the twelve mixed-derivative cross terms, nineteen points in
 * the interior. Counted exactly rather than as 19 per row, because the boundary planes lose
 * neighbors and at n=25 that is a 6% correction to every byte model downstream.
 *
 * Rows are what a participant owns, so a neighbor living on another slab still contributes its
 * nonzero here; only the z-extent of the *owned* rows enters. Passing the whole range
 * reproduces the nnz column of data/regime/regime_placement.csv exactly at every accepted
 * grid, which is the check that this count and the measured table describe one operator.
 *
 * @param n1 points per axis of the cubic global grid.
 * @param z0 first owned plane.
 * @param k  number of owned planes.
 */
[[nodiscard]] inline constexpr int64_t basket_nnz_slab(int n1, int z0, int k) noexcept
{
    if (n1 <= 0 || k <= 0) return 0;
    const int64_t n  = n1;
    const int64_t kk = k;
    // Owned planes that have a neighbor one step along +z / -z somewhere in the global grid.
    const int64_t up = std::max<int64_t>(0, std::min<int64_t>(z0 + k, n - 1) - z0);
    const int64_t dn = std::max<int64_t>(0, (z0 + k) - std::max<int64_t>(z0, 1));
    const int64_t z_pair = up + dn;

    return kk * n * n                       // the diagonal
         + 2 * kk * n * (n - 1)             // +-x
         + 2 * kk * n * (n - 1)             // +-y
         + z_pair * n * n                   // +-z
         + 4 * kk * (n - 1) * (n - 1)       // +-x +-y
         + 2 * z_pair * n * (n - 1)         // +-x +-z
         + 2 * z_pair * n * (n - 1);        // +-y +-z
}

/// Nonzeros of the whole operator: the slab count over every plane.
[[nodiscard]] inline constexpr int64_t basket_nnz(int n1) noexcept
{
    return basket_nnz_slab(n1, 0, n1);
}

/**
 * @brief Planes owned by the most heavily loaded participant of a z-slab decomposition.
 *
 * One-dimensional in z, matching gpu_pde_slab.cuh, which is the decomposition the distributed
 * matrix-powers kernel is written against. The remainder is spread one plane at a time, so the
 * heaviest slab carries ceil(n1 / participants); that participant sets the pace of every
 * reduction, which is why the map's vertical coordinate is a max rather than a mean.
 */
[[nodiscard]] inline constexpr int slab_planes_max(int n1, int participants) noexcept
{
    if (participants <= 1) return n1;
    return (n1 + participants - 1) / participants;
}

/// Rows owned by the most heavily loaded participant.
[[nodiscard]] inline constexpr int64_t slab_points_max(int n1, int participants) noexcept
{
    return static_cast<int64_t>(slab_planes_max(n1, participants))
         * static_cast<int64_t>(n1) * static_cast<int64_t>(n1);
}

/**
 * @brief Nonzeros owned by the most heavily loaded participant.
 *
 * Placed at an interior offset when there is room for one, since an interior slab carries more
 * nonzeros than a face slab and the max is what the coordinates need. With one participant the
 * slab is the whole grid and the faces are included, as they must be.
 */
[[nodiscard]] inline constexpr int64_t slab_nnz_max(int n1, int participants) noexcept
{
    const int k = slab_planes_max(n1, participants);
    if (k >= n1) return basket_nnz(n1);
    const int z0 = std::min(1, n1 - k);   // one plane in, where the grid allows it
    return basket_nnz_slab(n1, z0, k);
}

// Placement
/// A point of the trajectory figure: one (policy, topology, grid) key, placed and gated.
struct GpuTrajectoryPoint {
    const GpuTopology* topology = nullptr;
    int     n1_global      = 0;
    int64_t n_global       = 0;
    int64_t nnz_global     = 0;
    int64_t n_local_max    = 0;    ///< rows on the heaviest slab: the R_v numerator's operand
    int64_t nnz_local_max  = 0;
    int     m              = 0;    ///< measured converged Krylov dimension of the global solve
    int     planes_max     = 0;
    double  working_set    = 0.0;  ///< bytes, on the heaviest slab
    double  l2             = 0.0;  ///< one device's L2; never the sum over the allocation
    double  rv             = 0.0;
    double  rh             = 0.0;
    double  rh_q1          = 0.0;  ///< R_h from the collective's q1, so the spread is drawable
    double  rh_q3          = 0.0;
    double  cycle_s        = 0.0;  ///< modeled local Arnoldi cycle
    double  compute_between_reductions_s = 0.0;
    int64_t reductions     = 0;    ///< R(m) on the unmodified MGS baseline
    /// Whether each kernel's working set still fits one device's L2. The roof drops
    /// 4.4x on the V100 when it stops fitting, so a refinement trajectory bends
    /// sharply at the grid where this turns false; recorded so that bend is
    /// auditable rather than mysterious.
    bool    spmv_resident  = false;
    bool    mgs_resident   = false;
    RooflineVerdict gate_spmv;
    RooflineVerdict gate_mgs;
    RooflineVerdict gate_gram;
    bool    on_map         = false;
    bool    accepted       = false; ///< every gate held and the topology is calibrated
};

/**
 * @brief Place one operator, on one topology, at one grid.
 *
 * Both coordinates are taken on the heaviest slab. The vertical because caches are private, so
 * the reuse a participant gets is bounded by its own L2; the horizontal because the collective
 * cannot start until the slowest participant arrives, so the compute between two reductions is
 * the slowest local cycle rather than the average one.
 *
 * Nothing here sees a timer. `m` is a measured property of the operator and the stopping
 * tolerance rather than of the host, the collective cost is a measured machine constant, and
 * the compute term is the same predeclared byte model the CPU map uses. The predictor/outcome
 * firewall is what makes the resulting point a test of where CA pays rather than a restatement
 * of a run that already happened.
 *
 * @param gm  the device preset; its L2 and roofs are the machine capacities.
 * @param t   the arrangement, carrying its own measured collective cost.
 * @param n1  points per axis of the cubic global grid.
 * @param m   measured converged Krylov dimension of the global solve at this grid.
 * @param s   certified block width, for the Gram-matrix gate only.
 */
[[nodiscard]] inline GpuTrajectoryPoint place_gpu_trajectory(
    const GpuMachine& gm, const GpuTopology& t, int n1, int m,
    double x_reuse = 1.0, Precision p = Precision::FP64, int s = 8)
{
    GpuTrajectoryPoint pt;
    pt.topology      = &t;
    pt.n1_global     = n1;
    pt.n_global      = grid_points(n1, 3);
    pt.nnz_global    = basket_nnz(n1);
    pt.planes_max    = slab_planes_max(n1, t.participants);
    pt.n_local_max   = slab_points_max(n1, t.participants);
    pt.nnz_local_max = slab_nnz_max(n1, t.participants);
    pt.m             = m;

    const int P = gm.sm_count;   // each participant runs the whole device on its own slab
    pt.working_set = static_cast<double>(
        arnoldi_working_set_bytes(pt.nnz_local_max, pt.n_local_max, m));
    pt.l2 = static_cast<double>(aggregate_l2_bytes(gm, P));
    pt.rv = R_v(gm, P, arnoldi_working_set_bytes(pt.nnz_local_max, pt.n_local_max, m));

    const GpuCycleTime ct = gpu_arnoldi_cycle_seconds(
        gm, P, pt.nnz_local_max, pt.n_local_max, m, x_reuse, p);
    pt.cycle_s    = ct.total_s;
    pt.reductions = mgs_reductions(m);
    pt.compute_between_reductions_s =
        pt.reductions > 0 ? ct.total_s / static_cast<double>(pt.reductions) : 0.0;

    // The measured collective for this exact arrangement, not reduction_cost_s(gm, tier):
    // the rung alone cannot separate two participants on two nodes from four.
    const double denom = pt.compute_between_reductions_s;
    pt.rh    = denom > 0.0 ? t.t_reduce_s    / denom : 0.0;
    pt.rh_q1 = denom > 0.0 ? t.t_reduce_q1_s / denom : 0.0;
    pt.rh_q3 = denom > 0.0 ? t.t_reduce_q3_s / denom : 0.0;

    pt.spmv_resident = ct.spmv_resident;
    pt.mgs_resident  = ct.mgs_resident;
    pt.gate_spmv = ct.gate_spmv;
    pt.gate_mgs  = ct.gate_mgs;
    pt.gate_gram = roofline_gate(gm, P, gram_intensity(pt.n_local_max, s), p);
    pt.on_map    = pt.gate_spmv.memory_bound && pt.gate_mgs.memory_bound;
    pt.accepted  = pt.on_map && t.calibrated && !t.contended
                 && gm.roofline_gated && m > 0 && pt.rh > 0.0;
    return pt;
}
