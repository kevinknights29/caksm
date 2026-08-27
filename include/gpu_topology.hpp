/**
 * @file gpu_topology.hpp
 * @brief The measured arrangements of participants over devices and nodes, and the slab
 *        decomposition that turns a global grid into the per-GPU working set the vertical
 *        axis measures.
 *
 * gpu_machine.hpp describes one device. gpu_allocation.hpp describes the shape a launch
 * received. This header describes the third thing: an arrangement someone has actually
 * measured a collective on, and what it cost.
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
 *      the NODE rung and differ by 1.5x, because the ring is longer. Eight H200s of one node
 *      cost 2.01x what four of the same node cost, at the same DEVICE_P2P rung.
 *      `GpuMachine::t_reduce_s` cannot express either: it is indexed by rung alone. Each
 *      arrangement therefore carries its own measured collective latency, transcribed from the
 *      run that measured that exact arrangement.
 *
 * The table is keyed by machine, so one build serves every calibrated cluster and adding a
 * cluster adds rows rather than editing anyone else's. It is compiled in rather than loaded,
 * for the same reason gpu_machine.hpp's presets are: a measured constant entering the model is
 * a deliberate, reviewable edit, and a constant that can change without a diff is a constant
 * nobody is checking. What a person must not do is retype the numbers. Every field below that
 * a run already recorded is produced by scripts/regime/topology_entry.sh, which reads an
 * archived run directory and prints the entry ready to paste; only the judgments are typed.
 *
 * The provenance rule is inherited unchanged from gpu_machine.hpp: geometry is transcribed,
 * every latency is measured offline on an idle device over at least seven repeats, and the
 * source file of each measurement is named beside it. An arrangement whose collective cost has
 * not been measured for its own arrangement carries `calibrated = false`, and the trajectory
 * tool refuses to place it rather than borrowing a neighbouring rung's constant.
 *
 * @author Kevin Knights
 * @date 2026-08-23
 */
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <string_view>

#include "gpu_allocation.hpp"
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
    std::string_view machine;   ///< the GpuMachine key these devices are presets of
    std::string_view cluster;   ///< the installation, so two hosts of one card do not merge
    std::string_view key;       ///< canonical name, "8gpu-1node"; unique within a machine
    std::string_view label;     ///< how the figure names it

    int participants;           ///< GPUs taking part in the collective
    int nodes;                  ///< distinct hosts spanned
    int local_gpus;             ///< GPUs per host; participants == nodes * local_gpus
    std::string_view devices;   ///< the logical device ordinals this arrangement selected

    ReductionTier tier;         ///< the rung this arrangement reduces at

    /// Measured cost of one global reduction [s], the median of at least seven repeats on an
    /// idle device, with the spread this record publishes beside it. With `invocations == 1`
    /// the quartiles are the within-run interquartile range; with more they are the extremes
    /// over the invocation medians, which is the wider and the honest figure. Seven
    /// back-to-back repeats measure how steady one launch was, not how repeatable the constant
    /// is, so wherever both are known the between-invocation spread is the one to draw.
    double t_reduce_s;
    double t_reduce_q1_s;
    double t_reduce_q3_s;

    double interconnect_bw_gbs; ///< all-reduce bus bandwidth at THIS participant count
    int    repeats;             ///< repeats behind each invocation's median
    int    invocations;         ///< independent runs behind the published figure
    bool   contended;           ///< the idle-device gate; true blocks the point

    /// True only when this exact arrangement was measured and its evidence is complete. False
    /// keeps the row here as evidence while suppressing the point, which is what a run whose
    /// raw CSV was overwritten deserves: visible, and unplaceable.
    bool   calibrated;

    /// The observed local link matrix reduced to one token, then the devices it was taken
    /// over. It is what stops a calibration on one four-GPU subset from being applied to a
    /// differently connected four-GPU subset of the same node. "unknown" is a legitimate value
    /// and, with `calibrated`, is what blocks such a row from placing a point.
    std::string_view link_signature;
    std::string_view transport;    ///< the path NCCL selected: "nccl-p2p", "nccl-net-ib", ...
    std::string_view instrument;   ///< the binary that produced the number
    std::string_view source;       ///< repo-relative file the number was transcribed from
};

/**
 * The arrangements measured so far, in the order the trajectory figure connects them.
 *
 * synge, ordered by collective reach, which on that cluster is also ordered by cost: 5.54 us on
 * one device, 11.25 us across the two GPU/NUMA domains of one node, 20.06 us across two nodes,
 * and 29.20 us once both GPUs of both nodes take part. The monotonicity is the hypothesis panel
 * C tests, so it is worth naming that it is visible in the constants before any point is
 * placed; what the figure adds is whether the motion in R_h is large enough to cross theta_h.
 *
 * A single-participant entry is not an NCCL measurement and must not be one. With one
 * participant no collective leaves the device, so the global reduction IS the grid reduction,
 * and the number is the two-kernel grid form from calibrate-gpu-reduction.
 *
 * `link_signature` is "<link label>:<devices>". Two counts are absent for the H200 rather than
 * interpolated: nobody has calibrated two H200s or a sixteen-participant two-node run, so
 * lookup returns nothing and the caller stops, which is what should happen.
 */
inline constexpr std::array<GpuTopology, 7> kGpuTopologies {{
    {
        "v100-pcie-16gb", "synge", "1gpu-1node", "1 GPU, 1 node",
        1, 1, 1, "0", ReductionTier::GRID,
        // Equals reduction_cost_s(v100, GRID) to five figures, so the ladder and this table
        // agree where they overlap.
        5.538702475e-6, 5.537564556e-6, 5.542684363e-6,
        0.0, 7, 1, false, true,
        // synge's node has one pair, so a two-GPU arrangement has no other subset it could be
        // confused with and the signature carries no UUIDs. On a node that does, they must.
        "none:0", "on-device",
        "calibrate-gpu-reduction", "data/regime/calibrate_gpu_reduction.csv",
    },
    {
        "v100-pcie-16gb", "synge", "2gpu-1node", "2 GPUs, 1 node",
        2, 1, 2, "0-1", ReductionTier::DEVICE_P2P,
        // Both V100s of one node, spanning the two GPU/NUMA domains: GPU0 on NUMA 0, GPU1 on
        // NUMA 1, linked SYS, i.e. PCIe plus a cross-socket UPI hop. No NVLink on this cluster.
        1.124922050e-5, 1.113722375e-5, 1.145958500e-5,
        7.36, 7, 1, false, true,
        "SYS:0-1", "nccl-p2p",
        "calibrate-gpu-p2p", "data/regime/calibrate_gpu_p2p_device.csv",
    },
    {
        "v100-pcie-16gb", "synge", "2gpu-2node", "2 GPUs, 2 nodes",
        2, 2, 1, "0", ReductionTier::NODE,
        // allreduce/norm on one double: the reduction Arnoldi actually performs, rather than a
        // bandwidth payload. synge-n01 and synge-n02, one GPU each, over NET/IB on hfi1_0.
        2.006445450e-5, 1.991615250e-5, 2.150410275e-5,
        7.00, 7, 1, false, true,
        "SYS:0-per-node", "nccl-net-ib",
        "calibrate-gpu-p2p (participants)",
        "data/ca-participant-calibration-quiet/node_2participants.csv",
    },
    {
        "v100-pcie-16gb", "synge", "4gpu-2node", "4 GPUs, 2 nodes",
        4, 2, 2, "0-1", ReductionTier::NODE,
        // The same run at four participants: both GPUs of both nodes. 1.46x the two-node pair
        // at the same rung, which is the cost the tier index alone cannot see and the reason
        // this table exists.
        2.920146000e-5, 2.902727025e-5, 2.949645525e-5,
        7.00, 7, 1, false, true,
        "SYS:0-1-per-node", "nccl-net-ib",
        "calibrate-gpu-p2p (participants)",
        "data/ca-participant-calibration-quiet/node_4participants.csv",
    },
    {
        "h200", "h200-gpu03", "1gpu-1node", "1 GPU, 1 node",
        1, 1, 1, "0", ReductionTier::GRID,
        // The two-kernel grid form of the accepted invocation. The quartiles are NOT that
        // run's interquartile range, which spans only 5.2550 to 5.2582 us: they are the
        // extremes over the three independent invocations of the single-device calibrator on
        // this host, 5.2086 us (2026-08-25, transcript only) to 5.5528 us (2026-08-26 10:50).
        // Publishing the narrow figure would assert a precision this host does not have.
        5.256711377e-6, 5.208600000e-6, 5.552817716e-6,
        0.0, 7, 3, false, true,
        "none:0", "on-device", "calibrate-gpu-reduction",
        "data/calibration/h200/2026-08-26T1057-job20170/calibrate_gpu_reduction.csv",
    },
    {
        "h200", "h200-gpu03", "4gpu-1node", "4 GPUs, 1 node",
        4, 1, 4, "unknown", ReductionTier::DEVICE_P2P,
        // A real measurement, 2026-08-25, inside a `salloc -N 1 --gpus=4` allocation: seven
        // repeats on an idle device. It is nevertheless NOT calibrated, for a reason that is
        // not about the number. The eight-participant run wrote the same fixed filename and
        // destroyed the raw CSV, so only the stdout transcript survives; the allocation's
        // device UUIDs were never recorded, and on a node whose halves have different NUMA and
        // NIC affinity an unnamed four-GPU subset is not a reproducible arrangement. The row
        // is kept because deleting it would lose the evidence that four participants cost
        // 2.011x less than eight; it is blocked because a figure must not treat it as a point.
        // Re-measure it with the devices named and confirm 16.336 us within the predeclared
        // tolerance rather than averaging any mismatch away.
        1.633600000e-5, 1.632000000e-5, 1.634000000e-5,
        314.4, 7, 1, false, false,
        "unknown", "nccl-p2p", "calibrate-gpu-p2p",
        "data/calibration/h200/2026-08-26T1057-job20170/calibrate_gpu_p2p.log",
    },
    {
        "h200", "h200-gpu03", "8gpu-1node", "8 GPUs, 1 node",
        8, 1, 8, "0-7", ReductionTier::DEVICE_P2P,
        // The full node, every device named and checksummed. Latency is 2.011x the
        // four-participant figure while bus bandwidth rises 18.9%, which is why these are two
        // arrangements and not two samples of one constant. nvidia-smi topo -m reports NV18
        // between every one of the 28 GPU pairs, so the GPU-to-GPU graph is uniform; the
        // device set still matters, since GPUs 0-3 sit on NUMA 0 with mlx5_0..3 closest and
        // GPUs 4-7 on NUMA 1 with mlx5_4..7.
        3.284996450e-5, 3.283777425e-5, 3.287139625e-5,
        373.8386, 7, 1, false, true,
        "NV18:0-7", "nccl-p2p", "calibrate-gpu-p2p",
        "data/calibration/h200/2026-08-26T1057-job20170/calibrate_gpu_p2p_device.csv",
    },
}};

/// The arrangement of one machine under a name, or nullptr when it has not been measured. An
/// absent key is not an error: an arrangement the allocation never supplied has no row, and
/// interpolating one is exactly what this table exists to prevent.
[[nodiscard]] inline constexpr const GpuTopology* lookup_topology(
    std::string_view machine, std::string_view key) noexcept
{
    for (const auto& t : kGpuTopologies)
        if (t.machine == machine && t.key == key) return &t;
    return nullptr;
}

/// The arrangement of one machine matching an observed allocation, by shape rather than by
/// name. What a launcher needs once it has counted its hosts and devices and wants the row it
/// is allowed to publish against.
[[nodiscard]] inline constexpr const GpuTopology* lookup_topology(
    std::string_view machine, const GpuAllocation& a) noexcept
{
    for (const auto& t : kGpuTopologies)
        if (t.machine == machine && t.nodes == a.nodes && t.local_gpus == a.local_gpus)
            return &t;
    return nullptr;
}

/// The arrangement as an allocation, for the reachability and rung questions.
[[nodiscard]] inline GpuAllocation topology_allocation(const GpuTopology& t)
{
    return make_allocation(t.nodes, t.local_gpus);
}

/**
 * @brief Whether a launch's observed placement matches what the arrangement declares.
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

/**
 * @brief Whether a row is internally consistent.
 *
 * A static_assert-able check, so a mistyped entry is a build error rather than a plausible
 * number placed on a figure. The rung is checked against the split because a row naming
 * DEVICE_P2P on one local GPU describes an arrangement that cannot exist, and would price a
 * link the launch never crosses.
 */
[[nodiscard]] inline constexpr bool topology_consistent(const GpuTopology& t) noexcept
{
    return t.participants > 0 && t.nodes > 0 && t.local_gpus > 0
        && t.participants == t.nodes * t.local_gpus
        && t.tier == collective_tier({t.nodes, t.local_gpus})
        && t.t_reduce_s > 0.0
        && t.t_reduce_q1_s <= t.t_reduce_s && t.t_reduce_s <= t.t_reduce_q3_s
        && t.repeats > 0 && t.invocations > 0;
}

/// Whether a row may place a point: consistent, measured on an idle device, and carrying the
/// physical evidence that says which devices it was measured over.
[[nodiscard]] inline constexpr bool topology_placeable(const GpuTopology& t) noexcept
{
    return topology_consistent(t) && t.calibrated && !t.contended
        && !t.link_signature.empty() && t.link_signature != "unknown";
}

/// Every row is consistent, and no machine repeats a key. Checked at build time, because a
/// duplicate key would silently resolve to whichever entry came first.
consteval bool gpu_topologies_well_formed()
{
    for (std::size_t i = 0; i < kGpuTopologies.size(); ++i) {
        if (!topology_consistent(kGpuTopologies[i])) return false;
        for (std::size_t j = 0; j < i; ++j)
            if (kGpuTopologies[i].machine == kGpuTopologies[j].machine
                && kGpuTopologies[i].key == kGpuTopologies[j].key) return false;
    }
    return true;
}
static_assert(gpu_topologies_well_formed(),
              "a GpuTopology entry is inconsistent, or one machine repeats a topology key");

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
    pt.accepted  = pt.on_map && topology_placeable(t)
                 && gm.roofline_gated && m > 0 && pt.rh > 0.0;
    return pt;
}
