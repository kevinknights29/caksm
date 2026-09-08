/**
 * @file gpu_allocation.hpp
 * @brief The shape of the allocation a launch actually received, and which reduction rungs
 *        that shape can reach.
 *
 * gpu_machine.hpp describes one device. gpu_topology.hpp describes arrangements that have been
 * measured. This header describes the third thing, which used to have no owner: how many nodes
 * and how many local GPUs the scheduler handed this process right now.
 *
 * Reachability used to be read off the preset, from a `gpu_count` and a `node_count` stored
 * beside the SM count. That is correct only on a cluster whose shape never varies. On synge it
 * never did, so the error was invisible; on an H200 node the same preset must serve a one-GPU
 * development shell, a four-GPU subset, and the full eight-GPU node, and a preset-stored
 * maximum answers the wrong question in all three. Worse, it answers it plausibly: a preset
 * claiming eight GPUs makes DEVICE_P2P look reachable inside a one-GPU allocation, and the
 * ladder then prices a link the launch never crosses.
 *
 * So the rule is inverted. The preset says what one device can do; the allocation says what
 * this launch can reach; nothing about the cluster is stored in the preset at all. A caller
 * that has observed its allocation passes it. A caller that has not observed one has no
 * business publishing a DEVICE_P2P or NODE cost.
 *
 * The separation also fixes a provenance leak. Collective latency is keyed to the exact
 * participant count, not to the rung: eight H200s on one node reduce at the same DEVICE_P2P
 * rung as four and cost 1.89x as much. `GpuMachine::t_reduce_s` is indexed by rung alone and
 * cannot hold both, which is why a machine preset may carry the on-device rungs only, and the
 * off-device cost belongs to a measured arrangement in gpu_topology.hpp.
 *
 * @author Kevin Knights
 * @date 2026-08-27
 */
#pragma once

#include <stdexcept>
#include <string>

#include "gpu_machine.hpp"

/**
 * @brief The nodes and local GPUs one launch actually received.
 *
 * Both fields are observations, not requests. `local_gpus` is the count per node and the
 * allocation is assumed symmetric, which is what every launch path here produces and what the
 * trajectory figure's participant axis presumes; an asymmetric allocation has no single
 * participant count and must be rejected upstream rather than averaged into one here.
 */
struct GpuAllocation {
    int nodes;       ///< distinct hosts taking part
    int local_gpus;  ///< GPUs per host

    /// GPUs taking part in the collective, which on this code path is also the rank count.
    [[nodiscard]] constexpr int participants() const noexcept { return nodes * local_gpus; }
};

/// A validated allocation. Rejects the shapes that would silently mis-price a rung.
[[nodiscard]] inline GpuAllocation make_allocation(int nodes, int local_gpus)
{
    if (nodes < 1 || local_gpus < 1)
        throw std::invalid_argument(
            "allocation requires nodes >= 1 and local_gpus >= 1, got "
            + std::to_string(nodes) + " x " + std::to_string(local_gpus));
    return {nodes, local_gpus};
}

/**
 * @brief The allocation implied by a world size spread over a node count.
 *
 * The form an MPI launch has to hand: it counts its ranks and its distinct hostnames and needs
 * the per-node figure. Refuses a world that does not divide evenly, since an uneven spread is
 * an asymmetric allocation wearing a symmetric label.
 */
[[nodiscard]] inline GpuAllocation allocation_from_world(int world_gpus, int nodes)
{
    if (world_gpus < 1 || nodes < 1 || nodes > world_gpus)
        throw std::invalid_argument("allocation requires 1 <= nodes <= GPUs");
    if (world_gpus % nodes != 0)
        throw std::invalid_argument(
            "asymmetric allocation: " + std::to_string(world_gpus) + " GPU(s) over "
            + std::to_string(nodes) + " node(s) is not a whole number per node");
    return {nodes, world_gpus / nodes};
}

/// The canonical key for an arrangement, matching gpu_topology.hpp: "8gpu-1node".
[[nodiscard]] inline std::string allocation_key(const GpuAllocation& a)
{
    return std::to_string(a.participants()) + "gpu-" + std::to_string(a.nodes) + "node";
}

// Tier reachability
/**
 * @brief Whether this allocation can exercise a rung at all.
 *
 * The on-device rungs are always reachable; a single GPU still reduces across its warps,
 * blocks and grid. The two link rungs need the hardware to be there in this launch, which is
 * the question the preset could not answer.
 */
[[nodiscard]] inline constexpr bool tier_reachable(const GpuAllocation& a,
                                                   ReductionTier t) noexcept
{
    switch (t) {
        case ReductionTier::WARP:
        case ReductionTier::BLOCK:
        case ReductionTier::GRID:       return true;
        case ReductionTier::DEVICE_P2P: return a.local_gpus > 1;
        case ReductionTier::NODE:       return a.nodes > 1;
    }
    return false;
}

/**
 * @brief The rung an observed solver topology actually exercises.
 *
 * More than one host crosses NODE, more than one local GPU crosses DEVICE_P2P, and a lone
 * device stops at GRID. Independent of the MPI process count on purpose: one process may own
 * several GPUs, and a one-GPU process is not a P2P topology however many ranks surround it.
 */
[[nodiscard]] inline constexpr ReductionTier collective_tier(const GpuAllocation& a) noexcept
{
    if (a.nodes > 1)      return ReductionTier::NODE;
    if (a.local_gpus > 1) return ReductionTier::DEVICE_P2P;
    return ReductionTier::GRID;
}

/// The world-size form, for launch paths that have counted ranks and hosts rather than
/// built an allocation.
[[nodiscard]] inline ReductionTier collective_tier_for_topology(int world_gpus, int node_count)
{
    return collective_tier(allocation_from_world(world_gpus, node_count));
}

/// The most expensive rung this allocation can place a point on: the right-hand end of the
/// swept horizontal axis.
[[nodiscard]] inline constexpr ReductionTier highest_reachable_tier(
    const GpuAllocation& a) noexcept
{
    if (a.nodes > 1)      return ReductionTier::NODE;
    if (a.local_gpus > 1) return ReductionTier::DEVICE_P2P;
    return ReductionTier::GRID;
}

/// A cost may be published only when this allocation reaches the rung and the preset measured
/// it. An off-device rung the preset does not carry is not an error; it means the cost lives
/// in a measured arrangement instead. See gpu_topology.hpp.
[[nodiscard]] inline constexpr bool tier_cost_available(const GpuMachine& gm,
                                                        const GpuAllocation& a,
                                                        ReductionTier tier) noexcept
{
    return tier_reachable(a, tier) && gm.tier_calibrated[tier_index(tier)];
}

/**
 * @brief The most expensive rung that is both reachable here and measured on this preset.
 *
 * Distinct from highest_reachable_tier(). An uncalibrated rung carries a zero increment, so
 * reduction_cost_s() would return the cost of the rung below it while the caller believes it
 * asked for the higher one: a plausible number attached to the wrong hardware.
 */
[[nodiscard]] inline constexpr ReductionTier highest_calibrated_tier(
    const GpuMachine& gm, const GpuAllocation& a) noexcept
{
    ReductionTier top = ReductionTier::WARP;
    for (int i = 0; i < kTierCount; ++i) {
        const auto t = static_cast<ReductionTier>(i);
        if (tier_reachable(a, t) && gm.tier_calibrated[tier_index(t)]) top = t;
    }
    return top;
}

/// Whether every rung this allocation reaches carries a measured cost on this preset. The
/// honest precondition for trusting an R_h magnitude at the top of the ladder.
[[nodiscard]] inline constexpr bool all_reachable_tiers_calibrated(
    const GpuMachine& gm, const GpuAllocation& a) noexcept
{
    for (int i = 0; i < kTierCount; ++i) {
        const auto t = static_cast<ReductionTier>(i);
        if (tier_reachable(a, t) && !gm.tier_calibrated[tier_index(t)]) return false;
    }
    return true;
}
