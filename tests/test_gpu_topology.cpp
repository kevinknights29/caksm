/**
 * @file test_gpu_topology.cpp
 * @brief The H200 preset's transcribed constants, and the recorded arrangements that stop one
 *        participant count's collective cost from standing in for another's.
 *
 * The two subjects are one subject. The H200 preset carries no DEVICE_P2P increment for the
 * same reason kGpuTopologies exists: eight devices of one node cost 1.89x what four of the
 * same node cost, at the same rung, so a constant indexed by rung alone cannot hold both.
 * Every test here defends one half of that split.
 *
 * These are drift guards. They restate the archived measurements as literals, so an edit that
 * changes a magnitude has to change a test too and becomes a deliberate act. They check
 * relationships, not plausibility: a wrong constant that is merely plausible is exactly what a
 * predict-then-measure instrument cannot tolerate.
 *
 * The table's own well-formedness is a static_assert in gpu_topology.hpp rather than a case
 * here, so a mistyped entry fails the build instead of a test run. What the cases below add is
 * the part a consistency check cannot see: whether the numbers are the ones that were measured.
 *
 * @author Kevin Knights
 * @date 2026-08-27
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <vector>

#include "gpu_allocation.hpp"
#include "gpu_machine.hpp"
#include "gpu_topology.hpp"

using Catch::Matchers::WithinRel;

namespace {

constexpr GpuAllocation kOneDevice {1, 1};
constexpr GpuAllocation kH200Node  {1, 8};

/// Every arrangement recorded for one machine, in table order.
[[nodiscard]] std::vector<const GpuTopology*> arrangements_of(std::string_view machine)
{
    std::vector<const GpuTopology*> out;
    for (const auto& t : kGpuTopologies)
        if (t.machine == machine) out.push_back(&t);
    return out;
}

}  // namespace

// The H200 preset: guard the transcribed constants against silent drift

/// Every point placed on this card inherits these numbers, and all of them came from one
/// archived invocation.
/// Expected: the recorded fields agree with the archived CSVs and with each other.
TEST_CASE("the calibrated h200 preset is self-consistent", "[regime][gpu][preset]")
{
    const GpuMachine& gm = lookup_gpu_machine("h200");

    // Geometry, transcribed from the calibrator's own banner.
    REQUIRE(gm.sm_count == 132);
    REQUIRE(gm.l2_bytes == 60L << 20);
    REQUIRE(gm.shared_mem_bytes_per_sm == 228L << 10);

    // The footprint ceiling is rounded down from the reported 139.8 GiB, so it can only
    // under-state the device. A ceiling that over-states capacity would wave through a
    // configuration that does not fit.
    REQUIRE(gm.device_memory_bytes <= static_cast<int64_t>(139.8 * (1L << 30)));

    // Measured roofs. FP64 is first-class at 2:1, which is what puts the H200 on the same
    // arm as the V100 rather than the 3090.
    REQUIRE_THAT(gm.fp64_flops_peak * 1e-12, WithinRel(30.750, 1e-3));
    REQUIRE_THAT(gm.fp32_flops_peak * 1e-12, WithinRel(61.414, 1e-3));
    REQUIRE_THAT(gm.fp32_flops_peak / gm.fp64_flops_peak, WithinRel(2.0, 0.01));

    // Achieved bandwidths, and the gap the vertical mechanism converts. The cache/DRAM ratio
    // is 3.2x here against the V100's 4.4x, so the same L2 residency buys less.
    REQUIRE_THAT(gm.hbm_bw_gbs_achieved, WithinRel(4053.1, 1e-4));
    REQUIRE_THAT(gm.l2_bw_gbs_achieved,  WithinRel(12988.1, 1e-4));
    REQUIRE(gm.hbm_bw_gbs_achieved < gm.hbm_bw_gbs);   // measured below theoretical
    REQUIRE_THAT(gm.l2_bw_gbs_achieved / gm.hbm_bw_gbs_achieved, WithinRel(3.204, 0.01));
    REQUIRE(gm.roofline_gated);

    // The overlap capabilities are recorded, not assumed: gpu-device-probe ran on all eight
    // devices of gpu01 and they agreed. Three copy engines, concurrent kernels, and unified
    // addressing, which is what a copy/compute overlap claim would have to rest on.
    REQUIRE(gm.overlap.recorded);
    REQUIRE(gm.overlap.async_engine_count == 3);
    REQUIRE(gm.overlap.concurrent_kernels == 1);
    REQUIRE(gm.overlap.unified_addressing == 1);
}

/// The preset must carry the on-device ladder and nothing above it, because nothing above it
/// is a property of a device on this node.
/// Expected: WARP, BLOCK and GRID measured; DEVICE_P2P and NODE zero and unmeasured.
TEST_CASE("the h200 preset carries the on-device ladder only", "[regime][gpu][preset]")
{
    const GpuMachine& gm = lookup_gpu_machine("h200");

    REQUIRE(gm.reduction_calibrated);
    REQUIRE(gm.tier_calibrated[tier_index(ReductionTier::WARP)]);
    REQUIRE(gm.tier_calibrated[tier_index(ReductionTier::BLOCK)]);
    REQUIRE(gm.tier_calibrated[tier_index(ReductionTier::GRID)]);
    REQUIRE_FALSE(gm.tier_calibrated[tier_index(ReductionTier::DEVICE_P2P)]);
    REQUIRE_FALSE(gm.tier_calibrated[tier_index(ReductionTier::NODE)]);
    REQUIRE(gm.t_reduce_s[tier_index(ReductionTier::DEVICE_P2P)] == 0.0);
    REQUIRE(gm.t_reduce_s[tier_index(ReductionTier::NODE)] == 0.0);

    // The link bandwidth is deliberately absent for the same reason: 274.6 GB/s at two
    // participants against 373.9 at eight is not one number about a link.
    REQUIRE(gm.interconnect_bw_gbs == 0.0);
    REQUIRE(gm.interconnect == "NV18");

    // Each on-device rung must cost strictly more than the one below, or it is not a distinct
    // point on the swept horizontal axis.
    double prev = -1.0;
    for (int i = 0; i <= static_cast<int>(ReductionTier::GRID); ++i) {
        const double t = reduction_cost_s(gm, static_cast<ReductionTier>(i));
        REQUIRE(t > prev);
        prev = t;
    }

    // The accumulated grid rung must reproduce the two-kernel total the calibrator measured,
    // 5.3770 us. That equality is the whole point of transcribing one invocation rather than
    // per-rung statistics: the model reproduces a total that was actually observed.
    REQUIRE_THAT(reduction_cost_s(gm, ReductionTier::GRID) * 1e6, WithinRel(5.3770, 1e-4));

    // Launch is 36% of a grid reduction, so the rung structure stays visible behind it. Above
    // about 90% the ladder would collapse to a launch constant and R_h would have no shape.
    REQUIRE(launch_share(gm, ReductionTier::GRID) < 0.5);
    REQUIRE(tier_multiplier(gm, ReductionTier::GRID) > 5.0);
}

/// The replicate policy, restated as a test so it cannot quietly change.
/// Expected: the recorded increments accumulate to a two-kernel total that was measured, and
/// not to the per-rung average of several runs, which no run produces.
TEST_CASE("the h200 ladder is one invocation, not a per-rung average",
          "[regime][gpu][preset]")
{
    const GpuMachine& gm = lookup_gpu_machine("h200");
    const double recorded_us = reduction_cost_s(gm, ReductionTier::GRID) * 1e6;

    // The calibrator measured 5.376960 us for the two-kernel grid form, and the four
    // recorded entries must sum back to it.
    REQUIRE_THAT(recorded_us, WithinRel(5.376960, 1e-4));

    // Why one invocation and not a per-rung statistic. On an archived three-invocation set
    // the per-rung average accumulated to 5.2888 us while the measured totals were 5.2086,
    // 5.2567 and 5.5528: the average of rungs is a ladder no run ever exhibited. Whatever the
    // selection rule, the entries transcribed here must come from a single run together.
    constexpr double kPerRungAverageUs = 5.288833;
    REQUIRE(std::abs(recorded_us - kPerRungAverageUs) > 1e-3);

    // The selection rule is "the invocation nearest the mean of the invocations", which uses
    // every run to locate the center while still transcribing one. Checked on the set above,
    // the only evidence with several invocations: it picks 5.2567, agreeing with the median.
    const std::vector<double> invocations {5.2086, 5.256711377, 5.552817716};
    const double mean = (invocations[0] + invocations[1] + invocations[2]) / 3.0;
    const double nearest = *std::min_element(
        invocations.begin(), invocations.end(),
        [mean](double a, double b) { return std::abs(a - mean) < std::abs(b - mean); });
    REQUIRE_THAT(nearest, WithinRel(5.256711377, 1e-9));

    // The spread the rule exists to respect is real: those three invocations spanned 1.066x
    // on this total.
    REQUIRE(*std::max_element(invocations.begin(), invocations.end())
          / *std::min_element(invocations.begin(), invocations.end()) > 1.05);
}

/// Calibration completeness is a question about the launch, and the H200 answers it two
/// different ways on the same hardware.
/// Expected: complete on one device, incomplete on the eight-device node.
TEST_CASE("h200 ladder completeness depends on the allocation", "[regime][gpu][preset]")
{
    const GpuMachine& gm = lookup_gpu_machine("h200");

    // A one-GPU development shell reaches only the on-device rungs, so the ladder is whole.
    REQUIRE(all_reachable_tiers_calibrated(gm, kOneDevice));
    REQUIRE(highest_calibrated_tier(gm, kOneDevice) == ReductionTier::GRID);
    REQUIRE(highest_reachable_tier(kOneDevice) == ReductionTier::GRID);

    // The full node reaches the link rung, which this preset does not carry. The gap must be
    // visible rather than papered over: highest_calibrated_tier stops one rung short, and
    // asking for the link cost from the preset is refused.
    REQUIRE_FALSE(all_reachable_tiers_calibrated(gm, kH200Node));
    REQUIRE(highest_reachable_tier(kH200Node) == ReductionTier::DEVICE_P2P);
    REQUIRE(highest_calibrated_tier(gm, kH200Node) == ReductionTier::GRID);
    REQUIRE_FALSE(tier_cost_available(gm, kH200Node, ReductionTier::DEVICE_P2P));
}

/// The roofline gate decides whether a point is on the map at all, and the H200's answer must
/// be a measured one.
/// Expected: the baseline kernels are memory-bound, and the CA Gram kernel is too, which puts
/// the H200 on the same arm as the V100.
TEST_CASE("the h200 places on the map with a measured roof", "[regime][gpu][preset]")
{
    const GpuMachine& gm = lookup_gpu_machine("h200");

    // The ridge is measured on both sides, so no verdict here is provisional.
    const double ridge = ridge_ai(gm, Precision::FP64);
    REQUIRE_THAT(ridge, WithinRel(30.743e12 / (4054.6 * 1e9), 1e-3));

    const RooflineVerdict spmv = roofline_gate(gm, 0.135, Precision::FP64);
    const RooflineVerdict mgs  = roofline_gate(gm, 0.375, Precision::FP64);
    REQUIRE(spmv.memory_bound);
    REQUIRE(mgs.memory_bound);
    REQUIRE_FALSE(spmv.provisional);

    // The Gram kernel CA adds has intensity about s/4 and is what the 3090 fails on at s ~ 2.4.
    // The H200 clears it well past the certified s_max of 9, so the negative arm does not
    // reappear on this card and the treatment stays on the map.
    REQUIRE(gram_ridge_s(gm, Precision::FP64) > 9.0);
}

// The recorded arrangements

/// The V100 trajectory must reproduce from the machine-keyed table that replaced
/// kSyngeTopologies, or the generalization silently republished a different figure.
/// Expected: four arrangements, with the constants the retired array carried.
TEST_CASE("the synge arrangements reproduce the retired topology table",
          "[regime][gpu][topology]")
{
    REQUIRE(arrangements_of("v100-pcie-16gb").size() == 4);

    struct Expected { const char* key; int participants; int nodes; int local; double us; };
    constexpr std::array<Expected, 4> kExpected {{
        {"1gpu-1node", 1, 1, 1,  5.538702475},
        {"2gpu-1node", 2, 1, 2, 11.249220500},
        {"2gpu-2node", 2, 2, 1, 20.064454500},
        {"4gpu-2node", 4, 2, 2, 29.201460000},
    }};
    for (const auto& e : kExpected) {
        const GpuTopology* t = lookup_topology("v100-pcie-16gb", e.key);
        REQUIRE(t != nullptr);
        REQUIRE(t->participants == e.participants);
        REQUIRE(t->nodes == e.nodes);
        REQUIRE(t->local_gpus == e.local);
        REQUIRE_THAT(t->t_reduce_s * 1e6, WithinRel(e.us, 1e-7));
        REQUIRE(topology_placeable(*t));
    }

    // The single-GPU entry is the grid reduction, and must still agree with the preset's
    // ladder to five figures. That equality is what says the table and the ladder describe one
    // mechanism where they overlap.
    const GpuMachine& v100 = lookup_gpu_machine("v100-pcie-16gb");
    REQUIRE_THAT(lookup_topology("v100-pcie-16gb", "1gpu-1node")->t_reduce_s,
                 WithinRel(reduction_cost_s(v100, ReductionTier::GRID), 1e-5));

    // Two participants on two nodes and four on two nodes reduce at the same rung and differ
    // by 1.46x. That difference is the reason a row exists per arrangement.
    const GpuTopology* two  = lookup_topology("v100-pcie-16gb", "2gpu-2node");
    const GpuTopology* four = lookup_topology("v100-pcie-16gb", "4gpu-2node");
    REQUIRE(two->tier == ReductionTier::NODE);
    REQUIRE(four->tier == ReductionTier::NODE);
    REQUIRE_THAT(four->t_reduce_s / two->t_reduce_s, WithinRel(1.46, 0.01));
}

/// The H200 participant sweep, all four points from one host in one job.
/// Expected: the collective very nearly doubles with the participant count while bus bandwidth
/// climbs, which is the ring signature R_h's rise depends on.
TEST_CASE("the h200 collective scales with the participant count",
          "[regime][gpu][topology]")
{
    REQUIRE(arrangements_of("h200").size() == 8);

    const GpuTopology* one   = lookup_topology("h200", "1gpu-1node");
    const GpuTopology* two   = lookup_topology("h200", "2gpu-1node");
    const GpuTopology* four  = lookup_topology("h200", "4gpu-1node");
    const GpuTopology* eight = lookup_topology("h200", "8gpu-1node");
    for (const GpuTopology* t : {one, two, four, eight}) {
        REQUIRE(t != nullptr);
        REQUIRE(topology_placeable(*t));
        REQUIRE_FALSE(t->contended);
        REQUIRE(t->repeats >= 7);
    }

    // Every point comes from one host, which is what makes the participant axis a controlled
    // comparison: mixing hosts would put a node effect inside the axis.
    for (const GpuTopology* t : {one, two, four, eight})
        REQUIRE(t->cluster == "h200-gpu01");

    // Measured medians [us], transcribed by scripts/regime/topology_entry.sh.
    REQUIRE_THAT(two->t_reduce_s   * 1e6, WithinRel(10.5099, 1e-4));
    REQUIRE_THAT(four->t_reduce_s  * 1e6, WithinRel(20.6320, 1e-4));
    REQUIRE_THAT(eight->t_reduce_s * 1e6, WithinRel(39.0024, 1e-4));

    // Doubling the participants very nearly doubles the latency: 1.963x then 1.890x. That near
    // linearity is the finding, and it is why R_h climbs as a fixed problem is sharded wider.
    REQUIRE_THAT(four->t_reduce_s  / two->t_reduce_s,  WithinRel(1.963, 0.01));
    REQUIRE_THAT(eight->t_reduce_s / four->t_reduce_s, WithinRel(1.890, 0.01));

    // Bus bandwidth moves the other way, rising 36% from two participants to eight. Latency and
    // throughput are therefore separate facts about the link, and a row must carry both.
    REQUIRE(two->interconnect_bw_gbs < four->interconnect_bw_gbs);
    REQUIRE(four->interconnect_bw_gbs < eight->interconnect_bw_gbs);
    REQUIRE_THAT(eight->interconnect_bw_gbs / two->interconnect_bw_gbs, WithinRel(1.362, 0.01));

    // Every count above one crosses the link; one participant does not leave the device.
    REQUIRE(one->tier == ReductionTier::GRID);
    for (const GpuTopology* t : {two, four, eight})
        REQUIRE(t->tier == ReductionTier::DEVICE_P2P);

    // The four-GPU subset names its devices and its link signature. Without both, a
    // calibration cannot be tied to the subset it was taken over.
    REQUIRE(four->link_signature == "NV18:0-3");
    REQUIRE(four->devices == "0-3");
    REQUIRE(four->calibrated);
}

/// The launch structure is a field, not a remark, because on this machine it is worth more
/// than several of the effects the table exists to separate.
/// Expected: the two families are distinguishable, and the same participant count on the same
/// hardware carries two different costs depending only on how the processes were arranged.
TEST_CASE("the h200 table distinguishes launch structure from topology",
          "[regime][gpu][topology]")
{
    // One node: ncclCommInitAll, so one process holds every local device.
    for (const char* key : {"2gpu-1node", "4gpu-1node", "8gpu-1node"}) {
        const GpuTopology* t = lookup_topology("h200", key);
        REQUIRE(t != nullptr);
        REQUIRE(t->gpus_per_rank == t->local_gpus);
        REQUIRE(t->transport == "nccl-p2p");
    }
    // Two nodes: MPI, so each rank holds exactly one device.
    for (const char* key : {"2gpu-2node", "4gpu-2node", "8gpu-2node", "16gpu-2node"}) {
        const GpuTopology* t = lookup_topology("h200", key);
        REQUIRE(t != nullptr);
        REQUIRE(t->gpus_per_rank == 1);
        REQUIRE(t->transport == "nccl-net-ib");
    }

    // Eight participants, two arrangements, and they differ in BOTH the node split and the
    // launch structure. That is the confound, and it is large: 39.00 us against 18.24.
    //
    // Two framings of the same gap. These rows give 2.14x, since the device and fabric rows
    // come from different jobs; within the single job that measured both it is 2.10x
    // (38.32 / 18.24). The 2% between them is the node's own repeatability, so neither
    // framing changes the conclusion.
    //
    // Neither cause can be attributed on this evidence. The control that separates them is
    // eight MPI ranks on ONE node, which MPI_ON_ONE_NODE=1 in calibrate_gpu_p2p.sh now runs
    // and which would carry gpus_per_rank == 1 with nodes == 1.
    const GpuTopology* one_process = lookup_topology("h200", "8gpu-1node");
    const GpuTopology* many_ranks  = lookup_topology("h200", "8gpu-2node");
    REQUIRE(one_process->participants == many_ranks->participants);
    REQUIRE(one_process->gpus_per_rank != many_ranks->gpus_per_rank);
    REQUIRE(one_process->nodes != many_ranks->nodes);
    REQUIRE_THAT(one_process->t_reduce_s / many_ranks->t_reduce_s, WithinRel(2.138, 0.01));

    // That control has not been measured, so the arrangement it would produce is absent. An
    // absent row is the correct representation of an unmeasured one.
    bool mpi_on_one_node = false;
    for (const auto& t : kGpuTopologies)
        if (t.machine == "h200" && t.nodes == 1 && t.gpus_per_rank == 1 && t.participants > 1)
            mpi_on_one_node = true;
    REQUIRE_FALSE(mpi_on_one_node);
}

/// A launcher that has counted its hosts and devices must reach the row for that exact shape,
/// and must reach nothing when the shape was never measured.
/// Expected: lookup by allocation matches on the split, and an unmeasured count is absent.
TEST_CASE("an unmeasured participant count has no row to borrow", "[regime][gpu][topology]")
{
    const GpuTopology* eight = lookup_topology("h200", GpuAllocation{1, 8});
    REQUIRE(eight != nullptr);
    REQUIRE(eight->key == "8gpu-1node");

    // Nobody has calibrated three H200s, so there is no row and the caller stops. Returning a
    // neighboring count here is precisely the substitution this port removes.
    REQUIRE(lookup_topology("h200", GpuAllocation{1, 3}) == nullptr);
    REQUIRE(lookup_topology("h200", "3gpu-1node") == nullptr);

    // Two nodes ARE calibrated now, at 1, 2, 4 and 8 GPUs per node, so the fabric rung
    // resolves and the sixteen-participant arrangement is a real row.
    const GpuTopology* sixteen = lookup_topology("h200", GpuAllocation{2, 8});
    REQUIRE(sixteen != nullptr);
    REQUIRE(sixteen->key == "16gpu-2node");
    REQUIRE(sixteen->tier == ReductionTier::NODE);

    // Counts between the calibrated ones are still absent, on either split, and a third node
    // has never been allocated at all.
    REQUIRE(lookup_topology("h200", GpuAllocation{2, 3}) == nullptr);
    REQUIRE(lookup_topology("h200", "6gpu-2node") == nullptr);
    REQUIRE(lookup_topology("h200", GpuAllocation{4, 8}) == nullptr);

    // The same participant count on two different splits is two different rows carrying two
    // different costs, which is the whole reason the table is keyed by arrangement and not by
    // count: eight on one node cost 38.32 us, eight over two nodes 18.24 us.
    const GpuTopology* eight_1n = lookup_topology("h200", GpuAllocation{1, 8});
    const GpuTopology* eight_2n = lookup_topology("h200", GpuAllocation{2, 4});
    REQUIRE(eight_2n != nullptr);
    REQUIRE(eight_1n->participants == eight_2n->participants);
    REQUIRE(eight_1n->t_reduce_s != eight_2n->t_reduce_s);
    REQUIRE(eight_1n->tier != eight_2n->tier);

    // One machine's arrangements are never another's, even where the key is spelled the same.
    // Both clusters record a 1gpu-1node, and they differ by 5%.
    const GpuTopology* synge_one = lookup_topology("v100-pcie-16gb", "1gpu-1node");
    const GpuTopology* h200_one  = lookup_topology("h200", "1gpu-1node");
    REQUIRE(synge_one != h200_one);
    REQUIRE(synge_one->t_reduce_s != h200_one->t_reduce_s);
    REQUIRE(lookup_topology("rtx-3090", "1gpu-1node") == nullptr);
}
