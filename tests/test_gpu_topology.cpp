/**
 * @file test_gpu_topology.cpp
 * @brief The H200 preset's transcribed constants, and the recorded arrangements that stop one
 *        participant count's collective cost from standing in for another's.
 *
 * Two subjects, and they are one subject. The reason the H200 preset carries no DEVICE_P2P
 * increment is the reason kGpuTopologies exists: eight devices of one node cost 2.01x what four
 * of the same node cost, at the same rung, so a constant indexed by rung alone cannot hold
 * both. Every test here defends one half of that split.
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
/// archived invocation on gpu03.
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
    REQUIRE_THAT(gm.fp64_flops_peak * 1e-12, WithinRel(30.743, 1e-3));
    REQUIRE_THAT(gm.fp32_flops_peak * 1e-12, WithinRel(61.409, 1e-3));
    REQUIRE_THAT(gm.fp32_flops_peak / gm.fp64_flops_peak, WithinRel(2.0, 0.01));

    // Achieved bandwidths, and the gap the vertical mechanism converts. The cache/DRAM ratio
    // is 3.2x here against the V100's 4.4x, so the same L2 residency buys less.
    REQUIRE_THAT(gm.hbm_bw_gbs_achieved, WithinRel(4054.6, 1e-4));
    REQUIRE_THAT(gm.l2_bw_gbs_achieved,  WithinRel(13014.5, 1e-4));
    REQUIRE(gm.hbm_bw_gbs_achieved < gm.hbm_bw_gbs);   // measured below theoretical
    REQUIRE_THAT(gm.l2_bw_gbs_achieved / gm.hbm_bw_gbs_achieved, WithinRel(3.21, 0.01));
    REQUIRE(gm.roofline_gated);

    // The overlap capabilities are unknown, not assumed. gpu-device-probe has not run on
    // gpu03, and an architectural guess here would look exactly like a measurement.
    REQUIRE_FALSE(gm.overlap.recorded);
    REQUIRE(gm.overlap.async_engine_count == -1);
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

    // The link bandwidth is deliberately absent for the same reason: 314.4 GB/s at four
    // participants against 373.8 at eight is not one number about a link.
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
    // 5.2567 us. That equality is the whole point of transcribing one invocation rather than
    // per-rung medians: the model reproduces a total that was actually observed.
    REQUIRE_THAT(reduction_cost_s(gm, ReductionTier::GRID) * 1e6, WithinRel(5.2567, 1e-4));

    // Launch is 36% of a grid reduction, so the rung structure stays visible behind it. Above
    // about 90% the ladder would collapse to a launch constant and R_h would have no shape.
    REQUIRE(launch_share(gm, ReductionTier::GRID) < 0.5);
    REQUIRE(tier_multiplier(gm, ReductionTier::GRID) > 5.0);
}

/// The replicate policy, restated as a test so it cannot quietly change.
/// Expected: the recorded grid total is the median of the three archived invocations, and is
/// not the per-rung-median total, which no invocation produced.
TEST_CASE("the h200 ladder is one invocation, not an average of three",
          "[regime][gpu][preset]")
{
    const GpuMachine& gm = lookup_gpu_machine("h200");

    // The three independent invocations on gpu03, by two-kernel grid total [us]:
    //   2026-08-25 transcript 5.2086, 2026-08-26 10:57 5.2567, 2026-08-26 10:50 5.5528.
    const std::vector<double> observed_us {5.2086, 5.256711377, 5.552817716};
    std::vector<double> sorted = observed_us;
    std::sort(sorted.begin(), sorted.end());
    const double recorded_us = reduction_cost_s(gm, ReductionTier::GRID) * 1e6;

    REQUIRE_THAT(recorded_us, WithinRel(sorted[1], 1e-4));

    // The spread is real and material: 1.066x on the total, 1.128x on the grid increment
    // alone. A preset claiming the within-run interquartile range of one invocation, which
    // spans 0.06%, would assert a precision this host does not have.
    REQUIRE(sorted.back() / sorted.front() > 1.05);

    // Per-rung medians would accumulate to 5.2888 us, a grid cost no invocation produced.
    // The recorded value must not be that number.
    constexpr double kPerRungMedianTotalUs = 5.288833;
    REQUIRE(std::abs(recorded_us - kPerRungMedianTotalUs) > 1e-3);
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

/// The H200 rows carry a real measurement that must not place a point, which is the case the
/// whole provenance apparatus exists for.
/// Expected: the eight-GPU row places, the four-GPU row is kept and blocked.
TEST_CASE("the h200 table blocks the transcript-only arrangement",
          "[regime][gpu][topology]")
{
    REQUIRE(arrangements_of("h200").size() == 3);

    const GpuTopology* one   = lookup_topology("h200", "1gpu-1node");
    const GpuTopology* four  = lookup_topology("h200", "4gpu-1node");
    const GpuTopology* eight = lookup_topology("h200", "8gpu-1node");
    REQUIRE(one   != nullptr);
    REQUIRE(four  != nullptr);
    REQUIRE(eight != nullptr);

    // The four-GPU row is internally consistent, uncontended and carries a real median: the
    // only thing wrong with it is that its raw CSV was overwritten and its devices were never
    // named. That must be enough to block it, and must not be enough to delete it.
    REQUIRE(topology_consistent(*four));
    REQUIRE_FALSE(four->contended);
    REQUIRE_THAT(four->t_reduce_s * 1e6, WithinRel(16.336, 1e-4));
    REQUIRE_FALSE(four->calibrated);
    REQUIRE(four->link_signature == "unknown");
    REQUIRE_FALSE(topology_placeable(*four));

    REQUIRE(topology_placeable(*one));
    REQUIRE(topology_placeable(*eight));

    // The finding the table exists to preserve: same rung, same node, twice the cost at twice
    // the participants, while bus bandwidth moves only 18.9% the other way. Neither number may
    // be used for the other's participant count.
    REQUIRE(eight->tier == four->tier);
    REQUIRE(eight->nodes == four->nodes);
    REQUIRE_THAT(eight->t_reduce_s / four->t_reduce_s, WithinRel(2.011, 0.001));
    REQUIRE_THAT(eight->interconnect_bw_gbs / four->interconnect_bw_gbs,
                 WithinRel(1.189, 0.001));

    // The single-device row publishes the spread over three invocations, not one run's
    // interquartile range. Its quartiles must therefore be far wider than that run's.
    REQUIRE(one->invocations == 3);
    REQUIRE(one->t_reduce_q3_s / one->t_reduce_q1_s > 1.05);
}

/// A launcher that has counted its hosts and devices must reach the row for that exact shape,
/// and must reach nothing when the shape was never measured.
/// Expected: lookup by allocation matches on the split, and an unmeasured count is absent.
TEST_CASE("an unmeasured participant count has no row to borrow", "[regime][gpu][topology]")
{
    const GpuTopology* eight = lookup_topology("h200", GpuAllocation{1, 8});
    REQUIRE(eight != nullptr);
    REQUIRE(eight->key == "8gpu-1node");

    // Nobody has calibrated two H200s, so there is no row and the caller stops. Returning the
    // four- or eight-GPU figure here is precisely the substitution this port removes.
    REQUIRE(lookup_topology("h200", GpuAllocation{1, 2}) == nullptr);
    REQUIRE(lookup_topology("h200", "2gpu-1node") == nullptr);

    // Nor has anyone run two nodes, so the fabric rung has no row at any count.
    REQUIRE(lookup_topology("h200", GpuAllocation{2, 8}) == nullptr);
    REQUIRE(lookup_topology("h200", "16gpu-2node") == nullptr);

    // One machine's arrangements are never another's, even where the key is spelled the same.
    // Both clusters record a 1gpu-1node, and they differ by 5%.
    const GpuTopology* synge_one = lookup_topology("v100-pcie-16gb", "1gpu-1node");
    const GpuTopology* h200_one  = lookup_topology("h200", "1gpu-1node");
    REQUIRE(synge_one != h200_one);
    REQUIRE(synge_one->t_reduce_s != h200_one->t_reduce_s);
    REQUIRE(lookup_topology("rtx-3090", "1gpu-1node") == nullptr);
}
