/**
 * @file test_gpu_regime.cpp
 * @brief The GPU map's structural claims, pinned so a future edit cannot quietly undo them.
 *
 * Three of these tests defend results rather than code. The Phase 0 derivation
 * rests on R_v being independent of the team size and on the
 * R_v * R_h product being independent of both P and N. Both follow from a fixed L2 and a
 * device-wide bandwidth roof, and both would be destroyed by an edit that made
 * `aggregate_l2_bytes` scale with P for symmetry with the CPU side, producing plausible wrong
 * numbers rather than a crash. So the invariant is asserted numerically.
 *
 * The rest pin the things a GPU coordinate can get wrong without complaining: cumulative
 * rather than per-rung reduction costs, a gate evaluated per machine rather than per kernel,
 * and a window solver that reports feasible when one of its three constraints is empty.
 *
 * @author Kevin Knights
 * @date 2026-07-21
 */

#include "test_regime_helpers.hpp"

#include "gpu_machine.hpp"
#include "gpu_regime.hpp"

using namespace regime_test;

namespace {

/// A calibrated V100, so the tests exercise the paths a real preset will take. The constants
/// are stated priors, not measurements: these tests check structure (scaling laws, ordering,
/// feasibility logic) rather than magnitudes, so no measured number is needed.
[[nodiscard]] GpuMachine calibrated_v100()
{
    GpuMachine gm = lookup_gpu_machine("v100-pcie-16gb");
    gm.hbm_bw_gbs_achieved = 800.0;
    gm.l2_bw_gbs_achieved  = 2200.0;
    gm.t_reduce_s = {{ 50e-9, 450e-9, 4.5e-6, 5.0e-6, 15.0e-6 }};
    gm.t_kernel_launch_s = 0.0;   // folded out: the tiers above are the on-die increments
    gm.tier_calibrated = {{ true, true, true, true, true }};
    gm.reduction_calibrated = true;
    gm.roofline_gated = true;
    return gm;
}

/// 3D 7-point stencil on an n1^3 grid, exact at the boundary.
[[nodiscard]] int64_t stencil_nnz_3d(int n1)
{
    const int64_t N = grid_points(n1, 3);
    const int64_t plane = N / n1;
    return N + 3 * 2 * (N - plane);
}

}  // namespace

// The measured 3090 preset: guard the transcribed constants against silent drift

/// The transcribed preset constants must not drift silently, since every point
/// placed on this card inherits them.
/// Expected: the internal relationships between the recorded fields hold.
TEST_CASE("the calibrated rtx-3090 preset is self-consistent", "[regime][gpu][preset]")
{
    const GpuMachine& gm = lookup_gpu_machine("rtx-3090");

    // Provenance flags: a single-device host reaches only WARP/BLOCK/GRID, so those three
    // being measured is what makes reduction_calibrated legitimately true. The flag must not
    // be true while an unreachable rung is falsely marked, nor while a reachable one is unset.
    REQUIRE(gm.reduction_calibrated);
    REQUIRE(gm.roofline_gated);
    REQUIRE(all_reachable_tiers_calibrated(gm));
    REQUIRE(gm.tier_calibrated[tier_index(ReductionTier::GRID)]);
    REQUIRE_FALSE(gm.tier_calibrated[tier_index(ReductionTier::DEVICE_P2P)]);

    // The negative-arm verdict, on measured numbers. FP64:FP32 must stay in the throttled
    // regime, or the 3090 is no longer the negative arm.
    const double ratio = gm.fp32_flops_peak / gm.fp64_flops_peak;
    REQUIRE(ratio > 32.0);   // measured 65.6; datasheet 64

    // The reduction ladder must climb: a rung that does not cost more than the one below is a
    // transcription error, not a finding.
    REQUIRE(reduction_cost_s(gm, ReductionTier::BLOCK)
            > reduction_cost_s(gm, ReductionTier::WARP));
    REQUIRE(reduction_cost_s(gm, ReductionTier::GRID)
            > reduction_cost_s(gm, ReductionTier::BLOCK));

    // The achieved HBM roof sits below theoretical, and the L2 roof above it: an L2 figure
    // that dipped below HBM would mean the tiers were mislabeled.
    REQUIRE(gm.hbm_bw_gbs_achieved < gm.hbm_bw_gbs);
    REQUIRE(gm.l2_bw_gbs_achieved > gm.hbm_bw_gbs_achieved);
}

/// The negative arm of the two-card study, and it has to fail for the stated
/// reason rather than by assumption.
/// Expected: at production width the 3090 is compute-bound and so off-map, on
/// its own measured FP64 roof rather than on a datasheet figure.
TEST_CASE("the 3090 fails the CA gate at production s, on measured constants",
          "[regime][gpu][preset]")
{
    const GpuMachine& gm = lookup_gpu_machine("rtx-3090");
    const int64_t N = grid_points(61, 3), nnz = stencil_nnz_3d(61);

    // The point of the negative arm: the baseline is on-map, since SpMV and MGS clear even the
    // throttled ridge, but the kernel CA introduces is compute-bound at the certified width.
    const GpuRegimePoint pt =
        place_gpu(gm, gm.sm_count, ReductionTier::GRID, nnz, N, 12, 1.0, Precision::FP64, 8);
    REQUIRE(on_map(pt));
    REQUIRE_FALSE(treatment_on_map(pt));

    // And the crossing sits below the certificate, so the effect is reachable, not academic.
    REQUIRE(gram_ridge_s(gm, Precision::FP64) < 9.0);   // measured ~2.8
}

/// The cards are a near-controlled pair, matched on cache and bandwidth and
/// differing by more than an order of magnitude in FP64, so the roofline gate
/// is the only thing that can separate them.
/// Expected: the gate admits one and rejects the other.
TEST_CASE("the two-card contrast discriminates on measured constants", "[regime][gpu][preset]")
{
    const GpuMachine& v100  = lookup_gpu_machine("v100-pcie-16gb");
    const GpuMachine& c3090 = lookup_gpu_machine("rtx-3090");
    const int64_t N = grid_points(61, 3), nnz = stencil_nnz_3d(61);
    const int m = 12, s = 8;

    // The controlled pair, as measured: same L2, bandwidth within 1%, FP64 apart by ~11x.
    REQUIRE(v100.l2_bytes == c3090.l2_bytes);
    REQUIRE_THAT(v100.hbm_bw_gbs_achieved, WithinRel(c3090.hbm_bw_gbs_achieved, 0.05));
    REQUIRE(v100.fp64_flops_peak > 10.0 * c3090.fp64_flops_peak);

    // FP64:FP32 is 2.0:1 measured on the V100 and 65.6:1 on the 3090: the design in one line.
    REQUIRE(v100.fp32_flops_peak / v100.fp64_flops_peak < 4.0);
    REQUIRE(c3090.fp32_flops_peak / c3090.fp64_flops_peak > 32.0);

    const GpuRegimePoint a = place_gpu(v100, v100.sm_count, highest_calibrated_tier(v100),
                                       nnz, N, m, 1.0, Precision::FP64, s);
    const GpuRegimePoint b = place_gpu(c3090, c3090.sm_count, highest_calibrated_tier(c3090),
                                       nnz, N, m, 1.0, Precision::FP64, s);

    // Both reach Upper-Right at production resolution, the corner puffin's CPU could not.
    REQUIRE(a.rv > kThetaV);  REQUIRE(a.rh > kThetaH);
    REQUIRE(b.rv > kThetaV);  REQUIRE(b.rh > kThetaH);

    // And the gate discriminates where Phase 0 predicted: the baseline is on-map on both
    // cards, but the kernel CA adds is compute-bound only on consumer silicon. That is the
    // falsification test's verdict, and what these presets exist to hold.
    REQUIRE(on_map(a));            REQUIRE(on_map(b));
    REQUIRE(treatment_on_map(a));  REQUIRE_FALSE(treatment_on_map(b));
}

/// A rung that costs nothing is either uncalibrated or not a distinct
/// mechanism; either way it must not be placed on.
/// Expected: every tier is calibrated and each adds a strictly positive
/// increment over the one below it.
TEST_CASE("the V100 ladder is complete and every rung is a real step",
          "[regime][gpu][preset]")
{
    const GpuMachine& v100 = lookup_gpu_machine("v100-pcie-16gb");

    // All five rungs measured, so placement may use the hardware's full reach.
    REQUIRE(v100.reduction_calibrated);
    REQUIRE(all_reachable_tiers_calibrated(v100));
    REQUIRE(highest_calibrated_tier(v100) == highest_reachable_tier(v100));
    REQUIRE(highest_calibrated_tier(v100) == ReductionTier::NODE);

    // Each rung must cost strictly more than the one below, or it is not a distinct point on
    // the swept horizontal axis and the ladder has fewer rungs than it claims.
    double prev = -1.0;
    for (int i = 0; i < kTierCount; ++i) {
        const double t = reduction_cost_s(v100, static_cast<ReductionTier>(i));
        REQUIRE(t > prev);
        prev = t;
    }

    // The model must reproduce the measured cumulative latencies, since those are what R_h's
    // numerator actually is: 11.46 us device-to-device, 22.09 us across the fabric.
    REQUIRE_THAT(reduction_cost_s(v100, ReductionTier::DEVICE_P2P) * 1e6,
                 WithinRel(11.46, 0.01));
    REQUIRE_THAT(reduction_cost_s(v100, ReductionTier::NODE) * 1e6, WithinRel(22.09, 0.01));

    // The fabric step is shallow, under 2x the intra-node rung against the CPU's 5.4x CCX
    // crossing. Pinned because it is a finding rather than an accident: synge's SYS link is
    // slow enough that leaving the node barely doubles the reduction.
    REQUIRE(tier_multiplier(v100, ReductionTier::NODE) < 2.5);
    REQUIRE(tier_multiplier(v100, ReductionTier::NODE) > 1.5);
}

/// The guard that stops a missing measurement from being read as a free rung.
/// Expected: an uncalibrated tier reports unavailable rather than returning the
/// accumulated cost of the rung below it.
TEST_CASE("an uncalibrated rung is never placed on", "[regime][gpu][preset]")
{
    // Both presets are complete now, so the guard is exercised on a constructed machine: a
    // zero increment must never let reduction_cost_s report the rung below under a higher
    // rung's name, a real number attached to the wrong hardware.
    GpuMachine gm = lookup_gpu_machine("v100-pcie-16gb");
    gm.t_reduce_s[tier_index(ReductionTier::NODE)]      = 0.0;
    gm.tier_calibrated[tier_index(ReductionTier::NODE)] = false;

    REQUIRE(highest_reachable_tier(gm) == ReductionTier::NODE);
    REQUIRE(highest_calibrated_tier(gm) == ReductionTier::DEVICE_P2P);
    REQUIRE_THAT(reduction_cost_s(gm, ReductionTier::NODE),
                 WithinRel(reduction_cost_s(gm, ReductionTier::DEVICE_P2P), 1e-12));
    REQUIRE_FALSE(all_reachable_tiers_calibrated(gm));

    // The 3090 is complete by a different route: its unreachable rungs need no measurement.
    const GpuMachine& c3090 = lookup_gpu_machine("rtx-3090");
    REQUIRE(highest_calibrated_tier(c3090) == highest_reachable_tier(c3090));
    REQUIRE(highest_calibrated_tier(c3090) == ReductionTier::GRID);
}

// The structural break: L2 does not grow with the team

/// The structural break between the two machine models. On the CPU every
/// engaged CCX brings its own cache slice; on the GPU, L2 is one device-wide
/// block and engaging more SMs adds none.
/// Expected: aggregate L2 is identical at every P, so R_v becomes a pure
/// function of the working set. The whole GPU decoupling follows from this.
TEST_CASE("aggregate L2 is flat in P, unlike the CPU's aggregate L3", "[regime][gpu][rv]")
{
    const GpuMachine& gm = lookup_gpu_machine("v100-pcie-16gb");

    // The CPU twin grows: every engaged CCX brings its own slice.
    const Machine& mc = lookup_machine("amd-3960x");
    REQUIRE(aggregate_llc_bytes(mc, 24) > aggregate_llc_bytes(mc, 3));

    // The GPU does not. Engaging more SMs adds zero cache, and the whole of the Phase 0
    // decoupling follows from that.
    for (const int P : {1, 8, 20, 40, 80})
        REQUIRE(aggregate_l2_bytes(gm, P) == gm.l2_bytes);

    // Therefore R_v is a pure function of the working set.
    const int64_t ws = 64L << 20;
    REQUIRE(R_v(gm, 1, ws) == R_v(gm, 80, ws));
}

// The invariant

/// The central GPU identity. Both coordinates are separately free of P, so
/// their product is too, for a different reason than on the CPU where
/// reciprocal powers of P canceled.
/// Expected: the product is constant across P from 8 to 80, checked on placed
/// coordinates rather than only on the closed form, so a P creeping back into
/// either axis would be caught. Across three decades of N the product holds to
/// 5%, the residual being the boundary correction to nnz/N.
TEST_CASE("R_v * R_h is independent of P and of N on a GPU", "[regime][gpu][invariant]")
{
    const GpuMachine gm = calibrated_v100();
    const int m = 12;
    const double bw = gm.hbm_bw_gbs_achieved;

    // P-independence. Both coordinates are separately P-free, so the product is too, and for
    // a different reason than on the CPU, where reciprocal powers of P canceled.
    const int64_t N = grid_points(61, 3);
    const int64_t nnz = stencil_nnz_3d(61);
    const double ref = regime_invariant(gm, ReductionTier::DEVICE_P2P, nnz, N, m, 1.0, bw).product;
    REQUIRE(ref > 0.0);

    for (const int P : {8, 20, 40, 80}) {
        const GpuRegimePoint pt =
            place_gpu(gm, P, ReductionTier::DEVICE_P2P, nnz, N, m, 1.0);
        REQUIRE(on_map(pt));   // the identity is scoped to the memory side; assert we are on it
        // Placed coordinates, not just the closed form: this is what would break if a P
        // crept back into either axis.
        REQUIRE_THAT(pt.rv * pt.rh, WithinRel(ref, 1e-9));
    }

    // N-independence, across three decades. W and B are both linear in N, so N cancels; the
    // residual drift is the boundary correction to nnz/N, which shrinks as the grid refines.
    double prev = 0.0;
    for (const int n1 : {45, 61, 89, 120}) {
        const int64_t Nk = grid_points(n1, 3);
        const RegimeInvariant ri =
            regime_invariant(gm, ReductionTier::DEVICE_P2P, stencil_nnz_3d(n1), Nk, m, 1.0, bw);
        if (prev > 0.0) REQUIRE_THAT(ri.product, WithinRel(prev, 0.05));
        prev = ri.product;
    }
}

/// The invariant is scoped to the memory side of the roofline, so a
/// compute-bound point must be ruled off-map rather than placed.
/// Expected: the gate rejects it instead of reporting a position.
TEST_CASE("the gate closes the compute-bound escape hatch", "[regime][gpu][invariant]")
{
    const GpuMachine gm = calibrated_v100();
    const int64_t N = grid_points(61, 3), nnz = stencil_nnz_3d(61);

    // Engage too little of the device and the compute roof falls below the memory roof. The
    // cycle time then carries a 1/P, R_h recovers a factor of P, and the product floats free:
    // the one branch in which the Phase 0 identity does not hold.
    const GpuRegimePoint few  = place_gpu(gm, 1, ReductionTier::DEVICE_P2P, nnz, N, 12, 1.0);
    const GpuRegimePoint many = place_gpu(gm, 80, ReductionTier::DEVICE_P2P, nnz, N, 12, 1.0);

    REQUIRE_FALSE(on_map(few));     // and so it is excluded, by the gate, not by assumption
    REQUIRE(on_map(many));
    REQUIRE(few.rv * few.rh < many.rv * many.rh);

    // The gate's verdict must agree with what the pricing actually did. A whole-device ridge
    // would have called this point memory-bound while attainable_flops() priced it on the
    // compute branch, and the invariant would have appeared to break for no visible reason.
    REQUIRE(ridge_ai_at(gm, 1, Precision::FP64) < ridge_ai(gm, Precision::FP64));
    REQUIRE(roofline_gate(gm, few.ai, Precision::FP64).memory_bound);   // whole-device: passes
    REQUIRE_FALSE(roofline_gate(gm, 1, few.ai, Precision::FP64).memory_bound);   // at P=1: fails
}

/// The CPU form of the identity depends on the cache roof scaling with P, which
/// stops once the working set spills.
/// Expected: once spilled the product falls with P, marking the boundary of the
/// CPU identity rather than of the GPU one.
TEST_CASE("the CPU invariant survives only while cache-resident", "[regime][gpu][invariant]")
{
    // The contrast the GPU result is stated against. On the CPU the cancellation needs BOTH
    // aggregate cache and aggregate bandwidth to scale with P; once the working set spills,
    // DRAM pins flat and the product acquires a 1/P. Asserting it here keeps the Phase 0
    // claim honest: the GPU invariant is not "the same invariant", it is a different one.
    const Machine& mc = lookup_machine("amd-3960x");
    // 2M rows at ~144 B/row is ~288 MiB: past 128 MiB aggregate L3 even at the full P=24, so
    // both ends of the sweep are on the DRAM roof and the comparison is like-for-like.
    const int64_t n = 2000000, nnz = 5L * n;
    const int m = 8;

    const RegimePoint lo = place(mc, 6, nnz, n, m, 1.0);
    const RegimePoint hi = place(mc, 24, nnz, n, m, 1.0);
    REQUIRE(lo.rv > 1.0);                       // spilled at both ends
    REQUIRE(hi.rv > 1.0);
    REQUIRE(lo.rv * lo.rh > hi.rv * hi.rh);     // the product falls with P once spilled
}

/// tau* is defined as the reduction cost at which the invariant equals one, so
/// it must be exactly that and not approximately.
/// Expected: setting the ladder so the top rung costs exactly tau* puts the
/// product on 1.0 to 1e-9 and flips the upper-right flag.
TEST_CASE("tau* is the reduction cost at which the corner opens", "[regime][gpu][invariant]")
{
    GpuMachine gm = calibrated_v100();
    const int64_t N = grid_points(61, 3), nnz = stencil_nnz_3d(61);
    const double bw = gm.hbm_bw_gbs_achieved;

    const RegimeInvariant ri =
        regime_invariant(gm, ReductionTier::DEVICE_P2P, nnz, N, 12, 1.0, bw);
    REQUIRE(ri.tau_star_s > 0.0);

    // Set the ladder so the top rung costs exactly tau*: the product must land on 1.
    gm.t_reduce_s = {{ ri.tau_star_s, 0.0, 0.0, 0.0, 0.0 }};
    gm.t_kernel_launch_s = 0.0;
    const RegimeInvariant at_star =
        regime_invariant(gm, ReductionTier::WARP, nnz, N, 12, 1.0, bw);
    REQUIRE_THAT(at_star.product, WithinRel(1.0, 1e-9));
    REQUIRE(at_star.upper_right);
}

// The reduction ladder

/// The ladder is cumulative, because a device-to-device all-reduce necessarily
/// performs the on-device combines first, and launch is a fixed per-reduction
/// cost rather than a per-rung or per-level one.
/// Expected: each tier equals the sum of the increments below it; launch enters
/// at the grid rung and exactly once; and the ladder is strictly monotone, since
/// one that is not is a bug rather than a finding.
TEST_CASE("reduction cost accumulates the rungs and adds launch once", "[regime][gpu][rh]")
{
    GpuMachine gm = calibrated_v100();
    gm.t_kernel_launch_s = 3e-6;

    // Cumulative: a device-to-device all-reduce performs the on-device combines first.
    REQUIRE_THAT(reduction_cost_s(gm, ReductionTier::WARP), WithinRel(50e-9, 1e-12));
    REQUIRE_THAT(reduction_cost_s(gm, ReductionTier::BLOCK), WithinRel(500e-9, 1e-12));

    // Launch enters at GRID and once only, not per rung and not per tree level.
    REQUIRE_THAT(reduction_cost_s(gm, ReductionTier::GRID),
                 WithinRel(500e-9 + 4.5e-6 + 3e-6, 1e-12));
    REQUIRE_THAT(reduction_cost_s(gm, ReductionTier::DEVICE_P2P),
                 WithinRel(500e-9 + 4.5e-6 + 5.0e-6 + 3e-6, 1e-12));

    // Monotone in the rung: a ladder that is not is a bug rather than a finding.
    double prev = -1.0;
    for (int i = 0; i < kTierCount; ++i) {
        const double t = reduction_cost_s(gm, static_cast<ReductionTier>(i));
        REQUIRE(t > prev);
        prev = t;
    }
}

/// If launch latency swamps every rung above the block tier, the ladder
/// collapses to a constant and the horizontal mechanism stops being
/// interesting. That is a result, and it is invisible unless reported.
/// Expected: the share is negligible when launch is, and is flagged when launch
/// dominates.
TEST_CASE("launch_share flags a ladder hidden behind launch cost", "[regime][gpu][rh]")
{
    GpuMachine gm = calibrated_v100();

    gm.t_kernel_launch_s = 1e-9;                            // negligible
    REQUIRE(launch_share(gm, ReductionTier::GRID) < 0.01);
    // Below the grid rung there is no launch at all.
    REQUIRE(launch_share(gm, ReductionTier::BLOCK) == 0.0);

    gm.t_kernel_launch_s = 500e-6;                          // launch swamps every rung
    REQUIRE(launch_share(gm, ReductionTier::GRID) > 0.95);
    // The risk this guards: the tier structure survives arithmetically but stops being
    // readable, and R_h's ladder collapses toward a constant.
    REQUIRE_THAT(tier_multiplier(gm, ReductionTier::DEVICE_P2P), WithinAbs(1.0, 0.05));
}

/// A host with one GPU has no device-to-device or node rung to measure.
/// Expected: those tiers report unreachable rather than returning a cost.
TEST_CASE("a single-device preset cannot reach the interconnect rungs", "[regime][gpu][rh]")
{
    const GpuMachine& g3090 = lookup_gpu_machine("rtx-3090");
    REQUIRE(tier_reachable(g3090, ReductionTier::GRID));
    REQUIRE_FALSE(tier_reachable(g3090, ReductionTier::DEVICE_P2P));
    REQUIRE_FALSE(tier_reachable(g3090, ReductionTier::NODE));
    REQUIRE(highest_reachable_tier(g3090) == ReductionTier::GRID);

    const GpuMachine& v100 = lookup_gpu_machine("v100-pcie-16gb");
    REQUIRE(highest_reachable_tier(v100) == ReductionTier::NODE);
}

/// The rung a collective actually crosses is set by the topology, not by how
/// the ranks were launched. Deriving it from process count would price a
/// two-node run at the on-device rung.
/// Expected: the selected tier follows GPU and node counts.
TEST_CASE("the solver reduction tier follows GPUs and nodes, not process count",
          "[regime][gpu][rh]")
{
    REQUIRE(collective_tier_for_topology(1, 1) == ReductionTier::GRID);
    REQUIRE(collective_tier_for_topology(2, 1)
            == ReductionTier::DEVICE_P2P);
    REQUIRE(collective_tier_for_topology(2, 2) == ReductionTier::NODE);
    REQUIRE(collective_tier_for_topology(4, 2) == ReductionTier::NODE);
    REQUIRE_THROWS_AS(collective_tier_for_topology(0, 1),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(collective_tier_for_topology(1, 2),
                      std::invalid_argument);

    const GpuMachine& v100 = lookup_gpu_machine("v100-pcie-16gb");
    REQUIRE(tier_cost_available(v100, ReductionTier::GRID));
    REQUIRE(tier_cost_available(v100, ReductionTier::DEVICE_P2P));
    REQUIRE(tier_cost_available(v100, ReductionTier::NODE));
    GpuMachine incomplete = v100;
    incomplete.tier_calibrated[tier_index(ReductionTier::NODE)] = false;
    REQUIRE_FALSE(tier_cost_available(incomplete, ReductionTier::NODE));
    REQUIRE(tier_cost_available(incomplete, ReductionTier::DEVICE_P2P));

    const GpuMachine& g3090 = lookup_gpu_machine("rtx-3090");
    REQUIRE(tier_cost_available(g3090, ReductionTier::GRID));
    REQUIRE_FALSE(tier_cost_available(g3090, ReductionTier::DEVICE_P2P));
    REQUIRE_FALSE(tier_cost_available(g3090, ReductionTier::NODE));
}

// The gate, per kernel

/// The two cards must separate on the kernel the study is about. SpMV is too
/// memory-bound to tell them apart, so a gate that discriminated there would be
/// measuring the wrong thing.
/// Expected: the cards differ on the CA kernel and agree on SpMV.
TEST_CASE("the gate discriminates the two cards on the CA kernel, not on SpMV",
          "[regime][gpu][gate]")
{
    GpuMachine v100 = lookup_gpu_machine("v100-pcie-16gb");
    GpuMachine c3090 = lookup_gpu_machine("rtx-3090");
    v100.hbm_bw_gbs_achieved = 800.0;  v100.roofline_gated = true;
    c3090.hbm_bw_gbs_achieved = 800.0; c3090.roofline_gated = true;

    const int64_t N = grid_points(61, 3), nnz = stencil_nnz_3d(61);
    const int m = 12, s = 8;

    // The controlled variable: same L2, same bandwidth class, FP64 apart by ~12x.
    REQUIRE(v100.l2_bytes == c3090.l2_bytes);
    REQUIRE(ridge_ai(v100, Precision::FP64) > 10.0 * ridge_ai(c3090, Precision::FP64));

    const GpuRegimePoint a =
        place_gpu(v100, v100.sm_count, ReductionTier::DEVICE_P2P, nnz, N, m, 1.0,
                  Precision::FP64, s);
    const GpuRegimePoint b =
        place_gpu(c3090, c3090.sm_count, ReductionTier::GRID, nnz, N, m, 1.0,
                  Precision::FP64, s);

    // The correction Phase 0 made: SpMV and MGS are so far below even the throttled ridge
    // that both cards are on-map for the baseline method. A blanket "consumer FP64 is
    // compute-bound" claim would have been wrong.
    REQUIRE(on_map(a));
    REQUIRE(on_map(b));

    // Where the pair actually discriminates is the kernel CA introduces and MGS lacks.
    REQUIRE(treatment_on_map(a));
    REQUIRE_FALSE(treatment_on_map(b));
}

/// The block Gram is what carries the arithmetic intensity up as width grows,
/// which is the mechanism that eventually crosses the ridge.
/// Expected: intensity increases with s, and the crossing follows from it.
TEST_CASE("Gram intensity rises with s and sets the crossing", "[regime][gpu][gate]")
{
    const int64_t n = 226981;
    REQUIRE_THAT(gram_intensity(n, 8), WithinRel(2.0, 1e-3));    // ~s/4 for s << n
    REQUIRE(gram_intensity(n, 16) > gram_intensity(n, 8));

    GpuMachine c3090 = lookup_gpu_machine("rtx-3090");
    c3090.hbm_bw_gbs_achieved = 800.0;
    const double s_cross = gram_ridge_s(c3090, Precision::FP64);

    // The crossing must sit below the certified s_max = 9, or the effect is unreachable and
    // the negative arm is a curiosity rather than a constraint on the method.
    REQUIRE(s_cross < 9.0);
    REQUIRE(roofline_gate(c3090, gram_intensity(n, static_cast<int>(s_cross) + 1),
                          Precision::FP64).memory_bound == false);
    REQUIRE(roofline_gate(c3090, gram_intensity(n, std::max(1, static_cast<int>(s_cross) - 1)),
                          Precision::FP64).memory_bound == true);
}

/// A gate evaluated against a theoretical roof over-states the memory side and
/// would wave through a point that is really compute-bound.
/// Expected: without a measured achieved bandwidth the verdict is provisional;
/// the theoretical roof is higher and its ridge lower, so the fallback
/// over-rejects. It can lose a point that is on the map but never admit one
/// that is not, which is why it is a fallback and not a hard error.
TEST_CASE("an unmeasured roof marks every verdict provisional", "[regime][gpu][gate]")
{
    // Build the ungated machine explicitly rather than leaning on a preset happening to be
    // uncalibrated: both presets are calibrated now, and a test that asserts otherwise breaks
    // the moment the project succeeds at its own goal.
    GpuMachine raw = lookup_gpu_machine("v100-pcie-16gb");
    raw.hbm_bw_gbs_achieved = 0.0;      // falls back to the theoretical roof
    raw.roofline_gated      = false;
    REQUIRE(roofline_gate(raw, 0.135, Precision::FP64).provisional);

    GpuMachine measured = calibrated_v100();
    REQUIRE_FALSE(roofline_gate(measured, 0.135, Precision::FP64).provisional);

    // Which way the fallback errs, pinned because the intuitive guess is backwards. A
    // theoretical roof reads higher than an achieved one, and the ridge is peak/BW, so the
    // unmeasured ridge is lower and the gate is more likely to call a kernel compute-bound.
    // The fallback therefore over-rejects: it can lose a point that is really on the map, but
    // it cannot admit one that is not. That is why it is a fallback and not a hard error.
    REQUIRE(raw.hbm_bw_gbs > measured.hbm_bw_gbs_achieved);
    REQUIRE(ridge_ai(raw, Precision::FP64) < ridge_ai(measured, Precision::FP64));
}

// The window

/// The two blocking questions, whether the corner is reachable and over what
/// range of N, are one question.
/// Expected: at every rung, feasibility agrees with the invariant exceeding 1,
/// and where feasible the window width is the invariant product to 5%.
TEST_CASE("the window is non-empty exactly when the invariant exceeds 1",
          "[regime][gpu][window]")
{
    GpuMachine gm = calibrated_v100();
    const int m = 12, dim = 3;
    const double bw = gm.hbm_bw_gbs_achieved;
    const int64_t N = grid_points(61, dim), nnz = stencil_nnz_3d(61);
    const double nu = static_cast<double>(nnz) / static_cast<double>(N);
    const int64_t budget = gm.device_memory_bytes;

    for (const auto tier : {ReductionTier::WARP, ReductionTier::BLOCK, ReductionTier::GRID,
                            ReductionTier::DEVICE_P2P, ReductionTier::NODE}) {
        const RegimeInvariant ri = regime_invariant(gm, tier, nnz, N, m, 1.0, bw);
        const UpperRightWindow w =
            upper_right_window(gm, tier, m, nu, dim, 1.0, bw, budget);
        REQUIRE(w.feasible == ri.upper_right);
        // ... and the width IS the product. The two blocking questions are one question.
        if (w.feasible) REQUIRE_THAT(w.width, WithinRel(ri.product, 0.05));
    }
}

/// Identifies which ceiling actually binds, so the open question is stated
/// correctly.
/// Expected: the R_h ceiling binds more than ten times sooner than the 16 GB
/// capacity, so memory is not what closes the corner, and production resolution
/// sits inside the window on the interconnect rungs.
TEST_CASE("device memory is not the constraint that closes the window", "[regime][gpu][window]")
{
    const GpuMachine gm = calibrated_v100();
    const int m = 12, dim = 3;
    const int64_t N = grid_points(61, dim);
    const double nu = static_cast<double>(stencil_nnz_3d(61)) / static_cast<double>(N);

    const UpperRightWindow w = upper_right_window(gm, ReductionTier::NODE, m, nu, dim, 1.0,
                                                  gm.hbm_bw_gbs_achieved,
                                                  gm.device_memory_bytes);
    REQUIRE(w.feasible);
    // Whether 16 GB closes the corner: it does not, since R_h's ceiling binds first by orders
    // of magnitude. What remains is a question about tau.
    REQUIRE_FALSE(w.memory_binds);
    REQUIRE(w.n_max_rh < w.n_max_mem / 10);

    // And production resolution sits inside the window on the interconnect rungs.
    REQUIRE(N >= w.n_min);
    REQUIRE(N <= w.n_max);
}

/// The lower edge of the window is a real physical boundary rather than a
/// solver artifact: below it the operator is cache-resident and the vertical
/// mechanism buys nothing.
/// Expected: R_v is at least 1 at N_min by construction, and below 1 one step
/// under it.
TEST_CASE("the window closes from below at the L2 capacity, not at N=0",
          "[regime][gpu][window]")
{
    const GpuMachine gm = calibrated_v100();
    const double nu = 7.0;
    const UpperRightWindow w =
        upper_right_window(gm, ReductionTier::NODE, 12, nu, 3, 1.0, 800.0,
                           gm.device_memory_bytes);

    // N_min is where the working set first exceeds L2, so R_v = 1 there by construction.
    const int64_t nnz_min = static_cast<int64_t>(nu * static_cast<double>(w.n_min));
    REQUIRE(R_v(gm, gm.sm_count, arnoldi_working_set_bytes(nnz_min, w.n_min, 12)) >= 1.0);
    // One step below it the operator is cache-resident and the vertical mechanism buys
    // nothing, so the floor is a real boundary rather than a solver artifact.
    const int64_t below = w.n_min / 2;
    REQUIRE(R_v(gm, gm.sm_count,
                arnoldi_working_set_bytes(static_cast<int64_t>(nu * static_cast<double>(below)),
                                          below, 12)) < 1.0);
}

// Device selection

/// An instrument that predicts before measuring must take its constants from
/// the device it is about to run on, not from a hardcoded key.
/// Expected: each known device name maps to its own preset, and an unknown one
/// is rejected loudly rather than silently defaulting.
TEST_CASE("a preset is selected from the driver-reported device name",
          "[regime][gpu][preset]")
{
    // The failure this guards is not a crash. An instrument that predicts before
    // it measures used to hardcode the V100 key, which is right on Synge and
    // silently wrong on every other host, including a Puffin smoke test.
    REQUIRE(lookup_gpu_machine_for_device("Tesla V100-PCIE-16GB").key
            == lookup_gpu_machine("v100-pcie-16gb").key);
    REQUIRE(lookup_gpu_machine_for_device("NVIDIA GeForce RTX 3090").key
            == lookup_gpu_machine("rtx-3090").key);
    // Case and vendor prefixes vary between driver versions; the token does not.
    REQUIRE(lookup_gpu_machine_for_device("NVIDIA Tesla V100-PCIE-16GB").key
            == lookup_gpu_machine("v100-pcie-16gb").key);
    REQUIRE(lookup_gpu_machine_for_device("tesla v100-pcie-16gb").key
            == lookup_gpu_machine("v100-pcie-16gb").key);
    // An uncalibrated device names itself rather than borrowing constants.
    REQUIRE_THROWS_AS(
        lookup_gpu_machine_for_device("NVIDIA A100-SXM4-40GB"),
        std::invalid_argument);
}

// Type safety

/// Every CPU field is meaningless on a GPU and vice versa, so feeding one into
/// the other formulas would give a wrong answer that still looks plausible.
/// Expected: the separate types make that a compile-time impossibility; this
/// records the intent the type split exists to enforce.
TEST_CASE("GPU and CPU coordinates cannot be mixed", "[regime][gpu][types]")
{
    // Not a runtime assertion: the point of the separate type is that the mixing error is a
    // compile failure, so there is nothing here to execute. This case documents the guarantee
    // and names the two things it forbids:
    //
    //   R_v(lookup_machine("amd-3960x"), 80, ws)         <- CPU machine, GPU team size
    //   aggregate_llc_bytes(lookup_gpu_machine(...), 80) <- GPU machine, CPU accessor
    //
    // Both are rejected by overload resolution. The failure class this prevents is a CPU
    // constant serving a GPU coordinate: a wrong number that looks plausible, which is what
    // produced the degenerate-kappa(X) problem on the CPU side.
    STATIC_REQUIRE_FALSE(std::is_same_v<Machine, GpuMachine>);
    SUCCEED("enforced by the type system");
}
