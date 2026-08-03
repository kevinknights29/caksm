/**
 * @file test_regime_coordinates.cpp
 * @brief The regime map's dimensionless coordinates and their cost models.
 *
 * Reduction counts that must mirror the Arnoldi loop in src/scaling.cpp, R_v's aggregate
 * accounting frame, R_h's discipline and its two-roof denominator, the reachability claim
 * behind the contested corner, and the reduction cost model the machine preset encodes.
 *
 * @author Kevin Knights
 * @date 2026-07-10
 */

#include "test_regime_helpers.hpp"

using namespace regime_test;

// Reduction counts: must mirror the loop in src/scaling.cpp

/// The reduction counter must mirror the loop it models, or R_h is calibrated
/// against code nobody runs.
/// Expected: 1 + sum over j of (j+2) for every m from 1 to 16, and 45 at m=8.
TEST_CASE("MGS reduction count matches the scaling.cpp Arnoldi loop", "[regime][reductions]")
{
    // One norm for V.col(0); then per j in [0,m): (j+1) dots + 1 norm.
    for (int m = 1; m <= 16; ++m) {
        int64_t expected = 1;
        for (int j = 0; j < m; ++j) expected += (j + 1) + 1;
        REQUIRE(mgs_reductions(m) == expected);
    }
    REQUIRE(mgs_reductions(8) == 45);
}

/// The horizontal mechanism, stated as arithmetic: one reduction per block
/// instead of one per column.
/// Expected: a single width-m block costs 2 reductions (3 with the second
/// CholQR pass), four width-2 blocks cost 5, and a partial trailing block
/// still costs a full one. At m=8 MGS costs more than 20x the s-step count,
/// which is the ratio that motivates the method.
TEST_CASE("s-step reduction count collapses the quadratic term", "[regime][reductions]")
{
    // One block of size m => 1 normalization + 1 Gram reduction.
    REQUIRE(ca_reductions(8, 8, false) == 2);
    REQUIRE(ca_reductions(8, 8, true) == 3);
    // Four blocks of size 2.
    REQUIRE(ca_reductions(8, 2, false) == 5);
    // Partial trailing block still costs a reduction.
    REQUIRE(ca_reductions(9, 4, false) == 4);   // ceil(9/4) = 3

    // The claim that motivates the horizontal mechanism.
    REQUIRE(mgs_reductions(8) > 20 * ca_reductions(8, 8, false));
}

// R_v accounting frame

/// R_v must count cache and working set in the same frame. Counting cache per
/// slice against a per-process working set would give a 1/P^2 law.
/// Expected: doubling P exactly halves R_v, and a 256 MiB working set still
/// sits at R_v = 2 against 128 MiB of aggregate L3 at full P.
TEST_CASE("R_v uses an aggregate frame on both sides, so R_v ~ 1/P", "[regime][rv]")
{
    const Machine& mc = lookup_machine("amd-3960x");
    const int64_t ws = 256L << 20;  // 256 MiB, above puffin's 128 MiB aggregate L3

    // CCX-aligned points, where engaged slices == P/3 exactly.
    const double rv3  = R_v(mc, 3,  ws);
    const double rv6  = R_v(mc, 6,  ws);
    const double rv24 = R_v(mc, 24, ws);

    // Doubling P halves R_v: a 1/P law, never 1/P^2.
    REQUIRE_THAT(rv3 / rv6, WithinRel(2.0, 1e-12));
    REQUIRE_THAT(rv3 / rv24, WithinRel(8.0, 1e-12));

    // At full P the 256 MiB working set still overflows the 128 MiB of aggregate L3.
    REQUIRE(rv24 > 1.0);
    REQUIRE_THAT(rv24, WithinRel(2.0, 1e-12));
}

/// A thread spilling onto a second CCX commands that whole cache slice, not a
/// fraction of it.
/// Expected: P=3 engages one slice, P=4 engages two, P=24 engages eight, and
/// the aggregate is the full 128 MiB.
TEST_CASE("close binding engages whole slices, not fractional ones", "[regime][rv]")
{
    const Machine& mc = lookup_machine("amd-3960x");
    // P=4 spills onto a second CCX and commands its whole 16 MiB.
    REQUIRE(engaged_llc_slices(mc, 1) == 1);
    REQUIRE(engaged_llc_slices(mc, 3) == 1);
    REQUIRE(engaged_llc_slices(mc, 4) == 2);
    REQUIRE(engaged_llc_slices(mc, 24) == 8);
    REQUIRE(aggregate_llc_bytes(mc, 24) == (128L << 20));
}

// R_h discipline

/// The horizontal mechanism itself: adding cores raises R_h, because the
/// reduction outgrows the compute thinning beneath it.
/// Expected: R_h at P=16 exceeds R_h at P=2, and below saturation it grows
/// faster than 1/P alone, since the numerator climbs as the tree starts
/// crossing cache domains.
TEST_CASE("R_h grows with P while compute-bound", "[regime][rh]")
{
    // Calibration retired the old R_h ~ P log2(P) law: the reduction's cost is set by
    // cache-domain crossings, and it saturates. The mechanism the test exists for is
    // unchanged and is what is checked here: adding cores raises R_h, because the reduction
    // outgrows the compute thinning beneath it. Only the exponent moved, and the saturation
    // test below pins where it moved to.
    const Machine& mc = lookup_machine("amd-3960x");
    // Small enough that both kernels stay L3-resident across the whole P range, so the
    // roof scales with P and the denominator genuinely thins.
    const int64_t n = 100000, nnz = 5 * n;
    const int m = 8;
    auto rh_at = [&](int P) {
        const CycleTime ct = arnoldi_cycle_seconds(mc, P, nnz, n, m, 1.0);
        return R_h(mc, P, ct.total_s, mgs_reductions(m));
    };

    REQUIRE(rh_at(16) > rh_at(2));

    // Below saturation both ends drive growth: the numerator climbs as the tree starts
    // crossing cache domains, the denominator thins like 1/P. So R_h must outgrow 1/P alone.
    REQUIRE(rh_at(12) / rh_at(3) > 12.0 / 3.0);
}

/// R_h must be linear in the reduction count, which is the only lever s-step
/// pulls.
/// Expected: with the cycle held fixed, the MGS/CA ratio of R_h equals the
/// MGS/CA ratio of reduction counts exactly.
TEST_CASE("cutting reductions cuts R_h proportionally", "[regime][rh]")
{
    const Machine& mc = lookup_machine("amd-3960x");
    const int64_t n = 100000, nnz = 5 * n;
    const CycleTime ct = arnoldi_cycle_seconds(mc, 24, nnz, n, 8, 1.0);

    // R_h is defined on the baseline; this checks the lever s-step pulls. The cycle is
    // held fixed, s-step changes how many reductions that same work is split across.
    const double rh_mgs = R_h(mc, 24, ct.total_s, mgs_reductions(8));
    const double rh_ca  = R_h(mc, 24, ct.total_s, ca_reductions(8, 8, false));
    REQUIRE_THAT(rh_mgs / rh_ca,
                 WithinRel(static_cast<double>(mgs_reductions(8))
                           / static_cast<double>(ca_reductions(8, 8, false)), 1e-12));
}

// R_h's denominator: two kernels, two roofs.
// Pricing the cycle as one lump at SpMV's DRAM-bound intensity over-predicted it 6x at
// n=61/P=21 (R_h 0.010 against 0.060). The two intensities differ ~2x while their roofs
// differ over 10x at full P, so residency (which is R_v) picks the roof. These pin that.

/// Which roof applies is decided by residency, not by intensity, and only the
/// cache roof grows with core count.
/// Expected: a resident working set is priced at the L3 slice bandwidth times
/// the engaged slices and improves with P; a spilled one is priced at the
/// socket DRAM figure and does not. At full P the two differ by more than 4x,
/// which is the gap a single-roof model discards.
TEST_CASE("residency picks the roof, and only the cache roof scales with P",
          "[regime][rh][roofline][critical]")
{
    const Machine& mc = lookup_machine("amd-3960x");
    const int64_t small = 8L << 20;    // 8 MiB: resident at every P
    const int64_t huge  = 512L << 20;  // 512 MiB: spills even at full P

    // Resident, so L3: counted in slices on both sides, exactly as aggregate_llc_bytes is.
    // Counting cache per slice and bandwidth per core would over-provision it threefold.
    REQUIRE_THAT(memory_bw_gbs(mc, 24, small),
                 WithinRel(mc.l3_bw_gbs_slice * 8.0, 1e-12));   // 24 cores -> 8 slices
    REQUIRE(memory_bw_gbs(mc, 24, small) > memory_bw_gbs(mc, 12, small));

    // Spilled, so DRAM, a socket figure. Adding cores buys nothing.
    REQUIRE_THAT(memory_bw_gbs(mc, 24, huge), WithinRel(mc.dram_bw_gbs, 1e-12));
    REQUIRE_THAT(memory_bw_gbs(mc, 12, huge), WithinRel(mc.dram_bw_gbs, 1e-12));

    // The gap the old single-roof model was throwing away at full P.
    REQUIRE(memory_bw_gbs(mc, 24, small) / memory_bw_gbs(mc, 24, huge) > 4.0);
}

/// The case that refutes a single roof for the whole cycle: MGS resident while
/// SpMV spills, so any one roof misprices at least one of them.
/// Expected: at a 19-point stencil the two working sets straddle the aggregate
/// L3, each kernel is priced at its own roof, and the cycle is exactly their
/// sum. The wide stencil is required: at 5 nonzeros per row and m=8 the two
/// working sets are algebraically equal and no (n, P) separates them.
TEST_CASE("the two kernels are priced on their own working sets, not the cycle's",
          "[regime][rh][roofline][critical]")
{
    // The case that exposes the old model: MGS resident while SpMV spills, so one roof for
    // the whole cycle must misprice at least one of them.
    //
    // The stencil has to be wide for the two to straddle at all. At 5 nonzeros per row and
    // m=8 the working sets are algebraically identical (spmv_ws = 12*5n + 20n = 80n and
    // mgs_ws = 8n(m+2) = 80n), so no (n, P) separates them. The 19-point stencil of a
    // 3-asset operator with cross terms gives 248n against the same 80n, and the gap opens.
    // That the separation needs the mixed-derivative terms is itself the point.
    const Machine& mc = lookup_machine("amd-3960x");
    const int m = 8;
    const int64_t n = 500000, nnz = 19 * n;   // 1 + 2d + 4*C(d,2) at d=3
    const int P = 12;   // 4 slices -> 64 MiB aggregate

    const int64_t ws_spmv = spmv_working_set_bytes(nnz, n);
    const int64_t ws_mgs  = mgs_working_set_bytes(n, m);
    INFO("SpMV ws " << ws_spmv / (1 << 20) << " MiB, MGS ws " << ws_mgs / (1 << 20)
         << " MiB, aggregate L3 " << aggregate_llc_bytes(mc, P) / (1 << 20) << " MiB");
    REQUIRE(ws_spmv > aggregate_llc_bytes(mc, P));   // SpMV spills
    REQUIRE(ws_mgs  < aggregate_llc_bytes(mc, P));   // MGS does not

    const CycleTime ct = arnoldi_cycle_seconds(mc, P, nnz, n, m, 1.0);
    REQUIRE_FALSE(ct.spmv_resident);
    REQUIRE(ct.mgs_resident);
    REQUIRE_THAT(ct.bw_spmv_gbs, WithinRel(mc.dram_bw_gbs, 1e-12));
    REQUIRE_THAT(ct.bw_mgs_gbs,  WithinRel(mc.l3_bw_gbs_slice * 4.0, 1e-12));  // 12 -> 4 slices
    REQUIRE(ct.total_s > 0.0);
    REQUIRE_THAT(ct.total_s, WithinRel(ct.spmv_s + ct.mgs_s, 1e-12));
}

/// Locates the 6x mispricing in the roofs rather than in the intensities.
/// Expected: MGS is more intense than SpMV but by under 5x, so intensity alone
/// cannot explain a 6x error; and the quadratic basis re-read is more than
/// half of MGS traffic, confirming it is a bandwidth problem and not only a
/// synchronization one.
TEST_CASE("MGS's intensity is its own, and the roofs matter more than it does",
          "[regime][rh][roofline]")
{
    // MGS's cost is the basis re-reads: V_i is read once for every j >= i, so m(m+1)/2
    // column reads per cycle. That quadratic makes MGS a bandwidth problem, not only a
    // synchronization one, at its own intensity distinct from SpMV's.
    const int64_t n = 226984;
    const int m = 8;
    const double ai_mgs  = mgs_intensity(n, m);
    const double ai_spmv = spmv_intensity(4234686, n, 1.0);

    INFO("ai_mgs = " << ai_mgs << "  ai_spmv = " << ai_spmv);
    REQUIRE(ai_mgs > ai_spmv);
    REQUIRE(ai_mgs / ai_spmv < 5.0);   // ~2x: the intensities are NOT what differed 6x

    // The dominant term really is the quadratic column re-read.
    const double col_reads = 8.0 * static_cast<double>(n) * (m * (m + 1) / 2.0);
    REQUIRE(col_reads / mgs_dram_bytes(n, m) > 0.5);
}

/// The flop model must account for the whole cycle and nothing more.
/// Expected: cycle flops equal SpMV plus MGS exactly, and SpMV is exactly m
/// applications of the operator at 2*nnz each.
TEST_CASE("the cycle splits into exactly the two kernels", "[regime][rh][roofline]")
{
    const int64_t n = 226984, nnz = 4234686;
    const int m = 8;
    REQUIRE_THAT(arnoldi_cycle_flops(nnz, n, m),
                 WithinRel(spmv_cycle_flops(nnz, m) + mgs_cycle_flops(n, m), 1e-12));
    // m applications of the operator, and nothing else.
    REQUIRE_THAT(spmv_cycle_flops(nnz, m),
                 WithinRel(2.0 * static_cast<double>(nnz) * m, 1e-12));
}

// Reachability: the quantitative claim behind the contested corner

/// The reachability claim: on the physical operator, core count moves the two
/// coordinates in opposite directions, so no P reaches the upper-right corner.
/// Expected: raising P from 3 to 24 lifts R_h and drops R_v, carrying the point
/// from above the vertical threshold to below it as aggregate L3 swallows the
/// operator.
TEST_CASE("P traverses the map along an anti-diagonal for the real operator",
          "[regime][corners][critical]")
{
    const Machine& mc = lookup_machine("amd-3960x");
    constexpr int m = 8;

    // The physical Black-Scholes operator at n=61: N = 61^3, ~19 nonzeros/row.
    const int64_t n_real   = 61L * 61 * 61;
    const int64_t nnz_real = n_real * 19;

    const RegimePoint lo = place(mc, 3,  nnz_real, n_real, m, 1.0);
    const RegimePoint hi = place(mc, 24, nnz_real, n_real, m, 1.0);

    // Raising P lifts R_h and drops R_v, in opposite directions. There is no setting
    // of P that puts this operator in Upper-Right.
    REQUIRE(hi.rh > lo.rh);
    REQUIRE(hi.rv < lo.rv);
    REQUIRE(lo.rv > kThetaV);    // Upper-Left at low P
    REQUIRE(hi.rv < kThetaV);    // the 24 cores' aggregate L3 has swallowed the operator
}

/// The scaffold must reach where the physical operator cannot, or the contested
/// corner stays unmeasurable.
/// Expected: the physical operator falls below the vertical threshold at full
/// P, while a synthetic operator sized to overflow all 128 MiB stays above it.
TEST_CASE("the synthetic operator holds R_v above 1 at full P, which n cannot",
          "[regime][corners][critical]")
{
    const Machine& mc = lookup_machine("amd-3960x");
    constexpr int m = 8;

    const int64_t n_real   = 61L * 61 * 61;
    const int64_t nnz_real = n_real * 19;
    REQUIRE(place(mc, 24, nnz_real, n_real, m, 1.0).rv < kThetaV);

    // N is chosen so the working set overflows all 128 MiB of aggregate L3 at P = 24.
    SyntheticSpec big;
    big.n1 = 128; big.dim = 3;                      // N = 2,097,152
    const int64_t n_syn   = synthetic_dimension(big);
    const int64_t nnz_syn = 7 * n_syn - 6L * 128 * 128;   // 7-point stencil, minus faces

    const RegimePoint syn = place(mc, 24, nnz_syn, n_syn, m, 1.0);
    REQUIRE(syn.rv > kThetaV);   // vertical mechanism survives full core count
}

/// A finding, not a mechanism check: scatter cannot be used to reach the
/// upper-right corner, because it moves the two coordinates against each other.
/// Expected: a symmetric permutation leaves working set, flops and R_v exactly
/// unchanged, while lowering arithmetic intensity and with it R_h.
TEST_CASE("scattering trades the horizontal mechanism away for the vertical one",
          "[regime][corners][tension]")
{
    // A finding that qualifies the "large N + scattered pattern" recipe for upper-right.
    // Scatter raises DRAM traffic, lowering arithmetic intensity and so the attainable rate,
    // which makes the compute between two reductions take longer, so a reduction matters
    // relatively less and R_h falls. Scatter is the upper-left knob: it moves away from
    // upper-right.
    const Machine& mc = lookup_machine("amd-3960x");
    constexpr int m = 8;
    const int64_t n = 1L << 20;
    const int64_t nnz = 7 * n;

    const RegimePoint banded    = place(mc, 24, nnz, n, m, /*x_reuse=*/1.0);
    const RegimePoint scattered = place(mc, 24, nnz, n, m, /*x_reuse=*/0.0);

    // Same working set, same flops: a symmetric permutation changed only the traffic.
    REQUIRE(banded.working_set == scattered.working_set);
    REQUIRE_THAT(banded.W, WithinRel(scattered.W, 1e-12));
    REQUIRE(banded.rv == scattered.rv);

    // But intensity, and therefore R_h, moves down.
    REQUIRE(scattered.ai < banded.ai);
    REQUIRE(scattered.rh < banded.rh);
}

// The reduction cost model: cache-domain crossings, not uniform tree levels
//
// Calibration refuted t = alpha*log2(P) twice, against the O(P) scan the harness used to
// run and again against a correct tree. Levels do not cost the same here: one costs 5.4x
// more once its partner leaves the root's L3 slice, and nothing at all past L*. These tests
// pin the replacement, and the two claims whose breakage would be silent: the level split
// must match the tree the reducer actually walks, and P=1 must stay free.

/// The level counters must describe the tree reduction.hpp actually walks;
/// a mismatch calibrates the cost against a shape nobody runs.
/// Expected: at 3 cores per slice, P=3 gives 2 intra-domain levels and no
/// crossing, P=6 adds the first crossing, P=24 gives 2 intra and 3 cross. For
/// every P up to 24 the two counts sum to the tree depth, so no level is
/// uncounted or double-counted.
TEST_CASE("the level split matches the tree the reducer walks", "[regime][rh][critical]")
{
    // include/reduction.hpp combines thread 0 with partners at d = 1, 2, 4, ... < P, and
    // under close binding partner d sits on slice floor(d / cores_per_slice). If these
    // counters and that loop disagree, the reduction is calibrated against a tree shape
    // nobody runs: the defect that put the O(P) scan under an alpha*log2(P) model to
    // begin with.
    const Machine& mc = lookup_machine("amd-3960x");   // 3 cores/slice

    // P=3: partners at d=1,2, both on the root's own CCX.
    REQUIRE(intra_domain_levels(mc, 3) == 2);
    REQUIRE(cross_domain_levels(mc, 3) == 0);

    // P=6: adds d=4, the first partner on another CCX.
    REQUIRE(intra_domain_levels(mc, 6) == 2);
    REQUIRE(cross_domain_levels(mc, 6) == 1);

    // P=24: d=1,2 intra; d=4,8,16 cross.
    REQUIRE(intra_domain_levels(mc, 24) == 2);
    REQUIRE(cross_domain_levels(mc, 24) == 3);

    // Every level is one or the other, and together they are the tree's depth.
    for (int P = 1; P <= 24; ++P) {
        int depth = 0;
        for (int d = 1; d < P; d <<= 1) ++depth;
        INFO("P = " << P);
        REQUIRE(intra_domain_levels(mc, P) + cross_domain_levels(mc, P) == depth);
    }
}

/// A serial run walks no tree levels, which is what places it at the bottom of
/// the map by construction rather than by convention.
/// Expected: both level counts are zero at P=1, the reduction costs exactly
/// nothing, and R_h is exactly zero.
TEST_CASE("a serial run has no reduction, and that is measured not asserted",
          "[regime][rh]")
{
    // P=1 walks no levels, so it costs nothing, which is what puts a serial run at the
    // bottom of the map by construction. The calibrated tree measures 9.6 ns there: call
    // overhead and no synchronization whatever.
    const Machine& mc = lookup_machine("amd-3960x");
    REQUIRE(intra_domain_levels(mc, 1) == 0);
    REQUIRE(cross_domain_levels(mc, 1) == 0);
    REQUIRE(reduction_cost_s(mc, 1) == 0.0);
    REQUIRE(R_h(mc, 1, 1e-3, 45) == 0.0);
}

/// The crossing model needs a real mechanism: if a crossing cost the same as
/// an intra-domain level, R_h's numerator would be uniform in depth again.
/// Expected: the crossing multiplier stays above 3, and the first crossing
/// costs more than the entire intra-CCX tree beneath it.
TEST_CASE("a cache-domain crossing costs multiples of an intra-domain level",
          "[regime][rh][critical]")
{
    // `hop_cost` used to sit at 1.0, declared untestable on a single NUMA node. It is
    // testable: a CCX boundary is a cache-domain crossing too, and puffin has eight.
    // Measured at ~5.4x. If it ever collapses toward 1, the crossing model has lost its
    // mechanism and R_h's numerator is uniform in depth again.
    const Machine& mc = lookup_machine("amd-3960x");
    REQUIRE(crossing_multiplier(mc) > 3.0);
    REQUIRE(mc.t_level_cross_s > mc.t_level_intra_s);

    // The first crossing costs more than the entire intra-CCX tree beneath it.
    const double intra_only = reduction_cost_s(mc, 3);    // 2 intra levels, no crossing
    const double one_cross  = reduction_cost_s(mc, 6);    // + the first crossing
    REQUIRE(one_cross - intra_only > intra_only);
}

/// Past the measured crossing limit, extra tree depth is free, which drops
/// R_h's exponent from P log P to P on this core count.
/// Expected: P=18 and P=24 walk the same three crossings and are charged
/// identically; below saturation the cost still climbs crossing by crossing;
/// and with both kernels held resident, R_h scales exactly as P, which is
/// strictly below what the P log P law it replaced would predict.
TEST_CASE("the numerator saturates, so R_h ~ P and not P log P on this machine",
          "[regime][rh][critical]")
{
    // The correction calibration forced. Past L* crossings the subtrees have already
    // finished, so extra depth is free and the reduction cost stops growing. Over puffin's
    // upper range the numerator is constant, dropping R_h's exponent from P log P to P. The
    // mechanism is untouched (reductions still outgrow the thinning compute); 24 cores is
    // simply too few for tree depth to bind.
    const Machine& mc = lookup_machine("amd-3960x");

    // P=18 and P=24 both walk three crossings, past L* = 2.2, so the model charges them
    // identically despite a 33% difference in P. That is the saturation, and it is what
    // the measurement showed: a full extra level from P=15 to P=18 cost -15 ns.
    REQUIRE(cross_domain_levels(mc, 18) == 3);
    REQUIRE(cross_domain_levels(mc, 24) == 3);
    REQUIRE_THAT(reduction_cost_s(mc, 24),
                 WithinRel(reduction_cost_s(mc, 18), 1e-12));

    // Below saturation the cost still climbs, crossing by crossing.
    REQUIRE(reduction_cost_s(mc, 6) > reduction_cost_s(mc, 3));
    REQUIRE(reduction_cost_s(mc, 9) > reduction_cost_s(mc, 6));

    // With the numerator pinned, R_h rides entirely on the denominator's 1/P and so scales
    // like P exactly. Both kernels are held L3-resident so the roof scales with P and the
    // cycle really does thin like 1/P; once one spills onto the flat DRAM roof the
    // denominator stops shrinking and even this weakened law fails.
    const int64_t n = 100000, nnz = 5 * n;
    const int m = 8;
    auto rh_at = [&](int P) {
        const CycleTime ct = arnoldi_cycle_seconds(mc, P, nnz, n, m, 1.0);
        REQUIRE(ct.spmv_resident);
        REQUIRE(ct.mgs_resident);
        return R_h(mc, P, ct.total_s, mgs_reductions(m));
    };
    const double rh18 = rh_at(18);
    const double rh24 = rh_at(24);
    REQUIRE_THAT(rh24 / rh18, WithinRel(24.0 / 18.0, 1e-12));

    // And that is strictly less than the P log P law this replaced would have predicted.
    const double p_log_p = (24.0 * std::log2(24.0)) / (18.0 * std::log2(18.0));
    REQUIRE(rh24 / rh18 < p_log_p);
}

/// A preset for hardware nobody ran on would place points from numbers no one
/// measured.
/// Expected: exactly one machine exists, it is marked calibrated, an unknown
/// key throws, and every preset carries positive measured level costs.
TEST_CASE("the only preset is the machine that was measured", "[regime][rh]")
{
    // One compute environment, with its reduction parameters measured on it. A preset for
    // hardware nobody runs is not a convenience but a set of numbers that can silently place
    // a point, and R_h's magnitude would inherit them without complaint.
    REQUIRE(kMachines.size() == 1);
    REQUIRE(lookup_machine("amd-3960x").reduction_calibrated);
    REQUIRE_THROWS_AS(lookup_machine("intel-8358"), std::invalid_argument);

    // reduction_calibrated is vacuously true while there is one machine, and kept
    // deliberately: it guards against a second host's preset being trusted before
    // scripts/regime/calibrate_alpha.sh has run on it. Deleting it would mean re-deriving
    // that reason the next time a machine is added.
    for (const Machine& mc : kMachines) {
        INFO("preset " << mc.key);
        REQUIRE(mc.reduction_calibrated);
        REQUIRE(mc.t_level_cross_s > 0.0);
        REQUIRE(mc.t_level_intra_s > 0.0);
    }
}
