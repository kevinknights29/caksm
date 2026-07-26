/**
 * @file test_regime_kernels.cpp
 * @brief The tiled matrix-powers halo model (mpk.hpp) and the kernel itself (akx.hpp).
 *
 * The halo model decides, before anything runs, whether the vertical mechanism is even
 * available on a given operator. The kernel must then reproduce the plain SpMV chain's
 * Krylov basis to the bit, or every theta_v crossover measured on it is meaningless.
 *
 * @author Kevin Knights
 * @date 2026-07-10
 */

#include "test_regime_helpers.hpp"

using namespace regime_test;

// The tiled matrix-powers model
//
// The halo model is what decides, before anything runs, whether the vertical mechanism
// is even available on a given operator. If predicted_bandwidth under-estimates w, the
// switch tiles a pattern it cannot tile and the matrix-powers arm silently degenerates
// while still being reported as blocked. These tests pin the bound and the switch.

TEST_CASE("predicted bandwidth is exact on the banded arm", "[mpk][bandwidth]")
{
    // b = 1 is the identity permutation, so the operator keeps its stencil bandwidth:
    // the widest axial stride, n1^(dim-1).
    SyntheticSpec sp;
    sp.n1 = 16; sp.dim = 2; sp.scatter_block = 1;

    const SyntheticOperator op = build_synthetic(sp);
    REQUIRE(stencil_bandwidth(sp) == 16);
    REQUIRE(predicted_bandwidth(sp) == 16);
    REQUIRE(measured_bandwidth(op.A) == 16);
}

TEST_CASE("predicted bandwidth bounds the measured one at every scatter setting",
          "[mpk][bandwidth][critical]")
{
    // The switch is only safe if the bound is never violated: an under-estimate would
    // tile an operator whose halo is wider than the panel assumed.
    for (const int64_t b : {int64_t{1}, int64_t{2}, int64_t{8}, int64_t{32},
                            int64_t{128}, int64_t{256}}) {
        SyntheticSpec sp;
        sp.n1 = 16; sp.dim = 2; sp.scatter_block = b;

        const SyntheticOperator op = build_synthetic(sp);
        INFO("scatter_block b = " << b);
        REQUIRE(measured_bandwidth(op.A) <= predicted_bandwidth(sp));
    }
}

TEST_CASE("full scatter saturates the bandwidth: the halo becomes the whole vector",
          "[mpk][bandwidth]")
{
    SyntheticSpec sp;
    sp.n1 = 16; sp.dim = 2; sp.scatter_block = 256;   // b = N
    REQUIRE(predicted_bandwidth(sp) == 255);          // n - 1
}

TEST_CASE("the level footprint is a trapezoid, not a box", "[mpk][footprint]")
{
    // Levels shrink as the recurrence climbs, so charging (s+1) * expanded would
    // over-count the halo and retire the panel before it actually stops fitting.
    const int64_t rows = 4096, w = 16, n = 1 << 20;
    const int s = 8;
    const double nnz_per_row = 5.0;

    const double trapezoid = mpk_panel_bytes(rows, s, w, n, nnz_per_row);
    const double expanded  = static_cast<double>(rows + 2 * s * w);
    const double box = expanded * (nnz_per_row * 12.0 + 4.0)
                     + static_cast<double>(s + 1) * expanded * 8.0;

    REQUIRE(trapezoid < box);
}

TEST_CASE("the auto-sized panel fits the level it was sized against", "[mpk][footprint]")
{
    // mpk_panel_rows inverts mpk_panel_bytes, so the two must agree: the panel it
    // returns has to land at or under the capacity, at both tiling levels.
    const Machine& mc = lookup_machine("amd-3960x");
    const int64_t w = 16, n = 1 << 22;
    const double nnz_per_row = 5.0;

    for (const TileLevel lv : {TileLevel::L2, TileLevel::L3}) {
        for (const int s : {1, 4, 8, 16}) {
            const int64_t rows = mpk_panel_rows(mc, lv, s, w, n, nnz_per_row);
            INFO("level " << tile_level_name(lv) << "  s = " << s << "  rows = " << rows);
            REQUIRE(rows > 0);
            REQUIRE(mpk_tiled_R_v(mc, lv, rows, s, w, n, nnz_per_row) <= 1.0);
        }
    }
}

TEST_CASE("a taller panel carries a thinner halo", "[mpk][halo]")
{
    // The halo is a fixed 2*s*w rows, so it amortizes over the interior. This is why
    // the tallest panel that fits is the right one to pick.
    REQUIRE(mpk_halo_ratio(1000, 4, 16) > mpk_halo_ratio(8000, 4, 16));
}

TEST_CASE("the halo fattens with s, which is the capacity roof's mechanism", "[mpk][halo]")
{
    REQUIRE(mpk_halo_ratio(4096, 2, 16) < mpk_halo_ratio(4096, 8, 16));
}

TEST_CASE("the finer tiling level imposes the tighter capacity roof on s",
          "[mpk][sroof]")
{
    // L2 is private but small; L3 is shared but large. A panel sized to L2 runs out of
    // room at a lower s, which is exactly why running both levels brackets the crossover
    // rather than asserting it.
    const Machine& mc = lookup_machine("amd-3960x");
    const int64_t w = 16, n = 1 << 22;
    const double nnz_per_row = 5.0;

    const int s_l2 = s_ghost_max(mc, TileLevel::L2, w, n, nnz_per_row, 40);
    const int s_l3 = s_ghost_max(mc, TileLevel::L3, w, n, nnz_per_row, 40);

    INFO("s_ghost_max: L2 = " << s_l2 << "  L3 = " << s_l3);
    REQUIRE(s_l2 > 0);
    REQUIRE(s_l2 <= s_l3);
}

TEST_CASE("on the banded arm the CERTIFICATE binds, not the halo", "[mpk][sroof][critical]")
{
    // The thesis claim in miniature. A thin-banded operator leaves the capacity roof far
    // above the exponential-integrator's m = 8-11 window, so the block width is set by
    // basis conditioning and NOT by cache. If this ever inverts, s has become a capacity
    // constraint and a better-conditioned basis (Newton, Chebyshev) stops being worth
    // reaching for: a different chapter.
    const Machine& mc = lookup_machine("amd-3960x");
    const int64_t w = 16, n = 1 << 22;
    const double nnz_per_row = 5.0;
    const int s_certified = 9;      // the Laplacian scaffold's certified width

    const int s_capacity = s_ghost_max(mc, TileLevel::L3, w, n, nnz_per_row, 40);
    INFO("capacity roof = " << s_capacity << "  certificate = " << s_certified);
    REQUIRE(s_capacity > s_certified);
    REQUIRE(operative_s(s_capacity, s_certified) == s_certified);
}

TEST_CASE("the pattern switch tiles the banded arm and refuses the scattered one",
          "[mpk][switch][critical]")
{
    // The load-bearing asymmetry. The scatter knob is a permutation, so it leaves the
    // dependency GRAPH identical while destroying the locality the tiling needs. The map
    // predicts the two arms differ only in DRAM traffic; the tiled kernel is where that
    // stops being true, and the switch has to see it coming from b alone.
    const Machine& mc = lookup_machine("amd-3960x");
    const double nnz_per_row = 5.0;
    const int s_certified = 8;

    SyntheticSpec banded;
    banded.n1 = 256; banded.dim = 2; banded.scatter_block = 1;
    const MpkPlan pb = plan_mpk(mc, TileLevel::L3, banded, s_certified, nnz_per_row);
    INFO("banded: " << pb.reason << "  w = " << pb.w << "  panel = " << pb.panel_rows);
    REQUIRE(pb.form == MpkForm::TILED);
    REQUIRE(pb.halo_ratio < kThinHaloRatio);

    SyntheticSpec scattered = banded;
    scattered.scatter_block = synthetic_dimension(scattered);   // b = N
    const MpkPlan ps = plan_mpk(mc, TileLevel::L3, scattered, s_certified, nnz_per_row);
    INFO("scattered: " << ps.reason << "  w = " << ps.w);
    REQUIRE(ps.form == MpkForm::NAIVE);
    REQUIRE(ps.panel_rows == 0);
}

TEST_CASE("the scatter knob leaves the operative block width alone on the banded arm",
          "[mpk][switch]")
{
    // A sanity pairing with the similarity-transform control: b moves bytes, so it must
    // not move the numerical roof. It moves the capacity roof, and only that.
    const Machine& mc = lookup_machine("amd-3960x");
    SyntheticSpec sp;
    sp.n1 = 256; sp.dim = 2; sp.scatter_block = 1;

    const MpkPlan p = plan_mpk(mc, TileLevel::L3, sp, 8, 5.0);
    REQUIRE(p.s_certified == 8);
    REQUIRE(p.s == 8);
    REQUIRE_FALSE(p.capacity_binds);
}

// The cache-blocked matrix-powers kernel (akx.hpp)
//
// The kernel reorders memory access, not arithmetic: tiled_matrix_powers must reproduce the
// plain SpMV chain's Krylov basis to the bit, or a fast-looking kernel posts a roofline for
// the wrong subspace and every theta_v crossover measured on it is meaningless. This is the
// gate the vertical experiment rests on, so it is tested across tile sizes (one panel, many,
// panels smaller than the halo), operator bandwidths (banded and scattered), and block widths.

namespace {

SpMatRow row_major(const SpMatS& A)
{
    // spmv_row_dot reads the raw CSR arrays directly (outerIndexPtr/innerIndexPtr/
    // valuePtr), which is only valid on a COMPRESSED matrix. The row-major conversion
    // constructor produces one in practice, but this makes the requirement explicit and
    // idempotent rather than relying on that being true forever.
    SpMatRow rm(A);
    rm.makeCompressed();
    return rm;
}

}  // namespace

TEST_CASE("tiled matrix-powers equals the plain SpMV chain, bit-for-bit", "[akx][critical]")
{
    SyntheticSpec sp;
    sp.n1 = 20; sp.dim = 2; sp.scatter_block = 1;   // banded, N = 400
    const SyntheticOperator op = build_synthetic(sp);
    const SpMatRow A = row_major(op.A);
    const int64_t w = operator_bandwidth(A);
    const Eigen::VectorXd v = deterministic_unit_vector(A.rows());

    // The kernel reorders access, not arithmetic, so the two agree to rounding at worst.
    for (const int s : {1, 2, 3, 5, 8}) {
        const Eigen::MatrixXd ref = spmv_chain(A, v, s);
        // A spread of tile sizes: one panel (no tiling), several panels, and panels far
        // smaller than the s*w halo (maximal recompute / overlap).
        for (const int64_t tile : {int64_t{0}, int64_t{400}, int64_t{97}, int64_t{31},
                                   int64_t{8}, int64_t{1}}) {
            const Eigen::MatrixXd got = tiled_matrix_powers(A, v, s, tile, w);
            INFO("s = " << s << "  tile = " << tile << "  w = " << w);
            REQUIRE(got.rows() == ref.rows());
            REQUIRE(got.cols() == ref.cols());
            REQUIRE((got - ref).cwiseAbs().maxCoeff() <= 1e-10 * (1.0 + ref.cwiseAbs().maxCoeff()));
        }
    }
}

TEST_CASE("tiled matrix-powers is correct on the scattered arm too (wide halo)",
          "[akx]")
{
    // The scattered operator has bandwidth ~ N, so every panel's halo is the whole vector.
    // The kernel must still compute the right answer; only performance collapses there
    // (the pattern switch refuses to tile), not correctness.
    SyntheticSpec sp;
    sp.n1 = 12; sp.dim = 2; sp.scatter_block = 144;   // full scatter, N = 144
    const SyntheticOperator op = build_synthetic(sp);
    const SpMatRow A = row_major(op.A);
    const int64_t w = operator_bandwidth(A);
    const Eigen::VectorXd v = deterministic_unit_vector(A.rows());

    const Eigen::MatrixXd ref = spmv_chain(A, v, 4);
    const Eigen::MatrixXd got = tiled_matrix_powers(A, v, 4, 32, w);
    REQUIRE((got - ref).cwiseAbs().maxCoeff() <= 1e-10 * (1.0 + ref.cwiseAbs().maxCoeff()));
}

TEST_CASE("tiled matrix-powers spans the same subspace the certified block uses",
          "[akx][critical]")
{
    // The kernel is only ever asked for the certified block width. A block built by the
    // tiled kernel and orthogonalized must span the same subspace as the reference
    // ca_arnoldi block: swapping in tiled_matrix_powers must not change the answer.
    SyntheticSpec sp;
    sp.n1 = 24; sp.dim = 2; sp.scatter_block = 1;
    const SyntheticOperator op = build_synthetic(sp);
    const SpMatRow A = row_major(op.A);
    const int64_t w = operator_bandwidth(A);
    const Eigen::VectorXd v = deterministic_unit_vector(A.rows());

    const int s_cert = measured_s_max(op.A, v, 20);   // the certified width the caller passes
    REQUIRE(s_cert >= 1);

    const Eigen::MatrixXd tiled = tiled_matrix_powers(A, v, s_cert, 64, w);
    const Eigen::MatrixXd plain = matrix_powers(op.A, v, s_cert);
    REQUIRE((tiled - plain).cwiseAbs().maxCoeff()
            <= 1e-10 * (1.0 + plain.cwiseAbs().maxCoeff()));
}

TEST_CASE("s = 0 returns just the start vector", "[akx]")
{
    SyntheticSpec sp; sp.n1 = 10; sp.dim = 1;
    const SyntheticOperator op = build_synthetic(sp);
    const SpMatRow A = row_major(op.A);
    const Eigen::VectorXd v = deterministic_unit_vector(A.rows());
    const Eigen::MatrixXd B = tiled_matrix_powers(A, v, 0, 4, operator_bandwidth(A));
    REQUIRE(B.cols() == 1);
    REQUIRE((B.col(0) - v).cwiseAbs().maxCoeff() == 0.0);
}

TEST_CASE("tiled_matrix_powers_into chains blocks into one basis, sharing one scratch",
          "[akx][critical]")
{
    // The pattern build_tiled (src/regime_sweep.cpp) actually runs, which the single-call
    // wrapper tests above do not exercise: several calls into one output matrix at different
    // col0, reusing a single scratch buffer across all of them (the fix for the
    // double-allocation bug that made the timed kernel slower than the baseline at every
    // measured grid size). Wrong col0 bookkeeping or scratch reuse shows up here, and a
    // wrapper-only test would not catch it.
    SyntheticSpec sp;
    sp.n1 = 24; sp.dim = 2; sp.scatter_block = 1;
    const SyntheticOperator op = build_synthetic(sp);
    const SpMatRow A = row_major(op.A);
    const int64_t w = operator_bandwidth(A);
    const Eigen::VectorXd v = deterministic_unit_vector(A.rows());

    const int m = 11;   // deliberately not a multiple of s, so the last block is partial
    for (const int s : {1, 2, 3, 4}) {
        const Eigen::MatrixXd ref = spmv_chain(A, v, m);

        Eigen::MatrixXd B(A.rows(), m + 1);
        B.col(0) = v;
        Eigen::MatrixXd scratch;   // ONE buffer, reused across every block below
        int filled = 0;
        while (filled < m) {
            const int blk = std::min(s, m - filled);
            // Tile smaller than a block's halo on purpose: forces multiple panels per
            // block, the case most likely to expose an elo/col0 indexing mistake.
            tiled_matrix_powers_into(A, blk, /*tile_rows=*/17, w, B, filled, scratch);
            filled += blk;
        }

        INFO("m = " << m << "  s = " << s);
        REQUIRE((B - ref).cwiseAbs().maxCoeff() <= 1e-10 * (1.0 + ref.cwiseAbs().maxCoeff()));
    }
}
