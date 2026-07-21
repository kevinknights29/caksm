/**
 * @file test_regime_synthetic.cpp
 * @brief The synthetic operator's knobs, and the spectrum-alone basis prediction.
 *
 * The scaffold's three knobs (spectrum, scale/shift, scatter) must move independently.
 * In particular the scatter knob must be a similarity transform, or there is no pure R_v
 * axis, and the monomial basis conditioning must follow from the spectrum alone.
 *
 * @author Kevin Knights
 * @date 2026-07-10
 */

#include "test_regime_helpers.hpp"

using namespace regime_test;

// The synthetic operator's three knobs really are independent

TEST_CASE("analytic Kronecker-sum spectrum matches a dense eigensolver", "[synthetic][spectrum]")
{
    SyntheticSpec sp;
    sp.n1 = 12; sp.dim = 2;   // N = 144

    const SyntheticOperator op = build_synthetic(sp);
    Eigen::VectorXd analytic = op.lambda;
    std::sort(analytic.data(), analytic.data() + analytic.size());

    const Eigen::VectorXd numeric = sorted_spectrum(op.A);
    REQUIRE((analytic - numeric).cwiseAbs().maxCoeff() < 1e-11);
}

TEST_CASE("scale and shift move the spectrum analytically", "[synthetic][spectrum]")
{
    SyntheticSpec base;
    base.n1 = 10; base.dim = 1;

    SyntheticSpec tuned = base;
    tuned.spec_scale = 3.5;
    tuned.spec_shift = 0.25;

    const SyntheticOperator b = build_synthetic(base);
    const SyntheticOperator t = build_synthetic(tuned);

    for (Eigen::Index i = 0; i < b.lambda.size(); ++i)
        REQUIRE_THAT(t.lambda(i), WithinRel(3.5 * (b.lambda(i) - 0.25), 1e-12));

    // And the assembled matrix agrees with the analytic claim.
    const Eigen::VectorXd numeric = sorted_spectrum(t.A);
    Eigen::VectorXd analytic = t.lambda;
    std::sort(analytic.data(), analytic.data() + analytic.size());
    REQUIRE((analytic - numeric).cwiseAbs().maxCoeff() < 1e-10);
}

TEST_CASE("shift does not change the Krylov dimension: subspaces are shift-invariant",
          "[synthetic][spectrum]")
{
    SyntheticSpec base;
    base.n1 = 40; base.dim = 1;
    SyntheticSpec shifted = base;
    shifted.spec_shift = 1.3;

    const SyntheticOperator b = build_synthetic(base);
    const SyntheticOperator s = build_synthetic(shifted);

    // predict_krylov_dim reads the spread, which a shift leaves alone.
    REQUIRE(predict_krylov_dim(b.lambda_min, b.lambda_max, 1e-2, 1e-8)
            == predict_krylov_dim(s.lambda_min, s.lambda_max, 1e-2, 1e-8));
}

TEST_CASE("the scatter knob is a similarity transform: pure-axis motion exists",
          "[synthetic][scatter][critical]")
{
    // The single most load-bearing property of the instrument. If this fails, the
    // synthetic operator cannot decouple the axes, and we are back on the anti-diagonal.
    SyntheticSpec banded;
    banded.n1 = 14; banded.dim = 2;   // N = 196

    SyntheticSpec scattered = banded;
    scattered.scatter_block = 196;    // one global shuffle

    const SyntheticOperator b = build_synthetic(banded);
    const SyntheticOperator s = build_synthetic(scattered);

    SECTION("flop count is preserved exactly") {
        REQUIRE(b.nnz == s.nnz);
        REQUIRE(b.n == s.n);
    }

    SECTION("working set in bytes is preserved exactly") {
        REQUIRE(arnoldi_working_set_bytes(b.nnz, b.n, 8)
                == arnoldi_working_set_bytes(s.nnz, s.n, 8));
    }

    SECTION("the spectrum is preserved exactly") {
        const Eigen::VectorXd lb = sorted_spectrum(b.A);
        const Eigen::VectorXd ls = sorted_spectrum(s.A);
        REQUIRE((lb - ls).cwiseAbs().maxCoeff() < 1e-10);
    }

    SECTION("basis conditioning is preserved") {
        const Eigen::VectorXd v = deterministic_unit_vector(b.n);
        const double kb = condition_number(matrix_powers(b.A, apply_perm(b.perm, v), 6));
        const double ks = condition_number(matrix_powers(s.A, apply_perm(s.perm, v), 6));
        REQUIRE_THAT(kb, WithinRel(ks, 1e-8));
    }

    SECTION("but the modelled gather traffic does move") {
        const Machine& mc = lookup_machine("amd-3960x");
        // A window that fits in cache is reused; a global shuffle over a large N is not.
        REQUIRE(modelled_x_reuse(mc, 24, 1, 1L << 30) == 1.0);
        REQUIRE(modelled_x_reuse(mc, 24, 1L << 30, 1L << 30) < 1.0);
    }
}

TEST_CASE("scatter_block interpolates the gather window monotonically", "[synthetic][scatter]")
{
    const Machine& mc = lookup_machine("amd-3960x");
    const int64_t n = 1L << 30;
    const double r_small = modelled_x_reuse(mc, 24, 1L << 20, n);
    const double r_mid   = modelled_x_reuse(mc, 24, 1L << 25, n);
    const double r_big   = modelled_x_reuse(mc, 24, 1L << 30, n);
    REQUIRE(r_small >= r_mid);
    REQUIRE(r_mid   >  r_big);
}

// Basis conditioning: predicted from the spectrum alone

TEST_CASE("predicted basis condition matches the explicitly formed basis",
          "[regime][conditioning]")
{
    SyntheticSpec sp;
    sp.n1 = 16; sp.dim = 2;   // N = 256

    const SyntheticOperator op = build_synthetic(sp);
    const Eigen::VectorXd v = deterministic_unit_vector(op.n);
    const Eigen::VectorXd c = spectral_coefficients(sp, v);

    for (int s = 1; s <= 5; ++s) {
        const double measured  = condition_number(matrix_powers(op.A, apply_perm(op.perm, v), s));
        const double predicted = predicted_basis_condition(op.lambda, c, s);
        // Compare in decades: kappa spans orders of magnitude by construction.
        REQUIRE_THAT(std::log10(measured), WithinAbs(std::log10(predicted), 0.05));
    }
}

TEST_CASE("spectral coefficients reproduce the vector: Q is orthogonal", "[synthetic][dst]")
{
    SyntheticSpec sp;
    sp.n1 = 9; sp.dim = 2;
    const Eigen::VectorXd v = deterministic_unit_vector(synthetic_dimension(sp));
    const Eigen::VectorXd c = spectral_coefficients(sp, v);
    // A DST-I basis is orthonormal, so it preserves the 2-norm.
    REQUIRE_THAT(c.norm(), WithinRel(v.norm(), 1e-12));
}
