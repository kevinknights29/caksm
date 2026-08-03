/**
 * @file test_regime_krylov.cpp
 * @brief Building and orthogonalizing the Krylov basis.
 *
 * The two orthogonalization arms (MGS Arnoldi and the s-step/CholeskyQR pair), the
 * Hochbruck-Lubich prediction of the Krylov dimension m, and the per-vector certificate
 * blocks = ceil(m/(s_max+1)) with the anti-correlation the ledger must respect.
 *
 * @author Kevin Knights
 * @date 2026-07-10
 */

#include "test_regime_helpers.hpp"

using namespace regime_test;

// The two orthogonalization arms

/// The baseline arm, and the reference every other arm is compared against.
/// Expected: the requested dimension is reached, the basis loses no more than
/// 1e-12 of orthogonality, and H has the Hessenberg shape.
TEST_CASE("MGS Arnoldi produces an orthonormal basis and a Hessenberg H",
          "[arnoldi][mgs]")
{
    SyntheticSpec sp;
    sp.n1 = 20; sp.dim = 2;
    const SyntheticOperator op = build_synthetic(sp);
    const Eigen::VectorXd v = deterministic_unit_vector(op.n);

    const ArnoldiResult r = mgs_arnoldi(op.A, v, 10);
    REQUIRE(r.m_used == 10);
    REQUIRE(orthogonality_loss(r.V.leftCols(r.m_used)) < 1e-12);

    // Arnoldi relation: A V_m = V_{m+1} H.
    const Eigen::MatrixXd lhs = op.A * r.V.leftCols(10);
    const Eigen::MatrixXd rhs = r.V.leftCols(11) * r.H.topLeftCorner(11, 10);
    REQUIRE((lhs - rhs).norm() < 1e-11);

    REQUIRE(r.reductions == mgs_reductions(10));
}

/// The s-step arm must reach the same subspace as MGS, or it is solving a
/// different problem cheaply.
/// Expected: the orthogonal projectors onto the two spans coincide, even though
/// the bases themselves differ, and the s-step arm issues far fewer reductions.
/// That pairing is the entire horizontal mechanism.
TEST_CASE("s-step Arnoldi spans the same Krylov subspace as MGS", "[arnoldi][ca][critical]")
{
    SyntheticSpec sp;
    sp.n1 = 20; sp.dim = 2;
    const SyntheticOperator op = build_synthetic(sp);
    const Eigen::VectorXd v = deterministic_unit_vector(op.n);

    constexpr int m = 8;
    const ArnoldiResult   mgs = mgs_arnoldi(op.A, v, m);
    const CaArnoldiResult ca  = ca_arnoldi(op.A, v, m, /*s=*/4);

    REQUIRE_FALSE(ca.broke_down);
    REQUIRE(ca.m_used == m);

    // Same subspace: projectors onto span(V) coincide, even though the bases differ.
    const Eigen::MatrixXd Pm = mgs.V.leftCols(m) * mgs.V.leftCols(m).transpose();
    const Eigen::MatrixXd Pc = ca.V * ca.V.transpose();
    REQUIRE((Pm - Pc).norm() < 1e-8);

    // And it costs far fewer reductions. That is the entire horizontal mechanism.
    REQUIRE(10 * ca.reductions < mgs.reductions);
}

/// Forming the Gram matrix squares the condition number, so one CholeskyQR pass
/// loses orthogonality long before Cholesky itself complains.
/// Expected: at a width just inside the certificate, one pass has already lost
/// about six digits while two passes recover O(u). Both report the condition of
/// the input block rather than of the intermediate Q, which is what makes the
/// certificate checkable before the factorization runs.
TEST_CASE("plain CholeskyQR is unsafe; CholeskyQR2 restores O(u) orthogonality",
          "[arnoldi][cholqr][stability][critical]")
{
    SyntheticSpec sp;
    sp.n1 = 64; sp.dim = 1;
    const SyntheticOperator op = build_synthetic(sp);
    const Eigen::VectorXd v = deterministic_unit_vector(op.n);

    // s = 9 sits just inside the certificate (kappa = 2.6e7 < 9.5e7), yet one pass of
    // CholeskyQR has already lost six digits of orthogonality: the squared condition
    // number bites long before Cholesky itself complains.
    const Eigen::MatrixXd B = matrix_powers(op.A, v, 9);
    const CholQrResult one = cholesky_qr(B);
    const CholQrResult two = cholesky_qr2(B);

    REQUIRE_FALSE(one.llt_failed);
    REQUIRE_FALSE(two.llt_failed);
    REQUIRE(one.kappa < cholqr_kappa_limit());

    REQUIRE(orthogonality_loss(one.Q) > 1e-7);    // one pass: unusable
    REQUIRE(orthogonality_loss(two.Q) < 1e-13);   // two passes: O(u)

    // Both report the condition of the input block, not of the intermediate Q.
    REQUIRE_THAT(two.kappa, WithinRel(one.kappa, 1e-12));
}

/// Locates where the s-step arm loses orthogonality and what recovers it.
/// Expected: the second CholeskyQR pass earns about four orders of magnitude,
/// and repeating only the block Gram-Schmidt projection would not, because that
/// loss is intra-block. The cost is one extra Gram all-reduce per block.
TEST_CASE("s-step trades orthogonality for reductions, and CholeskyQR2 buys it back",
          "[arnoldi][ca][stability]")
{
    SyntheticSpec sp;
    sp.n1 = 30; sp.dim = 1;
    const SyntheticOperator op = build_synthetic(sp);
    const Eigen::VectorXd v = deterministic_unit_vector(op.n);

    const CaArnoldiResult plain  = ca_arnoldi(op.A, v, 12, 6, /*reorth=*/false);
    const CaArnoldiResult reorth = ca_arnoldi(op.A, v, 12, 6, /*reorth=*/true);

    REQUIRE_FALSE(plain.broke_down);
    REQUIRE_FALSE(reorth.broke_down);

    // Repeating only the block-GS projection would not touch intra-block loss; the
    // second CholeskyQR pass is what earns the four orders of magnitude.
    REQUIRE(reorth.ortho_loss < plain.ortho_loss / 100.0);
    REQUIRE(reorth.ortho_loss < 1e-12);

    // And it is not free: one extra Gram all-reduce per block.
    REQUIRE(reorth.reductions > plain.reductions);
    REQUIRE(reorth.reductions == ca_reductions(13, 6, true));
}

/// The s-step arm never accumulates H from dot products; it recovers it from
/// the recurrence and the orthogonalization coefficients. This is what makes
/// that equivalent rather than merely similar.
/// Expected: no breakdown, the full dimension reached, and H matching the
/// standard Arnoldi H to 1e-9 while keeping the Hessenberg shape.
TEST_CASE("CA factor assembly matches the standard Arnoldi Hessenberg",
          "[arnoldi][ca][assembly][critical]")
{
    SyntheticSpec sp;
    sp.n1 = 24; sp.dim = 2;
    const SyntheticOperator op = build_synthetic(sp);
    const Eigen::VectorXd v = deterministic_unit_vector(op.n);

    constexpr int m = 8;
    const ArnoldiResult mgs = mgs_arnoldi(op.A, v, m);
    const CaArnoldiResult ca = ca_arnoldi(op.A, v, m, /*s=*/4, /*reorth=*/true);

    REQUIRE_FALSE(ca.broke_down);
    REQUIRE(ca.m_used == m);
    REQUIRE(ca.H.rows() == m + 1);
    REQUIRE(ca.H.cols() == m);
    REQUIRE((ca.H - mgs.H.topLeftCorner(m + 1, m)).norm() < 1e-9);

    for (int col = 0; col < m; ++col)
        for (int row = col + 2; row < m + 1; ++row)
            REQUIRE(std::abs(ca.H(row, col)) < 1e-9);
}

/// The same equivalence under the Chebyshev recurrence, whose H assembly is a
/// different formula rather than the same one with different numbers.
/// Expected: H matches standard Arnoldi to 1e-9, and the Arnoldi relation
/// A*V = V_ext*H holds to 1e-8.
TEST_CASE("Chebyshev CA factor assembly matches standard Arnoldi",
          "[arnoldi][ca][chebyshev][assembly][critical]")
{
    SyntheticSpec sp;
    sp.n1 = 24; sp.dim = 2;
    const SyntheticOperator op = build_synthetic(sp);
    const Eigen::VectorXd v = deterministic_unit_vector(op.n);

    constexpr int m = 8;
    constexpr double center = -0.25;
    constexpr double half_width = 2.0;
    const ArnoldiResult mgs = mgs_arnoldi(op.A, v, m);
    const CaArnoldiResult ca = ca_arnoldi(
        op.A, v, m, /*s=*/4, /*reorth=*/true,
        CaPolynomialBasis::Chebyshev, center, half_width);

    REQUIRE_FALSE(ca.broke_down);
    REQUIRE(ca.m_used == m);
    REQUIRE((ca.H - mgs.H.topLeftCorner(m + 1, m)).norm() < 1e-9);
    REQUIRE((op.A * ca.V - ca.V_extended * ca.H).norm() < 1e-8);
}

/// The certificate kappa(B) <= u^(-1/2) must be both safe to trust inside and
/// strictly better than the obvious alternative of waiting for Cholesky to fail.
///
/// Expected, in three sections. Sound: everywhere inside the certificate,
/// CholeskyQR2 attains O(u). Conservative: plain Cholesky keeps succeeding well
/// past the certificate, and by the time it finally fails its Q has lost more
/// than 1e-3 of orthogonality, so LLT failure must never be the breakdown
/// criterion. And the applied claim: an m=8 cycle fits inside one safe block.
TEST_CASE("the CholQR certificate is sound, and conservative",
          "[arnoldi][cholqr][stability][critical]")
{
    SyntheticSpec sp;
    sp.n1 = 64; sp.dim = 1;
    const SyntheticOperator op = build_synthetic(sp);
    const Eigen::VectorXd v = deterministic_unit_vector(op.n);
    const Eigen::VectorXd c = spectral_coefficients(sp, v);

    const int s_pred = predicted_s_max(op.lambda, c, 40);
    INFO("predicted s_max (certificate) = " << s_pred);

    SECTION("sound: inside the certificate, CholeskyQR2 always attains O(u)") {
        for (int s = 1; s <= s_pred; ++s) {
            const CholQrResult qr = cholesky_qr2(matrix_powers(op.A, v, s));
            REQUIRE_FALSE(qr.llt_failed);
            REQUIRE(qr.kappa < cholqr_kappa_limit());
            REQUIRE(orthogonality_loss(qr.Q) < 1e-13);
        }
    }

    SECTION("conservative: Cholesky keeps succeeding well past the certificate") {
        // So LLT failure must never be used as the breakdown criterion: it lags the
        // real loss of orthogonality by several decades of kappa.
        int s_llt = 0;
        for (int s = 1; s <= 40; ++s) {
            if (cholesky_qr(matrix_powers(op.A, v, s)).llt_failed) break;
            s_llt = s;
        }
        INFO("first LLT failure at s = " << s_llt + 1);
        REQUIRE(s_llt > s_pred);

        // ...and by then plain CholeskyQR's "successful" Q is not a basis at all.
        const CholQrResult late = cholesky_qr(matrix_powers(op.A, v, s_llt));
        REQUIRE_FALSE(late.llt_failed);
        REQUIRE(orthogonality_loss(late.Q) > 1e-3);
    }

    SECTION("the NA claim: an m = 8 exponential-integrator cycle is one safe block") {
        REQUIRE(s_pred >= 8);
        REQUIRE(predicted_basis_condition(op.lambda, c, 8) < cholqr_kappa_limit());
    }
}

/// Conditioning must worsen monotonically with width, or the certificate could
/// not be a single threshold on s.
/// Expected: predicted kappa is non-decreasing across widths 1 to 10.
TEST_CASE("kappa of the monomial basis grows monotonically in s", "[regime][conditioning]")
{
    SyntheticSpec sp;
    sp.n1 = 32; sp.dim = 1;
    const SyntheticOperator op = build_synthetic(sp);
    const Eigen::VectorXd v = deterministic_unit_vector(op.n);
    const Eigen::VectorXd c = spectral_coefficients(sp, v);

    double prev = 0.0;
    for (int s = 1; s <= 10; ++s) {
        const double k = predicted_basis_condition(op.lambda, c, s);
        REQUIRE(k >= prev);
        prev = k;
    }
}

// Krylov dimension prediction

/// The predicted Krylov dimension is an upper bound, so it must not be exceeded
/// by measurement, and must not be so loose as to be useless.
/// Expected: the measured m sits at or below the predicted bound. The bound is
/// a Hermitian positive semidefinite theorem, so it is checked on the scaffold
/// where that hypothesis actually holds.
TEST_CASE("Hochbruck-Lubich bounds the measured Krylov dimension", "[regime][krylov]")
{
    SyntheticSpec sp;
    sp.n1 = 48; sp.dim = 1;
    const SyntheticOperator op = build_synthetic(sp);
    const Eigen::VectorXd v = deterministic_unit_vector(op.n);

    constexpr double h = 1e-2, tol = 1e-8;
    const int m_pred = predict_krylov_dim(op.lambda_min, op.lambda_max, h, tol);
    const int m_meas = measured_krylov_dim(op.A, v, -h, tol, 48);

    INFO("predicted m <= " << m_pred << ", measured m = " << m_meas);
    REQUIRE(m_meas <= m_pred);   // the bound must hold
    REQUIRE(m_pred > 0);
}

/// The Krylov dimension must respond to spectral spread, which is what makes it
/// a controllable coordinate rather than a constant.
/// Expected: widening the spectrum raises the required m.
TEST_CASE("a wider spectrum demands a larger m: the modulator knob works",
          "[regime][krylov][modulator]")
{
    SyntheticSpec narrow;
    narrow.n1 = 48; narrow.dim = 1;
    SyntheticSpec wide = narrow;
    wide.spec_scale = 20.0;

    const SyntheticOperator n_op = build_synthetic(narrow);
    const SyntheticOperator w_op = build_synthetic(wide);

    const int m_narrow = predict_krylov_dim(n_op.lambda_min, n_op.lambda_max, 1e-2, 1e-8);
    const int m_wide   = predict_krylov_dim(w_op.lambda_min, w_op.lambda_max, 1e-2, 1e-8);
    REQUIRE(m_wide > m_narrow);
}

// The per-vector certificate: blocks = ceil(m/(s_max+1)), and the anti-correlation the
// ledger must respect (an eigenvector start: s_max=0 but m=1, still one block).

/// The limiting adversary for the block-count ledger: an exact eigenvector start
/// makes [v, Av] rank-1, so the certified width collapses to zero, yet the
/// Krylov space is one-dimensional and one block is genuinely enough.
/// Expected: blocks = 1. A ledger that divided by the certified width alone
/// would divide by zero or demand infinitely many blocks here.
TEST_CASE("an eigenvector start needs exactly one block despite s_max=0",
          "[regime][certificate][critical]")
{
    // The limiting adversary the certificate must handle: v an exact eigenvector makes
    // [v, Av] rank-1 (s_max=0), but the Krylov space is 1-D so demand m=1. blocks=1.
    SyntheticSpec sp; sp.n1 = 20; sp.dim = 1;
    const SyntheticOperator op = build_synthetic(sp);

    // Build an exact eigenvector: q_k(i) = sin((i+1)(k+1) pi / (n1+1)).
    const int n = static_cast<int>(op.n);
    Eigen::VectorXd q(n);
    for (int i = 0; i < n; ++i)
        q(i) = std::sin((i + 1.0) * 3.0 * std::numbers::pi_v<double> / (n + 1.0));
    q.normalize();

    const int s_max = measured_s_max(op.A, q, 20);
    REQUIRE(s_max == 0);                          // supply collapses

    const int m = measured_krylov_dim(op.A, q, -1e-2, 1e-8, 40);
    REQUIRE(m == 1);                              // but demand collapses too

    const int blocks = (m + s_max) / (s_max + 1); // ceil(m/(s_max+1))
    REQUIRE(blocks == 1);                          // one block suffices -- not a failure
}

/// The physical justification for not certifying against the dominant-mode
/// adversary: a diffusive solution is smooth and therefore small-lambda heavy.
/// Expected: a smoothed low-frequency start conditions better than the top sine
/// mode, so the width the solver actually gets is wider than the worst case.
TEST_CASE("smooth (diffusive) vectors condition better than dominant-mode vectors",
          "[regime][certificate]")
{
    // The physical justification for not certifying against the dominant-mode adversary:
    // a diffusive solution is smooth (small-lambda heavy) and conditions better.
    SyntheticSpec sp; sp.n1 = 24; sp.dim = 2;
    const SyntheticOperator op = build_synthetic(sp);
    const int n = static_cast<int>(op.n);

    // Smooth vector: low-frequency (small-lambda) modes only. Dominant vector: the
    // largest-lambda eigenvector direction (here approximated by the top sine mode).
    std::mt19937_64 rng(1);
    std::normal_distribution<double> nd(0, 1);
    Eigen::VectorXd rnd(n);
    for (int i = 0; i < n; ++i) rnd(i) = nd(rng);

    // A few explicit-diffusion smoothing steps damp the high modes.
    double radius = 0.0;
    for (int k = 0; k < op.A.outerSize(); ++k) {
        double rs = 0.0;
        for (SpMatS::InnerIterator it(op.A, k); it; ++it) rs += std::abs(it.value());
        radius = std::max(radius, rs);
    }
    Eigen::VectorXd smooth = rnd;
    for (int k = 0; k < 32; ++k)
        smooth -= (0.5 / radius) * (op.A * smooth);
    smooth.normalize();

    Eigen::VectorXd dominant = rnd;
    for (int k = 0; k < 8; ++k) { dominant = op.A * dominant; dominant.normalize(); }

    REQUIRE(measured_s_max(op.A, smooth, 20) >= measured_s_max(op.A, dominant, 20));
}
