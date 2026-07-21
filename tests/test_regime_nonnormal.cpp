/**
 * @file test_regime_nonnormal.cpp
 * @brief Non-normality: the transferability bridge to the real Black-Scholes operator.
 *
 * Advection keeps an analytic spectrum but breaks symmetry, and its non-normality is
 * diagonally removable, correlation destroys separability and is not. The honest
 * distinction between the two is what scopes the spectrum-alone prediction.
 *
 * @author Kevin Knights
 * @date 2026-07-10
 */

#include "test_regime_helpers.hpp"

#include "regime_control_support.hpp"   // structured_kappa_X, unit_norm_kappa_X

using namespace regime_test;

// Non-normality: the transferability bridge to the real Black-Scholes operator

namespace {

// Real parts of a general (possibly non-symmetric) operator's eigenvalues, sorted.
Eigen::VectorXd sorted_general_spectrum(const SpMatS& A, double& max_imag_out)
{
    Eigen::EigenSolver<Eigen::MatrixXd> es(Eigen::MatrixXd(A), false);
    max_imag_out = es.eigenvalues().imag().cwiseAbs().maxCoeff();
    Eigen::VectorXd re = es.eigenvalues().real();
    std::sort(re.data(), re.data() + re.size());
    return re;
}

}  // namespace

TEST_CASE("advection keeps the spectrum real and analytic but breaks symmetry",
          "[synthetic][nonnormal][critical]")
{
    SyntheticSpec sp;
    sp.n1 = 12; sp.dim = 2; sp.advection = 0.2;
    const SyntheticOperator op = build_synthetic(sp);

    // The assembled operator is genuinely nonsymmetric...
    const Eigen::MatrixXd D(op.A);
    REQUIRE((D - D.transpose()).norm() > 1e-3);
    REQUIRE_FALSE(op.is_normal);

    // ...yet its spectrum is real and matches the analytic Toeplitz formula.
    double max_imag = 0.0;
    const Eigen::VectorXd numeric = sorted_general_spectrum(op.A, max_imag);
    Eigen::VectorXd analytic = op.lambda;
    std::sort(analytic.data(), analytic.data() + analytic.size());
    REQUIRE(max_imag < 1e-9);
    REQUIRE((analytic - numeric).cwiseAbs().maxCoeff() < 1e-9);
}

TEST_CASE("advection compresses the spectral spread by sqrt(1-gamma^2)",
          "[synthetic][nonnormal]")
{
    SyntheticSpec base; base.n1 = 40; base.dim = 1;
    SyntheticSpec adv = base; adv.advection = 0.6;

    const SyntheticOperator b = build_synthetic(base);
    const SyntheticOperator a = build_synthetic(adv);

    const double spread_b = b.lambda_max - b.lambda_min;
    const double spread_a = a.lambda_max - a.lambda_min;
    REQUIRE_THAT(spread_a / spread_b, WithinRel(std::sqrt(1.0 - 0.6 * 0.6), 1e-9));
}

TEST_CASE("eigenvector_condition matches the dense eigenvector matrix condition",
          "[synthetic][nonnormal][critical]")
{
    SyntheticSpec sp; sp.n1 = 10; sp.dim = 1; sp.advection = 0.2;
    const SyntheticOperator op = build_synthetic(sp);

    // Analytic kappa(X) = r^(n1-1), r = sqrt((1+g)/(1-g)).
    const double r = std::sqrt(1.2 / 0.8);
    REQUIRE_THAT(op.eigvec_condition, WithinRel(std::pow(r, 9.0), 1e-9));

    // Against the dense eigenvector matrix (my convention: columns r^i * sine).
    const int n = static_cast<int>(op.n);
    Eigen::MatrixXd X(n, n);
    const double sc = std::sqrt(2.0 / (n + 1));
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k)
            X(i, k) = std::pow(r, i) * sc
                    * std::sin((i + 1.0) * (k + 1.0) * std::numbers::pi_v<double> / (n + 1.0));
    REQUIRE_THAT(condition_number(X), WithinRel(op.eigvec_condition, 1e-6));
    REQUIRE(op.eigvec_condition > 5.0);   // genuinely non-normal
}

TEST_CASE("spectral_coefficients gives exact eigenbasis coordinates for non-normal A",
          "[synthetic][nonnormal][critical]")
{
    // The load-bearing check for the non-normal prediction: c = X^{-1} v must be exact,
    // i.e. X * M(c) reconstructs [v, Av, ..., A^s v]. If this drifts, the predicted
    // kappa is meaningless.
    SyntheticSpec sp; sp.n1 = 10; sp.dim = 1; sp.advection = 0.2;
    const SyntheticOperator op = build_synthetic(sp);
    const int n = static_cast<int>(op.n);
    const double r = std::sqrt(1.2 / 0.8);

    Eigen::MatrixXd X(n, n);
    const double sc = std::sqrt(2.0 / (n + 1));
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k)
            X(i, k) = std::pow(r, i) * sc
                    * std::sin((i + 1.0) * (k + 1.0) * std::numbers::pi_v<double> / (n + 1.0));

    const Eigen::VectorXd v = deterministic_unit_vector(n);
    const Eigen::VectorXd c = spectral_coefficients(sp, v);

    const int s = 5;
    Eigen::MatrixXd M(n, s + 1);
    M.col(0) = c;
    for (int k = 1; k <= s; ++k) M.col(k) = M.col(k - 1).cwiseProduct(op.lambda);

    const Eigen::MatrixXd via_X  = X * M;
    const Eigen::MatrixXd sparse = matrix_powers(op.A, v, s);
    REQUIRE((via_X - sparse).norm() / sparse.norm() < 1e-10);
}

TEST_CASE("the scatter knob stays a similarity transform under advection",
          "[synthetic][nonnormal][scatter]")
{
    // The similarity transform must survive non-normality: permutation preserves eigenvalues
    // AND the eigenvector conditioning, so pure-axis motion still exists for the BS-like regime.
    SyntheticSpec banded; banded.n1 = 12; banded.dim = 2; banded.advection = 0.15;
    SyntheticSpec scattered = banded; scattered.scatter_block = 12 * 12;

    const SyntheticOperator b = build_synthetic(banded);
    const SyntheticOperator s = build_synthetic(scattered);

    REQUIRE(b.nnz == s.nnz);
    REQUIRE_THAT(b.eigvec_condition, WithinRel(s.eigvec_condition, 1e-12));

    double imb = 0.0, ims = 0.0;
    const Eigen::VectorXd lb = sorted_general_spectrum(b.A, imb);
    const Eigen::VectorXd ls = sorted_general_spectrum(s.A, ims);
    REQUIRE((lb - ls).cwiseAbs().maxCoeff() < 1e-9);
}

TEST_CASE("the conditioning gap grows LOGARITHMICALLY in kappa(X), far under the bound",
          "[regime][nonnormal][critical]")
{
    // The rigorous bound gap <= log10(kappa(X)) is loose.
    // The empirical error is far smaller (shallow logarithmic slope), which is why the
    // spectral prediction survives non-normality. Assert the tight empirical form so the
    // test detects a real change, not the vacuous bound.
    for (double gamma : {0.0, 0.1, 0.2, 0.3}) {
        SyntheticSpec sp; sp.n1 = 16; sp.dim = 2; sp.advection = gamma;
        const SyntheticOperator op = build_synthetic(sp);
        const Eigen::VectorXd v = deterministic_unit_vector(op.n);
        const Eigen::VectorXd c = spectral_coefficients(sp, v);

        const int s = 5;
        const double k_meas = condition_number(matrix_powers(op.A, apply_perm(op.perm, v), s));
        const double k_pred = predicted_basis_condition(op.lambda, c, s);
        const double gap_decades = std::abs(std::log10(k_meas) - std::log10(k_pred));

        INFO("gamma=" << gamma << " kappa(X)=" << op.eigvec_condition
             << " gap=" << gap_decades << " decades");
        if (gamma == 0.0)
            REQUIRE(gap_decades < 0.02);                        // normal: exact
        else
            REQUIRE(gap_decades <= 0.20 * std::log10(op.eigvec_condition));  // empirical, tight
    }
}

TEST_CASE("advection is rejected outside |gamma| < 1", "[synthetic][nonnormal]")
{
    SyntheticSpec sp; sp.n1 = 8; sp.dim = 1; sp.advection = 1.0;
    REQUIRE_THROWS_AS(build_synthetic(sp), std::invalid_argument);
    sp.advection = -1.5;
    REQUIRE_THROWS_AS(build_synthetic(sp), std::invalid_argument);
}

// The honest distinction: advection is diagonally removable, correlation is not.

TEST_CASE("Henrici departure is zero iff the operator is normal", "[synthetic][nonnormal][henrici]")
{
    SyntheticSpec lap; lap.n1 = 10; lap.dim = 2;                       // symmetric
    SyntheticSpec adv = lap; adv.advection = 0.3;                      // non-normal
    SyntheticSpec cor = lap; cor.correlation = 0.3;                    // symmetric cross term

    REQUIRE(henrici_departure(build_synthetic(lap).A) < 1e-12);
    REQUIRE(henrici_departure(build_synthetic(adv).A) > 1.0);
    // A symmetric mixed-derivative term is still normal: correlation ALONE has zero
    // departure. Non-normality needs advection.
    REQUIRE(henrici_departure(build_synthetic(cor).A) < 1e-12);
}

TEST_CASE("advection non-normality is diagonally removable", "[synthetic][nonnormal][critical]")
{
    // The reviewer's required verification: A = D Atilde D^{-1}, so D^{-1} A D is
    // symmetric to machine precision. The "non-normality" is a fixed diagonal.
    SyntheticSpec sp; sp.n1 = 14; sp.dim = 2; sp.advection = 0.25;
    const SyntheticOperator op = build_synthetic(sp);
    REQUIRE(op.henrici > 1.0);   // genuinely nonsymmetric

    const Eigen::VectorXd d = advection_symmetrizer(sp);
    REQUIRE(d.size() == op.n);
    const Eigen::MatrixXd A(op.A);
    const Eigen::MatrixXd B = d.cwiseInverse().asDiagonal() * A * d.asDiagonal();
    REQUIRE((B - B.transpose()).norm() < 1e-10);   // symmetric after D: REMOVABLE
}

TEST_CASE("correlation destroys separability: no analytic spectrum, not removable",
          "[synthetic][nonnormal][correlation][critical]")
{
    SyntheticSpec sp; sp.n1 = 12; sp.dim = 2; sp.correlation = 0.3; sp.advection = 0.2;
    const SyntheticOperator op = build_synthetic(sp);

    // No analytic spectrum, and lambda is not populated.
    REQUIRE_FALSE(op.has_analytic_spectrum);
    REQUIRE(op.lambda.size() == 0);
    REQUIRE_FALSE(op.is_normal);               // advection makes it genuinely non-normal
    REQUIRE(op.henrici > 1.0);

    // The advection diagonal does NOT symmetrize it (advection_symmetrizer returns empty
    // when correlation != 0, precisely because no such diagonal exists).
    REQUIRE(advection_symmetrizer(sp).size() == 0);

    // Applying the pure-advection diagonal anyway leaves it far from symmetric.
    SyntheticSpec adv_only = sp; adv_only.correlation = 0.0;
    const Eigen::VectorXd d = advection_symmetrizer(adv_only);
    const Eigen::MatrixXd A(op.A);
    const Eigen::MatrixXd B = d.cwiseInverse().asDiagonal() * A * d.asDiagonal();
    REQUIRE((B - B.transpose()).norm() > 0.1);   // NOT removable
}

TEST_CASE("correlation alone is normal but non-separable", "[synthetic][nonnormal][correlation]")
{
    // A symmetric cross term: normal (measured departure ~0) yet no analytic spectrum.
    // The two flags are genuinely independent.
    SyntheticSpec sp; sp.n1 = 12; sp.dim = 2; sp.correlation = 0.3;
    const SyntheticOperator op = build_synthetic(sp);
    REQUIRE(op.is_normal);                 // measured from henrici
    REQUIRE_FALSE(op.has_analytic_spectrum);
}

TEST_CASE("normal non-separable: the DENSE-eigensolver Vandermonde prediction is exact",
          "[regime][nonnormal][correlation][critical]")
{
    // The reviewer's experiment 1: correlation has no closed-form spectrum but is normal,
    // so a dense symmetric eigensolver gives lambda and Q, and predicted_basis_condition
    // matches the measured basis conditioning exactly.
    SyntheticSpec sp; sp.n1 = 16; sp.dim = 2; sp.correlation = 0.3;
    const SyntheticOperator op = build_synthetic(sp);
    REQUIRE(op.is_normal);
    REQUIRE_FALSE(op.has_analytic_spectrum);

    const Eigen::VectorXd v = deterministic_unit_vector(op.n);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Eigen::MatrixXd(op.A));
    const Eigen::VectorXd lam = es.eigenvalues();
    const Eigen::VectorXd c   = es.eigenvectors().transpose() * v;   // Q^T v

    for (int s = 1; s <= 5; ++s) {
        const double meas = condition_number(matrix_powers(op.A, v, s));
        const double pred = predicted_basis_condition(lam, c, s);
        REQUIRE_THAT(std::log10(meas), WithinAbs(std::log10(pred), 0.02));  // exact
    }
}

TEST_CASE("genuinely non-normal: the prediction gap is far under the kappa(X) bound",
          "[regime][nonnormal][correlation][critical]")
{
    // The reviewer's experiment 2: rho+gamma is genuinely non-normal (henrici>0, not
    // diagonally removable). A complex eigendecomposition gives kappa(X), and the
    // measured/predicted conditioning gap is bounded by it: a quantitative law.
    SyntheticSpec sp; sp.n1 = 14; sp.dim = 2; sp.correlation = 0.3; sp.advection = 0.2;
    const SyntheticOperator op = build_synthetic(sp);
    REQUIRE_FALSE(op.is_normal);
    REQUIRE_FALSE(op.has_analytic_spectrum);

    const Eigen::VectorXd v = deterministic_unit_vector(op.n);
    Eigen::EigenSolver<Eigen::MatrixXd> es(Eigen::MatrixXd(op.A));
    const Eigen::VectorXcd lam = es.eigenvalues();
    const Eigen::MatrixXcd X   = es.eigenvectors();
    const double kappa_X = condition_number(X);
    REQUIRE(kappa_X > 10.0);   // genuinely non-normal

    const Eigen::VectorXcd cc =
        X.colPivHouseholderQr().solve(v.cast<std::complex<double>>());

    const int s = 5;
    Eigen::MatrixXcd M(lam.size(), s + 1);
    M.col(0) = cc;
    for (int k = 1; k <= s; ++k) M.col(k) = M.col(k - 1).cwiseProduct(lam);

    const double meas = condition_number(matrix_powers(op.A, v, s));
    const double pred = condition_number(M);
    const double gap  = std::abs(std::log10(meas) - std::log10(pred));
    INFO("kappa(X)=" << kappa_X << " gap=" << gap << " bound=" << std::log10(kappa_X));

    // The rigorous bound gap <= log10(kappa(X)) is nearly vacuous (decades of slack), so
    // it would never trip on a regression. Assert against the empirical law instead: the
    // error grows logarithmically with a shallow slope (~0.09), so gap stays well under
    // 0.2 * log10(kappa(X)). This threshold actually detects a change in the instrument.
    REQUIRE(gap <= 0.20 * std::log10(kappa_X));
}

TEST_CASE("correlation needs dim >= 2", "[synthetic][correlation]")
{
    SyntheticSpec sp; sp.n1 = 8; sp.dim = 1; sp.correlation = 0.2;
    REQUIRE_THROWS_AS(build_synthetic(sp), std::invalid_argument);
}

TEST_CASE("the prediction error is MECHANISM-dependent, not a function of kappa(X)",
          "[regime][nonnormal][mechanism][critical]")
{
    // Why the "law" is scoped to one family: at equal eigenvector conditioning kappa(X),
    // variable-coefficient advection degrades the spectral prediction far more than
    // constant-coefficient advection, so the gap is not a function of kappa(X) alone.
    // Mirroring the control, each operator uses its reliable prediction: analytic for a
    // diagonally-similar (real-spectrum) operator, dense-complex only where there is no
    // closed form (the general complex eigensolver returns inaccurate eigenvectors on a
    // real-spectrum nonsymmetric matrix and would spuriously inflate the gap). kappa(X) is
    // unit-2-norm dense for both, for a consistent x-axis.
    const int s = 6;
    auto gap_and_kappaX = [&](const SyntheticSpec& sp) {
        const SyntheticOperator op = build_synthetic(sp);
        const Eigen::VectorXd v = deterministic_unit_vector(op.n);
        const double meas = condition_number(matrix_powers(op.A, v, s));
        Eigen::EigenSolver<Eigen::MatrixXd> es(Eigen::MatrixXd(op.A));
        const double kX = condition_number(es.eigenvectors());

        double pred;
        if (op.has_analytic_spectrum) {                        // reliable analytic path
            const Eigen::VectorXd c = spectral_coefficients(sp, v);
            pred = predicted_basis_condition(op.lambda, c, s);
        } else {                                               // dense complex path
            const Eigen::VectorXcd lam = es.eigenvalues();
            const Eigen::VectorXcd cc = es.eigenvectors().colPivHouseholderQr()
                                            .solve(v.cast<std::complex<double>>());
            Eigen::MatrixXcd M(lam.size(), s + 1);
            M.col(0) = cc;
            for (int k = 1; k <= s; ++k) M.col(k) = M.col(k - 1).cwiseProduct(lam);
            pred = condition_number(M);
        }
        return std::pair{kX, std::abs(std::log10(meas) - std::log10(pred))};
    };

    // Tune each mechanism to a comparable kappa(X) ~ 1e4, then compare gaps.
    SyntheticSpec cst; cst.n1 = 24; cst.dim = 2; cst.advection = 0.1;      // kappa(X) ~ 4e4
    SyntheticSpec var; var.n1 = 24; var.dim = 2; var.var_advection = 0.8;  // kappa(X) ~ 4e4
    const auto [kx_c, gap_c] = gap_and_kappaX(cst);
    const auto [kx_v, gap_v] = gap_and_kappaX(var);

    INFO("const: kappa(X)=" << kx_c << " gap=" << gap_c
         << " | var: kappa(X)=" << kx_v << " gap=" << gap_v);
    // Comparable kappa(X) (within a decade)...
    REQUIRE(std::abs(std::log10(kx_c) - std::log10(kx_v)) < 1.0);
    // ...but the variable-coefficient gap is several times larger: not kappa(X) alone.
    REQUIRE(gap_v > 3.0 * gap_c);
}

TEST_CASE("variable-coefficient advection has no analytic spectrum and is non-normal",
          "[synthetic][nonnormal][var]")
{
    SyntheticSpec sp; sp.n1 = 16; sp.dim = 2; sp.var_advection = 0.5;
    const SyntheticOperator op = build_synthetic(sp);
    REQUIRE_FALSE(op.has_analytic_spectrum);   // not Toeplitz -> no closed form
    REQUIRE_FALSE(op.is_normal);               // genuinely non-normal
    REQUIRE(op.henrici > 1.0);
}

// The asset-dimension law. The cross term couples axes 0 and 1 and no others, so the
// operator factors as A_d = B01 (+) T (+) ... (+) T and kappa(X) is exactly
// kappa(X01)*kappa(X1)^(d-2). These pin the two halves of that claim: the canonical value is
// right wherever it can be checked, and the dense value it replaces is checkable nowhere
// else, because the leftover axes are interchangeable and the spectrum repeats.

TEST_CASE("structured_kappa_X matches the dense value wherever the spectrum is simple",
          "[synthetic][nonnormal][correlation][critical]")
{
    // dim - 2 cross-term-free axes; fewer than 2 of them means no interchange symmetry,
    // hence a simple spectrum and a dense kappa(X) that is genuinely an operator property.
    struct Case { int n1, dim; double gamma, rho; };
    for (const Case c : {Case{4, 2, 0.3, 0.3}, Case{4, 3, 0.3, 0.3},
                         Case{8, 3, 0.3, 0.6}, Case{8, 3, 0.3, 0.99}}) {
        SyntheticSpec sp;
        sp.n1 = c.n1; sp.dim = c.dim; sp.advection = c.gamma; sp.correlation = c.rho;

        const double dense = regime_control::unit_norm_kappa_X(build_synthetic(sp));
        const double structured = regime_control::structured_kappa_X(sp);

        INFO("n1=" << c.n1 << " dim=" << c.dim << " rho=" << c.rho
             << " dense=" << dense << " structured=" << structured);
        REQUIRE(std::isfinite(structured));
        REQUIRE_THAT(structured, WithinRel(dense, 1e-6));
    }
}

TEST_CASE("the tensor law makes log kappa(X) exactly linear in the asset dimension",
          "[synthetic][nonnormal][correlation]")
{
    SyntheticSpec sp;
    sp.n1 = 4; sp.advection = 0.3; sp.correlation = 0.3;

    // Each extra axis is one more Kronecker factor, so it multiplies kappa(X) by exactly
    // kappa(X1), the same ratio at every dimension, with no C(dim,2) term anywhere.
    SyntheticSpec s1 = sp;
    s1.dim = 1; s1.correlation = 0.0;
    const double k1 = regime_control::unit_norm_kappa_X(build_synthetic(s1));
    REQUIRE(k1 > 1.0);

    for (int d = 2; d <= 6; ++d) {
        sp.dim = d;
        SyntheticSpec next = sp;
        next.dim = d + 1;
        const double ratio = regime_control::structured_kappa_X(next)
                           / regime_control::structured_kappa_X(sp);
        INFO("dim " << d << " -> " << d + 1 << " ratio=" << ratio << " kappa(X1)=" << k1);
        REQUIRE_THAT(ratio, WithinRel(k1, 1e-9));
    }
}

TEST_CASE("interchangeable axes make the spectrum degenerate, so dense kappa(X) is "
          "basis-dependent", "[synthetic][nonnormal][correlation][critical]")
{
    // dim = 4 leaves axes 2 and 3 carrying nothing that distinguishes them, so swapping them
    // is an exact symmetry of A: eigenvalues repeat, every basis of a repeated eigenspace is
    // admissible, and the dense eigensolver's arbitrary choice is not a function of A.
    SyntheticSpec sp;
    sp.n1 = 4; sp.dim = 4; sp.advection = 0.3; sp.correlation = 0.3;
    const SyntheticOperator op = build_synthetic(sp);

    Eigen::EigenSolver<Eigen::MatrixXd> es(Eigen::MatrixXd(op.A), false);
    Eigen::VectorXd re = es.eigenvalues().real();
    std::sort(re.data(), re.data() + re.size());
    int distinct = 1;
    for (Eigen::Index i = 1; i < re.size(); ++i)
        if (std::abs(re(i) - re(i - 1)) > 1e-8 * (1.0 + std::abs(re(i)))) ++distinct;

    INFO("N=" << op.n << " distinct eigenvalues=" << distinct);
    REQUIRE(distinct < op.n * 3 / 4);   // measured: 144 of 256

    // The canonical value, by contrast, is built from the factors and cannot depend on how
    // the assembled operator happens to be ordered. The scatter knob is a similarity
    // (P A P^T), so it is the sharpest available restatement of "same operator".
    SyntheticSpec scattered = sp;
    scattered.scatter_block = synthetic_dimension(sp);
    REQUIRE_THAT(regime_control::structured_kappa_X(scattered),
                 WithinRel(regime_control::structured_kappa_X(sp), 1e-12));
}

TEST_CASE("a symmetric permutation moves dense kappa(X) but not the canonical value",
          "[synthetic][nonnormal][correlation][critical]")
{
    // The quantitative half of the claim above, and the source of the spread quoted in the
    // README. P A P^T is an exact similarity, so any function of the OPERATOR is invariant
    // under it. Sweeping the seed of the scatter permutation sweeps P.
    //
    // Measured over 8 permutations: the dense value is pinned to rounding while the
    // spectrum is simple and comes apart once it repeats.
    //
    //   dim     N   perm min   perm max      spread   structured
    //     2    16      6.671      6.671   1 + 3e-15        6.671
    //     3    64     16.951     16.951   1 + 8e-14       16.951
    //     4   256     81.815    314.338        3.84x       43.073
    //     5  1024    373.9     2903            7.76x      109.45     (not run here: cost)
    //
    // dim = 5 is left out of the loop because 8 dense eigensolves plus SVDs at N = 1024
    // dominate the suite's runtime; dim = 4 already exhibits the effect.
    constexpr int kSeeds = 8;

    struct Case { int dim; double max_spread; };   // dim 2,3: invariant. dim 4: it is not.
    for (const Case c : {Case{2, 1.0 + 1e-9}, Case{3, 1.0 + 1e-9}, Case{4, 0.0}}) {
        SyntheticSpec sp;
        sp.n1 = 4; sp.dim = c.dim; sp.advection = 0.3; sp.correlation = 0.3;

        double lo = std::numeric_limits<double>::max(), hi = 0.0;
        for (uint64_t seed = 1; seed <= kSeeds; ++seed) {
            SyntheticSpec p = sp;
            p.scatter_block = synthetic_dimension(sp);   // full symmetric shuffle
            p.seed          = seed;
            const double k = regime_control::unit_norm_kappa_X(build_synthetic(p));
            REQUIRE(std::isfinite(k));
            lo = std::min(lo, k);
            hi = std::max(hi, k);
            // The canonical value is a function of the factors, so every P leaves it fixed.
            REQUIRE_THAT(regime_control::structured_kappa_X(p),
                         WithinRel(regime_control::structured_kappa_X(sp), 1e-12));
        }

        INFO("dim=" << c.dim << " dense kappa(X) in [" << lo << ", " << hi
             << "], spread " << hi / lo << "x");
        if (c.dim < 4)
            REQUIRE(hi / lo <= c.max_spread);       // simple spectrum: a genuine invariant
        else
            REQUIRE(hi / lo > 2.0);                 // degenerate: not an operator property
    }
}
