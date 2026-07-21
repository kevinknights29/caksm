/**
 * @file regime_control_support.hpp
 * @brief Support layer for the regime-control gate (the regime-control executable).
 *
 * Everything the C1-C6 checks stand on -- the operators under test, argument parsing, the
 * check-recording ledger, the spectral decomposition, and the physical starting-vector
 * ensembles, factored out of regime_control.cpp so that file holds only the checks and
 * main(). Symbols live in namespace regime_control and the free functions are inline, so the
 * header is safe to include from more than one translation unit.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <format>
#include <limits>
#include <numeric>
#include <print>
#include <random>
#include <utility>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "ca_arnoldi.hpp"
#include "machine.hpp"
#include "pde_operators.hpp"
#include "regime.hpp"
#include "synthetic.hpp"

namespace regime_control {

/**
 * @brief The real discretised Black-Scholes basket operator, wrapped as a SyntheticOperator
 *        so it runs through the identical control machinery.
 *
 * Closes the transfer argument by measurement rather than category membership. Assembled
 * in log-price coordinates, where sigma, rho and r are constant, so despite the
 * correlation cross terms it belongs to the constant-coefficient family; C6 places it on
 * the same gap-vs-kappa(X) figure as the synthetic families, as its own marker excluded
 * from the law fits. Measured across n = 10,...,20 at the solver's own h = t_final/steps:
 *
 *     n:         10    12    14    16    18    20
 *     kappa(X):  46    39   706    85   140  2304
 *     gap (dec): 0.20  0.20  0.37  0.26  0.36  0.85
 *
 * kappa(X) is neither bounded nor monotone under refinement, and the gap climbs toward
 * one s-step at n=20 rather than staying benign, yet the load-bearing claim survives:
 * every gap stays under one s-step and the measured s_max held usable at 5-7 across every
 * grid, so s-step is safe here because the certificate is measured, not because kappa(X)
 * is benign.
 *
 * The spikes are physical, not a dense-eigensolver artifact: re-decomposing with LAPACK
 * zgeev reproduces every kappa(X) to four significant figures including the spikes, the
 * Wilkinson condition number spikes in lockstep, and eigenpair residuals stay at ~1e-13.
 * The operator is genuinely near-defective at those grids, where a mode's eigenvectors go
 * nearly orthogonal as the mesh brings an eigenvalue pair toward coalescence, which is why
 * the certificate must be measured rather than read from a spectrum blind to it.
 *
 * @param n      grid points per asset (N = n^3), keep small (C4/C6 dense-eigensolve).
 * @param v0_out the operator's actual starting vector: the discounted payoff u0.
 */
inline SyntheticOperator real_bs_operator(int n, Eigen::VectorXd& v0_out)
{
    const std::array<double, 3> sigma{0.30, 0.35, 0.40};
    const std::array<double, 3> rho_off{0.50, 0.50, 0.50};
    const std::array<double, 3> weight{1.0 / 3, 1.0 / 3, 1.0 / 3};
    const std::array<double, 3> s0{100.0, 100.0, 100.0};
    const PDESystem sys = build_pde_system(n, 100.0, 0.04, 1.0, sigma, rho_off,
                                           weight, s0, 2.85, /*rainbow_bc=*/false);

    SyntheticOperator op;
    op.A   = sys.A;
    op.n   = sys.N;
    op.nnz = sys.A.nonZeros();
    op.perm.resize(static_cast<std::size_t>(op.n));
    std::iota(op.perm.begin(), op.perm.end(), int64_t{0});   // identity: no scatter here

    op.henrici = henrici_departure(op.A);
    const double fro = op.A.norm();
    op.is_normal             = (op.henrici <= 1e-9 * fro * fro);
    op.has_analytic_spectrum = false;      // cross terms: no closed form, dense eigensolve
    op.lambda                = Eigen::VectorXd();
    op.lambda_min = op.lambda_max = std::numeric_limits<double>::quiet_NaN();
    op.eigvec_condition       = std::numeric_limits<double>::quiet_NaN();

    v0_out = sys.u0;
    const double nrm = v0_out.norm();
    if (nrm > 0.0) v0_out /= nrm;
    return op;
}

struct Args {
    SyntheticSpec spec{};
    /// Ceiling for the Krylov-dimension search. m itself is never a dial (it enters the
    /// working set and the reductions-per-cycle, moving both ratios), in the real problem it
    /// is an output of (n, tol), not an input.
    int         m_ceiling  = 64;
    double      h          = 1e-2;   ///< exponential-integrator time step
    double      tol        = 1e-8;   ///< KSM-EI convergence tolerance
    int         s_ceiling  = 64;     ///< how far the stability probe climbs
    std::string machine    = "amd-3960x";
    int         P          = 24;
    int         real_bs    = 0;   ///< >0: run the real Black-Scholes operator at this n (N=n^3)
    std::string csv_path;
};

inline Args parse(std::span<const char* const> argv)
{
    Args a;
    a.spec.n1  = 24;   // 24^2 = 576: small enough for a dense eigensolver in C1
    a.spec.dim = 2;

    for (std::size_t i = 1; i < argv.size(); ++i) {
        const std::string_view arg = argv[i];
        auto next = [&]() -> std::string_view {
            if (++i >= argv.size())
                throw std::invalid_argument("Missing value for " + std::string(arg));
            return argv[i];
        };
        if      (arg == "--n1")         a.spec.n1 = std::stoi(std::string(next()));
        else if (arg == "--dim")        a.spec.dim = std::stoi(std::string(next()));
        else if (arg == "--scale")      a.spec.spec_scale = std::stod(std::string(next()));
        else if (arg == "--shift")      a.spec.spec_shift = std::stod(std::string(next()));
        else if (arg == "--advection")  a.spec.advection  = std::stod(std::string(next()));
        else if (arg == "--correlation") a.spec.correlation = std::stod(std::string(next()));
        else if (arg == "--var-advection") a.spec.var_advection = std::stod(std::string(next()));
        else if (arg == "--seed")       a.spec.seed = std::stoull(std::string(next()));
        else if (arg == "--m-ceiling")   a.m_ceiling = std::stoi(std::string(next()));
        else if (arg == "--h")          a.h = std::stod(std::string(next()));
        else if (arg == "--tol")        a.tol = std::stod(std::string(next()));
        else if (arg == "--s-ceiling")  a.s_ceiling = std::stoi(std::string(next()));
        else if (arg == "--machine")    a.machine = std::string(next());
        else if (arg == "--P")          a.P = std::stoi(std::string(next()));
        else if (arg == "--real-bs")    a.real_bs = std::stoi(std::string(next()));
        else if (arg == "--csv")        a.csv_path = std::string(next());
        else if (arg == "--help") {
            std::println("Usage: ./regime-control [--n1 N] [--dim 1|2|3] [--h H] [--tol T]");
            std::println("                        [--scale S] [--shift MU] [--s-ceiling S]");
            std::println("                        [--m-ceiling M] [--seed SEED]");
            std::println("                        [--machine KEY] [--P P] [--csv PATH]");
            std::println("");
            std::println("  Control gate. Exits non-zero if any GATING check fails.");
            std::println("  --dim D        Laplacian scaffold dimension (d = 2D+1 nonzeros/row)");
            std::println("  --scale S      spectrum knob: lambda -> S * (lambda - shift)");
            std::println("  --shift MU     spectrum knob: translates, does NOT change m");
            std::println("  --advection G  DIAGONALLY-REMOVABLE non-normality: gamma in (-1,1).");
            std::println("                 Convection-diffusion Toeplitz; A = D Atilde D^-1.");
            std::println("  --correlation R  GENUINE non-normality: mixed-derivative cross term");
            std::println("                 (needs dim>=2). Destroys the Kronecker sum -> no");
            std::println("                 analytic spectrum -> s_max must be MEASURED (C6).");
            std::println("  --var-advection V  SECOND non-normality mechanism: advection ramps");
            std::println("                 with position (not Toeplitz). Tests whether the");
            std::println("                 gap-vs-kappa(X) law is mechanism-independent (C6).");
            std::println("  --h H / --tol  move m; m is MEASURED, never set directly");
            std::println("  --m-ceiling M  ceiling for the Krylov-dimension search (default 64)");
            std::println("  --real-bs n    run the ACTUAL 3-asset Black-Scholes basket operator");
            std::println("                 (N=n^3) with its payoff u0 as the start vector, instead");
            std::println("                 of the synthetic scaffold. Transfer by MEASUREMENT (C6);");
            std::println("                 overrides --n1/--dim/knobs. n<=32 (dense-eigensolved).");
            std::println("  --machine      amd-3960x (puffin, default and only preset)");
            std::exit(0);
        }
        else throw std::invalid_argument("Unknown flag: " + std::string(arg));
    }
    return a;
}

/// `gating` separates the two kinds of entry: a gating check validates the instrument (if
/// it fails, nothing downstream is trustworthy), while a non-gating check reports a finding
/// about the operator under study and must never hold the gate hostage.
struct Check {
    std::string name;
    std::string prediction;   ///< the mechanism claim, stated before the number
    bool        passed = false;
    bool        gating = true;
    std::string detail;
};

inline std::vector<Check> g_checks;

inline void record(std::string name, std::string prediction, bool passed, std::string detail,
            bool gating = true)
{
    g_checks.push_back({std::move(name), std::move(prediction), passed, gating,
                        std::move(detail)});
}

/// One point of the basis-conditioning curve: the a-priori prediction beside the
/// measurement, so C4's agreement can be plotted rather than only asserted.
struct KappaRow {
    int    s;
    double kappa_measured;
    double kappa_predicted;
    bool   trusted;   ///< measured kappa below the 1e12 roundoff horizon
};

/// Apply the operator's permutation to an unpermuted vector: v'[pi(i)] = v[i].
inline Eigen::VectorXd permute(const std::vector<int64_t>& perm, const Eigen::VectorXd& v)
{
    Eigen::VectorXd out(v.size());
    for (Eigen::Index i = 0; i < v.size(); ++i)
        out(static_cast<Eigen::Index>(perm[static_cast<std::size_t>(i)])) = v(i);
    return out;
}

/// The deterministic sine vector, used wherever a single comparable-across-runs
/// starting vector is wanted (the plotted conditioning curve, the spectrum-alone
/// prediction).
inline Eigen::VectorXd sine_vector(Eigen::Index n)
{
    Eigen::VectorXd v(n);
    for (Eigen::Index i = 0; i < n; ++i)
        v(i) = std::sin(0.7 * static_cast<double>(i) + 1.0);
    return v.normalized();
}

/// Is the operator Hermitian positive-semidefinite? Only the Hochbruck-Lubich m-bound
/// (C3) needs this. Distinct from normality: a shift makes the operator indefinite but
/// still normal (kappa(X)=1), and the basis-conditioning prediction survives that.
inline bool is_hermitian_psd(const SyntheticOperator& op) noexcept
{
    if (!op.is_normal || !op.has_analytic_spectrum) return false;
    const double scale = std::max(1.0, std::abs(op.lambda_max));
    return op.lambda_min >= -1e-9 * scale;
}

// The spectral prediction, from whatever source the operator admits.
//
// The load-bearing property is normality, not separability: for any normal operator the
// monomial-basis conditioning equals the row-scaled Vandermonde kappa(M) exactly, whether
// lambda comes from a cosine formula, a dense symmetric eigensolver, or a complex
// eigensolver. Done once per run.
struct EigenData {
    bool                normal = true;   ///< real (symmetric) vs complex (non-normal) path
    Eigen::VectorXd     lam_r;           ///< real eigenvalues        (normal path)
    Eigen::VectorXd     c_r;             ///< real coefficients Q^T v  (normal path)
    Eigen::VectorXcd    lam_c;           ///< complex eigenvalues      (non-normal path)
    Eigen::VectorXcd    c_c;             ///< complex coefficients X^-1 v
    double              kappa_X = 1.0;   ///< eigenvector conditioning (1 = normal)
    std::string         source  = "analytic";
    bool                ok      = false;
    // Retained so c = X^{-1} v can be recomputed for any vector without re-decomposing,
    // which lets C6 report predicted-vs-measured across an ensemble.
    bool                analytic_c = false;  ///< c comes from spectral_coefficients(spec,.)
    Eigen::MatrixXd     Q;                   ///< orthonormal eigenvectors (dense normal path)
    Eigen::MatrixXcd    X;                   ///< complex eigenvectors     (non-normal path)
};

/// Eigenbasis coordinates c = X^{-1} v for an arbitrary vector, reusing the stored
/// decomposition. `v_unperm`/`v_perm` mirror decompose()'s two conventions.
struct Coeffs { Eigen::VectorXd r; Eigen::VectorXcd c; };

/// The standard eigenvector conditioning: kappa_2 of the eigenvectors normalised to unit
/// 2-norm (Eigen's convention). Used for every non-normal operator so the gap-vs-kappa(X)
/// law compares like with like (the analytic r^(d(n1-1)) formula is a different,
/// un-normalised scaling). Returns NaN if the eigensolver fails.
inline double unit_norm_kappa_X(const SyntheticOperator& op)
{
    if (op.is_normal) return 1.0;
    Eigen::EigenSolver<Eigen::MatrixXd> es(Eigen::MatrixXd(op.A), /*eigenvectors=*/true);
    if (es.info() != Eigen::Success) return std::numeric_limits<double>::quiet_NaN();
    return condition_number(es.eigenvectors());
}

inline EigenData decompose(const SyntheticOperator& op, const SyntheticSpec& spec,
                    const Eigen::VectorXd& v_unperm, const Eigen::VectorXd& v_perm)
{
    EigenData e;
    if (op.has_analytic_spectrum) {
        // Closed-form spectrum: analytic lambda and c, but kappa(X) is the unit-2-norm
        // value (consistent with the dense paths), not the un-normalised eigvec_condition.
        e.normal  = true;
        e.lam_r   = op.lambda;
        e.c_r     = spectral_coefficients(spec, v_unperm);
        e.kappa_X = op.is_normal ? 1.0 : unit_norm_kappa_X(op);
        e.source  = op.is_normal ? "analytic (closed form)"
                                 : "analytic prediction, unit-2-norm kappa(X)";
        e.analytic_c = true;
        e.ok      = true;
        return e;
    }
    // No closed form: dense-eigensolve the operator under test. The spectrum still predicts.
    const Eigen::MatrixXd A(op.A);
    if (op.is_normal) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);
        if (es.info() != Eigen::Success) return e;
        e.normal  = true;
        e.lam_r   = es.eigenvalues();
        e.Q       = es.eigenvectors();
        e.c_r     = e.Q.transpose() * v_perm;                 // Q^T v (Q orthonormal)
        e.kappa_X = 1.0;
        e.source  = "dense symmetric eigensolver";
        e.ok      = true;
        return e;
    }
    // Genuinely non-normal: complex eigendecomposition, kappa(X) > 1 (unit-2-norm, the
    // standard conditioning). Near-defectiveness makes kappa(X) large but not unreliable:
    // it reproduces across independent eigensolvers (see the real_bs_operator note), a
    // faithful measure of eigenvectors the spectrum-alone prediction cannot see.
    Eigen::EigenSolver<Eigen::MatrixXd> es(A);
    if (es.info() != Eigen::Success) return e;
    e.normal  = false;
    e.lam_c   = es.eigenvalues();
    e.X       = es.eigenvectors();
    e.c_c     = e.X.colPivHouseholderQr()
                    .solve(v_perm.cast<std::complex<double>>());  // X^-1 v
    e.kappa_X = condition_number(e.X);
    e.source  = "dense general eigensolver";
    e.ok      = true;
    return e;
}

/// Predicted kappa of the row-scaled Vandermonde diag(c) V(lambda) at power s, from a
/// decomposition. Real for a normal operator, complex for a non-normal one.
inline double predicted_kappa_c(const EigenData& e, const Coeffs& cf, int s)
{
    if (!e.ok) return std::numeric_limits<double>::quiet_NaN();
    if (e.normal)
        return predicted_basis_condition(e.lam_r, cf.r, s);
    Eigen::MatrixXcd M(e.lam_c.size(), static_cast<Eigen::Index>(s) + 1);
    M.col(0) = cf.c;
    for (int k = 1; k <= s; ++k) M.col(k) = M.col(k - 1).cwiseProduct(e.lam_c);
    return condition_number(M);
}

inline double predicted_kappa(const EigenData& e, int s)
{
    return predicted_kappa_c(e, Coeffs{e.c_r, e.c_c}, s);
}

/// Coordinates for an arbitrary vector, reusing the stored decomposition. Analytic
/// operators need the spec (the DST path); dense ones reuse Q or X directly.
inline Coeffs coeffs_for(const EigenData& e, const SyntheticSpec& spec,
                  const Eigen::VectorXd& v_unperm, const Eigen::VectorXd& v_perm)
{
    Coeffs cf;
    if (e.analytic_c)      cf.r = spectral_coefficients(spec, v_unperm);
    else if (e.normal)     cf.r = e.Q.transpose() * v_perm;
    else                   cf.c = e.X.colPivHouseholderQr()
                                     .solve(v_perm.cast<std::complex<double>>());
    return cf;
}

/// Gershgorin upper bound on the spectral radius (max absolute row sum). Cheap, needs
/// no spectrum, valid for non-normal operators; used to pick a stable smoothing step.
inline double gershgorin_radius(const SpMatS& A)
{
    Eigen::VectorXd rowsum = Eigen::VectorXd::Zero(A.rows());
    for (int k = 0; k < A.outerSize(); ++k)
        for (SpMatS::InnerIterator it(A, k); it; ++it)
            rowsum(it.row()) += std::abs(it.value());
    return rowsum.maxCoeff();
}

/// A heat-smoothed random vector: forward-Euler diffusion on the symmetric part damps the
/// high-frequency modes, as the diffusive semigroup e^{-tau A} does to a KSM-EI solution.
/// The physically-grounded starting-vector model: the real solver presents smooth vectors
/// concentrated on small eigenvalues, not dominant-mode-heavy ones.
inline Eigen::VectorXd heat_smoothed(const SpMatS& A, Eigen::VectorXd v, int steps, double radius)
{
    const double dt = 0.5 / std::max(radius, 1e-30);   // stable: damps lam_max by ~0.5/step
    for (int k = 0; k < steps; ++k) {
        const Eigen::VectorXd Asv = 0.5 * (A * v + A.transpose() * v);  // symmetric part
        v -= dt * Asv;
    }
    const double nrm = v.norm();
    return (nrm > 0.0) ? Eigen::VectorXd(v / nrm) : v;
}

/**
 * @brief The physical starting-vector ensemble: vectors the diffusive solver produces.
 *
 * A diffusive KSM-EI solution smooths as it evolves (exp(-hA) damps large-lambda modes), so
 * its content concentrates on the small eigenvalues. Certifying against a dominant-mode
 * adversary is pessimism about a vector the solver runs away from. The ensemble is the
 * deterministic sine vector plus random vectors heat-smoothed to a range of smoothness, the
 * dominant-mode vector is reported separately as a stress case (stress_vector).
 */
inline std::vector<Eigen::VectorXd> physical_ensemble(const SyntheticOperator& op,
                                              const Eigen::VectorXd& base)
{
    const Eigen::Index n = op.n;
    const double radius = gershgorin_radius(op.A);
    std::vector<Eigen::VectorXd> vs;
    vs.push_back(base);   // the operator's actual starting vector (payoff u0 for real BS)

    std::mt19937_64 rng(0xC0FFEE);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (int steps : {2, 8, 32}) {
        Eigen::VectorXd w(n);
        for (Eigen::Index i = 0; i < n; ++i) w(i) = nd(rng);
        vs.push_back(heat_smoothed(op.A, w, steps, radius));
    }
    return vs;
}

/// The dominant-mode stress vector: power iteration toward the largest-|lambda| eigenvector.
/// Reported, never part of the certificate (its adversariality is an arbitrary dial).
inline Eigen::VectorXd stress_vector(const SyntheticOperator& op, int bias_power = 8)
{
    const Eigen::Index n = op.n;
    std::mt19937_64 rng(0xBADF00D);
    std::normal_distribution<double> nd(0.0, 1.0);
    Eigen::VectorXd w(n);
    for (Eigen::Index i = 0; i < n; ++i) w(i) = nd(rng);
    for (int it = 0; it < bias_power; ++it) {
        w = op.A * w;
        const double nrm = w.norm();
        if (nrm > 0.0) w /= nrm;
    }
    return w.normalized();
}

// A per-vector certificate row: demand m(v), supply s_max(v), and the blocks a stable
// s-step cycle then needs. A block of b basis vectors is certified iff b-1 <= s_max, so
// blocks = ceil(m/(s_max+1)).
// Cross-platform: a raw s_max can flip by +/-1 between toolchains (kappa(B_s) crosses the
// fixed u^(-1/2) line and roundoff moves the integer crossing), but `blocks` is invariant.
// Report and act on blocks, not raw s_max.
struct VectorCert {
    std::string label;
    int m      = 0;   ///< demand: Krylov dimension for this vector
    int s_max  = 0;   ///< supply (largest certified power); roundoff-sensitive at crossings
    int blocks = 0;   ///< ceil(m / (s_max+1)); platform-invariant, the certified quantity
};

inline VectorCert certify_vector(const SyntheticOperator& op, const Args& a,
                          const Eigen::VectorXd& v, std::string label)
{
    const int s_ceil = std::min(a.s_ceiling, static_cast<int>(op.n) - 1);
    const int m_ceil = std::min(a.m_ceiling, static_cast<int>(op.n));
    const Eigen::VectorXd vp = permute(op.perm, v);
    VectorCert r;
    r.label  = std::move(label);
    r.m      = measured_krylov_dim(op.A, vp, -a.h, a.tol, m_ceil);
    r.s_max  = measured_s_max(op.A, vp, s_ceil);
    r.blocks = (r.m + r.s_max) / (r.s_max + 1);   // ceil(m / (s_max+1))
    return r;
}

/// Departure from symmetry of D^{-1} A D, after the known advection diagonal. Near zero
/// means the non-normality was diagonally removable (an artefact of a fixed diagonal).
inline double departure_after_symmetrizer(const SyntheticOperator& op, const Eigen::VectorXd& d)
{
    if (d.size() != op.n) return std::numeric_limits<double>::quiet_NaN();
    const Eigen::MatrixXd A(op.A);
    const Eigen::MatrixXd B = d.cwiseInverse().asDiagonal() * A * d.asDiagonal();
    return (B - B.transpose()).norm();
}

/// Largest s for which the conservative certificate holds under non-normality:
/// kappa_pred(s) * kappa(X) <= u^(-1/2), from the two-sided bound kappa(B) <= kappa(X) kappa(M).
/// The gap to the measured s_max is how much the pessimism costs.
inline int conservative_s_max(const EigenData& eig, int ceiling)
{
    const double limit = cholqr_kappa_limit();
    int s_max = 0;
    for (int s = 1; s <= ceiling; ++s) {
        const double kp = predicted_kappa(eig, s);
        if (!(kp * eig.kappa_X < limit)) break;
        s_max = s;
    }
    return s_max;
}

/// C6's per-operator report: populated by check_nonnormality, read by main for the CSV row.
struct NonNormalReport {
    double kappa_X      = 1.0;   ///< unit-2-norm eigenvector conditioning
    double henrici_rel  = 0.0;   ///< relative departure ||[A,A^T]||_F / ||A||_F^2 (units-free)
    double gap_dec      = 0.0;
    int    s_pred_int   = 0;     ///< spectral-predicted s_max, same vector
    int    s_meas_same  = 0;     ///< measured s_max,           same vector
    int    s_cons       = 0;     ///< conservative certificate
    int    diff_worst   = 0;     ///< worst |pred-meas| over the vector ensemble
};

}  // namespace regime_control
