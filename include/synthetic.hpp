/**
 * @file synthetic.hpp
 * @brief The synthetic operator A': a constructed matrix whose working set, DRAM traffic,
 *        and spectrum are three independent knobs.
 *
 * The physical Black-Scholes operator welds these together (refining the grid lifts the
 * working set, fattens the compute, and shifts the spectrum at once), so `n` traverses the
 * (R_v, R_h) plane along an anti-diagonal. A' cuts those wires:
 *
 *   Knob 1  N = n1^dim          working set, flop count     (R_v numerator)
 *   Knob 2  scatter_block b     DRAM traffic at fixed rest   (pure-axis motion)
 *   Knob 3  (scale, shift)      spectrum                     (m, basis conditioning)
 *
 * Knob 2 is exact: scattering is a symmetric permutation A' = P A P^T, so the spectrum,
 * flop count, working set, and Krylov conditioning are all preserved, and only the SpMV's
 * gather window widens. That is the pure R_v line: it moves DRAM traffic and nothing else.
 *
 * A' characterizes the method only. Prices come solely from the real operator in
 * pde_operators.hpp.
 *
 * @author Kevin Knights
 * @date 2026-07-10
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <numeric>
#include <random>
#include <stdexcept>
#include <stdint.h>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "machine.hpp"

using SpMatS = Eigen::SparseMatrix<double>;

/// Sparsity pattern, as a named reading of the scatter_block dial.
enum class Pattern {
    BANDED,     ///< b = 1: stencil-contiguous columns, cache-friendly reuse
    SCATTERED,  ///< b = N: columns land anywhere, worst-case gather
};

struct SyntheticSpec {
    int      n1           = 64;   ///< points per dimension of the Laplacian scaffold
    int      dim          = 3;    ///< 1|2|3 gives 3|5|7 nonzeros per interior row
    int64_t  scatter_block = 1;   ///< b: gather window. 1 = banded, >= N = full scatter
    double   spec_scale   = 1.0;  ///< sigma: scales lambda to sigma * lambda. Widens/compresses.
    double   spec_shift   = 0.0;  ///< mu: shifts lambda to lambda - mu. Translates.
    // advection (gamma): asymmetric off-diagonals (-1+gamma) upper, (-1-gamma) lower.
    // Diagonally removable non-normality: A = D Atilde D^{-1} with Atilde symmetric, so the
    // monomial basis is the symmetric basis reweighted by a fixed diagonal. Real spectrum,
    // known kappa(X); the baseline the correlation knob is measured against. Requires |gamma| < 1.
    double   advection    = 0.0;
    // correlation (rho): the genuine non-normality knob. A mixed second-derivative cross
    // term coupling axes 0 and 1 (needs dim >= 2), the analogue of the basket operator's
    // rho*sigma_i*sigma_j term. Destroys the Kronecker sum: no analytic spectrum, so s_max
    // must be measured. The cross term is symmetric, so rho alone is non-separable but
    // normal; rho and advection together are non-removably non-normal.
    double   correlation  = 0.0;
    // var_advection: a second, independent route to non-normality. The advection ramps
    // linearly along axis 0: gamma_eff(i0) = advection + var_advection*i0/(n1-1). A varying
    // coefficient is not Toeplitz, so no single diagonal symmetrizes it. Tests whether the
    // gap-vs-kappa(X) law is a property of eigenvector conditioning or an artifact of the
    // constant-coefficient family.
    double   var_advection = 0.0;
    uint64_t seed          = 42;
};

/// The generated operator, together with everything about it known a priori.
struct SyntheticOperator {
    SpMatS               A;        ///< the operator, permuted and spectrum-adjusted
    Eigen::VectorXd      lambda;   ///< analytic eigenvalues, tensor order (not sorted)
    std::vector<int64_t> perm;     ///< the applied permutation pi: old index to new
    int64_t              n   = 0;
    int64_t              nnz = 0;
    double lambda_min = 0.0;
    double lambda_max = 0.0;
    // Non-normality diagnostics:
    //   eigvec_condition: kappa_2 of the eigenvector matrix X. Analytic (r^(d(n1-1))) only
    //                     when the spectrum is separable (correlation == 0), else NaN.
    //   henrici:          ||A^T A - A A^T||_F, measured from the assembled matrix. 0 iff
    //                     A is normal.
    //   has_analytic_spectrum: false once correlation != 0; `lambda` is then invalid.
    double   eigvec_condition = 1.0;
    double   henrici          = 0.0;
    bool     is_normal        = true;   ///< measured: Henrici departure ~ 0
    bool     has_analytic_spectrum = true;  ///< closed-form spectrum (no correlation/var-advection)
};

/// Henrici's departure from normality, ||A^T A - A A^T||_F. Zero iff A is normal;
/// measured directly from the operator.
[[nodiscard]] inline double henrici_departure(const SpMatS& A)
{
    const SpMatS At = SpMatS(A.transpose());
    const SpMatS comm = SpMatS(At * A) - SpMatS(A * At);
    return comm.norm();  // Frobenius for a sparse matrix
}

/// N = n1^dim.
[[nodiscard]] inline int64_t synthetic_dimension(const SyntheticSpec& sp) noexcept
{
    int64_t n = 1;
    for (int d = 0; d < sp.dim; ++d) n *= sp.n1;
    return n;
}

/**
 * @brief Analytic eigenvalues of the 1D convection-diffusion Toeplitz.
 *
 * tridiag(-1-gamma, 2, -1+gamma) is similar to a symmetric matrix by D = diag(r^j),
 * r = sqrt((1+gamma)/(1-gamma)), so its eigenvalues are analytic even when nonsymmetric:
 *
 *     lambda_k = 2 - 2 sqrt(1 - gamma^2) cos(k pi / (n1+1)),   k = 1..n1
 *
 * Real for |gamma| < 1, compressing toward gamma = 1, the non-normality lives entirely in
 * the eigenvectors (condition r^(n1-1)). This real-spectrum / ill-conditioned-eigenvectors
 * separation is the cell-Peclet regime, where the spectrum-alone prediction is known to
 * fail. gamma = 0 recovers the Dirichlet Laplacian.
 */
[[nodiscard]] inline Eigen::VectorXd convection_diffusion_1d_eigenvalues(int n1, double gamma)
{
    const double compress = std::sqrt(std::max(0.0, 1.0 - gamma * gamma));
    Eigen::VectorXd lam(n1);
    for (int k = 0; k < n1; ++k) {
        const double theta = static_cast<double>(k + 1) * std::numbers::pi_v<double>
                           / static_cast<double>(n1 + 1);
        lam(k) = 2.0 - 2.0 * compress * std::cos(theta);
    }
    return lam;
}

/// Analytic eigenvalues of the 1D Dirichlet Laplacian tridiag(-1, 2, -1): the
/// gamma = 0 special case, lambda_k = 2 - 2 cos(k pi / (n1+1)).
[[nodiscard]] inline Eigen::VectorXd laplacian_1d_eigenvalues(int n1)
{
    return convection_diffusion_1d_eigenvalues(n1, 0.0);
}

/**
 * @brief kappa_2 of the eigenvector matrix X: the operator's departure from normality.
 *
 * X = D S with D = diag(r^j), r = sqrt((1+gamma)/(1-gamma)) and S the orthogonal sine
 * matrix, so kappa(X_1) = r^(n1-1) and kappa(X) = r^(dim (n1-1)) for the Kronecker sum.
 * Analytic, no eigensolver. For the monomial basis B = X M, kappa(B) can exceed kappa(M)
 * by up to this factor: 1 for a normal operator, orders of magnitude for a non-normal one.
 */
[[nodiscard]] inline double eigenvector_condition(const SyntheticSpec& sp) noexcept
{
    if (sp.advection == 0.0) return 1.0;
    const double r = std::sqrt((1.0 + sp.advection) / (1.0 - sp.advection));
    return std::pow(r, static_cast<double>(sp.dim) * static_cast<double>(sp.n1 - 1));
}

/**
 * @brief The diagonal D that symmetrizes the pure-advection operator: D^{-1} A D = A^T.
 *
 * D = diag(r^{sum of per-axis indices}), r = sqrt((1+gamma)/(1-gamma)). This is the whole
 * content of "advection non-normality is diagonally removable": D^{-1} A D is symmetric.
 * Returns an empty vector when correlation != 0, since no such diagonal exists.
 */
[[nodiscard]] inline Eigen::VectorXd advection_symmetrizer(const SyntheticSpec& sp)
{
    if (sp.correlation != 0.0) return Eigen::VectorXd();
    const int64_t n = synthetic_dimension(sp);
    Eigen::VectorXd d(n);
    if (sp.advection == 0.0) { d.setOnes(); return d; }
    const double r = std::sqrt((1.0 + sp.advection) / (1.0 - sp.advection));
    for (int64_t gid = 0; gid < n; ++gid) {
        int64_t rest = gid, isum = 0;
        for (int dd = 0; dd < sp.dim; ++dd) { isum += rest % sp.n1; rest /= sp.n1; }
        d(static_cast<Eigen::Index>(gid)) = std::pow(r, static_cast<double>(isum));
    }
    return d;
}

/**
 * @brief Analytic eigenvalues of the dD Kronecker-sum Laplacian, after scale and shift.
 *
 * The Kronecker sum's spectrum is the sum-set of the 1D spectra, so no eigensolver is
 * called. Ordering matches the tensor layout: gid = sum_j k_j * n1^j, axis 0 fastest.
 */
[[nodiscard]] inline Eigen::VectorXd synthetic_eigenvalues(const SyntheticSpec& sp)
{
    const Eigen::VectorXd lam1 = convection_diffusion_1d_eigenvalues(sp.n1, sp.advection);
    const int64_t n = synthetic_dimension(sp);

    Eigen::VectorXd lam(n);
    for (int64_t gid = 0; gid < n; ++gid) {
        int64_t rest = gid;
        double  sum  = 0.0;
        for (int d = 0; d < sp.dim; ++d) {
            const int64_t k = rest % sp.n1;
            rest /= sp.n1;
            sum += lam1(static_cast<Eigen::Index>(k));
        }
        lam(static_cast<Eigen::Index>(gid)) = sp.spec_scale * (sum - sp.spec_shift);
    }
    return lam;
}

/**
 * @brief The symmetric permutation that realizes the scatter knob.
 *
 * Indices are partitioned into contiguous blocks of `b` and shuffled within each block, so
 * a stencil neighbor lands within about b positions of pi(i): b = 1 is the identity
 * (columns stay stencil-local), b = N is one global shuffle (every gather a potential miss).
 * A permutation for every b, so A' = P A P^T is similar to A at every setting: the spectrum
 * does not move as we scatter.
 */
[[nodiscard]] inline std::vector<int64_t> scatter_permutation(int64_t n, int64_t b, uint64_t seed)
{
    std::vector<int64_t> perm(static_cast<std::size_t>(n));
    std::iota(perm.begin(), perm.end(), int64_t{0});
    if (b <= 1) return perm;

    std::mt19937_64 rng(seed);
    const int64_t block = std::min(b, n);
    for (int64_t lo = 0; lo < n; lo += block) {
        const int64_t hi = std::min(n, lo + block);
        std::shuffle(perm.begin() + static_cast<std::ptrdiff_t>(lo),
                     perm.begin() + static_cast<std::ptrdiff_t>(hi), rng);
    }
    return perm;
}

/**
 * @brief Build A' = P * (scale * (T_d - shift I)) * P^T. dim = asset dimension.
 *
 * T_d is the dD Kronecker sum of the 1D convection-diffusion Toeplitz: diagonal 2*dim,
 * upper neighbor (-1+gamma), lower (-1-gamma). The permutation P is orthogonal, so A' is
 * similar to T_d at every scatter and advection setting: spectrum and eigenvector
 * conditioning are both scatter-invariant.
 */
[[nodiscard]] inline struct SyntheticOperator build_synthetic(const SyntheticSpec& sp)
{
    // dim is the asset dimension. Capped at 6 so N = n1^dim stays dense-eigensolvable
    // for the control; the kappa(X)-vs-d law is what needs d > 3.
    if (sp.dim < 1 || sp.dim > 6)
        throw std::invalid_argument("SyntheticSpec::dim must be in [1, 6]");
    if (sp.n1 < 2)
        throw std::invalid_argument("SyntheticSpec::n1 must be >= 2");
    if (std::abs(sp.advection) >= 1.0)
        throw std::invalid_argument("SyntheticSpec::advection must satisfy |gamma| < 1 "
                                    "(keeps the spectrum real)");
    if (sp.correlation != 0.0 && sp.dim < 2)
        throw std::invalid_argument("SyntheticSpec::correlation needs dim >= 2 "
                                    "(the cross term couples two axes)");

    const int64_t n = synthetic_dimension(sp);
    const std::vector<int64_t> perm = scatter_permutation(n, sp.scatter_block, sp.seed);

    std::vector<int64_t> stride(static_cast<std::size_t>(sp.dim));
    stride[0] = 1;
    for (int d = 1; d < sp.dim; ++d)
        stride[static_cast<std::size_t>(d)] = stride[static_cast<std::size_t>(d - 1)] * sp.n1;

    const double diag = sp.spec_scale * (2.0 * static_cast<double>(sp.dim) - sp.spec_shift);
    const double off_up  = sp.spec_scale * (-1.0 + sp.advection);  // super-diagonal
    const double off_lo  = sp.spec_scale * (-1.0 - sp.advection);  // sub-diagonal
    const double cross   = sp.spec_scale * sp.correlation * 0.25;  // mixed 2nd-deriv weight

    std::vector<Eigen::Triplet<double>> trs;
    trs.reserve(static_cast<std::size_t>(n)
                * static_cast<std::size_t>(2 * sp.dim + 1 + (sp.correlation != 0.0 ? 4 : 0)));

    for (int64_t gid = 0; gid < n; ++gid) {
        const int64_t r = perm[static_cast<std::size_t>(gid)];
        trs.emplace_back(static_cast<int>(r), static_cast<int>(r), diag);

        int64_t rest = gid;
        int k0 = 0, k1 = 0;  // per-axis indices of axes 0 and 1 (for the cross term)
        for (int d = 0; d < sp.dim; ++d) {
            const int64_t k = rest % sp.n1;
            rest /= sp.n1;
            if (d == 0) k0 = static_cast<int>(k);
            if (d == 1) k1 = static_cast<int>(k);
            const int64_t st = stride[static_cast<std::size_t>(d)];

            // Advection for this axis. Axis 0 optionally carries a position-dependent ramp
            // (var_advection): gamma_eff = advection + var*i0/(n1-1). Not Toeplitz, so no
            // single diagonal symmetrizes it: a second route to non-normality.
            double up = off_up, lo = off_lo;
            if (d == 0 && sp.var_advection != 0.0) {
                const double g = sp.advection + sp.var_advection
                               * static_cast<double>(k) / static_cast<double>(sp.n1 - 1);
                up = sp.spec_scale * (-1.0 + g);
                lo = sp.spec_scale * (-1.0 - g);
            }
            if (k > 0) {  // lower neighbor along this axis (column gid - st)
                const int64_t c = perm[static_cast<std::size_t>(gid - st)];
                trs.emplace_back(static_cast<int>(r), static_cast<int>(c), lo);
            }
            if (k < sp.n1 - 1) {  // upper neighbor (column gid + st)
                const int64_t c = perm[static_cast<std::size_t>(gid + st)];
                trs.emplace_back(static_cast<int>(r), static_cast<int>(c), up);
            }
        }

        // Correlation cross term: central-difference d^2/dx0 dx1 stencil coupling the four
        // diagonal neighbors in the (axis0, axis1) plane, breaking the Kronecker sum.
        //   (+1,+1): +c   (-1,-1): +c   (+1,-1): -c   (-1,+1): -c
        // Symmetric, so rho alone stays normal; rho with advection is non-removably non-normal.
        if (sp.correlation != 0.0) {
            const int64_t s0 = stride[0], s1 = stride[1];
            auto add_cross = [&](bool up0, bool up1, double w) {
                const int ka = up0 ? k0 + 1 : k0 - 1;
                const int kb = up1 ? k1 + 1 : k1 - 1;
                if (ka < 0 || ka >= sp.n1 || kb < 0 || kb >= sp.n1) return;
                const int64_t col = gid + (up0 ? s0 : -s0) + (up1 ? s1 : -s1);
                const int64_t c = perm[static_cast<std::size_t>(col)];
                trs.emplace_back(static_cast<int>(r), static_cast<int>(c), w);
            };
            add_cross(true,  true,  cross);
            add_cross(false, false, cross);
            add_cross(true,  false, -cross);
            add_cross(false, true,  -cross);
        }
    }

    SyntheticOperator op;
    op.A = SpMatS(static_cast<int>(n), static_cast<int>(n));
    op.A.setFromTriplets(trs.begin(), trs.end());
    op.A.makeCompressed();
    op.perm   = perm;
    op.n      = n;
    op.nnz    = op.A.nonZeros();
    // Non-normality is measured from the departure ||A^T A - A A^T||_F, not inferred from
    // knobs: correlation-alone is normal (departure 0) but non-separable, so the two flags
    // are independent.
    op.henrici              = henrici_departure(op.A);
    const double a_fro      = op.A.norm();
    op.is_normal            = (op.henrici <= 1e-9 * a_fro * a_fro);
    // Correlation or variable-coefficient advection destroys the closed-form spectrum.
    op.has_analytic_spectrum = (sp.correlation == 0.0 && sp.var_advection == 0.0);

    if (op.has_analytic_spectrum) {
        op.lambda = synthetic_eigenvalues(sp);
        op.lambda_min = op.lambda.minCoeff();
        op.lambda_max = op.lambda.maxCoeff();
        op.eigvec_condition = eigenvector_condition(sp);
    } else {
        // No analytic spectrum and X unknown in closed form: callers must compute the
        // spectrum densely, or measure what they were going to predict.
        op.lambda = Eigen::VectorXd();
        op.lambda_min = std::numeric_limits<double>::quiet_NaN();
        op.lambda_max = std::numeric_limits<double>::quiet_NaN();
        op.eigvec_condition = std::numeric_limits<double>::quiet_NaN();
    }
    return op;
}

/**
 * @brief Coefficients c = X^{-1} v of v in the operator's analytic eigenbasis.
 *
 * The eigenvectors are separable products of scaled 1D sine modes, X = diag(r^i) S per axis
 * with r = sqrt((1+gamma)/(1-gamma)) and S the orthogonal sine matrix, so c is the DST-I of
 * v pre-scaled by delta^{sum of indices}, delta = 1/r. No eigensolver is called.
 *
 * At gamma = 0 this is the plain DST-I. For gamma != 0 the resulting c feeds
 * predicted_basis_condition to give kappa(M), the conditioning A would have if normal, the
 * true kappa(X M) is larger by up to kappa(X).
 *
 * @param v the starting vector, in unpermuted tensor order.
 */
[[nodiscard]] inline Eigen::VectorXd spectral_coefficients(const SyntheticSpec& sp,
                                                           const Eigen::VectorXd& v)
{
    const int64_t n  = synthetic_dimension(sp);
    const int     n1 = sp.n1;
    if (v.size() != static_cast<Eigen::Index>(n))
        throw std::invalid_argument("spectral_coefficients: v has wrong length");

    // 1D DST-I matrix, applied along each axis (separable, so d passes of O(N*n1)).
    Eigen::MatrixXd S(n1, n1);
    const double scale = std::sqrt(2.0 / static_cast<double>(n1 + 1));
    for (int k = 0; k < n1; ++k)
        for (int i = 0; i < n1; ++i)
            S(k, i) = scale * std::sin(static_cast<double>((k + 1) * (i + 1))
                                       * std::numbers::pi_v<double>
                                       / static_cast<double>(n1 + 1));

    Eigen::VectorXd c = v;

    // Non-normal case: pre-scale by delta^{sum of per-axis indices} so that the
    // separable sine transform below computes X^{-1} v rather than S^T v.
    if (sp.advection != 0.0) {
        const double delta = std::sqrt((1.0 - sp.advection) / (1.0 + sp.advection));
        for (int64_t gid = 0; gid < n; ++gid) {
            int64_t rest = gid, isum = 0;
            for (int d = 0; d < sp.dim; ++d) { isum += rest % n1; rest /= n1; }
            c(static_cast<Eigen::Index>(gid)) *= std::pow(delta, static_cast<double>(isum));
        }
    }

    int64_t stride = 1;
    for (int d = 0; d < sp.dim; ++d) {
        Eigen::VectorXd out(n);
        const int64_t outer = n / (stride * n1);
        for (int64_t o = 0; o < outer; ++o) {
            for (int64_t s = 0; s < stride; ++s) {
                const int64_t base = o * stride * n1 + s;
                Eigen::VectorXd line(n1);
                for (int i = 0; i < n1; ++i)
                    line(i) = c(static_cast<Eigen::Index>(base + static_cast<int64_t>(i) * stride));
                const Eigen::VectorXd tl = S * line;
                for (int k = 0; k < n1; ++k)
                    out(static_cast<Eigen::Index>(base + static_cast<int64_t>(k) * stride)) = tl(k);
            }
        }
        c = out;
        stride *= n1;
    }
    return c;
}

/**
 * @brief Modeled x-vector reuse for a given gather window, on a given machine.
 *
 * A row's gather touches about `b` entries of x. While that window fits in the cache the
 * team commands, x is reused (reuse near 1), and once it spills reuse decays toward 0. A
 * model, feeding only the arithmetic intensity.
 */
[[nodiscard]] inline double modelled_x_reuse(const Machine& mc, int P, int64_t b, int64_t n)
{
    const double window_bytes = static_cast<double>(std::min(std::max(b, int64_t{1}), n)) * 8.0;
    const double cache_bytes  = static_cast<double>(aggregate_llc_bytes(mc, P));
    return std::min(1.0, cache_bytes / window_bytes);
}

/// Convenience: the pattern a given scatter_block reads as.
[[nodiscard]] inline Pattern pattern_of(const SyntheticSpec& sp) noexcept
{
    return sp.scatter_block <= 1 ? Pattern::BANDED : Pattern::SCATTERED;
}

[[nodiscard]] inline const char* pattern_name(Pattern p) noexcept
{
    return p == Pattern::BANDED ? "banded" : "scattered";
}
