/**
 * @file ca_arnoldi.hpp
 * @brief The two orthogonalization arms of the regime study, and the matrix-powers
 *        kernel that separates them.
 *
 * Baseline: Modified Gram-Schmidt Arnoldi, 1 + m(m+3)/2 global reductions per cycle.
 * Each dot product depends on the previous subtraction, so MGS cannot be made
 * communication-avoiding.
 *
 * CA arm: matrix-powers kernel + Block Gram-Schmidt + CholeskyQR. The basis
 * [v, Av, ..., A^s v] is built with zero reductions and orthogonalized with one Gram
 * all-reduce per block. The price is orthogonality: CholeskyQR squares the condition
 * number, so a stable arm uses CholeskyQR2 (a second Gram all-reduce), giving
 * 1 + 2*ceil(m/s) reductions, valid inside the certificate kappa(B) <= u^(-1/2).
 *
 * These are serial, Eigen-dense numerical kernels.
 * The OpenMP timing path lives in src/scaling.cpp.
 *
 * @author Kevin Knights
 * @date 2026-07-10
 */
#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/MatrixFunctions>

#include "regime.hpp"

/// Loss of orthogonality of a computed basis Q: ||I - Q^T Q||_F.
[[nodiscard]] inline double orthogonality_loss(const Eigen::MatrixXd& Q)
{
    const Eigen::Index k = Q.cols();
    return (Eigen::MatrixXd::Identity(k, k) - Q.transpose() * Q).norm();
}

// Matrix-powers kernel
/**
 * @brief The monomial Krylov basis B = [v, Av, A^2 v, ..., A^s v], s+1 columns.
 *
 * Zero global reductions, which is what the s-step formulation buys. The columns are
 * not orthogonal and kappa(B) grows geometrically in s, bounding s from above. Newton
 * or Chebyshev bases are the standard remedy when kappa(B) becomes the binding constraint.
 */
[[nodiscard]] inline Eigen::MatrixXd matrix_powers(const SpMatS& A,
                                                   const Eigen::VectorXd& v,
                                                   int s)
{
    if (s < 0) throw std::invalid_argument("matrix_powers: s must be >= 0");
    Eigen::MatrixXd B(A.rows(), static_cast<Eigen::Index>(s) + 1);
    B.col(0) = v;
    for (int k = 1; k <= s; ++k)
        B.col(k).noalias() = A * B.col(k - 1);
    return B;
}

/**
 * @brief Largest s whose monomial basis condition stays inside the CholQR certificate
 *        kappa <= u^(-1/2).
 *
 * Measured, so valid for non-normal and non-separable operators where the spectrum-alone
 * predicted_s_max has no meaning. A block of b basis vectors is certified iff b-1 <= s_max,
 * so the largest certified block width is s_max + 1 and a cycle needs ceil(m/(s_max+1)) blocks.
 */
[[nodiscard]] inline int measured_s_max(const SpMatS& A, const Eigen::VectorXd& v, int ceiling)
{
    const double limit = cholqr_kappa_limit();
    int s_max = 0;
    for (int s = 1; s <= ceiling; ++s) {
        if (condition_number(matrix_powers(A, v, s)) >= limit) break;
        s_max = s;
    }
    return s_max;
}

// CholeskyQR: one reduction, local factorization
struct CholQrResult {
    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;
    bool   llt_failed = false;  ///< Gram matrix lost numerical positive-definiteness
    double kappa      = 0.0;    ///< kappa_2(B), measured on the input block
};

/**
 * @brief Q R = B via the Gram matrix G = B^T B, formed in one all-reduce.
 *
 * The Cholesky factor and triangular solve are local; that single reduction, in place
 * of MGS's j+1 serialized ones, is the horizontal mechanism.
 *
 * Forming G squares the condition number, so the computed Q loses orthogonality like
 * O(u * kappa(B)^2) long before Cholesky hits a non-positive pivot: llt_failed is a
 * late, crude symptom, not the usable limit. The usable limit is the certificate
 * kappa(B) <= u^(-1/2) (Yamamoto, Nakatsukasa, Yanagisawa & Fukaya, 2015), under which
 * cholesky_qr2 recovers O(u).
 */
[[nodiscard]] inline CholQrResult cholesky_qr(const Eigen::MatrixXd& B)
{
    CholQrResult res;
    res.kappa = condition_number(B);

    const Eigen::MatrixXd G = B.transpose() * B;     // the single reduction
    const Eigen::LLT<Eigen::MatrixXd> llt(G);
    if (llt.info() != Eigen::Success) {
        res.llt_failed = true;
        return res;
    }
    res.R = llt.matrixU();
    // Solve R^T X = B^T for X = R^-T B^T, so Q = X^T = B R^-1. Materialise X: a
    // triangular solve expression must not be transposed lazily.
    const Eigen::MatrixXd X =
        res.R.transpose().triangularView<Eigen::Lower>().solve(B.transpose());
    res.Q = X.transpose();
    return res;
}

/**
 * @brief CholeskyQR2: run CholeskyQR twice. Two reductions, O(u) orthogonality.
 *
 * The first pass leaves kappa(Q1) ~ 1, so the second drives the error to O(u). Valid
 * inside the certificate kappa(B) <= u^(-1/2). Costs one extra Gram all-reduce per block,
 * which is what ca_reductions() charges when `reorth` is set.
 */
[[nodiscard]] inline CholQrResult cholesky_qr2(const Eigen::MatrixXd& B)
{
    CholQrResult first = cholesky_qr(B);
    if (first.llt_failed) return first;

    CholQrResult second = cholesky_qr(first.Q);
    if (second.llt_failed) {
        second.kappa = first.kappa;
        return second;
    }
    second.R     = second.R * first.R;   // B = Q2 (R2 R1)
    second.kappa = first.kappa;          // report the condition of the INPUT block
    return second;
}

// Baseline: Modified Gram-Schmidt Arnoldi
struct ArnoldiResult {
    Eigen::MatrixXd V;              ///< n x (m+1) basis (only m_used+1 columns valid)
    Eigen::MatrixXd H;              ///< (m+1) x m Hessenberg
    int     m_used     = 0;         ///< steps actually taken (breakdown may cut short)
    int64_t reductions = 0;         ///< global reductions this cycle would cost
    double  beta       = 0.0;       ///< ||v0||
    bool    broke_down = false;
};

/**
 * @brief m-step Arnoldi with Modified Gram-Schmidt: the communication-bound baseline.
 *
 * Mirrors the loop in src/scaling.cpp, so the reduction count is mgs_reductions(m_used).
 */
[[nodiscard]] inline ArnoldiResult mgs_arnoldi(const SpMatS& A, const Eigen::VectorXd& v0, int m,
                                               double breakdown_tol = 1e-14)
{
    const Eigen::Index n = A.rows();
    if (m <= 0 || static_cast<Eigen::Index>(m) > n)
        throw std::invalid_argument("mgs_arnoldi: require 0 < m <= n");

    ArnoldiResult r;
    r.V = Eigen::MatrixXd::Zero(n, m + 1);
    r.H = Eigen::MatrixXd::Zero(m + 1, m);
    r.beta = v0.norm();                              // reduction 1
    r.V.col(0) = v0 / r.beta;

    int j = 0;
    for (; j < m; ++j) {
        Eigen::VectorXd w = A * r.V.col(j);

        // MGS: each dot sees w already updated by the previous subtraction.
        for (int i = 0; i <= j; ++i) {
            r.H(i, j) = w.dot(r.V.col(i));           // reduction
            w -= r.H(i, j) * r.V.col(i);
        }
        r.H(j + 1, j) = w.norm();                    // reduction
        if (r.H(j + 1, j) <= breakdown_tol) {
            r.broke_down = true;
            ++j;
            break;
        }
        r.V.col(j + 1) = w / r.H(j + 1, j);
    }
    r.m_used     = j;
    r.reductions = mgs_reductions(r.m_used);
    return r;
}

// CA arm: s-step block Arnoldi (matrix-powers + BGS + CholeskyQR)
struct CaArnoldiResult {
    Eigen::MatrixXd V;               ///< n x m orthonormal basis of K_m(A, v0)
    Eigen::MatrixXd H;               ///< m x m projected operator V^T A V
    int     m_used      = 0;
    int64_t reductions  = 0;
    double  beta        = 0.0;
    double  max_kappa   = 0.0;       ///< worst monomial-block condition seen
    double  ortho_loss  = 0.0;       ///< ||I - V^T V||_F, the measured outcome
    bool    broke_down  = false;     ///< a CholeskyQR block failed
    int     broke_at_s  = 0;         ///< block size at which it failed
};

/**
 * @brief s-step Arnoldi: builds K_m(A, v0) in ceil(m/s) blocks, one reduction each.
 *
 * Per block: matrix-powers builds s vectors with no communication; Block Gram-Schmidt
 * projects off the existing basis; CholeskyQR orthonormalises with one Gram reduction.
 * H is recovered as V^T A V (correct but not communication-optimal), which suits the
 * numerical control this kernel serves.
 *
 * @param reorth  a second BGS pass and CholeskyQR2: one extra reduction per block,
 *                restoring orthogonality to O(u).
 */
[[nodiscard]] inline CaArnoldiResult ca_arnoldi(const SpMatS& A, const Eigen::VectorXd& v0,
                                                int m, int s, bool reorth = false)
{
    if (m <= 0 || s <= 0) throw std::invalid_argument("ca_arnoldi: require m > 0 and s > 0");

    const Eigen::Index n = A.rows();
    CaArnoldiResult r;
    r.beta = v0.norm();

    Eigen::MatrixXd V(n, m);
    Eigen::Index filled = 0;
    Eigen::VectorXd start = v0 / r.beta;

    while (filled < static_cast<Eigen::Index>(m)) {
        const int blk = static_cast<int>(std::min<Eigen::Index>(s, static_cast<Eigen::Index>(m) - filled));

        // No reductions here: the communication the s-step formulation avoids.
        Eigen::MatrixXd B = matrix_powers(A, start, blk - 1);

        // Block Gram-Schmidt against the existing basis (one reduction), optionally twice.
        if (filled > 0) {
            const Eigen::MatrixXd Vp = V.leftCols(filled);
            B.noalias() -= Vp * (Vp.transpose() * B);
            if (reorth) B.noalias() -= Vp * (Vp.transpose() * B);
        }

        // Intra-block orthogonality is set by CholeskyQR, not by the BGS pass above.
        const CholQrResult qr = reorth ? cholesky_qr2(B) : cholesky_qr(B);
        r.max_kappa = std::max(r.max_kappa, qr.kappa);
        if (qr.llt_failed) {
            r.broke_down = true;
            r.broke_at_s = blk;
            break;
        }

        V.middleCols(filled, blk) = qr.Q;
        filled += blk;
        if (filled < static_cast<Eigen::Index>(m))
            start = A * V.col(filled - 1);
    }

    r.m_used = static_cast<int>(filled);
    if (r.m_used > 0) {
        r.V = V.leftCols(r.m_used);
        r.H = r.V.transpose() * (A * r.V);
        r.ortho_loss = orthogonality_loss(r.V);
    }
    r.reductions = ca_reductions(r.m_used, s, reorth);
    return r;
}

// Measured Krylov dimension
/**
 * @brief Smallest m at which the Arnoldi approximation to exp(h A) v0 meets `tol`.
 *
 * Grows m and watches the a-posteriori residual beta * h_{m+1,m} * |e_m^T exp(h H_m) e_1|.
 * The same absolute residual solve_ksm_ei (solvers.hpp) converges on, so this m matches
 * the application's and is the one predict_krylov_dim() must reproduce.
 *
 * @note Not shift-invariant: under a shift A - mu I, the estimate picks up a factor
 *       e^{h mu}, so a large h*mu can shift m by a step. This mirrors the solver's
 *       absolute stopping test; the Krylov subspace itself is shift-invariant.
 */
[[nodiscard]] inline int measured_krylov_dim(const SpMatS& A, const Eigen::VectorXd& v0,
                                             double h, double tol, int m_ceiling = 512)
{
    const ArnoldiResult r = mgs_arnoldi(A, v0, std::min<int>(m_ceiling, static_cast<int>(A.rows())));

    for (int m = 1; m <= r.m_used; ++m) {
        const Eigen::MatrixXd Hm = r.H.topLeftCorner(m, m);
        const Eigen::MatrixXd E  = (h * Hm).exp();
        const double resid = r.beta * r.H(m, m - 1) * std::abs(E(m - 1, 0));
        if (resid <= tol) return m;
    }
    return r.m_used;
}
