/**
 * @file regime.hpp
 * @brief The dimensionless coordinates (R_v, R_h), their a-priori inputs, and the
 *        predicted boundaries the regime map exists to test.
 *
 * Strict separation: predictors (everything here, computed before a run
 * from N, pattern, spectrum, and the Machine) define a point's coordinates.
 * Outcomes (measured wall-clock, achieved bytes, orthogonality loss) test the boundaries and never
 * enter this header. The rule that keeps the map non-circular: R_h's numerator is a machine
 * constant times a tree depth, never a measured fraction of runtime.
 *
 * @author Kevin Knights
 * @date 2026-07-10
 */
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "machine.hpp"

/// The operator type shared by the synthetic generator and the Arnoldi kernels.
/// Identical to pde_operators.hpp's SpMat; named separately so the regime headers
/// stay independent of the Black-Scholes assembly.
using SpMatS = Eigen::SparseMatrix<double>;

// Byte models
// These reproduce src/scaling.cpp's accounting so the regime study and the
// existing strong/weak-scaling anchor remain directly comparable:
//   CSR value 8 B + column index 4 B = 12 B per nonzero; row pointer 4 B per row.

/// Bytes of the CSR operator itself (values + column indices + row pointers).
[[nodiscard]] inline int64_t matrix_bytes(int64_t nnz, int64_t n) noexcept
{
    return nnz * 12 + (n + 1) * 4;
}

/**
 * @brief Total working set resident across one Arnoldi cycle: the numerator of R_v.
 *
 * The operator, the (m+1)-column basis V, and the work vector w. Unlike scaling.cpp's
 * `spmv_ws` this does not separately charge x and y (x IS a column of V, y IS w); the two
 * models agree to within 8 B/row.
 */
[[nodiscard]] inline int64_t arnoldi_working_set_bytes(int64_t nnz, int64_t n, int m) noexcept
{
    const int64_t basis = (static_cast<int64_t>(m) + 1) * n * 8;  // V
    const int64_t work  = n * 8;                                  // w
    return matrix_bytes(nnz, n) + basis + work;
}

/**
 * @brief Modeled DRAM bytes moved by ONE SpMV.
 *
 * The operator streams once and y is written once. The pattern knob controls only the
 * gather x[col_idx]: x_reuse = 1 (banded) costs the compulsory n*8 read, x_reuse = 0
 * (scattered) makes every gather a potential miss at nnz*8. This moves bytes-from-DRAM at
 * fixed flops, working set, and spectrum (spec section 4).
 *
 * @note Modeled, not counted: AMD uncore DRAM counters need root on puffin, so spec section
 *       10's fallback applies (modeled bytes reconciled against the roofline).
 */
[[nodiscard]] inline double spmv_dram_bytes(int64_t nnz, int64_t n, double x_reuse) noexcept
{
    const double stream = static_cast<double>(matrix_bytes(nnz, n));
    const double y_out  = static_cast<double>(n) * 8.0;
    const double x_in   = x_reuse * static_cast<double>(n) * 8.0
                        + (1.0 - x_reuse) * static_cast<double>(nnz) * 8.0;
    return stream + y_out + x_in;
}

/// Arithmetic intensity [FLOP/byte] of one SpMV under the modeled byte traffic.
[[nodiscard]] inline double spmv_intensity(int64_t nnz, int64_t n, double x_reuse) noexcept
{
    return (2.0 * static_cast<double>(nnz)) / spmv_dram_bytes(nnz, n, x_reuse);
}

// Reduction counts (R): the quantity the horizontal mechanism attacks
/**
 * @brief Global reductions per Arnoldi cycle under Modified Gram-Schmidt.
 *
 * One norm for V.col(0), then per j: (j+1) dot products plus one norm.
 *
 *     R_mgs = 1 + m(m+3)/2       (45 at m=8)
 *
 * The quadratic term cannot be batched: MGS subtracts h_ij*V_i from w before dotting w
 * with V_{i+1}, so each reduction depends on the previous update. That serialization is
 * MGS's orthogonality and why it cannot be made communication-avoiding. See ca_reductions().
 */
[[nodiscard]] constexpr int64_t mgs_reductions(int m) noexcept
{
    const int64_t mm = m;
    return 1 + mm * (mm + 3) / 2;
}

/**
 * @brief Global reductions per Arnoldi cycle under s-step block orthogonalization.
 *
 * Matrix-powers builds s vectors with zero reductions; each block then costs one Gram
 * all-reduce, and re-orthogonalization a second.
 *
 *     R_ca = 1 + ceil(m/s) * (reorth ? 2 : 1) (2 at m=s=8, against MGS's 45)
 */
[[nodiscard]] constexpr int64_t ca_reductions(int m, int s, bool reorth) noexcept
{
    const int64_t blocks = (static_cast<int64_t>(m) + s - 1) / s;
    return 1 + blocks * (reorth ? 2 : 1);
}

// Work per Arnoldi cycle (W): the denominator's numerator
/**
 * @brief FLOPs of the m SpMVs in one Arnoldi cycle.
 *
 * W is defined on the BASELINE, so R_h measures how a reduction compares to the compute the
 * naive method does between reductions. Defining W on the CA variant would make the
 * coordinate depend on the treatment.
 */
[[nodiscard]] inline double spmv_cycle_flops(int64_t nnz, int m) noexcept
{
    return 2.0 * static_cast<double>(nnz) * static_cast<double>(m);
}

/**
 * @brief FLOPs of the MGS orthogonalization in one Arnoldi cycle.
 *
 * 3n to normalize V.col(0); then per j: (j+1) dot-and-axpy pairs at 4n, one norm at 2n,
 * one scale at n.
 */
[[nodiscard]] inline double mgs_cycle_flops(int64_t n, int m) noexcept
{
    const double dn = static_cast<double>(n);
    const double dm = static_cast<double>(m);
    return 3.0 * dn                     // normalise V.col(0)
         + 2.0 * dn * dm * (dm + 1.0)   // sum_j 4n(j+1)
         + 3.0 * dn * dm;               // per-j norm + scale
}

/// Total FLOPs in one m-step Arnoldi cycle. Kept for W in reports; the two kernels are
/// priced SEPARATELY in arnoldi_cycle_seconds because they sit on different roofs.
[[nodiscard]] inline double arnoldi_cycle_flops(int64_t nnz, int64_t n, int m) noexcept
{
    return spmv_cycle_flops(nnz, m) + mgs_cycle_flops(n, m);
}

/**
 * @brief Bytes MGS moves in one cycle.
 *
 * The dominant term is the basis re-reads: V_i is read once for every j >= i, so m(m+1)/2
 * columns at 8n bytes each. That quadratic is why MGS is a bandwidth problem, not only a
 * synchronization one. Then per j: w read and written (16n) and V_{j+1} written (8n).
 *
 * @note Charges each column read once, assuming a thread's slice survives in private L2
 *       between the dot and the axpy. Holds at P=21/n=61 (~86 KiB against 512 KiB),
 *       under-counts at P=1: the model's weakest joint.
 */
[[nodiscard]] inline double mgs_dram_bytes(int64_t n, int m) noexcept
{
    const double dn = static_cast<double>(n);
    const double dm = static_cast<double>(m);
    const double col_reads = 8.0 * dn * (dm * (dm + 1.0) / 2.0);
    const double w_traffic = 24.0 * dn * dm;   // read w, write w, write V_{j+1}
    const double init      = 24.0 * dn;        // norm + scale of V.col(0)
    return col_reads + w_traffic + init;
}

/// Arithmetic intensity [FLOP/byte] of MGS over one cycle. About 2x SpMV's, far less than
/// the gap between the two ROOFS they may sit on (see memory_bw_gbs).
[[nodiscard]] inline double mgs_intensity(int64_t n, int m) noexcept
{
    return mgs_cycle_flops(n, m) / mgs_dram_bytes(n, m);
}

/**
 * @brief SpMV's reuse window: the operator, plus the source and destination vectors.
 *
 * Whether the OPERATOR survives from one SpMV to the next: if it does, the cycle streams it
 * from DRAM once and from cache m-1 times. This is the quantity the strong-scaling knee
 * turns on, not the Arnoldi cycle's full working set.
 */
[[nodiscard]] inline int64_t spmv_working_set_bytes(int64_t nnz, int64_t n) noexcept
{
    return matrix_bytes(nnz, n) + n * 16;   // operator + x + y
}

/// MGS's reuse window: the (m+1)-column basis it re-reads, plus the work vector.
[[nodiscard]] inline int64_t mgs_working_set_bytes(int64_t n, int m) noexcept
{
    return (static_cast<int64_t>(m) + 1) * n * 8 + n * 8;
}

/// Per-kernel timing of one Arnoldi cycle, with the roof each kernel was priced against.
struct CycleTime {
    double spmv_s       = 0.0;
    double mgs_s        = 0.0;
    double total_s      = 0.0;
    double ai_spmv      = 0.0;
    double ai_mgs       = 0.0;
    double bw_spmv_gbs  = 0.0;   ///< the roof SpMV got: L3 (scales with P) or DRAM (flat)
    double bw_mgs_gbs   = 0.0;
    bool   spmv_resident = false;
    bool   mgs_resident  = false;
};

/**
 * @brief Seconds for one m-step Arnoldi cycle, pricing each kernel on its own roof.
 *
 * Each kernel gets its own byte model, residency test, and roof, then they
 * are summed.
 *
 * @param x_reuse 1 = banded (x stays cached), 0 = fully scattered (every gather misses).
 */
[[nodiscard]] inline CycleTime arnoldi_cycle_seconds(const Machine& mc, int P, int64_t nnz,
                                                     int64_t n, int m, double x_reuse)
{
    CycleTime t;
    const int64_t ws_spmv = spmv_working_set_bytes(nnz, n);
    const int64_t ws_mgs  = mgs_working_set_bytes(n, m);

    t.spmv_resident = ws_spmv <= aggregate_llc_bytes(mc, P);
    t.mgs_resident  = ws_mgs  <= aggregate_llc_bytes(mc, P);
    t.bw_spmv_gbs   = memory_bw_gbs(mc, P, ws_spmv);
    t.bw_mgs_gbs    = memory_bw_gbs(mc, P, ws_mgs);
    t.ai_spmv       = spmv_intensity(nnz, n, x_reuse);
    t.ai_mgs        = mgs_intensity(n, m);

    t.spmv_s = spmv_cycle_flops(nnz, m)
             / attainable_flops(mc, P, t.ai_spmv, t.bw_spmv_gbs);
    t.mgs_s  = mgs_cycle_flops(n, m)
             / attainable_flops(mc, P, t.ai_mgs, t.bw_mgs_gbs);
    t.total_s = t.spmv_s + t.mgs_s;
    return t;
}

// The coordinates
/**
 * @brief Vertical coordinate: working set relative to the cache the team commands.
 *
 * R_v < 1 fits in aggregate last-level cache (no DRAM traffic to avoid, matrix-powers buys
 * nothing); R_v > 1 the operator spills. Aggregate on both sides (spec section 2), so
 * R_v ~ 1/P: P retires the vertical opportunity it creates the horizontal one.
 */
[[nodiscard]] inline double R_v(const Machine& mc, int P, int64_t working_set) noexcept
{
    return static_cast<double>(working_set) / static_cast<double>(aggregate_llc_bytes(mc, P));
}

/**
 * @brief Horizontal coordinate: one reduction's latency against the compute between two.
 *
 *     numerator   = reduction_cost_s(mc, P)              [s]  a priori, never measured
 *     denominator = arnoldi_cycle_seconds(...).total_s / R    [s]
 *
 * While compute-bound the aggregate rate grows with P, so the denominator shrinks like 1/P;
 * once a kernel spills and the DRAM roof pins, it stops. The denominator is per-kernel:
 * pricing the whole cycle at SpMV's DRAM intensity over-predicted it 6x (R_h 0.010 against
 * 0.060), so R_v and R_h are coupled through which roof each kernel's working set selects.
 *
 * Documented as R_h ~ P log P, but calibration killed the log on this machine: puffin's
 * reduction cost saturates beyond P ~ 12, so over the reachable range R_h ~ P. The
 * mechanism is untouched; only the exponent moved, because 24 cores is too few for tree
 * depth to bind. At P=1 there is no tree, so R_h = 0 (measured: 9.6 ns, call overhead).
 * See reduction_cost_s.
 */
[[nodiscard]] inline double R_h(const Machine& mc, int P, double cycle_seconds,
                                int64_t reductions) noexcept
{
    if (P <= 1 || reductions <= 0 || !(cycle_seconds > 0.0)) return 0.0;
    const double t_reduce  = reduction_cost_s(mc, P);
    const double t_compute = cycle_seconds / static_cast<double>(reductions);
    return t_reduce / t_compute;
}

/// A point in the regime plane, with the inputs that placed it there.
struct RegimePoint {
    int64_t n            = 0;   ///< operator dimension
    int64_t nnz          = 0;
    int     m            = 0;   ///< Krylov dimension per step
    int     P            = 0;
    double  x_reuse      = 1.0; ///< 1 = banded, 0 = fully scattered
    double  working_set  = 0.0; ///< bytes, whole Arnoldi cycle
    double  llc          = 0.0; ///< bytes
    double  ai           = 0.0; ///< FLOP/byte, SpMV
    double  W            = 0.0; ///< FLOPs per Arnoldi cycle
    int64_t reductions   = 0;   ///< R, on the MGS baseline
    double  rv           = 0.0;
    double  rh           = 0.0;
    // Per-kernel breakdown of R_h's denominator: the two kernels can sit on different roofs,
    // and which roof each got is the largest term in the cycle time.
    double  cycle_s       = 0.0;  ///< modeled seconds per Arnoldi cycle
    double  ai_mgs        = 0.0;
    double  rv_spmv       = 0.0;  ///< SpMV's own working set / aggregate L3
    double  rv_mgs        = 0.0;  ///< MGS's own working set / aggregate L3
    bool    spmv_resident = false;
    bool    mgs_resident  = false;
};

/**
 * @brief Place a synthetic or real operator in the (R_v, R_h) plane. Pure prediction.
 */
[[nodiscard]] inline RegimePoint place(const Machine& mc, int P, int64_t nnz, int64_t n,
                                       int m, double x_reuse)
{
    RegimePoint pt;
    pt.n           = n;
    pt.nnz         = nnz;
    pt.m           = m;
    pt.P           = P;
    pt.x_reuse     = x_reuse;
    pt.working_set = static_cast<double>(arnoldi_working_set_bytes(nnz, n, m));
    pt.llc         = static_cast<double>(aggregate_llc_bytes(mc, P));
    pt.ai          = spmv_intensity(nnz, n, x_reuse);
    pt.W           = arnoldi_cycle_flops(nnz, n, m);
    pt.reductions  = mgs_reductions(m);
    pt.rv          = R_v(mc, P, arnoldi_working_set_bytes(nnz, n, m));

    const CycleTime ct = arnoldi_cycle_seconds(mc, P, nnz, n, m, x_reuse);
    pt.cycle_s       = ct.total_s;
    pt.ai_mgs        = ct.ai_mgs;
    pt.spmv_resident = ct.spmv_resident;
    pt.mgs_resident  = ct.mgs_resident;
    pt.rv_spmv       = R_v(mc, P, spmv_working_set_bytes(nnz, n));
    pt.rv_mgs        = R_v(mc, P, mgs_working_set_bytes(n, m));
    pt.rh            = R_h(mc, P, ct.total_s, pt.reductions);
    return pt;
}

// Predicted boundaries
/**
 * @brief theta_v: the vertical boundary, predicted from mechanism.
 *
 * Matrix-powers begins to beat baseline SpMV at R_v = 1. The claim under test is that the
 * measured SpMV-vs-matrix-powers crossover lands there and not elsewhere.
 */
inline constexpr double kThetaV = 1.0;

/**
 * @brief theta_h: the horizontal boundary, predicted from mechanism.
 *
 * s-step begins to beat MGS at R_h = 1. Below it reductions hide inside compute; above it
 * the m(m+3)/2 serialized reductions dominate the cycle.
 */
inline constexpr double kThetaH = 1.0;

// Stability: the CholeskyQR breakdown threshold
/// Unit roundoff u = eps/2.
inline constexpr double kUnitRoundoff = std::numeric_limits<double>::epsilon() / 2.0;

/**
 * @brief The CholeskyQR safety certificate: kappa(B) <= u^(-1/2) ~ 9.5e7.
 *
 * CholQR forms G = B^T B and squares the condition number; below this threshold kappa(B)^2
 * stays inside 1/u and CholeskyQR2 recovers O(u) orthogonality (Yamamoto, Nakatsukasa,
 * Yanagisawa & Fukaya, 2015).
 *
 * Sufficient, not necessary, and not where Cholesky errors out: LLT keeps "succeeding" to
 * kappa ~ 2e12 while returning a Q whose orthogonality error has grown to 0.65, so LLT
 * failure must never be the breakdown criterion. Measured ||I - Q^T Q|| alone is also
 * insufficient: past the certificate the basis can lose rank, and a rank-collapsed Q is
 * orthonormal while spanning the wrong subspace.
 */
[[nodiscard]] inline double cholqr_kappa_limit() noexcept
{
    return 1.0 / std::sqrt(kUnitRoundoff);
}

/**
 * @brief kappa_2 of a tall matrix, via QR then SVD of the small triangular factor.
 *
 * Never form M^T M: kappa here reaches u^(-1/2) ~ 1e8 by design, so the Gram matrix would
 * have condition ~1e16 and its small eigenvalues would be pure roundoff. Householder QR is
 * backward stable and kappa(M) = kappa(R).
 */
[[nodiscard]] inline double condition_number(const Eigen::MatrixXd& M)
{
    const Eigen::Index k = M.cols();
    const Eigen::MatrixXd R = Eigen::HouseholderQR<Eigen::MatrixXd>(M)
                                  .matrixQR()
                                  .topLeftCorner(k, k)
                                  .triangularView<Eigen::Upper>();
    const Eigen::VectorXd sv = R.jacobiSvd().singularValues();
    const double smax = sv(0);
    const double smin = sv(sv.size() - 1);
    if (!(smin > 0.0)) return std::numeric_limits<double>::infinity();
    return smax / smin;
}

/// kappa_2 of a complex matrix, for the non-normal eigendecomposition path.
///
/// BDCSVD, not JacobiSVD: this runs on the full N x N eigenvector matrix (N up to 4096 at
/// six assets), where Jacobi's unblocked ~n^3.7 scaling makes it hours against BDC's
/// ~1 minute, for identical kappa.
[[nodiscard]] inline double condition_number(const Eigen::MatrixXcd& M)
{
    const Eigen::VectorXd sv = M.bdcSvd().singularValues();
    const double smax = sv(0);
    const double smin = sv(sv.size() - 1);
    if (!(smin > 0.0)) return std::numeric_limits<double>::infinity();
    return smax / smin;
}

/**
 * @brief Condition number of the monomial basis from the spectrum, NORMAL only.
 *
 * For normal A = Q diag(lambda) Q^* with orthogonal Q and v = Q c, the basis is Q M with
 * M[i,k] = c_i lambda_i^k. Therefore, kappa is a closed function of the spectrum and the starting
 * vector's coefficients, with no matrix formed. This is what makes m a controllable input.
 *
 * CRITICAL: needs Q orthogonal, i.e. NORMAL A. For non-normal A = X Lambda X^{-1} the basis
 * is X M and the true kappa departs from this by up to kappa(X) in either direction, so this
 * value is UNRELIABLE there. The discretized Black-Scholes operator is non-normal, so its
 * s_max must be MEASURED (see synthetic.hpp eigenvector_condition and the C6 control). Feed
 * c = X^{-1} v (spectral_coefficients handles the scaling).
 *
 * @param lambda eigenvalues of A
 * @param c      coefficients of v in A's eigenbasis (c = X^{-1} v)
 * @param s      highest power; the basis has s+1 columns
 */
[[nodiscard]] inline double predicted_basis_condition(const Eigen::VectorXd& lambda,
                                                      const Eigen::VectorXd& c,
                                                      int s)
{
    const Eigen::Index rows = lambda.size();
    const Eigen::Index cols = static_cast<Eigen::Index>(s) + 1;
    Eigen::MatrixXd M(rows, cols);
    M.col(0) = c;
    for (Eigen::Index k = 1; k < cols; ++k)
        M.col(k) = M.col(k - 1).cwiseProduct(lambda);

    return condition_number(M);
}

/**
 * @brief Largest s for which the monomial basis still carries the CholQR certificate.
 *
 * Purely a-priori: walks the analytic Vandermonde condition number until it crosses
 * u^(-1/2), no matrix formed. On the 1D Laplacian scaffold this yields s_max = 9, so an
 * m = 8 cycle fits inside a single certified block (spec section 3).
 */
[[nodiscard]] inline int predicted_s_max(const Eigen::VectorXd& lambda,
                                         const Eigen::VectorXd& c,
                                         int s_ceiling)
{
    const double limit = cholqr_kappa_limit();
    int s_max = 0;
    for (int s = 1; s <= s_ceiling; ++s) {
        if (predicted_basis_condition(lambda, c, s) >= limit) break;
        s_max = s;
    }
    return s_max;
}

// Predicted Krylov dimension m from the spectrum
/**
 * @brief A-priori Krylov dimension for exp(h*A)v, HERMITIAN PSD operators only.
 *
 * Hochbruck & Lubich (SINUM 1997, Thm 2) for Hermitian A with spectrum in [-4rho, 0]:
 *
 *     err <= 10 * exp(-m^2 / (5 rho))              for sqrt(4 rho) <= m <= 2 rho
 *     err <= 10 * rho^-1 * e^-rho * (e rho / m)^m  for m >= 2 rho
 *
 * with rho = h * spread / 4. The bound uses the SPREAD, so a shift cannot change m: the
 * spectrum knob's `shift` moves basis conditioning while `scale` moves m.
 *
 * DOMAIN: a Hermitian PSD theorem, meaningless for an indefinite (shift across zero) or
 * non-normal (advection != 0) operator; C3 gates on this only when Hermitian PSD. The
 * constants (prefactor 10, exponent 1/5) are deliberately loosened so the bound stays a
 * safe upper bound.
 *
 * @return the smallest m meeting `tol`, or `m_ceiling` if the bound never does.
 */
[[nodiscard]] inline int predict_krylov_dim(double lambda_min, double lambda_max,
                                            double h, double tol, int m_ceiling = 512)
{
    const double rho = h * (lambda_max - lambda_min) / 4.0;
    if (rho <= 0.0) return 1;

    for (int m = 1; m <= m_ceiling; ++m) {
        const double dm = static_cast<double>(m);
        double err;
        if (dm >= 2.0 * rho) {
            err = 10.0 / rho * std::exp(-rho)
                * std::pow(std::numbers::e_v<double> * rho / dm, dm);
        } else if (dm >= std::sqrt(4.0 * rho)) {
            err = 10.0 * std::exp(-(dm * dm) / (5.0 * rho));
        } else {
            continue;  // below the bound's range of validity
        }
        if (err <= tol) return m;
    }
    return m_ceiling;
}
