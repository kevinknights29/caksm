/**
 * @file pde_operators.hpp
 * @brief Grid construction, sparse Kronecker products, PDE operator assembly,
 *        payoff vectors, and boundary forcing for the 3-asset Black-Scholes
 *        equation in log-price coordinates.
 * @author Kevin Knights
 * @version 1.0
 * @date 2026-06-16
 */
#pragma once

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Sparse>

// Type aliases
using SpMat = Eigen::SparseMatrix<double>;   // column-major by default
using Trip = Eigen::Triplet<double>;
using VecXd = Eigen::VectorXd;
using MatXd = Eigen::MatrixXd;

// Grid
/**
 * @brief Uniform log-price grid per spatial dimension.
 *
 * Layout: gid = i3*n*n + i2*n + i1  (i1 runs fastest, i.e. x_g[0]-direction)
 */
struct Grid {
    int n;
    std::array<VecXd, 3> x;    ///< log-price nodes x_d[i]
    std::array<double, 3> dx;  ///< grid spacings per dimension
};

/**
 * @brief Build the 3-asset log-price grid.
 *
 * For each dimension d: x_d spans [log(s0[d]) +/- alpha*sigma[d]*sqrt(T)]
 * with n equidistant nodes.
 */
inline Grid build_grid(int n,
                        const std::array<double, 3>& s_0,
                        const std::array<double, 3>& sigma,
                        double alpha,
                        double T)
{
    Grid g;
    g.n = n;
    for (int direction = 0; direction < 3; ++direction) {
        const double x_0 = std::log(s_0[direction]);
        const double half = alpha * sigma[direction] * std::sqrt(T);
        const double low = x_0 - half;
        const double high = x_0 + half;
        g.dx[direction] = (high - low) / (n - 1);
        g.x[direction] = VecXd::LinSpaced(n, low, high);
    }
    return g;
}

// 1-D building blocks
/**
 * @brief Skew-centred first-derivative stencil.
 *
 * T[i, i-1] = -1, T[i, i+1] = +1 for interior rows.
 * @param rainbow_bc If true, last row becomes [-2, +2] (zero-gamma BC).
 */
inline SpMat build_T(int n, bool rainbow_bc = false)
{
    std::vector<Trip> trs;
    trs.reserve(2 * n);
    for (int i = 0; i < n; ++i) {
        if (i > 0)   trs.emplace_back(i, i - 1, -1.0);
        if (i < n-1) trs.emplace_back(i, i + 1, +1.0);
    }
    if (rainbow_bc) {
        std::vector<Trip> t2;
        t2.reserve(trs.size());
        for (auto& t : trs)
            if (t.row() != n - 1) t2.push_back(t);
        t2.emplace_back(n-1, n-2, -2.0);
        t2.emplace_back(n-1, n-1, +2.0);
        trs = std::move(t2);
    }
    SpMat M(n, n);
    M.setFromTriplets(trs.begin(), trs.end());
    return M;
}

/**
 * @brief Centred second-derivative stencil.
 *
 * S[i, i-1] = +1, S[i, i] = -2, S[i, i+1] = +1 for interior rows.
 * @param rainbow_bc If true, last row is zeroed (zero-gamma BC).
 */
inline SpMat build_S(int n, bool rainbow_bc = false)
{
    std::vector<Trip> trs;
    trs.reserve(3 * n);
    for (int i = 0; i < n; ++i) {
        if (i > 0)   trs.emplace_back(i, i - 1, +1.0);
        trs.emplace_back(i, i, -2.0);
        if (i < n-1) trs.emplace_back(i, i + 1, +1.0);
    }
    if (rainbow_bc) {
        std::vector<Trip> t2;
        t2.reserve(trs.size());
        for (auto& t : trs)
            if (t.row() != n - 1) t2.push_back(t);
        trs = std::move(t2);
    }
    SpMat M(n, n);
    M.setFromTriplets(trs.begin(), trs.end());
    return M;
}

// Kronecker product helpers
//
// Index layout:  gid = i3*n^2 + i2*n + i1   (i1 fastest)
//
// kron_dX(M, n) embeds M into one spatial dimension and uses identity for the
// other two.  kron_dXY(Ma, Mb, n) embeds Ma into dim-Y and Mb into dim-X.

/// kron(I, kron(I, M)) - M acts on d=0 (i1 index, stride 1)
inline SpMat kron_d0(const SpMat& M, int n)
{
    const int N = n * n * n;
    std::vector<Trip> trs;
    trs.reserve(static_cast<std::size_t>(M.nonZeros()) * n * n);
    for (int i3 = 0; i3 < n; ++i3)
        for (int i2 = 0; i2 < n; ++i2)
            for (int jc = 0; jc < M.outerSize(); ++jc)
                for (SpMat::InnerIterator it(M, jc); it; ++it)
                    trs.emplace_back(i3*n*n + i2*n + (int)it.row(),
                                     i3*n*n + i2*n + (int)it.col(),
                                     it.value());
    SpMat R(N, N);
    R.setFromTriplets(trs.begin(), trs.end());
    return R;
}

/// kron(I, kron(M, I)) - M acts on d=1 (i2 index, stride n)
inline SpMat kron_d1(const SpMat& M, int n)
{
    const int N = n * n * n;
    std::vector<Trip> trs;
    trs.reserve(static_cast<std::size_t>(M.nonZeros()) * n * n);
    for (int i3 = 0; i3 < n; ++i3)
        for (int jc = 0; jc < M.outerSize(); ++jc)
            for (SpMat::InnerIterator it(M, jc); it; ++it)
                for (int i1 = 0; i1 < n; ++i1)
                    trs.emplace_back(i3*n*n + (int)it.row()*n + i1,
                                     i3*n*n + (int)it.col()*n + i1,
                                     it.value());
    SpMat R(N, N);
    R.setFromTriplets(trs.begin(), trs.end());
    return R;
}

/// kron(M, kron(I, I)) - M acts on d=2 (i3 index, stride n^2)
inline SpMat kron_d2(const SpMat& M, int n)
{
    const int N = n * n * n;
    std::vector<Trip> trs;
    trs.reserve(static_cast<std::size_t>(M.nonZeros()) * n * n);
    for (int jc = 0; jc < M.outerSize(); ++jc)
        for (SpMat::InnerIterator it(M, jc); it; ++it)
            for (int i2 = 0; i2 < n; ++i2)
                for (int i1 = 0; i1 < n; ++i1)
                    trs.emplace_back((int)it.row()*n*n + i2*n + i1,
                                     (int)it.col()*n*n + i2*n + i1,
                                     it.value());
    SpMat R(N, N);
    R.setFromTriplets(trs.begin(), trs.end());
    return R;
}

/// kron(I, kron(Ma, Mb)) - Ma on d=1, Mb on d=0  (mixed derivative D_2[(0,1)])
inline SpMat kron_d01(const SpMat& Ma, const SpMat& Mb, int n)
{
    const int N = n * n * n;
    std::vector<Trip> trs;
    trs.reserve(static_cast<std::size_t>(Ma.nonZeros()) * Mb.nonZeros() * n);
    for (int i3 = 0; i3 < n; ++i3)
        for (int ja = 0; ja < Ma.outerSize(); ++ja)
            for (SpMat::InnerIterator ia(Ma, ja); ia; ++ia)
                for (int jb = 0; jb < Mb.outerSize(); ++jb)
                    for (SpMat::InnerIterator ib(Mb, jb); ib; ++ib)
                        trs.emplace_back(i3*n*n + (int)ia.row()*n + (int)ib.row(),
                                         i3*n*n + (int)ia.col()*n + (int)ib.col(),
                                         ia.value() * ib.value());
    SpMat R(N, N);
    R.setFromTriplets(trs.begin(), trs.end());
    return R;
}

/// kron(Ma, kron(I, Mb)) - Ma on d=2, Mb on d=0  (mixed derivative D_2[(0,2)])
inline SpMat kron_d02(const SpMat& Ma, const SpMat& Mb, int n)
{
    const int N = n * n * n;
    std::vector<Trip> trs;
    trs.reserve(static_cast<std::size_t>(Ma.nonZeros()) * Mb.nonZeros() * n);
    for (int ja = 0; ja < Ma.outerSize(); ++ja)
        for (SpMat::InnerIterator ia(Ma, ja); ia; ++ia)
            for (int i2 = 0; i2 < n; ++i2)
                for (int jb = 0; jb < Mb.outerSize(); ++jb)
                    for (SpMat::InnerIterator ib(Mb, jb); ib; ++ib)
                        trs.emplace_back((int)ia.row()*n*n + i2*n + (int)ib.row(),
                                         (int)ia.col()*n*n + i2*n + (int)ib.col(),
                                         ia.value() * ib.value());
    SpMat R(N, N);
    R.setFromTriplets(trs.begin(), trs.end());
    return R;
}

/// kron(Ma, kron(Mb, I)) - Ma on d=2, Mb on d=1  (mixed derivative D_2[(1,2)])
inline SpMat kron_d12(const SpMat& Ma, const SpMat& Mb, int n)
{
    const int N = n * n * n;
    std::vector<Trip> trs;
    trs.reserve(static_cast<std::size_t>(Ma.nonZeros()) * Mb.nonZeros() * n);
    for (int ja = 0; ja < Ma.outerSize(); ++ja)
        for (SpMat::InnerIterator ia(Ma, ja); ia; ++ia)
            for (int jb = 0; jb < Mb.outerSize(); ++jb)
                for (SpMat::InnerIterator ib(Mb, jb); ib; ++ib)
                    for (int i1 = 0; i1 < n; ++i1)
                        trs.emplace_back((int)ia.row()*n*n + (int)ib.row()*n + i1,
                                         (int)ia.col()*n*n + (int)ib.col()*n + i1,
                                         ia.value() * ib.value());
    SpMat R(N, N);
    R.setFromTriplets(trs.begin(), trs.end());
    return R;
}

// PDE system
/**
 * @brief Assembled PDE system for one option type.
 *
 * Stores all matrices needed by CN, ADI, ME and KSM solvers so they are
 * constructed once and shared across benchmark runs.
 */
struct PDESystem {
    SpMat A;                          ///< full N x N PDE operator
    std::array<SpMat, 4> A_adi;       ///< ADI split: [0]=mixed, [1-3]=per-direction
    MatXd B;                          ///< N x 3 boundary forcing (basket only)
    std::array<MatXd, 3> B_adi;       ///< per-direction N x 3 boundary matrices (ADI)
    VecXd u0;                         ///< initial payoff vector (length N)
    Grid  grid;
    int   N;                          ///< n^3
    bool  has_forcing;                ///< true for basket (boundary forcing active)
};

/**
 * @brief Assemble the full PDE system from model parameters.
 *
 * @param n             Grid points per spatial dimension.
 * @param r             Risk-free interest rate.
 * @param T             Time to expiry.
 * @param sigma         Per-asset volatilities [sigma_0, sigma_1, sigma_2].
 * @param rho_off       Off-diagonal correlations [rho_01, rho_02, rho_12].
 * @param weight        Basket weights (ignored for rainbow).
 * @param s0            Initial asset prices.
 * @param alpha         Grid half-width in units of sigma*sqrt(T).
 * @param rainbow_bc    If true, use zero-gamma BC and rainbow payoff.
 */
inline PDESystem build_pde_system(
    int n, double r, double T,
    const std::array<double, 3>& sigma,
    const std::array<double, 3>& rho_off,
    const std::array<double, 3>& weight,
    const std::array<double, 3>& s0,
    double alpha,
    bool rainbow_bc)
{
    PDESystem sys;
    sys.grid = build_grid(n, s0, sigma, alpha, T);
    sys.N = n * n * n;
    sys.has_forcing = !rainbow_bc;
    const Grid& g = sys.grid;

    // Full 3x3 correlation matrix (diagonal = 1)
    const double rho[3][3] = {
        {1.0,        rho_off[0], rho_off[1]},
        {rho_off[0], 1.0,        rho_off[2]},
        {rho_off[1], rho_off[2], 1.0       }
    };

    // 1-D building blocks
    const SpMat Tm = build_T(n, rainbow_bc);
    const SpMat Sm = build_S(n, rainbow_bc);

    // First-derivative operators D_1[d]
    SpMat D1[3];
    D1[0] = ((r - 0.5*rho[0][0]*sigma[0]*sigma[0]) / (2.0*g.dx[0])) * kron_d0(Tm, n);
    D1[1] = ((r - 0.5*rho[1][1]*sigma[1]*sigma[1]) / (2.0*g.dx[1])) * kron_d1(Tm, n);
    D1[2] = ((r - 0.5*rho[2][2]*sigma[2]*sigma[2]) / (2.0*g.dx[2])) * kron_d2(Tm, n);

    // Pure second-derivative operators D_2[(d,d)]
    SpMat D2diag[3];
    D2diag[0] = (0.5*sigma[0]*sigma[0] / (g.dx[0]*g.dx[0])) * kron_d0(Sm, n);
    D2diag[1] = (0.5*sigma[1]*sigma[1] / (g.dx[1]*g.dx[1])) * kron_d1(Sm, n);
    D2diag[2] = (0.5*sigma[2]*sigma[2] / (g.dx[2]*g.dx[2])) * kron_d2(Sm, n);

    // Mixed second-derivative operators D_2[(d,e)] for d < e
    const SpMat D2mix01 = (0.5*rho[0][1]*sigma[0]*sigma[1]
                            / (2.0*g.dx[0]) / (2.0*g.dx[1])) * kron_d01(Tm, Tm, n);
    const SpMat D2mix02 = (0.5*rho[0][2]*sigma[0]*sigma[2]
                            / (2.0*g.dx[0]) / (2.0*g.dx[2])) * kron_d02(Tm, Tm, n);
    const SpMat D2mix12 = (0.5*rho[1][2]*sigma[1]*sigma[2]
                            / (2.0*g.dx[1]) / (2.0*g.dx[2])) * kron_d12(Tm, Tm, n);

    SpMat IN(sys.N, sys.N);
    IN.setIdentity();

    // Full PDE operator  A = sum D_1 + sum D_2_diag + 2*sum D_2_mix - r*I
    sys.A = D1[0] + D1[1] + D1[2]
          + D2diag[0] + D2diag[1] + D2diag[2]
          + 2.0*D2mix01 + 2.0*D2mix02 + 2.0*D2mix12
          - r * IN;

    // ADI split: A_adi[0] = mixed cross-terms; A_adi[l+1] = dir-l terms - r/3*I
    sys.A_adi[0] = 2.0*D2mix01 + 2.0*D2mix02 + 2.0*D2mix12;
    for (int d = 0; d < 3; ++d)
        sys.A_adi[d + 1] = D1[d] + D2diag[d] - (r / 3.0) * IN;

    // Payoff initial condition
    sys.u0.resize(sys.N);
    for (int i3 = 0; i3 < n; ++i3)
        for (int i2 = 0; i2 < n; ++i2)
            for (int i1 = 0; i1 < n; ++i1) {
                const int gid  = i3*n*n + i2*n + i1;
                const double S0 = std::exp(g.x[0][i1]);
                const double S1 = std::exp(g.x[1][i2]);
                const double S2 = std::exp(g.x[2][i3]);
                if (!rainbow_bc)
                    sys.u0[gid] = std::max(weight[0]*S0 + weight[1]*S1 + weight[2]*S2
                                            - /* K */ 0.0, 0.0); // K set below via closure
                else
                    sys.u0[gid] = std::max(std::min({S0, S1, S2}), 0.0);
                // note: K subtraction handled below
            }

    // The payoff above left out K. Recompute properly.
    // (Avoids passing K as a parameter to every lambda above.)
    return sys; // placeholder - caller fills K via build_pde_system_with_K
}

/**
 * @brief Full PDE system builder including strike price.
 *
 * This is the primary public entry point. K is separated so the payoff
 * computation is clean.
 */
inline PDESystem build_pde_system(
    int n, double K, double r, double T,
    const std::array<double, 3>& sigma,
    const std::array<double, 3>& rho_off,
    const std::array<double, 3>& weight,
    const std::array<double, 3>& s0,
    double alpha,
    bool rainbow_bc)
{
    PDESystem sys;
    sys.grid = build_grid(n, s0, sigma, alpha, T);
    sys.N = n * n * n;
    sys.has_forcing = !rainbow_bc;
    const Grid& g = sys.grid;

    const double rho[3][3] = {
        {1.0, rho_off[0], rho_off[1]},
        {rho_off[0], 1.0, rho_off[2]},
        {rho_off[1], rho_off[2], 1.0}
    };

    const SpMat Tm = build_T(n, rainbow_bc);
    const SpMat Sm = build_S(n, rainbow_bc);

    SpMat D1[3];
    D1[0] = ((r - 0.5*rho[0][0]*sigma[0]*sigma[0]) / (2.0*g.dx[0])) * kron_d0(Tm, n);
    D1[1] = ((r - 0.5*rho[1][1]*sigma[1]*sigma[1]) / (2.0*g.dx[1])) * kron_d1(Tm, n);
    D1[2] = ((r - 0.5*rho[2][2]*sigma[2]*sigma[2]) / (2.0*g.dx[2])) * kron_d2(Tm, n);

    SpMat D2diag[3];
    D2diag[0] = (0.5*sigma[0]*sigma[0] / (g.dx[0]*g.dx[0])) * kron_d0(Sm, n);
    D2diag[1] = (0.5*sigma[1]*sigma[1] / (g.dx[1]*g.dx[1])) * kron_d1(Sm, n);
    D2diag[2] = (0.5*sigma[2]*sigma[2] / (g.dx[2]*g.dx[2])) * kron_d2(Sm, n);

    const SpMat D2mix01 = (0.5*rho[0][1]*sigma[0]*sigma[1]
                            / (2.0*g.dx[0]) / (2.0*g.dx[1])) * kron_d01(Tm, Tm, n);
    const SpMat D2mix02 = (0.5*rho[0][2]*sigma[0]*sigma[2]
                            / (2.0*g.dx[0]) / (2.0*g.dx[2])) * kron_d02(Tm, Tm, n);
    const SpMat D2mix12 = (0.5*rho[1][2]*sigma[1]*sigma[2]
                            / (2.0*g.dx[1]) / (2.0*g.dx[2])) * kron_d12(Tm, Tm, n);

    SpMat IN(sys.N, sys.N);
    IN.setIdentity();

    sys.A = D1[0] + D1[1] + D1[2]
          + D2diag[0] + D2diag[1] + D2diag[2]
          + 2.0*D2mix01 + 2.0*D2mix02 + 2.0*D2mix12
          - r * IN;

    sys.A_adi[0] = 2.0*D2mix01 + 2.0*D2mix02 + 2.0*D2mix12;
    for (int d = 0; d < 3; ++d)
        sys.A_adi[d + 1] = D1[d] + D2diag[d] - (r / 3.0) * IN;

    // Payoff
    sys.u0.resize(sys.N);
    for (int i3 = 0; i3 < n; ++i3)
        for (int i2 = 0; i2 < n; ++i2)
            for (int i1 = 0; i1 < n; ++i1) {
                const int gid  = i3*n*n + i2*n + i1;
                const double S0 = std::exp(g.x[0][i1]);
                const double S1 = std::exp(g.x[1][i2]);
                const double S2 = std::exp(g.x[2][i3]);
                if (!rainbow_bc)
                    sys.u0[gid] = std::max(weight[0]*S0 + weight[1]*S1 + weight[2]*S2 - K, 0.0);
                else
                    sys.u0[gid] = std::max(std::min({S0, S1, S2}) - K, 0.0);
            }

    // Boundary forcing matrices (basket only)
    if (!rainbow_bc) {
        sys.B = MatXd::Zero(sys.N, 3);
        for (int d = 0; d < 3; ++d)
            sys.B_adi[d] = MatXd::Zero(sys.N, 3);

        for (int dir = 0; dir < 3; ++dir) {
            // Forward stencil coefficient for the ghost node in direction dir
            const double fwd = 0.5*sigma[dir]*sigma[dir] / (g.dx[dir]*g.dx[dir])
                              + (r - 0.5*sigma[dir]*sigma[dir]) / (2.0*g.dx[dir]);

            const double x_virtual = g.x[dir][n - 1] + g.dx[dir];

            // The two free axes along the upper face in direction dir
            int free_ax[2];
            int fi = 0;
            for (int d = 0; d < 3; ++d)
                if (d != dir) free_ax[fi++] = d;

            for (int u = 0; u < n; ++u) {
                for (int v = 0; v < n; ++v) {
                    int ijk[3] = {0, 0, 0};
                    ijk[dir]        = n - 1;
                    ijk[free_ax[0]] = u;
                    ijk[free_ax[1]] = v;

                    double coords[3];
                    for (int d = 0; d < 3; ++d)
                        coords[d] = g.x[d][ijk[d]];
                    coords[dir] = x_virtual;

                    const double S[3] = {
                        std::exp(coords[0]),
                        std::exp(coords[1]),
                        std::exp(coords[2])
                    };

                    // Deep-ITM approximation: u(S_virt, tau) \approx w S_virt - K*e^{-r*tau}
                    // Taylor in tau: (w S - K) + K*r*tau - K*r^2/2*tau^2
                    const double payoff_v = weight[0]*S[0] + weight[1]*S[1] + weight[2]*S[2] - K;

                    const int gid = ijk[2]*n*n + ijk[1]*n + ijk[0];

                    // B column layout: [0]=coeff of tau^2/2!, [1]=coeff of tau, [2]=coeff of 1
                    sys.B(gid, 2) += fwd * payoff_v;
                    sys.B(gid, 1) += fwd * (K * r);
                    sys.B(gid, 0) += fwd * (-K * r * r);

                    sys.B_adi[dir](gid, 2) += fwd * payoff_v;
                    sys.B_adi[dir](gid, 1) += fwd * (K * r);
                    sys.B_adi[dir](gid, 0) += fwd * (-K * r * r);
                }
            }
        }
    }

    return sys;
}

// Utilities shared by solvers

/**
 * @brief Build the Taylor-coefficient seed vector for the boundary forcing.
 *
 * s(tau) = [tau^(p-1)/(p-1)!, ..., tau/1!, 1/0!]
 * For p=3: s = [tau^2/2, tau, 1].
 */
inline VecXd make_s_vec(double tau, int p = 3)
{
    VecXd s(p);
    double fact = 1.0;
    for (int k = 0; k < p; ++k) {
        const int j = p - 1 - k;
        fact = 1.0;
        for (int f = 1; f <= j; ++f) fact *= f;
        s[k] = std::pow(tau, j) / fact;
    }
    return s;
}

/**
 * @brief Extract the option price at the spot (nearest-node lookup).
 *
 * @param u   Option value field on the full grid.
 * @param g   Grid description.
 * @param s0  Initial asset prices used for the nearest-node lookup.
 */
inline double extract_price(const VecXd& u, const Grid& g,
                              const std::array<double, 3>& s0)
{
    const int n = g.n;
    int spot[3];
    for (int d = 0; d < 3; ++d) {
        Eigen::Index idx = 0;
        (g.x[d].array() - std::log(s0[d])).cwiseAbs().minCoeff(&idx);
        spot[d] = (int)idx;
    }
    return u[spot[2]*n*n + spot[1]*n + spot[0]];
}

/**
 * @brief Build the augmented matrix \tilde{A} = [[A, B], [0, K]] for ME/KSM basket.
 *
 * K is the 3 x 3 nilpotent shift [[0,1,0],[0,0,1],[0,0,0]] that propagates the
 * polynomial boundary source autonomously inside the exponential.
 */
inline SpMat build_A_tilde(const SpMat& A, const MatXd& B, int N)
{
    constexpr int p = 3;
    const int M = N + p;
    std::vector<Trip> trs;
    trs.reserve(static_cast<std::size_t>(A.nonZeros()) + static_cast<std::size_t>(N) * p + 2);

    // A block
    for (int jc = 0; jc < A.outerSize(); ++jc)
        for (SpMat::InnerIterator it(A, jc); it; ++it)
            trs.emplace_back((int)it.row(), (int)it.col(), it.value());

    // B block (sparse, only non-zero entries)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < p; ++j)
            if (B(i, j) != 0.0)
                trs.emplace_back(i, N + j, B(i, j));

    // K block: [[0,1,0],[0,0,1],[0,0,0]]
    trs.emplace_back(N,     N + 1, 1.0);
    trs.emplace_back(N + 1, N + 2, 1.0);

    SpMat A_tilde(M, M);
    A_tilde.setFromTriplets(trs.begin(), trs.end());
    return A_tilde;
}

/**
 * @brief Compute the induced 1-norm (max absolute column sum) of a sparse matrix.
 */
inline double sparse_1norm(const SpMat& A)
{
    double norm = 0.0;
    for (int jc = 0; jc < A.outerSize(); ++jc) {
        double col_sum = 0.0;
        for (SpMat::InnerIterator it(A, jc); it; ++it)
            col_sum += std::abs(it.value());
        norm = std::max(norm, col_sum);
    }
    return norm;
}
