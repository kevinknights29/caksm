/**
 * @file solvers.hpp
 * @brief PDE solver implementations for the 3-asset European option pricer.
 *
 *   CN:     Crank-Nicolson
 *   ADI-DR: Douglas-Rachford ADI
 *   ADI-HV: Hundsdorfer-Verwer ADI
 *   ME:     Matrix Exponential (Al-Mohy & Higham)
 *   KSM-EI: Krylov Subspace Exponential Integrator
 *
 * All functions take a pre-assembled PDESystem and Config.
 * They return the option-value field u at time T.
 *
 * @author Kevin Knights
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <print>
#include <stdexcept>
#include <string>

#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <unsupported/Eigen/MatrixFunctions>

#include "config.hpp"
#include "me_theta.hpp"
#include "pde_operators.hpp"

// Crank-Nicolson
[[nodiscard]] inline VecXd solve_cn(const PDESystem& sys, const Config& cfg)
{
    const int N = sys.N;
    const double dt = cfg.t_final / cfg.temporal_steps;

    SpMat IN(N, N);
    IN.setIdentity();
    const SpMat LHS_mat = IN - (dt / 2.0) * sys.A;
    const SpMat RHS_mat = IN + (dt / 2.0) * sys.A;

    Eigen::SparseLU<SpMat> lu;
    lu.compute(LHS_mat);
    if (lu.info() != Eigen::Success)
        throw std::runtime_error("CN: sparse LU factorization failed");

    VecXd u = sys.u0;
    double t_curr = 0.0;

    for (int step = 0; step < cfg.temporal_steps; ++step) {
        const double h = std::min(dt, cfg.t_final - t_curr);
        const double tau_curr = t_curr;
        const double tau_next = tau_curr + h;

        VecXd rhs = RHS_mat * u;
        if (sys.has_forcing) {
            const VecXd b_curr = sys.B * make_s_vec(tau_curr);
            const VecXd b_next = sys.B * make_s_vec(tau_next);
            rhs += (h / 2.0) * (b_curr + b_next);
        }
        u = lu.solve(rhs);
        t_curr += h;
    }
    return u;
}

// ADI Douglas-Rachford
[[nodiscard]] inline VecXd solve_adi_dr(const PDESystem& sys, const Config& cfg)
{
    const int N = sys.N;
    const double dt = cfg.t_final / cfg.temporal_steps;
    constexpr double theta = 0.5;

    SpMat IN(N, N);
    IN.setIdentity();

    Eigen::SparseLU<SpMat> lu[3];
    for (int d = 0; d < 3; ++d) {
        lu[d].compute(IN - theta * dt * sys.A_adi[d + 1]);
        if (lu[d].info() != Eigen::Success)
            throw std::runtime_error("ADI-DR: sparse LU factorization failed");
    }

    SpMat A_full = sys.A_adi[0] + sys.A_adi[1] + sys.A_adi[2] + sys.A_adi[3];

    VecXd u = sys.u0;
    double t_curr = 0.0;

    for (int step = 0; step < cfg.temporal_steps; ++step) {
        const double h = std::min(dt, cfg.t_final - t_curr);
        const double tau_curr = t_curr;
        const double tau_next = tau_curr + h;

        const VecXd s_curr = make_s_vec(tau_curr);
        const VecXd s_next = make_s_vec(tau_next);

        VecXd Y = u + h * (A_full * u + (sys.has_forcing ? VecXd(sys.B * s_curr)
                                                           : VecXd::Zero(N)));
        for (int l = 0; l < 3; ++l) {
            VecXd rhs_l = Y - theta * h * (sys.A_adi[l + 1] * u);
            if (sys.has_forcing)
                rhs_l += theta * h * (sys.B_adi[l] * s_next - sys.B_adi[l] * s_curr);
            Y = lu[l].solve(rhs_l);
        }
        u = Y;
        t_curr += h;
    }
    return u;
}

// ADI Hundsdorfer-Verwer
[[nodiscard]] inline VecXd solve_adi_hv(const PDESystem& sys, const Config& cfg)
{
    const int N = sys.N;
    const double dt = cfg.t_final / cfg.temporal_steps;
    constexpr double theta  = 0.5;
    constexpr double sig_hv = 0.5;

    SpMat IN(N, N);
    IN.setIdentity();

    Eigen::SparseLU<SpMat> lu[3];
    for (int d = 0; d < 3; ++d) {
        lu[d].compute(IN - theta * dt * sys.A_adi[d + 1]);
        if (lu[d].info() != Eigen::Success)
            throw std::runtime_error("ADI-HV: sparse LU factorization failed");
    }

    SpMat A_full = sys.A_adi[0] + sys.A_adi[1] + sys.A_adi[2] + sys.A_adi[3];

    VecXd u = sys.u0;
    double t_curr = 0.0;

    for (int step = 0; step < cfg.temporal_steps; ++step) {
        const double h        = std::min(dt, cfg.t_final - t_curr);
        const double tau_curr = t_curr;
        const double tau_next = tau_curr + h;

        const VecXd s_curr = make_s_vec(tau_curr);
        const VecXd s_next = make_s_vec(tau_next);

        // Stage 1: DR predictor
        VecXd F_curr = A_full * u + (sys.has_forcing ? VecXd(sys.B * s_curr)
                                                      : VecXd::Zero(N));
        VecXd Y = u + h * F_curr;
        for (int l = 0; l < 3; ++l) {
            VecXd rhs_l = Y - theta * h * (sys.A_adi[l + 1] * u);
            if (sys.has_forcing)
                rhs_l += theta * h * (sys.B_adi[l] * s_next - sys.B_adi[l] * s_curr);
            Y = lu[l].solve(rhs_l);
        }
        const VecXd u_stage1 = Y;

        // Stage 2: HV correction
        VecXd F_next = A_full * u_stage1 + (sys.has_forcing ? VecXd(sys.B * s_next)
                                                             : VecXd::Zero(N));
        VecXd Yt = (u + h * F_curr) + sig_hv * h * (F_next - F_curr);
        for (int l = 0; l < 3; ++l) {
            const VecXd rhs_l = Yt - theta * h * (sys.A_adi[l + 1] * u_stage1);
            Yt = lu[l].solve(rhs_l);
        }
        u = Yt;
        t_curr += h;
    }
    return u;
}

// ADI Hundsdorfer-Verwer with start-up smoothing
namespace adi_hv {

/// The three directional LU factorizations for one (theta, step) pair.
struct Factorization {
    Eigen::SparseLU<SpMat> lu[3];
    double theta = 0.0;
    double step = 0.0;
    bool ready = false;
};

/// Factorize I - theta*h*A_l for each direction l, in place.
inline void factorize(Factorization& factors, const PDESystem& sys,
                      const SpMat& identity, double theta, double h)
{
    for (int d = 0; d < 3; ++d) {
        factors.lu[d].compute(identity - theta * h * sys.A_adi[d + 1]);
        if (factors.lu[d].info() != Eigen::Success)
            throw std::runtime_error("ADI-HV-S: sparse LU factorization failed");
    }
    factors.theta = theta;
    factors.step = h;
    factors.ready = true;
}

} // namespace adi_hv

/**
 * @brief Hundsdorfer-Verwer ADI with start-up smoothing (ADI-HV-S).
 *
 * The scheme is equations (14.1) to (14.5) of Dang, Christara and Jackson: an
 * explicit predictor, three unidirectional implicit correctors, an explicit
 * second-stage update, and three more unidirectional solves. Writing
 * F(tau, v) = A v + b(tau) and F_l for its direction-l part,
 *
 *   v_0    = u + h F(tau_n, u),
 *   v_l    = v_{l-1} + theta h (F_l(tau_{n+1}, v_l) - F_l(tau_n, u)),
 *   w_0    = v_0 + (h/2) (F(tau_{n+1}, v_3) - F(tau_n, u)),
 *   w_l    = w_{l-1} + theta h (F_l(tau_{n+1}, w_l) - F_l(tau_{n+1}, v_3)),
 *   u_next = w_3.
 *
 * The boundary forcing enters the first stage as the difference
 * b_l(tau_{n+1}) - b_l(tau_n) and cancels out of the second, where both terms
 * sit at the same time level.
 *
 * theta is 1 for the first @p smoothing_steps steps and 0.5 afterwards. theta = 1
 * damps the error the non-smooth payoff injects; theta = 0.5 is the more
 * accurate choice once that transient is gone. Both HV stages are kept during
 * start-up, so these steps are partially implicit rather than fully implicit.
 * That is what distinguishes this from Rannacher smoothing, and it should not
 * be described as Rannacher steps or as implicit Euler steps.
 *
 * The directional systems are factored once per distinct (theta, step) pair, so
 * a uniform partition needs exactly two sets. The accumulated time can still
 * leave the final step a rounding short of dt; that step is factored at its own
 * size rather than solved with an operator built for a different one.
 *
 * @param smoothing_steps Start-up steps at theta = 1. Zero reproduces the
 *                        unsmoothed solve_adi_hv, which is kept unchanged so
 *                        historical ADI-HV results do not shift meaning.
 */
[[nodiscard]] inline VecXd solve_adi_hv_smoothed(const PDESystem& sys,
                                                  const Config& cfg,
                                                  int smoothing_steps)
{
    if (smoothing_steps < 0)
        throw std::invalid_argument("ADI-HV-S: smoothing steps must not be negative");

    const int N = sys.N;
    const double dt = cfg.t_final / cfg.temporal_steps;
    constexpr double sig_hv = 0.5;
    constexpr double theta_startup = 1.0;
    constexpr double theta_main = 0.5;

    SpMat IN(N, N);
    IN.setIdentity();

    adi_hv::Factorization startup;
    adi_hv::Factorization main_stage;
    adi_hv::Factorization odd_step;
    adi_hv::factorize(main_stage, sys, IN, theta_main, dt);
    if (smoothing_steps > 0)
        adi_hv::factorize(startup, sys, IN, theta_startup, dt);

    const SpMat A_full = sys.A_adi[0] + sys.A_adi[1] + sys.A_adi[2] + sys.A_adi[3];

    VecXd u = sys.u0;
    double t_curr = 0.0;

    for (int step = 0; step < cfg.temporal_steps; ++step) {
        const double h        = std::min(dt, cfg.t_final - t_curr);
        const double tau_curr = t_curr;
        const double tau_next = tau_curr + h;
        const double theta    = step < smoothing_steps ? theta_startup : theta_main;

        adi_hv::Factorization* factors =
            theta == theta_startup ? &startup : &main_stage;
        if (h != dt) {
            if (!odd_step.ready || odd_step.theta != theta || odd_step.step != h)
                adi_hv::factorize(odd_step, sys, IN, theta, h);
            factors = &odd_step;
        }

        const VecXd s_curr = make_s_vec(tau_curr);
        const VecXd s_next = make_s_vec(tau_next);

        // Stage 1: explicit predictor and three unidirectional correctors
        VecXd F_curr = A_full * u + (sys.has_forcing ? VecXd(sys.B * s_curr)
                                                      : VecXd::Zero(N));
        VecXd Y = u + h * F_curr;
        for (int l = 0; l < 3; ++l) {
            VecXd rhs_l = Y - theta * h * (sys.A_adi[l + 1] * u);
            if (sys.has_forcing)
                rhs_l += theta * h * (sys.B_adi[l] * s_next - sys.B_adi[l] * s_curr);
            Y = factors->lu[l].solve(rhs_l);
        }
        const VecXd u_stage1 = Y;

        // Stage 2: HV correction, both forcing terms at the new time level
        VecXd F_next = A_full * u_stage1 + (sys.has_forcing ? VecXd(sys.B * s_next)
                                                             : VecXd::Zero(N));
        VecXd Yt = (u + h * F_curr) + sig_hv * h * (F_next - F_curr);
        for (int l = 0; l < 3; ++l) {
            const VecXd rhs_l = Yt - theta * h * (sys.A_adi[l + 1] * u_stage1);
            Yt = factors->lu[l].solve(rhs_l);
        }
        u = Yt;
        t_curr += h;
    }
    return u;
}

/// The configured smoothed solver, for the benchmark and harness dispatch tables.
[[nodiscard]] inline VecXd solve_adi_hv_s(const PDESystem& sys, const Config& cfg)
{
    return solve_adi_hv_smoothed(sys, cfg, cfg.hv_smoothing_steps);
}

// Matrix Exponential
[[nodiscard]] inline VecXd solve_me(const PDESystem& sys, const Config& cfg)
{
    const int N = sys.N;
    constexpr int p = 3;
    constexpr double method_tolerance = 6e-8;

    SpMat A_op;
    VecXd v;

    if (sys.has_forcing) {
        A_op = build_A_tilde(sys.A, sys.B, N);
        v.resize(N + p);
        v.head(N) = sys.u0;
        v.tail(p) = make_s_vec(0.0);
    } else {
        A_op = sys.A;
        v    = sys.u0;
    }

    const auto [m_star, s] = me_select_m_s(A_op, cfg.t_final);
    const double scale = cfg.t_final / s;

    for (int sub = 0; sub < s; ++sub) {
        VecXd prev = v;
        VecXd rsum = v;
        for (int k = 1; k <= m_star; ++k) {
            const VecXd curr = (scale / k) * (A_op * prev);
            rsum += curr;
            const double c_prev = prev.lpNorm<Eigen::Infinity>();
            const double c_curr = curr.lpNorm<Eigen::Infinity>();
            const double c_rsum = rsum.lpNorm<Eigen::Infinity>();
            if (c_rsum > 0.0 && (c_prev + c_curr) < method_tolerance * c_rsum)
                break;
            prev = curr;
        }
        v = rsum;
    }
    return sys.has_forcing ? VecXd(v.head(N)) : v;
}

// KSM Exponential Integrator
[[nodiscard]] inline VecXd solve_ksm_ei(const PDESystem& sys, const Config& cfg)
{
    const int N = sys.N;
    const int p = 3;
    const double dt  = cfg.t_final / cfg.ei_steps;
    const double tol = cfg.tol_ei;
    constexpr int    m_max          = 50;
    constexpr double breakdown_tol  = 1e-14;

    const bool is_basket = sys.has_forcing;
    const int  aug_N     = is_basket ? N + p : N;

    const SpMat A_op = is_basket ? build_A_tilde(sys.A, sys.B, N) : sys.A;

    MatXd V(aug_N, m_max + 1);
    MatXd H_full = MatXd::Zero(m_max + 1, m_max);

    VecXd u = sys.u0;
    double t_curr = 0.0;

    for (int step = 0; step < cfg.ei_steps; ++step) {
        const double h    = std::min(dt, cfg.t_final - t_curr);
        const double tau0 = t_curr;

        VecXd v_tilde(aug_N);
        if (is_basket) {
            v_tilde.head(N) = u;
            v_tilde.tail(p) = make_s_vec(tau0);
        } else {
            v_tilde = u;
        }

        const double beta = v_tilde.norm();
        if (beta < breakdown_tol) { t_curr += h; continue; }

        V.col(0) = v_tilde / beta;
        H_full.setZero();

        bool converged = false;

        for (int j = 0; j < m_max; ++j) {
            VecXd w = A_op * V.col(j);
            for (int i = 0; i <= j; ++i) {
                H_full(i, j) = w.dot(V.col(i));
                w -= H_full(i, j) * V.col(i);
            }
            H_full(j + 1, j) = w.norm();

            if (H_full(j + 1, j) < breakdown_tol) {
                const MatXd H_m = H_full.topLeftCorner(j + 1, j + 1);
                const VecXd f   = (h * H_m).exp().col(0);
                const VecXd w_r = beta * V.leftCols(j + 1) * f;
                u = is_basket ? VecXd(w_r.head(N)) : w_r;
                converged = true;
                break;
            }

            const MatXd H_m = H_full.topLeftCorner(j + 1, j + 1);
            const VecXd f   = (h * H_m).exp().col(0);
            const double err = beta * std::abs(H_full(j + 1, j)) * std::abs(f[j]);

            if (err < tol) {
                const VecXd w_r = beta * V.leftCols(j + 1) * f;
                u = is_basket ? VecXd(w_r.head(N)) : w_r;
                converged = true;
                break;
            }

            if (j + 1 < m_max)
                V.col(j + 1) = w / H_full(j + 1, j);
        }

        if (!converged) {
            const MatXd H_m = H_full.topLeftCorner(m_max, m_max);
            const VecXd f   = (h * H_m).exp().col(0);
            const VecXd w_r = beta * V.leftCols(m_max) * f;
            u = is_basket ? VecXd(w_r.head(N)) : w_r;
        }

        t_curr += h;
    }
    return u;
}

// ME referee
[[nodiscard]] inline VecXd compute_me_referee(const PDESystem& sys, const Config& cfg)
{
    constexpr int    m_me      = 55;
    constexpr double theta_ref = 9.9;
    constexpr double tolerance = 1.1e-16;

    const int N = sys.N;
    constexpr int p = 3;

    SpMat A_op;
    VecXd v;

    if (sys.has_forcing) {
        A_op = build_A_tilde(sys.A, sys.B, N);
        v.resize(N + p);
        v.head(N) = sys.u0;
        v.tail(p) = make_s_vec(0.0);
    } else {
        A_op = sys.A;
        v    = sys.u0;
    }

    const double norm_1 = sparse_1norm(A_op);
    const int s_me = std::max(1, static_cast<int>(
        std::ceil(cfg.t_final * norm_1 / theta_ref)));
    const double scale = cfg.t_final / s_me;

    std::println("  [ME referee: m={}, s={}]", m_me, s_me);

    for (int sub = 0; sub < s_me; ++sub) {
        VecXd prev = v;
        VecXd rsum = v;
        for (int k = 1; k <= m_me; ++k) {
            const VecXd curr = (scale / k) * (A_op * prev);
            rsum += curr;
            const double c_prev = prev.lpNorm<Eigen::Infinity>();
            const double c_curr = curr.lpNorm<Eigen::Infinity>();
            const double c_rsum = rsum.lpNorm<Eigen::Infinity>();
            if (c_rsum > 0.0 && (c_prev + c_curr) < tolerance * c_rsum)
                break;
            prev = curr;
        }
        v = rsum;
    }
    return sys.has_forcing ? VecXd(v.head(N)) : v;
}

// Referee file I/O
inline std::string referee_path(const std::string& dir, int n, bool rainbow)
{
    return dir + "/referee_n" + std::to_string(n)
               + (rainbow ? "_rainbow" : "_basket") + ".bin";
}

inline void save_referee_file(const VecXd& ref, const std::string& path)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write referee file: " + path);
    const int64_t sz = ref.size();
    f.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
    f.write(reinterpret_cast<const char*>(ref.data()), sz * sizeof(double));
}

inline VecXd load_referee_file(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    int64_t sz = 0;
    f.read(reinterpret_cast<char*>(&sz), sizeof(sz));
    VecXd ref(sz);
    f.read(reinterpret_cast<char*>(ref.data()), sz * sizeof(double));
    return ref;
}
