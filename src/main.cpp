/**
 * @file main.cpp
 * @brief European option pricer (basket call and rainbow min-call) using five
 *        PDE-based numerical methods on a 3-asset log-price grid.
 *
 *        Methods:
 *          CN: Crank-Nicolson (\theta = 0.5)
 *          ADI_DR: Alternating Direction Implicit, Douglas-Rachford (\theta = 0.5)
 *          ADI_HV: Alternating Direction Implicit, Hundsdorfer-Verwer (\theta = 0.5)
 *          ME: Matrix Exponential (Al-Mohy and Higham scaling-and-squaring)
 *          KSM_EI: Krylov Subspace Method / Exponential integrator
 *
 *        Usage:
 *          ./pricer --benchmark
 *          ./pricer --benchmark --option rainbow
 *          ./pricer --benchmark --n 20 --steps 100
 *
 * @author Kevin Knights
 * @version 1.1
 * @date 2026-06-16
 */

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <format>
#include <fstream>
#include <iostream>
#include <print>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <unsupported/Eigen/MatrixFunctions>

#include "me_theta.hpp"
#include "pde_operators.hpp"

// Types
enum class EuropeanOptionType {
    CALL_BASKET,
    CALL_MIN_RAINBOW,
};

struct Config {
    int n = 15; ///< grid points per spatial dimension
    int temporal_steps = 100;
    int ei_steps = 100; ///< KSM-EI substeps (= temporal_steps unless --tol given)
    double tol_ei = 1e-8; ///< KSM-EI convergence tolerance
    double strike_price = 100.0;
    double risk_free_rate = 0.04;
    double t_final = 1.0;
    std::array<double, 3> sigma = {0.30, 0.35, 0.40};
    std::array<double, 3> rho_off = {0.50, 0.50, 0.50};  // rho01, rho02, rho12
    std::array<double, 3> weight = {1.0/3, 1.0/3, 1.0/3};
    std::array<double, 3> initial_prices = {100.0, 100.0, 100.0};
    double alpha = 2.85;
    EuropeanOptionType option_type = EuropeanOptionType::CALL_BASKET;
    bool benchmark   = false;
    bool export_csv  = false;
};

// Solver: Crank-Nicolson
/**
 * @brief Price a European option via the Crank-Nicolson scheme (\theta = 0.5).
 *
 * Solves  (I - h/2 A) u_{n+1} = (I + h/2 A) u_n + h/2 (b(\tau_n) + b(\tau_{n+1}))
 * backward in pseudo-time \tau from 0 to T.
 *
 * @param sys  Assembled PDE system (grid, A, B, u_0).
 * @param cfg  Simulation parameters.
 * @return     Estimated option price at the initial spot.
 */
[[nodiscard]] VecXd solve_cn(const PDESystem& sys, const Config& cfg)
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
        throw std::runtime_error("CN: sparse LU factorisation failed");

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

// Solver: ADI Douglas-Rachford
/**
 * @brief Price a European option via the Douglas-Rachford ADI scheme (\theta = 0.5).
 *
 * Stage 1 (repeated for each direction l = 1, 2, 3):
 *   Y_0 = u_n + h (A u_n + b(\tau_n))
 *   (I - \theta h A_l) Y_l = Y_{l-1} - \theta h A_l u_n + \theta h (b_l(\tau_{n+1}) - b_l(\tau_n))
 *   u_{n+1} = Y_3
 *
 * @param sys  Assembled PDE system.
 * @param cfg  Simulation parameters.
 * @return     Estimated option price.
 */
[[nodiscard]] VecXd solve_adi_dr(const PDESystem& sys, const Config& cfg)
{
    const int N = sys.N;
    const double dt = cfg.t_final / cfg.temporal_steps;
    constexpr double theta = 0.5;

    SpMat IN(N, N);
    IN.setIdentity();

    // Precompute per-direction LU factors
    Eigen::SparseLU<SpMat> lu[3];
    for (int d = 0; d < 3; ++d) {
        lu[d].compute(IN - theta * dt * sys.A_adi[d + 1]);
        if (lu[d].info() != Eigen::Success)
            throw std::runtime_error("ADI-DR: sparse LU factorisation failed");
    }

    // Reconstruct full operator for the explicit predictor stage
    SpMat A_full = sys.A_adi[0] + sys.A_adi[1] + sys.A_adi[2] + sys.A_adi[3];

    VecXd u = sys.u0;
    double t_curr = 0.0;

    for (int step = 0; step < cfg.temporal_steps; ++step) {
        const double h        = std::min(dt, cfg.t_final - t_curr);
        const double tau_curr = t_curr;
        const double tau_next = tau_curr + h;

        const VecXd s_curr = make_s_vec(tau_curr);
        const VecXd s_next = make_s_vec(tau_next);

        // Y_0 = u_n + h F(\tau_n, u_n)
        VecXd Y = u + h * (A_full * u + (sys.has_forcing ? VecXd(sys.B * s_curr)
                                                          : VecXd::Zero(N)));

        // Sequential direction solves
        for (int l = 0; l < 3; ++l) {
            VecXd rhs_l = Y - theta * h * (sys.A_adi[l + 1] * u);
            if (sys.has_forcing) {
                rhs_l += theta * h * (sys.B_adi[l] * s_next
                                     - sys.B_adi[l] * s_curr);
            }
            Y = lu[l].solve(rhs_l);
        }

        u = Y;
        t_curr += h;
    }
    return u;
}

// Solver: ADI Hundsdorfer-Verwer
/**
 * @brief Price a European option via the Hundsdorfer-Verwer ADI scheme
 *        (\theta = 0.5, \sigma_HV = 0.5).
 *
 * Two-stage correction on top of the Douglas-Rachford predictor:
 *   Stage 2: \tilde{Y}_0 = Y_0^{DR} + \sigma_HV h (F(\tau_{n+1}, u^{DR}) − F(\tau_n, u_n))
 *            (I − \theta h A_l) \tilde{Y}_l = tilde{Y}_{l−1} − theta h A_l u^{DR}
 *
 * @param sys  Assembled PDE system.
 * @param cfg  Simulation parameters.
 * @return     Estimated option price.
 */
[[nodiscard]] VecXd solve_adi_hv(const PDESystem& sys, const Config& cfg)
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
            throw std::runtime_error("ADI-HV: sparse LU factorisation failed");
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

        // Stage 1: Douglas-Rachford predictor
        VecXd F_curr = A_full * u + (sys.has_forcing ? VecXd(sys.B * s_curr)
                                                      : VecXd::Zero(N));
        VecXd Y = u + h * F_curr;  // Y_0

        for (int l = 0; l < 3; ++l) {
            VecXd rhs_l = Y - theta * h * (sys.A_adi[l + 1] * u);
            if (sys.has_forcing)
                rhs_l += theta * h * (sys.B_adi[l] * s_next - sys.B_adi[l] * s_curr);
            Y = lu[l].solve(rhs_l);
        }
        const VecXd u_stage1 = Y;  // u^{DR}

        // Stage 2: HV correction
        VecXd F_next = A_full * u_stage1 + (sys.has_forcing ? VecXd(sys.B * s_next)
                                                             : VecXd::Zero(N));
        // \tilde{Y}_0 = Y_0^{DR} + \sigma_HV h (F_next - F_curr)
        // Y_0^{DR} = u + h*F_curr
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

// Solver: Matrix Exponential
/**
 * @brief Price a European option via the matrix exponential method
 *        (Al-Mohy and Higham scaling-and-squaring with Taylor polynomial).
 *
 * Computes u(T) = exp(T \tilde{A}) v_0 using s internal substeps of size T/s,
 * each approximated by the degree-m* Taylor polynomial with early termination.
 *
 * For basket options the augmented system \tilde{A} = [[A, B]; [0, K]] encodes the
 * time-varying boundary forcing autonomously.  For rainbow options A is used
 * directly (no forcing, no augmentation).
 *
 * @param sys  Assembled PDE system.
 * @param cfg  Simulation parameters.
 * @return     Estimated option price.
 */
[[nodiscard]] VecXd solve_me(const PDESystem& sys, const Config& cfg)
{
    const int N = sys.N;
    constexpr int p = 3;
    constexpr double method_tolerance = 6e-8; // fixed - single precision tolerance

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

            // Early exit when the last two terms are negligible relative to the sum
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

// Solver: Krylov Subspace Method / Exponential Integrator
/**
 * @brief Price a European option via the Krylov subspace exponential integrator.
 *
 * At each time step, approximates exp(h \tilde{A}) \tilde{v} via an incrementally-built
 * Arnoldi decomposition:
 *   \tilde{A} \approx V_m H_m V_m^T, exp(h \tilde{A}) \tilde{v} \approx \beta V_m exp(h H_m) e_1
 *
 * The Krylov dimension m is grown until the error estimate
 *   \epsilon = \beta |h_{m+1,m}| |f_m| < tolerance
 * is satisfied (Niessen and Wright criterion).
 *
 * @param sys  Assembled PDE system.
 * @param cfg  Simulation parameters.
 * @return     Estimated option price.
 */
[[nodiscard]] VecXd solve_ksm_ei(const PDESystem& sys, const Config& cfg)
{
    const int N = sys.N;
    const int p = 3;
    const double dt = cfg.t_final / cfg.ei_steps;
    const double tol = cfg.tol_ei;
    constexpr int m_max = 50;
    constexpr double breakdown_tol = 1e-14;

    const bool is_basket = sys.has_forcing;
    const int aug_N = is_basket ? N + p : N;

    const SpMat A_op = is_basket ? build_A_tilde(sys.A, sys.B, N) : sys.A;

    // Arnoldi workspace (preallocated; columns filled incrementally)
    MatXd V(aug_N, m_max + 1);
    MatXd H_full = MatXd::Zero(m_max + 1, m_max);

    VecXd u = sys.u0;
    double t_curr = 0.0;

    for (int step = 0; step < cfg.ei_steps; ++step) {
        const double h = std::min(dt, cfg.t_final - t_curr);
        const double tau0 = t_curr;

        // Augmented start vector \tilde{v} = [u_k; s(\tau_0)]  (or just u_k for rainbow)
        VecXd v_tilde(aug_N);
        if (is_basket) {
            v_tilde.head(N) = u;
            v_tilde.tail(p) = make_s_vec(tau0);
        } else {
            v_tilde = u;
        }

        const double beta = v_tilde.norm();
        if (beta < breakdown_tol) {
            t_curr += h;
            continue;
        }

        V.col(0) = v_tilde / beta;
        H_full.setZero();

        bool converged = false;

        for (int j = 0; j < m_max; ++j) {
            // Arnoldi step: w = A V[:,j], then orthogonalize
            VecXd w = A_op * V.col(j);
            for (int i = 0; i <= j; ++i) {
                H_full(i, j) = w.dot(V.col(i));
                w -= H_full(i, j) * V.col(i);
            }
            H_full(j + 1, j) = w.norm();

            if (H_full(j + 1, j) < breakdown_tol) {
                // Invariant subspace reached (exact in this Krylov space)
                const MatXd H_m = H_full.topLeftCorner(j + 1, j + 1);
                const VecXd f = (h * H_m).exp().col(0);
                const VecXd w_res = beta * V.leftCols(j + 1) * f;
                u = is_basket ? VecXd(w_res.head(N)) : w_res;
                converged = true;
                break;
            }

            // Convergence check before extending basis
            const MatXd H_m = H_full.topLeftCorner(j + 1, j + 1);
            const VecXd f = (h * H_m).exp().col(0);
            const double err = beta * std::abs(H_full(j + 1, j)) * std::abs(f[j]);

            if (err < tol) {
                const VecXd w_res = beta * V.leftCols(j + 1) * f;
                u = is_basket ? VecXd(w_res.head(N)) : w_res;
                converged = true;
                break;
            }

            if (j + 1 < m_max)
                V.col(j + 1) = w / H_full(j + 1, j);
        }

        if (!converged) {
            // Use the full m_max-dimensional Krylov approximation
            const MatXd H_m = H_full.topLeftCorner(m_max, m_max);
            const VecXd f = (h * H_m).exp().col(0);
            const VecXd w_res = beta * V.leftCols(m_max) * f;
            u = is_basket ? VecXd(w_res.head(N)) : w_res;
        }

        t_curr += h;
    }
    return u;
}

// Referee: Matrix Exponential (high-accuracy ODE reference)
/**
 * @brief Compute the ODE reference solution via the matrix exponential method.
 *
 * Uses fixed parameters m=55 (maximum Taylor degree) and theta=9.9 to choose
 * the substep count s = ceil(T * ||A_tilde||_1 / theta), with early-termination
 * tolerance 1.1e-16 (= 2^-53, machine epsilon).  This matches the Python
 * reference implementation exactly.
 *
 * @param sys  Assembled PDE system.
 * @param cfg  Simulation parameters.
 * @return     Full N-length solution vector at time T.
 */
[[nodiscard]] VecXd compute_me_referee(const PDESystem& sys, const Config& cfg)
{
    constexpr int    m_me      = 55;
    constexpr double theta_ref = 9.9;
    constexpr double tolerance = 1.1e-16;  // 2^-53

    const int N = sys.N;
    constexpr int p = 3;

    SpMat A_op;
    VecXd v;

    if (sys.has_forcing) {
        A_op = build_A_tilde(sys.A, sys.B, N);
        v.resize(N + p);
        v.head(N) = sys.u0;
        v.tail(p) = make_s_vec(0.0);  // [0, 0, 1] at tau=0
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

// CLI
/**
 * @brief Parse command-line arguments into a Config struct.
 *
 * Supported flags:
 *   --benchmark             Run all methods for both option types.
 *   --option  basket|rainbow  Select option type (default: basket).
 *   --n       N             Grid points per dimension (default: 15).
 *   --steps   M             Temporal steps (default: 100).
 *   --help                  Print usage and exit.
 *
 * @param args  Full argv span including program name at index 0.
 * @return      Populated Config struct.
 * @throws std::invalid_argument on unknown flags or missing values.
 */
Config parse_args(std::span<const char* const> args)
{
    Config cfg;
    bool tol_given = false;
    for (std::size_t i = 1; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        auto next = [&]() -> std::string_view {
            if (++i >= args.size())
                throw std::invalid_argument("Missing value for " + std::string(arg));
            return args[i];
        };
        if      (arg == "--benchmark") cfg.benchmark   = true;
        else if (arg == "--export")    cfg.export_csv  = true;
        else if (arg == "--n")         cfg.n              = std::stoi(std::string(next()));
        else if (arg == "--steps")     cfg.temporal_steps = std::stoi(std::string(next()));
        else if (arg == "--tol") {
            cfg.tol_ei = std::stod(std::string(next()));
            tol_given  = true;
        }
        else if (arg == "--option") {
            const auto opt = next();
            if      (opt == "basket")  cfg.option_type = EuropeanOptionType::CALL_BASKET;
            else if (opt == "rainbow") cfg.option_type = EuropeanOptionType::CALL_MIN_RAINBOW;
            else throw std::invalid_argument("Unknown option type: " + std::string(opt));
        }
        else if (arg == "--help") {
            std::println("Usage: ./pricer --benchmark [--option basket|rainbow]");
            std::println("               [--n N] [--steps M] [--tol T] [--export]");
            std::println("  --steps M   temporal steps for CN, ADI-DR, ADI-HV, KSM-EI");
            std::println("  --tol T     KSM-EI convergence tolerance (default 1e-8);");
            std::println("              when given, KSM-EI uses the default step count");
            std::println("              regardless of --steps");
            std::println("  --export    write results to caksm_export_<timestamp>.csv");
            std::exit(0);
        }
        else throw std::invalid_argument("Unknown flag: " + std::string(arg));
    }
    // When --tol is given, KSM-EI ignores --steps and uses the default count (100)
    cfg.ei_steps = tol_given ? 100 : cfg.temporal_steps;
    return cfg;
}

// Benchmark runner
/**
 * @brief Run all five methods for a given option type and print a results table.
 *
 * Builds the PDE system once and shares it across all solvers.  Each method is
 * timed and its price printed next to the known reference value.
 *
 * @param cfg         Simulation configuration (n, steps, option type, etc.).
 * @param option_type Option type to price.
 */
void run_benchmark(const Config& cfg, EuropeanOptionType option_type,
                   std::ofstream* csv = nullptr)
{
    const bool rainbow = (option_type == EuropeanOptionType::CALL_MIN_RAINBOW);

    const std::string_view opt_label = rainbow ? "Rainbow min-call" : "Basket call";
    const double reference           = rainbow ? 4.4450 : 13.2449;

    std::println("\n {}", opt_label);
    std::println("  Parameters: n={}, steps={}, K={:.0f}, r={:.2f}, T={:.1f}",
                 cfg.n, cfg.temporal_steps,
                 cfg.strike_price, cfg.risk_free_rate, cfg.t_final);
    std::println("  sigma=[{:.2f},{:.2f},{:.2f}]  rho_off=[{:.1f},{:.1f},{:.1f}]",
                 cfg.sigma[0], cfg.sigma[1], cfg.sigma[2],
                 cfg.rho_off[0], cfg.rho_off[1], cfg.rho_off[2]);
    std::println("  KSM-EI: tol={:.2e}, steps={}", cfg.tol_ei, cfg.ei_steps);
    std::println("  Reference price: {:.4f}\n", reference);

    // Build PDE system (shared across all solvers for this option type)
    std::println("  [Building PDE system...]");
    const PDESystem sys = build_pde_system(
        cfg.n, cfg.strike_price, cfg.risk_free_rate, cfg.t_final,
        cfg.sigma, cfg.rho_off, cfg.weight, cfg.initial_prices,
        cfg.alpha, rainbow);

    // Compute high-accuracy ODE reference (ME referee, m=55, tol=2^-53)
    std::println("  [Computing ME referee...]");
    const VecXd ref_u    = compute_me_referee(sys, cfg);
    const VecXd ref_cube = extract_cube(ref_u, sys.grid, cfg.initial_prices);

    std::println("  {:<12} {:>10}  {:>10}  {:>10}  {:>10}",
                 "Method", "Price", "PDE Err", "ODE Err", "Time(ms)");
    std::println("  {}", std::string(60, '-'));

    struct Run {
        std::string_view name;
        VecXd (*fn)(const PDESystem&, const Config&);
    };

    const Run runs[] = {
        {"CN",      solve_cn},
        {"ADI-DR",  solve_adi_dr},
        {"ADI-HV",  solve_adi_hv},
        {"ME",      solve_me},
        {"KSM-EI",  solve_ksm_ei},
    };

    for (const auto& run : runs) {
        const auto t0 = std::chrono::steady_clock::now();
        const VecXd u = run.fn(sys, cfg);
        const auto t1 = std::chrono::steady_clock::now();

        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double price = extract_price(u, sys.grid, cfg.initial_prices);
        const double pde_err = std::abs(price - reference);
        const double ode_err = (extract_cube(u, sys.grid, cfg.initial_prices) - ref_cube).norm();

        std::println("  {:<12} {:>10.4f}  {:>10.4f}  {:>10.3e}  {:>10.1f}",
                     run.name, price, pde_err, ode_err, ms);

        if (csv) {
            *csv << std::format("{},{},{},{},{:.6e},{},{:.6f},{:.6f},{:.6e},{:.3f}\n",
                                (rainbow ? "rainbow" : "basket"),
                                cfg.n,
                                cfg.temporal_steps,
                                cfg.ei_steps,
                                cfg.tol_ei,
                                run.name,
                                price,
                                pde_err,
                                ode_err,
                                ms);
        }
    }
    std::println("");
}

int main(const int argc, char* argv[])
{
    try {
        const Config cfg = parse_args(
            std::span<const char* const>(argv, static_cast<std::size_t>(argc)));

        if (!cfg.benchmark) {
            std::println("No action specified. Use --benchmark to run all methods.");
            std::println("Run with --help for usage information.");
            return EXIT_SUCCESS;
        }

        std::println("European Option PDE Pricer [Benchmark]");

        std::ofstream csv;
        if (cfg.export_csv) {
            std::time_t now = std::time(nullptr);
            char ts[20];
            std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", std::localtime(&now));
            const std::string csv_path = std::string("caksm_export_") + ts + ".csv";
            csv.open(csv_path);
            if (!csv)
                throw std::runtime_error("Cannot open CSV file: " + csv_path);
            csv << "option_type,n,temporal_steps,ei_steps,tol_ei,"
                   "method,price,pde_err,ode_err,time_ms\n";
            std::println("  Exporting to {}\n", csv_path);
        }

        std::ofstream* csv_ptr = cfg.export_csv ? &csv : nullptr;
        run_benchmark(cfg, EuropeanOptionType::CALL_BASKET,       csv_ptr);
        run_benchmark(cfg, EuropeanOptionType::CALL_MIN_RAINBOW,  csv_ptr);

    } catch (const std::exception& e) {
        std::println(std::cerr, "Error: {}", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
