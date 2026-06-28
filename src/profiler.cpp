/**
 * @file profiler.cpp
 * @brief CPU profiler for the KSM-EI (Exponential Integrator) solver.
 *
 * Answers three questions:
 *   1. Memory: how many bytes does A_op occupy (CSC layout)?
 *   2. Timing: how long does each kernel take (SpMV / GS / expm)?
 *   3. Roofline: what is the Arithmetic Intensity of each kernel,
 *                and where does it sit relative to the hardware ridge point?
 *
 * Usage:
 *   ./profiler [--n N] [--steps S] [--tol T] [--option basket|rainbow]
 *
 * @author Kevin Knights
 * @date 2026-06-26
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <format>
#include <iostream>
#include <numeric>
#include <print>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Sparse>
#include <unsupported/Eigen/MatrixFunctions>

#include "config.hpp"
#include "pde_operators.hpp"

// Timing alias
using Clock = std::chrono::steady_clock;
using Sec = std::chrono::duration<double>;

static inline double elapsed_s(Clock::time_point t0)
{
    return Sec(Clock::now() - t0).count();
}

// 1.  Memory analysis
struct MatMemory {
    int rows;
    int cols;
    long nnz; ///> number of non-zero entries
    double values_mb;  // nnz x 8 bytes  (double values)
    double indices_mb; // nnz x 4 bytes  (int32 inner indices)
    double ptrs_mb;    // (cols + 1) x 4 bytes  (int32 outer starts)
    double total_mb;
};

static MatMemory csc_memory(const SpMat& A)
{
    const long nnz = A.nonZeros();
    const long nc = A.outerSize(); // #columns for column-major (CSC)
    MatMemory m;
    m.rows = static_cast<int>(A.rows());
    m.cols = static_cast<int>(A.cols());
    m.nnz = nnz;
    m.values_mb = nnz * 8.0 / (1 << 20);
    m.indices_mb = nnz * 4.0 / (1 << 20);
    m.ptrs_mb = (nc + 1) * 4.0 / (1 << 20);
    m.total_mb = m.values_mb + m.indices_mb + m.ptrs_mb;
    return m;
}

static void print_memory_report(const MatMemory& m, std::string_view label)
{
    std::println("\n=== Memory: {} ===", label);
    std::println("||  Dimensions : {} x {}", m.rows, m.cols);
    std::println("||  Nonzeros   : {:L}", m.nnz);
    std::println("||  CSC layout :");
    std::println("||    values   {:8.3f} MB   (nnz x 8 bytes)", m.values_mb);
    std::println("||    indices  {:8.3f} MB   (nnz x 4 bytes)", m.indices_mb);
    std::println("||    col ptrs {:8.3f} MB   ((cols+1) x 4 bytes)", m.ptrs_mb);
    std::println("||    ---------------------------------------------");
    std::println("||    total    {:8.3f} MB", m.total_mb);
    std::println("===");
}

// 2.  Per-kernel statistics
struct KernelStats {
    double time_s = 0.0;
    long long flops = 0; // analytic FLOPs
    long long bytes = 0; // cold-cache bytes transferred

    [[nodiscard]] double gflops_achieved() const
    {
        return time_s > 0 ? static_cast<double>(flops) / (time_s * 1e9) : 0.0;
    }
    // Arithmetic Intensity [FLOPs / byte]
    [[nodiscard]] double ai() const
    {
        return bytes > 0 ? static_cast<double>(flops) / static_cast<double>(bytes) : 0.0;
    }
    // Attainable roof [GFLOP/s] given measured bandwidth
    [[nodiscard]] double attainable_gflops(const double bw_gbs, const double peak_gflops) const
    {
        return std::min(ai() * bw_gbs, peak_gflops);
    }
};

struct KSMEIStats {
    MatMemory matrix_mem;
    KernelStats spmv;
    KernelStats gs;
    KernelStats expm;        // dense (h * H_m).exp() (FLOPs are an upper-bound estimate)
    double t_total_s = 0.0;
    int n_steps = 0;
    int converged = 0;       // steps that exited the Arnoldi loop early
    double avg_krylov = 0.0; // mean Krylov dimension used per step
};

// 3. Instrumented KSM-EI
// FLOP counts (analytic, per call):
//   SpMV A_op * v: 2 * nnz (1 multiply + 1 add per nonzero)
//    GS at step j: (j+1) iterations, each:
//                      dot:   w \cdot V[:,i] (2 * aug_N FLOPs)
//                      daxpy: w -= h*v       (2 * aug_N FLOPs)
//                      w.norm():             (2 * aug_N FLOPs)
//                      total: 4 * (j + 1) * aug_N + 2 * aug_N
//   expm (j + 1) x (j + 1): Eigen uses Padé-13 + scaling-and-squaring.
//                       Upper-bound estimate: 13 * m^3 FLOPs  (m = j+1)
//                       (true cost for m < ~20 may be Padé-7 or lower)
// Byte counts (cold-cache model, CSC storage):
//   SpMV: nnz * 8 [values] + nnz * 4 [inner idx] + (aug_N+1) * 4 [outer ptrs]
//          + aug_N * 8 [read x] + aug_N*8 [write y]
//         total: nnz * 12 + aug_N * 16 + 4
//   GS at step j: reading V[:,0...j] = (j + 1) * aug_N * 8  (each col read twice:
//                  once for dot, once for daxpy) + w read/write
//                 = 2 * aug_N * 8
//                 = 2 * (j + 1) * aug_N * 8 + 2 * aug_N * 8
//                 = 2 * aug_N * 8 * (j + 2)
//                 = 6 * m^2 * 8 bytes read + 2 * m^2 * 8 bytes write
//                 = 64 * m^2  bytes  (upper bound, marked as estimated)
static KSMEIStats profile_ksm_ei(const PDESystem& sys, const Config& cfg)
{
    const int    N  = sys.N;
    const int    p  = 3;
    const double dt  = cfg.t_final / cfg.ei_steps;
    const double tol = cfg.tol_ei;
    constexpr int    m_max         = 50;
    constexpr double breakdown_tol = 1e-14;

    const bool is_basket = sys.has_forcing;
    const int  aug_N     = is_basket ? N + p : N;

    const SpMat A_op = is_basket ? build_A_tilde(sys.A, sys.B, N) : sys.A;

    KSMEIStats st;
    st.matrix_mem = csc_memory(A_op);
    st.n_steps    = cfg.ei_steps;

    const long nnz = A_op.nonZeros();

    // Pre-compute bytes-per-SpMV (cold-cache CSC model)
    const long long bytes_per_spmv =
        nnz * 12LL                       // values (8) + inner indices (4)
        + (long long)(aug_N + 1) * 4LL   // outer starts
        + (long long)aug_N * 16LL;       // read x (8) + write y (8)

    // Arnoldi workspace
    MatXd V(aug_N, m_max + 1);
    MatXd H_full = MatXd::Zero(m_max + 1, m_max);

    VecXd u = sys.u0;
    double t_curr   = 0.0;
    int    total_krylov = 0;

    const auto wall_start = Clock::now();

    for (int step = 0; step < cfg.ei_steps; ++step) {
        const double h    = std::min(dt, cfg.t_final - t_curr);
        const double tau0 = t_curr;

        // Build augmented start vector
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
        int  m_used    = 0;

        for (int j = 0; j < m_max; ++j) {

            // SpMV
            {
                const auto t0 = Clock::now();
                VecXd w = A_op * V.col(j);
                st.spmv.time_s += elapsed_s(t0);
                st.spmv.flops  += 2 * nnz;
                st.spmv.bytes  += bytes_per_spmv;

                // Gram-Schmidt
                const auto t1 = Clock::now();
                for (int i = 0; i <= j; ++i) {
                    H_full(i, j) = w.dot(V.col(i));
                    w -= H_full(i, j) * V.col(i);
                }
                H_full(j + 1, j) = w.norm();
                st.gs.time_s += elapsed_s(t1);
                // FLOPs: 4 * (j + 1) * aug_N (dot + daxpy per i) + 2 * aug_N (norm)
                st.gs.flops += 4LL * (j + 1) * aug_N + 2LL * aug_N;
                // Bytes: read V cols 0...j twice + r/w w
                st.gs.bytes += 2LL * (long long)(j + 2) * aug_N * 8LL;

                m_used = j + 1;

                // Exact invariant subspace
                if (H_full(j + 1, j) < breakdown_tol) {
                    const MatXd H_m = H_full.topLeftCorner(j + 1, j + 1);
                    const auto  te  = Clock::now();
                    const VecXd f = (h * H_m).exp().col(0);
                    st.expm.time_s += elapsed_s(te);
                    st.expm.flops += 13LL * (j + 1) * (j + 1) * (j + 1);
                    st.expm.bytes += 64LL * (j + 1) * (j + 1);

                    const VecXd w_r = beta * V.leftCols(j + 1) * f;
                    u = is_basket ? VecXd(w_r.head(N)) : w_r;
                    converged = true;
                    if (j + 1 < m_max) V.col(j + 1) = w / H_full(j + 1, j);
                    break;
                }

                // Convergence check (Niessen-Wright error estimate)
                {
                    const MatXd H_m = H_full.topLeftCorner(j + 1, j + 1);
                    const auto  te  = Clock::now();
                    const VecXd f   = (h * H_m).exp().col(0);
                    st.expm.time_s += elapsed_s(te);
                    st.expm.flops  += 13LL * (j + 1) * (j + 1) * (j + 1);
                    st.expm.bytes  += 64LL * (j + 1) * (j + 1);

                    const double err = beta * std::abs(H_full(j + 1, j)) * std::abs(f[j]);
                    if (err < tol) {
                        const VecXd w_r = beta * V.leftCols(j + 1) * f;
                        u = is_basket ? VecXd(w_r.head(N)) : w_r;
                        converged = true;
                        if (j + 1 < m_max) V.col(j + 1) = w / H_full(j + 1, j);
                        break;
                    }
                }

                if (j + 1 < m_max)
                    V.col(j + 1) = w / H_full(j + 1, j);
            }
        }

        if (!converged) {
            const MatXd H_m = H_full.topLeftCorner(m_max, m_max);
            const auto  te  = Clock::now();
            const VecXd f   = (h * H_m).exp().col(0);
            st.expm.time_s += elapsed_s(te);
            st.expm.flops  += 13LL * m_max * m_max * m_max;
            st.expm.bytes  += 64LL * m_max * m_max;

            const VecXd w_r = beta * V.leftCols(m_max) * f;
            u = is_basket ? VecXd(w_r.head(N)) : w_r;
            m_used = m_max;
        } else {
            ++st.converged;
        }

        total_krylov += m_used;
        t_curr += h;
    }

    st.t_total_s  = elapsed_s(wall_start);
    st.avg_krylov = static_cast<double>(total_krylov) / cfg.ei_steps;
    return st;
}

// 4.  Bandwidth benchmark (STREAM-Triad)
// Measures sustainable memory bandwidth by running:
//     c[i] = a[i] + scalar * b[i]
// across a 128 MB working set (3 x n x 8 bytes, n = 1 << 23 approx 8 M doubles)
// Two passes: one warmup (populate caches / TLB) then one timed pass
static double measure_bandwidth_gbs(int n = 1 << 23)
{
    const std::size_t sz = static_cast<std::size_t>(n);
    std::vector<double> a(sz, 1.0), b(sz, 2.0), c(sz, 0.0);
    constexpr double scalar = 0.5;

    // Warmup pass
    for (int i = 0; i < n; ++i) c[i] = a[i] + scalar * b[i];

    // Timed pass
    const auto t0 = Clock::now();
    for (int i = 0; i < n; ++i) c[i] = a[i] + scalar * b[i];
    const double elapsed = elapsed_s(t0);

    // Prevent dead-code elimination
    volatile double sink = c[n / 2];
    (void)sink;

    const double bytes = 3.0 * n * sizeof(double);  // read a, b; write c
    return bytes / elapsed / 1e9;
}

// 5.  Report printers
static void print_timing_report(const KSMEIStats& st)
{
    const double t_accounted = st.spmv.time_s + st.gs.time_s + st.expm.time_s;
    const double t_other     = std::max(0.0, st.t_total_s - t_accounted);

    auto pct = [&](double t) { return 100.0 * t / st.t_total_s; };

    std::println("\n=== Timing & Performance ===");
    std::println("||  Steps        : {}  (converged {}/{})",
                 st.n_steps, st.converged, st.n_steps);
    std::println("||  Avg Krylov m : {:.1f}", st.avg_krylov);
    std::println("||  Total wall   : {:.1f} ms\n||",
                 st.t_total_s * 1e3);
    std::println("||  {:<16} {:>10}  {:>7}  {:>12}  {:>12}",
                 "Kernel", "Time(ms)", "%Total", "GFLOPs", "AI (F/B)");
    std::println("||  {}", std::string(64, '-'));

    auto row = [&](std::string_view name, const KernelStats& k) {
        std::println("||  {:<16} {:>10.2f}  {:>6.1f}%  {:>12.4f}  {:>12.4f}",
                     name,
                     k.time_s * 1e3,
                     pct(k.time_s),
                     k.gflops_achieved(),
                     k.ai());
    };

    row("SpMV",           st.spmv);
    row("Gram-Schmidt",   st.gs);
    row("dense expm (*)", st.expm);

    std::println("||  {:<16} {:>10.2f}  {:>6.1f}%",
                 "other", t_other * 1e3, pct(t_other));
    std::println("||  {}", std::string(64, '-'));
    std::println("||  {:<16} {:>10.2f}  {:>6.1f}%",
                 "total", st.t_total_s * 1e3, 100.0);
    std::println("||\n||  (*) expm FLOPs are an upper-bound estimate (Padé-13 model).");
    std::println("||      True cost depends on Krylov dimension and Padé degree chosen.");
    std::println("===");
}

static void print_roofline_report(const KSMEIStats& st, double bw_gbs)
{
    // To correctly estimate the peak GFLOP/s, please run
    // taskset -c 2 ./build/fma-loop
    // The Intel Xeon Platinum 8358 (Ice Lake), achieved 86.1 GFLOP/s as peak_gflops
    constexpr double peak_gflops = 86.1;

    const double ridge = peak_gflops / bw_gbs;  // FLOPs/byte

    std::println("\n=== Roofline Model ===");
    std::println("||  Measured bandwidth  : {:.2f} GB/s  (STREAM-Triad, 128 MB)", bw_gbs);
    std::println("||  Peak FP64 estimate  : {:.1f} GFLOP/s",
                 peak_gflops);
    std::println("||  Ridge point         : {:.3f} FLOPs/byte\n||",
                 ridge);
    std::println("||  {:<16}  {:>8}  {:>12}  {:>14}  {}",
                 "Kernel", "AI(F/B)", "Attain(GF/s)", "Achieved(GF/s)", "Bound");
    std::println("||  {}", std::string(68, '-'));

    auto roofline_row = [&](std::string_view name, const KernelStats& k)
    {
        const double attain   = k.attainable_gflops(bw_gbs, peak_gflops);
        const double achieved = k.gflops_achieved();
        const bool   mem_bound = k.ai() < ridge;
        std::println("||  {:<16}  {:>8.3f}  {:>12.4f}  {:>14.4f}  {}",
                     name,
                     k.ai(),
                     attain,
                     achieved,
                     mem_bound ? "MEMORY" : "COMPUTE");
    };

    roofline_row("SpMV",           st.spmv);
    roofline_row("Gram-Schmidt",   st.gs);
    roofline_row("dense expm (*)", st.expm);

    std::println("||");
    std::println("||  Notes:");
    std::println("||  * AI is computed with a cold-cache byte model (worst case).");
    std::println("||    In practice, V columns fit in L2/L3 for small Krylov m,");
    std::println("||    so actual GS intensity is higher than reported.");
    std::println("||  * To get the exact peak FP64, run:");
    std::println("||      taskset -c 2 ./build/fma-loop");
    std::println("||  * expm AI is estimated, actual Padé degree may be < 13 for small m.");
    std::println("===");
}

// 6.  CLI and main
static Config parse_profiler_args(std::span<const char* const> args)
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
        if      (arg == "--n")      cfg.n              = std::stoi(std::string(next()));
        else if (arg == "--steps")  cfg.temporal_steps = std::stoi(std::string(next()));
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
            std::println("Usage: ./profiler [--n N] [--steps S] [--tol T]");
            std::println("                  [--option basket|rainbow]");
            std::println("  --n N           grid points per dimension (default 15)");
            std::println("  --steps S       KSM-EI time steps (default 100)");
            std::println("  --tol T         KSM-EI convergence tolerance (default 1e-8)");
            std::println("  --option TYPE   basket (default) or rainbow");
            std::exit(0);
        }
        else throw std::invalid_argument("Unknown flag: " + std::string(arg));
    }
    cfg.ei_steps = tol_given ? 100 : cfg.temporal_steps;
    return cfg;
}

int main(const int argc, char* argv[])
{
    try {
        const Config cfg = parse_profiler_args(
            std::span<const char* const>(argv, static_cast<std::size_t>(argc)));

        const bool rainbow = (cfg.option_type == EuropeanOptionType::CALL_MIN_RAINBOW);

        std::println("KSM-EI Profiler");
        std::println("  option={},  n={},  steps={},  tol={:.2e}",
                     rainbow ? "rainbow" : "basket",
                     cfg.n, cfg.ei_steps, cfg.tol_ei);

        // Build PDE system
        std::println("  [Building PDE system...]");
        const PDESystem sys = build_pde_system(
            cfg.n, cfg.strike_price, cfg.risk_free_rate, cfg.t_final,
            cfg.sigma, cfg.rho_off, cfg.weight, cfg.initial_prices,
            cfg.alpha, rainbow);

        // 1. Memory report
        // Build A_op once just for the memory report (profiler builds it again internally)
        {
            const SpMat A_op = sys.has_forcing
                ? build_A_tilde(sys.A, sys.B, sys.N)
                : sys.A;
            const MatMemory mem = csc_memory(A_op);
            const std::string_view label = sys.has_forcing ? "A_tilde (basket)" : "A (rainbow)";
            print_memory_report(mem, label);
        }

        // 2 and 3. Timing and Roofline
        std::println("\n  [Running instrumented KSM-EI...]");
        const KSMEIStats st = profile_ksm_ei(sys, cfg);
        print_timing_report(st);

        std::println("\n  [Measuring memory bandwidth...]");
        const double bw = measure_bandwidth_gbs();
        print_roofline_report(st, bw);

    } catch (const std::exception& e) {
        std::println(std::cerr, "Error: {}", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
