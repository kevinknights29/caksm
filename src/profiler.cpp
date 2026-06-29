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
 *   ./profiler --hw PRESET [--n N] [--steps S] [--tol T] [--option basket|rainbow]
 *              [--peak-gflops X] [--l2-kb X] [--l3-kb X]
 *
 * Hardware presets (--hw, required):
 *   intel-8358  Intel Xeon Platinum 8358 (Lambda AI cloud VM)
 *   amd-3960x   AMD Ryzen Threadripper 3960X (puffin local cluster)
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
#include <optional>
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

// Hardware configuration
// Two built-in presets cover the two known compute environments.
// Select at runtime with --hw <key>; override individual values with
// --peak-gflops / --l2-kb / --l3-kb.
// To add a new environment: add a preset entry below and rebuild once.
// To switch environments: no rebuild needed — just pass --hw.

struct HardwareConfig {
    std::string name;
    double peak_gflops;  // sustained FP64 peak [GFLOP/s], measured with fma-loop
    long   l2_bytes;     // per-core L2 cache size [bytes]
    long   l3_bytes;     // per-core (or per-CCX) L3 cache size [bytes]
};

struct HWPreset {
    std::string_view key;
    HardwareConfig   hw;
};

// Two presets for the two compute environments in the project.
// Cache sizes are per-core / per-CCX figures used for single-threaded probing.
// Intel figures are per-vCPU as reported by lscpu on the Lambda AI VM.
// AMD figures are per-CCX (3 cores share 16 MiB L3; L2 is private per core).
static const std::array<HWPreset, 2> kHWPresets {{
    {
        "intel-8358",
        {
            "Intel Xeon Platinum 8358 (Lambda AI)",
            86.1,        // measured: fma-loop AVX-512 path, taskset -c 2
            4L << 20,    // 4 MiB L2 per vCPU  (120 MiB / 30 vCPUs)
            16L << 20,   // 16 MiB L3 per vCPU  (480 MiB / 30 vCPUs)
        }
    },
    {
        "amd-3960x",
        {
            "AMD Ryzen Threadripper 3960X (puffin)",
            68.12,       // measured: fma-loop AVX2+FMA path, taskset -c 2
            512L << 10,  // 512 KiB L2 per core  (12 MiB / 24 cores)
            16L << 20,   // 16 MiB L3 per CCX    (128 MiB / 8 CCXes)
        }
    },
}};

static const HardwareConfig& lookup_hw_preset(std::string_view key)
{
    for (const auto& p : kHWPresets)
        if (p.key == key)
            return p.hw;
    throw std::invalid_argument("Unknown hardware preset: " + std::string(key)
        + ".  Valid keys: intel-8358, amd-3960x");
}

// Memory analysis
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

// Per-kernel statistics
struct KernelStats {
    double time_s = 0.0;
    long long flops = 0;    // analytic FLOPs
    long long bytes = 0;    // cold-cache bytes transferred
    long long ws_bytes = 0; // average working-set size (for cache-tier classification)

    [[nodiscard]] double gflops_achieved() const
    {
        return time_s > 0 ? static_cast<double>(flops) / (time_s * 1e9) : 0.0;
    }
    // Arithmetic Intensity [FLOPs / byte]
    [[nodiscard]] double ai() const
    {
        return bytes > 0 ? static_cast<double>(flops) / static_cast<double>(bytes) : 0.0;
    }
    // Attainable roof [GFLOP/s] given bandwidth and compute peak
    [[nodiscard]] double attainable_gflops(double bw_gbs, double peak_gflops) const
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

// Instrumented KSM-EI
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
// Working-set sizes (for cache-tier classification):
//   SpMV: full data footprint = bytes_per_spmv  (A + x + y; A dominates)
//   GS:   V matrix columns in use = avg_krylov * aug_N * 8 bytes
//   expm: H_m matrix = avg_krylov^2 * 8 bytes  (always small, typically L1/L2-resident)
static KSMEIStats profile_ksm_ei(const PDESystem& sys, const Config& cfg)
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

    KSMEIStats st;
    st.matrix_mem = csc_memory(A_op);
    st.n_steps = cfg.ei_steps;

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
        int m_used = 0;

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
                    const auto te  = Clock::now();
                    const VecXd f = (h * H_m).exp().col(0);
                    st.expm.time_s += elapsed_s(te);
                    st.expm.flops += 13LL * (j + 1) * (j + 1) * (j + 1);
                    st.expm.bytes += 64LL * (j + 1) * (j + 1);

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
            const auto te  = Clock::now();
            const VecXd f = (h * H_m).exp().col(0);
            st.expm.time_s += elapsed_s(te);
            st.expm.flops += 13LL * m_max * m_max * m_max;
            st.expm.bytes += 64LL * m_max * m_max;

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

    // Working-set sizes for cache-tier classification.
    // SpMV: full A matrix + vectors — whether A fits in L3 determines the roof.
    // GS:   V columns [0, ..., m - 1] must all be live simultaneously for the dot+daxpy loop.
    // expm: H_m is mxm, always tiny (at most 50x50x8=20 KB) -> permanently L1/L2-resident.
    st.spmv.ws_bytes = bytes_per_spmv;
    st.gs.ws_bytes = static_cast<long long>(st.avg_krylov * aug_N * 8.0);
    st.expm.ws_bytes = static_cast<long long>(st.avg_krylov * st.avg_krylov * 8.0);

    return st;
}

// Bandwidth benchmark (STREAM-Triad)
// Measures sustainable memory bandwidth by running:
//     c[i] = a[i] + scalar * b[i]
// One warmup pass (populate caches / TLB), then one timed pass.
// Choose n so that 3 * n * 8 bytes fits the target cache level.
static double measure_bandwidth_gbs(long n)
{
    const std::size_t sz = static_cast<std::size_t>(n);
    std::vector<double> a(sz, 1.0), b(sz, 2.0), c(sz, 0.0);
    constexpr double scalar = 0.5;

    // Warmup: brings data into the target cache level (or confirms DRAM path)
    for (long i = 0; i < n; ++i) c[i] = a[i] + scalar * b[i];

    // Timed pass
    const auto t0 = Clock::now();
    for (long i = 0; i < n; ++i) c[i] = a[i] + scalar * b[i];
    const double elapsed = elapsed_s(t0);

    // Prevent dead-code elimination
    volatile double sink = c[n / 2];
    (void)sink;

    const double bytes = 3.0 * static_cast<double>(n) * sizeof(double);  // read a, b; write c
    return bytes / elapsed / 1e9;
}

// Cache-aware bandwidth hierarchy
// Runs the STREAM-Triad probe at three working-set sizes derived from the
// hardware cache topology.  Probe sizes are rounded to powers of 2 and
// target 75 % of each cache level so the array fits comfortably inside it.

struct BandwidthHierarchy {
    double l2_gbs;
    double l3_gbs;
    double dram_gbs;
    long   l2_n;    // probe element count (3 * n * 8 bytes = working set)
    long   l3_n;
    long   dram_n;
};

static long pow2_floor(long n)
{
    long r = 1;
    while (r * 2 <= n) r *= 2;
    return r;
}

static BandwidthHierarchy measure_bandwidth_hierarchy(const HardwareConfig& hw)
{
    // Target ~75 % of the cache level: 3 * n * 8 = cache_bytes * 0.75
    const long n_l2   = pow2_floor(hw.l2_bytes * 3L / 4L / 24L);
    const long n_l3   = pow2_floor(hw.l3_bytes * 3L / 4L / 24L);
    const long n_dram = 1L << 23;  // 192 MB total — well above any L3

    return {
        measure_bandwidth_gbs(n_l2),
        measure_bandwidth_gbs(n_l3),
        measure_bandwidth_gbs(n_dram),
        n_l2, n_l3, n_dram,
    };
}

// Cache-tier classification
enum class CacheTier { L2, L3, DRAM };

static std::string_view tier_name(CacheTier t)
{
    switch (t) {
        case CacheTier::L2:   return "L2";
        case CacheTier::L3:   return "L3";
        case CacheTier::DRAM: return "DRAM";
    }
    return "?";
}

static CacheTier classify_tier(long long ws_bytes, const HardwareConfig& hw)
{
    if (ws_bytes <= hw.l2_bytes) return CacheTier::L2;
    if (ws_bytes <= hw.l3_bytes) return CacheTier::L3;
    return CacheTier::DRAM;
}

static double tier_bandwidth(CacheTier t, const BandwidthHierarchy& bw)
{
    switch (t) {
        case CacheTier::L2:   return bw.l2_gbs;
        case CacheTier::L3:   return bw.l3_gbs;
        case CacheTier::DRAM: return bw.dram_gbs;
    }
    return bw.dram_gbs;
}

static std::string fmt_bytes(long long b)
{
    if (b < 1024LL)          return std::format("{} B",    b);
    if (b < 1024LL * 1024LL) return std::format("{:.0f} KiB", static_cast<double>(b) / 1024.0);
    return                          std::format("{:.1f} MiB", static_cast<double>(b) / (1024.0 * 1024.0));
}

// Report printers
static void print_timing_report(const KSMEIStats& st)
{
    const double t_accounted = st.spmv.time_s + st.gs.time_s + st.expm.time_s;
    const double t_other = std::max(0.0, st.t_total_s - t_accounted);

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

static void print_roofline_report(const KSMEIStats& st,
                                  const HardwareConfig& hw,
                                  const BandwidthHierarchy& bw)
{
    const double ridge_l2 = hw.peak_gflops / bw.l2_gbs;
    const double ridge_l3 = hw.peak_gflops / bw.l3_gbs;
    const double ridge_dram = hw.peak_gflops / bw.dram_gbs;

    std::println("\n=== Cache-Aware Roofline ===");
    std::println("||  Hardware      : {}", hw.name);
    std::println("||  Peak FP64     : {:.1f} GFLOP/s  (fma-loop)", hw.peak_gflops);
    std::println("||  Avg Krylov m : {:.1f}  ->  GS working set {}", st.avg_krylov, fmt_bytes(st.gs.ws_bytes));
    std::println("||");

    // Bandwidth hierarchy table
    std::println("||  {:<10}  {:>11}  {:>14}  {:>10}",
                 "Mem tier", "BW (GB/s)", "Ridge (F/B)", "Probe WS");
    std::println("||  {}", std::string(52, '-'));
    std::println("||  {:<10}  {:>11.2f}  {:>14.3f}  {:>10}",
                 "DRAM",
                 bw.dram_gbs, ridge_dram,
                 fmt_bytes(3LL * bw.dram_n * 8LL));
    std::println("||  {:<10}  {:>11.2f}  {:>14.3f}  {:>10}",
                 "L3",
                 bw.l3_gbs, ridge_l3,
                 fmt_bytes(3LL * bw.l3_n * 8LL));
    std::println("||  {:<10}  {:>11.2f}  {:>14.3f}  {:>10}",
                 "L2",
                 bw.l2_gbs, ridge_l2,
                 fmt_bytes(3LL * bw.l2_n * 8LL));
    std::println("||");

    // Per-kernel roofline rows
    std::println("||  {:<16}  {:>8}  {:>12}  {:>6}  {:>12}  {:>14}  {}",
                 "Kernel", "AI(F/B)", "Working set", "Tier", "Attain(GF/s)", "Achieve(GF/s)", "Bound");
    std::println("||  {}", std::string(82, '-'));

    auto roofline_row = [&](std::string_view name, const KernelStats& k)
    {
        const CacheTier tier = classify_tier(k.ws_bytes, hw);
        const double bw_tier = tier_bandwidth(tier, bw);
        const double ridge = hw.peak_gflops / bw_tier;
        const double attain = k.attainable_gflops(bw_tier, hw.peak_gflops);
        const double achieved = k.gflops_achieved();
        const bool mem_bound = k.ai() < ridge;
        std::println("||  {:<16}  {:>8.3f}  {:>12}  {:>6}  {:>12.4f}  {:>14.4f}  {}",
                     name, k.ai(), fmt_bytes(k.ws_bytes), tier_name(tier),
                     attain, achieved,
                     mem_bound ? "MEMORY" : "COMPUTE");
    };

    roofline_row("SpMV",           st.spmv);
    roofline_row("Gram-Schmidt",   st.gs);
    roofline_row("dense expm (*)", st.expm);

    std::println("||");
    std::println("||  Notes:");
    std::println("||  * Working sets: SpMV = A + vectors (cold-cache footprint); GS = V cols");
    std::println("||    [0..m-1] in flight simultaneously; expm = H_m matrix (always tiny).");
    std::println("||  * AI uses a cold-cache byte model (lower bound); warm-cache AI is higher.");
    std::println("||  * Attainable roof is min(AI * BW_tier, peak).  Tier is derived from");
    std::println("||    whether the kernel working set fits in L2, L3, or spills to DRAM.");
    std::println("||  * GS tier crosses L2→L3→DRAM as N grows; the crossing is the");
    std::println("||    cache-residency boundary and the target for CA reblocking.");
    std::println("||  * expm AI is an upper-bound estimate; actual Padé degree < 13 for small m.");
    std::println("===");
}

// CLI
struct ProfilerConfig {
    Config        pde;
    HardwareConfig hw;
};

static ProfilerConfig parse_profiler_args(std::span<const char* const> args)
{
    Config pde_cfg;
    bool   tol_given = false;

    // Two-pass approach: resolve --hw first so the preset is loaded before any
    // per-field overrides (--peak-gflops / --l2-kb / --l3-kb) are applied.
    std::optional<HardwareConfig> hw_opt;
    for (std::size_t i = 1; i < args.size(); ++i) {
        if (std::string_view(args[i]) == "--hw" && i + 1 < args.size()) {
            hw_opt = lookup_hw_preset(args[i + 1]);
            break;
        }
    }

    for (std::size_t i = 1; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        auto next = [&]() -> std::string_view {
            if (++i >= args.size())
                throw std::invalid_argument("Missing value for " + std::string(arg));
            return args[i];
        };
        if      (arg == "--n")     pde_cfg.n              = std::stoi(std::string(next()));
        else if (arg == "--steps") pde_cfg.temporal_steps = std::stoi(std::string(next()));
        else if (arg == "--tol") {
            pde_cfg.tol_ei = std::stod(std::string(next()));
            tol_given = true;
        }
        else if (arg == "--option") {
            const auto opt = next();
            if      (opt == "basket")  pde_cfg.option_type = EuropeanOptionType::CALL_BASKET;
            else if (opt == "rainbow") pde_cfg.option_type = EuropeanOptionType::CALL_MIN_RAINBOW;
            else throw std::invalid_argument("Unknown option type: " + std::string(opt));
        }
        else if (arg == "--hw") {
            next();  // already consumed in the first pass, skip the value
        }
        else if (arg == "--peak-gflops") {
            if (!hw_opt) throw std::invalid_argument("--peak-gflops requires --hw to be given first");
            hw_opt->peak_gflops = std::stod(std::string(next()));
        }
        else if (arg == "--l2-kb") {
            if (!hw_opt) throw std::invalid_argument("--l2-kb requires --hw to be given first");
            hw_opt->l2_bytes = std::stol(std::string(next())) * 1024L;
        }
        else if (arg == "--l3-kb") {
            if (!hw_opt) throw std::invalid_argument("--l3-kb requires --hw to be given first");
            hw_opt->l3_bytes = std::stol(std::string(next())) * 1024L;
        }
        else if (arg == "--help") {
            std::println("Usage: ./profiler --hw PRESET [--n N] [--steps S] [--tol T]");
            std::println("                  [--option basket|rainbow]");
            std::println("                  [--peak-gflops X] [--l2-kb X] [--l3-kb X]");
            std::println("");
            std::println("  --hw PRESET       hardware preset (required): intel-8358 | amd-3960x");
            std::println("  --n N             grid points per dimension (default 15)");
            std::println("  --steps S         KSM-EI time steps (default 100)");
            std::println("  --tol T           KSM-EI convergence tolerance (default 1e-8)");
            std::println("  --option TYPE     basket (default) or rainbow");
            std::println("  --peak-gflops X   override sustained FP64 peak [GFLOP/s]");
            std::println("  --l2-kb X         override per-core L2 size [KiB]");
            std::println("  --l3-kb X         override per-core/per-CCX L3 size [KiB]");
            std::println("");
            std::println("  Preset cache sizes (single-core, per lscpu):");
            std::println("    intel-8358  L2=4096 KiB  L3=16384 KiB  peak=86.1 GFLOP/s  (fma-loop AVX-512)");
            std::println("    amd-3960x   L2=512 KiB   L3=16384 KiB  peak=68.12 GFLOP/s (fma-loop AVX2+FMA)");
            std::exit(0);
        }
        else throw std::invalid_argument("Unknown flag: " + std::string(arg));
    }
    if (!hw_opt)
        throw std::invalid_argument(
            "Missing required flag: --hw  (intel-8358 | amd-3960x)\n"
            "  Run ./profiler --help for usage.");

    pde_cfg.ei_steps = tol_given ? 100 : pde_cfg.temporal_steps;
    return { pde_cfg, *hw_opt };
}

int main(const int argc, char* argv[])
{
    try {
        auto [cfg, hw] = parse_profiler_args(
            std::span<const char* const>(argv, static_cast<std::size_t>(argc)));

        const bool rainbow = (cfg.option_type == EuropeanOptionType::CALL_MIN_RAINBOW);

        std::println("KSM-EI Profiler");
        std::println("  hw={},  option={},  n={},  steps={},  tol={:.2e}",
                     hw.name,
                     rainbow ? "rainbow" : "basket",
                     cfg.n, cfg.ei_steps, cfg.tol_ei);

        // Build PDE system
        std::println("  [Building PDE system...]");
        const PDESystem sys = build_pde_system(
            cfg.n, cfg.strike_price, cfg.risk_free_rate, cfg.t_final,
            cfg.sigma, cfg.rho_off, cfg.weight, cfg.initial_prices,
            cfg.alpha, rainbow);

        // 1. Memory report
        {
            const SpMat A_op = sys.has_forcing
                ? build_A_tilde(sys.A, sys.B, sys.N)
                : sys.A;
            const MatMemory mem = csc_memory(A_op);
            const std::string_view label = sys.has_forcing ? "A_tilde (basket)" : "A (rainbow)";
            print_memory_report(mem, label);
        }

        // 2. Timing
        std::println("\n  [Running instrumented KSM-EI...]");
        const KSMEIStats st = profile_ksm_ei(sys, cfg);
        print_timing_report(st);

        // 3. Cache-aware roofline
        std::println("\n  [Measuring bandwidth hierarchy (L2 / L3 / DRAM)...]");
        const BandwidthHierarchy bw = measure_bandwidth_hierarchy(hw);
        print_roofline_report(st, hw, bw);

    } catch (const std::exception& e) {
        std::println(std::cerr, "Error: {}", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
