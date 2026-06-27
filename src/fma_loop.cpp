/**
 * @file fma_loop.cpp
 * @brief Peak FP64 throughput microbenchmark using tight AVX-512 FMA loops.
 *
 * Measures the compute ceiling for the roofline model by running N independent
 * AVX-512 FMA accumulator chains, the "far-right" kernel whose only bottleneck
 * is the floating-point units themselves.
 *
 * Design:
 *   FMA:              vfmadd213pd on 512-bit ZMM registers (8 doubles, 2 FLOPs each)
 *   Register-blocked: N independent chains overcome the latency wall.
 *                      Ice Lake: 4-cycle latency x 2 FMA units => need >= 8 chains
 *                      to saturate.  Sweep reveals the plateau empirically.
 *   Tight:            nothing in the inner loop but FMAs, operands stay in ZMM
 *                      registers (no loads or stores, infinite arithmetic intensity).
 *   Anti-DCE:         accumulators are reduced into a volatile sink at the end so
 *                      the compiler cannot eliminate the loop as dead code.
 *
 * FLOP accounting:
 *   total_FLOPs = fma_count x 8 doubles x 2 FLOPs = fma_count x 16
 *
 * Expected ceiling (Xeon Platinum 8358, AVX-512 clock ~2.0-2.2 GHz):
 *   2 units x 8 doubles x 2 FLOPs/cycle x ~2.1 GHz approx 67-70 GFLOP/s
 *   (AVX-512 throttles the core below the 2.60 GHz base that is the honest
 *   ceiling for AVX-512 kernels, so the measured value is what matters.)
 *
 * Usage:
 *   taskset -c 2 ./build/fma-loop [--acc N] [--warmup T] [--run T]
 *
 * @author Kevin Knights
 * @date 2026-06-27
 */

#ifndef __AVX512F__
#error "fma_loop requires AVX-512F. Build with -march=native on Skylake-SP / Ice Lake or newer."
#endif

#include <immintrin.h>

#include <chrono>
#include <cstdlib>
#include <print>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

using Clock = std::chrono::steady_clock;
using Sec   = std::chrono::duration<double>;

static double elapsed_s(Clock::time_point t0)
{
    return Sec(Clock::now() - t0).count();
}

// N independent AVX-512 FMA chains with no cross-chain data dependencies.
//
// Each chain: a[i] = fma(a[i], mul, add)
//
// mul = 0.9999999, add = 1e-7 --> fixed point a* = add / (1 - mul) = 1.0.
// Values converge geometrically to 1.0: no overflow, no denormals, and the
// compiler cannot constant-fold because the loop bound is not static.
//
// Register budget: N ZMM (accumulators) + 1 ZMM (mul) + 1 ZMM (add) = N+2.
// AVX-512 provides 32 architectural ZMM registers, N <= 24 leaves 6 spare for
// loop counters and return-address / frame overhead.
template <int N>
double fma_gflops(double warmup_s, double run_s)
{
    const __m512d mul = _mm512_set1_pd(0.9999999);
    const __m512d add = _mm512_set1_pd(1e-7);

    __m512d a[N];
    for (int i = 0; i < N; ++i)
        a[i] = _mm512_set1_pd(1.0 + 1e-3 * i); // distinct, normal starting values

    // Warmup: let the core ramp to its AVX-512 frequency before timing begins
    {
        auto t0 = Clock::now();
        while (elapsed_s(t0) < warmup_s)
            for (int i = 0; i < N; ++i)
                a[i] = _mm512_fmadd_pd(a[i], mul, add);
    }

    // Timed run
    // BLOCK inner iterations between each chrono call amortize the ~20 ns
    // overhead of Clock::now() to well under 1% of total FMA time
    constexpr int BLOCK = 1000;
    long long fma_count = 0;
    const auto t0 = Clock::now();
    while (elapsed_s(t0) < run_s) {
        for (int b = 0; b < BLOCK; ++b)
            for (int i = 0; i < N; ++i)                        // N is compile-time constant
                a[i] = _mm512_fmadd_pd(a[i], mul, add); // GCC -O3 fully unrolls
        fma_count += static_cast<long long>(BLOCK) * N;
    }
    const double elapsed = elapsed_s(t0);

    // Anti-DCE: force every accumulator value to be materialized
    double s = 0.0;
    for (int i = 0; i < N; ++i)
        s += _mm512_reduce_add_pd(a[i]);
    volatile double sink = s;
    (void)sink;

    // Each vfmadd512pd: 8 doubles x 2 FLOPs = 16 FLOPs
    return static_cast<double>(fma_count) * 16.0 / elapsed / 1e9;
}

// Dispatch table
using MeasureFn = double (*)(double, double);
struct Entry { int n; MeasureFn fn; };

// All N in [1, 24] so --acc accepts any value in that range
static const Entry kAll[] = {
    { 1,  fma_gflops< 1>}, { 2,  fma_gflops< 2>}, { 3,  fma_gflops< 3>},
    { 4,  fma_gflops< 4>}, { 5,  fma_gflops< 5>}, { 6,  fma_gflops< 6>},
    { 7,  fma_gflops< 7>}, { 8,  fma_gflops< 8>}, { 9,  fma_gflops< 9>},
    {10,  fma_gflops<10>}, {11,  fma_gflops<11>}, {12,  fma_gflops<12>},
    {13,  fma_gflops<13>}, {14,  fma_gflops<14>}, {15,  fma_gflops<15>},
    {16,  fma_gflops<16>}, {17,  fma_gflops<17>}, {18,  fma_gflops<18>},
    {19,  fma_gflops<19>}, {20,  fma_gflops<20>}, {21,  fma_gflops<21>},
    {22,  fma_gflops<22>}, {23,  fma_gflops<23>}, {24,  fma_gflops<24>},
};

// Sparse sweep for the default (no --acc) run, covers the latency-bound,
// transitional, and saturated regimes without running all 24 configs
static constexpr int kSweepNs[] = {1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24};

// CLI
struct Config {
    int    acc      = 0;   // 0 => sparse sweep
    double warmup_s = 0.2;
    double run_s    = 1.0;
};

static Config parse_args(std::span<const char* const> args)
{
    Config cfg;
    for (std::size_t i = 1; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        auto next = [&]() -> std::string_view {
            if (++i >= args.size())
                throw std::invalid_argument("Missing value for " + std::string(arg));
            return args[i];
        };
        if      (arg == "--acc")    cfg.acc      = std::stoi(std::string(next()));
        else if (arg == "--warmup") cfg.warmup_s = std::stod(std::string(next()));
        else if (arg == "--run")    cfg.run_s    = std::stod(std::string(next()));
        else if (arg == "--help") {
            std::println("Usage: taskset -c 2 ./build/fma-loop [OPTIONS]");
            std::println("  --acc N      Single measurement with N_ACC = N  (1 <= N <= 24)");
            std::println("  --warmup T   Warmup per measurement in seconds (default 0.2)");
            std::println("  --run T      Timed run per measurement in seconds (default 1.0)");
            std::exit(0);
        }
        else throw std::invalid_argument("Unknown flag: " + std::string(arg));
    }
    return cfg;
}

int main(int argc, char* argv[])
{
    try {
        const Config cfg = parse_args(
            std::span<const char* const>(argv, static_cast<std::size_t>(argc)));

        std::println("AVX-512 FMA peak FP64 throughput -- Intel Xeon Platinum 8358 (Ice Lake)");
        std::println("Theory: 2 FMA units x 8 doubles x 2 FLOPs/cycle = 32 FLOP/cycle");
        std::println("Tip: pin to one core for stable results:  taskset -c 2 ./build/fma-loop\n");

        if (cfg.acc != 0) {
            if (cfg.acc < 1 || cfg.acc > 24)
                throw std::invalid_argument("--acc must be in [1, 24]");
            const double gf = kAll[cfg.acc - 1].fn(cfg.warmup_s, cfg.run_s);
            std::println("N_ACC = {}  ->  {:.2f} GFLOP/s", cfg.acc, gf);
        } else {
            std::println("  {:>6}  {:>12}", "N_ACC", "GFLOP/s");
            std::println("  {}", std::string(22, '-'));

            double peak = 0.0;
            for (const int n : kSweepNs) {
                const double gf = kAll[n - 1].fn(cfg.warmup_s, cfg.run_s);
                if (gf > peak) peak = gf;
                std::println("  {:>6}  {:>12.2f}", n, gf);
            }
            std::println("  {}", std::string(22, '-'));
            std::println("  {:>6}  {:>12.2f}  <-- ceiling", "peak", peak);
            std::println("\n  Use {:.1f} GFLOP/s as peak_gflops in profiler.cpp.", peak);
        }

    } catch (const std::exception& e) {
        std::println(std::cerr, "Error: {}", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
