/**
 * @file fma_loop.cpp
 * @brief Peak FP64 throughput microbenchmark using tight FMA loops.
 *
 * Measures the compute ceiling for the roofline model by running N independent
 * FMA accumulator chains — the "far-right" kernel whose only bottleneck is the
 * floating-point units themselves.
 *
 * Two ISA paths, selected at compile time via -march=native:
 *
 *   AVX-512 (Intel Ice Lake / Skylake-SP):
 *     Register: ZMM (512-bit, 8 fp64/reg)   Budget: 32 ZMM total
 *     Units:    2 FMA/cycle, 4-cycle latency => need >= 8 independent chains
 *     Theory:   2 x 8 x 2 = 32 FLOP/cycle   ~64-70 GFLOP/s at AVX-512 clock
 *     N_MAX:    24  (24+2 accumulators+operands = 26 of 32 ZMM)
 *
 *   AVX2+FMA (AMD Zen 2 / Zen 3, or Intel Haswell+):
 *     Register: YMM (256-bit, 4 fp64/reg)   Budget: 16 YMM total
 *     Units:    2 FMA/cycle, 5-cycle latency => need >= 10 independent chains
 *     Theory:   2 x 4 x 2 = 16 FLOP/cycle   ~48-60 GFLOP/s at boost clock
 *     N_MAX:    14  (14+2 = 16 of 16 YMM)
 *
 * Design:
 *   Register-blocked: N independent chains break the latency-throughput trap.
 *   Tight:            only FMAs in the inner loop; operands stay in registers.
 *   Anti-DCE:         accumulators are reduced to a volatile sink after timing.
 *   Numeric safety:   mul=0.9999999, add=1e-7 => fixed point a*=1.0, no overflow
 *                     or denormals; compiler cannot constant-fold the loop.
 *
 * FLOP accounting:
 *   total_FLOPs = fma_count x LANES x 2   (LANES = 8 for AVX-512, 4 for AVX2)
 *
 * Usage:
 *   taskset -c 2 ./build/fma-loop [--acc N] [--warmup T] [--run T]
 *
 * @author Kevin Knights
 * @date 2026-06-27
 */

// Architecture detection and abstraction
#if defined(__AVX512F__)

#include <immintrin.h>
using Vec = __m512d;
static constexpr int LANES = 8;

static Vec    vec_set1(double x)              { return _mm512_set1_pd(x); }
static Vec    vec_fmadd(Vec a, Vec b, Vec c)  { return _mm512_fmadd_pd(a, b, c); }
static double vec_hsum(Vec v)                 { return _mm512_reduce_add_pd(v); }

static constexpr const char* kArchLabel =
    "Intel AVX-512  (ZMM, 8 fp64/reg, 2 FMA units => 32 FLOP/cycle)";
static constexpr const char* kTheoryNote =
    "Expected ceiling: ~64-70 GFLOP/s (AVX-512 clock ~2.0-2.2 GHz)";

#elif defined(__AVX2__) && defined(__FMA__)

#include <immintrin.h>
using Vec = __m256d;
static constexpr int LANES = 4;

static Vec  vec_set1(double x)             { return _mm256_set1_pd(x); }
static Vec  vec_fmadd(Vec a, Vec b, Vec c) { return _mm256_fmadd_pd(a, b, c); }

// AVX2 has no horizontal-reduce intrinsic; split into two 128-bit halves
static double vec_hsum(Vec v) {
    __m128d lo = _mm256_castpd256_pd128(v);          // lower 128 bits (no instruction)
    __m128d hi = _mm256_extractf128_pd(v, 1);        // upper 128 bits
    __m128d s  = _mm_add_pd(lo, hi);                 // [a+c, b+d]
    return _mm_cvtsd_f64(_mm_add_pd(s, _mm_shuffle_pd(s, s, 1))); // (a+c)+(b+d)
}

static constexpr const char* kArchLabel =
    "AMD AVX2+FMA   (YMM, 4 fp64/reg, 2 FMA units => 16 FLOP/cycle)";
static constexpr const char* kTheoryNote =
    "Expected ceiling: ~48-60 GFLOP/s (boost clock up to 3.8 GHz on Threadripper 3960X)";

#else
#error "fma_loop requires AVX-512F or AVX2+FMA. Build with -march=native on a supported x86-64 CPU."
#endif

// Includes
#include <chrono>
#include <cstdlib>
#include <iostream>
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

// Kernel
//
// N independent FMA chains with no cross-chain dependencies.
// Each chain: a[i] = fmadd(a[i], mul, add)  i.e.  a[i] = a[i]*mul + add
//
// With N >= latency*ports chains the FMA units stay fully occupied.
//   AVX-512: need >= 4*2 = 8   (N_MAX=24 comfortably covers this)
//   AVX2:    need >= 5*2 = 10  (N_MAX=14 comfortably covers this)
//
// Register budget: N (accumulators) + 1 (mul) + 1 (add) = N+2.

template <int N>
double fma_gflops(double warmup_s, double run_s)
{
    const Vec mul = vec_set1(0.9999999);   // multiplier < 1: values converge to 1.0
    const Vec add = vec_set1(1e-7);        // addend: fixed point a* = 1e-7/1e-7 = 1.0

    Vec a[N];
    for (int i = 0; i < N; ++i)
        a[i] = vec_set1(1.0 + 1e-3 * i);  // distinct starting values, all normal

    // Warmup: drive the core to its FMA frequency before the timed window
    {
        auto t0 = Clock::now();
        while (elapsed_s(t0) < warmup_s)
            for (int i = 0; i < N; ++i)
                a[i] = vec_fmadd(a[i], mul, add);
    }

    // Timed run: BLOCK inner iterations between time checks amortize the
    // ~20 ns overhead of Clock::now() to well under 1% of FMA time.
    constexpr int BLOCK = 1000;
    long long fma_count = 0;
    const auto t0 = Clock::now();
    while (elapsed_s(t0) < run_s) {
        for (int b = 0; b < BLOCK; ++b)
            for (int i = 0; i < N; ++i)    // N is a compile-time constant;
                a[i] = vec_fmadd(a[i], mul, add); // GCC -O3 fully unrolls this
        fma_count += static_cast<long long>(BLOCK) * N;
    }
    const double elapsed = elapsed_s(t0);

    // Anti-DCE: force all accumulators to be materialized
    double s = 0.0;
    for (int i = 0; i < N; ++i)
        s += vec_hsum(a[i]);
    volatile double sink = s;
    (void)sink;

    // LANES doubles x 2 FLOPs per FMA instruction
    return static_cast<double>(fma_count) * (LANES * 2) / elapsed / 1e9;
}

// Dispatch table
using MeasureFn = double (*)(double, double);
struct Entry { int n; MeasureFn fn; };

// kAll: every N from 1 to N_MAX so --acc accepts any value in that range.
// AVX-512 registers: 32 ZMM  => cap at 24 (24+2 = 26, leaving 6 spare)
// AVX2    registers: 16 YMM  => cap at 14 (14+2 = 16, exactly fills all)
static const Entry kAll[] = {
    { 1,  fma_gflops< 1>}, { 2,  fma_gflops< 2>}, { 3,  fma_gflops< 3>},
    { 4,  fma_gflops< 4>}, { 5,  fma_gflops< 5>}, { 6,  fma_gflops< 6>},
    { 7,  fma_gflops< 7>}, { 8,  fma_gflops< 8>}, { 9,  fma_gflops< 9>},
    {10,  fma_gflops<10>}, {11,  fma_gflops<11>}, {12,  fma_gflops<12>},
    {13,  fma_gflops<13>}, {14,  fma_gflops<14>},
#if defined(__AVX512F__)
    {15,  fma_gflops<15>}, {16,  fma_gflops<16>}, {17,  fma_gflops<17>},
    {18,  fma_gflops<18>}, {19,  fma_gflops<19>}, {20,  fma_gflops<20>},
    {21,  fma_gflops<21>}, {22,  fma_gflops<22>}, {23,  fma_gflops<23>},
    {24,  fma_gflops<24>},
#endif
};
static constexpr int N_TABLE = static_cast<int>(sizeof(kAll) / sizeof(kAll[0]));

// Default sparse sweep: covers latency-bound, transitional, and saturated
// regimes without running every N in [1, N_TABLE].
#if defined(__AVX512F__)
static constexpr int kSweepNs[] = {1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24};
#else
static constexpr int kSweepNs[] = {1, 2, 4, 6, 8, 10, 12, 14};
#endif

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
            std::println("  --acc N      Single measurement with N_ACC = N  (1 <= N <= {})", N_TABLE);
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

        std::println("FMA peak FP64 throughput -- {}", kArchLabel);
        std::println("{}", kTheoryNote);
        std::println("Tip: pin to one core for stable results:  taskset -c 2 ./build/fma-loop\n");

        if (cfg.acc != 0) {
            if (cfg.acc < 1 || cfg.acc > N_TABLE)
                throw std::invalid_argument(
                    "--acc must be in [1, " + std::to_string(N_TABLE) + "]");
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
