/**
 * @file regime_gpu_mgs.cu
 * @brief Timed modified Gram-Schmidt on the GPU: the reduction-bound half of one Arnoldi cycle.
 *
 * MGS is what the horizontal coordinate charts: 1 + m(m+3)/2 global reductions per cycle, each
 * a dot product the next axpy depends on, so they serialize. This harness times one m-step
 * orthogonalization of an (m+1)-column basis with cuBLAS, the vector kernels a real
 * implementation calls, and reports the achieved bandwidth of the basis re-reads alongside the
 * compute between two reductions, timed as the axpy stream alone.
 *
 * Both are outcomes rather than coordinates: R_h stays predicted from constants, because the
 * predict-then-measure separation forbids a timed run from entering a coordinate. Given a
 * reduction cost, the harness reports its ratio to the measured compute between reductions,
 * which should track the predicted R_h and so tests the model's denominator rather than
 * replacing it.
 *
 * MGS is bandwidth-bound at large n, where the basis re-reads dominate and the rate approaches
 * the DRAM roof once the basis spills L2, and launch-bound at small n, where the serial
 * reductions each pay a cuBLAS call round trip. Values do not affect a timing, so the basis is
 * filled deterministically rather than built from a real operator.
 *
 * Usage:
 *   ./regime-gpu-mgs [--machine v100-pcie-16gb] [--device 0] [--m M]
 *                    [--n-list "1000 8000 61000 227000 705000"] [--repeats K]
 *                    [--dram-gbs G] [--t-reduce-us T] [--csv PATH]
 *
 * @author Kevin Knights
 * @date 2026-07-24
 */

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "gpu_contention.cuh"

namespace {

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        const cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                                    \
            std::fprintf(stderr, "CUDA error %s at %s:%d -- %s\n",                    \
                         cudaGetErrorName(err_), __FILE__, __LINE__,                  \
                         cudaGetErrorString(err_));                                   \
            std::exit(EXIT_FAILURE);                                                  \
        }                                                                             \
    } while (0)

#define CUBLAS_CHECK(call)                                                           \
    do {                                                                             \
        const cublasStatus_t st_ = (call);                                           \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                          \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n",                       \
                         static_cast<int>(st_), __FILE__, __LINE__);                 \
            std::exit(EXIT_FAILURE);                                                 \
        }                                                                            \
    } while (0)

/// Global reductions in one m-step cycle: one norm to start, then per j a dot for each earlier
/// column plus a norm. Mirrors the count the horizontal coordinate is defined on.
[[nodiscard]] int64_t mgs_reductions(int m)
{
    return 1 + static_cast<int64_t>(m) * (m + 3) / 2;
}

/// Bytes cuBLAS actually moves in one cycle. Dot and axpy are separate unfused passes, so per
/// inner (i, j) the dot reads V_i and w (16n). The axpy reads V_i and w and writes w (24n),
/// per column a norm reads w (8n) and a scale reads and writes w (16n). This is what the
/// achieved bandwidth is measured against, and it is larger than a fused kernel would move.
[[nodiscard]] double mgs_bytes(int64_t n, int m)
{
    const double dn = static_cast<double>(n);
    const double dm = static_cast<double>(m);
    const double inner = 40.0 * dn * (dm * (dm + 1.0) / 2.0);
    const double norm_scale = 24.0 * dn * (dm + 1.0);
    return inner + norm_scale;
}

/// FLOPs of the MGS orthogonalization in one cycle.
[[nodiscard]] double mgs_flops(int64_t n, int m)
{
    const double dn = static_cast<double>(n);
    const double dm = static_cast<double>(m);
    return 3.0 * dn + 2.0 * dn * dm * (dm + 1.0) + 3.0 * dn * dm;
}

template <typename F>
[[nodiscard]] double time_it(F&& f)
{
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(s));
    f();
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
    return static_cast<double>(ms) * 1e-3;
}

[[nodiscard]] double median(std::vector<double> v)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct Point {
    int64_t n          = 0;
    double  ws_mib     = 0.0;
    double  rv         = 0.0;
    double  cycle_s    = 0.0;    ///< full MGS orthogonalization, an outcome
    double  gbs        = 0.0;    ///< achieved bandwidth of the basis re-reads
    double  compute_us = 0.0;    ///< compute between two reductions, the axpy isolated
    double  ratio      = 0.0;    ///< reduction cost / compute between reductions, if supplied
    bool    resident   = false;
};

struct Args {
    std::string machine = "v100-pcie-16gb";
    int         device  = 0;
    int         m       = 12;
    int         repeats = 7;
    double      dram_gbs = 0.0;
    double      t_reduce_us = 0.0;   ///< reduction cost for a chosen tier; 0 = skip R_h
    std::vector<int64_t> n_list{1000, 8000, 61000, 227000, 705000,
                                1728000, 4096000, 8000000, 13824000};
    std::string csv_path;
};

[[nodiscard]] std::vector<int64_t> parse_int_list(const std::string& s)
{
    std::vector<int64_t> out;
    std::istringstream in{s};
    int64_t v = 0;
    while (in >> v) out.push_back(v);
    return out;
}

[[nodiscard]] Args parse_args(int argc, char** argv)
{
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) { std::fprintf(stderr, "Missing value for %s\n", arg.c_str());
                               std::exit(EXIT_FAILURE); }
            return argv[i];
        };
        if      (arg == "--machine")     a.machine = next();
        else if (arg == "--device")      a.device  = std::stoi(next());
        else if (arg == "--m")           a.m       = std::stoi(next());
        else if (arg == "--repeats")     a.repeats = std::stoi(next());
        else if (arg == "--dram-gbs")    a.dram_gbs = std::stod(next());
        else if (arg == "--t-reduce-us") a.t_reduce_us = std::stod(next());
        else if (arg == "--n-list")      a.n_list = parse_int_list(next());
        else if (arg == "--csv")         a.csv_path = next();
        else if (arg == "--help") {
            std::printf(
                "Usage: ./regime-gpu-mgs [--machine KEY] [--device D] [--m M]\n"
                "                        [--n-list \"1000 61000 227000\"] [--repeats K]\n"
                "                        [--dram-gbs G] [--t-reduce-us T] [--csv PATH]\n\n"
                "  --m M            Krylov dimension per cycle (default 12).\n"
                "  --n-list         operator dimensions n to sweep.\n"
                "  --dram-gbs G     DRAM roof to report the achieved rate against.\n"
                "  --t-reduce-us T  reduction cost for one tier, in microseconds. When given,\n"
                "                   the ratio T / (compute between reductions) is printed, an\n"
                "                   outcome that should track the predicted R_h.\n");
            std::exit(0);
        }
        else { std::fprintf(stderr, "Unknown flag: %s\n", arg.c_str()); std::exit(EXIT_FAILURE); }
    }
    if (a.m < 1) { std::fprintf(stderr, "--m must be >= 1\n"); std::exit(1); }
    return a;
}

/// Time one m-step MGS on a basis of dimension n, returning the median cycle time.
[[nodiscard]] Point measure(const Args& a, cublasHandle_t blas, int64_t n, int64_t l2_bytes)
{
    const int cols = a.m + 1;
    const std::size_t elems = static_cast<std::size_t>(n) * cols;
    double* d_v  = nullptr;   // the working basis, orthonormalized in place each run
    double* d_v0 = nullptr;   // a pristine device copy, used to reset between timed runs
    CUDA_CHECK(cudaMalloc(&d_v,  elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_v0, elems * sizeof(double)));

    std::vector<double> h_v(elems);
    for (std::size_t i = 0; i < elems; ++i)
        h_v[i] = std::sin(0.7 * static_cast<double>(i) + 1.0);
    CUDA_CHECK(cudaMemcpy(d_v0, h_v.data(), elems * sizeof(double), cudaMemcpyHostToDevice));

    // One column-major basis: column i is d_v + i*n, contiguous.
    auto col = [&](int i) { return d_v + static_cast<std::size_t>(i) * n; };
    const int ni = static_cast<int>(n);

    // Reset is device-to-device and stays outside the timed region: MGS orthonormalizes in
    // place, so each run needs a fresh basis, but a host copy would swamp the timing.
    auto reset = [&] {
        CUDA_CHECK(cudaMemcpy(d_v, d_v0, elems * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    };
    auto mgs = [&] {
        double nrm = 0.0, inv = 0.0, h = 0.0;
        CUBLAS_CHECK(cublasDnrm2(blas, ni, col(0), 1, &nrm));
        inv = 1.0 / nrm;
        CUBLAS_CHECK(cublasDscal(blas, ni, &inv, col(0), 1));
        for (int j = 1; j <= a.m; ++j) {
            double* w = col(j);
            for (int i = 0; i < j; ++i) {
                CUBLAS_CHECK(cublasDdot(blas, ni, col(i), 1, w, 1, &h));
                const double neg = -h;
                CUBLAS_CHECK(cublasDaxpy(blas, ni, &neg, col(i), 1, w, 1));
            }
            CUBLAS_CHECK(cublasDnrm2(blas, ni, w, 1, &nrm));
            inv = 1.0 / nrm;
            CUBLAS_CHECK(cublasDscal(blas, ni, &inv, w, 1));
        }
    };

    // The compute between two reductions is one axpy, w -= h_ij V_i. Timing the axpy stream
    // alone isolates that compute from the reduction and launch overhead the full cycle also
    // carries, and it is what a reduction cost should be compared against. Same axpy count as
    // the inner MGS loop.
    const double scal = 1e-6;
    auto axpy_only = [&] {
        for (int j = 1; j <= a.m; ++j)
            for (int i = 0; i < j; ++i)
                CUBLAS_CHECK(cublasDaxpy(blas, ni, &scal, col(i), 1, col(j), 1));
    };

    reset();
    (void)time_it(mgs);   // warm-up
    std::vector<double> secs;
    secs.reserve(static_cast<std::size_t>(a.repeats));
    for (int r = 0; r < a.repeats; ++r) { reset(); secs.push_back(time_it(mgs)); }
    const double t = median(secs);

    reset();
    (void)time_it(axpy_only);
    std::vector<double> asecs;
    asecs.reserve(static_cast<std::size_t>(a.repeats));
    for (int r = 0; r < a.repeats; ++r) { reset(); asecs.push_back(time_it(axpy_only)); }
    const double t_axpy = median(asecs);

    const double ws = static_cast<double>(cols) * static_cast<double>(n) * 8.0;
    const int64_t R = mgs_reductions(a.m);
    const double compute_per_red = t_axpy / static_cast<double>(R);

    Point p;
    p.n          = n;
    p.ws_mib     = ws / (1024.0 * 1024.0);
    p.rv         = ws / static_cast<double>(l2_bytes);
    p.cycle_s    = t;
    p.gbs        = t > 0.0 ? mgs_bytes(n, a.m) / t * 1e-9 : 0.0;
    p.compute_us = compute_per_red * 1e6;
    p.resident   = ws <= static_cast<double>(l2_bytes);
    if (a.t_reduce_us > 0.0 && compute_per_red > 0.0)
        p.ratio = (a.t_reduce_us * 1e-6) / compute_per_red;

    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_v0));
    return p;
}

}  // namespace

int main(int argc, char** argv)
{
    const Args a = parse_args(argc, argv);
    CUDA_CHECK(cudaSetDevice(a.device));
    const DeviceContention contention = check_device_contention();

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, a.device));

    const double dram_roof = a.dram_gbs > 0.0
        ? a.dram_gbs
        : 2.0 * static_cast<double>(prop.memoryClockRate) * 1e3
              * (static_cast<double>(prop.memoryBusWidth) / 8.0) * 1e-9;

    std::printf("GPU MGS timing -- the reduction-bound half of one Arnoldi cycle\n");
    std::printf("  machine=%s  device=%d (%s)  m=%d  reductions/cycle=%lld\n",
                a.machine.c_str(), a.device, prop.name, a.m, mgs_reductions(a.m));
    std::printf("  L2=%.1f MiB  DRAM roof=%.0f GB/s (%s)\n",
                static_cast<double>(prop.l2CacheSize) / (1024.0 * 1024.0),
                dram_roof, a.dram_gbs > 0.0 ? "measured" : "theoretical");
    report_toolkit();
    report_contention(contention);

    cublasHandle_t blas;
    CUBLAS_CHECK(cublasCreate(&blas));
    CUBLAS_CHECK(cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST));

    // With a reduction cost supplied, the last column is its ratio to the isolated compute
    // between reductions.
    const bool show_ratio = a.t_reduce_us > 0.0;
    std::printf("  %10s %10s %10s %12s %10s %12s%s\n",
                "n", "ws (MiB)", "R_v", "cycle (us)", "GB/s", "compute (us)",
                show_ratio ? "  t_red/comp" : "");
    std::printf("  %s\n", std::string(show_ratio ? 80 : 68, '-').c_str());

    std::vector<Point> pts;
    for (const int64_t n : a.n_list) {
        const Point p = measure(a, blas, n, prop.l2CacheSize);
        pts.push_back(p);
        if (show_ratio)
            std::printf("  %10lld %10.2f %10.4g %12.2f %10.1f %12.4f %11.4g  %s\n",
                        p.n, p.ws_mib, p.rv, p.cycle_s * 1e6, p.gbs, p.compute_us, p.ratio,
                        p.resident ? "L2-resident" : "spilled");
        else
            std::printf("  %10lld %10.2f %10.4g %12.2f %10.1f %12.4f  %s\n",
                        p.n, p.ws_mib, p.rv, p.cycle_s * 1e6, p.gbs, p.compute_us,
                        p.resident ? "L2-resident" : "spilled");
    }
    std::printf("  %s\n\n", std::string(show_ratio ? 80 : 68, '-').c_str());

    // A plateau is only claimed when the last two steps are both flat: one flat step can be a
    // slowing climb rather than a settled rate.
    double g2 = 0.0, g1 = 0.0, g0 = 0.0, ref_peak = 0.0;
    int spilled = 0;
    for (const Point& p : pts) {
        if (p.resident) continue;
        g2 = g1; g1 = g0; g0 = p.gbs;
        ref_peak = std::max(ref_peak, p.gbs);
        ++spilled;
    }
    if (ref_peak > 0.0 && dram_roof > 0.0) {
        const bool plateaued = spilled >= 3 && g0 <= g1 * 1.02 && g1 <= g2 * 1.02;
        std::printf("  spilled MGS peaks at %.0f%% of the DRAM roof and the tail is %s.\n",
                    ref_peak / dram_roof * 100.0, plateaued ? "plateaued" : "still rising");
        if (ref_peak > dram_roof)
            std::printf("  Above the roof because that roof is the STREAM triad (2 reads, 1 write)\n"
                        "  and MGS is dot-heavy (nearer 4 reads per write); HBM sustains reads faster\n"
                        "  than writes, so a read-heavy kernel exceeds the triad while staying under\n"
                        "  the theoretical peak.\n");
    }
    if (show_ratio)
        std::printf("  t_red/comp uses --t-reduce-us=%.3f. It measures the reduction cost against\n"
                    "  the compute between two reductions, and should track the predicted R_h.\n",
                    a.t_reduce_us);
    if (!may_record(contention))
        std::printf("\n  Contended device: treat these as a smoke test, not a measurement.\n");

    CUBLAS_CHECK(cublasDestroy(blas));

    if (!a.csv_path.empty()) {
        std::FILE* f = std::fopen(a.csv_path.c_str(), "w");
        if (!f) { std::fprintf(stderr, "Cannot open CSV: %s\n", a.csv_path.c_str()); return 1; }
        std::fprintf(f, "machine,device_name,m,reductions,n,ws_mib,rv,dram_roof_gbs,"
                        "cycle_s,gbs,compute_us_per_red,t_reduce_us,t_red_over_compute,"
                        "resident,contended\n");
        for (const Point& p : pts)
            std::fprintf(f, "%s,\"%s\",%d,%lld,%lld,%.4f,%.6f,%.4f,%.9e,%.4f,%.6f,%.4f,%.6f,%d,%d\n",
                         a.machine.c_str(), prop.name, a.m, mgs_reductions(a.m), p.n,
                         p.ws_mib, p.rv, dram_roof, p.cycle_s, p.gbs, p.compute_us,
                         a.t_reduce_us, p.ratio, p.resident ? 1 : 0, contention.contended ? 1 : 0);
        std::fclose(f);
        std::printf("\n  [Wrote %s]\n", a.csv_path.c_str());
    }
    return EXIT_SUCCESS;
}
