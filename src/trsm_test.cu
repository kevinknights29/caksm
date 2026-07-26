/**
 * @file trsm_test.cu
 * @brief Validate and time the tall-skinny trsm against cuBLAS: is it correct, and memory-bound?
 *
 * Correctness compares the in-place X R = B solve against cuBLAS on a small tall block.
 * Rate times both on a tall block and reports each against the card's DRAM roof.
 * The claim under test is that the per-row solve reaches the memory roofline where cuBLAS
 * dtrsm stays compute-bound at a tiny fraction of useful FP64, so the triangular solve
 * stops dominating s-step's block compute.
 *
 * Usage:
 *   ./trsm-test [--machine v100-pcie-16gb] [--device 0] [--n-rate N] [--n-check N]
 *               [--s-list "1 2 4 6 8"] [--repeats K] [--fp64-tflops F] [--dram-gbs G]
 *               [--blocks-per-sm B] [--csv PATH]
 *
 * @author Kevin Knights
 * @date 2026-07-26
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
#include "trsm_tallskinny.cuh"

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

/// A well-conditioned s x s upper-triangular R (column-major), diagonal-dominant so the solve
/// does not amplify roundoff and the comparison against cuBLAS is clean.
[[nodiscard]] double* make_R(int s)
{
    std::vector<double> h(static_cast<std::size_t>(s) * s, 0.0);
    for (int j = 0; j < s; ++j)
        for (int i = 0; i <= j; ++i)
            h[static_cast<std::size_t>(j) * s + i] = (i == j) ? (2.0 + j) : 0.3;
    double* d_R = nullptr;
    CUDA_CHECK(cudaMalloc(&d_R, h.size() * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_R, h.data(), h.size() * sizeof(double), cudaMemcpyHostToDevice));
    return d_R;
}

/// A tall B (m x s, column-major), filled deterministically. Returns a device copy.
[[nodiscard]] double* make_B(int64_t m, int s)
{
    const std::size_t elems = static_cast<std::size_t>(m) * static_cast<std::size_t>(s);
    std::vector<double> h(elems);
    for (int c = 0; c < s; ++c)
        for (int64_t r = 0; r < m; ++r)
            h[static_cast<std::size_t>(c) * static_cast<std::size_t>(m) + static_cast<std::size_t>(r)] =
                std::sin(0.7 * static_cast<double>(c + 1) * static_cast<double>(r) + 1.0);
    double* d_B = nullptr;
    CUDA_CHECK(cudaMalloc(&d_B, elems * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_B, h.data(), elems * sizeof(double), cudaMemcpyHostToDevice));
    return d_B;
}

struct Args {
    std::string machine = "v100-pcie-16gb";
    int         device  = 0;
    int64_t     n_rate  = 1000000;
    int64_t     n_check = 4096;
    int         repeats = 7;
    int         blocks_per_sm = 8;
    double      fp64_tflops = 0.0;
    double      dram_gbs    = 0.0;
    std::vector<int> s_list{1, 2, 4, 6, 8};
    std::string csv_path;
};

[[nodiscard]] std::vector<int> parse_int_list(const std::string& s)
{
    std::vector<int> out;
    std::istringstream in{s};
    int v = 0;
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
        if      (arg == "--machine")       a.machine = next();
        else if (arg == "--device")        a.device  = std::stoi(next());
        else if (arg == "--n-rate")        a.n_rate  = std::stoll(next());
        else if (arg == "--n-check")       a.n_check = std::stoll(next());
        else if (arg == "--repeats")       a.repeats = std::stoi(next());
        else if (arg == "--blocks-per-sm") a.blocks_per_sm = std::stoi(next());
        else if (arg == "--fp64-tflops")   a.fp64_tflops = std::stod(next());
        else if (arg == "--dram-gbs")      a.dram_gbs = std::stod(next());
        else if (arg == "--s-list")        a.s_list = parse_int_list(next());
        else if (arg == "--csv")           a.csv_path = next();
        else if (arg == "--help") {
            std::printf(
                "Usage: ./trsm-test [--machine KEY] [--device D] [--n-rate N] [--n-check N]\n"
                "                   [--s-list \"1 2 4 8\"] [--repeats K] [--fp64-tflops F]\n"
                "                   [--dram-gbs G] [--blocks-per-sm B] [--csv PATH]\n\n"
                "  --n-rate N       rows for the timing pass (default 1e6).\n"
                "  --n-check N      rows for the correctness pass (default 4096).\n");
            std::exit(0);
        }
        else { std::fprintf(stderr, "Unknown flag: %s\n", arg.c_str()); std::exit(EXIT_FAILURE); }
    }
    return a;
}

/// Worst relative error of the per-row solve against cuBLAS dtrsm on an n_check block.
[[nodiscard]] double check_one(cublasHandle_t blas, int64_t m, int s, int sm_count, int bps)
{
    double* d_R  = make_R(s);
    double* d_B0 = make_B(m, s);
    const std::size_t elems = static_cast<std::size_t>(m) * static_cast<std::size_t>(s);
    double *d_mine = nullptr, *d_ref = nullptr;
    CUDA_CHECK(cudaMalloc(&d_mine, elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ref,  elems * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_mine, d_B0, elems * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_ref,  d_B0, elems * sizeof(double), cudaMemcpyDeviceToDevice));

    CUDA_CHECK(trsm_tallskinny(d_R, d_mine, m, s, sm_count, bps));
    const double one = 1.0;
    CUBLAS_CHECK(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                             CUBLAS_DIAG_NON_UNIT, static_cast<int>(m), s, &one, d_R, s,
                             d_ref, static_cast<int>(m)));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> h_mine(elems), h_ref(elems);
    CUDA_CHECK(cudaMemcpy(h_mine.data(), d_mine, elems * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ref.data(),  d_ref,  elems * sizeof(double), cudaMemcpyDeviceToHost));
    double worst = 0.0;
    for (std::size_t k = 0; k < elems; ++k) {
        const double denom = std::abs(h_ref[k]) > 0.0 ? std::abs(h_ref[k]) : 1.0;
        worst = std::max(worst, std::abs(h_mine[k] - h_ref[k]) / denom);
    }

    CUDA_CHECK(cudaFree(d_R));
    CUDA_CHECK(cudaFree(d_B0));
    CUDA_CHECK(cudaFree(d_mine));
    CUDA_CHECK(cudaFree(d_ref));
    return worst;
}

struct Rate {
    double mine_gflops = 0.0, mine_gbs = 0.0, mine_pct_fp64 = 0.0, mine_pct_roof = 0.0;
    double cublas_gflops = 0.0, speedup = 0.0;
};

[[nodiscard]] Rate rate_one(cublasHandle_t blas, int64_t m, int s, int repeats, int sm_count,
                            int bps, double fp64_peak, double dram_roof)
{
    double* d_R = make_R(s);
    double* d_B = make_B(m, s);   // solved in place, values drift but the cost is value-independent
    const double one = 1.0;

    auto mine = [&] { CUDA_CHECK(trsm_tallskinny(d_R, d_B, m, s, sm_count, bps)); };
    auto ref  = [&] {
        CUBLAS_CHECK(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                 CUBLAS_DIAG_NON_UNIT, static_cast<int>(m), s, &one, d_R, s,
                                 d_B, static_cast<int>(m)));
    };

    (void)time_it(mine);
    std::vector<double> ms, cbs;
    for (int r = 0; r < repeats; ++r) ms.push_back(time_it(mine));
    (void)time_it(ref);
    for (int r = 0; r < repeats; ++r) cbs.push_back(time_it(ref));
    const double t_mine = median(ms), t_cb = median(cbs);

    // Forward substitution: s(s+1)/2 mul-adds + s divides per row ~ s(s+1) flops. B read and
    // written once, so bytes = 16 m s. Intensity ~(s+1)/16, below both cards' FP64 ridges.
    const double flops = static_cast<double>(m) * static_cast<double>(s) * static_cast<double>(s + 1);
    const double bytes = 16.0 * static_cast<double>(m) * s + 8.0 * static_cast<double>(s) * s;
    Rate rt;
    rt.mine_gflops = t_mine > 0.0 ? flops / t_mine * 1e-9 : 0.0;
    rt.mine_gbs    = t_mine > 0.0 ? bytes / t_mine * 1e-9 : 0.0;
    rt.mine_pct_fp64 = fp64_peak > 0.0 ? rt.mine_gflops * 1e9 / fp64_peak * 100.0 : 0.0;
    rt.mine_pct_roof = dram_roof > 0.0 ? rt.mine_gbs / dram_roof * 100.0 : 0.0;
    rt.cublas_gflops = t_cb > 0.0 ? flops / t_cb * 1e-9 : 0.0;
    rt.speedup = t_mine > 0.0 ? t_cb / t_mine : 0.0;

    CUDA_CHECK(cudaFree(d_R));
    CUDA_CHECK(cudaFree(d_B));
    return rt;
}

}  // namespace

int main(int argc, char** argv)
{
    const Args a = parse_args(argc, argv);
    CUDA_CHECK(cudaSetDevice(a.device));
    const DeviceContention contention = check_device_contention();

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, a.device));
    const int sm_count = prop.multiProcessorCount;

    const double dram_roof = a.dram_gbs > 0.0
        ? a.dram_gbs
        : 2.0 * static_cast<double>(prop.memoryClockRate) * 1e3
              * (static_cast<double>(prop.memoryBusWidth) / 8.0) * 1e-9;
    const double fp64_peak = a.fp64_tflops * 1e12;

    std::printf("Tall-skinny trsm vs cuBLAS -- correctness and the memory-roofline rate\n");
    std::printf("  machine=%s  device=%d (%s)  SMs=%d  blocks/SM=%d\n",
                a.machine.c_str(), a.device, prop.name, sm_count, a.blocks_per_sm);
    std::printf("  FP64 peak=%.2f TFLOP/s  DRAM roof=%.0f GB/s  ridge=%.2f FLOP/B\n",
                a.fp64_tflops, dram_roof,
                fp64_peak > 0.0 && dram_roof > 0.0 ? fp64_peak / (dram_roof * 1e9) : 0.0);
    report_toolkit();
    report_contention(contention);

    cublasHandle_t blas;
    CUBLAS_CHECK(cublasCreate(&blas));
    CUBLAS_CHECK(cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST));

    std::printf("  correctness (n=%lld, vs cuBLAS dtrsm):\n", static_cast<long long>(a.n_check));
    std::printf("  %4s %14s %8s\n", "s", "max rel err", "verdict");
    std::printf("  %s\n", std::string(30, '-').c_str());
    bool all_ok = true;
    for (const int s : a.s_list) {
        const double err = check_one(blas, a.n_check, s, sm_count, a.blocks_per_sm);
        const bool ok = err < 1e-11;
        all_ok = all_ok && ok;
        std::printf("  %4d %14.2e %8s\n", s, err, ok ? "ok" : "MISMATCH");
    }
    std::printf("  %s\n\n", std::string(30, '-').c_str());

    std::printf("  rate (n=%lld):\n", static_cast<long long>(a.n_rate));
    std::printf("  %4s %11s %8s %8s %11s %10s\n",
                "s", "trsm Gf", "%fp64", "%roof", "cuBLAS Gf", "speedup");
    std::printf("  %s\n", std::string(62, '-').c_str());
    std::vector<Rate> rates;
    for (const int s : a.s_list) {
        const Rate rt = rate_one(blas, a.n_rate, s, a.repeats, sm_count, a.blocks_per_sm,
                                 fp64_peak, dram_roof);
        rates.push_back(rt);
        std::printf("  %4d %11.1f %7.0f%% %7.0f%% %11.1f %9.1fx\n",
                    s, rt.mine_gflops, rt.mine_pct_fp64, rt.mine_pct_roof, rt.cublas_gflops,
                    rt.speedup);
    }
    std::printf("  %s\n\n", std::string(62, '-').c_str());

    if (!all_ok)
        std::printf("  A MISMATCH means the solve is wrong; do not read the rate.\n");
    if (!may_record(contention))
        std::printf("  Contended device: treat these as a smoke test, not a measurement.\n");

    CUBLAS_CHECK(cublasDestroy(blas));

    if (!a.csv_path.empty()) {
        std::FILE* f = std::fopen(a.csv_path.c_str(), "w");
        if (!f) { std::fprintf(stderr, "Cannot open CSV: %s\n", a.csv_path.c_str()); return 1; }
        std::fprintf(f, "machine,device_name,n_rate,s,trsm_gflops,trsm_gbs,trsm_pct_fp64,"
                        "trsm_pct_roof,cublas_gflops,speedup,contended\n");
        for (std::size_t k = 0; k < rates.size(); ++k)
            std::fprintf(f, "%s,\"%s\",%lld,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n",
                         a.machine.c_str(), prop.name, static_cast<long long>(a.n_rate),
                         a.s_list[k], rates[k].mine_gflops, rates[k].mine_gbs,
                         rates[k].mine_pct_fp64, rates[k].mine_pct_roof, rates[k].cublas_gflops,
                         rates[k].speedup, contention.contended ? 1 : 0);
        std::fclose(f);
        std::printf("\n  [Wrote %s]\n", a.csv_path.c_str());
    }
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
