/**
 * @file gram_splitk_test.cu
 * @brief Validate and time the split-K Gram against cuBLAS: is it correct, and roofline-bound?
 *
 * Two passes. Correctness compares the split-K G = B^T B against a cuBLAS gemm reference on a
 * small block and reports the worst relative error. Rate times the split-K kernel and cuBLAS
 * dsyrk on a tall block and reports each against the card's FP64 peak and DRAM roof. The claim
 * under test is that split-K reaches the ~s/4 roofline where cuBLAS dsyrk stays latency-bound, so
 * the negative-arm rate becomes readable and s-step's block compute stops dominating.
 *
 * Usage:
 *   ./gram-splitk-test [--machine v100-pcie-16gb] [--device 0] [--n-rate N] [--n-check N]
 *                      [--s-list "1 2 4 6 8"] [--repeats K] [--fp64-tflops F] [--dram-gbs G]
 *                      [--blocks-per-sm B] [--csv PATH]
 *
 * @author Kevin Knights
 * @date 2026-07-25
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
#include "gram_splitk.cuh"

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

[[nodiscard]] double* make_block(int64_t n, int s)
{
    const std::size_t belems = static_cast<std::size_t>(n) * static_cast<std::size_t>(s);
    std::vector<double> h(belems);
    for (int c = 0; c < s; ++c)
        for (int64_t r = 0; r < n; ++r)
            h[static_cast<std::size_t>(c) * static_cast<std::size_t>(n) + static_cast<std::size_t>(r)] =
                std::sin(0.7 * static_cast<double>(c + 1) * static_cast<double>(r) + 1.0);
    double* d_B = nullptr;
    CUDA_CHECK(cudaMalloc(&d_B, belems * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_B, h.data(), belems * sizeof(double), cudaMemcpyHostToDevice));
    return d_B;
}

struct Args {
    std::string machine = "v100-pcie-16gb";
    int         device  = 0;
    int64_t     n_rate  = 1000000;
    int64_t     n_check = 4096;
    int         repeats = 7;
    int         blocks_per_sm = 4;
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
                "Usage: ./gram-splitk-test [--machine KEY] [--device D] [--n-rate N]\n"
                "                          [--n-check N] [--s-list \"1 2 4 8\"] [--repeats K]\n"
                "                          [--fp64-tflops F] [--dram-gbs G] [--blocks-per-sm B]\n"
                "                          [--csv PATH]\n\n"
                "  --n-rate N       rows for the timing pass (default 1e6); large enough to spill\n"
                "                   L2 so the split-K rate is a clean DRAM-roof read.\n"
                "  --n-check N      rows for the correctness pass (default 4096).\n"
                "  --blocks-per-sm B  grid fill for the split-K launch (default 4).\n");
            std::exit(0);
        }
        else { std::fprintf(stderr, "Unknown flag: %s\n", arg.c_str()); std::exit(EXIT_FAILURE); }
    }
    return a;
}

/// Worst relative error of the split-K Gram against a cuBLAS gemm reference on an n_check block.
[[nodiscard]] double check_one(cublasHandle_t blas, int64_t n, int s, int sm_count, int bps)
{
    double* d_B = make_block(n, s);
    double *d_sk = nullptr, *d_ref = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sk,  static_cast<std::size_t>(s) * s * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ref, static_cast<std::size_t>(s) * s * sizeof(double)));

    CUDA_CHECK(gram_splitk(d_B, n, s, d_sk, sm_count, bps));
    const double one = 1.0, zero = 0.0;
    CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, s, s, static_cast<int>(n),
                             &one, d_B, static_cast<int>(n), d_B, static_cast<int>(n),
                             &zero, d_ref, s));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> h_sk(static_cast<std::size_t>(s) * s), h_ref(h_sk.size());
    CUDA_CHECK(cudaMemcpy(h_sk.data(),  d_sk,  h_sk.size()  * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ref.data(), d_ref, h_ref.size() * sizeof(double), cudaMemcpyDeviceToHost));
    double worst = 0.0;
    for (std::size_t k = 0; k < h_sk.size(); ++k) {
        const double denom = std::abs(h_ref[k]) > 0.0 ? std::abs(h_ref[k]) : 1.0;
        worst = std::max(worst, std::abs(h_sk[k] - h_ref[k]) / denom);
    }

    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_sk));
    CUDA_CHECK(cudaFree(d_ref));
    return worst;
}

struct Rate {
    double sk_gflops = 0.0, sk_gbs = 0.0, sk_pct_fp64 = 0.0, sk_pct_roof = 0.0;
    double cublas_gflops = 0.0, speedup = 0.0;
};

[[nodiscard]] Rate rate_one(cublasHandle_t blas, int64_t n, int s, int repeats, int sm_count,
                            int bps, double fp64_peak, double dram_roof)
{
    double* d_B = make_block(n, s);
    double* d_G = nullptr;
    CUDA_CHECK(cudaMalloc(&d_G, static_cast<std::size_t>(s) * s * sizeof(double)));
    const double one = 1.0, zero = 0.0;

    auto sk = [&] { CUDA_CHECK(gram_splitk(d_B, n, s, d_G, sm_count, bps)); };
    auto syrk = [&] {
        CUBLAS_CHECK(cublasDsyrk(blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, s,
                                 static_cast<int>(n), &one, d_B, static_cast<int>(n),
                                 &zero, d_G, s));
    };

    (void)time_it(sk);
    std::vector<double> sks, cbs;
    for (int r = 0; r < repeats; ++r) sks.push_back(time_it(sk));
    (void)time_it(syrk);
    for (int r = 0; r < repeats; ++r) cbs.push_back(time_it(syrk));
    const double t_sk = median(sks), t_cb = median(cbs);

    // Triangle work: both split-K and dsyrk compute the s(s+1)/2 upper elements, so 2 flops per
    // FMA over that triangle. Bytes read B once (8ns), the s x s write is negligible; the syrk
    // intensity is thus ~(s+1)/8, half the full-gemm s/4 the map quotes.
    const double flops = static_cast<double>(n) * static_cast<double>(s) * static_cast<double>(s + 1);
    const double bytes = 8.0 * static_cast<double>(n) * s + 8.0 * static_cast<double>(s) * s;
    Rate rt;
    rt.sk_gflops = t_sk > 0.0 ? flops / t_sk * 1e-9 : 0.0;
    rt.sk_gbs    = t_sk > 0.0 ? bytes / t_sk * 1e-9 : 0.0;
    rt.sk_pct_fp64 = fp64_peak > 0.0 ? rt.sk_gflops * 1e9 / fp64_peak * 100.0 : 0.0;
    rt.sk_pct_roof = dram_roof > 0.0 ? rt.sk_gbs / dram_roof * 100.0 : 0.0;
    rt.cublas_gflops = t_cb > 0.0 ? flops / t_cb * 1e-9 : 0.0;
    rt.speedup = t_sk > 0.0 ? t_cb / t_sk : 0.0;

    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_G));
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

    std::printf("Split-K Gram vs cuBLAS -- correctness and the roofline rate\n");
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

    std::printf("  correctness (n=%lld, vs cuBLAS gemm):\n", static_cast<long long>(a.n_check));
    std::printf("  %4s %14s %8s\n", "s", "max rel err", "verdict");
    std::printf("  %s\n", std::string(30, '-').c_str());
    bool all_ok = true;
    for (const int s : a.s_list) {
        const double err = check_one(blas, a.n_check, s, sm_count, a.blocks_per_sm);
        const bool ok = err < 1e-12;
        all_ok = all_ok && ok;
        std::printf("  %4d %14.2e %8s\n", s, err, ok ? "ok" : "MISMATCH");
    }
    std::printf("  %s\n\n", std::string(30, '-').c_str());

    std::printf("  rate (n=%lld):\n", static_cast<long long>(a.n_rate));
    std::printf("  %4s %11s %8s %8s %11s %10s\n",
                "s", "split-K Gf", "%fp64", "%roof", "cuBLAS Gf", "speedup");
    std::printf("  %s\n", std::string(62, '-').c_str());
    std::vector<Rate> rates;
    for (const int s : a.s_list) {
        const Rate rt = rate_one(blas, a.n_rate, s, a.repeats, sm_count, a.blocks_per_sm,
                                 fp64_peak, dram_roof);
        rates.push_back(rt);
        std::printf("  %4d %11.1f %7.0f%% %7.0f%% %11.1f %9.1fx\n",
                    s, rt.sk_gflops, rt.sk_pct_fp64, rt.sk_pct_roof, rt.cublas_gflops, rt.speedup);
    }
    std::printf("  %s\n\n", std::string(62, '-').c_str());

    if (!all_ok)
        std::printf("  A MISMATCH means the kernel is wrong; do not read the rate.\n");
    if (!may_record(contention))
        std::printf("  Contended device: treat these as a smoke test, not a measurement.\n");

    CUBLAS_CHECK(cublasDestroy(blas));

    if (!a.csv_path.empty()) {
        std::FILE* f = std::fopen(a.csv_path.c_str(), "w");
        if (!f) { std::fprintf(stderr, "Cannot open CSV: %s\n", a.csv_path.c_str()); return 1; }
        std::fprintf(f, "machine,device_name,n_rate,s,sk_gflops,sk_gbs,sk_pct_fp64,sk_pct_roof,"
                        "cublas_gflops,speedup,contended\n");
        for (std::size_t k = 0; k < rates.size(); ++k)
            std::fprintf(f, "%s,\"%s\",%lld,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n",
                         a.machine.c_str(), prop.name, static_cast<long long>(a.n_rate),
                         a.s_list[k], rates[k].sk_gflops, rates[k].sk_gbs, rates[k].sk_pct_fp64,
                         rates[k].sk_pct_roof, rates[k].cublas_gflops, rates[k].speedup,
                         contention.contended ? 1 : 0);
        std::fclose(f);
        std::printf("\n  [Wrote %s]\n", a.csv_path.c_str());
    }
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
