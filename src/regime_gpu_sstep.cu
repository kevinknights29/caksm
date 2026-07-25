/**
 * @file regime_gpu_sstep.cu
 * @brief Timed s-step orthogonalization: the communication-avoiding treatment MGS is baseline for.
 *
 * s-step orthogonalizes a block of s Krylov vectors at once with CholeskyQR: form the Gram
 * matrix G = B^T B, factor it, and solve Q = B R^-1, repeated once for O(u) orthogonality
 * (CholQR2). One reduction per block of s, against the m(m+3)/2 of MGS.
 *
 * Two measurements. The modeled horizontal crossover: each method's local compute plus its
 * reduction count times a calibrated reduction cost, so the cost at which s-step overtakes MGS
 * locates the crossover against the predicted R_h = 1. And the Gram block's FP64 rate, whose
 * ~s/4 intensity places it compute-bound on a throttled card and memory-bound on a datacenter
 * one.
 *
 * The basis is checked for orthogonality: a fast basis that is not orthonormal is no win.
 *
 * Usage:
 *   ./regime-gpu-sstep [--machine v100-pcie-16gb] [--device 0] [--m M]
 *                      [--n-list "61000 227000 705000"] [--s-list "1 2 4 6 8"]
 *                      [--repeats K] [--fp64-tflops F] [--dram-gbs G] [--t-reduce-us T]
 *                      [--csv PATH]
 *
 * @author Kevin Knights
 * @date 2026-07-24
 */

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

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

#define CUSOLVER_CHECK(call)                                                         \
    do {                                                                             \
        const cusolverStatus_t st_ = (call);                                         \
        if (st_ != CUSOLVER_STATUS_SUCCESS) {                                        \
            std::fprintf(stderr, "cuSOLVER error %d at %s:%d\n",                     \
                         static_cast<int>(st_), __FILE__, __LINE__);                 \
            std::exit(EXIT_FAILURE);                                                 \
        }                                                                            \
    } while (0)

/// Global reductions for MGS: one norm, then per j a dot for each earlier column plus a norm.
[[nodiscard]] int64_t mgs_reductions(int m)
{
    return 1 + static_cast<int64_t>(m) * (m + 3) / 2;
}

/// Global reductions for s-step: one to start, then per block a Gram reduce, doubled for the
/// CholQR2 re-orthogonalization.
[[nodiscard]] int64_t ca_reductions(int m, int s)
{
    const int64_t blocks = (m + s - 1) / s;
    return 1 + blocks * 2;
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

struct Args {
    std::string machine = "v100-pcie-16gb";
    int         device  = 0;
    int         m       = 12;
    int         repeats = 7;
    double      fp64_tflops = 0.0;
    double      dram_gbs    = 0.0;
    double      t_reduce_us = 0.0;
    std::vector<int64_t> n_list{61000, 227000, 705000};
    std::vector<int>     s_list{1, 2, 4, 6, 8};
    int         s_max   = 9;   // certified block-width cap: kappa(B) <= u^-1/2 holds below it,
                               // the largest s the monomial basis carries on the Laplacian scaffold
    int64_t     gram_n  = 20000;  // rows per block for the batched Gram rate, small on purpose,
                                  // the ~s/4 intensity is n-independent so a modest n suffices
    int         gram_batch = 512; // independent blocks contracted at once, to raise occupancy
    std::string csv_path;
};

template <typename T>
[[nodiscard]] std::vector<T> parse_list(const std::string& s)
{
    std::vector<T> out;
    std::istringstream in{s};
    double v = 0.0;
    while (in >> v) out.push_back(static_cast<T>(v));
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
        else if (arg == "--fp64-tflops") a.fp64_tflops = std::stod(next());
        else if (arg == "--dram-gbs")    a.dram_gbs = std::stod(next());
        else if (arg == "--t-reduce-us") a.t_reduce_us = std::stod(next());
        else if (arg == "--s-max")       a.s_max = std::stoi(next());
        else if (arg == "--gram-n")      a.gram_n = std::stoll(next());
        else if (arg == "--gram-batch")  a.gram_batch = std::stoi(next());
        else if (arg == "--n-list")      a.n_list = parse_list<int64_t>(next());
        else if (arg == "--s-list")      a.s_list = parse_list<int>(next());
        else if (arg == "--csv")         a.csv_path = next();
        else if (arg == "--help") {
            std::printf(
                "Usage: ./regime-gpu-sstep [--machine KEY] [--device D] [--m M]\n"
                "                          [--n-list \"61000 227000\"] [--s-list \"1 2 4 8\"]\n"
                "                          [--repeats K] [--fp64-tflops F] [--dram-gbs G]\n"
                "                          [--t-reduce-us T] [--csv PATH]\n\n"
                "  --fp64-tflops F  measured FP64 peak, for the Gram block's compute roof.\n"
                "  --dram-gbs G     DRAM roof, for the Gram block's memory roof.\n"
                "  --s-max S        certified block-width cap (default 9); s beyond it is\n"
                "                   dropped, since the block loses orthogonality there.\n"
                "  --gram-n N       rows per block for the batched Gram rate (default 20000).\n"
                "  --gram-batch B   independent blocks contracted at once (default 512), to raise\n"
                "                   the occupancy of the Gram-rate measurement.\n"
                "  --t-reduce-us T  reduction cost at one tier. The crossover against MGS is\n"
                "                   compared to it and to the point where R_h crosses 1.\n");
            std::exit(0);
        }
        else { std::fprintf(stderr, "Unknown flag: %s\n", arg.c_str()); std::exit(EXIT_FAILURE); }
    }
    if (a.m < 1)     { std::fprintf(stderr, "--m must be >= 1\n"); std::exit(1); }
    if (a.s_max < 1) { std::fprintf(stderr, "--s-max must be >= 1\n"); std::exit(1); }
    return a;
}

struct GramRate {
    double gflops   = 0.0;
    double gbs      = 0.0;   ///< achieved effective bandwidth against the block footprint
    double pct_fp64 = 0.0;
    double pct_roof = 0.0;
    double ai       = 0.0;   ///< arithmetic intensity, ~s/4 for s << n
};

/// The negative-arm measurement: the Gram block's achieved FP64 rate, isolated from the block.
///
/// One tall-skinny G = B^T B has only s x s outputs, too few threads to hide memory latency, so
/// timing it alone is latency-bound. Contracting nb independent blocks with a strided-batched
/// gemm raises occupancy; cuBLAS still serializes the batch at s >= 2, so the rate is a floor,
/// not the roofline, and the verdict rests on the intensity (~s/4), not the achieved rate. The
/// intensity is n-independent, so a modest n keeps memory bounded. Values do not matter: only the
/// contraction is timed, no factorization.
[[nodiscard]] GramRate measure_gram_rate(cublasHandle_t blas, int s, int64_t n, int nb,
                                         int repeats, double fp64_peak, double dram_roof)
{
    const int ni = static_cast<int>(n);
    const std::size_t belems = static_cast<std::size_t>(n) * static_cast<std::size_t>(s)
                             * static_cast<std::size_t>(nb);
    double *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_b, belems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_c, static_cast<std::size_t>(s) * static_cast<std::size_t>(s)
                                    * static_cast<std::size_t>(nb) * sizeof(double)));

    std::vector<double> h_b(belems);
    for (std::size_t i = 0; i < belems; ++i)
        h_b[i] = std::sin(0.7 * static_cast<double>(i) + 1.0);
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), belems * sizeof(double), cudaMemcpyHostToDevice));

    const double one = 1.0, zero = 0.0;
    const long long strideB = static_cast<long long>(n) * s;
    const long long strideC = static_cast<long long>(s) * s;
    // C_i = B_i^T B_i: op(A)=T gives s x n, op(B)=N gives n x s, output s x s. A and B are the
    // same block; the blocks are distinct memory, so L2 reuse across the batch cannot inflate it.
    auto batched = [&] {
        CUBLAS_CHECK(cublasDgemmStridedBatched(
            blas, CUBLAS_OP_T, CUBLAS_OP_N, s, s, ni,
            &one, d_b, ni, strideB, d_b, ni, strideB,
            &zero, d_c, s, strideC, nb));
    };
    (void)time_it(batched);
    std::vector<double> secs;
    secs.reserve(static_cast<std::size_t>(repeats));
    for (int r = 0; r < repeats; ++r) secs.push_back(time_it(batched));
    const double t = median(secs);

    const double dnb   = static_cast<double>(nb);
    const double flops = 2.0 * static_cast<double>(n) * s * s * dnb;
    const double bytes = (8.0 * static_cast<double>(n) * s + 8.0 * static_cast<double>(s) * s) * dnb;
    GramRate g;
    g.gflops   = t > 0.0 ? flops / t * 1e-9 : 0.0;
    g.gbs      = t > 0.0 ? bytes / t * 1e-9 : 0.0;
    g.pct_fp64 = fp64_peak > 0.0 ? g.gflops * 1e9 / fp64_peak * 100.0 : 0.0;
    g.pct_roof = dram_roof > 0.0 ? g.gbs / dram_roof * 100.0 : 0.0;
    g.ai       = bytes > 0.0 ? flops / bytes : 0.0;

    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return g;
}

/// Full MGS orthogonalization time on an (m+1)-column basis: the baseline local compute.
[[nodiscard]] double measure_mgs(cublasHandle_t blas, int m, int64_t n, int repeats)
{
    const int cols = m + 1;
    const std::size_t elems = static_cast<std::size_t>(n) * cols;
    double *d_v = nullptr, *d_v0 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_v,  elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_v0, elems * sizeof(double)));
    std::vector<double> h_v(elems);
    for (std::size_t i = 0; i < elems; ++i)
        h_v[i] = std::sin(0.7 * static_cast<double>(i) + 1.0);
    CUDA_CHECK(cudaMemcpy(d_v0, h_v.data(), elems * sizeof(double), cudaMemcpyHostToDevice));

    const int ni = static_cast<int>(n);
    auto col = [&](int i) { return d_v + static_cast<std::size_t>(i) * n; };
    auto reset = [&] {
        CUDA_CHECK(cudaMemcpy(d_v, d_v0, elems * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    };
    auto mgs = [&] {
        double nrm = 0.0, inv = 0.0, h = 0.0;
        CUBLAS_CHECK(cublasDnrm2(blas, ni, col(0), 1, &nrm));
        inv = 1.0 / nrm;
        CUBLAS_CHECK(cublasDscal(blas, ni, &inv, col(0), 1));
        for (int j = 1; j <= m; ++j) {
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
    reset();
    (void)time_it(mgs);
    std::vector<double> secs;
    for (int r = 0; r < repeats; ++r) { reset(); secs.push_back(time_it(mgs)); }

    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_v0));
    return median(secs);
}

struct SstepResult {
    double block_s     = 0.0;   ///< time for one CholQR2 block of width s
    double ortho       = 0.0;   ///< ||I - Q^T Q|| after CholQR2
    bool   potrf_ok    = true;
};

/// One CholQR2 block of width s on an n x s block, timed, with the orthogonality of the result
/// measured. The Gram rate is measured separately and saturated (see measure_gram_rate).
[[nodiscard]] SstepResult measure_sstep(cublasHandle_t blas, cusolverDnHandle_t solver,
                                        int64_t n, int s, int repeats)
{
    const int ni = static_cast<int>(n);
    const std::size_t belems = static_cast<std::size_t>(n) * s;
    double *d_b = nullptr, *d_b0 = nullptr, *d_g = nullptr;
    int    *d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_b,  belems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b0, belems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_g,  static_cast<std::size_t>(s) * s * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    // Fill the block full rank: each column a distinct-frequency sinusoid. A single running index
    // (as in the MGS harness) makes every column a phase shift of one sinusoid, so the block spans
    // only {sin, cos} and is rank 2; the Gram is then singular for s > 2 and CholQR2 divides by a
    // zero pivot. Distinct frequencies keep it well conditioned, so the check measures CholQR2,
    // not the fill.
    std::vector<double> h_b(belems);
    for (int c = 0; c < s; ++c)
        for (int64_t r = 0; r < n; ++r)
            h_b[static_cast<std::size_t>(c) * static_cast<std::size_t>(n) + static_cast<std::size_t>(r)] =
                std::sin(0.7 * static_cast<double>(c + 1) * static_cast<double>(r) + 1.0);
    CUDA_CHECK(cudaMemcpy(d_b0, h_b.data(), belems * sizeof(double), cudaMemcpyHostToDevice));

    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER, s, d_g, s, &lwork));
    double* d_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_work, static_cast<std::size_t>(std::max(lwork, 1)) * sizeof(double)));

    const double one = 1.0, zero = 0.0;
    auto reset = [&] {
        CUDA_CHECK(cudaMemcpy(d_b, d_b0, belems * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    };
    // One CholeskyQR pass: G = B^T B, G = R^T R, B <- B R^-1.
    auto cholqr = [&] {
        CUBLAS_CHECK(cublasDsyrk(blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, s, ni,
                                 &one, d_b, ni, &zero, d_g, s));
        CUSOLVER_CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_UPPER, s, d_g, s,
                                        d_work, lwork, d_info));
        CUBLAS_CHECK(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                                 CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, ni, s,
                                 &one, d_g, s, d_b, ni));
    };
    auto cholqr2 = [&] { cholqr(); cholqr(); };
    auto gram_only = [&] {
        CUBLAS_CHECK(cublasDsyrk(blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, s, ni,
                                 &one, d_b, ni, &zero, d_g, s));
    };

    reset();
    (void)time_it(cholqr2);
    std::vector<double> block;
    block.reserve(static_cast<std::size_t>(repeats));
    for (int r = 0; r < repeats; ++r) { reset(); block.push_back(time_it(cholqr2)); }

    // Orthogonality and breakdown on a fresh block: run CholQR2, then form Q^T Q and compare to
    // the identity on the host, and read the Cholesky status.
    reset();
    cholqr2();
    int info = 0;
    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    gram_only();   // Q^T Q into d_g
    std::vector<double> h_g(static_cast<std::size_t>(s) * s);
    CUDA_CHECK(cudaMemcpy(h_g.data(), d_g, h_g.size() * sizeof(double), cudaMemcpyDeviceToHost));
    double err = 0.0;
    for (int j = 0; j < s; ++j)
        for (int i = 0; i <= j; ++i) {   // upper triangle filled by dsyrk
            const double e = h_g[static_cast<std::size_t>(j) * s + i] - (i == j ? 1.0 : 0.0);
            err += (i == j ? 1.0 : 2.0) * e * e;
        }

    SstepResult res;
    res.block_s  = median(block);
    res.ortho    = std::sqrt(err);
    res.potrf_ok = info == 0;

    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_b0));
    CUDA_CHECK(cudaFree(d_g));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));
    return res;
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
    const double fp64_peak = a.fp64_tflops * 1e12;   // FLOP/s; 0 if not supplied

    std::printf("GPU s-step orthogonalization -- the treatment MGS is baseline for\n");
    std::printf("  machine=%s  device=%d (%s)  m=%d\n",
                a.machine.c_str(), a.device, prop.name, a.m);
    std::printf("  FP64 peak=%.2f TFLOP/s  DRAM roof=%.0f GB/s  ridge=%.2f FLOP/B\n",
                a.fp64_tflops, dram_roof,
                fp64_peak > 0.0 && dram_roof > 0.0 ? fp64_peak / (dram_roof * 1e9) : 0.0);

    // Cap the sweep at the certified block width. Beyond it the monomial basis crosses the
    // kappa(B) <= u^-1/2 certificate and the block is no longer orthogonalizable, so a wider s
    // would time a correct method against a broken one. Report the cap.
    std::vector<int> s_list;
    for (const int s : a.s_list)
        if (s >= 1 && s <= a.s_max) s_list.push_back(s);
    if (s_list.empty()) {
        std::fprintf(stderr, "No s in --s-list falls within 1..%d (--s-max)\n", a.s_max);
        return EXIT_FAILURE;
    }
    const std::size_t dropped = a.s_list.size() - s_list.size();
    std::printf("  s sweep capped at s_max=%d (certified block width)", a.s_max);
    if (dropped > 0) std::printf(", %zu value(s) above it dropped", dropped);
    std::printf("\n");
    report_toolkit();
    report_contention(contention);

    cublasHandle_t blas;
    cusolverDnHandle_t solver;
    CUBLAS_CHECK(cublasCreate(&blas));
    CUSOLVER_CHECK(cusolverDnCreate(&solver));

    // The negative-arm reading. The verdict is the intensity against the ridge (the predicted
    // column); the achieved rate is a batched floor, since cuBLAS serializes the tall-skinny Gram
    // at s >= 2. See measure_gram_rate.
    const double ridge = fp64_peak > 0.0 && dram_roof > 0.0 ? fp64_peak / (dram_roof * 1e9) : 0.0;
    std::vector<GramRate> grates;
    grates.reserve(s_list.size());
    std::printf("  Gram FP64 rate (batched: %d independent %lld x s blocks contracted at once)\n",
                a.gram_batch, static_cast<long long>(a.gram_n));
    std::printf("  %4s %8s %8s %11s %8s %8s   %s\n",
                "s", "AI", "ridge", "GFLOP/s", "%fp64", "%roof", "predicted");
    std::printf("  %s\n", std::string(70, '-').c_str());
    for (const int s : s_list) {
        const GramRate g = measure_gram_rate(blas, s, a.gram_n, a.gram_batch, a.repeats,
                                             fp64_peak, dram_roof);
        grates.push_back(g);
        std::printf("  %4d %8.2f %8.2f %11.1f %7.0f%% %7.0f%%   %s\n",
                    s, g.ai, ridge, g.gflops, g.pct_fp64, g.pct_roof,
                    ridge > 0.0 ? (g.ai > ridge ? "compute-bound" : "memory-bound") : "no roof");
    }
    std::printf("  %s\n", std::string(70, '-').c_str());
    // A measurement caveat, not a verdict: cuBLAS serializes the tall-skinny batch at s >= 2, so
    // the achieved rate is a floor. The predicted column carries the roofline call.
    std::printf("  (achieved GFLOP/s is a cuBLAS floor for s>=2, not the roofline)\n\n");

    auto gram_for = [&](int s) -> GramRate {
        for (std::size_t i = 0; i < s_list.size(); ++i)
            if (s_list[i] == s) return grates[i];
        return GramRate{};
    };

    const bool show_cross = a.t_reduce_us > 0.0;

    struct Row {
        int64_t n; int s; int64_t blocks; int64_t r_ca;
        double sstep_us; double gram_gflops; double gram_pct_fp64; double gram_pct_roof;
        double ortho; bool potrf_ok; double t_cross_us; double rh1_us;
    };
    std::vector<Row> rows;

    for (const int64_t n : a.n_list) {
        const double mgs_local = measure_mgs(blas, a.m, n, a.repeats);
        const int64_t r_mgs = mgs_reductions(a.m);
        const double rh1_us = mgs_local / static_cast<double>(r_mgs) * 1e6;   // t_reduce at R_h=1

        std::printf("  n=%lld  MGS baseline: local %.1f us, R=%lld  (R_h crosses 1 at "
                    "t_reduce=%.3f us)\n", n, mgs_local * 1e6, r_mgs, rh1_us);
        std::printf("  %4s %8s %6s %11s %11s %10s%s\n",
                    "s", "blocks", "R_ca", "s-step (us)", "ortho", "potrf",
                    show_cross ? "   t_cross" : "");
        std::printf("  %s\n", std::string(show_cross ? 67 : 57, '-').c_str());

        for (const int s : s_list) {
            const SstepResult r = measure_sstep(blas, solver, n, s, a.repeats);
            const int64_t blocks = (a.m + s - 1) / s;
            const int64_t r_ca   = ca_reductions(a.m, s);
            const double sstep_local = static_cast<double>(blocks) * r.block_s;
            const GramRate g = gram_for(s);   // n-independent, from the saturated measurement above

            // Crossover: sstep_local + R_ca*t = mgs_local + R_mgs*t. Below t_cross MGS wins,
            // above it s-step does. Negative means s-step wins even with free reductions.
            const double denom = static_cast<double>(r_mgs - r_ca);
            const double t_cross_us = denom != 0.0
                ? (sstep_local - mgs_local) / denom * 1e6 : 0.0;

            rows.push_back({n, s, blocks, r_ca, sstep_local * 1e6, g.gflops,
                            g.pct_fp64, g.pct_roof, r.ortho, r.potrf_ok, t_cross_us, rh1_us});

            if (show_cross)
                std::printf("  %4d %8lld %6lld %11.2f %11.2e %10s %9.3f\n",
                            s, blocks, r_ca, sstep_local * 1e6, r.ortho,
                            r.potrf_ok ? "ok" : "FAILED", t_cross_us);
            else
                std::printf("  %4d %8lld %6lld %11.2f %11.2e %10s\n",
                            s, blocks, r_ca, sstep_local * 1e6, r.ortho,
                            r.potrf_ok ? "ok" : "FAILED");
        }
        std::printf("  %s\n\n", std::string(show_cross ? 67 : 57, '-').c_str());
    }

    // The breakdown reading: the largest block width that both factored and stayed orthonormal,
    // against the certified cap. Reaching the cap means no breakdown was seen in range; a smaller
    // value means cuSOLVER potrf gives out before the certificate says it should.
    constexpr double kOrthoTol = 1e-8;   // O(u) ~ 1e-15 under CholQR2; well clear of a broken block
    int s_ortho_max = 0;
    for (const Row& r : rows)
        if (r.potrf_ok && r.ortho < kOrthoTol && r.s > s_ortho_max) s_ortho_max = r.s;
    std::printf("  largest s that factored and stayed orthonormal: %d (certified cap %d). %s\n",
                s_ortho_max, a.s_max,
                s_ortho_max >= s_list.back()
                    ? "No breakdown up to the swept cap."
                    : "Breakdown below the cap: potrf gives out before the certificate.");

    if (!may_record(contention))
        std::printf("\n  Contended device: treat these as a smoke test, not a measurement.\n");

    CUBLAS_CHECK(cublasDestroy(blas));
    CUSOLVER_CHECK(cusolverDnDestroy(solver));

    if (!a.csv_path.empty()) {
        std::FILE* f = std::fopen(a.csv_path.c_str(), "w");
        if (!f) { std::fprintf(stderr, "Cannot open CSV: %s\n", a.csv_path.c_str()); return 1; }
        std::fprintf(f, "machine,device_name,m,n,s,blocks,r_ca,r_mgs,sstep_us,gram_gflops,"
                        "gram_pct_fp64,gram_pct_roof,ortho,potrf_ok,t_cross_us,rh1_us,contended\n");
        for (const Row& r : rows)
            std::fprintf(f, "%s,\"%s\",%d,%lld,%d,%lld,%lld,%lld,%.4f,%.4f,%.4f,%.4f,%.6e,%d,"
                            "%.6f,%.6f,%d\n",
                         a.machine.c_str(), prop.name, a.m, r.n, r.s, r.blocks, r.r_ca,
                         mgs_reductions(a.m), r.sstep_us, r.gram_gflops, r.gram_pct_fp64,
                         r.gram_pct_roof, r.ortho, r.potrf_ok ? 1 : 0, r.t_cross_us, r.rh1_us,
                         contention.contended ? 1 : 0);
        std::fclose(f);
        std::printf("\n  [Wrote %s]\n", a.csv_path.c_str());
    }
    return EXIT_SUCCESS;
}
