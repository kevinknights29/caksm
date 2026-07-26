/**
 * @file regime_gpu_spmv.cu
 * @brief Timed baseline SpMV on the GPU: does it fall onto the DRAM roof as R_v crosses 1?
 *
 * The vertical mechanism pays only where baseline SpMV is DRAM-bound, and this harness tests
 * that precondition. It times an m-application Krylov chain y <- A y across a grid sweep and
 * reports the achieved bandwidth against the measured DRAM roof. Below R_v = 1 the operator is
 * cache-resident and the rate should sit near the L2 roof; above it the operator spills and the
 * rate should plateau at the DRAM roof, which is where cache-blocked matrix-powers has traffic
 * to convert.
 *
 * Two arms. A scalar-per-row CSR kernel is the naive baseline, and cuSPARSE is the reference a
 * practitioner would actually run. The crossover must be read against the reference, or a weak
 * kernel would be mistaken for an operator that cannot saturate bandwidth.
 *
 * Usage:
 *   ./regime-gpu-spmv [--machine v100-pcie-16gb] [--device 0] [--m M] [--dim D]
 *                     [--n1-list "31 45 61 74 89"] [--repeats K] [--dram-gbs G] [--csv PATH]
 *
 * @author Kevin Knights
 * @date 2026-07-24
 */

#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <utility>
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

#define CUSPARSE_CHECK(call)                                                          \
    do {                                                                              \
        const cusparseStatus_t st_ = (call);                                          \
        if (st_ != CUSPARSE_STATUS_SUCCESS) {                                         \
            std::fprintf(stderr, "cuSPARSE error %d at %s:%d\n",                      \
                         static_cast<int>(st_), __FILE__, __LINE__);                  \
            std::exit(EXIT_FAILURE);                                                  \
        }                                                                             \
    } while (0)

/// CSR bytes: 8 B value + 4 B column index per nonzero, 4 B row pointer per row. Matches the
/// accounting the host-side byte model uses so the two are directly comparable.
constexpr double kValIdxBytes = 12.0;

/// One row per thread. Seven nonzeros per interior row makes a scalar row cheap, and the gather
/// x[colidx[k]] over a banded operator hits contiguous neighbors.
__global__ void spmv_csr(const int* __restrict__ rowptr, const int* __restrict__ colidx,
                         const double* __restrict__ val, const double* __restrict__ x,
                         double* __restrict__ y, int n)
{
    const int r = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                + static_cast<int>(threadIdx.x);
    if (r >= n) return;
    double acc = 0.0;
    for (int k = rowptr[r]; k < rowptr[r + 1]; ++k)
        acc += val[k] * x[colidx[k]];
    y[r] = acc;
}

struct Csr {
    std::vector<int>    rowptr;
    std::vector<int>    colidx;
    std::vector<double> val;
    int64_t n   = 0;
    int64_t nnz = 0;
};

/// A (2*dim+1)-point stencil on an n1^dim grid in natural ordering. Diagonal 2*dim, off-diagonal
/// -1, which is enough for a bandwidth measurement: the values do not enter the byte count.
[[nodiscard]] Csr build_stencil(int n1, int dim)
{
    int64_t stride[3] = {1, 1, 1};
    for (int d = 1; d < dim; ++d) stride[d] = stride[d - 1] * n1;
    int64_t n = 1;
    for (int d = 0; d < dim; ++d) n *= n1;

    Csr a;
    a.n = n;
    a.rowptr.reserve(static_cast<std::size_t>(n) + 1);
    a.rowptr.push_back(0);

    for (int64_t r = 0; r < n; ++r) {
        a.colidx.push_back(static_cast<int>(r));
        a.val.push_back(2.0 * dim);
        for (int d = 0; d < dim; ++d) {
            const int64_t coord = (r / stride[d]) % n1;
            if (coord > 0)      { a.colidx.push_back(static_cast<int>(r - stride[d])); a.val.push_back(-1.0); }
            if (coord < n1 - 1) { a.colidx.push_back(static_cast<int>(r + stride[d])); a.val.push_back(-1.0); }
        }
        a.rowptr.push_back(static_cast<int>(a.colidx.size()));
    }
    a.nnz = static_cast<int64_t>(a.colidx.size());
    return a;
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

/// Bytes one SpMV moves: the operator streamed once, x read once (banded reuse), y written once.
[[nodiscard]] double spmv_bytes(int64_t nnz, int64_t n)
{
    const double matrix = kValIdxBytes * static_cast<double>(nnz)
                        + 4.0 * static_cast<double>(n + 1);
    return matrix + 8.0 * static_cast<double>(n) + 8.0 * static_cast<double>(n);
}

struct Point {
    int     n1            = 0;
    int64_t n             = 0;
    int64_t nnz           = 0;
    double  ws_mib        = 0.0;
    double  rv            = 0.0;
    double  gbs_naive     = 0.0;   ///< achieved effective bandwidth, scalar-per-row kernel
    double  gflops_naive  = 0.0;
    double  gbs_ref       = 0.0;   ///< achieved effective bandwidth, cuSPARSE
    double  gflops_ref    = 0.0;
    bool    resident      = false;
};

struct Args {
    std::string machine = "v100-pcie-16gb";
    int         device  = 0;
    int         m       = 12;
    int         dim     = 3;
    int         repeats = 7;
    double      dram_gbs = 0.0;   ///< the DRAM roof to compare against; 0 = derive from device
    std::vector<int> n1_list{31, 45, 61, 74, 89, 120, 160, 200, 240, 280, 320};
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
        if      (arg == "--machine")  a.machine = next();
        else if (arg == "--device")   a.device  = std::stoi(next());
        else if (arg == "--m")        a.m       = std::stoi(next());
        else if (arg == "--dim")      a.dim     = std::stoi(next());
        else if (arg == "--repeats")  a.repeats = std::stoi(next());
        else if (arg == "--dram-gbs") a.dram_gbs = std::stod(next());
        else if (arg == "--n1-list")  a.n1_list = parse_int_list(next());
        else if (arg == "--csv")      a.csv_path = next();
        else if (arg == "--help") {
            std::printf(
                "Usage: ./regime-gpu-spmv [--machine KEY] [--device D] [--m M] [--dim D]\n"
                "                         [--n1-list \"31 45 61\"] [--repeats K]\n"
                "                         [--dram-gbs G] [--csv PATH]\n\n"
                "  --m M         SpMV applications per timed chain (default 12).\n"
                "  --dim D       spatial dimensions; sets the stencil (default 3).\n"
                "  --dram-gbs G  DRAM roof to report the achieved rate against. Pass the value\n"
                "                gpu-stream measured; 0 derives a theoretical roof from clocks.\n");
            std::exit(0);
        }
        else { std::fprintf(stderr, "Unknown flag: %s\n", arg.c_str()); std::exit(EXIT_FAILURE); }
    }
    if (a.m < 1)   { std::fprintf(stderr, "--m must be >= 1\n"); std::exit(1); }
    if (a.dim < 1 || a.dim > 3) { std::fprintf(stderr, "--dim must be 1, 2 or 3\n"); std::exit(1); }
    return a;
}

/// Median achieved rate over `repeats` timings of a chain, given the per-application byte and
/// flop counts. `chain` runs one full m-application Krylov chain.
template <typename Chain>
void time_chain(Chain&& chain, double bytes, double flops, int repeats,
                double& gbs, double& gflops)
{
    (void)time_it(chain);   // warm-up: first touch and the clock ramp
    CUDA_CHECK(cudaGetLastError());
    std::vector<double> g_bw, g_fl;
    g_bw.reserve(static_cast<std::size_t>(repeats));
    g_fl.reserve(static_cast<std::size_t>(repeats));
    for (int r = 0; r < repeats; ++r) {
        const double t = time_it(chain);
        g_bw.push_back(t > 0.0 ? bytes / t * 1e-9 : 0.0);
        g_fl.push_back(t > 0.0 ? flops / t * 1e-9 : 0.0);
    }
    gbs    = median(g_bw);
    gflops = median(g_fl);
}

/// Time the naive kernel and cuSPARSE on one grid, returning both achieved rates.
[[nodiscard]] Point measure(const Args& a, cusparseHandle_t sparse, int n1, int64_t l2_bytes)
{
    const Csr h = build_stencil(n1, a.dim);

    int *d_rowptr = nullptr, *d_colidx = nullptr;
    double *d_val = nullptr, *d_x = nullptr, *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rowptr, h.rowptr.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colidx, h.colidx.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_val,    h.val.size()    * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x, static_cast<std::size_t>(h.n) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, static_cast<std::size_t>(h.n) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_rowptr, h.rowptr.data(), h.rowptr.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colidx, h.colidx.data(), h.colidx.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_val, h.val.data(), h.val.size() * sizeof(double),
                          cudaMemcpyHostToDevice));

    std::vector<double> x0(static_cast<std::size_t>(h.n), 1.0);
    CUDA_CHECK(cudaMemcpy(d_x, x0.data(), x0.size() * sizeof(double), cudaMemcpyHostToDevice));

    const double bytes = spmv_bytes(h.nnz, h.n) * a.m;
    const double flops = 2.0 * static_cast<double>(h.nnz) * a.m;

    // Naive arm: scalar per row, ping-ponging the two vectors.
    const int threads = 256;
    const int blocks  = static_cast<int>((h.n + threads - 1) / threads);
    auto naive_chain = [&] {
        double* in  = d_x;
        double* out = d_y;
        for (int k = 0; k < a.m; ++k) {
            spmv_csr<<<blocks, threads>>>(d_rowptr, d_colidx, d_val, in, out,
                                          static_cast<int>(h.n));
            std::swap(in, out);
        }
    };

    // Reference arm: cuSPARSE csrmv. The matrix descriptor is fixed; only the x and y vector
    // descriptors alternate to build the chain.
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    CUSPARSE_CHECK(cusparseCreateCsr(&matA, h.n, h.n, h.nnz, d_rowptr, d_colidx, d_val,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, h.n, d_x, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, h.n, d_y, CUDA_R_64F));

    const double alpha = 1.0, beta = 0.0;
    std::size_t buf_size = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                           CUSPARSE_SPMV_ALG_DEFAULT, &buf_size));
    void* d_buf = nullptr;
    if (buf_size > 0) CUDA_CHECK(cudaMalloc(&d_buf, buf_size));

    auto ref_chain = [&] {
        cusparseDnVecDescr_t in = vecX, out = vecY;
        for (int k = 0; k < a.m; ++k) {
            CUSPARSE_CHECK(cusparseSpMV(sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, matA, in, &beta, out, CUDA_R_64F,
                                        CUSPARSE_SPMV_ALG_DEFAULT, d_buf));
            std::swap(in, out);
        }
    };

    Point p;
    p.n1       = n1;
    p.n        = h.n;
    p.nnz      = h.nnz;
    p.ws_mib   = spmv_bytes(h.nnz, h.n) / (1024.0 * 1024.0);
    p.rv       = spmv_bytes(h.nnz, h.n) / static_cast<double>(l2_bytes);
    p.resident = spmv_bytes(h.nnz, h.n) <= static_cast<double>(l2_bytes);
    time_chain(naive_chain, bytes, flops, a.repeats, p.gbs_naive, p.gflops_naive);
    time_chain(ref_chain,   bytes, flops, a.repeats, p.gbs_ref,   p.gflops_ref);

    CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
    if (d_buf) CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFree(d_rowptr));
    CUDA_CHECK(cudaFree(d_colidx));
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
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

    std::printf("GPU SpMV roofline -- is the baseline DRAM-bound where R_v > 1?\n");
    std::printf("  machine=%s  device=%d (%s)  m=%d  dim=%d\n",
                a.machine.c_str(), a.device, prop.name, a.m, a.dim);
    std::printf("  L2=%.1f MiB  DRAM roof=%.0f GB/s (%s)\n",
                static_cast<double>(prop.l2CacheSize) / (1024.0 * 1024.0),
                dram_roof, a.dram_gbs > 0.0 ? "measured" : "theoretical");
    report_toolkit();
    report_contention(contention);

    cusparseHandle_t sparse;
    CUSPARSE_CHECK(cusparseCreate(&sparse));

    std::printf("  %5s %12s %10s %10s   %8s %7s   %8s %7s\n",
                "n1", "n", "ws (MiB)", "R_v",
                "naive", "% roof", "cuSPARSE", "% roof");
    std::printf("  %s\n", std::string(74, '-').c_str());

    std::vector<Point> pts;
    for (const int n1 : a.n1_list) {
        const Point p = measure(a, sparse, n1, prop.l2CacheSize);
        pts.push_back(p);
        std::printf("  %5d %12lld %10.2f %10.4g   %8.1f %6.0f%%   %8.1f %6.0f%%  %s\n",
                    p.n1, p.n, p.ws_mib, p.rv,
                    p.gbs_naive, dram_roof > 0.0 ? p.gbs_naive / dram_roof * 100.0 : 0.0,
                    p.gbs_ref,   dram_roof > 0.0 ? p.gbs_ref   / dram_roof * 100.0 : 0.0,
                    p.resident ? "L2-resident" : "spilled");
    }
    std::printf("  %s\n\n", std::string(74, '-').c_str());

    // The reading uses the cuSPARSE reference. As R_v grows the operator stream dominates, so a
    // bandwidth-bound SpMV climbs toward the DRAM roof and then flattens; a latency-bound one
    // never approaches it. A plateau is only claimed when the last two steps are both flat: one
    // flat step can just be a slowing climb, not a settled rate.
    double g2 = 0.0, g1 = 0.0, g0 = 0.0, ref_peak = 0.0;   // last three spilled rates
    int spilled = 0;
    for (const Point& p : pts) {
        if (p.resident) continue;
        g2 = g1; g1 = g0; g0 = p.gbs_ref;
        ref_peak = std::max(ref_peak, p.gbs_ref);
        ++spilled;
    }
    if (ref_peak > 0.0 && dram_roof > 0.0) {
        const double peak_frac = ref_peak / dram_roof;
        const bool plateaued = spilled >= 3 && g0 <= g1 * 1.02 && g1 <= g2 * 1.02;
        std::printf("  spilled cuSPARSE peaks at %.0f%% of the DRAM roof and the tail is %s.\n",
                    peak_frac * 100.0, plateaued ? "plateaued" : "still rising");
        if (!plateaued)
            std::printf("  READING: still climbing at the largest grid. Extend --n1-list until two\n"
                        "           consecutive steps are flat before reading the plateau.\n");
        else if (peak_frac > 0.7)
            std::printf("  READING: plateaus near the DRAM roof, so baseline SpMV is DRAM-bound\n"
                        "           above R_v = 1 and the vertical crossover is a real target.\n");
        else
            std::printf("  READING: plateaus below the roof, so SpMV on this operator is gather- or\n"
                        "           latency-bound. A tiled arm would have little traffic to convert;\n"
                        "           report the boundary, not a reached crossover.\n");
    }
    if (!may_record(contention))
        std::printf("\n  Contended device: treat these as a smoke test, not a measurement.\n");

    CUSPARSE_CHECK(cusparseDestroy(sparse));

    if (!a.csv_path.empty()) {
        std::FILE* f = std::fopen(a.csv_path.c_str(), "w");
        if (!f) { std::fprintf(stderr, "Cannot open CSV: %s\n", a.csv_path.c_str()); return 1; }
        std::fprintf(f, "machine,device_name,m,dim,n1,n,nnz,ws_mib,rv,dram_roof_gbs,"
                        "gbs_naive,gflops_naive,gbs_cusparse,gflops_cusparse,resident,contended\n");
        for (const Point& p : pts)
            std::fprintf(f, "%s,\"%s\",%d,%d,%d,%lld,%lld,%.4f,%.6f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d\n",
                         a.machine.c_str(), prop.name, a.m, a.dim, p.n1, p.n, p.nnz,
                         p.ws_mib, p.rv, dram_roof, p.gbs_naive, p.gflops_naive,
                         p.gbs_ref, p.gflops_ref, p.resident ? 1 : 0,
                         contention.contended ? 1 : 0);
        std::fclose(f);
        std::printf("\n  [Wrote %s]\n", a.csv_path.c_str());
    }
    return EXIT_SUCCESS;
}
