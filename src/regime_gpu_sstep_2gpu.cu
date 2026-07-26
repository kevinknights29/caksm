/**
 * @file regime_gpu_sstep_2gpu.cu
 * @brief Two-GPU measured horizontal crossover: MGS against s-step with real inter-GPU reductions.
 *
 * The single-GPU harness models the crossover, adding R * t_reduce to each method's local
 * compute. This one measures it. The n basis rows are split across both local V100s, so every
 * MGS dot and norm, and every s-step Gram, becomes a real NCCL all-reduce across the PCIe/UPI
 * link. At the DEVICE_P2P rung the reduction is expensive, so s-step's smaller reduction count is
 * a wall-clock saving, and the n at which s-step stops beating MGS is the measured theta_h.
 *
 * Single process, both devices via ncclCommInitAll, the rung reachable in a one-node allocation.
 * Values are immaterial to a timing, as in the single-GPU harness: the all-reduces are real
 * and the kernels move real traffic, but no scalar is fed forward, and orthogonality is the
 * single-GPU harness's job. The reduction counts are exact: R_mgs = 1 + m(m+3)/2 all-reduces
 * of one double, R_ca = 1 + 2*ceil(m/s) all-reduces of an s x s block.
 *
 * The MGS baseline is cuBLAS BLAS-1, so its local compute is launch-bound (flat in n) until the
 * grid is large enough to turn bandwidth-bound. The measured R_h is formed from that local
 * compute, so it sits below a model that assumes a bandwidth-bound baseline. The reduction cost
 * in the numerator is the same either way, and is what this harness measures against the ladder.
 *
 * Usage:
 *   ./regime-gpu-sstep-2gpu [--machine v100-pcie-16gb] [--m M] [--n-list "8000 61000 227000"]
 *                           [--s-list "1 2 4 6 8"] [--s-max 9] [--repeats K]
 *                           [--t-reduce-us T] [--csv PATH]
 *
 * @author Kevin Knights
 * @date 2026-07-25
 */

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <nccl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "gpu_contention.cuh"
#include "gram_splitk.cuh"
#include "trsm_tallskinny.cuh"

namespace {

using Clock = std::chrono::steady_clock;
using Sec   = std::chrono::duration<double>;

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

#define NCCL_CHECK(call)                                                             \
    do {                                                                             \
        const ncclResult_t res_ = (call);                                            \
        if (res_ != ncclSuccess) {                                                   \
            std::fprintf(stderr, "NCCL error at %s:%d -- %s\n", __FILE__, __LINE__,  \
                         ncclGetErrorString(res_));                                  \
            std::exit(EXIT_FAILURE);                                                 \
        }                                                                            \
    } while (0)

[[nodiscard]] int64_t mgs_reductions(int m)
{
    return 1 + static_cast<int64_t>(m) * (m + 3) / 2;
}

[[nodiscard]] int64_t ca_reductions(int m, int s)
{
    const int64_t blocks = (m + s - 1) / s;
    return 1 + blocks * 2;
}

[[nodiscard]] double median(std::vector<double> v)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

/// One local device and its persistent handles. Buffers are per measurement, allocated in the
/// runners, since their sizes depend on n and s.
struct Dev {
    int                id     = 0;
    cudaStream_t       stream = nullptr;
    ncclComm_t         comm   = nullptr;
    cublasHandle_t     blas   = nullptr;
    cusolverDnHandle_t solver = nullptr;
    double*            d_one    = nullptr;   ///< 1.0, device-resident alpha
    double*            d_zero   = nullptr;   ///< 0.0, device-resident beta
    double*            d_scalar = nullptr;   ///< scratch for a 1-double reduction
    int64_t            n      = 0;           ///< local rows for the current measurement
};

/// Rows on device d when n_global is split across ndev, larger remainder to the low devices.
[[nodiscard]] int64_t local_rows(int64_t n_global, int ndev, int d)
{
    return n_global / ndev + (d < static_cast<int>(n_global % ndev) ? 1 : 0);
}

/// Sync every device's stream: a collective costs what its slowest participant costs, so the
/// clock may not stop until both have drained.
void sync_all(std::vector<Dev>& devs)
{
    for (Dev& dv : devs) {
        CUDA_CHECK(cudaSetDevice(dv.id));
        CUDA_CHECK(cudaStreamSynchronize(dv.stream));
    }
}

template <typename F>
[[nodiscard]] double time_cycle(std::vector<Dev>& devs, F&& fn)
{
    sync_all(devs);
    const Clock::time_point t0 = Clock::now();
    fn();
    sync_all(devs);
    return Sec(Clock::now() - t0).count();
}

/// A deterministic distinct-frequency fill. Column c on the local slice. Values are immaterial
/// to a timing, but a finite, full-rank block keeps the kernels honest across repeats.
[[nodiscard]] std::vector<double> fill_slice(int64_t n_local, int cols)
{
    std::vector<double> h(static_cast<std::size_t>(n_local) * cols);
    for (int c = 0; c < cols; ++c)
        for (int64_t r = 0; r < n_local; ++r)
            h[static_cast<std::size_t>(c) * static_cast<std::size_t>(n_local)
              + static_cast<std::size_t>(r)] =
                std::sin(0.7 * static_cast<double>(c + 1) * static_cast<double>(r) + 1.0);
    return h;
}

/// Distributed MGS local compute on an (m+1)-column basis. with_reduce toggles the all-reduces,
/// so the same call gives the wall-clock (with) and the local compute the R_h denominator needs
/// (without).
[[nodiscard]] double run_mgs(std::vector<Dev>& devs, int m, int64_t n_global, int repeats,
                             bool with_reduce)
{
    const int ndev = static_cast<int>(devs.size());
    const int cols = m + 1;
    std::vector<double*> d_V(devs.size(), nullptr), d_V0(devs.size(), nullptr);

    for (int d = 0; d < ndev; ++d) {
        devs[static_cast<std::size_t>(d)].n = local_rows(n_global, ndev, d);
        const int64_t nd = devs[static_cast<std::size_t>(d)].n;
        const std::size_t elems = static_cast<std::size_t>(nd) * cols;
        CUDA_CHECK(cudaSetDevice(devs[static_cast<std::size_t>(d)].id));
        CUDA_CHECK(cudaMalloc(&d_V[static_cast<std::size_t>(d)],  elems * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_V0[static_cast<std::size_t>(d)], elems * sizeof(double)));
        const std::vector<double> h = fill_slice(nd, cols);
        CUDA_CHECK(cudaMemcpy(d_V0[static_cast<std::size_t>(d)], h.data(),
                              elems * sizeof(double), cudaMemcpyHostToDevice));
    }

    auto col = [](double* base, int i, int64_t nd) {
        return base + static_cast<std::size_t>(i) * static_cast<std::size_t>(nd);
    };
    auto reset = [&] {
        for (int d = 0; d < ndev; ++d) {
            const int64_t nd = devs[static_cast<std::size_t>(d)].n;
            CUDA_CHECK(cudaSetDevice(devs[static_cast<std::size_t>(d)].id));
            CUDA_CHECK(cudaMemcpy(d_V[static_cast<std::size_t>(d)], d_V0[static_cast<std::size_t>(d)],
                                  static_cast<std::size_t>(nd) * cols * sizeof(double),
                                  cudaMemcpyDeviceToDevice));
        }
        sync_all(devs);
    };
    auto reduce_scalar = [&] {
        if (!with_reduce) return;
        NCCL_CHECK(ncclGroupStart());
        for (Dev& dv : devs)
            NCCL_CHECK(ncclAllReduce(dv.d_scalar, dv.d_scalar, 1, ncclDouble, ncclSum,
                                     dv.comm, dv.stream));
        NCCL_CHECK(ncclGroupEnd());
    };
    // One m-step orthogonalization. Each dot and norm is a partial on every device followed by an
    // all-reduce. Rhe axpy and scale are local. The scalars are never fed forward (timing), but
    // the sequence and the reduction count are exactly MGS's.
    auto orthogonalize = [&] {
        auto dot = [&](int i, int j) {
            for (int d = 0; d < ndev; ++d) {
                Dev& dv = devs[static_cast<std::size_t>(d)];
                CUDA_CHECK(cudaSetDevice(dv.id));
                CUBLAS_CHECK(cublasDdot(dv.blas, static_cast<int>(dv.n),
                                        col(d_V[static_cast<std::size_t>(d)], i, dv.n), 1,
                                        col(d_V[static_cast<std::size_t>(d)], j, dv.n), 1,
                                        dv.d_scalar));
            }
            reduce_scalar();
        };
        auto axpy = [&](int i, int j) {
            for (int d = 0; d < ndev; ++d) {
                Dev& dv = devs[static_cast<std::size_t>(d)];
                CUDA_CHECK(cudaSetDevice(dv.id));
                CUBLAS_CHECK(cublasDaxpy(dv.blas, static_cast<int>(dv.n), dv.d_one,
                                         col(d_V[static_cast<std::size_t>(d)], i, dv.n), 1,
                                         col(d_V[static_cast<std::size_t>(d)], j, dv.n), 1));
            }
        };
        auto scale = [&](int j) {
            for (int d = 0; d < ndev; ++d) {
                Dev& dv = devs[static_cast<std::size_t>(d)];
                CUDA_CHECK(cudaSetDevice(dv.id));
                CUBLAS_CHECK(cublasDscal(dv.blas, static_cast<int>(dv.n), dv.d_one,
                                         col(d_V[static_cast<std::size_t>(d)], j, dv.n), 1));
            }
        };
        dot(0, 0); scale(0);                 // norm of column 0
        for (int j = 1; j <= m; ++j) {
            for (int i = 0; i < j; ++i) { dot(i, j); axpy(i, j); }
            dot(j, j); scale(j);             // norm of column j
        }
    };

    reset();
    (void)time_cycle(devs, orthogonalize);   // warm-up
    std::vector<double> secs;
    secs.reserve(static_cast<std::size_t>(repeats));
    for (int r = 0; r < repeats; ++r) { reset(); secs.push_back(time_cycle(devs, orthogonalize)); }

    for (int d = 0; d < ndev; ++d) {
        CUDA_CHECK(cudaSetDevice(devs[static_cast<std::size_t>(d)].id));
        CUDA_CHECK(cudaFree(d_V[static_cast<std::size_t>(d)]));
        CUDA_CHECK(cudaFree(d_V0[static_cast<std::size_t>(d)]));
    }
    return median(secs);
}

/// Distributed s-step: ceil(m/s) blocks of CholQR2, each Gram a real all-reduce of the s x s
/// partial. Cholesky and the triangular solve are local. After the all-reduce every device holds
/// the same G, so each factors it and solves its own row slice.
[[nodiscard]] double run_sstep(std::vector<Dev>& devs, int m, int64_t n_global, int s, int repeats,
                               int sm_count)
{
    const int ndev = static_cast<int>(devs.size());
    const int64_t blocks = (m + s - 1) / s;
    std::vector<double*> d_B(devs.size(), nullptr), d_B0(devs.size(), nullptr),
                         d_G(devs.size(), nullptr), d_work(devs.size(), nullptr);
    std::vector<int*>    d_info(devs.size(), nullptr);
    int lwork = 0;

    for (int d = 0; d < ndev; ++d) {
        devs[static_cast<std::size_t>(d)].n = local_rows(n_global, ndev, d);
        const int64_t nd = devs[static_cast<std::size_t>(d)].n;
        const std::size_t belems = static_cast<std::size_t>(nd) * s;
        Dev& dv = devs[static_cast<std::size_t>(d)];
        CUDA_CHECK(cudaSetDevice(dv.id));
        CUDA_CHECK(cudaMalloc(&d_B[static_cast<std::size_t>(d)],  belems * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_B0[static_cast<std::size_t>(d)], belems * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_G[static_cast<std::size_t>(d)],
                              static_cast<std::size_t>(s) * s * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_info[static_cast<std::size_t>(d)], sizeof(int)));
        const std::vector<double> h = fill_slice(nd, s);
        CUDA_CHECK(cudaMemcpy(d_B0[static_cast<std::size_t>(d)], h.data(),
                              belems * sizeof(double), cudaMemcpyHostToDevice));
        CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(dv.solver, CUBLAS_FILL_MODE_UPPER, s,
                                                   d_G[static_cast<std::size_t>(d)], s, &lwork));
        CUDA_CHECK(cudaMalloc(&d_work[static_cast<std::size_t>(d)],
                              static_cast<std::size_t>(std::max(lwork, 1)) * sizeof(double)));
    }

    auto reset = [&] {
        for (int d = 0; d < ndev; ++d) {
            const int64_t nd = devs[static_cast<std::size_t>(d)].n;
            CUDA_CHECK(cudaSetDevice(devs[static_cast<std::size_t>(d)].id));
            CUDA_CHECK(cudaMemcpy(d_B[static_cast<std::size_t>(d)], d_B0[static_cast<std::size_t>(d)],
                                  static_cast<std::size_t>(nd) * s * sizeof(double),
                                  cudaMemcpyDeviceToDevice));
        }
        sync_all(devs);
    };
    auto reduce_gram = [&] {
        NCCL_CHECK(ncclGroupStart());
        for (int d = 0; d < ndev; ++d) {
            Dev& dv = devs[static_cast<std::size_t>(d)];
            NCCL_CHECK(ncclAllReduce(d_G[static_cast<std::size_t>(d)],
                                     d_G[static_cast<std::size_t>(d)],
                                     static_cast<std::size_t>(s) * s, ncclDouble, ncclSum,
                                     dv.comm, dv.stream));
        }
        NCCL_CHECK(ncclGroupEnd());
    };
    auto reduce_scalar = [&] {
        NCCL_CHECK(ncclGroupStart());
        for (Dev& dv : devs)
            NCCL_CHECK(ncclAllReduce(dv.d_scalar, dv.d_scalar, 1, ncclDouble, ncclSum,
                                     dv.comm, dv.stream));
        NCCL_CHECK(ncclGroupEnd());
    };
    auto cholqr = [&] {
        for (int d = 0; d < ndev; ++d) {
            Dev& dv = devs[static_cast<std::size_t>(d)];
            CUDA_CHECK(cudaSetDevice(dv.id));
            CUDA_CHECK(gram_splitk(d_B[static_cast<std::size_t>(d)], dv.n, s,
                                   d_G[static_cast<std::size_t>(d)], sm_count, 4, dv.stream));
        }
        reduce_gram();   // partial Gram matrices summed to the full G on every device
        for (int d = 0; d < ndev; ++d) {
            Dev& dv = devs[static_cast<std::size_t>(d)];
            CUDA_CHECK(cudaSetDevice(dv.id));
            CUSOLVER_CHECK(cusolverDnDpotrf(dv.solver, CUBLAS_FILL_MODE_UPPER, s,
                                            d_G[static_cast<std::size_t>(d)], s,
                                            d_work[static_cast<std::size_t>(d)], lwork,
                                            d_info[static_cast<std::size_t>(d)]));
            CUDA_CHECK(trsm_tallskinny(d_G[static_cast<std::size_t>(d)],
                                       d_B[static_cast<std::size_t>(d)], dv.n, s, sm_count, 4,
                                       dv.stream));
        }
    };
    // One m-step cycle: an opening reduction, then CholQR2 on each block. The reduction count is
    // 1 + 2*blocks, matching ca_reductions.
    auto cycle = [&] {
        reduce_scalar();
        for (int64_t b = 0; b < blocks; ++b) { cholqr(); cholqr(); }
    };

    reset();
    (void)time_cycle(devs, cycle);   // warm-up
    std::vector<double> secs;
    secs.reserve(static_cast<std::size_t>(repeats));
    for (int r = 0; r < repeats; ++r) { reset(); secs.push_back(time_cycle(devs, cycle)); }

    for (int d = 0; d < ndev; ++d) {
        CUDA_CHECK(cudaSetDevice(devs[static_cast<std::size_t>(d)].id));
        CUDA_CHECK(cudaFree(d_B[static_cast<std::size_t>(d)]));
        CUDA_CHECK(cudaFree(d_B0[static_cast<std::size_t>(d)]));
        CUDA_CHECK(cudaFree(d_G[static_cast<std::size_t>(d)]));
        CUDA_CHECK(cudaFree(d_work[static_cast<std::size_t>(d)]));
        CUDA_CHECK(cudaFree(d_info[static_cast<std::size_t>(d)]));
    }
    return median(secs);
}

struct Args {
    std::string machine = "v100-pcie-16gb";
    int         m       = 12;
    int         repeats = 7;
    int         s_max   = 9;
    double      t_reduce_us = 0.0;   ///< calibrated DEVICE_P2P rung, for the R_h prediction
    std::vector<int64_t> n_list{8000, 61000, 227000, 705000, 1728000};
    std::vector<int>     s_list{1, 2, 4, 6, 8};
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
        else if (arg == "--m")           a.m       = std::stoi(next());
        else if (arg == "--repeats")     a.repeats = std::stoi(next());
        else if (arg == "--s-max")       a.s_max   = std::stoi(next());
        else if (arg == "--t-reduce-us") a.t_reduce_us = std::stod(next());
        else if (arg == "--n-list")      a.n_list = parse_list<int64_t>(next());
        else if (arg == "--s-list")      a.s_list = parse_list<int>(next());
        else if (arg == "--csv")         a.csv_path = next();
        else if (arg == "--help") {
            std::printf(
                "Usage: ./regime-gpu-sstep-2gpu [--machine KEY] [--m M] [--n-list \"...\"]\n"
                "                               [--s-list \"1 2 4 8\"] [--s-max S] [--repeats K]\n"
                "                               [--t-reduce-us T] [--csv PATH]\n\n"
                "  Splits the n basis rows across both local GPUs, so every MGS dot/norm and every\n"
                "  s-step Gram is a real NCCL all-reduce. Sweeps n at the DEVICE_P2P rung; the n\n"
                "  where s-step stops beating MGS is the measured horizontal crossover.\n\n"
                "  --t-reduce-us T  calibrated DEVICE_P2P reduction cost, for the predicted R_h the\n"
                "                   measured crossover is checked against.\n");
            std::exit(0);
        }
        else { std::fprintf(stderr, "Unknown flag: %s\n", arg.c_str()); std::exit(EXIT_FAILURE); }
    }
    if (a.m < 1)     { std::fprintf(stderr, "--m must be >= 1\n"); std::exit(1); }
    if (a.s_max < 1) { std::fprintf(stderr, "--s-max must be >= 1\n"); std::exit(1); }
    return a;
}

struct Row {
    int64_t n; int s; int64_t blocks; int64_t r_ca;
    double mgs_local_us; double mgs_2gpu_us; double sstep_2gpu_us; double speedup;
    double meas_treduce_us; double rh_pred;
};

}  // namespace

int main(int argc, char** argv)
{
    const Args a = parse_args(argc, argv);

    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev < 2) {
        std::fprintf(stderr,
            "Need at least 2 visible GPUs: the measured crossover is a real inter-GPU reduction,\n"
            "and one device has nothing to reduce across. Check CUDA_VISIBLE_DEVICES and that the\n"
            "allocation requested --gpus-per-node=2.\n");
        return EXIT_FAILURE;
    }

    // Every participating device must be idle: a collective costs what its slowest participant
    // costs, so one contended GPU contaminates the whole measurement.
    bool contended = false;
    for (int d = 0; d < ndev; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        const DeviceContention c = check_device_contention();
        if (c.contended) { std::printf("  device %d:", d); report_contention(c); contended = true; }
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // Comms first, then per-device streams and handles bound to their device and stream. Device
    // pointer mode keeps the vector kernels asynchronous: a host-pointer dot would block to
    // return its scalar and serialize the pipeline the all-reduce is meant to dominate.
    std::vector<Dev> devs(static_cast<std::size_t>(ndev));
    std::vector<int> ids(static_cast<std::size_t>(ndev));
    for (int d = 0; d < ndev; ++d) ids[static_cast<std::size_t>(d)] = d;
    std::vector<ncclComm_t> comms(static_cast<std::size_t>(ndev));
    NCCL_CHECK(ncclCommInitAll(comms.data(), ndev, ids.data()));

    const double one = 1.0, zero = 0.0;
    for (int d = 0; d < ndev; ++d) {
        Dev& dv = devs[static_cast<std::size_t>(d)];
        dv.id = d;
        dv.comm = comms[static_cast<std::size_t>(d)];
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaStreamCreate(&dv.stream));
        CUBLAS_CHECK(cublasCreate(&dv.blas));
        CUBLAS_CHECK(cublasSetStream(dv.blas, dv.stream));
        CUBLAS_CHECK(cublasSetPointerMode(dv.blas, CUBLAS_POINTER_MODE_DEVICE));
        CUSOLVER_CHECK(cusolverDnCreate(&dv.solver));
        CUSOLVER_CHECK(cusolverDnSetStream(dv.solver, dv.stream));
        CUDA_CHECK(cudaMalloc(&dv.d_one, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dv.d_zero, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dv.d_scalar, sizeof(double)));
        CUDA_CHECK(cudaMemcpy(dv.d_one, &one, sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dv.d_zero, &zero, sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dv.d_scalar, 0, sizeof(double)));
    }

    std::printf("GPU s-step vs MGS on two devices -- the measured horizontal crossover\n");
    std::printf("  machine=%s  devices=%d (%s)  m=%d  rung=DEVICE_P2P\n",
                a.machine.c_str(), ndev, prop.name, a.m);
    std::printf("  rows split across devices; every reduction is a real NCCL all-reduce\n");
    report_toolkit();

    std::vector<int> s_list;
    for (const int s : a.s_list) if (s >= 1 && s <= a.s_max) s_list.push_back(s);
    if (s_list.empty()) {
        std::fprintf(stderr, "No s in --s-list falls within 1..%d (--s-max)\n", a.s_max);
        return EXIT_FAILURE;
    }

    std::vector<Row> rows;
    for (const int64_t n : a.n_list) {
        const double mgs_local = run_mgs(devs, a.m, n, a.repeats, false);
        const double mgs_2gpu  = run_mgs(devs, a.m, n, a.repeats, true);
        const int64_t r_mgs = mgs_reductions(a.m);
        const double meas_treduce_us = r_mgs > 0
            ? (mgs_2gpu - mgs_local) / static_cast<double>(r_mgs) * 1e6 : 0.0;
        const double rh_pred = mgs_local > 0.0 && a.t_reduce_us > 0.0
            ? (a.t_reduce_us * 1e-6) / (mgs_local / static_cast<double>(r_mgs)) : 0.0;

        std::printf("\n  n=%lld  MGS: local %.1f us, 2-GPU %.1f us, R=%lld"
                    "  (measured t_reduce %.2f us; predicted R_h %.3f)\n",
                    n, mgs_local * 1e6, mgs_2gpu * 1e6, r_mgs, meas_treduce_us, rh_pred);
        std::printf("  %4s %8s %6s %13s %13s %10s\n",
                    "s", "blocks", "R_ca", "s-step 2gpu", "MGS 2gpu", "MGS/sstep");
        std::printf("  %s\n", std::string(58, '-').c_str());

        for (const int s : s_list) {
            const double sstep_2gpu = run_sstep(devs, a.m, n, s, a.repeats, prop.multiProcessorCount);
            const int64_t blocks = (a.m + s - 1) / s;
            const int64_t r_ca   = ca_reductions(a.m, s);
            const double speedup = sstep_2gpu > 0.0 ? mgs_2gpu / sstep_2gpu : 0.0;
            rows.push_back({n, s, blocks, r_ca, mgs_local * 1e6, mgs_2gpu * 1e6,
                            sstep_2gpu * 1e6, speedup, meas_treduce_us, rh_pred});
            std::printf("  %4d %8lld %6lld %13.2f %13.2f %10.3f\n",
                        s, blocks, r_ca, sstep_2gpu * 1e6, mgs_2gpu * 1e6, speedup);
        }
        std::printf("  %s\n", std::string(58, '-').c_str());
    }

    std::printf("\n  MGS/sstep > 1 means s-step is faster at that n. Predicted R_h is the MGS\n"
                "  baseline coordinate at the calibrated rung; the crossover in n (MGS/sstep = 1)\n"
                "  is the measured theta_h to read against R_h = 1.\n");
    if (contended)
        std::printf("\n  Contended device(s): treat these as a smoke test, not a measurement.\n");

    for (int d = 0; d < ndev; ++d) {
        Dev& dv = devs[static_cast<std::size_t>(d)];
        CUDA_CHECK(cudaSetDevice(dv.id));
        CUDA_CHECK(cudaFree(dv.d_one));
        CUDA_CHECK(cudaFree(dv.d_zero));
        CUDA_CHECK(cudaFree(dv.d_scalar));
        CUBLAS_CHECK(cublasDestroy(dv.blas));
        CUSOLVER_CHECK(cusolverDnDestroy(dv.solver));
        CUDA_CHECK(cudaStreamDestroy(dv.stream));
        NCCL_CHECK(ncclCommDestroy(dv.comm));
    }

    if (!a.csv_path.empty()) {
        std::FILE* f = std::fopen(a.csv_path.c_str(), "w");
        if (!f) { std::fprintf(stderr, "Cannot open CSV: %s\n", a.csv_path.c_str()); return 1; }
        std::fprintf(f, "machine,device_name,devices,m,n,s,blocks,r_ca,r_mgs,mgs_local_us,"
                        "mgs_2gpu_us,sstep_2gpu_us,speedup,meas_treduce_us,t_reduce_us,rh_pred,"
                        "contended\n");
        for (const Row& r : rows)
            std::fprintf(f, "%s,\"%s\",%d,%d,%lld,%d,%lld,%lld,%lld,%.4f,%.4f,%.4f,%.6f,%.6f,"
                            "%.6f,%.6f,%d\n",
                         a.machine.c_str(), prop.name, ndev, a.m, r.n, r.s, r.blocks, r.r_ca,
                         mgs_reductions(a.m), r.mgs_local_us, r.mgs_2gpu_us, r.sstep_2gpu_us,
                         r.speedup, r.meas_treduce_us, a.t_reduce_us, r.rh_pred,
                         contended ? 1 : 0);
        std::fclose(f);
        std::printf("\n  [Wrote %s]\n", a.csv_path.c_str());
    }
    return EXIT_SUCCESS;
}
