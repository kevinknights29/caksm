/**
 * @file ca_matrix_powers_2gpu.cu
 * @brief Two-GPU slab decomposition and deep-halo matrix-powers validation.
 *
 * Does the slab recurrence produce the same basis as a single GPU would? Each
 * device owns a contiguous range of z-planes, exchanges an s-deep halo once,
 * then runs the whole width locally. Correctness rests entirely on that halo
 * being deep enough, so the check here is against a global single-device
 * reference rather than against another distributed run.
 *
 * Usage:
 *
 *   ./ca-matrix-powers-2gpu [--n N] [--s S] [--scale H]
 *       [--option basket|rainbow] [--devices 0,1] [--repeats K]
 *
 * @author Kevin Knights
 * @date 2026-07-27
 */


#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ca_pricing_gpu.hpp"
#include "gpu_contention.cuh"
#include "gpu_pde_slab.cuh"
#include "regime.hpp"

#define CUDA_CHECK(call) do {                                                     \
    const cudaError_t e_ = (call);                                                \
    if (e_ != cudaSuccess) {                                                      \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                     cudaGetErrorString(e_));                                     \
        std::exit(EXIT_FAILURE);                                                  \
    }                                                                             \
} while (0)

#define NCCL_CHECK(call) do {                                                     \
    const ncclResult_t e_ = (call);                                               \
    if (e_ != ncclSuccess) {                                                      \
        std::fprintf(stderr, "NCCL error at %s:%d: %s\n", __FILE__, __LINE__,     \
                     ncclGetErrorString(e_));                                     \
        std::exit(EXIT_FAILURE);                                                  \
    }                                                                             \
} while (0)

namespace {

using Clock = std::chrono::steady_clock;
using Seconds = std::chrono::duration<double>;

/// Command line, already validated by parse_args.
struct Args {
    int n = 31;
    int s = 4;
    int repeats = 7;
    double scale = 0.01;
    bool rainbow = false;
    std::vector<int> devices{0, 1};
};

/// Split a "0,1" device list into ordinals. Slab rank follows list order, so
/// the order given here is the order of the z ranges.
[[nodiscard]] std::vector<int> parse_devices(const std::string& text)
{
    std::vector<int> devices;
    std::istringstream input(text);
    std::string token;
    while (std::getline(input, token, ','))
        devices.push_back(std::stoi(token));
    return devices;
}

/// Parse and validate the command line, or throw. Two distinct devices and at
/// least s planes per slab are both required: a slab thinner than the halo it
/// exchanges would need data from a neighbor's neighbor.
[[nodiscard]] Args parse_args(int argc, char** argv)
{
    Args args;
    for (int q = 1; q < argc; ++q) {
        const std::string arg = argv[q];
        auto next = [&]() -> const char* {
            if (++q >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", arg.c_str());
                std::exit(EXIT_FAILURE);
            }
            return argv[q];
        };
        if      (arg == "--n")       args.n = std::stoi(next());
        else if (arg == "--s")       args.s = std::stoi(next());
        else if (arg == "--repeats") args.repeats = std::stoi(next());
        else if (arg == "--scale")   args.scale = std::stod(next());
        else if (arg == "--devices") args.devices = parse_devices(next());
        else if (arg == "--option") {
            const std::string option = next();
            if      (option == "basket")  args.rainbow = false;
            else if (option == "rainbow") args.rainbow = true;
            else throw std::invalid_argument("--option must be basket or rainbow");
        } else if (arg == "--help") {
            std::printf(
                "Usage: ./ca-matrix-powers-2gpu [--n N] [--s S] [--scale H]\n"
                "                                    [--option basket|rainbow]\n"
                "                                    [--devices 0,1] [--repeats K]\n");
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::invalid_argument("unknown flag: " + arg);
        }
    }
    if (args.n < 3) throw std::invalid_argument("--n must be >= 3");
    if (args.s < 1 || args.s > kMpkMaxS)
        throw std::invalid_argument("--s is outside the supported range");
    if (args.repeats < 1)
        throw std::invalid_argument("--repeats must be positive");
    if (args.devices.size() != 2 || args.devices[0] == args.devices[1])
        throw std::invalid_argument("--devices must name two distinct GPUs");
    if (args.n < 2 * args.s)
        throw std::invalid_argument("each slab must own at least s planes");
    return args;
}

/// One GPU's share: the z range it owns, its stream and communicator, and the
/// device buffers that stay resident on it for the whole run.
struct Slab {
    int device = 0;
    int z_begin = 0;
    int z_count = 0;
    int64_t local_N = 0;
    int64_t ld = 0;
    cudaStream_t stream = nullptr;
    ncclComm_t comm = nullptr;
    double* start = nullptr;
    double* halo = nullptr;
    double* B = nullptr;
    double* face_b = nullptr;
};

/// Wait for every slab's stream. Timing brackets and correctness reads both
/// need all devices quiet, not just the one currently selected.
void sync_all(std::vector<Slab>& slabs)
{
    for (Slab& slab : slabs) {
        CUDA_CHECK(cudaSetDevice(slab.device));
        CUDA_CHECK(cudaStreamSynchronize(slab.stream));
    }
}

/// Zero each halo buffer and copy the owned planes into its middle.
///
/// The zeroing is what gives the outermost slabs their domain boundary: the s
/// planes beyond the global edge are never received from anyone, so they must
/// read as zero rather than as whatever was there last iteration.
void prepare_halos(
    std::vector<Slab>& slabs, int n, int s)
{
    const int64_t n2 = static_cast<int64_t>(n) * n;
    for (Slab& slab : slabs) {
        CUDA_CHECK(cudaSetDevice(slab.device));
        const std::size_t halo_values =
            static_cast<std::size_t>(slab.z_count + 2 * s)
            * static_cast<std::size_t>(n2);
        CUDA_CHECK(cudaMemsetAsync(
            slab.halo, 0, halo_values * sizeof(double), slab.stream));
        CUDA_CHECK(cudaMemcpyAsync(
            slab.halo + static_cast<int64_t>(s) * n2,
            slab.start,
            static_cast<std::size_t>(slab.local_N) * sizeof(double),
            cudaMemcpyDeviceToDevice, slab.stream));
    }
}

/// Trade s planes with each neighbor, in one NCCL group.
///
/// Grouping matters: every rank posts both its receives and its sends before
/// the group closes, so the pairwise exchange completes in one round instead of
/// deadlocking on an ordering where both sides send first.
void exchange_halos(
    std::vector<Slab>& slabs, int n, int s)
{
    const int64_t n2 = static_cast<int64_t>(n) * n;
    const std::size_t count =
        static_cast<std::size_t>(s) * static_cast<std::size_t>(n2);
    NCCL_CHECK(ncclGroupStart());
    for (std::size_t rank = 0; rank < slabs.size(); ++rank) {
        Slab& slab = slabs[rank];
        CUDA_CHECK(cudaSetDevice(slab.device));
        if (rank > 0) {
            NCCL_CHECK(ncclRecv(
                slab.halo, count, ncclDouble, static_cast<int>(rank - 1),
                slab.comm, slab.stream));
            NCCL_CHECK(ncclSend(
                slab.start, count, ncclDouble, static_cast<int>(rank - 1),
                slab.comm, slab.stream));
        }
        if (rank + 1 < slabs.size()) {
            NCCL_CHECK(ncclRecv(
                slab.halo + static_cast<int64_t>(s + slab.z_count) * n2,
                count, ncclDouble, static_cast<int>(rank + 1),
                slab.comm, slab.stream));
            NCCL_CHECK(ncclSend(
                slab.start + static_cast<int64_t>(slab.z_count - s) * n2,
                count, ncclDouble, static_cast<int>(rank + 1),
                slab.comm, slab.stream));
        }
    }
    NCCL_CHECK(ncclGroupEnd());
}

/// Run the full-width recurrence on every slab from its staged halo. One
/// launch per device, and no communication: that is the whole point of having
/// exchanged a deep halo first.
void launch_blocks(
    std::vector<Slab>& slabs, const GpuPdeOperator& op,
    int s, double scale)
{
    for (Slab& slab : slabs) {
        CUDA_CHECK(cudaSetDevice(slab.device));
        CUDA_CHECK(gpu_pde_slab_matrix_powers(
            slab.start, slab.halo, slab.B, slab.ld, s,
            op, slab.face_b, scale,
            slab.z_begin, slab.z_count, slab.stream));
    }
}

/// Time one phase across all devices: quiesce, run, quiesce, report seconds.
/// The leading sync is what keeps a previous phase's tail out of this one's
/// measurement.
template <typename Function>
[[nodiscard]] double time_component(
    std::vector<Slab>& slabs, Function&& function)
{
    sync_all(slabs);
    const auto begin = Clock::now();
    function();
    sync_all(slabs);
    return Seconds(Clock::now() - begin).count();
}

/// Median of a repeat distribution. Takes its argument by value because it
/// sorts in place.
[[nodiscard]] double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const Args args = parse_args(argc, argv);
        const CaPricingModel model;
        CUDA_CHECK(cudaSetDevice(args.devices[0]));
        report_toolkit();
        const PDESystem sys = build_pde_system(
            args.n, model.strike, model.rate, model.expiry,
            model.sigma, model.rho, model.weight, model.spot,
            model.alpha, args.rainbow);
        const GpuPdeOperator op =
            make_gpu_pde_operator(sys, model, args.rainbow);
        const int64_t n2 = static_cast<int64_t>(args.n) * args.n;
        const int64_t global_ld = static_cast<int64_t>(sys.N) + 3;

        Eigen::VectorXd start = Eigen::VectorXd::Zero(global_ld);
        start.head(sys.N) = sys.u0;
        if (!args.rainbow) start.tail(3) = make_s_vec(0.0);
        start.normalize();

        const SpMat A =
            args.rainbow ? sys.A : build_A_tilde(sys.A, sys.B, sys.N);
        Eigen::MatrixXd reference(global_ld, args.s + 1);
        reference.col(0) = start;
        for (int step = 1; step <= args.s; ++step) {
            if (args.rainbow) {
                reference.col(step).head(sys.N).noalias() =
                    args.scale * (A * reference.col(step - 1).head(sys.N));
                reference.col(step).tail(3).setZero();
            } else {
                reference.col(step).noalias() =
                    args.scale * (A * reference.col(step - 1));
            }
        }

        std::vector<Slab> slabs(2);
        const int low_planes = args.n / 2;
        slabs[0].device = args.devices[0];
        slabs[0].z_begin = 0;
        slabs[0].z_count = low_planes;
        slabs[1].device = args.devices[1];
        slabs[1].z_begin = low_planes;
        slabs[1].z_count = args.n - low_planes;

        ncclComm_t comms[2]{};
        int devices[2] = {args.devices[0], args.devices[1]};
        NCCL_CHECK(ncclCommInitAll(comms, 2, devices));
        const std::vector<double> face_b = make_gpu_face_b(sys, model);

        bool recordable = true;
        for (std::size_t rank = 0; rank < slabs.size(); ++rank) {
            Slab& slab = slabs[rank];
            slab.comm = comms[rank];
            slab.local_N = static_cast<int64_t>(slab.z_count) * n2;
            slab.ld = slab.local_N + 3;
            CUDA_CHECK(cudaSetDevice(slab.device));
            cudaDeviceProp properties{};
            CUDA_CHECK(cudaGetDeviceProperties(
                &properties, slab.device));
            std::printf(
                "  rank %zu: device %d %s | z=[%d,%d)\n",
                rank, slab.device, properties.name,
                slab.z_begin, slab.z_begin + slab.z_count);
            const DeviceContention contention = check_device_contention();
            report_contention(contention);
            recordable = recordable && !contention.contended;
            CUDA_CHECK(cudaStreamCreate(&slab.stream));
            CUDA_CHECK(cudaMalloc(
                &slab.start,
                static_cast<std::size_t>(slab.ld) * sizeof(double)));
            CUDA_CHECK(cudaMalloc(
                &slab.halo,
                static_cast<std::size_t>(slab.z_count + 2 * args.s)
                    * static_cast<std::size_t>(n2) * sizeof(double)));
            CUDA_CHECK(cudaMalloc(
                &slab.B,
                static_cast<std::size_t>(slab.ld)
                    * static_cast<std::size_t>(args.s + 1) * sizeof(double)));
            if (!args.rainbow) {
                CUDA_CHECK(cudaMalloc(
                    &slab.face_b, face_b.size() * sizeof(double)));
                CUDA_CHECK(cudaMemcpy(
                    slab.face_b, face_b.data(),
                    face_b.size() * sizeof(double),
                    cudaMemcpyHostToDevice));
            }

            std::vector<double> local(
                static_cast<std::size_t>(slab.ld), 0.0);
            const int64_t offset = static_cast<int64_t>(slab.z_begin) * n2;
            std::copy_n(
                start.data() + offset, slab.local_N, local.data());
            std::copy_n(
                start.data() + sys.N, 3,
                local.data() + slab.local_N);
            CUDA_CHECK(cudaMemcpy(
                slab.start, local.data(),
                static_cast<std::size_t>(slab.ld) * sizeof(double),
                cudaMemcpyHostToDevice));
        }

        prepare_halos(slabs, args.n, args.s);
        exchange_halos(slabs, args.n, args.s);
        launch_blocks(slabs, op, args.s, args.scale);
        sync_all(slabs);

        Eigen::MatrixXd measured =
            Eigen::MatrixXd::Zero(global_ld, args.s + 1);
        double replica_error = 0.0;
        for (Slab& slab : slabs) {
            CUDA_CHECK(cudaSetDevice(slab.device));
            Eigen::MatrixXd local(slab.ld, args.s + 1);
            CUDA_CHECK(cudaMemcpy(
                local.data(), slab.B,
                static_cast<std::size_t>(slab.ld)
                    * static_cast<std::size_t>(args.s + 1) * sizeof(double),
                cudaMemcpyDeviceToHost));
            const int64_t offset = static_cast<int64_t>(slab.z_begin) * n2;
            for (int column = 0; column <= args.s; ++column) {
                measured.col(column).segment(offset, slab.local_N) =
                    local.col(column).head(slab.local_N);
                if (&slab == &slabs.front())
                    measured.col(column).tail(3) = local.col(column).tail(3);
                else
                    replica_error = std::max(
                        replica_error,
                        (local.col(column).tail(3)
                         - measured.col(column).tail(3))
                            .lpNorm<Eigen::Infinity>());
            }
        }

        double worst_error = 0.0;
        double interface_error = 0.0;
        int worst_column = 0;
        const int interface_begin =
            std::max(0, low_planes - args.s);
        const int interface_planes =
            std::min(args.n, low_planes + args.s) - interface_begin;
        for (int column = 0; column <= args.s; ++column) {
            const double error =
                (measured.col(column) - reference.col(column))
                    .lpNorm<Eigen::Infinity>();
            if (error > worst_error) {
                worst_error = error;
                worst_column = column;
            }
            interface_error = std::max(
                interface_error,
                (measured.col(column).segment(
                     static_cast<int64_t>(interface_begin) * n2,
                     static_cast<int64_t>(interface_planes) * n2)
                 - reference.col(column).segment(
                     static_cast<int64_t>(interface_begin) * n2,
                     static_cast<int64_t>(interface_planes) * n2))
                    .lpNorm<Eigen::Infinity>());
        }

        std::vector<double> halo_times;
        std::vector<double> kernel_times;
        std::vector<double> cycle_times;
        for (int repeat = 0; repeat < args.repeats; ++repeat) {
            prepare_halos(slabs, args.n, args.s);
            halo_times.push_back(time_component(
                slabs, [&] { exchange_halos(slabs, args.n, args.s); }));
            kernel_times.push_back(time_component(
                slabs, [&] {
                    launch_blocks(slabs, op, args.s, args.scale);
                }));

            cycle_times.push_back(time_component(
                slabs, [&] {
                    prepare_halos(slabs, args.n, args.s);
                    exchange_halos(slabs, args.n, args.s);
                    launch_blocks(slabs, op, args.s, args.scale);
                }));
        }

        int peer_access = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(
            &peer_access, args.devices[0], args.devices[1]));
        const int64_t halo_values =
            2LL * args.s * n2;
        const int64_t redundant_points =
            n2 * args.s * (args.s - 1);
        std::printf("Two-GPU CA matrix powers\n");
        std::printf(
            "  option=%s | n=%d | s=%d | devices=%d,%d | decomposition=%d+%d z-planes\n",
            args.rainbow ? "rainbow" : "basket",
            args.n, args.s, args.devices[0], args.devices[1],
            slabs[0].z_count, slabs[1].z_count);
        std::printf(
            "  backend=NCCL send/recv | peer-access=%s | recordable=%s\n",
            peer_access ? "yes" : "no", recordable ? "yes" : "no");
        std::printf(
            "  halo: %lld values %.3f MiB/block | redundant interface work=%lld stencil points\n",
            static_cast<long long>(halo_values),
            static_cast<double>(halo_values) * sizeof(double)
                / (1024.0 * 1024.0),
            static_cast<long long>(redundant_points));
        std::printf(
            "  error: global=%.6e interface=%.6e replica=%.6e at column=%d\n",
            worst_error, interface_error, replica_error, worst_column);
        std::printf(
            "  median: halo=%.6e s kernel=%.6e s cycle=%.6e s\n",
            median(halo_times), median(kernel_times), median(cycle_times));

        for (Slab& slab : slabs) {
            CUDA_CHECK(cudaSetDevice(slab.device));
            CUDA_CHECK(cudaFree(slab.start));
            CUDA_CHECK(cudaFree(slab.halo));
            CUDA_CHECK(cudaFree(slab.B));
            if (slab.face_b != nullptr) CUDA_CHECK(cudaFree(slab.face_b));
            CUDA_CHECK(cudaStreamDestroy(slab.stream));
            NCCL_CHECK(ncclCommDestroy(slab.comm));
        }

        if (worst_error > 5.0e-12
            || interface_error > 5.0e-12
            || replica_error > 5.0e-12) {
            std::fprintf(stderr, "FAIL: distributed matrix-powers mismatch\n");
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Error: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
