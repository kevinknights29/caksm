/**
 * @file ca_integrator_multigpu.cu
 * @brief Slab-decomposed CA exponential integrator over N local GPUs.
 *
 * @author Kevin Knights
 * @date 2026-07-27
 */

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <nccl.h>
#include <nvToolsExt.h>
#ifdef CAKSM_HAVE_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ca_integrator_memory.hpp"
#include "ca_integrator_slab.cuh"
#include "ca_pricing_gpu.hpp"
#include "gpu_allocation.hpp"
#include "gpu_ca_arnoldi.cuh"
#include "gpu_contention.cuh"
#include "gpu_pde_slab.cuh"
#include "gpu_regime.hpp"
#include "gram_splitk.cuh"
#include "regime.hpp"
#include "trsm_tallskinny.cuh"

namespace {

/// Command line, already validated by parse_args.
struct Args {
    int n = 31;
    int m = 24;
    int s = 4;
    int steps = 100;
    int repeats = 1;
    double tol = 1.0e-8;
    double single_state_atol = 0.0;
    double single_state_rtol = 0.0;
    bool single_state_tolerance_set = false;
    double expiry = 1.0;
    bool rainbow = false;
    std::vector<int> devices{0, 1};
    std::string referee_dir;
    std::string single_gpu_state;
    std::string save_state;
    int profile_start = -1;
    int profile_steps = 5;
    std::string halo_backend = "auto";
    std::string halo_overlap = "off";
    std::string halo_memset = "full";
    std::string shared_carveout = "default";
    bool require_mpi = false;
    bool memory_report = false;
    double memory_reserve = 0.10;
    std::string arm = "as-measured";
    std::string certificate = "immediate";
    std::string agreement = "step";
    bool agreement_self_test = false;
};

/// Split a "0,1" device list into ordinals. Slab rank follows list order, so
/// this order is the order of the z ranges within a process.
[[nodiscard]] std::vector<int> parse_devices(const std::string& text)
{
    std::vector<int> devices;
    std::istringstream input(text);
    std::string token;
    while (std::getline(input, token, ','))
        devices.push_back(std::stoi(token));
    return devices;
}

/// Parse and validate the command line, or throw.
[[nodiscard]] Args parse_args(int argc, char** argv)
{
    Args args;
    for (int q = 1; q < argc; ++q) {
        const std::string arg = argv[q];
        auto next = [&]() -> const char* {
            if (++q >= argc)
                throw std::invalid_argument("missing value for " + arg);
            return argv[q];
        };
        if      (arg == "--n") args.n = std::stoi(next());
        else if (arg == "--m") args.m = std::stoi(next());
        else if (arg == "--s") args.s = std::stoi(next());
        else if (arg == "--steps") args.steps = std::stoi(next());
        else if (arg == "--repeats") args.repeats = std::stoi(next());
        else if (arg == "--tol") args.tol = std::stod(next());
        else if (arg == "--single-state-atol" || arg == "--single-state-tol") {
            args.single_state_atol = std::stod(next());
            args.single_state_tolerance_set = true;
        }
        else if (arg == "--single-state-rtol") {
            args.single_state_rtol = std::stod(next());
            args.single_state_tolerance_set = true;
        }
        else if (arg == "--expiry") args.expiry = std::stod(next());
        else if (arg == "--devices") args.devices = parse_devices(next());
        else if (arg == "--referee-dir") args.referee_dir = next();
        else if (arg == "--single-gpu-state") args.single_gpu_state = next();
        else if (arg == "--save-state") args.save_state = next();
        else if (arg == "--profile-start") args.profile_start = std::stoi(next());
        else if (arg == "--profile-steps") args.profile_steps = std::stoi(next());
        else if (arg == "--halo-backend") args.halo_backend = next();
        else if (arg == "--halo-overlap") args.halo_overlap = next();
        else if (arg == "--halo-memset") args.halo_memset = next();
        else if (arg == "--shared-carveout") args.shared_carveout = next();
        else if (arg == "--require-mpi") args.require_mpi = true;
        else if (arg == "--memory-report") args.memory_report = true;
        else if (arg == "--memory-reserve")
            args.memory_reserve = std::stod(next());
        else if (arg == "--arm") args.arm = next();
        else if (arg == "--certificate") args.certificate = next();
        else if (arg == "--agreement-check") args.agreement = next();
        else if (arg == "--no-agreement-check") args.agreement = "off";
        else if (arg == "--agreement-self-test") {
            args.agreement_self_test = true;
            args.agreement = "strict";
        }
        else if (arg == "--basis") {
            const std::string basis = next();
            if (basis != "monomial")
                throw std::invalid_argument(
                    "the distributed production path currently accepts --basis monomial");
        } else if (arg == "--orth") {
            const std::string orth = next();
            if (orth != "cholqr2")
                throw std::invalid_argument(
                    "the distributed production path currently accepts --orth cholqr2");
        } else if (arg == "--option") {
            const std::string option = next();
            if      (option == "basket") args.rainbow = false;
            else if (option == "rainbow") args.rainbow = true;
            else throw std::invalid_argument("--option must be basket or rainbow");
        } else if (arg == "--help") {
            std::printf(
                "Usage: ./ca-integrator-multigpu [--n N] [--m M] [--s S]\n"
                "       [--steps K] [--repeats K] [--tol T]\n"
                "       [--single-state-atol T] [--single-state-rtol T]\n"
                "       [--expiry T]\n"
                "       [--option basket|rainbow]\n"
                "       [--devices 0,1,2,3] [--basis monomial] [--orth cholqr2]\n"
                "       [--referee-dir DIR] [--single-gpu-state FILE]\n"
                "       [--save-state FILE]\n"
                "       [--profile-start STEP] [--profile-steps K]\n"
                "       [--halo-backend auto|peer|nccl]\n"
                "       [--halo-overlap off|on] [--halo-memset full|shells]\n"
                "       [--shared-carveout default|max] [--require-mpi]\n"
                "       [--memory-report] [--memory-reserve F]\n"
                "       [--arm as-measured|exact-depth]\n"
                "       [--certificate immediate|deferred]\n"
                "       [--agreement-check off|step|strict]\n"
                "       [--no-agreement-check] [--agreement-self-test]\n");
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::invalid_argument("unknown flag: " + arg);
        }
    }
    if (args.n < 3) throw std::invalid_argument("--n must be >= 3");
    if (args.m < 1 || args.m > kGpuCaMaxM)
        throw std::invalid_argument("--m is outside the supported range");
    if (args.s < 1 || args.s > kMpkMaxS)
        throw std::invalid_argument("--s is outside the supported range");
    if (args.steps < 1) throw std::invalid_argument("--steps must be positive");
    if (args.repeats < 1) throw std::invalid_argument("--repeats must be positive");
    if (!(args.tol > 0.0) || !std::isfinite(args.tol))
        throw std::invalid_argument("--tol must be finite and positive");
    if (!args.single_state_tolerance_set)
        args.single_state_atol = args.tol;
    if (!(args.single_state_atol >= 0.0)
        || !std::isfinite(args.single_state_atol))
        throw std::invalid_argument(
            "--single-state-atol must be finite and nonnegative");
    if (!(args.single_state_rtol >= 0.0)
        || !std::isfinite(args.single_state_rtol))
        throw std::invalid_argument(
            "--single-state-rtol must be finite and nonnegative");
    if (args.single_state_atol == 0.0 && args.single_state_rtol == 0.0)
        throw std::invalid_argument(
            "one single-state tolerance must be positive");
    if (!(args.expiry > 0.0) || !std::isfinite(args.expiry))
        throw std::invalid_argument("--expiry must be finite and positive");
    if (args.profile_start < -1 || args.profile_start >= args.steps)
        throw std::invalid_argument("--profile-start must name a timed step");
    if (args.profile_steps < 1
        || (args.profile_start >= 0
            && args.profile_start + args.profile_steps > args.steps))
        throw std::invalid_argument("--profile-steps exceeds the timed solve");
    if (args.devices.empty())
        throw std::invalid_argument("--devices must name at least one GPU");
    {
        // Distinct, not merely pairwise distinct. Two slabs sharing a device would each
        // believe they own their planes exclusively and would overwrite each other's halo.
        std::vector<int> ordered = args.devices;
        std::sort(ordered.begin(), ordered.end());
        if (std::adjacent_find(ordered.begin(), ordered.end()) != ordered.end())
            throw std::invalid_argument("--devices must name distinct GPUs");
    }
    if (args.halo_backend != "auto"
        && args.halo_backend != "peer"
        && args.halo_backend != "nccl")
        throw std::invalid_argument(
            "--halo-backend must be auto, peer, or nccl");
    if (args.halo_overlap != "off" && args.halo_overlap != "on")
        throw std::invalid_argument(
            "--halo-overlap must be off or on");
    if (args.halo_memset != "full" && args.halo_memset != "shells")
        throw std::invalid_argument(
            "--halo-memset must be full or shells");
    if (args.shared_carveout != "default"
        && args.shared_carveout != "max")
        throw std::invalid_argument(
            "--shared-carveout must be default or max");
    if (!(args.memory_reserve >= 0.0 && args.memory_reserve < 1.0))
        throw std::invalid_argument("--memory-reserve must be in [0,1)");
    if (args.arm != "as-measured" && args.arm != "exact-depth")
        throw std::invalid_argument(
            "--arm must be as-measured or exact-depth");
    if (args.certificate != "immediate" && args.certificate != "deferred")
        throw std::invalid_argument(
            "--certificate must be immediate or deferred");
    if (args.agreement != "off" && args.agreement != "step"
        && args.agreement != "strict")
        throw std::invalid_argument(
            "--agreement-check must be off, step, or strict");
    if (args.agreement_self_test && args.agreement != "strict")
        throw std::invalid_argument(
            "--agreement-self-test requires the strict agreement check");
    if (args.agreement_self_test && args.memory_report)
        throw std::invalid_argument(
            "--agreement-self-test requires a solve");
    return args;
}

/// Initialize the required halo cells and copy the owned planes into the middle.
///
/// Full initialization is the historical control. Shell initialization writes
/// only true domain boundaries; neighboring slabs overwrite every other ghost
/// shell before a boundary kernel can consume it.
void prepare_halos(
    std::vector<Slab>& slabs, const std::vector<const double*>& input,
    int n, int depth, int world_gpus, bool shell_initialization)
{
    const NvtxRange range("halo pack");
    const int64_t n2 = static_cast<int64_t>(n) * n;
    for (std::size_t rank = 0; rank < slabs.size(); ++rank) {
        Slab& slab = slabs[rank];
        CUDA_CHECK(cudaSetDevice(slab.device));
        const std::size_t shell_values =
            static_cast<std::size_t>(depth)
            * static_cast<std::size_t>(n2);
        if (!shell_initialization) {
            const std::size_t values =
                static_cast<std::size_t>(slab.z_count + 2 * depth)
                * static_cast<std::size_t>(n2);
            CUDA_CHECK(cudaMemsetAsync(
                slab.halo, 0, values * sizeof(double), slab.stream));
        } else {
            if (slab.global_rank == 0) {
                CUDA_CHECK(cudaMemsetAsync(
                    slab.halo, 0, shell_values * sizeof(double), slab.stream));
            }
            if (slab.global_rank + 1 == world_gpus) {
                CUDA_CHECK(cudaMemsetAsync(
                    slab.halo
                        + static_cast<int64_t>(depth + slab.z_count) * n2,
                    0, shell_values * sizeof(double), slab.stream));
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(
            slab.halo + static_cast<int64_t>(depth) * n2,
            input[rank], static_cast<std::size_t>(slab.local_N) * sizeof(double),
            cudaMemcpyDeviceToDevice, slab.stream));
        CUDA_CHECK(cudaEventRecord(slab.halo_ready, slab.stream));
    }
}

/// Trade halo planes with each neighbor over NCCL, in one group, charging the
/// exchange and its depth to the account. The transport for multi-node runs and
/// for single-node runs without peer access.
void exchange_halos_nccl(
    std::vector<Slab>& slabs, const std::vector<const double*>& input,
    int n, int depth, int world_gpus, bool overlap)
{
    const NvtxRange range("NCCL halo exchange");
    const int64_t n2 = static_cast<int64_t>(n) * n;
    const std::size_t count =
        static_cast<std::size_t>(depth) * static_cast<std::size_t>(n2);
    if (overlap) {
        for (Slab& slab : slabs) {
            CUDA_CHECK(cudaSetDevice(slab.device));
            CUDA_CHECK(cudaStreamWaitEvent(
                slab.halo_stream, slab.halo_ready, 0));
        }
    }
    NCCL_CHECK(ncclGroupStart());
    for (std::size_t local_rank = 0; local_rank < slabs.size(); ++local_rank) {
        Slab& slab = slabs[local_rank];
        CUDA_CHECK(cudaSetDevice(slab.device));
        const cudaStream_t stream =
            overlap ? slab.halo_stream : slab.stream;
        if (slab.global_rank > 0) {
            NCCL_CHECK(ncclRecv(
                slab.halo, count, ncclDouble, slab.global_rank - 1,
                slab.comm, stream));
            NCCL_CHECK(ncclSend(
                input[local_rank], count, ncclDouble, slab.global_rank - 1,
                slab.comm, stream));
        }
        if (slab.global_rank + 1 < world_gpus) {
            NCCL_CHECK(ncclRecv(
                slab.halo + static_cast<int64_t>(depth + slab.z_count) * n2,
                count, ncclDouble, slab.global_rank + 1,
                slab.comm, stream));
            NCCL_CHECK(ncclSend(
                input[local_rank]
                    + static_cast<int64_t>(slab.z_count - depth) * n2,
                count, ncclDouble, slab.global_rank + 1,
                slab.comm, stream));
        }
    }
    NCCL_CHECK(ncclGroupEnd());
    if (overlap) {
        for (Slab& slab : slabs) {
            CUDA_CHECK(cudaSetDevice(slab.device));
            CUDA_CHECK(cudaEventRecord(
                slab.halo_received, slab.halo_stream));
        }
    }
}

/// The same exchange as direct device-to-device copies, when every slab is on one node and
/// adjacent slabs can address each other. Skips the NCCL stack entirely; the events keep the
/// copies ordered against the compute stream.
///
/// The decomposition is one-dimensional in z, so a slab talks only to the slabs below and
/// above it. Each slab PULLS its own halo: its lower halo from the previous slab's last
/// `depth` owned planes, its upper from the next slab's first `depth`. Every copy is issued on
/// the receiving slab's own stream, so an interior slab's two incoming copies are ordered
/// against each other for free and `halo_received` is recorded once, after both. Pulling
/// rather than pushing makes the N-slab case a loop instead of a special case: the ends simply
/// have one neighbor instead of two.
///
/// Local neighbors only. The caller guarantees that, since peer_halo requires a single node;
/// the end slabs of a multi-node run have a neighbor on another rank, which only NCCL reaches.
void exchange_halos_peer(
    std::vector<Slab>& slabs, const std::vector<const double*>& input,
    int n, int depth, bool overlap)
{
    const NvtxRange range("peer halo exchange");
    const int64_t n2 = static_cast<int64_t>(n) * n;
    const std::size_t bytes =
        static_cast<std::size_t>(depth) * static_cast<std::size_t>(n2)
        * sizeof(double);
    const std::size_t count = slabs.size();

    for (std::size_t rank = 0; rank < count; ++rank) {
        Slab& slab = slabs[rank];
        const cudaStream_t stream = overlap ? slab.halo_stream : slab.stream;

        CUDA_CHECK(cudaSetDevice(slab.device));
        // Wait for this slab's own staging, and for each neighbor it is about to read from.
        // Reading a neighbor's owned planes before that neighbor has finished writing them
        // would copy the previous step's values, which is a wrong answer rather than a stall.
        if (overlap)
            CUDA_CHECK(cudaStreamWaitEvent(stream, slab.halo_ready, 0));
        if (rank > 0)
            CUDA_CHECK(cudaStreamWaitEvent(stream, slabs[rank - 1].halo_ready, 0));
        if (rank + 1 < count)
            CUDA_CHECK(cudaStreamWaitEvent(stream, slabs[rank + 1].halo_ready, 0));

        if (rank > 0) {
            const Slab& below = slabs[rank - 1];
            CUDA_CHECK(cudaMemcpyPeerAsync(
                slab.halo, slab.device,
                input[rank - 1]
                    + static_cast<int64_t>(below.z_count - depth) * n2,
                below.device, bytes, stream));
        }
        if (rank + 1 < count) {
            const Slab& above = slabs[rank + 1];
            CUDA_CHECK(cudaMemcpyPeerAsync(
                slab.halo + static_cast<int64_t>(depth + slab.z_count) * n2,
                slab.device, input[rank + 1], above.device, bytes, stream));
        }
        if (overlap)
            CUDA_CHECK(cudaEventRecord(slab.halo_received, stream));
    }
}

enum class HaloPurpose {
    BlockBuild,
    Transition
};

/// Build one block of the local basis: stage the halo, exchange it once, then
/// run the whole recurrence locally.
///
/// The depth requested here is the correction. A block of width s consumes columns 0 to s-1,
/// so it needs s-1 recurrence steps and a halo s-1 planes deep; the as-measured arm asks for s
/// of each and discards the extra column. At exact-depth s=1 that leaves nothing to exchange,
/// and the copy path fires with no communication at all.
void build_matrix_powers(
    std::vector<Slab>& slabs, const std::vector<const double*>& input,
    int n, int steps, const GpuPdeOperator& op, double scale,
    bool peer_halo, bool overlap, bool shell_halo_initialization,
    MpkSharedCarveout shared_carveout, int world_gpus, HaloPurpose purpose,
    ReductionStats& stats)
{
    if (steps == 0) {
        // A width-one block is the start vector itself. There is no recurrence to
        // run, so there is no ghost region to fetch and no interface to exchange.
        const NvtxRange range("width-one block copy");
        for (std::size_t rank = 0; rank < slabs.size(); ++rank) {
            Slab& slab = slabs[rank];
            CUDA_CHECK(cudaSetDevice(slab.device));
            CUDA_CHECK(cudaMemcpyAsync(
                slab.B, input[rank],
                static_cast<std::size_t>(slab.ld) * sizeof(double),
                cudaMemcpyDeviceToDevice, slab.stream));
        }
        return;
    }
    if (world_gpus == 1) {
        // A singleton owns the complete grid. Use the global one-GPU kernel
        // directly: staging a slab halo would perform a vacuous NCCL exchange
        // and needlessly select the distributed kernel.
        Slab& slab = slabs.front();
        CUDA_CHECK(cudaSetDevice(slab.device));
        CUDA_CHECK(gpu_pde_matrix_powers(
            input.front(), slab.B, slab.ld, steps, op, slab.face_b, scale,
            slab.stream, shared_carveout));
        stats.operator_steps += steps;
        return;
    }
    prepare_halos(
        slabs, input, n, steps, world_gpus, shell_halo_initialization);
    if (!overlap) {
        if (peer_halo)
            exchange_halos_peer(slabs, input, n, steps, false);
        else
            exchange_halos_nccl(
                slabs, input, n, steps, world_gpus, false);
        for (Slab& slab : slabs) {
            CUDA_CHECK(cudaSetDevice(slab.device));
            CUDA_CHECK(gpu_pde_slab_matrix_powers(
                input[static_cast<std::size_t>(slab.rank)],
                slab.halo, slab.B, slab.ld, steps, op, slab.face_b, scale,
                slab.z_begin, slab.z_count, slab.stream, shared_carveout));
        }
    } else {
        // Queue both interiors before the short peer copies. If communication
        // is submitted first, it can finish before the host reaches either
        // kernel launch and no device work is available to overlap it.
        for (Slab& slab : slabs) {
            CUDA_CHECK(cudaSetDevice(slab.device));
            CUDA_CHECK(gpu_pde_slab_tail_powers(
                input[static_cast<std::size_t>(slab.rank)],
                slab.B, slab.ld, steps, op, scale, slab.z_count, slab.stream));
            const int interior_begin = steps;
            const int interior_end = slab.z_count - steps;
            if (interior_begin < interior_end) {
                CUDA_CHECK(gpu_pde_slab_matrix_powers_range(
                    slab.halo, slab.B, slab.ld, steps, op, slab.face_b, scale,
                    slab.z_begin, slab.z_count, interior_begin,
                    interior_end - interior_begin, slab.stream,
                    shared_carveout));
            }
        }

        if (peer_halo)
            exchange_halos_peer(slabs, input, n, steps, true);
        else
            exchange_halos_nccl(
                slabs, input, n, steps, world_gpus, true);

        for (Slab& slab : slabs) {
            CUDA_CHECK(cudaSetDevice(slab.device));
            const int interior_begin = steps;
            const int interior_end = slab.z_count - steps;
            CUDA_CHECK(cudaStreamWaitEvent(
                slab.stream, slab.halo_received, 0));
            if (interior_begin < interior_end) {
                CUDA_CHECK(gpu_pde_slab_matrix_powers_range(
                    slab.halo, slab.B, slab.ld, steps, op, slab.face_b, scale,
                    slab.z_begin, slab.z_count, 0, steps, slab.stream,
                    shared_carveout));
                CUDA_CHECK(gpu_pde_slab_matrix_powers_range(
                    slab.halo, slab.B, slab.ld, steps, op, slab.face_b, scale,
                    slab.z_begin, slab.z_count, interior_end, steps,
                    slab.stream, shared_carveout));
            } else {
                CUDA_CHECK(gpu_pde_slab_matrix_powers_range(
                    slab.halo, slab.B, slab.ld, steps, op, slab.face_b, scale,
                    slab.z_begin, slab.z_count, 0, slab.z_count, slab.stream,
                    shared_carveout));
            }
        }
    }
    stats.operator_steps += steps;

    ++stats.halo_exchanges;
    const int64_t exchanged_values =
        2LL * static_cast<int64_t>(steps)
        * static_cast<int64_t>(n) * n
        * static_cast<int64_t>(world_gpus - 1);
    stats.halo_values += exchanged_values;
    if (purpose == HaloPurpose::BlockBuild) {
        ++stats.build_halo_exchanges;
        stats.build_halo_values += exchanged_values;
        stats.build_operator_steps += steps;
        stats.build_depth_min = std::min(stats.build_depth_min, steps);
        stats.build_depth_max = std::max(stats.build_depth_max, steps);
        ++stats.build_halo_depth_hist[static_cast<std::size_t>(steps)];
    } else {
        ++stats.transition_halo_exchanges;
        stats.transition_halo_values += exchanged_values;
    }
}

/// Global 2-norm of a distributed vector: local dot products, one all-reduce,
/// one square root. Each slab contributes only the rows it owns, so the
/// replicated augmented tail is counted once rather than once per rank.
[[nodiscard]] double distributed_norm(
    std::vector<Slab>& slabs, const std::vector<const double*>& input,
    ReductionStats& stats)
{
    const NvtxRange range("distributed norm");
    std::vector<double*> scalars;
    scalars.reserve(slabs.size());
    for (std::size_t rank = 0; rank < slabs.size(); ++rank) {
        Slab& slab = slabs[rank];
        CUDA_CHECK(cudaSetDevice(slab.device));
        CUBLAS_CHECK(cublasSetPointerMode(
            slab.blas, CUBLAS_POINTER_MODE_DEVICE));
        CUBLAS_CHECK(cublasDdot(
            slab.blas, static_cast<int>(slab.reduction_rows),
            input[rank], 1, input[rank], 1, slab.scalar));
        scalars.push_back(slab.scalar);
    }
    allreduce_in_place(slabs, scalars, 1, stats, 'n');
    Slab& norm_root = slabs.front();
    CUDA_CHECK(cudaSetDevice(norm_root.device));
    blocking_read(
        norm_root, norm_root.host_scalar, norm_root.scalar,
        sizeof(double), stats);
    const double norm2 = *norm_root.host_scalar;
    for (Slab& slab : slabs) {
        CUDA_CHECK(cudaSetDevice(slab.device));
        CUBLAS_CHECK(cublasSetPointerMode(
            slab.blas, CUBLAS_POINTER_MODE_HOST));
    }
    return std::sqrt(norm2);
}

/// Project a block against the accepted basis, twice.
///
/// Two passes because one leaves the block only as orthogonal as the first
/// projection's rounding allows. The second pass's coefficients are added to the
/// first's, so the assembled Hessenberg reflects both.
void project_twice(
    std::vector<Slab>& slabs, int filled, int block, ReductionStats& stats)
{
    const NvtxRange range("block projection and reorthogonalization");
    if (filled == 0) return;
    const double one = 1.0;
    const double zero = 0.0;
    const double minus_one = -1.0;
    const std::size_t matrix_values =
        static_cast<std::size_t>(slabs.front().target)
        * static_cast<std::size_t>(block);

    auto pass = [&](bool second) {
        std::vector<double*> buffers;
        buffers.reserve(slabs.size());
        for (Slab& slab : slabs) {
            CUDA_CHECK(cudaSetDevice(slab.device));
            double* C = second ? slab.C2 : slab.C;
            CUDA_CHECK(cudaMemsetAsync(
                C, 0, matrix_values * sizeof(double), slab.stream));
            CUBLAS_CHECK(cublasDgemm(
                slab.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                filled, block, static_cast<int>(slab.reduction_rows),
                &one, slab.V, static_cast<int>(slab.ld),
                slab.B, static_cast<int>(slab.ld),
                &zero, C, slab.target));
            buffers.push_back(C);
        }
        allreduce_in_place(slabs, buffers, matrix_values, stats, 'p');
        for (Slab& slab : slabs) {
            CUDA_CHECK(cudaSetDevice(slab.device));
            double* C = second ? slab.C2 : slab.C;
            CUBLAS_CHECK(cublasDgemm(
                slab.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                static_cast<int>(slab.ld), block, filled,
                &minus_one, slab.V, static_cast<int>(slab.ld),
                C, slab.target, &one, slab.B, static_cast<int>(slab.ld)));
        }
    };

    pass(false);
    pass(true);
    const dim3 threads(16, 16);
    const dim3 grid(
        static_cast<unsigned>((filled + 15) / 16),
        static_cast<unsigned>((block + 15) / 16));
    for (Slab& slab : slabs) {
        CUDA_CHECK(cudaSetDevice(slab.device));
        ca_add_matrix<<<grid, threads, 0, slab.stream>>>(
            slab.C, slab.C2, filled, block, slab.target);
        CUDA_CHECK(cudaGetLastError());
    }
}

/// Form the block Gram matrix across all slabs: a local split-K Gram on the
/// rows each slab owns, then one all-reduce. This single reduction, in place of
/// one per column, is the horizontal mechanism the whole method rests on.
void global_gram(
    std::vector<Slab>& slabs, int block, ReductionStats& stats)
{
    const NvtxRange range("global Gram");
    std::vector<double*> grams;
    grams.reserve(slabs.size());
    for (Slab& slab : slabs) {
        CUDA_CHECK(cudaSetDevice(slab.device));
        CUDA_CHECK(gram_splitk_ld(
            slab.B, slab.reduction_rows, slab.ld, block, slab.G,
            slab.sm_count, 4, slab.stream));
        grams.push_back(slab.G);
    }
    allreduce_in_place(
        slabs, grams,
        static_cast<std::size_t>(block) * static_cast<std::size_t>(block),
        stats, 'g');
}

/// Redundantly Cholesky-factor the replicated Gram on every slab. Redundant on
/// purpose: the factor is tiny, and computing it locally is cheaper than one
/// more collective to distribute it.
void factor_all(std::vector<Slab>& slabs, int block, bool second)
{
    const NvtxRange range("redundant Cholesky");
    for (Slab& slab : slabs) {
        CUDA_CHECK(cudaSetDevice(slab.device));
        CUSOLVER_CHECK(cusolverDnDpotrf(
            slab.solver, CUBLAS_FILL_MODE_UPPER, block, slab.G, block,
            slab.potrf_work, slab.potrf_lwork,
            second ? slab.potrf_info2 : slab.potrf_info));
    }
}

/// Read one Cholesky status word through pinned staging, counted in the
/// host-synchronization account.
[[nodiscard]] int read_factor_info(
    std::vector<Slab>& slabs, bool second, ReductionStats& stats)
{
    Slab& root = slabs.front();
    CUDA_CHECK(cudaSetDevice(root.device));
    blocking_read(
        root, root.host_info, second ? root.potrf_info2 : root.potrf_info,
        sizeof(int), stats);
    return *root.host_info;
}

/// Combine the two triangular factors into the block's R. The first pass gives
/// B = Q1 R1 and the second Q1 = Q R2, so the block's factor is R2 R1.
void finish_cholqr2(std::vector<Slab>& slabs, int block)
{
    const dim3 threads(
        static_cast<unsigned>(block), static_cast<unsigned>(block));
    for (Slab& slab : slabs) {
        CUDA_CHECK(cudaSetDevice(slab.device));
        CUDA_CHECK(trsm_tallskinny(
            slab.G, slab.B, slab.ld, block, slab.sm_count, 8, slab.stream));
        ca_combine_upper<<<1, threads, 0, slab.stream>>>(
            slab.G, slab.R1, slab.local_R, block);
        CUDA_CHECK(cudaGetLastError());
    }
}

/// Save the first pass's triangular factor before the second overwrites the
/// Gram buffer it shares.
void record_first_factor(std::vector<Slab>& slabs, int block)
{
    const std::size_t factor_bytes =
        static_cast<std::size_t>(block) * static_cast<std::size_t>(block)
        * sizeof(double);
    for (Slab& slab : slabs) {
        CUDA_CHECK(cudaSetDevice(slab.device));
        CUDA_CHECK(cudaMemcpyAsync(
            slab.R1, slab.G, factor_bytes,
            cudaMemcpyDeviceToDevice, slab.stream));
        CUDA_CHECK(trsm_tallskinny(
            slab.G, slab.B, slab.ld, block, slab.sm_count, 8, slab.stream));
    }
}

/// CholQR2 with the certificate read as it is decided.
///
/// The condition estimate and both Cholesky results are read as they are produced: three
/// blocking reads per block, so a rejected width costs no wasted work. At four checkpoints a
/// block this is the dominant term in the host-synchronization account, which is why the
/// deferred form exists to be measured against it.
[[nodiscard]] bool cholqr2_immediate(
    std::vector<Slab>& slabs, int block, double& kappa,
    bool agreement_strict, bool agreement_self_test, int step,
    ReductionStats& stats)
{
    global_gram(slabs, block, stats);
    for (Slab& slab : slabs) {
        CUDA_CHECK(cudaSetDevice(slab.device));
        ca_gram_condition<<<1, 1, 0, slab.stream>>>(
            slab.G, block, slab.kappa);
        CUDA_CHECK(cudaGetLastError());
    }
    Slab& root = slabs.front();
    CUDA_CHECK(cudaSetDevice(root.device));
    blocking_read(
        root, root.host_scalar, root.kappa, sizeof(double), stats);
    kappa = *root.host_scalar;
    const bool kappa_accepted = kappa < cholqr_kappa_limit();
    if (agreement_strict)
        check_certificate_agreement(
            slabs, block, 1, step, agreement_self_test, stats);
    if (!kappa_accepted) return false;

    factor_all(slabs, block, false);
    const int first_info = read_factor_info(slabs, false, stats);
    if (agreement_strict)
        check_certificate_agreement(
            slabs, block, 2, step, agreement_self_test, stats);
    if (first_info < 0)
        throw std::runtime_error(
            "Cholesky factorization rejected argument "
            + std::to_string(-first_info));
    if (first_info != 0) return false;
    record_first_factor(slabs, block);

    global_gram(slabs, block, stats);
    factor_all(slabs, block, true);
    const int second_info = read_factor_info(slabs, true, stats);
    if (agreement_strict)
        check_certificate_agreement(
            slabs, block, 3, step, agreement_self_test, stats);
    if (second_info < 0)
        throw std::runtime_error(
            "Cholesky factorization rejected argument "
            + std::to_string(-second_info));
    if (second_info != 0) return false;
    finish_cholqr2(slabs, block);
    return true;
}

/// CholQR2 with the certificate folded on the device and read once.
///
/// Same decisions as the immediate form, one round-trip instead of three: the block is
/// completed speculatively, and a rejected width discards it so the retry rebuilds the basis,
/// which the immediate form does anyway. Trades wasted work on the rare rejection for two
/// fewer host round-trips on every accepted block. The acceptance gate checks the decisions
/// exactly and the saved state within its declared tolerance.
[[nodiscard]] bool cholqr2_deferred(
    std::vector<Slab>& slabs, int block, double& kappa,
    bool agreement_strict, bool agreement_self_test, int step,
    ReductionStats& stats)
{
    global_gram(slabs, block, stats);
    for (Slab& slab : slabs) {
        CUDA_CHECK(cudaSetDevice(slab.device));
        ca_gram_condition<<<1, 1, 0, slab.stream>>>(
            slab.G, block, slab.kappa);
        CUDA_CHECK(cudaGetLastError());
    }

    factor_all(slabs, block, false);
    record_first_factor(slabs, block);
    global_gram(slabs, block, stats);
    factor_all(slabs, block, true);
    finish_cholqr2(slabs, block);

    for (Slab& slab : slabs) {
        CUDA_CHECK(cudaSetDevice(slab.device));
        ca_certificate_flag<<<1, 1, 0, slab.stream>>>(
            slab.kappa, cholqr_kappa_limit(),
            slab.potrf_info, slab.potrf_info2, slab.certificate);
        CUDA_CHECK(cudaGetLastError());
    }
    Slab& root = slabs.front();
    CUDA_CHECK(cudaSetDevice(root.device));
    blocking_read(
        root, root.host_certificate, root.certificate,
        sizeof(CertificateVerdict), stats);
    const CertificateVerdict verdict = *root.host_certificate;
    kappa = verdict.kappa;
    if (agreement_strict)
        check_certificate_agreement(
            slabs, block, 4, step, agreement_self_test, stats);
    if (verdict.rejected_argument < 0)
        throw std::runtime_error(
            "Cholesky factorization rejected argument "
            + std::to_string(-verdict.rejected_argument));
    return verdict.accepted != 0;
}

/// Dispatch to whichever certificate form the run selected.
[[nodiscard]] bool distributed_cholqr2(
    std::vector<Slab>& slabs, int block, double& kappa,
    const SolverOptions& options, int step, ReductionStats& stats)
{
    return options.deferred_certificate
        ? cholqr2_deferred(
            slabs, block, kappa, options.agreement_strict(),
            options.agreement_self_test, step, stats)
        : cholqr2_immediate(
            slabs, block, kappa, options.agreement_strict(),
            options.agreement_self_test, step, stats);
}

struct StepResult {
    int m = 0;
    double residual = std::numeric_limits<double>::infinity();
    double max_kappa = 0.0;
    int min_effective_s = 0;
    int fallback_blocks = 0;
    int certificate_retries = 0;
    int accepted_blocks = 0;
    bool converged = false;
};

/// Build one block, orthogonalize it against the accepted basis, and append it.
///
/// The distributed s-step cycle in one place: build with no reductions, project
/// against what is accepted, certify the conditioning, factor, and recover the
/// Hessenberg columns from the recurrence rather than from dot products.
/// Returns the certified width, which narrows when the certificate rejects the
/// block and the width falls back.
[[nodiscard]] double append_block(
    std::vector<Slab>& slabs, const GpuPdeOperator& op, int n,
    double scale, bool peer_halo, int world_gpus,
    int& filled, bool allow_fallback, const SolverOptions& options,
    int step, ReductionStats& stats, int* used_block)
{
    const int requested =
        std::min(slabs.front().block_max, slabs.front().target - filled);
    for (int block = requested; block >= 1; --block) {
        std::vector<const double*> input;
        input.reserve(slabs.size());
        for (const Slab& slab : slabs) input.push_back(slab.start);
        {
            // The block consumes columns 0 to block-1, so the recurrence needs
            // block-1 steps and a halo block-1 planes deep. The as-measured arm
            // requests one more of each and discards the extra column.
            const int steps = options.exact_depth ? block - 1 : block;
            const NvtxRange range("deep-halo matrix powers");
            build_matrix_powers(
                slabs, input, n, steps, op, scale,
                peer_halo, options.halo_overlap,
                options.shell_halo_initialization, options.shared_carveout,
                world_gpus,
                HaloPurpose::BlockBuild, stats);
        }
        project_twice(slabs, filled, block, stats);

        double kappa = 0.0;
        const bool certificate_accepted = distributed_cholqr2(
            slabs, block, kappa, options, step, stats);
        if (!certificate_accepted) {
            if (allow_fallback) continue;
            throw std::runtime_error("distributed block orthogonalization failed");
        }

        for (Slab& slab : slabs) {
            CUDA_CHECK(cudaSetDevice(slab.device));
            ca_assemble_block<<<1, 32, 0, slab.stream>>>(
                slab.H, slab.target, slab.target - 1,
                slab.C, slab.target, slab.local_R, filled, block,
                0, 0.0, 0.0);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(
                slab.V + static_cast<int64_t>(filled) * slab.ld,
                slab.B,
                static_cast<std::size_t>(slab.ld)
                    * static_cast<std::size_t>(block) * sizeof(double),
                cudaMemcpyDeviceToDevice, slab.stream));
        }
        filled += block;

        if (filled < slabs.front().target) {
            input.clear();
            for (const Slab& slab : slabs)
                input.push_back(
                    slab.V + static_cast<int64_t>(filled - 1) * slab.ld);
            {
                const NvtxRange range("boundary operator transition");
                build_matrix_powers(
                    slabs, input, n, 1, op, scale,
                    peer_halo, options.halo_overlap,
                    options.shell_halo_initialization, options.shared_carveout,
                    world_gpus,
                    HaloPurpose::Transition, stats);
            }
            for (Slab& slab : slabs) {
                CUDA_CHECK(cudaSetDevice(slab.device));
                CUDA_CHECK(cudaMemcpyAsync(
                    slab.start, slab.B + slab.ld,
                    static_cast<std::size_t>(slab.ld) * sizeof(double),
                    cudaMemcpyDeviceToDevice, slab.stream));
            }
        }
        if (used_block != nullptr) *used_block = block;
        return kappa;
    }
    throw std::runtime_error("matrix-powers block has no distributed certified width");
}

/// Advance one time step: build the basis, select m adaptively, and update.
///
/// Only the first slab evaluates the candidates, because the Hessenberg is
/// replicated and every rank would compute the same answer. That replication is
/// what the agreement checkpoints exist to assert rather than assume.
[[nodiscard]] StepResult advance_step(
    std::vector<Slab>& slabs, const GpuPdeOperator& op, int n,
    double scale, double tol, bool peer_halo, int world_gpus,
    const SolverOptions& options, int step,
    const std::vector<const double*>& current,
    const std::vector<double*>& next, ReductionStats& stats)
{
    constexpr double breakdown_tol = 1.0e-14;
    StepResult result;
    result.min_effective_s = slabs.front().block_max;
    const double beta = distributed_norm(slabs, current, stats);
    const bool breakdown = beta < breakdown_tol;
    if (options.agreement_enabled()) {
        const std::array<int, kDecisionFields> decisions{
            breakdown ? 1 : 0, slabs.front().block_max, 0, 0};
        check_decision_agreement(
            slabs, decisions, AgreementPoint::Breakdown, step,
            options.agreement_strict(), stats);
    }
    if (breakdown) {
        for (std::size_t rank = 0; rank < slabs.size(); ++rank) {
            Slab& slab = slabs[rank];
            CUDA_CHECK(cudaSetDevice(slab.device));
            CUDA_CHECK(cudaMemcpyAsync(
                next[rank], current[rank],
                static_cast<std::size_t>(slab.ld) * sizeof(double),
                cudaMemcpyDeviceToDevice, slab.stream));
        }
        result.m = 1;
        result.residual = 0.0;
        result.max_kappa = 1.0;
        result.converged = true;
        return result;
    }

    const double inv_beta = 1.0 / beta;
    for (std::size_t rank = 0; rank < slabs.size(); ++rank) {
        Slab& slab = slabs[rank];
        CUDA_CHECK(cudaSetDevice(slab.device));
        CUDA_CHECK(cudaMemcpyAsync(
            slab.start, current[rank],
            static_cast<std::size_t>(slab.ld) * sizeof(double),
            cudaMemcpyDeviceToDevice, slab.stream));
        CUBLAS_CHECK(cublasDscal(
            slab.blas, static_cast<int>(slab.ld),
            &inv_beta, slab.start, 1));
        CUDA_CHECK(cudaMemsetAsync(
            slab.H, 0,
            static_cast<std::size_t>(slab.target)
                * static_cast<std::size_t>(slab.target - 1) * sizeof(double),
            slab.stream));
    }

    int filled = 0;
    int checked = 0;
    while (filled < slabs.front().target && !result.converged) {
        const int requested =
            std::min(slabs.front().block_max, slabs.front().target - filled);
        int used = requested;
        const double kappa = append_block(
            slabs, op, n, scale, peer_halo, world_gpus,
            filled, true, options, step, stats, &used);
        result.max_kappa = std::max(result.max_kappa, kappa);
        ++result.accepted_blocks;
        if (used < requested) {
            ++result.fallback_blocks;
            result.certificate_retries += requested - used;
            result.min_effective_s = std::min(result.min_effective_s, used);
        }

        const int available =
            std::min(filled - 1, slabs.front().target - 1);
        if (available > checked) {
            const NvtxRange range("root residual and replicated exponential");
            Slab& root = slabs.front();
            CUDA_CHECK(cudaSetDevice(root.device));
            ca_expm_candidates<kGpuCaMaxM><<<1, 256, 0, root.stream>>>(
                root.H, root.target, checked + 1, available,
                beta, tol, root.f, root.candidate);
            CUDA_CHECK(cudaGetLastError());
            blocking_read(
                root, root.host_candidate, root.candidate,
                sizeof(GpuCaCandidateResult), stats);
            const GpuCaCandidateResult root_candidate = *root.host_candidate;
            const bool candidate_finite =
                std::isfinite(root_candidate.residual);
            result.residual = candidate_finite
                ? root_candidate.residual
                : std::numeric_limits<double>::infinity();
            result.m = candidate_finite
                ? (root_candidate.selected_m > 0
                    ? root_candidate.selected_m : available)
                : -1;
            result.converged =
                candidate_finite && root_candidate.selected_m > 0;
            // Per block only in strict mode. The step mode makes the same
            // comparison once, after the loop, on the decisions the step
            // actually finished with.
            if (options.agreement_strict()) {
                const std::array<int, kDecisionFields> decisions{
                    result.m, used,
                    candidate_finite
                        ? (result.converged ? 1 : 0) : -1,
                    result.fallback_blocks};
                check_decision_agreement(
                    slabs, decisions, AgreementPoint::Candidate,
                    step, true, stats);
            }
            if (!candidate_finite)
                throw std::runtime_error(
                    "non-finite Arnoldi residual estimate");
            for (Slab& slab : slabs) {
                CUDA_CHECK(cudaSetDevice(slab.device));
                ca_expm_action<kGpuCaMaxM><<<1, 256, 0, slab.stream>>>(
                    slab.H, slab.target, result.m, slab.f, nullptr, nullptr);
                CUDA_CHECK(cudaGetLastError());
            }
        }
        checked = available;
    }

    if (options.agreement_enabled() && !options.agreement_strict()) {
        const std::array<int, kDecisionFields> decisions{
            result.m, result.min_effective_s,
            result.converged ? 1 : 0, result.fallback_blocks};
        check_decision_agreement(
            slabs, decisions, AgreementPoint::Candidate, step, false, stats);
    }

    const double alpha = beta;
    const double zero = 0.0;
    const NvtxRange update_range("local state update");
    for (std::size_t rank = 0; rank < slabs.size(); ++rank) {
        Slab& slab = slabs[rank];
        CUDA_CHECK(cudaSetDevice(slab.device));
        CUBLAS_CHECK(cublasDgemv(
            slab.blas, CUBLAS_OP_N, static_cast<int>(slab.ld), result.m,
            &alpha, slab.V, static_cast<int>(slab.ld), slab.f, 1,
            &zero, next[rank], 1));
    }
    return result;
}

struct IntegratorResult {
    Eigen::VectorXd state;
    double elapsed_ms = 0.0;
    double avg_m = 0.0;
    double max_residual = 0.0;
    double max_kappa = 0.0;
    int min_m = 0;
    int max_m = 0;
    int min_effective_s = 0;
    int fallback_blocks = 0;
    int first_fallback_step = 0;
    int64_t accepted_blocks = 0;
    int64_t certificate_retries = 0;
    int unconverged = 0;
    ReductionStats reductions;
};

/// Integrate to expiry and return the final state with the run's full account.
[[nodiscard]] IntegratorResult solve(
    const Args& args, std::vector<Slab>& slabs,
    const GpuPdeOperator& op, const Eigen::VectorXd& initial,
    double operator_scale, const SolverOptions& options,
    bool peer_halo, int world_gpus, int mpi_rank, int mpi_size)
{
    const int64_t n2 = static_cast<int64_t>(args.n) * args.n;
    for (Slab& slab : slabs) {
        std::vector<double> local(static_cast<std::size_t>(slab.ld), 0.0);
        const int64_t offset = static_cast<int64_t>(slab.z_begin) * n2;
        std::copy_n(
            initial.data() + offset, slab.local_N, local.data());
        std::copy_n(
            initial.data() + initial.size() - 3, 3,
            local.data() + slab.local_N);
        CUDA_CHECK(cudaSetDevice(slab.device));
        CUDA_CHECK(cudaMemcpy(
            slab.state, local.data(),
            static_cast<std::size_t>(slab.ld) * sizeof(double),
            cudaMemcpyHostToDevice));
    }
    sync_all(slabs);

    std::vector<const double*> warm_current;
    std::vector<double*> warm_next;
    for (const Slab& slab : slabs) {
        warm_current.push_back(slab.state);
        warm_next.push_back(slab.action);
    }
    ReductionStats warm_stats;
    (void)advance_step(
        slabs, op, args.n, operator_scale,
        args.tol, peer_halo, world_gpus, options, -1,
        warm_current, warm_next, warm_stats);
    sync_all(slabs);

    for (Slab& slab : slabs) {
        std::vector<double> local(static_cast<std::size_t>(slab.ld), 0.0);
        const int64_t offset = static_cast<int64_t>(slab.z_begin) * n2;
        std::copy_n(
            initial.data() + offset, slab.local_N, local.data());
        std::copy_n(
            initial.data() + initial.size() - 3, 3,
            local.data() + slab.local_N);
        CUDA_CHECK(cudaSetDevice(slab.device));
        CUDA_CHECK(cudaMemcpy(
            slab.state, local.data(),
            static_cast<std::size_t>(slab.ld) * sizeof(double),
            cudaMemcpyHostToDevice));
        // Clear whatever the warm-up latched: only the timed steps are reported.
        CUDA_CHECK(cudaMemsetAsync(
            slab.verdict, 0, kDecisionFields * sizeof(int), slab.stream));
        slab.block_max = args.s;
    }
    sync_all(slabs);

    IntegratorResult result;
    result.state = Eigen::VectorXd::Zero(initial.size());
    result.min_m = args.m;
    result.min_effective_s = args.s;
    std::vector<double*> current;
    std::vector<double*> next;
    for (Slab& slab : slabs) {
        current.push_back(slab.state);
        next.push_back(slab.action);
    }

#ifdef CAKSM_HAVE_MPI
    if (mpi_size > 1) MPI_Barrier(MPI_COMM_WORLD);
#else
    (void)mpi_rank;
    (void)mpi_size;
#endif
    const auto begin = Clock::now();
    int64_t m_sum = 0;
    bool profile_range_open = false;
    for (int step = 0; step < args.steps; ++step) {
        if (step == args.profile_start) {
            nvtxRangePushA("profile window");
            profile_range_open = true;
        }
        std::vector<const double*> current_const(
            current.begin(), current.end());
        const StepResult step_result = advance_step(
            slabs, op, args.n, operator_scale,
            args.tol, peer_halo, world_gpus, options, step,
            current_const, next, result.reductions);
        m_sum += step_result.m;
        result.min_m = std::min(result.min_m, step_result.m);
        result.max_m = std::max(result.max_m, step_result.m);
        result.max_residual =
            std::max(result.max_residual, step_result.residual);
        result.max_kappa =
            std::max(result.max_kappa, step_result.max_kappa);
        result.min_effective_s =
            std::min(result.min_effective_s, step_result.min_effective_s);
        result.fallback_blocks += step_result.fallback_blocks;
        result.accepted_blocks += step_result.accepted_blocks;
        result.certificate_retries += step_result.certificate_retries;
        if (step_result.fallback_blocks > 0
            && result.first_fallback_step == 0)
            result.first_fallback_step = step + 1;
        if (step_result.fallback_blocks > 0)
            for (Slab& slab : slabs)
                slab.block_max =
                    std::min(slab.block_max, step_result.min_effective_s);
        if (!step_result.converged) ++result.unconverged;
        std::swap(current, next);
        if (profile_range_open
            && step + 1 == args.profile_start + args.profile_steps) {
            nvtxRangePop();
            profile_range_open = false;
        }
    }
    sync_all(slabs);
    result.elapsed_ms = Milliseconds(Clock::now() - begin).count();
#ifdef CAKSM_HAVE_MPI
    if (mpi_size > 1) {
        double elapsed_max = 0.0;
        double host_sync_max = 0.0;
        MPI_Allreduce(
            &result.elapsed_ms, &elapsed_max, 1, MPI_DOUBLE,
            MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(
            &result.reductions.host_sync_ms, &host_sync_max, 1, MPI_DOUBLE,
            MPI_MAX, MPI_COMM_WORLD);
        result.elapsed_ms = elapsed_max;
        result.reductions.host_sync_ms = host_sync_max;
    }
#endif
    result.avg_m =
        static_cast<double>(m_sum) / static_cast<double>(args.steps);

    Eigen::Vector3d root_tail = Eigen::Vector3d::Zero();
    for (Slab& slab : slabs) {
        CUDA_CHECK(cudaSetDevice(slab.device));
        Eigen::VectorXd local(slab.ld);
        CUDA_CHECK(cudaMemcpy(
            local.data(), current[static_cast<std::size_t>(slab.rank)],
            static_cast<std::size_t>(slab.ld) * sizeof(double),
            cudaMemcpyDeviceToHost));
        const int64_t offset = static_cast<int64_t>(slab.z_begin) * n2;
        result.state.segment(offset, slab.local_N) =
            local.head(slab.local_N);
        if (slab.rank == 0) root_tail = local.tail(3);
        else if ((local.tail(3) - root_tail).lpNorm<Eigen::Infinity>()
                 > 32.0 * std::numeric_limits<double>::epsilon())
            throw std::runtime_error("replicated augmented states diverged");
    }
    result.state.tail(3) = root_tail;
#ifdef CAKSM_HAVE_MPI
    if (mpi_size > 1) {
        if (mpi_rank != 0) result.state.tail(3).setZero();
        MPI_Allreduce(
            MPI_IN_PLACE, result.state.data(),
            static_cast<int>(result.state.size()), MPI_DOUBLE,
            MPI_SUM, MPI_COMM_WORLD);
    }
#endif
    return result;
}

/// Allocate one slab's device and pinned buffers and bring up its handles.
///
/// The basis keeps s+1 columns even though exact-depth consumes only s: both
/// arms live in one binary and share the frozen allocation model, so shrinking
/// it would silently move the recorded largest-common-grid prediction.
void allocate_slab(
    Slab& slab, int n, int m, int s, bool rainbow,
    const std::vector<double>& face_b)
{
    CUDA_CHECK(cudaSetDevice(slab.device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, slab.device));
    slab.sm_count = prop.multiProcessorCount;
    slab.target = m + 1;
    slab.block_max = s;
    slab.local_N =
        static_cast<int64_t>(slab.z_count) * n * n;
    slab.ld = slab.local_N + 3;
    slab.reduction_rows =
        slab.global_rank == 0 ? slab.ld : slab.local_N;

    CUDA_CHECK(cudaStreamCreate(&slab.stream));
    CUDA_CHECK(cudaStreamCreateWithFlags(
        &slab.halo_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(
        &slab.halo_ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(
        &slab.halo_received, cudaEventDisableTiming));
    CUBLAS_CHECK(cublasCreate(&slab.blas));
    CUSOLVER_CHECK(cusolverDnCreate(&slab.solver));
    CUBLAS_CHECK(cublasSetStream(slab.blas, slab.stream));
    CUSOLVER_CHECK(cusolverDnSetStream(slab.solver, slab.stream));

    const std::size_t ld = static_cast<std::size_t>(slab.ld);
    const std::size_t target = static_cast<std::size_t>(slab.target);
    const std::size_t block = static_cast<std::size_t>(s);
    CUDA_CHECK(cudaMalloc(&slab.state, ld * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&slab.start, ld * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&slab.action, ld * sizeof(double)));
    CUDA_CHECK(cudaMalloc(
        &slab.halo,
        static_cast<std::size_t>(slab.z_count + 2 * s)
            * static_cast<std::size_t>(n) * static_cast<std::size_t>(n)
            * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&slab.B, ld * (block + 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&slab.V, ld * target * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&slab.H, target * (target - 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&slab.C, target * block * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&slab.C2, target * block * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&slab.G, block * block * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&slab.R1, block * block * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&slab.local_R, block * block * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&slab.kappa, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&slab.f, target * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&slab.scalar, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&slab.candidate, sizeof(GpuCaCandidateResult)));
    CUDA_CHECK(cudaMalloc(&slab.potrf_info, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&slab.potrf_info2, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&slab.certificate, sizeof(CertificateVerdict)));
    CUDA_CHECK(cudaMalloc(&slab.decision, kDecisionFields * sizeof(int)));
    CUDA_CHECK(cudaMalloc(
        &slab.decision_extreme, kDecisionFields * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&slab.verdict, kDecisionFields * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(
        &slab.host_verdict, kDecisionFields * sizeof(int)));
    std::fill_n(slab.host_verdict, kDecisionFields, 0);
    CUDA_CHECK(cudaMallocHost(&slab.host_scalar, sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&slab.host_info, sizeof(int)));
    CUDA_CHECK(cudaMallocHost(
        &slab.host_certificate, sizeof(CertificateVerdict)));
    CUDA_CHECK(cudaMallocHost(
        &slab.host_candidate, sizeof(GpuCaCandidateResult)));
    CUDA_CHECK(cudaMemsetAsync(
        slab.verdict, 0, kDecisionFields * sizeof(int), slab.stream));
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(
        slab.solver, CUBLAS_FILL_MODE_UPPER, s, slab.G, s,
        &slab.potrf_lwork));
    CUDA_CHECK(cudaMalloc(
        &slab.potrf_work,
        static_cast<std::size_t>(slab.potrf_lwork) * sizeof(double)));
    if (!rainbow) {
        CUDA_CHECK(cudaMalloc(
            &slab.face_b, face_b.size() * sizeof(double)));
        CUDA_CHECK(cudaMemcpyAsync(
            slab.face_b, face_b.data(), face_b.size() * sizeof(double),
            cudaMemcpyHostToDevice, slab.stream));
    }
}

/// Release everything allocate_slab took, handles before buffers.
void free_slab(Slab& slab)
{
    CUDA_CHECK(cudaSetDevice(slab.device));
    CUDA_CHECK(cudaFree(slab.state));
    CUDA_CHECK(cudaFree(slab.start));
    CUDA_CHECK(cudaFree(slab.action));
    CUDA_CHECK(cudaFree(slab.halo));
    CUDA_CHECK(cudaFree(slab.B));
    CUDA_CHECK(cudaFree(slab.V));
    CUDA_CHECK(cudaFree(slab.H));
    CUDA_CHECK(cudaFree(slab.C));
    CUDA_CHECK(cudaFree(slab.C2));
    CUDA_CHECK(cudaFree(slab.G));
    CUDA_CHECK(cudaFree(slab.R1));
    CUDA_CHECK(cudaFree(slab.local_R));
    CUDA_CHECK(cudaFree(slab.potrf_work));
    CUDA_CHECK(cudaFree(slab.kappa));
    CUDA_CHECK(cudaFree(slab.f));
    CUDA_CHECK(cudaFree(slab.scalar));
    CUDA_CHECK(cudaFree(slab.candidate));
    CUDA_CHECK(cudaFree(slab.potrf_info));
    CUDA_CHECK(cudaFree(slab.potrf_info2));
    CUDA_CHECK(cudaFree(slab.certificate));
    CUDA_CHECK(cudaFree(slab.decision));
    CUDA_CHECK(cudaFree(slab.decision_extreme));
    CUDA_CHECK(cudaFree(slab.verdict));
    CUDA_CHECK(cudaFreeHost(slab.host_verdict));
    CUDA_CHECK(cudaFreeHost(slab.host_scalar));
    CUDA_CHECK(cudaFreeHost(slab.host_info));
    CUDA_CHECK(cudaFreeHost(slab.host_certificate));
    CUDA_CHECK(cudaFreeHost(slab.host_candidate));
    if (slab.face_b != nullptr) CUDA_CHECK(cudaFree(slab.face_b));
    CUBLAS_CHECK(cublasDestroy(slab.blas));
    CUSOLVER_CHECK(cusolverDnDestroy(slab.solver));
    CUDA_CHECK(cudaEventDestroy(slab.halo_ready));
    CUDA_CHECK(cudaEventDestroy(slab.halo_received));
    CUDA_CHECK(cudaStreamDestroy(slab.halo_stream));
    CUDA_CHECK(cudaStreamDestroy(slab.stream));
    NCCL_CHECK(ncclCommDestroy(slab.comm));
}

/// Ask cuSOLVER how much scratch a width-s Cholesky needs on this device. Only
/// the library can answer, so the memory model takes it as an argument.
[[nodiscard]] int query_potrf_lwork(int device, int s)
{
    CUDA_CHECK(cudaSetDevice(device));
    cusolverDnHandle_t solver = nullptr;
    double* matrix = nullptr;
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnCreate(&solver));
    CUDA_CHECK(cudaMalloc(
        &matrix,
        static_cast<std::size_t>(s) * static_cast<std::size_t>(s)
            * sizeof(double)));
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(
        solver, CUBLAS_FILL_MODE_UPPER, s, matrix, s, &lwork));
    CUDA_CHECK(cudaFree(matrix));
    CUSOLVER_CHECK(cusolverDnDestroy(solver));
    return lwork;
}

/// Print the per-slab allocation model against this topology and stop,
/// allocating nothing. Deciding whether a grid fits before a job spends minutes
/// finding out that it does not.
void report_integrator_memory(
    const Args& args, int mpi_rank, int mpi_size)
{
    const int local_gpus = static_cast<int>(args.devices.size());
    const int world_gpus = local_gpus * mpi_size;
    int local_recommended = 1289;
    int local_fit = args.n >= world_gpus * args.s ? 1 : 0;
    int local_idle = 1;
    const double mib = 1024.0 * 1024.0;

    for (int local_rank = 0; local_rank < local_gpus; ++local_rank) {
        const int device = args.devices[static_cast<std::size_t>(local_rank)];
        const int global_rank = mpi_rank * local_gpus + local_rank;
        const int potrf_lwork = query_potrf_lwork(device, args.s);
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        CUDA_CHECK(cudaSetDevice(device));
        const DeviceContention contention = check_device_contention();
        report_contention(contention);
        if (contention.contended) local_idle = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        const std::uint64_t capacity =
            ca_integrator_memory::usable_bytes(
                free_bytes, total_bytes, args.memory_reserve);
        const auto estimator = [&](int n) {
            return ca_integrator_memory::distributed_slab(
                n, args.m, args.s, !args.rainbow,
                world_gpus, global_rank, potrf_lwork,
                sizeof(GpuCaCandidateResult), sizeof(CertificateVerdict),
                kDecisionFields);
        };
        const auto requested = estimator(args.n);
        const int recommended =
            ca_integrator_memory::largest_odd_grid(capacity, estimator);
        local_recommended = std::min(local_recommended, recommended);
        local_fit =
            local_fit && requested.total_bytes <= capacity ? 1 : 0;

        std::printf(
            "  memory process=%d global-rank=%d device=%d: total=%.3f MiB free=%.3f MiB usable=%.3f MiB\n",
            mpi_rank, global_rank, device,
            static_cast<double>(total_bytes) / mib,
            static_cast<double>(free_bytes) / mib,
            static_cast<double>(capacity) / mib);
        std::printf(
            "    candidate n=%d: vectors=%.3f MiB halo=%.3f MiB faces=%.3f MiB small=%.3f MiB total=%.3f MiB | %s | local maximum odd n=%d\n",
            args.n,
            static_cast<double>(requested.vector_bytes) / mib,
            static_cast<double>(requested.halo_bytes) / mib,
            static_cast<double>(requested.face_bytes) / mib,
            static_cast<double>(requested.small_bytes) / mib,
            static_cast<double>(requested.total_bytes) / mib,
            requested.total_bytes <= capacity ? "FIT" : "NO FIT",
            recommended);
    }

    int recommended = local_recommended;
    int candidate_fit = local_fit;
    int all_idle = local_idle;
#ifdef CAKSM_HAVE_MPI
    if (mpi_size > 1) {
        MPI_Allreduce(
            &local_recommended, &recommended, 1, MPI_INT,
            MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(
            &local_fit, &candidate_fit, 1, MPI_INT,
            MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(
            &local_idle, &all_idle, 1, MPI_INT,
            MPI_MIN, MPI_COMM_WORLD);
    }
#endif
    if (all_idle == 0)
        throw std::runtime_error(
            "memory report requires every device to be idle");
    if (mpi_rank == 0) {
        std::printf("CA integrator device-memory report\n");
        std::printf(
            "  topology=%d GPU(s), %d process(es), %d local GPU(s)/process | option=%s | m=%d | s=%d | reserve=%.1f%%\n",
            world_gpus, mpi_size, local_gpus,
            args.rainbow ? "rainbow" : "basket",
            args.m, args.s, 100.0 * args.memory_reserve);
        std::printf(
            "  candidate n=%d: %s across every slab\n",
            args.n, candidate_fit != 0 ? "FIT" : "NO FIT");
        std::printf(
            "  recommended largest common odd n for this topology: %d\n",
            recommended);
        std::printf(
            "MEMORY_REPORT topology=%d_gpu processes=%d local_gpus=%d option=%s s=%d candidate_n=%d candidate_fit=%s recommended_n=%d reserve=%.6f\n",
            world_gpus, mpi_size, local_gpus,
            args.rainbow ? "rainbow" : "basket", args.s, args.n,
            candidate_fit != 0 ? "yes" : "no",
            recommended, args.memory_reserve);
    }
}

/// Where the independent referee's state for this grid and option lives.
[[nodiscard]] std::string referee_path(
    const std::string& dir, int n, bool rainbow)
{
    return dir + "/referee_n" + std::to_string(n)
               + (rainbow ? "_rainbow" : "_basket") + ".bin";
}

/// Read a state file: an int64 length followed by that many doubles.
[[nodiscard]] Eigen::VectorXd load_vector_file(const std::string& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open vector file: " + path);
    int64_t size = 0;
    stream.read(reinterpret_cast<char*>(&size), sizeof(size));
    if (!stream || size <= 0)
        throw std::runtime_error("invalid vector file: " + path);
    Eigen::VectorXd vector(size);
    stream.read(
        reinterpret_cast<char*>(vector.data()),
        static_cast<std::streamsize>(size * sizeof(double)));
    if (!stream) throw std::runtime_error("truncated vector file: " + path);
    return vector;
}

/// Write a state file in the layout load_vector_file reads.
void save_vector_file(const std::string& path, const Eigen::VectorXd& vector)
{
    std::ofstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot create state file: " + path);
    const int64_t size = vector.size();
    stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
    stream.write(
        reinterpret_cast<const char*>(vector.data()),
        static_cast<std::streamsize>(size * sizeof(double)));
    if (!stream) throw std::runtime_error("failed to write state file: " + path);
}

} // namespace

int main(int argc, char** argv)
{
    int mpi_rank = 0;
    int mpi_size = 1;
#ifdef CAKSM_HAVE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif
    int exit_code = EXIT_SUCCESS;
    try {
        const Args args = parse_args(argc, argv);
        if (args.require_mpi && mpi_size < 2) {
#ifdef CAKSM_HAVE_MPI
            throw std::runtime_error(
                "--require-mpi requested, but MPI_COMM_WORLD has fewer than two ranks; "
                "the launcher created singleton MPI worlds");
#else
            throw std::runtime_error(
                "--require-mpi requested, but this executable was built without MPI support; "
                "load MPI, reconfigure CMake, and rebuild");
#endif
        }
        CUDA_CHECK(cudaSetDevice(args.devices[0]));
        if (mpi_rank == 0) report_toolkit();
        if (args.memory_report) {
            report_integrator_memory(args, mpi_rank, mpi_size);
#ifdef CAKSM_HAVE_MPI
            MPI_Finalize();
#endif
            return EXIT_SUCCESS;
        }

        CaPricingModel model;
        model.expiry = args.expiry;
        SolverOptions options;
        options.exact_depth = args.arm == "exact-depth";
        options.deferred_certificate = args.certificate == "deferred";
        options.halo_overlap = args.halo_overlap == "on";
        options.shell_halo_initialization = args.halo_memset == "shells";
        options.shared_carveout = args.shared_carveout == "max"
            ? MpkSharedCarveout::Maximum : MpkSharedCarveout::Default;
        // Columns 0 to s-1 reach the basis, so the recurrence needs s-1 steps and
        // a halo s-1 planes deep. The as-measured arm asks for one more of each.
        const int block_depth = options.exact_depth ? args.s - 1 : args.s;
        // The one-GPU solver integrates to model.expiry. Taking the same scale here
        // keeps the two solvers on one final time when expiry stops being 1.
        const double operator_scale =
            model.expiry / static_cast<double>(args.steps);

        const PDESystem sys =
            build_gpu_pde_system(args.n, model, args.rainbow);
        const GpuPdeOperator op =
            make_gpu_pde_operator(sys, model, args.rainbow);
        const std::vector<double> face_b = make_gpu_face_b(sys, model);
        const int64_t global_ld = static_cast<int64_t>(sys.N) + 3;

        Eigen::VectorXd initial = Eigen::VectorXd::Zero(global_ld);
        initial.head(sys.N) = sys.u0;
        if (!args.rainbow) initial.tail(3) = make_s_vec(0.0);

        const int local_gpus = static_cast<int>(args.devices.size());
        const int world_gpus = local_gpus * mpi_size;
        // A one-slab run has no peer whose adaptive schedule can diverge. Keeping
        // the NCCL min/max and pinned verdict round-trips enabled there would turn
        // the one-GPU/one-slab comparison into a measurement of a vacuous
        // correctness check.
        options.agreement =
            world_gpus > 1
                ? (args.agreement == "strict" ? AgreementMode::Strict
                   : args.agreement == "step" ? AgreementMode::Step
                   : AgreementMode::Off)
                : AgreementMode::Off;
        if (args.agreement_self_test && !options.agreement_enabled())
            throw std::invalid_argument(
                "--agreement-self-test requires at least two global GPU ranks");
        options.agreement_self_test = args.agreement_self_test;
        if (args.n < world_gpus * args.s)
            throw std::invalid_argument(
                "every global slab must own at least s planes");
        std::vector<Slab> slabs(local_gpus);
        for (int local_rank = 0; local_rank < local_gpus; ++local_rank) {
            Slab& slab = slabs[static_cast<std::size_t>(local_rank)];
            slab.rank = local_rank;
            slab.global_rank = mpi_rank * local_gpus + local_rank;
            slab.device = args.devices[static_cast<std::size_t>(local_rank)];
            slab.z_begin =
                slab.global_rank * args.n / world_gpus;
            const int z_end =
                (slab.global_rank + 1) * args.n / world_gpus;
            slab.z_count = z_end - slab.z_begin;
        }

        std::vector<ncclComm_t> comms(
            static_cast<std::size_t>(local_gpus), nullptr);
        std::vector<int> devices = args.devices;
        if (mpi_size == 1) {
            NCCL_CHECK(ncclCommInitAll(
                comms.data(), local_gpus, devices.data()));
        } else {
#ifdef CAKSM_HAVE_MPI
            ncclUniqueId id{};
            if (mpi_rank == 0) NCCL_CHECK(ncclGetUniqueId(&id));
            MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
            NCCL_CHECK(ncclGroupStart());
            for (int local_rank = 0; local_rank < local_gpus; ++local_rank) {
                CUDA_CHECK(cudaSetDevice(devices[local_rank]));
                NCCL_CHECK(ncclCommInitRank(
                    &comms[local_rank], world_gpus, id,
                    mpi_rank * local_gpus + local_rank));
            }
            NCCL_CHECK(ncclGroupEnd());
#endif
        }

        // Only ADJACENT slabs exchange halos, because the decomposition is one-dimensional in
        // z, so the peer path needs bidirectional access across the N-1 adjacent pairs and not
        // the full N x N matrix. Every pair is checked: one pair without reciprocal access is
        // enough to make the whole chain fall back to NCCL, since a halo that arrives for some
        // slabs and not others is worse than one that never takes the fast path.
        bool peer_capable = local_gpus > 1;
        for (int rank = 0; rank + 1 < local_gpus && peer_capable; ++rank) {
            int forward = 0;
            int reverse = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(
                &forward, args.devices[static_cast<std::size_t>(rank)],
                args.devices[static_cast<std::size_t>(rank + 1)]));
            CUDA_CHECK(cudaDeviceCanAccessPeer(
                &reverse, args.devices[static_cast<std::size_t>(rank + 1)],
                args.devices[static_cast<std::size_t>(rank)]));
            peer_capable = forward != 0 && reverse != 0;
        }
        if (args.halo_backend == "peer"
            && (mpi_size > 1 || !peer_capable))
            throw std::runtime_error(
                "peer halo backend requires one node and bidirectional peer access");
        const bool peer_halo =
            mpi_size == 1
            && (args.halo_backend == "peer"
                || (args.halo_backend == "auto" && peer_capable));
        if (peer_halo) {
            // Each slab enables access to the neighbors it will read from. Enabling is
            // directional, so both ends of every adjacent pair do it; `1 - rank` served that
            // for two devices and names the wrong device for any other count.
            for (int rank = 0; rank < local_gpus; ++rank) {
                CUDA_CHECK(cudaSetDevice(devices[static_cast<std::size_t>(rank)]));
                for (const int neighbor : {rank - 1, rank + 1}) {
                    if (neighbor < 0 || neighbor >= local_gpus) continue;
                    const cudaError_t status = cudaDeviceEnablePeerAccess(
                        devices[static_cast<std::size_t>(neighbor)], 0);
                    if (status == cudaErrorPeerAccessAlreadyEnabled)
                        (void)cudaGetLastError();
                    else
                        CUDA_CHECK(status);
                }
            }
        }

        int local_recordable = 1;
        for (std::size_t rank = 0; rank < slabs.size(); ++rank) {
            Slab& slab = slabs[rank];
            slab.comm = comms[rank];
            CUDA_CHECK(cudaSetDevice(slab.device));
            cudaDeviceProp prop{};
            CUDA_CHECK(cudaGetDeviceProperties(&prop, slab.device));
            std::printf(
                "  global rank %d: process %d device %d %s | z=[%d,%d)\n",
                slab.global_rank, mpi_rank, slab.device, prop.name,
                slab.z_begin, slab.z_begin + slab.z_count);
            const DeviceContention contention = check_device_contention();
            report_contention(contention);
            local_recordable =
                local_recordable && !contention.contended ? 1 : 0;
            allocate_slab(
                slab, args.n, args.m, args.s, args.rainbow, face_b);
        }
        int recordable_int = local_recordable;
#ifdef CAKSM_HAVE_MPI
        if (mpi_size > 1)
            MPI_Allreduce(
                &local_recordable, &recordable_int, 1, MPI_INT,
                MPI_MIN, MPI_COMM_WORLD);
#endif
        const bool recordable = recordable_int != 0;

        int distinct_hosts = 1;
#ifdef CAKSM_HAVE_MPI
        if (mpi_size > 1) {
            char host[MPI_MAX_PROCESSOR_NAME] = {};
            int host_length = 0;
            MPI_Get_processor_name(host, &host_length);
            std::vector<char> hosts(
                static_cast<std::size_t>(mpi_size) * MPI_MAX_PROCESSOR_NAME, 0);
            MPI_Allgather(
                host, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                hosts.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                MPI_COMM_WORLD);
            distinct_hosts = 0;
            for (int rank = 0; rank < mpi_size; ++rank) {
                const char* candidate =
                    hosts.data()
                    + static_cast<std::size_t>(rank) * MPI_MAX_PROCESSOR_NAME;
                bool seen = false;
                for (int prior = 0; prior < rank; ++prior) {
                    const char* previous =
                        hosts.data()
                        + static_cast<std::size_t>(prior) * MPI_MAX_PROCESSOR_NAME;
                    if (std::strcmp(candidate, previous) == 0) {
                        seen = true;
                        break;
                    }
                }
                if (!seen) ++distinct_hosts;
            }
            if (distinct_hosts != mpi_size)
                throw std::runtime_error(
                    "multi-node mode requires one MPI process per node");
        }
#endif

        std::vector<int64_t> local_nnz(
            static_cast<std::size_t>(local_gpus), 0);
        const int64_t n2 = static_cast<int64_t>(args.n) * args.n;
        for (std::size_t local_rank = 0;
             local_rank < slabs.size(); ++local_rank) {
            // The placement model only needs a conservative traffic count.
            // The matrix-free 19-point stencil has at most 19 entries per row.
            local_nnz[local_rank] = 19 * slabs[local_rank].local_N;
        }
        const int model_rank = static_cast<int>(
            std::distance(
                local_nnz.begin(),
                std::max_element(local_nnz.begin(), local_nnz.end())));
        int64_t model_nnz = local_nnz[model_rank];
        int64_t model_rows = slabs[model_rank].local_N;
#ifdef CAKSM_HAVE_MPI
        if (mpi_size > 1) {
            int64_t global_model_nnz = 0;
            int64_t global_model_rows = 0;
            MPI_Allreduce(
                &model_nnz, &global_model_nnz, 1, MPI_INT64_T,
                MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(
                &model_rows, &global_model_rows, 1, MPI_INT64_T,
                MPI_MAX, MPI_COMM_WORLD);
            model_nnz = global_model_nnz;
            model_rows = global_model_rows;
        }
#endif
        cudaDeviceProp model_prop{};
        CUDA_CHECK(cudaSetDevice(slabs.front().device));
        CUDA_CHECK(cudaGetDeviceProperties(&model_prop, slabs.front().device));
        const GpuMachine& machine =
            lookup_gpu_machine_for_device(model_prop.name);
        // The allocation this launch actually received, not a shape the preset assumed.
        // It decides which rungs exist here, and therefore whether the cost about to be
        // published describes a link this run crosses.
        const GpuAllocation observed =
            allocation_from_world(world_gpus, distinct_hosts);
        const ReductionTier collective_tier = ::collective_tier(observed);
        if (!tier_cost_available(machine, observed, collective_tier))
            throw std::runtime_error(
                "the " + std::string(tier_name(collective_tier))
                + " collective tier required by " + std::to_string(world_gpus)
                + " GPU(s) across " + std::to_string(distinct_hosts)
                + " node(s) is unreachable or uncalibrated for "
                + std::string(machine.key)
                + "; refusing to publish a cost from a lower rung");
        const GpuRegimePoint placement = place_gpu(
            machine, machine.sm_count, collective_tier,
            model_nnz, model_rows,
            args.m, 1.0, Precision::FP64, args.s);
        const int64_t target_columns =
            static_cast<int64_t>(args.m) + 1;
        const int64_t certified_width_blocks =
            (static_cast<int64_t>(args.m) + 1 + args.s - 1) / args.s;
        // With the requested width certified, each block uses at most two
        // projection and two Gram collectives. Agreement adds one min/max pair
        // at breakdown and at the candidate branch. The immediate certificate
        // has three internal branches whose schedules must agree; the deferred
        // certificate has one final verdict branch.
        const int64_t certificate_agreement_per_attempt =
            args.certificate == "deferred" ? 2 : 6;
        const int64_t certified_width_collectives =
            options.agreement_strict()
                ? (6 + certificate_agreement_per_attempt)
                    * certified_width_blocks + 1
                : options.agreement_enabled()
                    ? 4 * certified_width_blocks - 1 + 4
                    : 4 * certified_width_blocks - 1;
        // A strict adaptive bound also permits every requested width down to one
        // to be rejected independently at every accepted column.
        const int64_t max_attempts =
            target_columns * static_cast<int64_t>(args.s);
        const int64_t strict_collective_upper =
            options.agreement_strict()
                ? 3
                    + (4 + certificate_agreement_per_attempt) * max_attempts
                    + 2 * target_columns
                : options.agreement_enabled()
                    ? 5 + 4 * max_attempts
                    : 1 + 4 * max_attempts;
        const double reduction_us =
            reduction_cost_s(machine, collective_tier) * 1.0e6;
        const double certified_width_collective_ms =
            static_cast<double>(certified_width_collectives)
            * reduction_us * 1.0e-3;
        const double strict_collective_ms =
            static_cast<double>(strict_collective_upper)
            * reduction_us * 1.0e-3;
        const double requested_halo_kib =
            static_cast<double>(2LL * block_depth * n2 * sizeof(double))
            / 1024.0;
        const double requested_halo_floor_us =
            machine.interconnect_bw_gbs > 0.0
                ? requested_halo_kib * 1024.0
                    / (machine.interconnect_bw_gbs * 1.0e9) * 1.0e6
                : 0.0;
        if (mpi_rank == 0) {
            std::printf("Distributed CA a-priori placement\n");
            std::printf(
                "  machine: %s selected from device \"%s\"\n",
                std::string(machine.key).c_str(), model_prop.name);
            std::printf(
                "  local model: rows=%lld nnz=%lld working-set=%.3f MiB | R_v=%.3f R_h=%.3f\n",
                static_cast<long long>(model_rows),
                static_cast<long long>(model_nnz),
                placement.working_set / (1024.0 * 1024.0),
                placement.rv, placement.rh);
            std::printf(
                "  %s reduction=%.3f us | m_max certified-width collectives=%lld (%.3f ms/step) | strict adaptive upper bound=%lld (%.3f ms/step)\n",
                tier_name(collective_tier), reduction_us,
                static_cast<long long>(certified_width_collectives),
                certified_width_collective_ms,
                static_cast<long long>(strict_collective_upper),
                strict_collective_ms);
            std::printf(
                "  requested halo=%.3f KiB/interface/block | bandwidth floor=%.3f us/interface/block\n\n",
                requested_halo_kib, requested_halo_floor_us);
        }

        double host_sync_probe_us = 0.0;
        for (Slab& slab : slabs)
            host_sync_probe_us =
                std::max(host_sync_probe_us, probe_host_sync_us(slab));
#ifdef CAKSM_HAVE_MPI
        if (mpi_size > 1) {
            double global_probe = 0.0;
            MPI_Allreduce(
                &host_sync_probe_us, &global_probe, 1, MPI_DOUBLE,
                MPI_MAX, MPI_COMM_WORLD);
            host_sync_probe_us = global_probe;
        }
#endif

        std::vector<IntegratorResult> runs;
        runs.reserve(static_cast<std::size_t>(args.repeats));
        for (int repeat = 0; repeat < args.repeats; ++repeat)
            runs.push_back(solve(
                args, slabs, op, initial, operator_scale, options, peer_halo,
                world_gpus, mpi_rank, mpi_size));
        std::sort(
            runs.begin(), runs.end(),
            [](const IntegratorResult& lhs, const IntegratorResult& rhs) {
                return lhs.elapsed_ms < rhs.elapsed_ms;
            });
        const double solve_min_ms = runs.front().elapsed_ms;
        const double solve_max_ms = runs.back().elapsed_ms;
        const IntegratorResult result =
            runs[static_cast<std::size_t>(args.repeats / 2)];

        const Eigen::VectorXd solution = result.state.head(sys.N);
        const double price = extract_price(solution, sys.grid, model.spot);
        // The four-decimal values published by Dang, Christara and Jackson. A
        // historical comparison, not a reference: they carry no stated
        // uncertainty. The accepted references, with theirs, are written by
        // ./financial-reference into data/financial-validation.
        const double historical_comparison_price =
            args.rainbow ? 4.4450 : 13.2449;
        const double literature_error =
            model.expiry == 1.0
                ? std::abs(price - historical_comparison_price)
                : std::numeric_limits<double>::quiet_NaN();
        const double tail_error = args.rainbow
            ? result.state.tail(3).norm()
            : (result.state.tail(3) - make_s_vec(model.expiry)).norm();

        const bool single_requested = !args.single_gpu_state.empty();
        const bool referee_requested = !args.referee_dir.empty();
        double single_error = std::numeric_limits<double>::quiet_NaN();
        double single_scale = std::numeric_limits<double>::quiet_NaN();
        double single_relative_error =
            std::numeric_limits<double>::quiet_NaN();
        double single_error_limit =
            std::numeric_limits<double>::quiet_NaN();
        double ode_error = std::numeric_limits<double>::quiet_NaN();
        double referee_price_error =
            std::numeric_limits<double>::quiet_NaN();
        int postprocess_ok = 1;
        std::string postprocess_error;
        if (mpi_rank == 0) {
            try {
                if (!args.save_state.empty())
                    save_vector_file(args.save_state, result.state);
                if (single_requested) {
                    const Eigen::VectorXd single =
                        load_vector_file(args.single_gpu_state);
                    if (single.size() != result.state.size())
                        throw std::runtime_error(
                            "single-GPU state size does not match distributed state");
                    single_error =
                        (result.state - single).lpNorm<Eigen::Infinity>();
                    single_scale = std::max(
                        1.0, single.lpNorm<Eigen::Infinity>());
                    single_relative_error = single_error / single_scale;
                    single_error_limit = args.single_state_atol
                        + args.single_state_rtol * single_scale;
                }
                if (referee_requested) {
                    const Eigen::VectorXd referee = load_vector_file(
                        referee_path(args.referee_dir, args.n, args.rainbow));
                    if (referee.size() != sys.N)
                        throw std::runtime_error(
                            "referee size does not match the PDE system");
                    ode_error =
                        (extract_cube(solution, sys.grid, model.spot)
                         - extract_cube(referee, sys.grid, model.spot)).norm();
                    referee_price_error = std::abs(
                        price - extract_price(referee, sys.grid, model.spot));
                }
            } catch (const std::exception& error) {
                postprocess_ok = 0;
                postprocess_error = error.what();
            }
        }
#ifdef CAKSM_HAVE_MPI
        if (mpi_size > 1)
            MPI_Bcast(
                &postprocess_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
        if (postprocess_ok == 0)
            throw std::runtime_error(
                mpi_rank == 0
                    ? "rank-0 postprocessing failed: " + postprocess_error
                    : "rank-0 postprocessing failed; see rank 0 for details");

        const int64_t halo_values =
            2LL * block_depth * n2 * (world_gpus - 1);
        const double cycle_ms =
            result.elapsed_ms / static_cast<double>(args.steps);
        const double reductions_per_step =
            static_cast<double>(result.reductions.operations)
            / static_cast<double>(args.steps);
        const double payload_per_step =
            static_cast<double>(result.reductions.payload_bytes)
            / static_cast<double>(args.steps);
        const double modeled_collective_ms_per_step =
            static_cast<double>(result.reductions.operations)
            / static_cast<double>(args.steps)
            * reduction_cost_s(machine, collective_tier) * 1.0e3;
        const double modeled_halo_floor_ms_per_step =
            machine.interconnect_bw_gbs > 0.0
                ? static_cast<double>(result.reductions.halo_values)
                    * sizeof(double) / static_cast<double>(args.steps)
                    / (machine.interconnect_bw_gbs * 1.0e9) * 1.0e3
                : 0.0;
        const double host_syncs_per_step =
            static_cast<double>(result.reductions.host_syncs)
            / static_cast<double>(args.steps);
        // The probe times an empty round-trip on an idle stream, so it is a
        // lower bound and not the cost. A real read waits for whatever the
        // stream still holds: an Nsight capture of the production point put the
        // measured cost about 24 times above this bound. Both are printed, and
        // the measured one is the max over ranks, because the ranks wait on
        // each other and a rank-local figure describes one process only.
        const double host_sync_lower_bound_ms_per_step =
            host_syncs_per_step * host_sync_probe_us * 1.0e-3;
        const double host_sync_measured_ms_per_step =
            result.reductions.host_sync_ms / static_cast<double>(args.steps);
        const double build_halos_per_step =
            static_cast<double>(result.reductions.build_halo_exchanges)
            / static_cast<double>(args.steps);
        const double build_kib_per_step =
            static_cast<double>(result.reductions.build_halo_values)
            * sizeof(double) / static_cast<double>(args.steps) / 1024.0;
        const double build_depth_avg =
            result.reductions.build_halo_exchanges > 0
                ? static_cast<double>(result.reductions.build_operator_steps)
                    / static_cast<double>(
                        result.reductions.build_halo_exchanges)
                : 0.0;
        const int build_depth_min =
            result.reductions.build_halo_exchanges > 0
                ? result.reductions.build_depth_min : 0;
        const int build_depth_max =
            result.reductions.build_halo_exchanges > 0
                ? result.reductions.build_depth_max : 0;
        const double transition_halos_per_step =
            static_cast<double>(result.reductions.transition_halo_exchanges)
            / static_cast<double>(args.steps);
        const double transition_kib_per_step =
            static_cast<double>(result.reductions.transition_halo_values)
            * sizeof(double) / static_cast<double>(args.steps) / 1024.0;

        int passed_int = 1;
        if (mpi_rank == 0) {
        std::printf("Distributed CA exponential integrator\n");
        std::printf(
            "  option=%s | basis=monomial | orth=cholqr2 | n=%d | N=%d | steps=%d | tol=%.3e | m_max=%d | s=%d | expiry=%.6g\n",
            args.rainbow ? "rainbow" : "basket",
            args.n, sys.N, args.steps, args.tol, args.m, args.s,
            args.expiry);
        std::printf(
            "  arm=%s | certificate=%s | agreement check=%s | "
            "halo overlap=%s | halo memset=%s | shared carveout=%s\n",
            args.arm.c_str(), args.certificate.c_str(),
            agreement_mode_name(options.agreement),
            options.halo_overlap ? "on" : "off",
            options.shell_halo_initialization ? "shells" : "full",
            options.shared_carveout == MpkSharedCarveout::Maximum
                ? "max" : "default");
        std::printf(
            "  decomposition=%d slabs across %d node(s) | local GPUs/process=%d | reductions=NCCL/%s | halo=%s | recordable=%s\n",
            world_gpus, distinct_hosts,
            local_gpus,
            tier_name(collective_tier),
            peer_halo ? "peer-copy" : "NCCL send/recv",
            recordable ? "yes" : "no");
        std::printf(
            "  halo: %lld values %.3f MiB/matrix-powers block\n",
            static_cast<long long>(halo_values),
            static_cast<double>(halo_values) * sizeof(double)
                / (1024.0 * 1024.0));
        std::printf(
            "  Krylov m: min=%d avg=%.2f max=%d | unconverged=%d\n",
            result.min_m, result.avg_m, result.max_m, result.unconverged);
        std::printf(
            "  max residual: %.6e | max block kappa: %.6e\n",
            result.max_residual, result.max_kappa);
        std::printf(
            "  certified width: requested=%d min=%d | fallback blocks=%d | first fallback step=%d\n",
            args.s, result.min_effective_s, result.fallback_blocks,
            result.first_fallback_step);
        std::printf(
            "  CA blocks: %.2f/step | certificate retries=%lld\n",
            static_cast<double>(result.accepted_blocks)
                / static_cast<double>(args.steps),
            static_cast<long long>(result.certificate_retries));
        std::printf(
            "  collectives: %.2f/step | norm=%lld projection=%lld gram=%lld agreement=%lld | payload=%.1f bytes/step | tier=%s\n",
            reductions_per_step,
            static_cast<long long>(result.reductions.norm_operations),
            static_cast<long long>(result.reductions.projection_operations),
            static_cast<long long>(result.reductions.gram_operations),
            static_cast<long long>(result.reductions.agreement_operations),
            payload_per_step, tier_name(collective_tier));
        if (recordable)
            std::printf(
                "  host synchronizations: %.2f/step | measured=%.3f ms/step (max over ranks) | empty round-trip=%.3f us | lower bound=%.3f ms/step\n",
                host_syncs_per_step, host_sync_measured_ms_per_step,
                host_sync_probe_us, host_sync_lower_bound_ms_per_step);
        else
            std::printf(
                "  host synchronizations: %.2f/step | cost withheld, a participating device is contended\n",
                host_syncs_per_step);
        std::printf(
            "  operator communication: %.2f halos/step | %.1f KiB/step | recurrence steps=%lld\n",
            static_cast<double>(result.reductions.halo_exchanges)
                / static_cast<double>(args.steps),
            static_cast<double>(result.reductions.halo_values) * sizeof(double)
                / static_cast<double>(args.steps) / 1024.0,
            static_cast<long long>(result.reductions.operator_steps));
        std::printf(
            "  operator communication detail: build_halos_per_step=%.2f build_kib_per_step=%.1f build_depth_avg=%.3f build_depth_min=%d build_depth_max=%d transition_halos_per_step=%.2f transition_kib_per_step=%.1f transition_depth=1\n",
            build_halos_per_step, build_kib_per_step, build_depth_avg,
            build_depth_min, build_depth_max, transition_halos_per_step,
            transition_kib_per_step);
        std::printf(
            "  operator communication depth histogram: units=total_exchanges build_depth_hist=");
        for (int depth = 0; depth <= args.s; ++depth) {
            if (depth > 0) std::printf(",");
            std::printf(
                "%d:%lld", depth,
                static_cast<long long>(
                    result.reductions.build_halo_depth_hist[
                        static_cast<std::size_t>(depth)]));
        }
        std::printf(
            " transition_depth_hist=1:%lld\n",
            static_cast<long long>(
                result.reductions.transition_halo_exchanges));
        std::printf(
            "  calibrated floors: collectives=%.3f ms/step halo-bandwidth=%.3f ms/step\n",
            modeled_collective_ms_per_step, modeled_halo_floor_ms_per_step);
        if (std::isfinite(literature_error))
            std::printf(
                "  price: %.8f | historical comparison error: %.6e\n",
                price, literature_error);
        else
            std::printf(
                "  price: %.8f | historical comparison error: not defined for expiry=%.6g\n",
                price, model.expiry);
        std::printf("  boundary-state error: %.6e\n", tail_error);
        if (single_requested)
            std::printf(
                "  single-GPU state error: absolute=%.6e relative=%.6e "
                "| scale=%.6e limit=%.6e\n",
                single_error, single_relative_error,
                single_scale, single_error_limit);
        else
            std::printf("  single-GPU state error: not requested\n");
        if (referee_requested)
            std::printf(
                "  ODE referee: cube=%.6e price=%.6e\n",
                ode_error, referee_price_error);
        else
            std::printf("  ODE referee error: not requested\n");
        const bool state_passed = result.state.allFinite();
        const bool boundary_passed =
            std::isfinite(tail_error) && tail_error <= args.tol;
        const bool single_passed =
            !single_requested
            || (std::isfinite(single_error)
                && std::isfinite(single_error_limit)
                && single_error <= single_error_limit);
        const bool referee_passed =
            !referee_requested
            || (std::isfinite(ode_error) && ode_error <= args.tol);
        const bool convergence_passed = result.unconverged == 0;
        const bool passed =
            state_passed && boundary_passed && single_passed
            && referee_passed && convergence_passed;
        passed_int = passed ? 1 : 0;
        std::printf(
            "  validation: %s | state=%s convergence=%s boundary=%s single=%s referee=%s\n",
            passed ? "PASS" : "FAIL",
            state_passed ? "PASS" : "FAIL",
            convergence_passed ? "PASS" : "FAIL",
            boundary_passed ? "PASS" : "FAIL",
            single_passed ? "PASS" : "FAIL",
            referee_passed ? "PASS" : "FAIL");
        // A co-tenant moves every timing at once and plausibly, so a contended run
        // is a correctness run and nothing else. This matches what
        // calibrate-gpu-p2p --production does, except that the solve is allowed to
        // finish and report its numerical gates.
        if (recordable)
            std::printf(
                "  solve median: %.3f ms | cycle: %.3f ms/step | distribution: [%.3f, %.3f] ms (%d runs)\n",
                result.elapsed_ms, cycle_ms,
                solve_min_ms, solve_max_ms, args.repeats);
        else
            std::printf(
                "  solve timing: withheld, a participating device is contended | correctness run only (%d runs)\n",
                args.repeats);
        }

#ifdef CAKSM_HAVE_MPI
        if (mpi_size > 1)
            MPI_Bcast(&passed_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
        for (Slab& slab : slabs) free_slab(slab);
        exit_code = passed_int != 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        if (mpi_rank == 0)
            std::fprintf(stderr, "Error: %s\n", error.what());
        exit_code = EXIT_FAILURE;
    }
#ifdef CAKSM_HAVE_MPI
    MPI_Finalize();
#endif
    return exit_code;
}
