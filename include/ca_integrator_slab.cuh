/**
 * @file ca_integrator_slab.cuh
 * @brief One slab of the distributed CA integrator, its communication
 *        primitives, and the cross-rank agreement instrument.
 *
 * A slab is the contiguous range of z-planes of the grid that one GPU owns.
 * Everything here follows from that ownership: the slab and the buffers resident
 * on it, the account of what it communicated, the blocking-read and all-reduce
 * primitives the rest of the solver is built from, and the agreement checkpoints
 * that assert every rank is still making the same adaptive decisions.
 *
 * The solver itself is not here. It lives in src/ca_integrator_2gpu.cu, where
 * append_block() builds and orthogonalizes one block, distributed_cholqr2()
 * dispatches to the cholqr2_immediate() and cholqr2_deferred() arms,
 * advance_step() takes one time step, and solve() runs the integration.
 *
 * @see src/ca_integrator_2gpu.cu for every caller of the primitives below.
 *
 * @author Kevin Knights
 * @date 2026-07-27
 */
#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <nccl.h>
#include <nvToolsExt.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_ca_arnoldi.cuh"      // GpuCaCandidateResult
#include "gpu_pde_matrix_powers.cuh"  // kMpkMaxS
#include "regime.hpp"              // cholqr_kappa_limit

#define CUDA_CHECK(call) do {                                                     \
    const cudaError_t e_ = (call);                                                \
    if (e_ != cudaSuccess) {                                                      \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                     cudaGetErrorString(e_));                                     \
        std::exit(EXIT_FAILURE);                                                  \
    }                                                                             \
} while (0)

#define CUBLAS_CHECK(call) do {                                                   \
    const cublasStatus_t e_ = (call);                                             \
    if (e_ != CUBLAS_STATUS_SUCCESS) {                                            \
        std::fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__,   \
                     static_cast<int>(e_));                                       \
        std::exit(EXIT_FAILURE);                                                  \
    }                                                                             \
} while (0)

#define CUSOLVER_CHECK(call) do {                                                 \
    const cusolverStatus_t e_ = (call);                                           \
    if (e_ != CUSOLVER_STATUS_SUCCESS) {                                          \
        std::fprintf(stderr, "cuSOLVER error at %s:%d: %d\n", __FILE__, __LINE__, \
                     static_cast<int>(e_));                                       \
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
using Milliseconds = std::chrono::duration<double, std::milli>;

/**
 * @brief Scoped NVTX range around one phase of the cycle.
 *
 * Every phase is wrapped in one, which is what lets an Nsight trace attribute
 * the cycle to phases rather than to kernels.
 */
class NvtxRange {
public:
    explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }
    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

/**
 * @brief The communication account for one solve, counted rather than modeled.
 *
 * Collectives are split by kind because they carry different payloads, and halos
 * are split into the block build and the transition apply because the two
 * exchange different depths. The depth histogram lets a figure price a halo at
 * the depth it actually moved rather than infer one from aggregate bytes.
 */
struct ReductionStats {
    int64_t operations = 0;
    int64_t payload_bytes = 0;
    int64_t norm_operations = 0;
    int64_t projection_operations = 0;
    int64_t gram_operations = 0;
    int64_t agreement_operations = 0;
    int64_t halo_exchanges = 0;
    int64_t halo_values = 0;
    int64_t operator_steps = 0;
    int64_t build_halo_exchanges = 0;
    int64_t build_halo_values = 0;
    int64_t build_operator_steps = 0;
    int build_depth_min = kMpkMaxS + 1;
    int build_depth_max = 0;
    std::array<int64_t, kMpkMaxS + 1> build_halo_depth_hist{};
    int64_t transition_halo_exchanges = 0;
    int64_t transition_halo_values = 0;
    /// Blocking device-to-host reads inside the timed region, with the wall time
    /// the host spent stalled in them. The stall includes whatever work was still
    /// queued on the stream, so it is an upper bound on the synchronization cost;
    /// the lower bound is the count against the separately probed empty
    /// round-trip.
    int64_t host_syncs = 0;
    double host_sync_ms = 0.0;
};

/**
 * @brief Width of the agreement tuple: four integers name every branch
 *        checkpoint.
 *
 * The meanings are checkpoint-specific and are reported by
 * agreement_field_name().
 */
inline constexpr int kDecisionFields = 4;
static_assert(
    sizeof(int) == sizeof(std::int32_t),
    "NCCL agreement buffers require a 32-bit int");

/**
 * @brief How hard the cross-rank agreement check works.
 *
 * `Step` is the production form: one seeded min/max per step, latched on the
 * device, no host round-trip. It detects a disagreement but cannot prevent the
 * hang one would cause on a schedule-changing branch, because by the time the
 * step-end check runs the ranks have already enqueued different collectives.
 *
 * `Strict` closes that gap by checking before every such branch and reading each
 * verdict immediately, so a disagreement aborts instead. That costs a host
 * round-trip at roughly four checkpoints per block, about a quarter of the cycle
 * by Nsight, so it is a correctness pass and not a timed configuration: run once
 * under `strict`, then time under `step`.
 */
enum class AgreementMode {
    Off,
    Step,
    Strict
};

/**
 * @brief Mode name for the record.
 */
[[nodiscard]] constexpr const char* agreement_mode_name(
    AgreementMode mode) noexcept
{
    switch (mode) {
        case AgreementMode::Off:    return "off";
        case AgreementMode::Step:   return "step";
        case AgreementMode::Strict: return "strict";
    }
    return "unknown";
}

/**
 * @brief Behavior chosen on the command line and held fixed for the whole solve.
 */
struct SolverOptions {
    bool exact_depth = false;
    bool deferred_certificate = false;
    bool halo_overlap = false;
    bool shell_halo_initialization = false;
    MpkSharedCarveout shared_carveout = MpkSharedCarveout::Default;
    AgreementMode agreement = AgreementMode::Step;
    bool agreement_self_test = false;

    [[nodiscard]] bool agreement_enabled() const
    {
        return agreement != AgreementMode::Off;
    }
    [[nodiscard]] bool agreement_strict() const
    {
        return agreement == AgreementMode::Strict;
    }
};

/**
 * @brief One CholQR2 certificate decision, sized so the host reads it in a
 *        single transfer.
 */
struct CertificateVerdict {
    double kappa = 0.0;
    int accepted = 0;
    int rejected_argument = 0;
};

/**
 * @brief One GPU's share of the problem and everything resident on it.
 *
 * The z range it owns, its stream, communicator and library handles, every
 * device buffer, and the pinned host staging every blocking read goes through.
 * Large state stays here for the whole run and is gathered only after timing.
 */
struct Slab {
    int rank = 0;
    int global_rank = 0;
    int device = 0;
    int z_begin = 0;
    int z_count = 0;
    int sm_count = 0;
    int64_t local_N = 0;
    int64_t ld = 0;
    int64_t reduction_rows = 0;
    int target = 0;
    int block_max = 0;
    int potrf_lwork = 0;
    cudaStream_t stream = nullptr;
    cudaStream_t halo_stream = nullptr;
    cudaEvent_t halo_ready = nullptr;
    cudaEvent_t halo_received = nullptr;
    ncclComm_t comm = nullptr;
    cublasHandle_t blas = nullptr;
    cusolverDnHandle_t solver = nullptr;
    double* state = nullptr;
    double* start = nullptr;
    double* action = nullptr;
    double* halo = nullptr;
    double* B = nullptr;
    double* V = nullptr;
    double* H = nullptr;
    double* C = nullptr;
    double* C2 = nullptr;
    double* G = nullptr;
    double* R1 = nullptr;
    double* local_R = nullptr;
    double* potrf_work = nullptr;
    double* kappa = nullptr;
    double* f = nullptr;
    double* scalar = nullptr;
    GpuCaCandidateResult* candidate = nullptr;
    int* potrf_info = nullptr;
    int* potrf_info2 = nullptr;
    CertificateVerdict* certificate = nullptr;
    int* decision = nullptr;
    int* decision_extreme = nullptr;
    int* verdict = nullptr;
    double* face_b = nullptr;
    /// Pinned staging for every blocking read. A device-to-host copy out of
    /// pageable memory synchronizes the stream and stages through a driver
    /// buffer, so the copy itself absorbs the pipeline drain; Nsight measured
    /// that path at about 90 microseconds average against a 7.7 microsecond
    /// median for the device-to-device copies beside it.
    int* host_verdict = nullptr;
    double* host_scalar = nullptr;
    int* host_info = nullptr;
    CertificateVerdict* host_certificate = nullptr;
    GpuCaCandidateResult* host_candidate = nullptr;
};

/**
 * @brief Fold a CholQR2 attempt into one device-side verdict.
 *
 * Lets the host read once instead of three times. A rejected cuSOLVER argument
 * is carried through so the host can still raise it.
 */
__global__ void ca_certificate_flag(
    const double* kappa, double limit, const int* info, const int* info2,
    CertificateVerdict* verdict)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    verdict->kappa = kappa[0];
    verdict->rejected_argument =
        info[0] < 0 ? info[0] : (info2[0] < 0 ? info2[0] : 0);
    verdict->accepted =
        (kappa[0] < limit && info[0] == 0 && info2[0] == 0) ? 1 : 0;
}

/**
 * @brief Seed both reduction buffers with this rank's replicated decisions.
 *
 * The values arrive as kernel arguments rather than as a host-to-device copy: an
 * asynchronous copy out of pageable host memory synchronizes the stream before
 * it starts, which is exactly the cost this check exists to avoid.
 */
__global__ void ca_decision_seed(
    int field0, int field1, int field2, int field3,
    int* decision, int* extreme)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const int values[kDecisionFields] = {field0, field1, field2, field3};
    for (int q = 0; q < kDecisionFields; ++q) {
        decision[q] = values[q];
        extreme[q] = values[q];
    }
}

/**
 * @brief Seed a certificate checkpoint from this slab's own device result.
 *
 * Seeding from the device is the point. A checkpoint fed by one process-root
 * host value would compare what the host believed rather than what each device
 * decided, and would hide a disagreement between two GPUs owned by one process.
 */
__global__ void ca_certificate_decision_seed(
    int block, int stage, double limit, const double* kappa,
    const int* info, const CertificateVerdict* certificate,
    int invert_acceptance, int* decision, int* extreme)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int accepted = 0;
    int status = 0;
    if (stage == 1) {
        accepted = kappa[0] < limit ? 1 : 0;
    } else if (stage == 4) {
        accepted = certificate->accepted;
        status = certificate->rejected_argument;
    } else {
        status = info[0];
        accepted = status == 0 ? 1 : 0;
    }
    if (invert_acceptance != 0) accepted = accepted == 0 ? 1 : 0;
    const int values[kDecisionFields] = {
        block, stage, accepted, status};
    for (int q = 0; q < kDecisionFields; ++q) {
        decision[q] = values[q];
        extreme[q] = values[q];
    }
}

/**
 * @brief Latch a disagreement between the reduced minimum and maximum tuples.
 *
 * Latching rather than reading is what keeps step mode free of host round-trips:
 * the verdict is read once after the timed region. Strict mode reads it at every
 * checkpoint instead, so no rank can enter a different collective schedule.
 */
__global__ void ca_decision_verdict(
    const int* minimum, const int* maximum, int step, int* verdict)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (verdict[0] != 0) return;
    for (int q = 0; q < kDecisionFields; ++q) {
        if (minimum[q] != maximum[q]) {
            verdict[0] = q + 1;
            verdict[1] = step;
            verdict[2] = minimum[q];
            verdict[3] = maximum[q];
            return;
        }
    }
}

/**
 * @brief Wait for every slab's compute and halo streams.
 *
 * Timing brackets and correctness reads need all devices quiet, not just the one
 * currently selected.
 */
void sync_all(std::vector<Slab>& slabs)
{
    for (Slab& slab : slabs) {
        CUDA_CHECK(cudaSetDevice(slab.device));
        CUDA_CHECK(cudaStreamSynchronize(slab.stream));
        CUDA_CHECK(cudaStreamSynchronize(slab.halo_stream));
    }
}

/**
 * @brief A blocking device-to-host read, counted and timed, through pinned
 *        staging.
 *
 * Every host round-trip in the timed region goes through here, for two reasons:
 * the synchronization account has one place to read from, and the pinned path is
 * forced. A copy out of pageable memory synchronizes the stream and stages
 * through a driver buffer, which is where an Nsight capture found most of the
 * cycle hiding.
 */
void blocking_read(
    Slab& slab, void* host, const void* device, std::size_t bytes,
    ReductionStats& stats)
{
    const auto begin = Clock::now();
    CUDA_CHECK(cudaMemcpyAsync(
        host, device, bytes, cudaMemcpyDeviceToHost, slab.stream));
    CUDA_CHECK(cudaStreamSynchronize(slab.stream));
    ++stats.host_syncs;
    stats.host_sync_ms += Milliseconds(Clock::now() - begin).count();
}

/**
 * @brief Time the empty round-trip: an 8-byte read of a device word on an
 *        otherwise idle stream.
 *
 * A lower bound on one blocking read, and explicitly not the floor: a real read
 * waits for whatever is queued behind it, which an Nsight capture put at roughly
 * twenty-four times this. Reported as a bound, so the account cannot be read as
 * having measured the stall.
 */
[[nodiscard]] double probe_host_sync_us(Slab& slab)
{
    constexpr int warmup = 32;
    constexpr int repeats = 512;
    double value = 0.0;
    CUDA_CHECK(cudaSetDevice(slab.device));
    for (int q = 0; q < warmup; ++q) {
        CUDA_CHECK(cudaMemcpyAsync(
            &value, slab.scalar, sizeof(double),
            cudaMemcpyDeviceToHost, slab.stream));
        CUDA_CHECK(cudaStreamSynchronize(slab.stream));
    }
    const auto begin = Clock::now();
    for (int q = 0; q < repeats; ++q) {
        CUDA_CHECK(cudaMemcpyAsync(
            &value, slab.scalar, sizeof(double),
            cudaMemcpyDeviceToHost, slab.stream));
        CUDA_CHECK(cudaStreamSynchronize(slab.stream));
    }
    return Milliseconds(Clock::now() - begin).count() * 1.0e3
         / static_cast<double>(repeats);
}

/**
 * @brief One NCCL all-reduce across every slab, with the payload charged to the
 *        account.
 *
 * Grouping matters: the local GPUs of one process must post together or the
 * collective serializes.
 */
void allreduce_in_place(
    std::vector<Slab>& slabs, const std::vector<double*>& buffers,
    std::size_t count, ReductionStats& stats, char kind)
{
    NCCL_CHECK(ncclGroupStart());
    for (std::size_t rank = 0; rank < slabs.size(); ++rank) {
        Slab& slab = slabs[rank];
        CUDA_CHECK(cudaSetDevice(slab.device));
        NCCL_CHECK(ncclAllReduce(
            buffers[rank], buffers[rank], count, ncclDouble, ncclSum,
            slab.comm, slab.stream));
    }
    NCCL_CHECK(ncclGroupEnd());
    ++stats.operations;
    stats.payload_bytes +=
        static_cast<int64_t>(count) * static_cast<int64_t>(sizeof(double));
    if (kind == 'n') ++stats.norm_operations;
    if (kind == 'p') ++stats.projection_operations;
    if (kind == 'g') ++stats.gram_operations;
}

/**
 * @brief Which adaptive branch a checkpoint stands in front of.
 *
 * Each names a place where the ranks could diverge into different collective
 * schedules.
 */
enum class AgreementPoint {
    Breakdown,
    BlockCertificate,
    Candidate
};

/**
 * @brief Which branch a checkpoint protects, for the abort message.
 *
 * A disagreement is only actionable if it names the branch that diverged.
 */
[[nodiscard]] constexpr const char* agreement_point_name(
    AgreementPoint point) noexcept
{
    switch (point) {
        case AgreementPoint::Breakdown:        return "breakdown";
        case AgreementPoint::BlockCertificate: return "block certificate";
        case AgreementPoint::Candidate:        return "Arnoldi candidate";
    }
    return "adaptive decision";
}

/**
 * @brief Which of the four tuple fields disagreed.
 *
 * The meanings are checkpoint-specific, so the name depends on the checkpoint as
 * well as the field.
 */
[[nodiscard]] constexpr const char* agreement_field_name(
    AgreementPoint point, int field) noexcept
{
    constexpr const char* breakdown[kDecisionFields] = {
        "breakdown flag", "current block width", "reserved", "reserved"};
    constexpr const char* certificate[kDecisionFields] = {
        "attempted width", "certificate stage", "certificate acceptance",
        "solver status"};
    constexpr const char* candidate[kDecisionFields] = {
        "selected m", "current block width", "convergence", "fallback blocks"};
    const char* const* names = candidate;
    if (point == AgreementPoint::Breakdown) names = breakdown;
    if (point == AgreementPoint::BlockCertificate) names = certificate;
    return field >= 0 && field < kDecisionFields
        ? names[field] : "unknown field";
}

/**
 * @brief Close one agreement checkpoint: reduce the already-seeded tuples across
 *        ranks, then settle the verdict.
 *
 * Two integer collectives per checkpoint. Step mode latches the verdict on the
 * device and reads it once after the solve; strict mode reads it here through
 * pinned host memory and aborts, which is what turns a would-be hang into a
 * named failure before an adaptive branch can change the collective schedule.
 */
void finish_decision_agreement(
    std::vector<Slab>& slabs, AgreementPoint point, int step,
    bool immediate, ReductionStats& stats)
{
    NCCL_CHECK(ncclGroupStart());
    for (Slab& slab : slabs) {
        CUDA_CHECK(cudaSetDevice(slab.device));
        NCCL_CHECK(ncclAllReduce(
            slab.decision, slab.decision, kDecisionFields, ncclInt32,
            ncclMin, slab.comm, slab.stream));
    }
    NCCL_CHECK(ncclGroupEnd());
    NCCL_CHECK(ncclGroupStart());
    for (Slab& slab : slabs) {
        CUDA_CHECK(cudaSetDevice(slab.device));
        NCCL_CHECK(ncclAllReduce(
            slab.decision_extreme, slab.decision_extreme, kDecisionFields,
            ncclInt32, ncclMax, slab.comm, slab.stream));
    }
    NCCL_CHECK(ncclGroupEnd());
    for (Slab& slab : slabs) {
        CUDA_CHECK(cudaSetDevice(slab.device));
        ca_decision_verdict<<<1, 1, 0, slab.stream>>>(
            slab.decision, slab.decision_extreme, step, slab.verdict);
        CUDA_CHECK(cudaGetLastError());
    }
    stats.operations += 2;
    stats.payload_bytes +=
        2LL * kDecisionFields * static_cast<int64_t>(sizeof(int));
    stats.agreement_operations += 2;

    // The verdict stays latched on the device unless the caller must branch on
    // it now. Reading it here is what makes a disagreement abort instead of
    // hang, and it is also what costs a host round-trip, so only the strict
    // mode pays for it; the step mode reads the latch once after the solve.
    if (!immediate) return;

    Slab& root = slabs.front();
    CUDA_CHECK(cudaSetDevice(root.device));
    blocking_read(
        root, root.host_verdict, root.verdict,
        kDecisionFields * sizeof(int), stats);

    const int field = root.host_verdict[0] - 1;
    if (field >= 0) {
        const std::string where = step >= 0
            ? "step " + std::to_string(step + 1)
            : "the warm-up step";
        throw std::runtime_error(
            "ranks disagreed at the "
            + std::string(agreement_point_name(point)) + " checkpoint on "
            + agreement_field_name(point, field) + " in " + where
            + ": minimum " + std::to_string(root.host_verdict[2])
            + ", maximum " + std::to_string(root.host_verdict[3]));
    }
}

/**
 * @brief Assert that a replicated adaptive decision matches across ranks.
 *
 * Breakdown and Arnoldi candidate decisions are computed from a replicated
 * Hessenberg, so they should agree bitwise, and they arrive here as kernel
 * arguments to avoid a pageable host-to-device copy. A divergence would not show
 * up as a wrong answer: the ranks would enqueue different collective schedules
 * and hang.
 */
void check_decision_agreement(
    std::vector<Slab>& slabs, const std::array<int, kDecisionFields>& decisions,
    AgreementPoint point, int step, bool immediate, ReductionStats& stats)
{
    const NvtxRange range("cross-rank decision agreement");
    for (Slab& slab : slabs) {
        CUDA_CHECK(cudaSetDevice(slab.device));
        ca_decision_seed<<<1, 1, 0, slab.stream>>>(
            decisions[0], decisions[1], decisions[2], decisions[3],
            slab.decision, slab.decision_extreme);
        CUDA_CHECK(cudaGetLastError());
    }
    finish_decision_agreement(slabs, point, step, immediate, stats);
}

/**
 * @brief The same assertion for a certificate stage.
 *
 * Seeded from each GPU's own device verdict rather than from a replicated host
 * value.
 */
void check_certificate_agreement(
    std::vector<Slab>& slabs, int block, int stage, int step,
    bool agreement_self_test, ReductionStats& stats)
{
    const NvtxRange range("cross-rank certificate agreement");
    for (Slab& slab : slabs) {
        CUDA_CHECK(cudaSetDevice(slab.device));
        const int* info = stage == 2
            ? slab.potrf_info : (stage == 3 ? slab.potrf_info2 : nullptr);
        // The diagnostic corrupts one device-derived field at the first
        // certificate checkpoint of the warm-up. It therefore exercises the
        // exact min/max and device-latched failure path used in production,
        // without changing any host-side branch decision.
        const int invert_acceptance =
            agreement_self_test && step < 0
                && (stage == 1 || stage == 4)
                && slab.global_rank == 1;
        ca_certificate_decision_seed<<<1, 1, 0, slab.stream>>>(
            block, stage, cholqr_kappa_limit(),
            stage == 1 ? slab.kappa : nullptr,
            info, stage == 4 ? slab.certificate : nullptr,
            invert_acceptance, slab.decision, slab.decision_extreme);
        CUDA_CHECK(cudaGetLastError());
    }
    finish_decision_agreement(
        slabs, AgreementPoint::BlockCertificate, step, true, stats);
}

} // namespace
