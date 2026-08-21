/**
 * @file ca_arnoldi_gpu.cu
 * @brief GPU CA-Arnoldi integrator and numerical validation harness.
 *
 * @author Kevin Knights
 * @date 2026-07-26
 */

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unsupported/Eigen/MatrixFunctions>

#include "ca_arnoldi.hpp"
#include "ca_integrator_memory.hpp"
#include "ca_pricing_gpu.hpp"
#include "ca_referee_scaled.hpp"
#include "gpu_ca_arnoldi.cuh"
#include "gpu_contention.cuh"
#include "gpu_pde_matrix_powers_streamed.cuh"
#include "gram_splitk.cuh"
#include "trsm_tallskinny.cuh"

// The solver indexes the state logically in the payoff initialization, the Gram
// and norm reductions, the final vector assembly and the host copies. A padded
// row stride would have to be threaded through all of them, and the padded arm
// has to clear its profiling gate in the matrix-powers harness before that is
// worth doing, so a pitched build of the solver is refused rather than run.
static_assert(kMpkPitchAlignment == 1,
              "the one-GPU solver indexes the state logically; see the pitched "
              "arm gate in the matrix-powers harness");

// Which swept configuration this binary was built as, supplied by the build
// definition. The default describes an ordinary hand-built binary rather than
// pretending one was not built.
#ifndef CAKSM_MPK_CONFIGURATION
#define CAKSM_MPK_CONFIGURATION "default"
#endif

#define CUDA_CHECK(call) do {                                                     \
    const cudaError_t e_ = (call);                                                \
    if (e_ != cudaSuccess) {                                                      \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                     cudaGetErrorString(e_));                                     \
        std::exit(EXIT_FAILURE);                                                  \
    }                                                                             \
} while (0)

#define CUBLAS_CHECK(call) do {                                                   \
    const cublasStatus_t s_ = (call);                                             \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                            \
        std::fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__,   \
                     static_cast<int>(s_));                                       \
        std::exit(EXIT_FAILURE);                                                  \
    }                                                                             \
} while (0)

#define CUSOLVER_CHECK(call) do {                                                 \
    const cusolverStatus_t s_ = (call);                                           \
    if (s_ != CUSOLVER_STATUS_SUCCESS) {                                          \
        std::fprintf(stderr, "cuSOLVER error at %s:%d: %d\n", __FILE__, __LINE__, \
                     static_cast<int>(s_));                                       \
        std::exit(EXIT_FAILURE);                                                  \
    }                                                                             \
} while (0)

namespace {

/// How a block of basis vectors is made orthonormal. CholQR2 is the production
/// arm; the other two pass the same stability gates and are kept as research
/// arms so the cost rejection can be measured rather than asserted.
enum class Orthogonalization {
    CholQr2,
    Tsqr,
    BlockGramSchmidt
};

/// Which recurrence builds the block. All three span the same Krylov subspace
/// and differ only in how well conditioned the columns spanning it are.
enum class PolynomialBasis {
    Monomial,
    Newton,
    Chebyshev
};

/// Which matrix-powers kernel the solver dispatches to.
///
/// Auto is the measured dispatch rather than a preference: the plane-streamed
/// family was faster than the production tile at every grid from 61 upward and
/// slower at 31, because a small grid cannot give it enough blocks to fill the
/// device. Auto encodes that boundary and nothing else.
enum class MpkFamilySelection {
    FullVolume,
    PlaneStreamed,
    Auto
};

/// Behavior chosen on the command line and held fixed for the whole solve.
struct SolverOptions {
    bool exact_depth = false;
    /// Recurrence steps per kernel launch; 0 dispatches the block in one launch.
    int mpk_chunk = kMpkPreferredS;
    /// The promoted production policy. Small grids retain the accepted
    /// full-volume kernel; grids from the measured boundary upward use the
    /// plane-streamed family when its width fits the device.
    MpkFamilySelection family = MpkFamilySelection::Auto;
    /// Streamed segment length. Negative asks for the measured rule for the
    /// grid; zero streams the whole extent, which was never the fastest choice.
    int stream_height = -1;
    /// The device's opt-in shared-memory limit, queried once. Needed because
    /// admissibility is a property of the device, and a width the streamed
    /// family cannot hold has to fall back rather than abort the run.
    std::size_t shared_optin_bytes = 0;
};

/// Whether this block runs streamed under the selected policy.
///
/// Admissibility is checked here rather than left to the launch. The streamed
/// footprint grows faster in the block width than the full-volume tile's does,
/// so a wide Chebyshev block can exceed what the device will opt into; the
/// accepted kernel always fits, and falling back to it is what keeps a width the
/// streamed family cannot hold from ending the run.
[[nodiscard]] bool use_streamed(
    const SolverOptions& options, int n, int steps, bool chebyshev)
{
    bool selected = false;
    switch (options.family) {
        case MpkFamilySelection::FullVolume: selected = false; break;
        case MpkFamilySelection::PlaneStreamed: selected = true; break;
        case MpkFamilySelection::Auto:
            selected = n >= kMpkStreamMinimumGrid;
            break;
    }
    if (!selected || steps < 1 || steps > kMpkMaxS) return false;
    const std::size_t needed = mpk_stream_shared_bytes(steps, chebyshev);
    return needed > 0
        && (options.shared_optin_bytes == 0
            || needed <= options.shared_optin_bytes);
}

/// The segment length a streamed launch uses at this grid.
[[nodiscard]] int stream_height_for(const SolverOptions& options, int n)
{
    return options.stream_height >= 0
        ? options.stream_height
        : mpk_stream_auto_height(n);
}

[[nodiscard]] const char* family_selection_name(MpkFamilySelection family)
{
    switch (family) {
        case MpkFamilySelection::FullVolume: return "full-volume";
        case MpkFamilySelection::PlaneStreamed: return "plane-streamed";
        case MpkFamilySelection::Auto: return "auto";
    }
    return "unknown";
}

/// Command line, already validated by parse_args.
struct Args {
    int n = 15;
    int m = 8;
    int s = 4;
    int steps = 0;
    int repeats = 1;
    int device = 0;
    double scale = 0.01;
    double expiry = 1.0;
    double tol = 1.0e-8;
    bool rainbow = false;
    PolynomialBasis basis = PolynomialBasis::Monomial;
    Orthogonalization orthogonalization = Orthogonalization::CholQr2;
    std::string referee_dir;
    std::string save_state;
    bool memory_report = false;
    double memory_reserve = 0.10;
    std::string arm = "as-measured";
    int mpk_chunk = -1;
    MpkFamilySelection family = MpkFamilySelection::Auto;
    int stream_height = -1;
};

/// Lower-case name for the record. Matches the string the figures key on.
[[nodiscard]] const char* basis_name(PolynomialBasis basis)
{
    switch (basis) {
        case PolynomialBasis::Monomial: return "monomial";
        case PolynomialBasis::Newton: return "newton";
        case PolynomialBasis::Chebyshev: return "chebyshev";
    }
    return "unknown";
}

/// Lower-case name for the record.
[[nodiscard]] const char* orthogonalization_name(Orthogonalization method)
{
    switch (method) {
        case Orthogonalization::CholQr2: return "cholqr2";
        case Orthogonalization::Tsqr: return "tsqr";
        case Orthogonalization::BlockGramSchmidt: return "bgs2";
    }
    return "unknown";
}

/// Global reductions one block costs, which is the quantity s-step exists to
/// reduce. Counted rather than derived, so a research arm with a different
/// reduction pattern is charged for what it actually issues.
[[nodiscard]] int block_collectives(
    Orthogonalization method, int block, bool has_prior_basis)
{
    int local_factorization = 0;
    switch (method) {
        case Orthogonalization::CholQr2:
            local_factorization = 2;
            break;
        case Orthogonalization::Tsqr:
            local_factorization = 1;
            break;
        case Orthogonalization::BlockGramSchmidt:
            // The input Gram used to report kappa, then two block projections
            // for every column after the first and one norm per column.
            local_factorization = 3 * block - 1;
            break;
    }
    return local_factorization + (has_prior_basis ? 2 : 0);
}

/// Doubles those reductions carry. Latency follows the count and bandwidth
/// follows this, and the two move in opposite directions with block width, so
/// the account keeps them apart.
[[nodiscard]] int64_t block_reduction_values(
    Orthogonalization method, int filled, int block)
{
    const int64_t projection =
        filled > 0
            ? 2LL * static_cast<int64_t>(filled) * block
            : 0;
    // CholQR2 forms two Grams. TSQR forms one diagnostic input Gram before its
    // local Householder factorization. BGS2 forms that same diagnostic Gram in
    // addition to b^2 values from its two-pass projections and column norms.
    const int gram_count =
        method == Orthogonalization::Tsqr ? 1 : 2;
    return projection
         + static_cast<int64_t>(gram_count) * block * block;
}

/// Parse and validate the command line, or throw.
[[nodiscard]] Args parse_args(int argc, char** argv)
{
    Args a;
    for (int q = 1; q < argc; ++q) {
        const std::string arg = argv[q];
        auto next = [&]() -> const char* {
            if (++q >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", arg.c_str());
                std::exit(EXIT_FAILURE);
            }
            return argv[q];
        };
        if      (arg == "--n")      a.n = std::stoi(next());
        else if (arg == "--m")      a.m = std::stoi(next());
        else if (arg == "--s")      a.s = std::stoi(next());
        else if (arg == "--steps")  a.steps = std::stoi(next());
        else if (arg == "--repeats") a.repeats = std::stoi(next());
        else if (arg == "--device") a.device = std::stoi(next());
        else if (arg == "--scale")  a.scale = std::stod(next());
        else if (arg == "--expiry") a.expiry = std::stod(next());
        else if (arg == "--tol")    a.tol = std::stod(next());
        else if (arg == "--referee-dir") a.referee_dir = next();
        else if (arg == "--save-state") a.save_state = next();
        else if (arg == "--memory-report") a.memory_report = true;
        else if (arg == "--memory-reserve")
            a.memory_reserve = std::stod(next());
        else if (arg == "--arm") a.arm = next();
        else if (arg == "--mpk-chunk") a.mpk_chunk = std::stoi(next());
        else if (arg == "--stream-height") a.stream_height = std::stoi(next());
        else if (arg == "--kernel-family") {
            const std::string family = next();
            if (family == "full-volume")
                a.family = MpkFamilySelection::FullVolume;
            else if (family == "plane-streamed")
                a.family = MpkFamilySelection::PlaneStreamed;
            else if (family == "auto")
                a.family = MpkFamilySelection::Auto;
            else {
                std::fprintf(
                    stderr,
                    "--kernel-family must be full-volume, plane-streamed or auto\n");
                std::exit(EXIT_FAILURE);
            }
        }
        else if (arg == "--basis") {
            const std::string basis = next();
            if      (basis == "monomial")  a.basis = PolynomialBasis::Monomial;
            else if (basis == "newton")    a.basis = PolynomialBasis::Newton;
            else if (basis == "chebyshev") a.basis = PolynomialBasis::Chebyshev;
            else {
                std::fprintf(
                    stderr,
                    "--basis must be monomial, newton, or chebyshev\n");
                std::exit(EXIT_FAILURE);
            }
        }
        else if (arg == "--orth") {
            const std::string orth = next();
            if      (orth == "cholqr2")
                a.orthogonalization = Orthogonalization::CholQr2;
            else if (orth == "tsqr")
                a.orthogonalization = Orthogonalization::Tsqr;
            else if (orth == "bgs2")
                a.orthogonalization = Orthogonalization::BlockGramSchmidt;
            else {
                std::fprintf(stderr, "--orth must be cholqr2, tsqr, or bgs2\n");
                std::exit(EXIT_FAILURE);
            }
        }
        else if (arg == "--option") {
            const std::string option = next();
            if      (option == "basket")  a.rainbow = false;
            else if (option == "rainbow") a.rainbow = true;
            else {
                std::fprintf(stderr, "--option must be basket or rainbow\n");
                std::exit(EXIT_FAILURE);
            }
        } else if (arg == "--help") {
            std::printf(
                "Usage: ./ca-integrator [--n N] [--m M] [--s S] [--device D]\n"
                "                       [--option basket|rainbow] [--scale H] [--expiry T]\n"
                "                       [--basis monomial|newton|chebyshev]\n"
                "                       [--orth cholqr2|tsqr|bgs2]\n"
                "       ./ca-integrator --steps K [--tol T] [--n N] [--m M]\n"
                "                       [--s S] [--repeats K]\n"
                "                       [--option basket|rainbow] [--expiry T]\n"
                "                       [--basis monomial|newton|chebyshev]\n"
                "                       [--orth cholqr2|tsqr|bgs2]\n"
                "                       [--referee-dir DIR] [--save-state FILE]\n"
                "                       [--memory-report] [--memory-reserve F]\n"
                "                       [--arm as-measured|exact-depth|scaled-augmentation]\n"
                "                       [--mpk-chunk S] [--device D]\n"
                "                       [--kernel-family full-volume|plane-streamed|auto] (default: auto)\n"
                "                       [--stream-height H]\n");
            std::exit(EXIT_SUCCESS);
        } else {
            std::fprintf(stderr, "Unknown flag: %s\n", arg.c_str());
            std::exit(EXIT_FAILURE);
        }
    }
    if (a.n < 3) {
        std::fprintf(stderr, "--n must be >= 3\n");
        std::exit(EXIT_FAILURE);
    }
    if (a.m < 1 || a.m > kGpuCaMaxM) {
        std::fprintf(stderr, "--m must be in [1, %d]\n", kGpuCaMaxM);
        std::exit(EXIT_FAILURE);
    }
    if (a.s < 1 || a.s > kMpkMaxS) {
        std::fprintf(stderr, "--s must be in [1, %d]\n", kMpkMaxS);
        std::exit(EXIT_FAILURE);
    }
    if (a.steps < 0) {
        std::fprintf(stderr, "--steps must be >= 0\n");
        std::exit(EXIT_FAILURE);
    }
    if (a.repeats < 1) {
        std::fprintf(stderr, "--repeats must be positive\n");
        std::exit(EXIT_FAILURE);
    }
    if (!(a.tol > 0.0) || !std::isfinite(a.tol)) {
        std::fprintf(stderr, "--tol must be finite and positive\n");
        std::exit(EXIT_FAILURE);
    }
    if (!(a.expiry > 0.0) || !std::isfinite(a.expiry)) {
        std::fprintf(stderr, "--expiry must be finite and positive\n");
        std::exit(EXIT_FAILURE);
    }
    if (!(a.memory_reserve >= 0.0 && a.memory_reserve < 1.0)) {
        std::fprintf(stderr, "--memory-reserve must be in [0,1)\n");
        std::exit(EXIT_FAILURE);
    }
    if (a.arm != "as-measured" && a.arm != "exact-depth"
        && a.arm != "scaled-augmentation") {
        std::fprintf(
            stderr,
            "--arm must be as-measured, exact-depth, or scaled-augmentation\n");
        std::exit(EXIT_FAILURE);
    }
    if (a.arm == "scaled-augmentation"
        && (a.steps == 0 || a.rainbow
            || a.basis != PolynomialBasis::Monomial
            || a.orthogonalization != Orthogonalization::CholQr2)) {
        std::fprintf(
            stderr,
            "--arm scaled-augmentation is a Basket monomial+CholQR2 "
            "integrator postmortem only\n");
        std::exit(EXIT_FAILURE);
    }
    if (a.mpk_chunk < -1 || a.mpk_chunk > kMpkMaxS) {
        std::fprintf(
            stderr, "--mpk-chunk must be 0 (one launch) or at most %d\n",
            kMpkMaxS);
        std::exit(EXIT_FAILURE);
    }
    if (a.basis == PolynomialBasis::Chebyshev
        && a.mpk_chunk >= 0 && a.mpk_chunk != kMpkPreferredS) {
        std::fprintf(
            stderr,
            "Chebyshev currently requires --mpk-chunk %d; its recurrence "
            "uses that fixed launch cap\n",
            kMpkPreferredS);
        std::exit(EXIT_FAILURE);
    }
    if (a.memory_report
        && (a.orthogonalization != Orthogonalization::CholQr2
            || a.basis != PolynomialBasis::Monomial)) {
        std::fprintf(
            stderr,
            "--memory-report currently describes the production monomial+CholQR2 arm\n");
        std::exit(EXIT_FAILURE);
    }
    return a;
}

/// Every device allocation the solve owns, plus the host-synchronization
/// account. Allocated once up front so nothing is allocated inside a timed
/// step, and sized for the widest block the run may request.
struct DeviceWorkspace {
    int64_t ld = 0;
    int target = 0;
    int block_max = 0;
    int sm_count = 0;
    double* start = nullptr;
    double* scratch = nullptr;
    double* B = nullptr;
    double* V = nullptr;
    double* H = nullptr;
    double* C = nullptr;
    double* C2 = nullptr;
    double* G = nullptr;
    double* R1 = nullptr;
    double* local_R = nullptr;
    double* potrf_work = nullptr;
    double* qr_tau = nullptr;
    double* qr_work = nullptr;
    double* newton_shifts = nullptr;
    double* kappa = nullptr;
    double* f = nullptr;
    double* action = nullptr;
    GpuCaCandidateResult* candidate = nullptr;
    int* potrf_info = nullptr;
    int* exp_squarings = nullptr;
    int* exp_terms = nullptr;
    int potrf_lwork = 0;
    int qr_lwork = 0;
    // Blocking host reads inside the timed region, and the wall time spent stalled
    // in them. The stall includes queued device work, so it bounds the cost from
    // above; the floor is the count against a separately probed empty round-trip.
    int64_t host_syncs = 0;
    double host_sync_ms = 0.0;
};

/// Blocking host/device transfers inside the timed region are counted and timed
/// through these helpers so research orthogonalization arms remain in the same
/// synchronization account as the production arm.
void blocking_read(
    DeviceWorkspace& w, void* host, const void* device, std::size_t bytes)
{
    const auto begin = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost));
    ++w.host_syncs;
    w.host_sync_ms +=
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
}

void blocking_write(
    DeviceWorkspace& w, void* device, const void* host, std::size_t bytes)
{
    const auto begin = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice));
    ++w.host_syncs;
    w.host_sync_ms +=
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
}

/// The empty round-trip on an idle stream: the floor a host synchronization cannot
/// go below, and the denominator that turns counted synchronizations into a term.
/// Time an empty device-to-host round-trip on an idle stream.
///
/// A lower bound on what one blocking read costs, and explicitly not the floor:
/// a real read waits for whatever is queued behind it, which an Nsight capture
/// put more than an order of magnitude above this. Reported as a bound so the
/// account cannot be read as having measured the stall.
[[nodiscard]] double probe_host_sync_us(DeviceWorkspace& w)
{
    constexpr int warmup = 32;
    constexpr int repeats = 512;
    double value = 0.0;
    for (int q = 0; q < warmup; ++q)
        CUDA_CHECK(cudaMemcpy(
            &value, w.kappa, sizeof(double), cudaMemcpyDeviceToHost));
    const auto begin = std::chrono::steady_clock::now();
    for (int q = 0; q < repeats; ++q)
        CUDA_CHECK(cudaMemcpy(
            &value, w.kappa, sizeof(double), cudaMemcpyDeviceToHost));
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - begin).count()
         * 1.0e3 / static_cast<double>(repeats);
}

/// Allocate the whole workspace and query the factorization scratch sizes.
///
/// The basis buffer keeps s+1 columns even though exact-depth consumes only s.
/// Both arms are selected at run time inside one binary and share the frozen
/// allocation model, so shrinking it would silently move the recorded
/// largest-common-grid prediction.
void allocate_workspace(
    DeviceWorkspace& w, int64_t ld, int m, int s,
    Orthogonalization orthogonalization, cusolverDnHandle_t solver)
{
    w.ld = ld;
    w.target = m + 1;
    w.block_max = s;

    cudaDeviceProp prop{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    w.sm_count = prop.multiProcessorCount;

    CUDA_CHECK(cudaMalloc(&w.start, static_cast<std::size_t>(ld) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&w.scratch, static_cast<std::size_t>(ld) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(
        &w.B, static_cast<std::size_t>(ld) * static_cast<std::size_t>(s + 1)
            * sizeof(double)));
    CUDA_CHECK(cudaMalloc(
        &w.V, static_cast<std::size_t>(ld) * static_cast<std::size_t>(m + 1)
            * sizeof(double)));
    CUDA_CHECK(cudaMalloc(
        &w.H, static_cast<std::size_t>(m + 1) * static_cast<std::size_t>(m)
            * sizeof(double)));
    CUDA_CHECK(cudaMalloc(
        &w.C, static_cast<std::size_t>(m + 1) * static_cast<std::size_t>(s)
            * sizeof(double)));
    CUDA_CHECK(cudaMalloc(
        &w.C2, static_cast<std::size_t>(m + 1) * static_cast<std::size_t>(s)
            * sizeof(double)));
    CUDA_CHECK(cudaMalloc(
        &w.G, static_cast<std::size_t>(s) * static_cast<std::size_t>(s) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(
        &w.R1, static_cast<std::size_t>(s) * static_cast<std::size_t>(s) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(
        &w.local_R, static_cast<std::size_t>(s) * static_cast<std::size_t>(s)
            * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&w.kappa, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&w.f, static_cast<std::size_t>(m) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&w.action, static_cast<std::size_t>(ld) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&w.candidate, sizeof(GpuCaCandidateResult)));
    CUDA_CHECK(cudaMalloc(&w.potrf_info, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&w.exp_squarings, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&w.exp_terms, sizeof(int)));

    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(
        solver, CUBLAS_FILL_MODE_UPPER, s, w.G, s, &w.potrf_lwork));
    CUDA_CHECK(cudaMalloc(
        &w.potrf_work, static_cast<std::size_t>(w.potrf_lwork) * sizeof(double)));
    if (orthogonalization == Orthogonalization::Tsqr) {
        CUDA_CHECK(cudaMalloc(
            &w.qr_tau, static_cast<std::size_t>(s) * sizeof(double)));
        int geqrf_lwork = 0;
        int orgqr_lwork = 0;
        CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(
            solver, static_cast<int>(ld), s, w.B, static_cast<int>(ld),
            &geqrf_lwork));
        CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(
            solver, static_cast<int>(ld), s, s, w.B, static_cast<int>(ld),
            w.qr_tau, &orgqr_lwork));
        w.qr_lwork = std::max(geqrf_lwork, orgqr_lwork);
        CUDA_CHECK(cudaMalloc(
            &w.qr_work,
            static_cast<std::size_t>(std::max(w.qr_lwork, 1))
                * sizeof(double)));
    }
}

/// Release everything allocate_workspace took.
void free_workspace(DeviceWorkspace& w)
{
    CUDA_CHECK(cudaFree(w.start));
    CUDA_CHECK(cudaFree(w.scratch));
    CUDA_CHECK(cudaFree(w.B));
    CUDA_CHECK(cudaFree(w.V));
    CUDA_CHECK(cudaFree(w.H));
    CUDA_CHECK(cudaFree(w.C));
    CUDA_CHECK(cudaFree(w.C2));
    CUDA_CHECK(cudaFree(w.G));
    CUDA_CHECK(cudaFree(w.R1));
    CUDA_CHECK(cudaFree(w.local_R));
    CUDA_CHECK(cudaFree(w.potrf_work));
    if (w.qr_tau != nullptr) CUDA_CHECK(cudaFree(w.qr_tau));
    if (w.qr_work != nullptr) CUDA_CHECK(cudaFree(w.qr_work));
    if (w.newton_shifts != nullptr) CUDA_CHECK(cudaFree(w.newton_shifts));
    CUDA_CHECK(cudaFree(w.kappa));
    CUDA_CHECK(cudaFree(w.f));
    CUDA_CHECK(cudaFree(w.action));
    CUDA_CHECK(cudaFree(w.candidate));
    CUDA_CHECK(cudaFree(w.potrf_info));
    CUDA_CHECK(cudaFree(w.exp_squarings));
    CUDA_CHECK(cudaFree(w.exp_terms));
}

/// Ask cuSOLVER how much scratch a width-s Cholesky needs. Only the library
/// can answer, so the memory model takes it as an argument rather than guessing.
[[nodiscard]] int query_potrf_lwork(int s)
{
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

/// Print the allocation model against this device and stop, allocating nothing.
/// The point is to decide whether a grid fits before a job spends minutes
/// finding out that it does not.
void report_integrator_memory(const Args& args)
{
    const int potrf_lwork = query_potrf_lwork(args.s);
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const std::uint64_t capacity =
        ca_integrator_memory::usable_bytes(
            free_bytes, total_bytes, args.memory_reserve);
    const auto estimator = [&](int n) {
        return ca_integrator_memory::one_gpu(
            n, args.m, args.s, !args.rainbow, potrf_lwork,
            sizeof(GpuCaCandidateResult));
    };
    const auto requested = estimator(args.n);
    const int largest =
        ca_integrator_memory::largest_odd_grid(capacity, estimator);
    const double mib = 1024.0 * 1024.0;

    std::printf("CA integrator device-memory report\n");
    std::printf(
        "  topology=one-gpu | option=%s | m=%d | s=%d | reserve=%.1f%%\n",
        args.rainbow ? "rainbow" : "basket",
        args.m, args.s, 100.0 * args.memory_reserve);
    std::printf(
        "  device total=%.3f MiB free=%.3f MiB usable=%.3f MiB\n",
        static_cast<double>(total_bytes) / mib,
        static_cast<double>(free_bytes) / mib,
        static_cast<double>(capacity) / mib);
    std::printf(
        "  candidate n=%d: vectors=%.3f MiB faces=%.3f MiB small=%.3f MiB total=%.3f MiB | %s\n",
        args.n,
        static_cast<double>(requested.vector_bytes) / mib,
        static_cast<double>(requested.face_bytes) / mib,
        static_cast<double>(requested.small_bytes) / mib,
        static_cast<double>(requested.total_bytes) / mib,
        requested.total_bytes <= capacity ? "FIT" : "NO FIT");
    std::printf(
        "  recommended largest odd n: %d\n", largest);
    std::printf(
        "MEMORY_REPORT topology=one_gpu option=%s s=%d candidate_n=%d candidate_fit=%s recommended_n=%d required_bytes=%llu usable_bytes=%llu total_bytes=%zu free_bytes=%zu reserve=%.6f\n",
        args.rainbow ? "rainbow" : "basket", args.s, args.n,
        requested.total_bytes <= capacity ? "yes" : "no", largest,
        static_cast<unsigned long long>(requested.total_bytes),
        static_cast<unsigned long long>(capacity),
        total_bytes, free_bytes, args.memory_reserve);
}

/// Run one Cholesky and read its status. Returns false on a non-positive
/// pivot, which is the late, crude symptom of a block too ill-conditioned to
/// certify; the usable limit is the kappa certificate checked before this.
[[nodiscard]] bool potrf_certified(
    cusolverDnHandle_t solver, DeviceWorkspace& w, int block)
{
    CUSOLVER_CHECK(cusolverDnDpotrf(
        solver, CUBLAS_FILL_MODE_UPPER, block, w.G, block,
        w.potrf_work, w.potrf_lwork, w.potrf_info));
    int info = 0;
    blocking_read(w, &info, w.potrf_info, sizeof(int));
    if (info < 0)
        throw std::runtime_error(
            "Cholesky factorization rejected argument "
            + std::to_string(-info));
    return info == 0;
}

/// CholeskyQR twice: the production orthogonalization arm.
///
/// One Gram all-reduce, a local Cholesky and a triangular solve, repeated. The
/// first pass squares the condition number, so a single pass loses orthogonality
/// like u*kappa^2; the second recovers O(u) inside the certificate
/// kappa(B) <= u^(-1/2), which is checked first and rejects the block if it
/// fails. The two triangular factors are combined into the block's R.
[[nodiscard]] bool cholqr2(
    cusolverDnHandle_t solver, DeviceWorkspace& w, int block, double& kappa)
{
    CUDA_CHECK(gram_splitk(w.B, w.ld, block, w.G, w.sm_count));
    ca_gram_condition<<<1, 1>>>(w.G, block, w.kappa);
    CUDA_CHECK(cudaGetLastError());
    kappa = 0.0;
    blocking_read(w, &kappa, w.kappa, sizeof(double));
    if (!(kappa < cholqr_kappa_limit())) return false;

    if (!potrf_certified(solver, w, block)) return false;
    CUDA_CHECK(cudaMemcpyAsync(
        w.R1, w.G, static_cast<std::size_t>(block) * static_cast<std::size_t>(block)
            * sizeof(double), cudaMemcpyDeviceToDevice, nullptr));
    CUDA_CHECK(trsm_tallskinny(w.G, w.B, w.ld, block, w.sm_count));

    CUDA_CHECK(gram_splitk(w.B, w.ld, block, w.G, w.sm_count));
    if (!potrf_certified(solver, w, block)) return false;
    CUDA_CHECK(trsm_tallskinny(w.G, w.B, w.ld, block, w.sm_count));

    const dim3 threads(
        static_cast<unsigned>(block), static_cast<unsigned>(block));
    ca_combine_upper<<<1, threads>>>(w.G, w.R1, w.local_R, block);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

/// Read a cuSOLVER info word. A negative value is a programming error and
/// throws; a positive one is a numerical failure the caller handles.
[[nodiscard]] bool solver_succeeded(DeviceWorkspace& w)
{
    int info = 0;
    blocking_read(w, &info, w.potrf_info, sizeof(int));
    if (info < 0)
        throw std::runtime_error(
            "QR factorization rejected argument " + std::to_string(-info));
    return info == 0;
}

/// TSQR research arm: a Householder QR of the whole tall block.
///
/// Unconditionally stable, unlike CholQR2, and rejected on cost rather than on
/// correctness. The sign fix afterward is what makes it comparable: Householder
/// leaves R's diagonal signs arbitrary while Cholesky forces them positive, so
/// without it the two arms would differ by a column sign.
[[nodiscard]] bool householder_qr(
    cusolverDnHandle_t solver, DeviceWorkspace& w, int block, double& kappa)
{
    CUDA_CHECK(gram_splitk(w.B, w.ld, block, w.G, w.sm_count));
    ca_gram_condition<<<1, 1>>>(w.G, block, w.kappa);
    CUDA_CHECK(cudaGetLastError());
    blocking_read(w, &kappa, w.kappa, sizeof(double));

    CUSOLVER_CHECK(cusolverDnDgeqrf(
        solver, static_cast<int>(w.ld), block, w.B, static_cast<int>(w.ld),
        w.qr_tau, w.qr_work, w.qr_lwork, w.potrf_info));
    if (!solver_succeeded(w)) return false;

    const dim3 matrix_threads(16, 16);
    const dim3 matrix_grid(
        static_cast<unsigned>((block + 15) / 16),
        static_cast<unsigned>((block + 15) / 16));
    ca_extract_upper<<<matrix_grid, matrix_threads>>>(
        w.B, static_cast<int>(w.ld), w.local_R,
        static_cast<int>(w.ld), block);
    CUDA_CHECK(cudaGetLastError());

    CUSOLVER_CHECK(cusolverDnDorgqr(
        solver, static_cast<int>(w.ld), block, block,
        w.B, static_cast<int>(w.ld), w.qr_tau,
        w.qr_work, w.qr_lwork, w.potrf_info));
    if (!solver_succeeded(w)) return false;

    ca_qr_signs<<<1, 32>>>(w.local_R, block, w.qr_tau);
    CUDA_CHECK(cudaGetLastError());
    const dim3 q_grid(
        static_cast<unsigned>((w.ld + 255) / 256),
        static_cast<unsigned>(block));
    ca_apply_q_signs<<<q_grid, 256>>>(
        w.B, w.ld, block, w.qr_tau);
    CUDA_CHECK(cudaGetLastError());
    ca_apply_r_signs<<<matrix_grid, matrix_threads>>>(
        w.local_R, block, w.qr_tau);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

/**
 * Two-pass classical block Gram-Schmidt on the columns of B.
 *
 * The projection against the previously accepted Arnoldi basis is performed
 * twice in append_ca_block. This routine supplies the distinct intra-block
 * BGS2 factorization: for each column it projects against the accepted columns
 * twice, accumulates both coefficient vectors in R, and then normalizes. It is
 * intentionally not routed through cuSOLVER's Householder QR; that is the TSQR
 * arm's local factorization.
 */
/// Two-pass block Gram-Schmidt research arm.
///
/// Orthogonalizes column by column within the block, so it pays a reduction per
/// column where CholQR2 pays one per block. That is precisely the cost s-step
/// set out to avoid, and measuring it here is what turns the rejection into a
/// recorded comparison.
[[nodiscard]] bool block_gram_schmidt2(
    cublasHandle_t blas, DeviceWorkspace& w, int block, double& kappa)
{
    CUDA_CHECK(gram_splitk(w.B, w.ld, block, w.G, w.sm_count));
    ca_gram_condition<<<1, 1>>>(w.G, block, w.kappa);
    CUDA_CHECK(cudaGetLastError());
    blocking_read(w, &kappa, w.kappa, sizeof(double));

    CUDA_CHECK(cudaMemsetAsync(
        w.local_R, 0,
        static_cast<std::size_t>(w.block_max)
            * static_cast<std::size_t>(w.block_max) * sizeof(double),
        nullptr));

    const double one = 1.0;
    const double zero = 0.0;
    const double minus_one = -1.0;
    for (int col = 0; col < block; ++col) {
        double* const vector =
            w.B + static_cast<int64_t>(col) * w.ld;
        for (int pass = 0; pass < 2 && col > 0; ++pass) {
            CUBLAS_CHECK(cublasDgemv(
                blas, CUBLAS_OP_T, static_cast<int>(w.ld), col,
                &one, w.B, static_cast<int>(w.ld), vector, 1,
                &zero, w.G, 1));
            CUBLAS_CHECK(cublasDgemv(
                blas, CUBLAS_OP_N, static_cast<int>(w.ld), col,
                &minus_one, w.B, static_cast<int>(w.ld), w.G, 1,
                &one, vector, 1));
            CUBLAS_CHECK(cublasDaxpy(
                blas, col, &one, w.G, 1,
                w.local_R + static_cast<int64_t>(col) * block, 1));
        }

        double norm = 0.0;
        const auto begin = std::chrono::steady_clock::now();
        CUBLAS_CHECK(cublasDnrm2(
            blas, static_cast<int>(w.ld), vector, 1, &norm));
        ++w.host_syncs;
        w.host_sync_ms +=
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - begin).count();
        if (!(norm > std::numeric_limits<double>::min())
            || !std::isfinite(norm))
            return false;

        blocking_write(
            w,
            w.local_R + col + static_cast<int64_t>(col) * block,
            &norm, sizeof(double));
        const double inverse_norm = 1.0 / norm;
        CUBLAS_CHECK(cublasDscal(
            blas, static_cast<int>(w.ld), &inverse_norm, vector, 1));
    }
    return true;
}

/// Assemble the Hessenberg columns a Newton block contributes.
///
/// The Newton recurrence is (A - shift_j I) applied to the previous column, so
/// unlike the monomial case the shift enters H on the diagonal. Kept separate
/// from the shared assembler because folding a per-column shift into it would
/// make the common path carry a branch it never takes.
__global__ void ca_assemble_newton_block(
    double* H, int ldh, int m, const double* C, int ldc,
    const double* R, int filled, int block, const double* shifts,
    double normalization)
{
    const int target = m + 1;
    const int row = static_cast<int>(threadIdx.x);
    if (blockIdx.x != 0) return;
    const bool active = row < target;

    if (active && filled > 0 && filled - 1 < m) {
        const int col = filled - 1;
        H[row + col * ldh] =
            row <= filled
                ? ca_block_factor(C, ldc, R, filled, block, row, 0)
                : 0.0;
    }
    __syncthreads();

    for (int j = 0; j + 1 < block; ++j) {
        const int col = filled + j;
        if (col >= m) break;
        if (active) {
            const double diagonal =
                ca_block_factor(C, ldc, R, filled, block, col, j);
            double rhs =
                shifts[j]
                    * ca_block_factor(C, ldc, R, filled, block, row, j)
                + normalization
                    * ca_block_factor(C, ldc, R, filled, block, row, j + 1);
            for (int k = 0; k < col; ++k)
                rhs -= H[row + k * ldh]
                     * ca_block_factor(C, ldc, R, filled, block, k, j);
            H[row + col * ldh] = rhs / diagonal;
        }
        __syncthreads();
    }
}

/// Build the raw block for the selected basis, in as few launches as possible.
///
/// The monomial and Newton recurrences fit one launch per chunk; Chebyshev
/// needs two predecessors live and so carries a third shared tile, which lowers
/// the width one launch can hold. mpk_chunk exposes that boundary explicitly:
/// the chunked and single-launch dispatches have different ghost redundancy and
/// launch counts, so which one ran is a configuration, not an implementation
/// detail.
///
/// The two kernel families differ in what a chunk costs them. The full-volume
/// Chebyshev path carries a third full tile and so chunks internally; the
/// streamed one carries a third plane per level and does not chunk at all,
/// which is why its Chebyshev branch takes the whole width in one launch.
void build_polynomial_block(
    DeviceWorkspace& w, const double* input, double* output, int steps,
    const GpuPdeOperator& op, const double* face_b, double scale,
    PolynomialBasis basis, double center, double half_width, int chunk_width,
    const SolverOptions& options)
{
    if (steps == 0) {
        // A width-one block is the start vector itself: no recurrence, no ghosts.
        CUDA_CHECK(cudaMemcpyAsync(
            output, input,
            static_cast<std::size_t>(w.ld) * sizeof(double),
            cudaMemcpyDeviceToDevice, nullptr));
        return;
    }
    const bool streamed = use_streamed(
        options, op.n, steps, basis == PolynomialBasis::Chebyshev);
    const int height = stream_height_for(options, op.n);

    if (basis == PolynomialBasis::Chebyshev) {
        if (streamed) {
            CUDA_CHECK(gpu_pde_stream_chebyshev_basis(
                input, output, w.ld, steps, op, face_b, scale,
                center, half_width, height));
        } else {
            CUDA_CHECK(gpu_pde_chebyshev_basis(
                input, output, w.ld, steps, op, face_b, scale,
                center, half_width));
        }
        return;
    }

    // chunk_width 0 dispatches the whole recurrence in one kernel, which is what
    // the slab path does. A positive width splits it, which costs a full-vector
    // copy at every boundary and changes the ghost redundancy of the launch.
    const int width = chunk_width > 0 ? chunk_width : steps;
    const double* current = input;
    int offset = 0;
    while (offset < steps) {
        const int chunk = std::min(width, steps - offset);
        double* column = output + static_cast<int64_t>(offset) * w.ld;
        if (basis == PolynomialBasis::Newton) {
            // offset is the degree within this Arnoldi block. Chunk boundaries
            // advance through the Leja sequence; append_ca_block starts every
            // later Arnoldi block again at shift zero with its new start vector.
            const double* shifts = w.newton_shifts + offset;
            CUDA_CHECK(streamed
                ? gpu_pde_stream_newton_basis(
                      current, column, w.ld, chunk, op, face_b, scale,
                      shifts, half_width, height)
                : gpu_pde_newton_basis(
                      current, column, w.ld, chunk, op, face_b, scale,
                      shifts, half_width));
        } else {
            CUDA_CHECK(streamed
                ? gpu_pde_stream_matrix_powers(
                      current, column, w.ld, chunk, op, face_b, scale, height)
                : gpu_pde_matrix_powers(
                      current, column, w.ld, chunk, op, face_b, scale));
        }
        offset += chunk;
        if (offset < steps) {
            CUDA_CHECK(cudaMemcpyAsync(
                w.scratch, output + static_cast<int64_t>(offset) * w.ld,
                static_cast<std::size_t>(w.ld) * sizeof(double),
                cudaMemcpyDeviceToDevice, nullptr));
            current = w.scratch;
        }
    }
}

/// Build one block, orthogonalize it against the accepted basis, and append it.
///
/// The whole s-step cycle in one place: build with no reductions, project
/// against what is already accepted, certify the block's conditioning, factor
/// it, and recover the Hessenberg columns from the recurrence rather than from
/// dot products. Returns the certified width, which is narrower than requested
/// when the certificate rejects the block and the width falls back.
[[nodiscard]] double append_ca_block(
    cublasHandle_t blas, cusolverDnHandle_t solver, DeviceWorkspace& w,
    const GpuPdeOperator& op, const double* face_b, double scale, int& filled,
    PolynomialBasis basis, double center, double half_width,
    Orthogonalization orthogonalization, const SolverOptions& options,
    bool allow_fallback, int* used_block = nullptr)
{
    const double one = 1.0;
    const double zero = 0.0;
    const double minus_one = -1.0;
    const int requested = std::min(w.block_max, w.target - filled);
    for (int block = requested; block >= 1; --block) {
        // The block consumes columns 0 to block-1, so the recurrence needs block-1
        // steps. The as-measured arm runs one more and discards the extra column.
        const int steps = options.exact_depth ? block - 1 : block;
        build_polynomial_block(
            w, w.start, w.B, steps, op, face_b, scale,
            basis, center, half_width, options.mpk_chunk, options);

        if (filled > 0) {
            CUBLAS_CHECK(cublasDgemm(
                blas, CUBLAS_OP_T, CUBLAS_OP_N,
                filled, block, static_cast<int>(w.ld),
                &one, w.V, static_cast<int>(w.ld),
                w.B, static_cast<int>(w.ld),
                &zero, w.C, w.target));
            CUBLAS_CHECK(cublasDgemm(
                blas, CUBLAS_OP_N, CUBLAS_OP_N,
                static_cast<int>(w.ld), block, filled,
                &minus_one, w.V, static_cast<int>(w.ld),
                w.C, w.target,
                &one, w.B, static_cast<int>(w.ld)));

            CUBLAS_CHECK(cublasDgemm(
                blas, CUBLAS_OP_T, CUBLAS_OP_N,
                filled, block, static_cast<int>(w.ld),
                &one, w.V, static_cast<int>(w.ld),
                w.B, static_cast<int>(w.ld),
                &zero, w.C2, w.target));
            CUBLAS_CHECK(cublasDgemm(
                blas, CUBLAS_OP_N, CUBLAS_OP_N,
                static_cast<int>(w.ld), block, filled,
                &minus_one, w.V, static_cast<int>(w.ld),
                w.C2, w.target,
                &one, w.B, static_cast<int>(w.ld)));

            const dim3 threads(16, 16);
            const dim3 grid(
                static_cast<unsigned>((filled + 15) / 16),
                static_cast<unsigned>((block + 15) / 16));
            ca_add_matrix<<<grid, threads>>>(
                w.C, w.C2, filled, block, w.target);
            CUDA_CHECK(cudaGetLastError());
        }

        double kappa = 0.0;
        bool factored = false;
        switch (orthogonalization) {
            case Orthogonalization::CholQr2:
                factored = cholqr2(solver, w, block, kappa);
                break;
            case Orthogonalization::Tsqr:
                factored = householder_qr(solver, w, block, kappa);
                break;
            case Orthogonalization::BlockGramSchmidt:
                factored = block_gram_schmidt2(blas, w, block, kappa);
                break;
        }
        if (!factored) {
            if (allow_fallback) continue;
            throw std::runtime_error("block orthogonalization failed");
        }

        if (basis == PolynomialBasis::Newton) {
            // The raw Newton basis for each Arnoldi block uses the prefix
            // shifts[0:block-1], matching the per-block reset above.
            ca_assemble_newton_block<<<1, 32>>>(
                w.H, w.target, w.target - 1,
                w.C, w.target, w.local_R, filled, block,
                w.newton_shifts, half_width);
        } else {
            ca_assemble_block<<<1, 32>>>(
                w.H, w.target, w.target - 1,
                w.C, w.target, w.local_R, filled, block,
                basis == PolynomialBasis::Chebyshev ? 1 : 0,
                center, half_width);
        }
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(
            w.V + static_cast<int64_t>(filled) * w.ld, w.B,
            static_cast<std::size_t>(w.ld) * static_cast<std::size_t>(block)
                * sizeof(double),
            cudaMemcpyDeviceToDevice, nullptr));
        filled += block;

        if (filled < w.target) {
            // The next block's start vector, one application of the operator.
            // It follows the same family as the block itself: a solver that
            // built its basis with one kernel and its restart vector with the
            // other would not be measuring either of them.
            const double* previous =
                w.V + static_cast<int64_t>(filled - 1) * w.ld;
            CUDA_CHECK(use_streamed(options, op.n, 1, false)
                ? gpu_pde_stream_matrix_powers(
                      previous, w.B, w.ld, 1, op, face_b, scale,
                      stream_height_for(options, op.n))
                : gpu_pde_matrix_powers(
                      previous, w.B, w.ld, 1, op, face_b, scale));
            CUDA_CHECK(cudaMemcpyAsync(
                w.start, w.B + w.ld,
                static_cast<std::size_t>(w.ld) * sizeof(double),
                cudaMemcpyDeviceToDevice, nullptr));
        }
        if (used_block != nullptr) *used_block = block;
        return kappa;
    }
    throw std::runtime_error("matrix-powers block has no certified width");
}

/// Fill the Arnoldi basis to the requested dimension, block by block, and
/// report the certified width each block achieved.
[[nodiscard]] std::vector<double> build_ca_basis(
    cublasHandle_t blas, cusolverDnHandle_t solver, DeviceWorkspace& w,
    const GpuPdeOperator& op, const double* face_b, double scale,
    PolynomialBasis basis, double center, double half_width,
    Orthogonalization orthogonalization, const SolverOptions& options)
{
    CUDA_CHECK(cudaMemsetAsync(
        w.H, 0,
        static_cast<std::size_t>(w.target) * static_cast<std::size_t>(w.target - 1)
            * sizeof(double), nullptr));

    std::vector<double> kappas;
    int filled = 0;
    while (filled < w.target)
        kappas.push_back(
            append_ca_block(
                blas, solver, w, op, face_b, scale, filled,
                basis, center, half_width, orthogonalization, options,
                false));
    return kappas;
}

struct StepResult {
    int m = 0;
    double residual = std::numeric_limits<double>::infinity();
    double max_kappa = 0.0;
    int min_effective_s = 0;
    int fallback_blocks = 0;
    int accepted_blocks = 0;
    int certificate_retries = 0;
    int reduction_syncs = 1;
    int64_t reduction_values = 1;
    bool converged = false;
};

/// Advance one time step: build the basis, choose m adaptively, and update.
///
/// The Krylov dimension is selected on the device by sweeping candidates and
/// stopping at the first whose residual estimate clears the tolerance, so
/// adaptivity costs no host round-trip per candidate. A step that exhausts
/// m_max without converging is recorded rather than accepted.
[[nodiscard]] StepResult advance_ca_step(
    cublasHandle_t blas, cusolverDnHandle_t solver, DeviceWorkspace& w,
    const GpuPdeOperator& op, const double* face_b, double scale, double tol,
    PolynomialBasis basis, double center, double half_width,
    Orthogonalization orthogonalization, const SolverOptions& options,
    const double* current, double* next)
{
    constexpr double breakdown_tol = 1.0e-14;
    double beta = 0.0;
    {
        // cuBLAS with a host result pointer synchronizes, so this is the step's
        // first host round-trip and belongs in the same account as the reads.
        const auto begin = std::chrono::steady_clock::now();
        CUBLAS_CHECK(cublasDnrm2(
            blas, static_cast<int>(w.ld), current, 1, &beta));
        ++w.host_syncs;
        w.host_sync_ms +=
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - begin).count();
    }
    if (beta < breakdown_tol) {
        CUDA_CHECK(cudaMemcpyAsync(
            next, current, static_cast<std::size_t>(w.ld) * sizeof(double),
            cudaMemcpyDeviceToDevice, nullptr));
        StepResult result;
        result.m = 1;
        result.residual = 0.0;
        result.max_kappa = 1.0;
        result.min_effective_s = w.block_max;
        result.converged = true;
        return result;
    }

    CUDA_CHECK(cudaMemcpyAsync(
        w.start, current, static_cast<std::size_t>(w.ld) * sizeof(double),
        cudaMemcpyDeviceToDevice, nullptr));
    const double inv_beta = 1.0 / beta;
    CUBLAS_CHECK(cublasDscal(
        blas, static_cast<int>(w.ld), &inv_beta, w.start, 1));
    CUDA_CHECK(cudaMemsetAsync(
        w.H, 0,
        static_cast<std::size_t>(w.target) * static_cast<std::size_t>(w.target - 1)
            * sizeof(double), nullptr));

    StepResult result;
    result.min_effective_s = w.block_max;
    int filled = 0;
    int checked = 0;
    while (filled < w.target && !result.converged) {
        const int filled_before = filled;
        const int requested = std::min(w.block_max, w.target - filled);
        int used_block = requested;
        result.max_kappa = std::max(
            result.max_kappa,
                append_ca_block(
                    blas, solver, w, op, face_b, scale, filled,
                    basis, center, half_width, orthogonalization, options,
                    orthogonalization == Orthogonalization::CholQr2,
                    &used_block));
        ++result.accepted_blocks;
        result.reduction_syncs += block_collectives(
            orthogonalization, used_block, filled_before > 0);
        result.reduction_values += block_reduction_values(
            orthogonalization, filled_before, used_block);
        if (used_block < requested) {
            ++result.fallback_blocks;
            const int retries = requested - used_block;
            result.certificate_retries += retries;
            result.reduction_syncs +=
                retries * (filled_before > 0 ? 3 : 1);
            for (int failed = requested; failed > used_block; --failed) {
                result.reduction_values +=
                    (filled_before > 0
                         ? 2LL * static_cast<int64_t>(filled_before) * failed
                         : 0)
                    + static_cast<int64_t>(failed) * failed;
            }
            result.min_effective_s =
                std::min(result.min_effective_s, used_block);
        }

        const int available = std::min(filled - 1, w.target - 1);
        if (available > checked) {
            ca_expm_candidates<kGpuCaMaxM><<<1, 256>>>(
                w.H, w.target, checked + 1, available, beta, tol,
                w.f, w.candidate);
            CUDA_CHECK(cudaGetLastError());
            GpuCaCandidateResult candidate{};
            blocking_read(w, &candidate, w.candidate, sizeof(candidate));
            if (!std::isfinite(candidate.residual))
                throw std::runtime_error("non-finite Arnoldi residual estimate");
            result.residual = candidate.residual;
            result.m =
                candidate.selected_m > 0 ? candidate.selected_m : available;
            result.converged = candidate.selected_m > 0;
        }
        checked = available;
    }

    const double alpha = beta;
    const double zero = 0.0;
    CUBLAS_CHECK(cublasDgemv(
        blas, CUBLAS_OP_N, static_cast<int>(w.ld), result.m,
        &alpha, w.V, static_cast<int>(w.ld), w.f, 1,
        &zero, next, 1));
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
    int64_t modeled_reductions = 0;
    int64_t reduction_values = 0;
    int64_t host_syncs = 0;
    double host_sync_ms = 0.0;
    int unconverged = 0;
};

/// Integrate to expiry and return the final state with the run's statistics.
[[nodiscard]] IntegratorResult solve_ca_integrator(
    const Args& args, cublasHandle_t blas, cusolverDnHandle_t solver,
    DeviceWorkspace& w, const GpuPdeOperator& op, const double* face_b,
    const Eigen::VectorXd& initial, double t_final,
    double center, double half_width, const SolverOptions& options)
{
    const double h = t_final / static_cast<double>(args.steps);
    double* d_state = nullptr;
    CUDA_CHECK(cudaMalloc(
        &d_state, static_cast<std::size_t>(w.ld) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(
        d_state, initial.data(), static_cast<std::size_t>(w.ld) * sizeof(double),
        cudaMemcpyHostToDevice));

    (void)advance_ca_step(
        blas, solver, w, op, face_b, h, args.tol,
        args.basis, center, half_width, args.orthogonalization, options,
        d_state, w.action);
    CUDA_CHECK(cudaDeviceSynchronize());

    IntegratorResult result;
    result.state.resize(w.ld);
    result.min_m = args.m;
    result.min_effective_s = args.s;

    double* current = d_state;
    double* next = w.action;
    // Only the timed steps are accounted; the warm-up above is discarded.
    w.host_syncs = 0;
    w.host_sync_ms = 0.0;
    cudaEvent_t begin{}, end{};
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(begin));
    int m_sum = 0;
    for (int step = 0; step < args.steps; ++step) {
        const StepResult step_result = advance_ca_step(
            blas, solver, w, op, face_b, h, args.tol,
            args.basis, center, half_width, args.orthogonalization, options,
            current, next);
        m_sum += step_result.m;
        result.min_m = std::min(result.min_m, step_result.m);
        result.max_m = std::max(result.max_m, step_result.m);
        result.max_residual = std::max(result.max_residual, step_result.residual);
        result.max_kappa = std::max(result.max_kappa, step_result.max_kappa);
        result.min_effective_s =
            std::min(result.min_effective_s, step_result.min_effective_s);
        result.fallback_blocks += step_result.fallback_blocks;
        result.accepted_blocks += step_result.accepted_blocks;
        result.certificate_retries += step_result.certificate_retries;
        result.modeled_reductions += step_result.reduction_syncs;
        result.reduction_values += step_result.reduction_values;
        if (step_result.fallback_blocks > 0 && result.first_fallback_step == 0)
            result.first_fallback_step = step + 1;
        if (step_result.fallback_blocks > 0)
            w.block_max = std::min(w.block_max, step_result.min_effective_s);
        if (!step_result.converged) ++result.unconverged;
        std::swap(current, next);
    }
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, begin, end));
    result.elapsed_ms = static_cast<double>(elapsed_ms);
    result.avg_m = static_cast<double>(m_sum) / static_cast<double>(args.steps);
    result.host_syncs = w.host_syncs;
    result.host_sync_ms = w.host_sync_ms;

    CUDA_CHECK(cudaMemcpy(
        result.state.data(), current,
        static_cast<std::size_t>(w.ld) * sizeof(double),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(begin));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaFree(d_state));
    return result;
}

/// Where the independent referee's state for this grid and option lives.
[[nodiscard]] std::string referee_path(
    const std::string& dir, int n, bool rainbow)
{
    return dir + "/referee_n" + std::to_string(n)
               + (rainbow ? "_rainbow" : "_basket") + ".bin";
}

/// Read a referee state file: an int64 length followed by that many doubles.
[[nodiscard]] Eigen::VectorXd load_referee_file(const std::string& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return {};
    int64_t size = 0;
    stream.read(reinterpret_cast<char*>(&size), sizeof(size));
    if (!stream || size <= 0)
        throw std::runtime_error("invalid referee file: " + path);
    Eigen::VectorXd referee(size);
    stream.read(
        reinterpret_cast<char*>(referee.data()),
        static_cast<std::streamsize>(size * sizeof(double)));
    if (!stream) throw std::runtime_error("truncated referee file: " + path);
    return referee;
}

/// Write a state file in the layout load_referee_file reads, so a distributed
/// run can be compared against this one.
void save_state_file(const std::string& path, const Eigen::VectorXd& state)
{
    std::ofstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot create state file: " + path);
    const int64_t size = state.size();
    stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
    stream.write(
        reinterpret_cast<const char*>(state.data()),
        static_cast<std::streamsize>(size * sizeof(double)));
    if (!stream) throw std::runtime_error("failed to write state file: " + path);
}

/// The assembled operator, augmented for Basket. Used only by the correctness
/// comparison: the production path never assembles a matrix.
[[nodiscard]] SpMat augmented_operator(const PDESystem& sys, bool rainbow)
{
    if (!rainbow) return build_A_tilde(sys.A, sys.B, sys.N);
    const MatXd zero = MatXd::Zero(sys.N, 3);
    return build_A_tilde(sys.A, zero, sys.N);
}

} // namespace

/// One executable, two modes, which is why main branches once and at one place.
///
/// With --steps it is the production integrator: advance to expiry, gate the
/// result against the referee, optionally save the state. Without it, it is the
/// single-cycle instrument: build one basis, apply one exponential, and report
/// the basis and phi timings that the regime map is calibrated from.
int main(int argc, char** argv)
{
    try {
        // Select the device and report what it is before anything is measured
        // on it; a contended device makes every timing below diagnostic.
        const Args args = parse_args(argc, argv);
        CaPricingModel model;
        model.expiry = args.expiry;

        CUDA_CHECK(cudaSetDevice(args.device));
        const DeviceContention contention = check_device_contention();
        report_toolkit();
        report_contention(contention);

        // The device's opt-in shared-memory ceiling, queried once. The streamed
        // family's footprint grows with the block width faster than the accepted
        // kernel's, so whether a width is admissible is a device fact and the
        // dispatch needs it before it can choose.
        cudaDeviceProp device_properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&device_properties, args.device));
        const std::size_t shared_optin_bytes =
            static_cast<std::size_t>(device_properties.sharedMemPerBlockOptin);

        // Which matrix-powers configuration this binary was built on. The tile
        // extents and the thread count are compile-time constants, so a solver
        // timing cannot be attributed to a configuration from its command line;
        // this line is how a run says which one it is.
        {
            // Reported at the requested block width, clamped to what the kernel
            // instantiates. The staged volume and the shared request are
            // properties of that width, so a width the dispatch would reject
            // must not be used to describe the launch.
            const int width = std::clamp(args.s, 1, kMpkMaxS);
            // The family the dispatch resolves to at this grid, not the one
            // requested: under auto they differ, and it is the resolved one
            // that describes the launch a timing belongs to.
            SolverOptions probe;
            probe.family = args.family;
            probe.stream_height = args.stream_height;
            probe.shared_optin_bytes = shared_optin_bytes;
            const bool streamed = use_streamed(
                probe, args.n, width,
                args.basis == PolynomialBasis::Chebyshev);
            const int height = stream_height_for(probe, args.n);
            const MpkLaunchRecord launch =
                streamed
                    ? mpk_stream_launch_record(
                          args.n, width, height,
                          args.basis == PolynomialBasis::Chebyshev)
                    : mpk_launch_record(
                          width, gpu_pde_matrix_powers_shared_bytes(width));
            std::printf(
                "MPK_LAUNCH configuration=%s requested_family=%s family=%s "
                "tile_x=%d tile_y=%d "
                "tile_z=%d stream_height=%d threads_per_block=%d shared_pad_x=%d "
                "pitch_alignment=%d elide_basis_barrier=%d restrict=%d "
                "interior_points=%lld staged_points=%lld "
                "redundant_fraction=%.6f points_per_thread=%.6f "
                "dynamic_shared_bytes=%zu\n",
                CAKSM_MPK_CONFIGURATION, family_selection_name(args.family),
                launch.family, launch.tile_x,
                launch.tile_y, launch.tile_z, launch.stream_height,
                launch.threads_per_block,
                launch.shared_pad_x, launch.pitch_alignment,
                launch.elide_basis_barrier, launch.use_restrict,
                static_cast<long long>(launch.interior_points),
                static_cast<long long>(launch.staged_points),
                launch.redundant_fraction, launch.points_per_thread,
                launch.dynamic_shared_bytes);
        }

        // Resolve the arm and the matrix-powers dispatch it implies.
        SolverOptions options;
        options.exact_depth = args.arm == "exact-depth";
        // The exact-depth arm dispatches the whole block in one launch, which is
        // what the slab path already does, so the two solvers become comparable at
        // a given width. --mpk-chunk overrides either default for monomial and
        // Newton. Chebyshev carries a two-vector recurrence across launches and
        // currently uses the kernel's fixed S=3 cap.
        options.mpk_chunk =
            args.basis == PolynomialBasis::Chebyshev
                ? kMpkPreferredS
                : (args.mpk_chunk >= 0
                       ? args.mpk_chunk
                       : (options.exact_depth ? 0 : kMpkPreferredS));
        options.family = args.family;
        options.stream_height = args.stream_height;
        options.shared_optin_bytes = shared_optin_bytes;

        if (args.memory_report) {
            if (contention.contended)
                throw std::runtime_error(
                    "memory report requires an idle device");
            report_integrator_memory(args);
            return EXIT_SUCCESS;
        }

        // Build the problem. The production path never assembles a matrix, so
        // the matrix-free builder is used wherever it suffices; the other bases
        // and the single-cycle mode still need the assembled operator.
        const PDESystem sys =
            args.steps > 0 && args.basis == PolynomialBasis::Monomial
                ? build_gpu_pde_system(args.n, model, args.rainbow)
                : build_pde_system(
                    args.n, model.strike, model.rate, model.expiry,
                    model.sigma, model.rho, model.weight, model.spot,
                    model.alpha, args.rainbow);
        const GpuPdeOperator op =
            make_gpu_pde_operator(sys, model, args.rainbow);
        std::vector<double> face_b = make_gpu_face_b(sys, model);
        const bool scaled_augmentation =
            args.arm == "scaled-augmentation";
        ca_referee::AugmentationScaling augmentation_scaling;
        if (scaled_augmentation) {
            augmentation_scaling = ca_referee::make_augmentation_scaling(
                ca_referee::face_forcing_1norm(face_b, args.n));
            for (double& value : face_b)
                value *= augmentation_scaling.eta;
        }
        const int64_t ld = static_cast<int64_t>(sys.N) + 3;
        const double operator_scale =
            args.steps > 0
                ? model.expiry / static_cast<double>(args.steps)
                : args.scale;
        // Newton and Chebyshev need an interval enclosing the spectrum: one
        // supplies the shifts, the other the affine map. Monomial needs neither.
        const GershgorinInterval interval =
            args.basis != PolynomialBasis::Monomial
                ? scaled_gershgorin_interval(
                    sys.A, operator_scale, true)
                : GershgorinInterval{};
        const double basis_center =
            0.5 * (interval.lower + interval.upper);
        const double basis_half_width =
            0.5 * (interval.upper - interval.lower);
        if (args.basis != PolynomialBasis::Monomial
            && !(basis_half_width > 0.0))
            throw std::runtime_error("spectral enclosure has zero width");
        const std::vector<double> newton_shifts =
            args.basis == PolynomialBasis::Newton
                ? real_leja_shifts(interval, args.s)
                : std::vector<double>{};

        // The augmented start vector: the payoff, plus the three-component tail
        // that carries the Basket boundary forcing inside the exponential.
        Eigen::VectorXd initial(ld);
        initial.setZero();
        initial.head(sys.N) = sys.u0;
        if (!args.rainbow) {
            initial.tail(3) = make_s_vec(0.0);
            if (scaled_augmentation)
                initial.tail(3) *= augmentation_scaling.eta_inverse;
        }
        Eigen::VectorXd start = initial;
        start.normalize();

        // Bring up the libraries and take every device allocation at once, so
        // no allocation happens inside a timed region.
        cublasHandle_t blas{};
        cusolverDnHandle_t solver{};
        CUBLAS_CHECK(cublasCreate(&blas));
        CUSOLVER_CHECK(cusolverDnCreate(&solver));

        DeviceWorkspace w;
        allocate_workspace(
            w, ld, args.m, args.s, args.orthogonalization, solver);
        if (args.basis == PolynomialBasis::Newton) {
            CUDA_CHECK(cudaMalloc(
                &w.newton_shifts,
                static_cast<std::size_t>(args.s) * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(
                w.newton_shifts, newton_shifts.data(),
                static_cast<std::size_t>(args.s) * sizeof(double),
                cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemcpy(
            w.start, start.data(), static_cast<std::size_t>(ld) * sizeof(double),
            cudaMemcpyHostToDevice));

        double* d_face_b = nullptr;
        if (!args.rainbow) {
            CUDA_CHECK(cudaMalloc(&d_face_b, face_b.size() * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(
                d_face_b, face_b.data(), face_b.size() * sizeof(double),
                cudaMemcpyHostToDevice));
        }

        if (args.steps > 0) {
            // Integrator mode. Repeat the whole solve and keep the median run,
            // so a reported cycle time is a distribution and not one sample.
            const double host_sync_probe_us = probe_host_sync_us(w);
            std::vector<IntegratorResult> runs;
            runs.reserve(static_cast<std::size_t>(args.repeats));
            for (int repeat = 0; repeat < args.repeats; ++repeat) {
                w.block_max = args.s;
                runs.push_back(solve_ca_integrator(
                    args, blas, solver, w, op, d_face_b, initial, model.expiry,
                    basis_center, basis_half_width, options));
            }
            std::sort(
                runs.begin(), runs.end(),
                [](const IntegratorResult& lhs, const IntegratorResult& rhs) {
                    return lhs.elapsed_ms < rhs.elapsed_ms;
                });
            const double solve_min_ms = runs.front().elapsed_ms;
            const double solve_max_ms = runs.back().elapsed_ms;
            IntegratorResult result =
                runs[static_cast<std::size_t>(args.repeats / 2)];
            // The solver advances the similarity-transformed state. Restore
            // the three forcing coordinates before any validation or output so
            // every reported quantity remains in the original representation.
            if (scaled_augmentation)
                result.state.tail(3) *= augmentation_scaling.eta;
            // Extract the price and the boundary tail. The tail has a known
            // closed form, so its error measures whether the augmentation was
            // propagated correctly rather than merely finitely.
            const Eigen::VectorXd solution = result.state.head(sys.N);
            // The four-decimal values published by Dang, Christara and Jackson.
            // A historical comparison, not a reference: they carry no stated
            // uncertainty. The accepted references, with theirs, are written by
            // ./financial-reference into data/financial-validation.
            const double historical_comparison_price =
                args.rainbow ? 4.4450 : 13.2449;
            const double price = extract_price(solution, sys.grid, model.spot);
            const double literature_error =
                model.expiry == 1.0
                    ? std::abs(price - historical_comparison_price)
                    : std::numeric_limits<double>::quiet_NaN();
            const double tail_error = args.rainbow
                ? result.state.tail(3).norm()
                : (result.state.tail(3) - make_s_vec(model.expiry)).norm();
            if (!args.save_state.empty())
                save_state_file(args.save_state, result.state);

            // Compare against the independent referee, which shares no
            // arithmetic with this solver.
            const bool referee_requested = !args.referee_dir.empty();
            double ode_error = std::numeric_limits<double>::quiet_NaN();
            double referee_price_error =
                std::numeric_limits<double>::quiet_NaN();
            if (referee_requested) {
                const std::string path =
                    referee_path(args.referee_dir, args.n, args.rainbow);
                const Eigen::VectorXd referee = load_referee_file(path);
                if (referee.size() != sys.N)
                    throw std::runtime_error(
                        "referee size does not match the PDE system: " + path);
                ode_error =
                    (extract_cube(solution, sys.grid, model.spot)
                     - extract_cube(referee, sys.grid, model.spot)).norm();
                referee_price_error = std::abs(
                    price - extract_price(referee, sys.grid, model.spot));
            }

            // Report the run in full, whether it passed, so a failure is
            // diagnosable from its own transcript.
            std::printf("GPU CA exponential integrator\n");
            std::printf(
                "  option: %s | basis=%s | orth=%s | n=%d | N=%d | steps=%d | tol=%.3e | m_max=%d | s=%d | expiry=%.6g\n",
                args.rainbow ? "rainbow" : "basket", basis_name(args.basis),
                orthogonalization_name(args.orthogonalization),
                args.n, sys.N, args.steps, args.tol,
                args.m, args.s, model.expiry);
            const std::string dispatch =
                args.basis == PolynomialBasis::Chebyshev
                    ? "Chebyshev chunks capped at S="
                        + std::to_string(
                            std::min(kMpkPreferredS, args.s))
                    : (options.mpk_chunk > 0
                           ? "chunked at S="
                               + std::to_string(options.mpk_chunk)
                           : "one launch per block");
            std::printf(
                "  arm=%s | matrix-powers dispatch=%s\n",
                args.arm.c_str(), dispatch.c_str());
            if (scaled_augmentation)
                std::printf(
                    "  augmentation: exact power-of-two similarity | "
                    "exponent=%d eta=%.17g | forcing 1-norm=%.17g "
                    "scaled=%.17g\n",
                    augmentation_scaling.exponent,
                    augmentation_scaling.eta,
                    augmentation_scaling.forcing_norm_1,
                    augmentation_scaling.scaled_forcing_norm_1);
            if (args.basis == PolynomialBasis::Chebyshev)
                std::printf(
                    "  spectral enclosure: [%.6e, %.6e] | center=%.6e | half-width=%.6e\n",
                    interval.lower, interval.upper,
                    basis_center, basis_half_width);
            else if (args.basis == PolynomialBasis::Newton) {
                std::printf(
                    "  spectral enclosure: [%.6e, %.6e] | normalization=%.6e | shifts:",
                    interval.lower, interval.upper, basis_half_width);
                for (double shift : newton_shifts)
                    std::printf(" %.3e", shift);
                std::printf("\n");
            }
            std::printf(
                "  Krylov m: min=%d avg=%.2f max=%d | unconverged=%d\n",
                result.min_m, result.avg_m, result.max_m, result.unconverged);
            std::printf(
                "  max residual: %.6e | max block kappa: %.6e\n",
                result.max_residual, result.max_kappa);
            std::printf(
                "  certified width: requested=%d min=%d | fallback blocks=%d\n",
                args.s, result.min_effective_s, result.fallback_blocks);
            std::printf(
                "  CA blocks: %.2f/step | certificate retries=%lld | first fallback step=%d\n",
                static_cast<double>(result.accepted_blocks)
                    / static_cast<double>(args.steps),
                static_cast<long long>(result.certificate_retries),
                result.first_fallback_step);
            std::printf(
                "  reduction operations: %.2f/step | payload: %.1f bytes/step | tier=device-local\n",
                static_cast<double>(result.modeled_reductions)
                    / static_cast<double>(args.steps),
                static_cast<double>(result.reduction_values) * sizeof(double)
                    / static_cast<double>(args.steps));
            const double host_syncs_per_step =
                static_cast<double>(result.host_syncs)
                / static_cast<double>(args.steps);
            if (!contention.contended)
                std::printf(
                    "  host synchronizations: %.2f/step | probed round-trip=%.3f us | floor=%.3f ms/step | observed stall=%.3f ms/step\n",
                    host_syncs_per_step, host_sync_probe_us,
                    host_syncs_per_step * host_sync_probe_us * 1.0e-3,
                    result.host_sync_ms / static_cast<double>(args.steps));
            else
                std::printf(
                    "  host synchronizations: %.2f/step | cost withheld, the device is contended\n",
                    host_syncs_per_step);
            if (std::isfinite(literature_error))
                std::printf(
                    "  price: %.8f | historical comparison error: %.6e\n",
                    price, literature_error);
            else
                std::printf(
                    "  price: %.8f | historical comparison error: not defined for expiry=%.6g\n",
                    price, model.expiry);
            std::printf("  boundary-state error: %.6e\n", tail_error);
            if (referee_requested) {
                std::printf(
                    "  ODE referee: cube=%.6e price=%.6e\n",
                    ode_error, referee_price_error);
            } else {
                std::printf("  ODE referee error: not requested\n");
            }
            const bool state_passed = result.state.allFinite();
            const bool convergence_passed = result.unconverged == 0;
            const bool boundary_passed =
                std::isfinite(tail_error) && tail_error <= args.tol;
            const bool referee_passed =
                !referee_requested
                || (std::isfinite(ode_error) && ode_error <= args.tol);
            const bool passed =
                state_passed && convergence_passed
                && boundary_passed && referee_passed;
            std::printf(
                "  validation: %s | state=%s convergence=%s boundary=%s referee=%s\n",
                passed ? "PASS" : "FAIL",
                state_passed ? "PASS" : "FAIL",
                convergence_passed ? "PASS" : "FAIL",
                boundary_passed ? "PASS" : "FAIL",
                referee_passed ? "PASS" : "FAIL");
            // A co-tenant moves every timing at once and plausibly, so a contended
            // run reports its numerical gates and no timing distribution, matching
            // what calibrate-gpu-p2p --production refuses to measure.
            if (!contention.contended)
                std::printf(
                    "  solve median: %.3f ms | cycle: %.3f ms/step | distribution: [%.3f, %.3f] ms (%d runs)\n",
                    result.elapsed_ms,
                    result.elapsed_ms / static_cast<double>(args.steps),
                    solve_min_ms, solve_max_ms, args.repeats);
            else
                std::printf(
                    "  solve timing: withheld, the device is contended | correctness run only (%d runs)\n",
                    args.repeats);

            if (d_face_b != nullptr) CUDA_CHECK(cudaFree(d_face_b));
            free_workspace(w);
            CUBLAS_CHECK(cublasDestroy(blas));
            CUSOLVER_CHECK(cusolverDnDestroy(solver));
            if (!state_passed) {
                std::fprintf(stderr, "FAIL: state contains non-finite values\n");
                return EXIT_FAILURE;
            }
            if (result.unconverged != 0) {
                std::fprintf(
                    stderr,
                    "FAIL: %d step(s) reached m_max without meeting tolerance\n",
                    result.unconverged);
                return EXIT_FAILURE;
            }
            if (!boundary_passed) {
                std::fprintf(
                    stderr,
                    "FAIL: boundary-state error exceeds tolerance\n");
                return EXIT_FAILURE;
            }
            if (!referee_passed) {
                std::fprintf(
                    stderr,
                    "FAIL: ODE referee error exceeds tolerance\n");
                return EXIT_FAILURE;
            }
            return EXIT_SUCCESS;
        }

        // Single-cycle mode: one basis and one exponential, timed separately.
        // Device events rather than host clocks, because the two phases are
        // what the regime map's horizontal and vertical coordinates are built
        // from and a host stall would land in the wrong one.
        cudaEvent_t begin{}, basis_end{}, phi_end{};
        CUDA_CHECK(cudaEventCreate(&begin));
        CUDA_CHECK(cudaEventCreate(&basis_end));
        CUDA_CHECK(cudaEventCreate(&phi_end));

        // Warm up: first-call allocation and JIT would otherwise land in the
        // first timed repeat.
        const double one = 1.0;
        const double zero = 0.0;
        (void)build_ca_basis(
            blas, solver, w, op, d_face_b, args.scale,
            args.basis, basis_center, basis_half_width,
            args.orthogonalization, options);
        ca_expm_action<kGpuCaMaxM><<<1, 256>>>(
            w.H, args.m + 1, args.m, w.f, w.exp_squarings, w.exp_terms);
        CUDA_CHECK(cudaGetLastError());
        CUBLAS_CHECK(cublasDgemv(
            blas, CUBLAS_OP_N, static_cast<int>(ld), args.m,
            &one, w.V, static_cast<int>(ld), w.f, 1,
            &zero, w.action, 1));
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> basis_timings;
        std::vector<double> phi_timings;
        std::vector<double> total_timings;
        basis_timings.reserve(static_cast<std::size_t>(args.repeats));
        phi_timings.reserve(static_cast<std::size_t>(args.repeats));
        total_timings.reserve(static_cast<std::size_t>(args.repeats));
        std::vector<double> kappas;
        for (int repeat = 0; repeat < args.repeats; ++repeat) {
            CUDA_CHECK(cudaMemcpy(
                w.start, start.data(),
                static_cast<std::size_t>(ld) * sizeof(double),
                cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaEventRecord(begin));
            std::vector<double> repeat_kappas = build_ca_basis(
                blas, solver, w, op, d_face_b, args.scale,
                args.basis, basis_center, basis_half_width,
                args.orthogonalization, options);
            CUDA_CHECK(cudaEventRecord(basis_end));

            ca_expm_action<kGpuCaMaxM><<<1, 256>>>(
                w.H, args.m + 1, args.m,
                w.f, w.exp_squarings, w.exp_terms);
            CUDA_CHECK(cudaGetLastError());
            CUBLAS_CHECK(cublasDgemv(
                blas, CUBLAS_OP_N, static_cast<int>(ld), args.m,
                &one, w.V, static_cast<int>(ld), w.f, 1,
                &zero, w.action, 1));
            CUDA_CHECK(cudaEventRecord(phi_end));
            CUDA_CHECK(cudaEventSynchronize(phi_end));

            float repeat_basis_ms = 0.0f;
            float repeat_total_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(
                &repeat_basis_ms, begin, basis_end));
            CUDA_CHECK(cudaEventElapsedTime(
                &repeat_total_ms, begin, phi_end));
            basis_timings.push_back(static_cast<double>(repeat_basis_ms));
            phi_timings.push_back(
                static_cast<double>(repeat_total_ms - repeat_basis_ms));
            total_timings.push_back(static_cast<double>(repeat_total_ms));
            kappas = std::move(repeat_kappas);
        }
        std::sort(basis_timings.begin(), basis_timings.end());
        std::sort(phi_timings.begin(), phi_timings.end());
        std::sort(total_timings.begin(), total_timings.end());
        const std::size_t median_index =
            static_cast<std::size_t>(args.repeats / 2);
        const double basis_ms = basis_timings[median_index];
        const double phi_ms = phi_timings[median_index];
        const double total_ms = total_timings[median_index];

        Eigen::MatrixXd gpu_H(args.m + 1, args.m);
        Eigen::MatrixXd gpu_V(ld, args.m);
        Eigen::VectorXd gpu_f(args.m);
        Eigen::VectorXd gpu_action(ld);
        CUDA_CHECK(cudaMemcpy(
            gpu_H.data(), w.H,
            static_cast<std::size_t>(args.m + 1) * static_cast<std::size_t>(args.m)
                * sizeof(double),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            gpu_V.data(), w.V,
            static_cast<std::size_t>(ld) * static_cast<std::size_t>(args.m)
                * sizeof(double),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            gpu_f.data(), w.f, static_cast<std::size_t>(args.m) * sizeof(double),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            gpu_action.data(), w.action, static_cast<std::size_t>(ld) * sizeof(double),
            cudaMemcpyDeviceToHost));
        int exp_squarings = 0, exp_terms = 0;
        CUDA_CHECK(cudaMemcpy(
            &exp_squarings, w.exp_squarings, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            &exp_terms, w.exp_terms, sizeof(int), cudaMemcpyDeviceToHost));

        const SpMat A_scaled = args.scale * augmented_operator(sys, args.rainbow);
        const ArnoldiResult cpu =
            mgs_arnoldi(A_scaled, start, args.m);
        const Eigen::VectorXd cpu_f =
            cpu.H.topLeftCorner(args.m, args.m).exp().col(0);
        const Eigen::VectorXd cpu_action =
            cpu.V.leftCols(args.m) * cpu_f;

        const Eigen::MatrixXd cpu_H =
            cpu.H.topLeftCorner(args.m + 1, args.m);
        const double h_abs = (gpu_H - cpu_H).lpNorm<Eigen::Infinity>();
        const double h_rel =
            h_abs / std::max(1.0, cpu_H.lpNorm<Eigen::Infinity>());
        const double f_abs = (gpu_f - cpu_f).lpNorm<Eigen::Infinity>();
        const double f_rel = f_abs / std::max(1.0, cpu_f.lpNorm<Eigen::Infinity>());
        const double action_abs =
            (gpu_action - cpu_action).lpNorm<Eigen::Infinity>();
        const double action_rel =
            action_abs / std::max(1.0, cpu_action.lpNorm<Eigen::Infinity>());
        const double ortho =
            (gpu_V.transpose() * gpu_V
             - Eigen::MatrixXd::Identity(args.m, args.m)).norm();

        std::printf("GPU CA-Arnoldi\n");
        std::printf(
            "  option: %s | basis=%s | orth=%s | n=%d | N=%d | m=%d | s=%d | scale=%.3e | expiry=%.6g\n",
            args.rainbow ? "rainbow" : "basket", basis_name(args.basis),
            orthogonalization_name(args.orthogonalization),
            args.n, sys.N, args.m, args.s, args.scale, model.expiry);
        if (args.basis == PolynomialBasis::Chebyshev)
            std::printf(
                "  spectral enclosure: [%.6e, %.6e] | center=%.6e | half-width=%.6e\n",
                interval.lower, interval.upper,
                basis_center, basis_half_width);
        else if (args.basis == PolynomialBasis::Newton) {
            std::printf(
                "  spectral enclosure: [%.6e, %.6e] | normalization=%.6e | shifts:",
                interval.lower, interval.upper, basis_half_width);
            for (double shift : newton_shifts)
                std::printf(" %.3e", shift);
            std::printf("\n");
        }
        std::printf("  block kappa:");
        for (double kappa : kappas) std::printf(" %.3e", kappa);
        std::printf("\n");
        int orth_collectives = 0;
        int64_t orth_values = 0;
        int fixed_filled = 0;
        for (std::size_t index = 0; index < kappas.size(); ++index) {
            const int block = std::min(args.s, args.m + 1 - fixed_filled);
            orth_collectives += block_collectives(
                args.orthogonalization, block, fixed_filled > 0);
            orth_values += block_reduction_values(
                args.orthogonalization, fixed_filled, block);
            fixed_filled += block;
        }
        std::printf(
            "  orthogonalization: reductions=%d | payload=%lld values | tier=device-local | input=%.3f MiB\n",
            orth_collectives, static_cast<long long>(orth_values),
            static_cast<double>(ld) * static_cast<double>(args.m + 1)
                * sizeof(double) / (1024.0 * 1024.0));
        if (args.orthogonalization == Orthogonalization::Tsqr)
            std::printf(
                "  TSQR tree: ranks=1 height=0 | local Householder QR only\n");
        else if (args.orthogonalization
                 == Orthogonalization::BlockGramSchmidt)
            std::printf(
                "  BGS2: two intra-block projection passes per column | positive-norm diagonal\n");
        std::printf("  orthogonality loss: %.6e\n", ortho);
        std::printf("  H error:      abs=%.6e rel=%.6e\n", h_abs, h_rel);
        std::printf("  exp(H)e1:     abs=%.6e rel=%.6e | squarings=%d terms=%d\n",
                    f_abs, f_rel, exp_squarings, exp_terms);
        std::printf("  Krylov action: abs=%.6e rel=%.6e\n", action_abs, action_rel);
        if (!contention.contended) {
            std::printf(
                "  basis+assembly: %.3f ms median [%.3f, %.3f] | phi+update: %.3f ms median [%.3f, %.3f] | total: %.3f ms median [%.3f, %.3f] (%d runs)\n",
                basis_ms, basis_timings.front(), basis_timings.back(),
                phi_ms, phi_timings.front(), phi_timings.back(),
                total_ms, total_timings.front(), total_timings.back(),
                args.repeats);
            std::printf(
                "ARNOLDI_TIMING basis_min_ms=%.9e basis_median_ms=%.9e basis_max_ms=%.9e phi_min_ms=%.9e phi_median_ms=%.9e phi_max_ms=%.9e total_min_ms=%.9e total_median_ms=%.9e total_max_ms=%.9e repeats=%d contended=0\n",
                basis_timings.front(), basis_ms, basis_timings.back(),
                phi_timings.front(), phi_ms, phi_timings.back(),
                total_timings.front(), total_ms, total_timings.back(),
                args.repeats);
        } else {
            std::printf(
                "  basis+assembly timing: withheld, the device is contended | correctness run only (%d runs)\n",
                args.repeats);
            std::printf(
                "ARNOLDI_TIMING repeats=%d contended=1\n",
                args.repeats);
        }

        CUDA_CHECK(cudaEventDestroy(begin));
        CUDA_CHECK(cudaEventDestroy(basis_end));
        CUDA_CHECK(cudaEventDestroy(phi_end));
        if (d_face_b != nullptr) CUDA_CHECK(cudaFree(d_face_b));
        free_workspace(w);
        CUBLAS_CHECK(cublasDestroy(blas));
        CUSOLVER_CHECK(cusolverDnDestroy(solver));

        const bool correct =
            ortho < 5e-12 && h_rel < 5e-10
            && f_rel < 5e-11 && action_rel < 5e-10;
        if (!correct) {
            std::fprintf(stderr, "FAIL: GPU CA-Arnoldi validation mismatch\n");
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return EXIT_FAILURE;
    }
}
