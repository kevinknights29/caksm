/**
 * @file ca_referee_gpu.cu
 * @brief Full-accuracy GPU matrix-exponential referee using assembled CSR.
 *
 * The independent check on the production solver. It assembles the operator as
 * CSR and applies the Al-Mohy and Higham Taylor action with cuSPARSE, so it
 * shares no arithmetic with the matrix-free stencil it referees: agreement
 * between the two is evidence precisely because they have no code in common.
 *
 * Two augmentation methods. fixed-compatibility is the reproduction control and
 * the Rainbow baseline. scaled-augmentation is the promoted Basket baseline; the
 * Basket forcing columns dominate the augmented 1-norm, and scaling them by an
 * exact power of two is what makes a large grid reachable at all. Each writes to
 * its own namespace so a changed method can never overwrite a promoted result.
 *
 * Usage:
 *
 *   ./ca-referee-gpu --n N --referee-dir DIR [--option basket|rainbow|both]
 *       [--method fixed-compatibility|scaled-augmentation]
 *       [--verify FILE_OR_DIR] [--repeats K] [--device D] [--profiled]
 *
 * Pass --profiled when running under a replay profiler: it labels the timing
 * mode in the sidecar so a profiled wall time is never read as a performance
 * constant.
 *
 * @author Kevin Knights
 * @date 2026-07-28
 */

#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Sparse>

#include "ca_referee_scaled.hpp"
#include "gpu_contention.cuh"
#include "pde_operators.hpp"

#ifndef CAKSM_GIT_REVISION
#define CAKSM_GIT_REVISION "unknown"
#endif

#define CUDA_CHECK(call) do {                                                     \
    const cudaError_t e_ = (call);                                                \
    if (e_ != cudaSuccess) {                                                      \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                     cudaGetErrorString(e_));                                     \
        std::exit(EXIT_FAILURE);                                                  \
    }                                                                             \
} while (0)

#define CUSPARSE_CHECK(call) do {                                                  \
    const cusparseStatus_t e_ = (call);                                           \
    if (e_ != CUSPARSE_STATUS_SUCCESS) {                                          \
        std::fprintf(stderr, "cuSPARSE error at %s:%d: %d\n", __FILE__, __LINE__, \
                     static_cast<int>(e_));                                       \
        std::exit(EXIT_FAILURE);                                                  \
    }                                                                             \
} while (0)

namespace {

using Clock = std::chrono::steady_clock;
using Milliseconds = std::chrono::duration<double, std::milli>;
using RowSpMat = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;

constexpr int kTaylorMax = 55;
constexpr double kThetaRef = 9.9;
constexpr double kToleranceRef = 1.1e-16;
constexpr int kVectorThreads = 256;
constexpr int kMaxReductionBlocks = 1024;

/// Which payoff to referee. Both runs the two in one invocation, which is how
/// the weak-scaling reference set is generated.
enum class OptionMode {
    Basket,
    Rainbow,
    Both
};

/// How the boundary forcing is represented. The two agree on the answer and
/// differ enormously in the substep count they need to reach it.
enum class RefereeMethod {
    FixedCompatibility,
    ScaledAugmentation
};

/// Command line, already validated by parse_args.
struct Args {
    int n = 31;
    int device = 0;
    int repeats = 1;
    bool profiled = false;
    OptionMode option = OptionMode::Both;
    RefereeMethod method = RefereeMethod::FixedCompatibility;
    std::string referee_dir;
    std::string verify;
};

/// The option the referee prices. Held here rather than shared with the solver
/// so the two cannot drift through a common edit.
struct Model {
    double strike = 100.0;
    double rate = 0.04;
    double expiry = 1.0;
    std::array<double, 3> sigma{0.30, 0.35, 0.40};
    std::array<double, 3> rho{0.50, 0.50, 0.50};
    std::array<double, 3> weight{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
    std::array<double, 3> spot{100.0, 100.0, 100.0};
    double alpha = 2.85;
};

/// Host-side CSR, in the exact layout cuSPARSE expects, so upload is a copy.
struct HostCsr {
    std::vector<int> row_offsets;
    std::vector<int> column_indices;
    std::vector<double> values;
    int rows = 0;
    int nnz = 0;
};

/// What the Taylor action did: the degree each substep stopped at, the total
/// SpMV count, and the norm maxima the stopping test compared. An unconverged
/// substep is counted rather than tolerated; it blocks the write gate.
struct ActionStats {
    std::vector<int> degrees;
    int64_t spmv_count = 0;
    int unconverged_substeps = 0;
    double max_previous = 0.0;
    double max_current = 0.0;
    double max_sum = 0.0;
};

/// Distance from a stored reference, on the whole field, the cube around the
/// spot, and the price.
struct ErrorMetrics {
    double full_abs = 0.0;
    double full_rel = 0.0;
    double cube_abs = 0.0;
    double price_abs = 0.0;
    bool passed = true;
};

/// Distance between the device SpMV and the host Eigen product on one vector.
/// This gate runs before any action: an operator that disagrees makes every
/// later number meaningless.
struct OperatorMetrics {
    double full_abs = 0.0;
    double full_rel = 0.0;
    double boundary_abs = 0.0;
    double augmented_abs = 0.0;
    bool finite = true;
};

/// Every device allocation and cuSPARSE descriptor the run owns. Two state and
/// two term buffers, because the Taylor recurrence alternates between them
/// rather than aliasing an SpMV's input and output.
struct Workspace {
    int rows = 0;
    int reduction_blocks = 0;
    int* row_offsets = nullptr;
    int* column_indices = nullptr;
    double* values = nullptr;
    double* initial = nullptr;
    double* state_a = nullptr;
    double* state_b = nullptr;
    double* term_a = nullptr;
    double* term_b = nullptr;
    double* partial_maxima = nullptr;
    double* maxima = nullptr;
    void* spmv_buffer = nullptr;
    std::size_t spmv_buffer_bytes = 0;
    cusparseSpMatDescr_t matrix = nullptr;
    cusparseDnVecDescr_t term_a_desc = nullptr;
    cusparseDnVecDescr_t term_b_desc = nullptr;
};

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

        if (arg == "--n") args.n = std::stoi(next());
        else if (arg == "--device") args.device = std::stoi(next());
        else if (arg == "--repeats") args.repeats = std::stoi(next());
        else if (arg == "--profiled") args.profiled = true;
        else if (arg == "--referee-dir") args.referee_dir = next();
        else if (arg == "--verify") args.verify = next();
        else if (arg == "--method") {
            const std::string method = next();
            if (method == "fixed-compatibility")
                args.method = RefereeMethod::FixedCompatibility;
            else if (method == "scaled-augmentation")
                args.method = RefereeMethod::ScaledAugmentation;
            else
                throw std::invalid_argument(
                    "--method must be fixed-compatibility or "
                    "scaled-augmentation");
        }
        else if (arg == "--option") {
            const std::string option = next();
            if (option == "basket") args.option = OptionMode::Basket;
            else if (option == "rainbow") args.option = OptionMode::Rainbow;
            else if (option == "both") args.option = OptionMode::Both;
            else
                throw std::invalid_argument(
                    "--option must be basket, rainbow, or both");
        } else if (arg == "--help") {
            std::printf(
                "Usage: ./ca-referee-gpu [--n N] [--option basket|rainbow|both]\n"
                "                        [--device ID] --referee-dir DIR\n"
                "                        [--verify FILE_OR_DIR] [--repeats K]\n"
                "                        [--profiled]\n"
                "                        [--method fixed-compatibility|\n"
                "                                  scaled-augmentation]\n"
                "Successful baseline runs are saved automatically as\n"
                "DIR/referee_n<N>_<option>.bin. Scaled augmentation is the\n"
                "Basket baseline; fixed compatibility is the Rainbow baseline.\n"
                "Fixed Basket controls use DIR/compatibility/fixed-compatibility/.\n");
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::invalid_argument("unknown flag: " + arg);
        }
    }

    if (args.n < 9 || args.n % 2 == 0)
        throw std::invalid_argument("--n must be an odd integer at least 9");
    if (args.repeats < 1)
        throw std::invalid_argument("--repeats must be positive");
    if (args.referee_dir.empty())
        throw std::invalid_argument("--referee-dir is required");
    if (!args.verify.empty()
        && args.option == OptionMode::Both
        && !std::filesystem::is_directory(args.verify))
        throw std::invalid_argument(
            "--verify must name a directory when --option both is used");
    if (args.method == RefereeMethod::ScaledAugmentation
        && args.option != OptionMode::Basket)
        throw std::invalid_argument(
            "scaled-augmentation requires --option basket");
    return args;
}

/// The method's name as it appears in paths and sidecars. This string is part
/// of a result's identity, not a label.
[[nodiscard]] std::string method_name(RefereeMethod method)
{
    return method == RefereeMethod::ScaledAugmentation
        ? "scaled-augmentation" : "fixed-compatibility";
}

/// "basket" or "rainbow", for paths and sidecars.
[[nodiscard]] std::string option_name(bool rainbow)
{
    return rainbow ? "rainbow" : "basket";
}

/// Where a promoted baseline for this grid and option lives.
[[nodiscard]] std::string referee_path(
    const std::string& directory, int n, bool rainbow)
{
    return (
        std::filesystem::path(directory)
        / ("referee_n" + std::to_string(n) + "_"
           + option_name(rainbow) + ".bin")).string();
}

/// Resolve --verify, which accepts either a state file or a directory holding
/// the canonical one. Empty when no verification was requested.
[[nodiscard]] std::string verification_path(
    const Args& args, bool rainbow)
{
    if (args.verify.empty()) return {};
    if (std::filesystem::is_directory(args.verify))
        return referee_path(args.verify, args.n, rainbow);
    return args.verify;
}

/// Where this run may write.
///
/// Only the promoted method writes to the baseline root. Everything else goes
/// under a method-named subdirectory, which is the mechanism that stops a
/// research arm from overwriting a frozen result.
[[nodiscard]] std::string output_path(
    const Args& args, bool rainbow)
{
    if (args.method == RefereeMethod::FixedCompatibility && !rainbow)
        return (
            std::filesystem::path(args.referee_dir)
            / "compatibility"
            / "fixed-compatibility"
            / ("referee_n" + std::to_string(args.n)
               + "_basket.bin")).string();
    return referee_path(args.referee_dir, args.n, rainbow);
}

/// Convert Eigen's column-major sparse matrix to row-major CSR. cuSPARSE reads
/// rows, and doing the transpose here keeps it out of the timed action.
[[nodiscard]] HostCsr make_host_csr(const SpMat& matrix)
{
    RowSpMat row_major = matrix;
    row_major.makeCompressed();
    if (row_major.rows() > std::numeric_limits<int>::max()
        || row_major.nonZeros() > std::numeric_limits<int>::max())
        throw std::overflow_error("CSR dimensions exceed 32-bit cuSPARSE indices");

    HostCsr csr;
    csr.rows = static_cast<int>(row_major.rows());
    csr.nnz = static_cast<int>(row_major.nonZeros());
    csr.row_offsets.assign(
        row_major.outerIndexPtr(),
        row_major.outerIndexPtr() + row_major.outerSize() + 1);
    csr.column_indices.assign(
        row_major.innerIndexPtr(),
        row_major.innerIndexPtr() + row_major.nonZeros());
    csr.values.assign(
        row_major.valuePtr(),
        row_major.valuePtr() + row_major.nonZeros());
    return csr;
}

/// Seed one Taylor substep: the running sum and the first term both start at
/// the incoming state.
__global__ void initialize_substep(
    const double* state, double* previous, double* sum, int rows)
{
    for (int row = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows;
         row += static_cast<int>(gridDim.x) * blockDim.x) {
        const double value = state[row];
        previous[row] = value;
        sum[row] = value;
    }
}

/// Add the new Taylor term into the sum and reduce the three infinity norms the
/// stopping test needs, in one pass over the vector.
///
/// Fused deliberately: the term, the previous term and the running sum are all
/// resident in this loop, so computing their maxima separately would cost three
/// more full passes over a vector far larger than any cache. Each block emits
/// three partial maxima; finish_maxima closes them.
__global__ void accumulate_and_partial_max(
    const double* previous, const double* current, double* sum,
    int rows, double* partial)
{
    __shared__ double previous_max[kVectorThreads];
    __shared__ double current_max[kVectorThreads];
    __shared__ double sum_max[kVectorThreads];

    double local_previous = 0.0;
    double local_current = 0.0;
    double local_sum = 0.0;
    for (int row = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows;
         row += static_cast<int>(gridDim.x) * blockDim.x) {
        const double next_sum = sum[row] + current[row];
        sum[row] = next_sum;
        local_previous = fmax(local_previous, fabs(previous[row]));
        local_current = fmax(local_current, fabs(current[row]));
        local_sum = fmax(local_sum, fabs(next_sum));
    }

    previous_max[threadIdx.x] = local_previous;
    current_max[threadIdx.x] = local_current;
    sum_max[threadIdx.x] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            previous_max[threadIdx.x] =
                fmax(previous_max[threadIdx.x],
                     previous_max[threadIdx.x + offset]);
            current_max[threadIdx.x] =
                fmax(current_max[threadIdx.x],
                     current_max[threadIdx.x + offset]);
            sum_max[threadIdx.x] =
                fmax(sum_max[threadIdx.x],
                     sum_max[threadIdx.x + offset]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial[blockIdx.x] = previous_max[0];
        partial[gridDim.x + blockIdx.x] = current_max[0];
        partial[2 * gridDim.x + blockIdx.x] = sum_max[0];
    }
}

/// The same fused step for the scaled method, with the augmented tail weighted
/// by eta.
///
/// The weight is what puts the norms back in original coordinates. Without it
/// the similarity transform would move the stopping decision, and the two
/// methods would no longer be the same algorithm on two representations.
__global__ void accumulate_and_partial_weighted_max(
    const double* previous, const double* current, double* sum,
    int rows, int physical_rows, double eta, double* partial)
{
    __shared__ double previous_max[kVectorThreads];
    __shared__ double current_max[kVectorThreads];
    __shared__ double sum_max[kVectorThreads];

    double local_previous = 0.0;
    double local_current = 0.0;
    double local_sum = 0.0;
    for (int row = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows;
         row += static_cast<int>(gridDim.x) * blockDim.x) {
        const double next_sum = sum[row] + current[row];
        sum[row] = next_sum;
        const double weight = row < physical_rows ? 1.0 : eta;
        local_previous = fmax(
            local_previous, weight * fabs(previous[row]));
        local_current = fmax(
            local_current, weight * fabs(current[row]));
        local_sum = fmax(local_sum, weight * fabs(next_sum));
    }

    previous_max[threadIdx.x] = local_previous;
    current_max[threadIdx.x] = local_current;
    sum_max[threadIdx.x] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            previous_max[threadIdx.x] =
                fmax(previous_max[threadIdx.x],
                     previous_max[threadIdx.x + offset]);
            current_max[threadIdx.x] =
                fmax(current_max[threadIdx.x],
                     current_max[threadIdx.x + offset]);
            sum_max[threadIdx.x] =
                fmax(sum_max[threadIdx.x],
                     sum_max[threadIdx.x + offset]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial[blockIdx.x] = previous_max[0];
        partial[gridDim.x + blockIdx.x] = current_max[0];
        partial[2 * gridDim.x + blockIdx.x] = sum_max[0];
    }
}

/// Reduce the per-block partial maxima to three scalars. A second kernel rather
/// than an atomic, so the result is independent of block completion order and a
/// rerun reproduces it bit for bit.
__global__ void finish_maxima(
    const double* partial, int blocks, double* maxima)
{
    __shared__ double storage[3][kVectorThreads];
    for (int channel = 0; channel < 3; ++channel) {
        double local = 0.0;
        for (int q = static_cast<int>(threadIdx.x); q < blocks;
             q += static_cast<int>(blockDim.x))
            local = fmax(local, partial[channel * blocks + q]);
        storage[channel][threadIdx.x] = local;
    }
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset)
            for (int channel = 0; channel < 3; ++channel)
                storage[channel][threadIdx.x] =
                    fmax(storage[channel][threadIdx.x],
                         storage[channel][threadIdx.x + offset]);
        __syncthreads();
    }

    if (threadIdx.x < 3)
        maxima[threadIdx.x] = storage[threadIdx.x][0];
}

/// Allocate every device buffer and build the cuSPARSE descriptors. Sized once
/// from the CSR, so nothing is allocated inside the timed action.
void allocate_workspace(
    Workspace& workspace, const HostCsr& csr, cusparseHandle_t sparse)
{
    workspace.rows = csr.rows;
    workspace.reduction_blocks = std::min(
        kMaxReductionBlocks,
        (csr.rows + kVectorThreads - 1) / kVectorThreads);

    CUDA_CHECK(cudaMalloc(
        &workspace.row_offsets,
        csr.row_offsets.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(
        &workspace.column_indices,
        csr.column_indices.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(
        &workspace.values,
        csr.values.size() * sizeof(double)));

    const std::size_t vector_bytes =
        static_cast<std::size_t>(csr.rows) * sizeof(double);
    CUDA_CHECK(cudaMalloc(&workspace.initial, vector_bytes));
    CUDA_CHECK(cudaMalloc(&workspace.state_a, vector_bytes));
    CUDA_CHECK(cudaMalloc(&workspace.state_b, vector_bytes));
    CUDA_CHECK(cudaMalloc(&workspace.term_a, vector_bytes));
    CUDA_CHECK(cudaMalloc(&workspace.term_b, vector_bytes));
    CUDA_CHECK(cudaMalloc(
        &workspace.partial_maxima,
        static_cast<std::size_t>(3 * workspace.reduction_blocks)
            * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&workspace.maxima, 3 * sizeof(double)));

    CUSPARSE_CHECK(cusparseCreateCsr(
        &workspace.matrix, csr.rows, csr.rows, csr.nnz,
        workspace.row_offsets, workspace.column_indices, workspace.values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(
        &workspace.term_a_desc, csr.rows, workspace.term_a, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(
        &workspace.term_b_desc, csr.rows, workspace.term_b, CUDA_R_64F));

    const double alpha = 1.0;
    const double beta = 0.0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, workspace.matrix, workspace.term_a_desc,
        &beta, workspace.term_b_desc, CUDA_R_64F,
        CUSPARSE_SPMV_CSR_ALG2, &workspace.spmv_buffer_bytes));
    if (workspace.spmv_buffer_bytes > 0)
        CUDA_CHECK(cudaMalloc(
            &workspace.spmv_buffer, workspace.spmv_buffer_bytes));
}

/// Release everything allocate_workspace took, descriptors before buffers.
void free_workspace(Workspace& workspace)
{
    CUSPARSE_CHECK(cusparseDestroySpMat(workspace.matrix));
    CUSPARSE_CHECK(cusparseDestroyDnVec(workspace.term_a_desc));
    CUSPARSE_CHECK(cusparseDestroyDnVec(workspace.term_b_desc));
    if (workspace.spmv_buffer != nullptr)
        CUDA_CHECK(cudaFree(workspace.spmv_buffer));
    CUDA_CHECK(cudaFree(workspace.row_offsets));
    CUDA_CHECK(cudaFree(workspace.column_indices));
    CUDA_CHECK(cudaFree(workspace.values));
    CUDA_CHECK(cudaFree(workspace.initial));
    CUDA_CHECK(cudaFree(workspace.state_a));
    CUDA_CHECK(cudaFree(workspace.state_b));
    CUDA_CHECK(cudaFree(workspace.term_a));
    CUDA_CHECK(cudaFree(workspace.term_b));
    CUDA_CHECK(cudaFree(workspace.partial_maxima));
    CUDA_CHECK(cudaFree(workspace.maxima));
}

/// Copy the CSR and the initial state to the device. Counted as setup, never
/// as part of the action.
void upload_problem(
    Workspace& workspace, const HostCsr& csr, const Eigen::VectorXd& initial)
{
    CUDA_CHECK(cudaMemcpy(
        workspace.row_offsets, csr.row_offsets.data(),
        csr.row_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        workspace.column_indices, csr.column_indices.data(),
        csr.column_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        workspace.values, csr.values.data(),
        csr.values.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        workspace.initial, initial.data(),
        static_cast<std::size_t>(initial.size()) * sizeof(double),
        cudaMemcpyHostToDevice));
}

/// Let cuSPARSE analyze the matrix once.
///
/// Worth a separate step because it removes a repeated partitioning kernel from
/// every SpMV, and because it must sit outside the timed region: leaving it
/// inside would charge the first repeat for work the others do not do.
void preprocess_spmv(Workspace& workspace, cusparseHandle_t sparse)
{
    const double alpha = 1.0;
    const double beta = 0.0;
    CUSPARSE_CHECK(cusparseSpMV_preprocess(
        sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, workspace.matrix, workspace.term_a_desc,
        &beta, workspace.term_b_desc, CUDA_R_64F,
        CUSPARSE_SPMV_CSR_ALG2, workspace.spmv_buffer));
}

/// Apply exp(t A) to the initial state and return the device buffer holding it.
///
/// Scaling and squaring in the Al-Mohy and Higham sense: the time is split into
/// substeps small enough that ||t A||_1 stays under theta_ref, and each substep
/// sums the Taylor series until the strict stopping inequality fires or the
/// degree cap is reached. A substep that reaches the cap is recorded as
/// unconverged rather than accepted quietly, because that is the one failure a
/// checksum comparison could not reveal.
[[nodiscard]] std::pair<double*, ActionStats> run_action(
    Workspace& workspace, cusparseHandle_t sparse,
    int scaling_steps, double action_scale,
    bool weighted_norms = false, int physical_rows = 0,
    double eta = 1.0)
{
    const std::size_t vector_bytes =
        static_cast<std::size_t>(workspace.rows) * sizeof(double);
    CUDA_CHECK(cudaMemcpy(
        workspace.state_a, workspace.initial, vector_bytes,
        cudaMemcpyDeviceToDevice));

    double* state = workspace.state_a;
    double* sum = workspace.state_b;
    double* previous = workspace.term_a;
    double* current = workspace.term_b;
    cusparseDnVecDescr_t previous_desc = workspace.term_a_desc;
    cusparseDnVecDescr_t current_desc = workspace.term_b_desc;

    ActionStats stats;
    stats.degrees.reserve(static_cast<std::size_t>(scaling_steps));
    for (int substep = 0; substep < scaling_steps; ++substep) {
        initialize_substep<<<
            workspace.reduction_blocks, kVectorThreads>>>(
                state, previous, sum, workspace.rows);
        CUDA_CHECK(cudaGetLastError());

        int degree = kTaylorMax;
        bool converged = false;
        for (int k = 1; k <= kTaylorMax; ++k) {
            const double alpha = action_scale / static_cast<double>(k);
            const double beta = 0.0;
            CUSPARSE_CHECK(cusparseSpMV(
                sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, workspace.matrix, previous_desc,
                &beta, current_desc, CUDA_R_64F,
                CUSPARSE_SPMV_CSR_ALG2, workspace.spmv_buffer));

            if (weighted_norms)
                accumulate_and_partial_weighted_max<<<
                    workspace.reduction_blocks, kVectorThreads>>>(
                        previous, current, sum, workspace.rows,
                        physical_rows, eta, workspace.partial_maxima);
            else
                accumulate_and_partial_max<<<
                    workspace.reduction_blocks, kVectorThreads>>>(
                        previous, current, sum, workspace.rows,
                        workspace.partial_maxima);
            CUDA_CHECK(cudaGetLastError());
            finish_maxima<<<1, kVectorThreads>>>(
                workspace.partial_maxima,
                workspace.reduction_blocks, workspace.maxima);
            CUDA_CHECK(cudaGetLastError());

            double maxima[3]{};
            CUDA_CHECK(cudaMemcpy(
                maxima, workspace.maxima, sizeof(maxima),
                cudaMemcpyDeviceToHost));
            if (!std::isfinite(maxima[0])
                || !std::isfinite(maxima[1])
                || !std::isfinite(maxima[2]))
                throw std::runtime_error(
                    "non-finite Taylor recurrence norm");
            stats.max_previous =
                std::max(stats.max_previous, maxima[0]);
            stats.max_current =
                std::max(stats.max_current, maxima[1]);
            stats.max_sum =
                std::max(stats.max_sum, maxima[2]);

            if (maxima[2] > 0.0
                && maxima[0] + maxima[1] < kToleranceRef * maxima[2]) {
                degree = k;
                converged = true;
                break;
            }
            std::swap(previous, current);
            std::swap(previous_desc, current_desc);
        }
        stats.degrees.push_back(degree);
        stats.spmv_count += degree;
        if (!converged)
            ++stats.unconverged_substeps;
        std::swap(state, sum);
    }
    return {state, stats};
}

/// Compare one device SpMV against the host Eigen product.
///
/// The precondition for everything else. It is checked on the assembled
/// operator, with the boundary rows and the augmented tail reported separately,
/// because those are the parts an augmentation change would break first.
[[nodiscard]] OperatorMetrics check_operator(
    Workspace& workspace, cusparseHandle_t sparse,
    const SpMat& matrix, int spatial_rows, int grid_n)
{
    Eigen::VectorXd input(workspace.rows);
    for (int row = 0; row < workspace.rows; ++row)
        input[row] =
            std::sin(0.013 * static_cast<double>(row + 1))
            + std::cos(0.007 * static_cast<double>(row + 3));
    const Eigen::VectorXd expected = matrix * input;

    CUDA_CHECK(cudaMemcpy(
        workspace.term_a, input.data(),
        static_cast<std::size_t>(workspace.rows) * sizeof(double),
        cudaMemcpyHostToDevice));
    const double alpha = 1.0;
    const double beta = 0.0;
    CUSPARSE_CHECK(cusparseSpMV(
        sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, workspace.matrix, workspace.term_a_desc,
        &beta, workspace.term_b_desc, CUDA_R_64F,
        CUSPARSE_SPMV_CSR_ALG2, workspace.spmv_buffer));
    Eigen::VectorXd measured(workspace.rows);
    CUDA_CHECK(cudaMemcpy(
        measured.data(), workspace.term_b,
        static_cast<std::size_t>(workspace.rows) * sizeof(double),
        cudaMemcpyDeviceToHost));

    OperatorMetrics metrics;
    const Eigen::VectorXd error = measured - expected;
    metrics.full_abs = error.lpNorm<Eigen::Infinity>();
    metrics.full_rel =
        metrics.full_abs
        / std::max(expected.lpNorm<Eigen::Infinity>(),
                   std::numeric_limits<double>::min());
    const int64_t grid_n2 = static_cast<int64_t>(grid_n) * grid_n;
    for (int row = 0; row < spatial_rows; ++row) {
        const int i = row % grid_n;
        const int j = (row / grid_n) % grid_n;
        const int k = static_cast<int>(
            static_cast<int64_t>(row) / grid_n2);
        if (i == 0 || i == grid_n - 1
            || j == 0 || j == grid_n - 1
            || k == 0 || k == grid_n - 1)
            metrics.boundary_abs =
                std::max(metrics.boundary_abs, std::abs(error[row]));
    }
    for (int row = spatial_rows; row < workspace.rows; ++row)
        metrics.augmented_abs =
            std::max(metrics.augmented_abs, std::abs(error[row]));
    metrics.finite = measured.allFinite();
    return metrics;
}

/// Read a state file: an int64 length followed by that many doubles.
[[nodiscard]] Eigen::VectorXd load_vector_file(const std::string& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("cannot open verification file: " + path);
    int64_t size = 0;
    stream.read(reinterpret_cast<char*>(&size), sizeof(size));
    if (!stream || size <= 0)
        throw std::runtime_error("invalid verification file: " + path);
    Eigen::VectorXd vector(size);
    stream.read(
        reinterpret_cast<char*>(vector.data()),
        static_cast<std::streamsize>(size * sizeof(double)));
    if (!stream)
        throw std::runtime_error("truncated verification file: " + path);
    return vector;
}

/// Write a state file in the layout load_vector_file reads.
void save_vector_file(const std::string& path, const Eigen::VectorXd& vector)
{
    std::ofstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("cannot create referee file: " + path);
    const int64_t size = vector.size();
    stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
    stream.write(
        reinterpret_cast<const char*>(vector.data()),
        static_cast<std::streamsize>(size * sizeof(double)));
    if (!stream)
        throw std::runtime_error("failed to write referee file: " + path);
}

/// FNV-1a over a byte range, continuing from an existing hash. Short and
/// exactly reproducible across hosts, which is all a result identity needs.
[[nodiscard]] uint64_t fnv1a_bytes(
    uint64_t hash, const void* data, std::size_t bytes)
{
    const auto* input = static_cast<const unsigned char*>(data);
    for (std::size_t q = 0; q < bytes; ++q) {
        hash ^= input[q];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

/// Identity of a computed state, length folded in first.
[[nodiscard]] uint64_t vector_file_checksum(const Eigen::VectorXd& vector)
{
    uint64_t hash = UINT64_C(14695981039346656037);
    const int64_t size = vector.size();
    hash = fnv1a_bytes(hash, &size, sizeof(size));
    return fnv1a_bytes(
        hash, vector.data(),
        static_cast<std::size_t>(size) * sizeof(double));
}

/// Identity of the assembled operator. Two runs reporting the same result
/// checksum are only comparable if this matches too.
[[nodiscard]] uint64_t csr_checksum(const HostCsr& csr)
{
    uint64_t hash = UINT64_C(14695981039346656037);
    hash = fnv1a_bytes(hash, &csr.rows, sizeof(csr.rows));
    hash = fnv1a_bytes(hash, &csr.nnz, sizeof(csr.nnz));
    hash = fnv1a_bytes(
        hash, csr.row_offsets.data(),
        csr.row_offsets.size() * sizeof(int));
    hash = fnv1a_bytes(
        hash, csr.column_indices.data(),
        csr.column_indices.size() * sizeof(int));
    return fnv1a_bytes(
        hash, csr.values.data(),
        csr.values.size() * sizeof(double));
}

/// Write the .meta sidecar beside a saved state.
///
/// This is what makes a checksum usable as evidence. A bare hash says two files
/// match; these fields say which method, which parity version, which Taylor
/// constants, which device and which timing mode produced them. A checksum
/// without this sidecar cannot serve as an oracle, which is why the packaging
/// gate treats a sidecar-less binary as unverified.
void save_metadata(
    const std::string& binary_path, const Args& args, bool rainbow,
    const HostCsr& csr, double norm_1, double physical_norm_1,
    double forcing_norm_1, double original_forcing_norm_1,
    double original_norm_1, int scaling_steps,
    int original_scaling_steps,
    const ca_referee::AugmentationScaling& augmentation,
    const ActionStats& stats, const std::string& device_name,
    bool contended, const Eigen::VectorXd& vector,
    double tail_error, const ErrorMetrics& verification,
    bool verification_requested, double action_ms)
{
    std::ofstream stream(binary_path + ".meta");
    if (!stream)
        throw std::runtime_error(
            "cannot create referee metadata: " + binary_path + ".meta");
    const auto degree_minmax =
        std::minmax_element(stats.degrees.begin(), stats.degrees.end());
    const double degree_mean =
        std::accumulate(stats.degrees.begin(), stats.degrees.end(), 0.0)
        / static_cast<double>(stats.degrees.size());
    const bool scaled_method =
        args.method == RefereeMethod::ScaledAugmentation;
    const bool baseline = scaled_method || rainbow;
    stream << std::setprecision(17);
    stream << "generator=ca-referee-gpu\n";
    stream << "method=" << method_name(args.method) << "\n";
    stream << "referee_role="
           << (baseline ? "baseline" : "compatibility") << "\n";
    stream << "canonical_eligible=" << (baseline ? "true" : "false") << "\n";
    stream << "canonical_output=" << (baseline ? "true" : "false") << "\n";
    stream << "output_admissible=true\n";
    stream << "result_finite=true\n";
    stream << "parity_version=2\n";
    stream << "option=" << option_name(rainbow) << "\n";
    stream << "n=" << args.n << "\n";
    stream << "rows=" << csr.rows << "\n";
    stream << "nnz=" << csr.nnz << "\n";
    stream << "matrix_norm_1=" << norm_1 << "\n";
    stream << "physical_norm_1=" << physical_norm_1 << "\n";
    stream << "forcing_norm_1=" << forcing_norm_1 << "\n";
    stream << "original_forcing_norm_1="
           << original_forcing_norm_1 << "\n";
    stream << "original_matrix_norm_1=" << original_norm_1 << "\n";
    stream << "m_max=" << kTaylorMax << "\n";
    stream << "theta_ref=" << kThetaRef << "\n";
    stream << "tol_ref=" << kToleranceRef << "\n";
    stream << "scaling_steps=" << scaling_steps << "\n";
    stream << "original_scaling_steps="
           << original_scaling_steps << "\n";
    if (scaled_method) {
        stream << "scaling_exponent="
               << augmentation.exponent << "\n";
        stream << "eta=" << augmentation.eta << "\n";
        stream << "eta_inverse=" << augmentation.eta_inverse << "\n";
        stream << "unshifted_forcing_norm_1="
               << augmentation.forcing_norm_1 << "\n";
        stream << "scaled_unshifted_forcing_norm_1="
               << augmentation.scaled_forcing_norm_1 << "\n";
        stream << "stopping_norm=original-coordinate-weighted-infinity\n";
    } else {
        stream << "stopping_norm=infinity\n";
    }
    stream << "taylor_degree_min=" << *degree_minmax.first << "\n";
    stream << "taylor_degree_mean=" << degree_mean << "\n";
    stream << "taylor_degree_max=" << *degree_minmax.second << "\n";
    stream << "spmv_count=" << stats.spmv_count << "\n";
    stream << "unconverged_substeps="
           << stats.unconverged_substeps << "\n";
    stream << "max_stopping_previous=" << stats.max_previous << "\n";
    stream << "max_stopping_current=" << stats.max_current << "\n";
    stream << "max_stopping_sum=" << stats.max_sum << "\n";
    uint64_t degree_hash = UINT64_C(14695981039346656037);
    degree_hash = fnv1a_bytes(
        degree_hash, stats.degrees.data(),
        stats.degrees.size() * sizeof(int));
    stream << "degree_checksum_algorithm=fnv1a64\n";
    stream << "degree_checksum=" << std::hex << std::setw(16)
           << std::setfill('0') << degree_hash << std::dec << "\n";
    if (stats.degrees.size() <= 10000) {
        stream << "taylor_degrees=";
        for (std::size_t q = 0; q < stats.degrees.size(); ++q) {
            if (q != 0) stream << ",";
            stream << stats.degrees[q];
        }
        stream << "\n";
    }
    stream << "precision=fp64\n";
    stream << "spmv_algorithm=CUSPARSE_SPMV_CSR_ALG2\n";
    stream << "spmv_preprocess=1\n";
    stream << "git_revision=" << CAKSM_GIT_REVISION << "\n";
    stream << "device=" << device_name << "\n";
    stream << "contended=" << (contended ? 1 : 0) << "\n";
    stream << "timing_mode="
           << (args.profiled ? "profiled" : "normal") << "\n";
    stream << "action_ms=" << action_ms << "\n";
    stream << "repeats=" << args.repeats << "\n";
    stream << "restored_tail_error=" << tail_error << "\n";
    stream << "verification_requested="
           << (verification_requested ? 1 : 0) << "\n";
    if (verification_requested) {
        stream << "verification_full_abs="
               << verification.full_abs << "\n";
        stream << "verification_full_rel="
               << verification.full_rel << "\n";
        stream << "verification_cube_abs="
               << verification.cube_abs << "\n";
        stream << "verification_price_abs="
               << verification.price_abs << "\n";
        stream << "verification_passed="
               << (verification.passed ? 1 : 0) << "\n";
    }
    stream << "compile_optimization=-O3\n";
    stream << "fast_math=false\n";
    stream << "cuda_toolkit=" << __CUDACC_VER_MAJOR__
           << "." << __CUDACC_VER_MINOR__ << "\n";
    int runtime_version = 0;
    int driver_version = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    stream << "cuda_runtime=" << runtime_version << "\n";
    stream << "cuda_driver=" << driver_version << "\n";
    stream << "csr_checksum_algorithm=fnv1a64\n";
    stream << "csr_checksum=" << std::hex << std::setw(16)
           << std::setfill('0') << csr_checksum(csr) << "\n";
    stream << "result_checksum_algorithm=fnv1a64\n";
    stream << "result_checksum=" << std::hex << std::setw(16)
           << std::setfill('0') << vector_file_checksum(vector) << "\n";
}

/// Compare a generated state against a stored one and decide whether it passes.
/// Three measures because they fail differently: a global drift, a local defect
/// near the spot that a field norm would average away, and the reported price.
[[nodiscard]] ErrorMetrics compare_reference(
    const Eigen::VectorXd& measured, const Eigen::VectorXd& reference,
    const PDESystem& system, const Model& model)
{
    if (measured.size() != reference.size())
        throw std::runtime_error(
            "verification vector size does not match generated field");
    ErrorMetrics metrics;
    const Eigen::VectorXd error = measured - reference;
    metrics.full_abs = error.lpNorm<Eigen::Infinity>();
    metrics.full_rel =
        metrics.full_abs
        / std::max(reference.lpNorm<Eigen::Infinity>(),
                   std::numeric_limits<double>::min());
    metrics.cube_abs =
        (extract_cube(measured, system.grid, model.spot)
         - extract_cube(reference, system.grid, model.spot))
            .lpNorm<Eigen::Infinity>();
    metrics.price_abs = std::abs(
        extract_price(measured, system.grid, model.spot)
        - extract_price(reference, system.grid, model.spot));
    metrics.passed =
        metrics.full_rel <= 1.0e-12
        && metrics.cube_abs <= 1.0e-10
        && metrics.price_abs <= 1.0e-10;
    return metrics;
}

/// Referee one option end to end, and return whether it passed.
///
/// Assemble, upload, gate the operator, run the action, restore original
/// coordinates, verify against any stored reference, and write only if every
/// gate held. The write gate is shared with the CPU audit so the two cannot
/// drift on what counts as admissible.
[[nodiscard]] bool run_option(
    const Args& args, const Model& model, bool rainbow,
    const DeviceContention& contention, const std::string& device_name)
{
    // Load the state to compare against, if one was named.
    const std::string verify_path = verification_path(args, rainbow);
    Eigen::VectorXd verification;
    if (!verify_path.empty()) verification = load_vector_file(verify_path);

    // Assemble the operator this method actually exponentiates. Rainbow needs
    // no augmentation; Basket gets the N+3 augmented form, scaled or not.
    const auto setup_begin = Clock::now();
    const PDESystem system = build_pde_system(
        args.n, model.strike, model.rate, model.expiry,
        model.sigma, model.rho, model.weight, model.spot,
        model.alpha, rainbow);
    const bool scaled_method =
        args.method == RefereeMethod::ScaledAugmentation;
    ca_referee::AugmentationScaling augmentation;
    if (scaled_method)
        augmentation =
            ca_referee::make_augmentation_scaling(system.B);

    SpMat matrix;
    Eigen::VectorXd initial;
    if (rainbow) {
        matrix = system.A;
        initial = system.u0;
    } else if (scaled_method) {
        matrix = ca_referee::build_scaled_A_tilde(
            system.A, system.B, system.N, augmentation.eta);
        initial.resize(system.N + 3);
        initial.head(system.N) = system.u0;
        initial.tail(3) =
            augmentation.eta_inverse * make_s_vec(0.0);
    } else {
        matrix = build_A_tilde(system.A, system.B, system.N);
        initial.resize(system.N + 3);
        initial.head(system.N) = system.u0;
        initial.tail(3) = make_s_vec(0.0);
    }
    // Choose the substep count from the operator norm, and record what the
    // unscaled representation would have needed, so the saving is measured
    // rather than asserted.
    const double physical_norm_1 = sparse_1norm(system.A);
    const double original_forcing_norm_1 =
        rainbow
            ? 0.0
            : ca_referee::augmented_forcing_1norm(system.B);
    const double forcing_norm_1 =
        scaled_method
            ? ca_referee::augmented_forcing_1norm(
                  system.B, augmentation.eta)
            : original_forcing_norm_1;
    const double norm_1 = sparse_1norm(matrix);
    const int scaling_steps =
        ca_referee::select_scaling_steps(model.expiry, norm_1);
    const double original_norm_1 =
        scaled_method
            ? std::max(physical_norm_1, original_forcing_norm_1)
            : norm_1;
    const int original_scaling_steps =
        ca_referee::select_scaling_steps(
            model.expiry, original_norm_1);
    const double action_scale =
        model.expiry / static_cast<double>(scaling_steps);
    const HostCsr csr = make_host_csr(matrix);
    const double host_setup_ms =
        Milliseconds(Clock::now() - setup_begin).count();

    // Bring up cuSPARSE, allocate, upload, and analyze the matrix. Each phase
    // is timed separately because none of them belongs in the action.
    cusparseHandle_t sparse = nullptr;
    CUSPARSE_CHECK(cusparseCreate(&sparse));
    int cusparse_version = 0;
    CUSPARSE_CHECK(cusparseGetVersion(sparse, &cusparse_version));

    std::size_t free_before = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total_bytes));
    const auto device_setup_begin = Clock::now();
    Workspace workspace;
    allocate_workspace(workspace, csr, sparse);
    CUDA_CHECK(cudaDeviceSynchronize());
    const double device_setup_ms =
        Milliseconds(Clock::now() - device_setup_begin).count();

    const auto transfer_begin = Clock::now();
    upload_problem(workspace, csr, initial);
    CUDA_CHECK(cudaDeviceSynchronize());
    const double h2d_ms =
        Milliseconds(Clock::now() - transfer_begin).count();

    const auto preprocess_begin = Clock::now();
    preprocess_spmv(workspace, sparse);
    CUDA_CHECK(cudaDeviceSynchronize());
    const double preprocess_ms =
        Milliseconds(Clock::now() - preprocess_begin).count();

    std::size_t free_after = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_after, &total_bytes));
    const std::size_t allocated_bytes =
        free_before > free_after ? free_before - free_after : 0;

    // Gate the operator before exponentiating it. A device product that
    // disagrees with the host makes every number after this point meaningless,
    // so this throws rather than recording a failure.
    const OperatorMetrics operator_metrics =
        check_operator(workspace, sparse, matrix, system.N, args.n);
    const bool operator_passed =
        operator_metrics.finite
        && (operator_metrics.full_rel <= 5.0e-13
            || operator_metrics.full_abs <= 5.0e-13);
    if (!operator_passed)
        throw std::runtime_error(
            "GPU CSR operator product does not match the assembled operator");

    // Restore the initial state, which check_operator overwrote, then run the
    // action once to warm up and again for each timed repeat. A repeat whose
    // stopping degrees differ is a nondeterministic build, and stops the run.
    CUDA_CHECK(cudaMemcpy(
        workspace.initial, initial.data(),
        static_cast<std::size_t>(initial.size()) * sizeof(double),
        cudaMemcpyHostToDevice));
    const auto execute_action = [&]() {
        return run_action(
            workspace, sparse, scaling_steps, action_scale,
            scaled_method, system.N, augmentation.eta);
    };
    const auto warmup =
        execute_action();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> action_times;
    action_times.reserve(static_cast<std::size_t>(args.repeats));
    ActionStats selected_stats;
    double* result_pointer = nullptr;
    for (int repeat = 0; repeat < args.repeats; ++repeat) {
        const auto action_begin = Clock::now();
        auto action = execute_action();
        CUDA_CHECK(cudaDeviceSynchronize());
        action_times.push_back(
            Milliseconds(Clock::now() - action_begin).count());
        if (action.second.degrees != warmup.second.degrees)
            throw std::runtime_error(
                "Taylor stopping degrees changed between identical repeats");
        result_pointer = action.first;
        selected_stats = std::move(action.second);
    }

    // Bring the state back and split it: the physical field is the result, and
    // the augmented tail is a known closed form, so its error measures whether
    // the augmentation was undone exactly.
    Eigen::VectorXd full_result(initial.size());
    const auto d2h_begin = Clock::now();
    CUDA_CHECK(cudaMemcpy(
        full_result.data(), result_pointer,
        static_cast<std::size_t>(full_result.size()) * sizeof(double),
        cudaMemcpyDeviceToHost));
    const double d2h_ms =
        Milliseconds(Clock::now() - d2h_begin).count();

    const Eigen::VectorXd field = full_result.head(system.N);
    const bool result_finite = full_result.allFinite();
    Eigen::Vector3d tail = Eigen::Vector3d::Zero();
    if (!rainbow)
        tail =
            (scaled_method ? augmentation.eta : 1.0)
            * full_result.tail(3);
    const double tail_error =
        rainbow
            ? 0.0
            : (tail - make_s_vec(model.expiry)).lpNorm<Eigen::Infinity>();

    // Compare against the stored state.
    ErrorMetrics verification_metrics;
    bool verification_requested = !verify_path.empty();
    if (verification_requested)
        verification_metrics =
            compare_reference(field, verification, system, model);

    // Reduce the repeats, degrees and footprint to the summary the printout
    // and the sidecar both report.
    const auto action_minmax =
        std::minmax_element(action_times.begin(), action_times.end());
    const double action_mean =
        std::accumulate(action_times.begin(), action_times.end(), 0.0)
        / static_cast<double>(action_times.size());
    const auto degree_minmax =
        std::minmax_element(
            selected_stats.degrees.begin(), selected_stats.degrees.end());
    const double degree_mean =
        std::accumulate(
            selected_stats.degrees.begin(),
            selected_stats.degrees.end(), 0.0)
        / static_cast<double>(selected_stats.degrees.size());
    const std::size_t csr_bytes =
        csr.row_offsets.size() * sizeof(int)
        + csr.column_indices.size() * sizeof(int)
        + csr.values.size() * sizeof(double);
    const std::size_t vector_bytes =
        5 * static_cast<std::size_t>(csr.rows) * sizeof(double);
    const std::size_t working_set =
        csr_bytes + vector_bytes + workspace.spmv_buffer_bytes
        + static_cast<std::size_t>(3 * workspace.reduction_blocks + 3)
            * sizeof(double);
    const double price =
        extract_price(field, system.grid, model.spot);

    // Report everything, admissible or not, so a failed referee is still
    // diagnosable from its own output.
    std::printf("GPU full-accuracy referee\n");
    const bool baseline = scaled_method || rainbow;
    std::printf(
        "  method=%s | role=%s | canonical_eligible=%s\n",
        method_name(args.method).c_str(),
        baseline ? "baseline" : "compatibility",
        baseline ? "true" : "false");
    std::printf(
        "  option=%s | n=%d | rows=%d | nnz=%d\n",
        option_name(rainbow).c_str(), args.n, csr.rows, csr.nnz);
    std::printf(
        "  CSR=%.3f MiB | device working set=%.3f MiB | allocated peak=%.3f MiB\n",
        static_cast<double>(csr_bytes) / (1024.0 * 1024.0),
        static_cast<double>(working_set) / (1024.0 * 1024.0),
        static_cast<double>(allocated_bytes) / (1024.0 * 1024.0));
    std::printf(
        "  ||A||_1=%.17e | m_max=%d | theta_ref=%.17g | tol_ref=%.17g | s=%d\n",
        norm_1, kTaylorMax, kThetaRef, kToleranceRef, scaling_steps);
    std::printf(
        "  norm components: physical=%.17e | forcing=%.17e | dominant=%s\n",
        physical_norm_1, forcing_norm_1,
        forcing_norm_1 > physical_norm_1 ? "forcing" : "physical");
    if (scaled_method)
        std::printf(
            "  eta=2^-%d=%.17e | original forcing=%.17e | original norm=%.17e s=%d\n",
            augmentation.exponent, augmentation.eta,
            original_forcing_norm_1, original_norm_1,
            original_scaling_steps);
    std::printf(
        "  Taylor degree: min=%d mean=%.2f max=%d | SpMV count=%lld | unconverged=%d\n",
        *degree_minmax.first, degree_mean, *degree_minmax.second,
        static_cast<long long>(selected_stats.spmv_count),
        selected_stats.unconverged_substeps);
    std::printf(
        "  stopping norm maxima: previous=%.6e current=%.6e sum=%.6e\n",
        selected_stats.max_previous, selected_stats.max_current,
        selected_stats.max_sum);
    std::printf(
        "  CSR operator check: full abs=%.6e rel=%.6e | boundary=%.6e | augmented=%.6e | PASS\n",
        operator_metrics.full_abs, operator_metrics.full_rel,
        operator_metrics.boundary_abs, operator_metrics.augmented_abs);
    std::printf(
        "  cuSPARSE=%d | algorithm=CUSPARSE_SPMV_CSR_ALG2 | preprocess=enabled\n",
        cusparse_version);
    std::printf(
        "  setup: host=%.3f ms device=%.3f ms | H2D=%.3f ms | preprocess=%.3f ms | D2H=%.3f ms\n",
        host_setup_ms, device_setup_ms, h2d_ms, preprocess_ms, d2h_ms);
    std::printf(
        "  action: min=%.3f ms mean=%.3f ms max=%.3f ms (%d repeats)\n",
        *action_minmax.first, action_mean, *action_minmax.second,
        args.repeats);
    std::printf(
        "  total representative=%.3f ms | price=%.10f\n",
        host_setup_ms + device_setup_ms + h2d_ms
            + preprocess_ms + action_mean + d2h_ms,
        price);
    if (!rainbow)
        std::printf(
            "  augmented tail: [%.17e %.17e %.17e] | error=%.6e\n",
            tail[0], tail[1], tail[2], tail_error);
    if (verification_requested) {
        std::printf(
            "  verification: full abs=%.6e rel=%.6e | cube=%.6e | price=%.6e | %s\n",
            verification_metrics.full_abs, verification_metrics.full_rel,
            verification_metrics.cube_abs, verification_metrics.price_abs,
            verification_metrics.passed ? "PASS" : "FAIL");
    } else {
        std::printf("  verification: not requested\n");
    }

    // The Taylor substep counter is method-independent, so both methods answer to it.
    // A canonical vector must not be writable past a silent non-convergence, whichever
    // augmentation produced it.
    const bool passed = ca_referee::referee_output_admissible(
        operator_passed, result_finite, selected_stats.unconverged_substeps,
        tail_error, verification_requested, verification_metrics.passed);
    if (passed) {
        const std::string output = output_path(args, rainbow);
        std::filesystem::create_directories(
            std::filesystem::path(output).parent_path());
        if (!verify_path.empty()
            && std::filesystem::weakly_canonical(output)
               == std::filesystem::weakly_canonical(verify_path))
            throw std::runtime_error(
                "output path would overwrite the verification referee");
        save_vector_file(output, field);
        save_metadata(
            output, args, rainbow, csr, norm_1,
            physical_norm_1, forcing_norm_1,
            original_forcing_norm_1, original_norm_1,
            scaling_steps, original_scaling_steps,
            augmentation, selected_stats, device_name,
            contention.contended, field, tail_error,
            verification_metrics, verification_requested,
            action_mean);
        std::printf(
            "  saved %s: %s\n",
            baseline ? "baseline" : "compatibility",
            output.c_str());
        std::printf(
            "  result checksum (fnv1a64): %016llx\n",
            static_cast<unsigned long long>(vector_file_checksum(field)));
        std::printf(
            "REFEREE_RESULT method=%s option=%s n=%d"
            " checksum=%016llx admissible=1\n",
            method_name(args.method).c_str(), option_name(rainbow).c_str(),
            args.n,
            static_cast<unsigned long long>(vector_file_checksum(field)));
        std::printf("  metadata: %s.meta\n", output.c_str());
    } else {
        std::printf(
            "  output withheld: referee gate failed"
            " | result_finite=%d | unconverged=%d"
            " | tail_error=%.6e | verification=%s\n",
            result_finite ? 1 : 0,
            selected_stats.unconverged_substeps,
            tail_error,
            verification_requested
                ? (verification_metrics.passed ? "PASS" : "FAIL")
                : "not-requested");
    }
    std::printf("\n");

    free_workspace(workspace);
    CUSPARSE_CHECK(cusparseDestroy(sparse));
    return passed;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const Args args = parse_args(argc, argv);
        const Model model;
        CUDA_CHECK(cudaSetDevice(args.device));

        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, args.device));
        const DeviceContention contention = check_device_contention();
        report_toolkit();
        report_contention(contention);
        std::printf(
            "GPU referee device: %d %s | git=%s\n\n",
            args.device, properties.name, CAKSM_GIT_REVISION);

        bool passed = true;
        if (args.option == OptionMode::Basket
            || args.option == OptionMode::Both)
            passed =
                run_option(
                    args, model, false, contention, properties.name)
                && passed;
        if (args.option == OptionMode::Rainbow
            || args.option == OptionMode::Both)
            passed =
                run_option(
                    args, model, true, contention, properties.name)
                && passed;
        return passed ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Error: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
