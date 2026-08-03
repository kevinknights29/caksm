/**
 * @file gpu_ca_arnoldi.cuh
 * @brief Small kernels used by GPU CA-Arnoldi assembly and phi evaluation.
 *
 * @author Kevin Knights
 * @date 2026-07-26
 */
#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

inline constexpr int kGpuCaMaxM = 24;
inline constexpr int kGpuCaMaxS = 9;

/// dst += src over an rows x cols column-major block. The two-pass block Gram-Schmidt
/// accumulation: the second projection's coefficients are added to the first pass's, so the
/// assembled H reflects both passes rather than only the last.
__global__ void ca_add_matrix(double* dst, const double* src, int rows, int cols, int ld)
{
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int col = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (row < rows && col < cols) dst[row + col * ld] += src[row + col * ld];
}

/// R = R2 R1 for two s x s upper triangular factors, the CholQR2 combination: the first pass
/// gives B = Q1 R1 and the second Q1 = Q R2, so the block's factor is the product. One thread
/// per entry, s <= 9, and only the upper triangle is touched because the product of two upper
/// triangular matrices is upper triangular.
__global__ void ca_combine_upper(const double* R2, const double* R1, double* R, int s)
{
    const int row = static_cast<int>(threadIdx.x);
    const int col = static_cast<int>(threadIdx.y);
    if (row >= s || col >= s) return;
    double sum = 0.0;
    if (row <= col)
        for (int k = row; k <= col; ++k)
            sum += R2[row + k * s] * R1[k + col * s];
    R[row + col * s] = sum;
}

/// Pull a packed cols x cols upper triangular R out of an lda-strided factorization. cuSOLVER's
/// geqrf leaves R in the upper triangle of the tall input and the Householder vectors below it,
/// so the strictly lower part is explicitly zeroed rather than copied.
__global__ void ca_extract_upper(
    const double* A, int lda, double* R, int rows, int cols)
{
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int col = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (row < cols && col < cols)
        R[row + col * cols] =
            row <= col && row < rows ? A[row + col * lda] : 0.0;
}

/// Sign of each diagonal entry of R. Householder QR leaves those signs arbitrary while Cholesky
/// forces them positive, so the three sign kernels below normalize the Householder arm to the
/// unique factorization CholQR2 produces. Without that the two orthogonalization arms would
/// differ by a column sign and could not be compared state for state.
__global__ void ca_qr_signs(const double* R, int s, double* signs)
{
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (col < s) signs[col] = R[col + col * s] < 0.0 ? -1.0 : 1.0;
}

/// Scale Q's columns by those signs. Paired with ca_apply_r_signs, which scales R's rows, so the
/// product Q R is unchanged.
__global__ void ca_apply_q_signs(
    double* Q, int64_t rows, int cols, const double* signs)
{
    const int64_t row =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int col = static_cast<int>(blockIdx.y);
    if (row < rows && col < cols) Q[row + static_cast<int64_t>(col) * rows] *= signs[col];
}

/// Scale R's rows by the same signs, leaving R with a positive diagonal.
__global__ void ca_apply_r_signs(double* R, int s, const double* signs)
{
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int col = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (row < s && col < s && row <= col) R[row + col * s] *= signs[row];
}

/// Read entry (row, col) of the block's coefficient matrix, which lives in two pieces: the
/// projection C against the filled columns of the existing basis on top, and the block's own
/// triangular factor R below. Together they are the change of basis from the raw powers to the
/// orthonormal columns, and reading them through one accessor keeps the H assembly from having
/// to know where the seam falls.
__device__ __forceinline__ double ca_block_factor(
    const double* C, int ldc, const double* R, int filled, int block,
    int row, int col)
{
    if (row < filled) return C[row + col * ldc];
    const int local_row = row - filled;
    if (local_row < 0 || local_row > col || local_row >= block) return 0.0;
    return R[local_row + col * block];
}

/**
 * @brief The columns of the Hessenberg H that one orthogonalized block contributes.
 *
 * This is what makes the CA arm equivalent to standard Arnoldi rather than merely similar. The
 * basis was built with no reductions, so H is never accumulated from dot products; it is
 * recovered from the polynomial recurrence the block satisfies and the coefficients that
 * orthogonalized it. Column j is obtained by expressing the recurrence's next power in the
 * accepted basis, subtracting the columns of H already known, and dividing by the diagonal
 * coefficient, one column at a time because each depends on its predecessors.
 *
 * The chebyshev flag selects which recurrence the right-hand side comes from: the next power for
 * the monomial basis, or M T_j = center T_j + half_width (T_{j+1} + T_{j-1})/2 for Chebyshev,
 * unhalved at j = 0 where M T_0 = center T_0 + half_width T_1. One row per thread in a single
 * block, synchronizing between columns.
 */
__global__ void ca_assemble_block(double* H, int ldh, int m,
                                  const double* C, int ldc, const double* R,
                                  int filled, int block, int chebyshev,
                                  double center, double half_width)
{
    const int target = m + 1;
    const int row = static_cast<int>(threadIdx.x);
    if (blockIdx.x != 0) return;
    const bool active = row < target;

    if (active && filled > 0 && filled - 1 < m) {
        const int col = filled - 1;
        H[row + col * ldh] =
            row <= filled ? ca_block_factor(C, ldc, R, filled, block, row, 0) : 0.0;
    }
    __syncthreads();

    for (int j = 0; j + 1 < block; ++j) {
        const int col = filled + j;
        if (col >= m) break;
        if (active) {
            const double diagonal =
                ca_block_factor(C, ldc, R, filled, block, col, j);
            double rhs = 0.0;
            if (chebyshev != 0) {
                rhs =
                    center * ca_block_factor(C, ldc, R, filled, block, row, j);
                if (j == 0) {
                    rhs +=
                        half_width
                        * ca_block_factor(C, ldc, R, filled, block, row, j + 1);
                } else {
                    rhs +=
                        0.5 * half_width
                        * (ca_block_factor(C, ldc, R, filled, block, row, j + 1)
                           + ca_block_factor(C, ldc, R, filled, block, row, j - 1));
                }
            } else {
                rhs = ca_block_factor(C, ldc, R, filled, block, row, j + 1);
            }
            for (int k = 0; k < col; ++k)
                rhs -= H[row + k * ldh]
                     * ca_block_factor(C, ldc, R, filled, block, k, j);
            H[row + col * ldh] = rhs / diagonal;
        }
        __syncthreads();
    }
}

/**
 * @brief kappa_2(B) from its Gram, by a serial two-sided Jacobi sweep. The CholQR certificate.
 *
 * G = B^T B is symmetric positive definite, so Jacobi rotations drive it to diagonal form and
 * its eigenvalues are the squared singular values of B; kappa_2(B) is then sqrt(hi / lo), or
 * infinity if the smallest has been lost to rounding. The caller admits the block only while
 * kappa(B) <= u^(-1/2), inside which CholQR2 recovers orthogonality to O(u); a non-positive
 * Cholesky pivot is a much later and cruder symptom.
 *
 * Launched <<<1, 1>>> on purpose: s <= 9, so the matrix is at most 81 doubles and any parallel
 * form would cost more in launch and synchronization than the sweep itself. The expensive part
 * is not this kernel but the blocking read of kappa that follows it, which is counted in the
 * host-synchronization term of the communication account.
 */
__global__ void ca_gram_condition(const double* G, int s, double* kappa)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    double a[kGpuCaMaxS * kGpuCaMaxS];
    for (int col = 0; col < s; ++col)
        for (int row = 0; row < s; ++row)
            a[row + col * s] = G[row + col * s];

    for (int sweep = 0; sweep < 64; ++sweep) {
        double largest = 0.0;
        for (int q = 1; q < s; ++q) {
            for (int p = 0; p < q; ++p) {
                const double apq = a[p + q * s];
                largest = fmax(largest, fabs(apq));
                const double app = a[p + p * s];
                const double aqq = a[q + q * s];
                if (fabs(apq) <= 2.0e-16 * sqrt(fabs(app * aqq))) continue;

                const double tau = (aqq - app) / (2.0 * apq);
                const double t = copysign(1.0, tau)
                               / (fabs(tau) + sqrt(1.0 + tau * tau));
                const double c = 1.0 / sqrt(1.0 + t * t);
                const double sn = t * c;

                for (int k = 0; k < s; ++k) {
                    if (k == p || k == q) continue;
                    const double akp = a[k + p * s];
                    const double akq = a[k + q * s];
                    const double np = c * akp - sn * akq;
                    const double nq = sn * akp + c * akq;
                    a[k + p * s] = np;
                    a[p + k * s] = np;
                    a[k + q * s] = nq;
                    a[q + k * s] = nq;
                }
                a[p + p * s] = c * c * app - 2.0 * c * sn * apq + sn * sn * aqq;
                a[q + q * s] = sn * sn * app + 2.0 * c * sn * apq + c * c * aqq;
                a[p + q * s] = 0.0;
                a[q + p * s] = 0.0;
            }
        }
        if (largest <= 1.0e-15) break;
    }

    double lo = a[0];
    double hi = a[0];
    for (int i = 1; i < s; ++i) {
        lo = fmin(lo, a[i + i * s]);
        hi = fmax(hi, a[i + i * s]);
    }
    *kappa = lo > 0.0 ? sqrt(hi / lo) : HUGE_VAL;
}

struct GpuCaCandidateResult {
    double residual;   ///< estimate at the accepted m, or at last_m if none converged
    int selected_m;    ///< 0 when no candidate met the tolerance
};

/**
 * @brief exp(H) e_1 for the small m x m Hessenberg, entirely in shared memory.
 *
 * Scaling and squaring with a fixed 24-term Taylor series: the squaring count is chosen so that
 * ||H||_1 * 2^-squarings <= 0.5, where 24 terms are comfortably converged, and the result is
 * squared back up. m <= kGpuCaMaxM, so the whole thing fits in shared memory and never touches
 * device memory, which is why the replicated exponential costs no communication even though
 * every rank computes it.
 */
template <int MAX_M>
__device__ void ca_expm_action_shared(
    const double* H, int ldh, int m, double* f,
    double* A, double* E, double* term, double* work, int* squarings)
{
    const int tid = static_cast<int>(threadIdx.x);
    const int elements = m * m;
    if (tid == 0) {
        double norm1 = 0.0;
        for (int col = 0; col < m; ++col) {
            double sum = 0.0;
            for (int row = 0; row < m; ++row)
                sum += fabs(H[row + col * ldh]);
            norm1 = fmax(norm1, sum);
        }
        *squarings = norm1 > 0.5
            ? static_cast<int>(ceil(log2(norm1 / 0.5))) : 0;
    }
    __syncthreads();

    const double scale = ldexp(1.0, -*squarings);
    for (int q = tid; q < elements; q += static_cast<int>(blockDim.x)) {
        const int row = q % m;
        const int col = q / m;
        const int dst = row + col * MAX_M;
        A[dst] = scale * H[row + col * ldh];
        E[dst] = row == col ? 1.0 : 0.0;
        term[dst] = E[dst];
    }
    __syncthreads();

    constexpr int terms_used = 24;
    for (int order = 1; order <= terms_used; ++order) {
        for (int q = tid; q < elements; q += static_cast<int>(blockDim.x)) {
            const int row = q % m;
            const int col = q / m;
            double sum = 0.0;
            for (int k = 0; k < m; ++k)
                sum += term[row + k * MAX_M] * A[k + col * MAX_M];
            work[row + col * MAX_M] = sum / static_cast<double>(order);
        }
        __syncthreads();
        for (int q = tid; q < elements; q += static_cast<int>(blockDim.x)) {
            const int row = q % m;
            const int col = q / m;
            const int dst = row + col * MAX_M;
            term[dst] = work[dst];
            E[dst] += term[dst];
        }
        __syncthreads();
    }

    for (int q = 0; q < *squarings; ++q) {
        for (int e = tid; e < elements; e += static_cast<int>(blockDim.x)) {
            const int row = e % m;
            const int col = e / m;
            double sum = 0.0;
            for (int k = 0; k < m; ++k)
                sum += E[row + k * MAX_M] * E[k + col * MAX_M];
            work[row + col * MAX_M] = sum;
        }
        __syncthreads();
        for (int e = tid; e < elements; e += static_cast<int>(blockDim.x)) {
            const int row = e % m;
            const int col = e / m;
            E[row + col * MAX_M] = work[row + col * MAX_M];
        }
        __syncthreads();
    }

    for (int row = tid; row < m; row += static_cast<int>(blockDim.x)) f[row] = E[row];
    __syncthreads();
}

/// The same at one fixed m, as its own launch. squarings_out and terms_out are optional and let
/// the caller record the work the exponential actually did rather than assume it.
template <int MAX_M>
__global__ void ca_expm_action(const double* H, int ldh, int m, double* f,
                               int* squarings_out, int* terms_out)
{
    __shared__ double A[MAX_M * MAX_M];
    __shared__ double E[MAX_M * MAX_M];
    __shared__ double term[MAX_M * MAX_M];
    __shared__ double work[MAX_M * MAX_M];
    __shared__ int squarings;

    const int tid = static_cast<int>(threadIdx.x);
    ca_expm_action_shared<MAX_M>(
        H, ldh, m, f, A, E, term, work, &squarings);
    if (tid == 0) {
        if (squarings_out != nullptr) *squarings_out = squarings;
        if (terms_out != nullptr) *terms_out = 24;
    }
}

/**
 * @brief Adaptive m: walk first_m, ..., last_m and stop at the first that meets the tolerance.
 *
 * The stopping test is the standard a posteriori Krylov estimate, beta |h_{m+1,m}| |e_m^T f|,
 * which costs one already-computed subdiagonal entry and one component of f rather than a
 * residual vector. Sweeping the candidates inside one kernel is what keeps adaptivity off the
 * critical path: the alternative is a launch and a host round-trip per trial m, and the m that
 * is finally accepted would have been recomputed several times over.
 */
template <int MAX_M>
__global__ void ca_expm_candidates(
    const double* H, int ldh, int first_m, int last_m, double beta, double tol,
    double* f, GpuCaCandidateResult* result)
{
    __shared__ double A[MAX_M * MAX_M];
    __shared__ double E[MAX_M * MAX_M];
    __shared__ double term[MAX_M * MAX_M];
    __shared__ double work[MAX_M * MAX_M];
    __shared__ int squarings;
    __shared__ int accepted;

    const int tid = static_cast<int>(threadIdx.x);
    for (int m = first_m; m <= last_m; ++m) {
        ca_expm_action_shared<MAX_M>(
            H, ldh, m, f, A, E, term, work, &squarings);
        if (tid == 0) {
            result->residual =
                beta * fabs(H[m + (m - 1) * ldh]) * fabs(f[m - 1]);
            accepted = result->residual < tol;
            result->selected_m = accepted ? m : 0;
        }
        __syncthreads();
        if (accepted) return;
    }
}
