/**
 * @file trsm_tallskinny.cuh
 * @brief Right-side triangular solve X R = B for a tall B (m x s) and a tiny s x s factor R.
 *
 * The B <- B R^-1 step of CholeskyQR. cuBLAS's trsm tiles the m dimension into thousands of small
 * blocks and runs compute-bound at ~2% of useful FP64 on this tall-skinny shape (measured with
 * Nsight Compute). Here R is tiny (s <= 9) and the m row-solves x_row R = b_row are independent,
 * so one thread per row does an s-element forward substitution with R staged in shared memory.
 * That makes the kernel memory-bound (read B, write X once), which its intensity ~(s+1)/16
 * implies below both cards' FP64 ridges, so unlike the Gram this solve is memory-bound on both.
 *
 * The recursive-into-GEMM schemes (Carrica et al. 2025) target a large triangular factor, and the
 * level-scheduled sparse solvers (Chen et al. 2016, Hogg 2012) target real dependency chains,
 * neither applies here, where R is small and the rows are independent.
 *
 * The diagonal reciprocals are computed once per block and the substitution multiplies by them:
 * an FP64 divide is a long software sequence, so eight per row would both add compute (compute-
 * bound on a throttled card) and spill registers (Nsight Compute showed both). Multiplying by a
 * once-computed reciprocal keeps the kernel memory-bound and low on registers.
 *
 * Solves X R = B in place (B overwritten by X), R upper-triangular, non-unit diagonal, alpha = 1,
 * matching cublasDtrsm(SIDE_RIGHT, UPPER, OP_N, DIAG_NON_UNIT) as CholQR calls it.
 *
 * @author Kevin Knights
 * @date 2026-07-26
 */
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

/**
 * @brief One row per thread: solve x R = b by forward substitution, R staged in shared memory.
 *
 * R is s x s upper, column-major (leading dimension s); B/X is m x s column-major (leading
 * dimension m). x_j = (b_j - sum_{k<j} x_k R[k,j]) / R[j,j], j ascending. B is read and written
 * once, both coalesced (consecutive threads are consecutive rows), so the kernel is memory-bound.
 */
template <int S>
__global__ void trsm_tallskinny_kernel(const double* __restrict__ R, double* __restrict__ B,
                                       int64_t m)
{
    __shared__ double sR[S * S];
    __shared__ double sInv[S];   // reciprocal of the diagonal, computed once for the whole block
    for (int e = static_cast<int>(threadIdx.x); e < S * S; e += static_cast<int>(blockDim.x))
        sR[e] = R[e];
    __syncthreads();
    if (static_cast<int>(threadIdx.x) < S)
        sInv[threadIdx.x] = 1.0 / sR[threadIdx.x + threadIdx.x * S];
    __syncthreads();

    const int64_t stride = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x)
                     + static_cast<int64_t>(threadIdx.x);
         row < m; row += stride) {
        double x[S];
        #pragma unroll
        for (int j = 0; j < S; ++j) x[j] = B[row + static_cast<int64_t>(j) * m];
        #pragma unroll
        for (int j = 0; j < S; ++j) {
            double acc = x[j];
            #pragma unroll
            for (int k = 0; k < j; ++k) acc -= x[k] * sR[k + j * S];
            x[j] = acc * sInv[j];
        }
        #pragma unroll
        for (int j = 0; j < S; ++j) B[row + static_cast<int64_t>(j) * m] = x[j];
    }
}

/**
 * @brief Solve X R = B in place for a run-time s in [1, 9]. Fills the device with a grid-stride
 *        launch. R is s x s upper (lead dim s); B is m x s (lead dim m), overwritten with X.
 * @return cudaErrorInvalidValue for an unsupported s, else the launch status.
 */
[[nodiscard]] inline cudaError_t trsm_tallskinny(const double* d_R, double* d_B, int64_t m, int s,
                                                 int sm_count, int blocks_per_sm = 8,
                                                 cudaStream_t stream = nullptr)
{
    const int block = 256;
    long long want = static_cast<long long>(sm_count > 0 ? sm_count : 32)
                   * static_cast<long long>(blocks_per_sm > 0 ? blocks_per_sm : 8);
    const long long cover = (m + block - 1) / block;   // no more blocks than the rows need
    if (want > cover) want = cover;
    if (want < 1) want = 1;
    const int grid = static_cast<int>(want);

    switch (s) {
        case 1: trsm_tallskinny_kernel<1><<<grid, block, 0, stream>>>(d_R, d_B, m); break;
        case 2: trsm_tallskinny_kernel<2><<<grid, block, 0, stream>>>(d_R, d_B, m); break;
        case 3: trsm_tallskinny_kernel<3><<<grid, block, 0, stream>>>(d_R, d_B, m); break;
        case 4: trsm_tallskinny_kernel<4><<<grid, block, 0, stream>>>(d_R, d_B, m); break;
        case 5: trsm_tallskinny_kernel<5><<<grid, block, 0, stream>>>(d_R, d_B, m); break;
        case 6: trsm_tallskinny_kernel<6><<<grid, block, 0, stream>>>(d_R, d_B, m); break;
        case 7: trsm_tallskinny_kernel<7><<<grid, block, 0, stream>>>(d_R, d_B, m); break;
        case 8: trsm_tallskinny_kernel<8><<<grid, block, 0, stream>>>(d_R, d_B, m); break;
        case 9: trsm_tallskinny_kernel<9><<<grid, block, 0, stream>>>(d_R, d_B, m); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}
