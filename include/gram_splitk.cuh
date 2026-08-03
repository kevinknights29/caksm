/**
 * @file gram_splitk.cuh
 * @brief Split-K Gram G = B^T B for a tall-skinny n x s block, roofline-bound where cuBLAS is not.
 *
 * cuBLAS syrk/gemm on a tall-skinny block (s x s output, huge k = n) launches only ~s^2 output
 * tiles, far too few threads to hide memory latency, so it runs orders of magnitude below the
 * roofline. Splitting the long k = n dimension across a device-filling grid and reducing the
 * partial Grams makes the kernel memory-bound (B is read once), which is the roofline its
 * arithmetic intensity implies. The technique is the TSMTTSM kernel of Ernst, Hager, Thies and
 * Wellein, "Performance Engineering for Real and Complex Tall & Skinny Matrix Multiplication
 * Kernels on GPUs" (2020). This is a straightforward parallel-over-k version, not their autotuned
 * code generator.
 *
 * Each thread reads its rows' s values once and accumulates only the upper triangle of the s x s
 * partial in registers (the Gram is symmetric), which halves the accumulator and so doubles
 * occupancy against a full s x s. The partials are combined with a warp-shuffle reduction rather
 * than shared-memory atomics, which Nsight Compute showed dominated the stall time. Only per-warp
 * leaders touch shared memory, and the block leader writes the mirrored full matrix to G.
 *
 * @author Kevin Knights
 * @date 2026-07-25
 */
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

/**
 * @brief One split-K Gram, templated on s so the packed-triangle accumulator is register-held.
 *
 * B is column-major with leading dimension ld, so consecutive threads read consecutive rows and
 * each per-column load coalesces. G (s x s, column-major) must be zeroed before launch; the
 * kernel accumulates the full symmetric matrix over the first n rows.
 */
template <int S>
__global__ void __launch_bounds__(128)
gram_splitk_kernel(
    const double* __restrict__ B, int64_t n, int64_t ld,
    double* __restrict__ G)
{
    constexpr int T = S * (S + 1) / 2;   // upper-triangle packed length
    double acc[T];
    #pragma unroll
    for (int t = 0; t < T; ++t) acc[t] = 0.0;

    // Grid-stride over the rows: read the s values of a row once, accumulate the triangle.
    const int64_t stride = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x)
                     + static_cast<int64_t>(threadIdx.x);
         row < n; row += stride) {
        double b[S];
        #pragma unroll
        for (int c = 0; c < S; ++c) b[c] = B[row + static_cast<int64_t>(c) * ld];
        int t = 0;
        #pragma unroll
        for (int j = 0; j < S; ++j)
            #pragma unroll
            for (int i = 0; i <= j; ++i)
                acc[t++] += b[i] * b[j];
    }

    // Warp-shuffle reduction: lane 0 of each warp ends with the warp's partial for every element.
    #pragma unroll
    for (int t = 0; t < T; ++t)
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc[t] += __shfl_down_sync(0xffffffffu, acc[t], off);

    // Per-warp leaders sum into a shared triangle; the block leader writes the mirrored full G.
    __shared__ double cs[T];
    for (int t = static_cast<int>(threadIdx.x); t < T; t += static_cast<int>(blockDim.x))
        cs[t] = 0.0;
    __syncthreads();
    if ((threadIdx.x & 31u) == 0u)
        #pragma unroll
        for (int t = 0; t < T; ++t) atomicAdd(&cs[t], acc[t]);
    __syncthreads();
    if (threadIdx.x == 0) {
        int t = 0;
        #pragma unroll
        for (int j = 0; j < S; ++j)
            #pragma unroll
            for (int i = 0; i <= j; ++i) {
                atomicAdd(&G[i + j * S], cs[t]);
                if (i != j) atomicAdd(&G[j + i * S], cs[t]);
                ++t;
            }
    }
}

/**
 * @brief Launch the split-K Gram for a run-time s in [1, 9]. Zeros G and fills the device.
 *
 * @param sm_count      device SMs, so the grid fills the machine; a grid-stride loop then covers
 *                      any n with a bounded number of blocks (small global-atomic tail).
 * @param blocks_per_sm resident blocks to aim for per SM (tuning knob).
 * @return cudaErrorInvalidValue for an unsupported s, else the launch status.
 */
[[nodiscard]] inline cudaError_t gram_splitk_ld(
    const double* d_B, int64_t n, int64_t ld, int s, double* d_G,
    int sm_count, int blocks_per_sm = 4, cudaStream_t stream = nullptr)
{
    if (n < 0 || ld < n) return cudaErrorInvalidValue;
    const cudaError_t z = cudaMemsetAsync(
        d_G, 0, static_cast<std::size_t>(s) * static_cast<std::size_t>(s) * sizeof(double), stream);
    if (z != cudaSuccess) return z;

    const int block = 128;
    long long want = static_cast<long long>(sm_count > 0 ? sm_count : 32)
                   * static_cast<long long>(blocks_per_sm > 0 ? blocks_per_sm : 8);
    const long long cover = (n + block - 1) / block;   // no more blocks than the rows need
    if (want > cover) want = cover;
    if (want < 1) want = 1;
    const int grid = static_cast<int>(want);

    switch (s) {
        case 1: gram_splitk_kernel<1><<<grid, block, 0, stream>>>(d_B, n, ld, d_G); break;
        case 2: gram_splitk_kernel<2><<<grid, block, 0, stream>>>(d_B, n, ld, d_G); break;
        case 3: gram_splitk_kernel<3><<<grid, block, 0, stream>>>(d_B, n, ld, d_G); break;
        case 4: gram_splitk_kernel<4><<<grid, block, 0, stream>>>(d_B, n, ld, d_G); break;
        case 5: gram_splitk_kernel<5><<<grid, block, 0, stream>>>(d_B, n, ld, d_G); break;
        case 6: gram_splitk_kernel<6><<<grid, block, 0, stream>>>(d_B, n, ld, d_G); break;
        case 7: gram_splitk_kernel<7><<<grid, block, 0, stream>>>(d_B, n, ld, d_G); break;
        case 8: gram_splitk_kernel<8><<<grid, block, 0, stream>>>(d_B, n, ld, d_G); break;
        case 9: gram_splitk_kernel<9><<<grid, block, 0, stream>>>(d_B, n, ld, d_G); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

/**
 * @brief The same Gram for a block whose columns are contiguous, that is ld == n.
 *
 * The one-GPU basis is packed that way. The slab path keeps a separate leading dimension so it
 * can reduce over the rows it owns rather than the whole column, and calls gram_splitk_ld.
 */
[[nodiscard]] inline cudaError_t gram_splitk(
    const double* d_B, int64_t n, int s, double* d_G,
    int sm_count, int blocks_per_sm = 4, cudaStream_t stream = nullptr)
{
    return gram_splitk_ld(
        d_B, n, n, s, d_G, sm_count, blocks_per_sm, stream);
}
