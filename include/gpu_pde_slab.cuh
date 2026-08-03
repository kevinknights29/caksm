/**
 * @file gpu_pde_slab.cuh
 * @brief Slab-local matrix-powers kernel with an exchanged deep halo.
 *
 * The decomposition is one-dimensional in z: a slab is the contiguous range of z-planes of the
 * n-cubed grid that one GPU owns, and it owns those rows of the state for the whole run. A
 * recurrence of S steps reaches S planes past the slab boundary, so the caller hands this
 * kernel a halo_start buffer holding the owned planes flanked by S planes from each neighbor.
 * Exchanging that depth once, instead of one plane before each of S applications, is what
 * s-step buys: the same bytes in one message rather than S.
 *
 * @author Kevin Knights
 * @date 2026-07-27
 */
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "gpu_pde_matrix_powers.cuh"

/**
 * @brief Build columns 0, ..., S of the local basis from one staged halo, S fixed at compile time.
 *
 * The slab counterpart of mpk_tiled_kernel: same shared-tile recurrence, but indices are split
 * into a local k that addresses this slab's rows of B and a global k that decides which boundary
 * condition a point sees, so a point knows whether it is at a true domain face or merely at a
 * slab face. Reads come from halo_start, which is offset by S planes, and writes go to B in
 * local coordinates. Monomial only: the Newton shift and Chebyshev recurrence are one-GPU paths.
 *
 * The valid region of the shared tile retreats one plane per side per step, so an S-deep ghost
 * shell is exactly consumed by step S and the interior the block writes stays correct
 * throughout. The augmented Basket tail is replicated on every slab, so mpk_apply reads it
 * locally and no rank has to own it.
 */
template <int S>
__global__ void mpk_slab_kernel(
    const double* halo_start, double* B, int64_t ld,
    GpuPdeOperator op, const double* face_b, double scale,
    int z_begin, int z_count)
{
    static_assert(S >= 1 && S <= kMpkMaxS);
    constexpr int ex = kMpkBlockX + 2 * S;
    constexpr int ey = kMpkBlockY + 2 * S;
    constexpr int ez = kMpkBlockZ + 2 * S;
    constexpr int volume = ex * ey * ez;

    extern __shared__ double storage[];
    double* in = storage;
    double* out = storage + volume;

    const int tid = static_cast<int>(threadIdx.x);
    const int ox = static_cast<int>(blockIdx.x) * kMpkBlockX;
    const int oy = static_cast<int>(blockIdx.y) * kMpkBlockY;
    const int oz = static_cast<int>(blockIdx.z) * kMpkBlockZ;
    const int n = op.n;
    const int64_t n2 = static_cast<int64_t>(n) * n;

    for (int q = tid; q < volume; q += static_cast<int>(blockDim.x)) {
        const int lx = q % ex;
        const int ly = (q / ex) % ey;
        const int lz = q / (ex * ey);
        const int i = ox + lx - S;
        const int j = oy + ly - S;
        const int local_k = oz + lz - S;
        const int global_k = z_begin + local_k;
        const int halo_k = local_k + S;
        if (i >= 0 && i < n && j >= 0 && j < n
            && global_k >= 0 && global_k < n
            && halo_k >= 0 && halo_k < z_count + 2 * S) {
            in[q] =
                halo_start[static_cast<int64_t>(halo_k) * n2
                           + static_cast<int64_t>(j) * n + i];
        } else {
            in[q] = 0.0;
        }
    }
    __syncthreads();

    for (int q = tid; q < kMpkBlockX * kMpkBlockY * kMpkBlockZ;
         q += static_cast<int>(blockDim.x)) {
        const int tx = q % kMpkBlockX;
        const int ty = (q / kMpkBlockX) % kMpkBlockY;
        const int tz = q / (kMpkBlockX * kMpkBlockY);
        const int i = ox + tx;
        const int j = oy + ty;
        const int local_k = oz + tz;
        if (i < n && j < n && local_k < z_count) {
            const int64_t gid =
                static_cast<int64_t>(local_k) * n2
                + static_cast<int64_t>(j) * n + i;
            B[gid] = in[mpk_index(tx + S, ty + S, tz + S, ex, ey)];
        }
    }

    const int64_t local_N = static_cast<int64_t>(z_count) * n2;
    for (int step = 1; step <= S; ++step) {
        const int lo = step;
        const int hi_x = ex - step;
        const int hi_y = ey - step;
        const int hi_z = ez - step;
        const double* tail =
            B + static_cast<int64_t>(step - 1) * ld + local_N;

        for (int q = tid; q < volume; q += static_cast<int>(blockDim.x)) {
            const int lx = q % ex;
            const int ly = (q / ex) % ey;
            const int lz = q / (ex * ey);
            if (lx < lo || lx >= hi_x || ly < lo || ly >= hi_y
                || lz < lo || lz >= hi_z)
                continue;

            const int i = ox + lx - S;
            const int j = oy + ly - S;
            const int local_k = oz + lz - S;
            const int global_k = z_begin + local_k;
            if (i >= 0 && i < n && j >= 0 && j < n
                && global_k >= 0 && global_k < n) {
                out[q] =
                    scale * mpk_apply(
                        in, lx, ly, lz, ex, ey,
                        i, j, global_k, op, face_b, tail);
            } else {
                out[q] = 0.0;
            }
        }
        __syncthreads();

        for (int q = tid; q < kMpkBlockX * kMpkBlockY * kMpkBlockZ;
             q += static_cast<int>(blockDim.x)) {
            const int tx = q % kMpkBlockX;
            const int ty = (q / kMpkBlockX) % kMpkBlockY;
            const int tz = q / (kMpkBlockX * kMpkBlockY);
            const int i = ox + tx;
            const int j = oy + ty;
            const int local_k = oz + tz;
            if (i < n && j < n && local_k < z_count) {
                const int64_t gid =
                    static_cast<int64_t>(local_k) * n2
                    + static_cast<int64_t>(j) * n + i;
                B[gid + static_cast<int64_t>(step) * ld] =
                    out[mpk_index(tx + S, ty + S, tz + S, ex, ey)];
            }
        }
        __syncthreads();

        double* swap = in;
        in = out;
        out = swap;
    }
}

/**
 * @brief Opt in to the shared memory the width needs, then launch the tail and the field.
 *
 * Two tiles of doubles exceed the 48 KiB default at these widths, so the dynamic limit is raised
 * per kernel before the first launch. The 3-component tail runs first, in its own single-thread
 * kernel, because mpk_apply reads column q-1 of the tail while producing column q of the field.
 */
template <int S>
[[nodiscard]] inline cudaError_t launch_mpk_slab(
    const double* start, const double* halo_start, double* B, int64_t ld,
    const GpuPdeOperator& op, const double* face_b, double scale,
    int z_begin, int z_count, cudaStream_t stream)
{
    constexpr std::size_t volume =
        static_cast<std::size_t>(kMpkBlockX + 2 * S)
        * static_cast<std::size_t>(kMpkBlockY + 2 * S)
        * static_cast<std::size_t>(kMpkBlockZ + 2 * S);
    constexpr std::size_t shared_bytes = 2 * volume * sizeof(double);

    cudaError_t error = cudaFuncSetAttribute(
        mpk_slab_kernel<S>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes));
    if (error != cudaSuccess) return error;

    const int64_t local_N =
        static_cast<int64_t>(z_count) * op.n * op.n;
    mpk_tail_powers<<<1, 1, 0, stream>>>(
        start, B, local_N, ld, S, scale, nullptr, 1.0);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;

    const dim3 grid(
        static_cast<unsigned>((op.n + kMpkBlockX - 1) / kMpkBlockX),
        static_cast<unsigned>((op.n + kMpkBlockY - 1) / kMpkBlockY),
        static_cast<unsigned>((z_count + kMpkBlockZ - 1) / kMpkBlockZ));
    mpk_slab_kernel<S><<<grid, 128, shared_bytes, stream>>>(
        halo_start, B, ld, op, face_b, scale, z_begin, z_count);
    return cudaGetLastError();
}

/**
 * @brief Run-time entry point: dispatch a run-time step count onto the compile-time width.
 *
 * S is a template parameter because the shared tile extents and the unrolled recurrence bounds
 * have to be compile-time constants, so the only way to accept a run-time --s is this switch.
 * Widths past kMpkMaxS are rejected rather than clamped, since a silently narrowed recurrence
 * would return a shorter basis than the caller sized its workspace for.
 */
[[nodiscard]] inline cudaError_t gpu_pde_slab_matrix_powers(
    const double* start, const double* halo_start,
    double* B, int64_t ld, int steps,
    const GpuPdeOperator& op, const double* face_b, double scale,
    int z_begin, int z_count, cudaStream_t stream)
{
    switch (steps) {
        case 1:
            return launch_mpk_slab<1>(
                start, halo_start, B, ld, op, face_b, scale,
                z_begin, z_count, stream);
        case 2:
            return launch_mpk_slab<2>(
                start, halo_start, B, ld, op, face_b, scale,
                z_begin, z_count, stream);
        case 3:
            return launch_mpk_slab<3>(
                start, halo_start, B, ld, op, face_b, scale,
                z_begin, z_count, stream);
        case 4:
            return launch_mpk_slab<4>(
                start, halo_start, B, ld, op, face_b, scale,
                z_begin, z_count, stream);
        case 5:
            return launch_mpk_slab<5>(
                start, halo_start, B, ld, op, face_b, scale,
                z_begin, z_count, stream);
        case 6:
            return launch_mpk_slab<6>(
                start, halo_start, B, ld, op, face_b, scale,
                z_begin, z_count, stream);
        default:
            return cudaErrorInvalidValue;
    }
}
