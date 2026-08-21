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

// The slab layout is packed: a plane is n x n and a halo message is a whole
// number of planes. A padded row stride would change what a neighbor exchange
// sends and is not admitted here until the one-GPU pitched arm has passed its
// numerical and profiling gates, so building this path pitched is an error
// rather than a silently different message size.
static_assert(kMpkPitchAlignment == 1,
              "the distributed halo layout is packed; see the pitched arm gate");

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
    const double* CAKSM_MPK_RESTRICT halo_start,
    double* CAKSM_MPK_RESTRICT B, int64_t ld,
    GpuPdeOperator op, const double* CAKSM_MPK_RESTRICT face_b,
    double scale,
    int z_begin, int z_count, int output_z_begin, int output_z_count)
{
    static_assert(S >= 1 && S <= kMpkMaxS);
    constexpr int logical_ex = kMpkBlockX + 2 * S;
    constexpr int ex = logical_ex + kMpkSharedPadX;
    constexpr int ey = kMpkBlockY + 2 * S;
    constexpr int ez = kMpkBlockZ + 2 * S;
    constexpr int logical_volume = logical_ex * ey * ez;
    constexpr int shared_volume = ex * ey * ez;

    extern __shared__ double storage[];
    double* in = storage;
    double* out = storage + shared_volume;

    const int tid = static_cast<int>(threadIdx.x);
    const int ox = static_cast<int>(blockIdx.x) * kMpkBlockX;
    const int oy = static_cast<int>(blockIdx.y) * kMpkBlockY;
    const int oz = output_z_begin
        + static_cast<int>(blockIdx.z) * kMpkBlockZ;
    const int output_z_end = output_z_begin + output_z_count;
    const int n = op.n;
    const int64_t n2 = static_cast<int64_t>(n) * n;

    for (int q = tid; q < logical_volume;
         q += static_cast<int>(blockDim.x)) {
        const int lx = q % logical_ex;
        const int ly = (q / logical_ex) % ey;
        const int lz = q / (logical_ex * ey);
        const int shared_q = mpk_index(lx, ly, lz, ex, ey);
        const int i = ox + lx - S;
        const int j = oy + ly - S;
        const int local_k = oz + lz - S;
        const int global_k = z_begin + local_k;
        const int halo_k = local_k + S;
        if (i >= 0 && i < n && j >= 0 && j < n
            && global_k >= 0 && global_k < n
            && halo_k >= 0 && halo_k < z_count + 2 * S) {
            in[shared_q] =
                halo_start[static_cast<int64_t>(halo_k) * n2
                           + static_cast<int64_t>(j) * n + i];
        } else {
            in[shared_q] = 0.0;
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
        if (i < n && j < n && local_k >= output_z_begin
            && local_k < output_z_end && local_k < z_count) {
            const int64_t gid =
                static_cast<int64_t>(local_k) * n2
                + static_cast<int64_t>(j) * n + i;
            B[gid] = in[mpk_index(tx + S, ty + S, tz + S, ex, ey)];
        }
    }

    const int64_t local_N = static_cast<int64_t>(z_count) * n2;
    for (int step = 1; step <= S; ++step) {
        const int lo = step;
        const int hi_x = logical_ex - step;
        const int hi_y = ey - step;
        const int hi_z = ez - step;
        const double* tail =
            B + static_cast<int64_t>(step - 1) * ld + local_N;

        for (int q = tid; q < logical_volume;
             q += static_cast<int>(blockDim.x)) {
            const int lx = q % logical_ex;
            const int ly = (q / logical_ex) % ey;
            const int lz = q / (logical_ex * ey);
            const int shared_q = mpk_index(lx, ly, lz, ex, ey);
            if (lx < lo || lx >= hi_x || ly < lo || ly >= hi_y
                || lz < lo || lz >= hi_z)
                continue;

            const int i = ox + lx - S;
            const int j = oy + ly - S;
            const int local_k = oz + lz - S;
            const int global_k = z_begin + local_k;
            if (i >= 0 && i < n && j >= 0 && j < n
                && global_k >= 0 && global_k < n) {
                out[shared_q] =
                    scale * mpk_apply(
                        in, lx, ly, lz, ex, ey,
                        i, j, global_k, op, face_b, tail);
            } else {
                out[shared_q] = 0.0;
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
            if (i < n && j < n && local_k >= output_z_begin
                && local_k < output_z_end && local_k < z_count) {
                const int64_t gid =
                    static_cast<int64_t>(local_k) * n2
                    + static_cast<int64_t>(j) * n + i;
                B[gid + static_cast<int64_t>(step) * ld] =
                    out[mpk_index(tx + S, ty + S, tz + S, ex, ey)];
            }
        }
        mpk_step_barrier();

        double* swap = in;
        in = out;
        out = swap;
    }
}

/**
 * @brief Opt in to the shared memory the width needs and launch one field range.
 *
 * Two tiles of doubles exceed the 48 KiB default at these widths, so the dynamic limit is raised
 * per kernel before the launch. The caller builds the replicated tail before any field range,
 * because mpk_apply reads tail column q-1 while producing field column q.
 */
template <int S>
[[nodiscard]] inline cudaError_t launch_mpk_slab_range(
    const double* halo_start, double* B, int64_t ld,
    const GpuPdeOperator& op, const double* face_b, double scale,
    int z_begin, int z_count, int output_z_begin, int output_z_count,
    cudaStream_t stream, MpkSharedCarveout carveout)
{
    if (output_z_count <= 0) return cudaSuccess;
#if defined(CAKSM_MPK_USE_RESTRICT)
    if (halo_start == B || face_b == B)
        return cudaErrorInvalidValue;
#endif
    constexpr std::size_t volume =
        static_cast<std::size_t>(kMpkBlockX + 2 * S + kMpkSharedPadX)
        * static_cast<std::size_t>(kMpkBlockY + 2 * S)
        * static_cast<std::size_t>(kMpkBlockZ + 2 * S);
    constexpr std::size_t shared_bytes = 2 * volume * sizeof(double);

    cudaError_t error = cudaFuncSetAttribute(
        mpk_slab_kernel<S>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes));
    if (error != cudaSuccess) return error;
    error = cudaFuncSetAttribute(
        mpk_slab_kernel<S>, cudaFuncAttributePreferredSharedMemoryCarveout,
        mpk_carveout_value(carveout));
    if (error != cudaSuccess) return error;

    const dim3 grid(
        static_cast<unsigned>((op.n + kMpkBlockX - 1) / kMpkBlockX),
        static_cast<unsigned>((op.n + kMpkBlockY - 1) / kMpkBlockY),
        static_cast<unsigned>(
            (output_z_count + kMpkBlockZ - 1) / kMpkBlockZ));
    mpk_slab_kernel<S><<<grid, kMpkThreadsPerBlock, shared_bytes, stream>>>(
        halo_start, B, ld, op, face_b, scale, z_begin, z_count,
        output_z_begin, output_z_count);
    return cudaGetLastError();
}

/// Build the replicated augmented tail once before field ranges consume it.
[[nodiscard]] inline cudaError_t gpu_pde_slab_tail_powers(
    const double* start, double* B, int64_t ld, int steps,
    const GpuPdeOperator& op, double scale, int z_count,
    cudaStream_t stream)
{
    const int64_t local_N =
        static_cast<int64_t>(z_count) * op.n * op.n;
    mpk_tail_powers<<<1, 1, 0, stream>>>(
        start, B, local_N, ld, steps, scale, nullptr, 1.0);
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
[[nodiscard]] inline cudaError_t gpu_pde_slab_matrix_powers_range(
    const double* halo_start,
    double* B, int64_t ld, int steps,
    const GpuPdeOperator& op, const double* face_b, double scale,
    int z_begin, int z_count, int output_z_begin, int output_z_count,
    cudaStream_t stream,
    MpkSharedCarveout carveout = MpkSharedCarveout::Default)
{
    switch (steps) {
        case 1:
            return launch_mpk_slab_range<1>(
                halo_start, B, ld, op, face_b, scale,
                z_begin, z_count, output_z_begin, output_z_count, stream,
                carveout);
        case 2:
            return launch_mpk_slab_range<2>(
                halo_start, B, ld, op, face_b, scale,
                z_begin, z_count, output_z_begin, output_z_count, stream,
                carveout);
        case 3:
            return launch_mpk_slab_range<3>(
                halo_start, B, ld, op, face_b, scale,
                z_begin, z_count, output_z_begin, output_z_count, stream,
                carveout);
        case 4:
            return launch_mpk_slab_range<4>(
                halo_start, B, ld, op, face_b, scale,
                z_begin, z_count, output_z_begin, output_z_count, stream,
                carveout);
        case 5:
            return launch_mpk_slab_range<5>(
                halo_start, B, ld, op, face_b, scale,
                z_begin, z_count, output_z_begin, output_z_count, stream,
                carveout);
        case 6:
            return launch_mpk_slab_range<6>(
                halo_start, B, ld, op, face_b, scale,
                z_begin, z_count, output_z_begin, output_z_count, stream,
                carveout);
        default:
            return cudaErrorInvalidValue;
    }
}

[[nodiscard]] inline cudaError_t gpu_pde_slab_matrix_powers(
    const double* start, const double* halo_start,
    double* B, int64_t ld, int steps,
    const GpuPdeOperator& op, const double* face_b, double scale,
    int z_begin, int z_count, cudaStream_t stream,
    MpkSharedCarveout carveout = MpkSharedCarveout::Default)
{
    cudaError_t error = gpu_pde_slab_tail_powers(
        start, B, ld, steps, op, scale, z_count, stream);
    if (error != cudaSuccess) return error;
    return gpu_pde_slab_matrix_powers_range(
        halo_start, B, ld, steps, op, face_b, scale,
        z_begin, z_count, 0, z_count, stream, carveout);
}
