/**
 * @file gpu_pde_matrix_powers_streamed.cuh
 * @brief Plane-streamed matrix-powers kernel: an x-y tile held on chip, z streamed through it.
 *
 * A separate kernel family from the accepted full-volume implementation. Both
 * remain available because the promoted dispatch keeps the full-volume family
 * for small grids and inadmissible streamed widths. The full-volume tile stages
 * (X+2S)(Y+2S)(Z+2S) doubles twice, so its shared-memory cost grows with all
 * three halo dimensions and the z extent it can afford is small. That is what
 * fixes its redundancy: a small z tile pays a 2S-deep ghost shell in z as well.
 *
 * Here the block owns an x-y tile and walks a contiguous z segment, holding a
 * short circular queue of planes per recurrence level. The shared-memory cost is
 * (S+1) levels times the queue depth times one plane, independent of how far the
 * segment runs, so the z redundancy (H+2S)/H falls as the segment lengthens
 * while the footprint does not move. What it does not remove is the compulsory
 * basis output: every column is still written once per interior point, because
 * Krylov orthogonalization consumes all of them.
 *
 * @author Kevin Knights
 * @date 2026-08-08
 */
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "gpu_pde_matrix_powers.cuh"

// The x-y tile one block owns, and the default z segment it streams. The
// segment length is the one tuning parameter that costs no shared memory, so it
// is a run-time argument with a compile-time default rather than a separate
// binary; the tile extents fix the plane size and must stay compile-time.
#ifndef CAKSM_MPK_STREAM_TILE_X
#define CAKSM_MPK_STREAM_TILE_X 16
#endif
#ifndef CAKSM_MPK_STREAM_TILE_Y
#define CAKSM_MPK_STREAM_TILE_Y 8
#endif
#ifndef CAKSM_MPK_STREAM_HEIGHT
#define CAKSM_MPK_STREAM_HEIGHT 8
#endif
#ifndef CAKSM_MPK_STREAM_THREADS
#define CAKSM_MPK_STREAM_THREADS 128
#endif
#ifndef CAKSM_MPK_STREAM_SHARED_PAD_X
#define CAKSM_MPK_STREAM_SHARED_PAD_X 0
#endif

inline constexpr int kMpkStreamTileX = CAKSM_MPK_STREAM_TILE_X;
inline constexpr int kMpkStreamTileY = CAKSM_MPK_STREAM_TILE_Y;
inline constexpr int kMpkStreamHeight = CAKSM_MPK_STREAM_HEIGHT;
inline constexpr int kMpkStreamThreads = CAKSM_MPK_STREAM_THREADS;
inline constexpr int kMpkStreamSharedPadX = CAKSM_MPK_STREAM_SHARED_PAD_X;

static_assert(kMpkStreamTileX >= 1 && kMpkStreamTileY >= 1);
static_assert(kMpkStreamHeight >= 1);
static_assert(kMpkStreamSharedPadX >= 0);
static_assert(kMpkStreamThreads >= 32 && kMpkStreamThreads <= 1024);
static_assert(kMpkStreamThreads % 32 == 0,
              "a partial warp wastes a scheduler slot on every launch");

/**
 * Planes one level keeps live.
 *
 * The schedule below lags each level two planes behind the one under it, so a
 * level reads its predecessor's planes from the three preceding pipeline steps
 * while writing its own for the current one: four slots. Chebyshev additionally
 * reads the level two below at the same plane, which the same arithmetic places
 * four steps back, so it needs one more.
 *
 * Declared for both sides: the kernel sizes its slot arithmetic from this, and
 * the launch sizes the shared-memory request from it, and the two must be the
 * same number rather than the same literal written twice.
 */
[[nodiscard]] __host__ __device__ inline constexpr int mpk_stream_queue_depth(
    bool chebyshev)
{
    return chebyshev ? 5 : 4;
}

/// Doubles one staged plane occupies, optional row padding included.
[[nodiscard]] inline constexpr int64_t mpk_stream_plane_doubles(int steps)
{
    return static_cast<int64_t>(
               kMpkStreamTileX + 2 * steps + kMpkStreamSharedPadX)
         * static_cast<int64_t>(kMpkStreamTileY + 2 * steps);
}

/// Dynamic shared memory one block requests: every level's whole queue.
[[nodiscard]] inline constexpr std::size_t mpk_stream_shared_bytes(
    int steps, bool chebyshev)
{
    if (steps < 1 || steps > kMpkMaxS) return 0;
    return static_cast<std::size_t>(steps + 1)
         * static_cast<std::size_t>(mpk_stream_queue_depth(chebyshev))
         * static_cast<std::size_t>(mpk_stream_plane_doubles(steps))
         * sizeof(double);
}

/// Interior points one block writes per basis column at this segment length.
[[nodiscard]] inline constexpr int64_t mpk_stream_interior_points(int height)
{
    return static_cast<int64_t>(kMpkStreamTileX)
         * static_cast<int64_t>(kMpkStreamTileY)
         * static_cast<int64_t>(height);
}

/// Points one block stages to produce them: the plane plus its z prologue and
/// epilogue. Unlike the full-volume tile this grows with the segment rather than
/// with the shared-memory budget, so a longer segment is cheaper per point.
[[nodiscard]] inline constexpr int64_t mpk_stream_staged_points(
    int steps, int height)
{
    return static_cast<int64_t>(kMpkStreamTileX + 2 * steps)
         * static_cast<int64_t>(kMpkStreamTileY + 2 * steps)
         * static_cast<int64_t>(height + 2 * steps);
}

/// Staged points in excess of interior points, per interior point.
[[nodiscard]] inline constexpr double mpk_stream_redundant_fraction(
    int steps, int height)
{
    const double interior =
        static_cast<double>(mpk_stream_interior_points(height));
    return (static_cast<double>(mpk_stream_staged_points(steps, height))
            - interior) / interior;
}

/// Segment length a run actually uses: zero requests the whole z extent.
[[nodiscard]] inline constexpr int mpk_stream_effective_height(
    int n, int height)
{
    return height > 0 ? height : n;
}

/// Grid z blocks the segment sweep found the optimum sitting at.
///
/// Measured, not derived. Across grids of 31, 61, 77 and 97 and both option
/// types, the fastest segment length was never a fixed number of planes: it was
/// whatever kept the launch at roughly this many blocks in z. Shorter segments
/// stage more ghost planes per interior point, longer ones leave the device
/// short of blocks, and this is where the two crossed on a V100.
inline constexpr int kMpkStreamTargetZBlocks = 8;

/// The segment length that rule implies for a grid, from the swept lengths.
///
/// Restricted to the lengths that were actually measured rather than solving
/// the rule exactly, so a dispatched run lands on a configuration the sweep
/// covered instead of interpolating into one it never timed.
[[nodiscard]] inline int mpk_stream_auto_height(int n)
{
    constexpr int candidates[] = {4, 8, 16, 32};
    int best = candidates[0];
    int best_distance = -1;
    for (const int height : candidates) {
        const int blocks = (n + height - 1) / height;
        const int distance =
            blocks > kMpkStreamTargetZBlocks
                ? blocks - kMpkStreamTargetZBlocks
                : kMpkStreamTargetZBlocks - blocks;
        if (best_distance < 0 || distance < best_distance) {
            best_distance = distance;
            best = height;
        }
    }
    return best;
}

/// Smallest grid at which the streamed family was measured to win.
///
/// Below this the launch cannot fill the device: at n=31 the streamed tile
/// leaves 32 to 64 blocks against 80 SMs, and the production tile's much larger
/// block count beats it despite moving several times more data. The boundary is
/// a property of the device and the grid, not of the recurrence.
inline constexpr int kMpkStreamMinimumGrid = 61;

/// Thread blocks one launch dispatches.
[[nodiscard]] inline constexpr int64_t mpk_stream_grid_blocks(
    int n, int height)
{
    const int extent = mpk_stream_effective_height(n, height);
    return static_cast<int64_t>((n + kMpkStreamTileX - 1) / kMpkStreamTileX)
         * static_cast<int64_t>((n + kMpkStreamTileY - 1) / kMpkStreamTileY)
         * static_cast<int64_t>((n + extent - 1) / extent);
}

/// Block-wide barriers one block executes: one per pipeline step, and a step per
/// streamed plane plus the depth the temporal halo adds.
[[nodiscard]] inline constexpr int64_t mpk_stream_barriers_per_block(
    int steps, int height)
{
    return 3 * static_cast<int64_t>(steps) + static_cast<int64_t>(height);
}

/// The launch record for this family, so a tuning row names what it measured.
[[nodiscard]] inline MpkLaunchRecord mpk_stream_launch_record(
    int n, int steps, int height, bool chebyshev)
{
    const int extent = mpk_stream_effective_height(n, height);
    MpkLaunchRecord record;
    record.family = mpk_family_name(MpkKernelFamily::PlaneStreamed);
    record.tile_x = kMpkStreamTileX;
    record.tile_y = kMpkStreamTileY;
    record.tile_z = extent;
    record.stream_height = extent;
    record.threads_per_block = kMpkStreamThreads;
    record.shared_pad_x = kMpkStreamSharedPadX;
    record.interior_points = mpk_stream_interior_points(extent);
    record.staged_points =
        steps >= 1 ? mpk_stream_staged_points(steps, extent) : 0;
    record.redundant_fraction =
        steps >= 1 ? mpk_stream_redundant_fraction(steps, extent) : 0.0;
    record.points_per_thread =
        static_cast<double>(record.interior_points)
        / static_cast<double>(kMpkStreamThreads);
    record.dynamic_shared_bytes = mpk_stream_shared_bytes(steps, chebyshev);
    return record;
}

/// Flat offset into one staged plane of row stride ex, x fastest.
__device__ __forceinline__ int mpk_plane_index(int lx, int ly, int ex)
{
    return ly * ex + lx;
}

/// The queue slot holding the plane a level produced at a given pipeline step.
__device__ __forceinline__ double* mpk_stream_slot(
    double* storage, int level, int step, int queue, int plane_doubles)
{
    return storage
         + (static_cast<int64_t>(level) * queue
            + static_cast<int64_t>(step % queue))
             * static_cast<int64_t>(plane_doubles);
}

/// Which of the three live planes a z offset of -1, 0 or +1 refers to.
__device__ __forceinline__ const double* mpk_stream_plane(
    const double* below, const double* mid, const double* above, int dz)
{
    if (dz < 0) return below;
    if (dz > 0) return above;
    return mid;
}

/**
 * One row of the pricing operator, read from three separate staged planes.
 *
 * This mirrors mpk_apply term for term. It exists as a second implementation
 * because the two families store the z neighbors differently: the full-volume
 * tile is one contiguous volume, so a z step there is a constant stride, while
 * here the planes at k-1, k and k+1 live in unrelated queue slots and have to be
 * addressed as separate pointers. The full 19-point stencil is retained: the xz
 * and yz mixed derivatives read diagonal values from the planes above and below,
 * which is why the queue holds whole planes rather than a z pencil per thread.
 * Both implementations are gated by the same column-by-column comparison against
 * the assembled operator, which is what keeps them from drifting apart.
 */
__device__ __forceinline__ double mpk_stream_apply(
    const double* below, const double* mid, const double* above,
    int lx, int ly, int ex, int i, int j, int k,
    const GpuPdeOperator& op, const double* face_b, const double* tail)
{
    const int coord[3] = {i, j, k};
    const int center = mpk_plane_index(lx, ly, ex);
    double y = op.reaction * mid[center]
             - op.diffusion[0] * 2.0 * mid[center]
             - op.diffusion[1] * 2.0 * mid[center]
             - op.diffusion[2] * 2.0 * mid[center];

#pragma unroll
    for (int d = 0; d < 3; ++d) {
        double low, high;
        if (d == 0) {
            low = mid[center - 1];
            high = mid[center + 1];
        } else if (d == 1) {
            low = mid[center - ex];
            high = mid[center + ex];
        } else {
            low = below[center];
            high = above[center];
        }
        const int q = coord[d];
        if (op.rainbow && q == op.n - 1) {
            y += op.drift[d] * (-2.0 * low + 2.0 * mid[center]);
            y += 2.0 * op.diffusion[d] * mid[center];
        } else {
            if (q > 0) y += (op.diffusion[d] - op.drift[d]) * low;
            if (q + 1 < op.n) y += (op.diffusion[d] + op.drift[d]) * high;
        }
    }

    // The three mixed second derivatives, each a tensor product of two T rows.
    // They are written out rather than looped because the pair decides which
    // plane a neighbor lives in, and that is not an offset that can be folded
    // into a stride the way the full-volume tile folds it.
    int dd[2], de[2], nd = 0, ne = 0;
    double td[2], te[2];
    const bool rainbow = op.rainbow != 0;

    mpk_t_row(i, op.n, rainbow, dd, td, nd);
    mpk_t_row(j, op.n, rainbow, de, te, ne);
    for (int a = 0; a < nd; ++a)
        for (int b = 0; b < ne; ++b)
            y += op.mixed[0] * td[a] * te[b]
               * mid[center + dd[a] + de[b] * ex];

    mpk_t_row(i, op.n, rainbow, dd, td, nd);
    mpk_t_row(k, op.n, rainbow, de, te, ne);
    for (int a = 0; a < nd; ++a)
        for (int b = 0; b < ne; ++b) {
            const double* plane =
                mpk_stream_plane(below, mid, above, de[b]);
            y += op.mixed[1] * td[a] * te[b] * plane[center + dd[a]];
        }

    mpk_t_row(j, op.n, rainbow, dd, td, nd);
    mpk_t_row(k, op.n, rainbow, de, te, ne);
    for (int a = 0; a < nd; ++a)
        for (int b = 0; b < ne; ++b) {
            const double* plane =
                mpk_stream_plane(below, mid, above, de[b]);
            y += op.mixed[2] * td[a] * te[b] * plane[center + dd[a] * ex];
        }

    if (!op.rainbow && face_b != nullptr && tail != nullptr) {
        const int n2 = op.n * op.n;
        for (int d = 0; d < 3; ++d) {
            if (coord[d] != op.n - 1) continue;
            const int f = mpk_face_index(d, i, j, k, op.n);
            for (int c = 0; c < 3; ++c)
                y += face_b[(d * 3 + c) * n2 + f] * tail[c];
        }
    }
    return y;
}

/**
 * @brief Build columns 0, ..., S of the basis by streaming z through an x-y tile.
 *
 * The schedule is the whole design. At pipeline step t the block loads input
 * plane z0 - S + t and each level q writes plane z0 - S + t - 2q, two planes
 * behind the level under it. That lag is what makes every value a level reads
 * older than the current step: level q reads level q-1 at steps t-1, t-2 and
 * t-3, and for Chebyshev level q-2 at step t-4. Nothing written during step t is
 * read during step t, so one block-wide barrier per streamed plane is sufficient
 * rather than one per level.
 *
 * The prologue and epilogue fall out of the same arithmetic: level q is active
 * only while its plane lies in the range its consumer needs, so the deeper
 * levels start later and finish later, and 3S extra steps cover the temporal
 * halo at both ends. Every level writes its interior points to global memory as
 * it produces them, which is the compulsory basis output and not an optimization
 * this kernel is free to skip.
 *
 * With shifts null and inverse_normalization 1 the recurrence is monomial;
 * otherwise it is the Newton form. CHEBYSHEV selects the three-term recurrence,
 * which needs no chunking here because a third live level costs one plane
 * instead of a third full volume. That is also why these pointers can be
 * restrict-qualified where the chunked full-volume Chebyshev kernel's cannot:
 * with no chunk boundary there is no launch in which the input and the output
 * column are the same address.
 */
template <int S, bool CHEBYSHEV>
__global__ void mpk_stream_kernel(
    const double* CAKSM_MPK_RESTRICT start,
    double* CAKSM_MPK_RESTRICT B, int64_t ld,
    GpuPdeOperator op, const double* CAKSM_MPK_RESTRICT face_b,
    double scale, const double* CAKSM_MPK_RESTRICT shifts,
    double inverse_normalization, double center, double half_width,
    int height)
{
    static_assert(S >= 1 && S <= kMpkMaxS);
    constexpr int queue = mpk_stream_queue_depth(CHEBYSHEV);
    constexpr int logical_ex = kMpkStreamTileX + 2 * S;
    constexpr int ex = logical_ex + kMpkStreamSharedPadX;
    constexpr int ey = kMpkStreamTileY + 2 * S;
    constexpr int plane_doubles = ex * ey;
    constexpr int stage_points = logical_ex * ey;

    extern __shared__ double storage[];

    const int tid = static_cast<int>(threadIdx.x);
    const int ox = static_cast<int>(blockIdx.x) * kMpkStreamTileX;
    const int oy = static_cast<int>(blockIdx.y) * kMpkStreamTileY;
    const int extent = height > 0 ? height : op.n;
    const int z_begin = static_cast<int>(blockIdx.z) * extent;
    if (z_begin >= op.n) return;
    const int z_end =
        z_begin + extent < op.n ? z_begin + extent : op.n;
    const int owned = z_end - z_begin;
    const int steps_total = 3 * S + owned;

    for (int t = 0; t < steps_total; ++t) {
        // Level zero: stage one input plane and, where it is an owned plane,
        // emit column zero from the value already in hand.
        if (t <= owned - 1 + 2 * S) {
            const int zl = z_begin - S + t;
            double* level0 =
                mpk_stream_slot(storage, 0, t, queue, plane_doubles);
            const bool owned_plane = zl >= z_begin && zl < z_end;
            for (int q = tid; q < stage_points; q += kMpkStreamThreads) {
                const int lx = q % logical_ex;
                const int ly = q / logical_ex;
                const int i = ox + lx - S;
                const int j = oy + ly - S;
                const bool inside =
                    i >= 0 && i < op.n && j >= 0 && j < op.n
                    && zl >= 0 && zl < op.n;
                const double value =
                    inside ? start[mpk_field_index(i, j, zl, op)] : 0.0;
                level0[mpk_plane_index(lx, ly, ex)] = value;
                if (owned_plane && lx >= S && lx < S + kMpkStreamTileX
                    && ly >= S && ly < S + kMpkStreamTileY
                    && i < op.n && j < op.n)
                    B[mpk_field_index(i, j, zl, op)] = value;
            }
        }

#pragma unroll
        for (int level = 1; level <= S; ++level) {
            if (t < 3 * level || t > level + 2 * S + owned - 1) continue;

            const int zc = z_begin - S + t - 2 * level;
            const double* above =
                mpk_stream_slot(storage, level - 1, t - 1, queue, plane_doubles);
            const double* mid =
                mpk_stream_slot(storage, level - 1, t - 2, queue, plane_doubles);
            const double* below =
                mpk_stream_slot(storage, level - 1, t - 3, queue, plane_doubles);
            // The second predecessor the three-term recurrence needs, at the same
            // plane. The indices are clamped rather than left to a guard so that
            // the pointer is always inside the allocation: it is read only when
            // the level has a second predecessor, but a clamped pointer cannot
            // fault even if the load is speculated. Dead for the other two
            // recurrences, which never read it.
            const double* older = mpk_stream_slot(
                storage, level >= 2 ? level - 2 : 0, t >= 4 ? t - 4 : 0,
                queue, plane_doubles);
            double* out =
                mpk_stream_slot(storage, level, t, queue, plane_doubles);
            const double* tail =
                B + static_cast<int64_t>(level - 1) * ld + op.N;

            // The valid region retreats one point per side per level, exactly as
            // it does in the full-volume tile; iterating it directly rather than
            // masking the whole plane keeps the idle threads out of the loop.
            const int low = level;
            const int valid_x = logical_ex - 2 * level;
            const int valid_y = ey - 2 * level;
            const int valid_points = valid_x * valid_y;
            const bool owned_plane = zc >= z_begin && zc < z_end;

            for (int q = tid; q < valid_points; q += kMpkStreamThreads) {
                const int lx = low + q % valid_x;
                const int ly = low + q / valid_x;
                const int i = ox + lx - S;
                const int j = oy + ly - S;
                const int slot = mpk_plane_index(lx, ly, ex);
                double value = 0.0;
                if (i >= 0 && i < op.n && j >= 0 && j < op.n
                    && zc >= 0 && zc < op.n) {
                    const double applied = mpk_stream_apply(
                        below, mid, above, lx, ly, ex, i, j, zc, op,
                        face_b, tail);
                    if constexpr (CHEBYSHEV) {
                        const double x =
                            (scale * applied - center * mid[slot])
                            / half_width;
                        value = level > 1 ? 2.0 * x - older[slot] : x;
                    } else {
                        const double shift =
                            shifts != nullptr ? shifts[level - 1] : 0.0;
                        value = inverse_normalization
                              * (scale * applied - shift * mid[slot]);
                    }
                }
                out[slot] = value;
                if (owned_plane && lx >= S && lx < S + kMpkStreamTileX
                    && ly >= S && ly < S + kMpkStreamTileY
                    && i < op.n && j < op.n)
                    B[static_cast<int64_t>(level) * ld
                      + mpk_field_index(i, j, zc, op)] = value;
            }
        }
        __syncthreads();
    }
}

/// Opt in to the shared memory the width needs, then launch the tail and field.
template <int S, bool CHEBYSHEV>
[[nodiscard]] inline cudaError_t launch_mpk_stream(
    const double* start, double* B, int64_t ld, const GpuPdeOperator& op,
    const double* face_b, double scale, const double* shifts,
    double inverse_normalization, double center, double half_width,
    int height, cudaStream_t stream, MpkSharedCarveout carveout)
{
#if defined(CAKSM_MPK_USE_RESTRICT)
    if (start == B || face_b == B || shifts == B)
        return cudaErrorInvalidValue;
#endif
    constexpr std::size_t shared_bytes =
        mpk_stream_shared_bytes(S, CHEBYSHEV);

    cudaError_t e = cudaFuncSetAttribute(
        mpk_stream_kernel<S, CHEBYSHEV>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes));
    if (e != cudaSuccess) return e;
    e = cudaFuncSetAttribute(
        mpk_stream_kernel<S, CHEBYSHEV>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        mpk_carveout_value(carveout));
    if (e != cudaSuccess) return e;

    if constexpr (CHEBYSHEV) {
        mpk_chebyshev_tail_chunk<<<1, 1, 0, stream>>>(
            nullptr, start, B, op.N, ld, S, scale, center, half_width, 0);
    } else {
        mpk_tail_powers<<<1, 1, 0, stream>>>(
            start, B, op.N, ld, S, scale, shifts, inverse_normalization);
    }
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;

    const int extent = mpk_stream_effective_height(op.n, height);
    const dim3 grid(
        static_cast<unsigned>(
            (op.n + kMpkStreamTileX - 1) / kMpkStreamTileX),
        static_cast<unsigned>(
            (op.n + kMpkStreamTileY - 1) / kMpkStreamTileY),
        static_cast<unsigned>((op.n + extent - 1) / extent));
    mpk_stream_kernel<S, CHEBYSHEV>
        <<<grid, kMpkStreamThreads, shared_bytes, stream>>>(
            start, B, ld, op, face_b, scale, shifts, inverse_normalization,
            center, half_width, extent);
    return cudaGetLastError();
}

/**
 * @brief Monomial basis [v, Av, ..., A^steps v] from the plane-streamed family.
 *
 * Same contract as gpu_pde_matrix_powers: steps+1 columns, one launch, no
 * reductions, and a width past kMpkMaxS rejected rather than clamped. height
 * zero streams the whole z extent in one segment, which is the lowest-redundancy
 * and lowest-parallelism end of the same tuning axis.
 */
[[nodiscard]] inline cudaError_t gpu_pde_stream_matrix_powers(
    const double* start, double* B, int64_t ld, int steps,
    const GpuPdeOperator& op, const double* face_b, double scale,
    int height = kMpkStreamHeight, cudaStream_t stream = nullptr,
    MpkSharedCarveout carveout = MpkSharedCarveout::Default)
{
    switch (steps) {
        case 1: return launch_mpk_stream<1, false>(start, B, ld, op, face_b, scale, nullptr, 1.0, 0.0, 1.0, height, stream, carveout);
        case 2: return launch_mpk_stream<2, false>(start, B, ld, op, face_b, scale, nullptr, 1.0, 0.0, 1.0, height, stream, carveout);
        case 3: return launch_mpk_stream<3, false>(start, B, ld, op, face_b, scale, nullptr, 1.0, 0.0, 1.0, height, stream, carveout);
        case 4: return launch_mpk_stream<4, false>(start, B, ld, op, face_b, scale, nullptr, 1.0, 0.0, 1.0, height, stream, carveout);
        case 5: return launch_mpk_stream<5, false>(start, B, ld, op, face_b, scale, nullptr, 1.0, 0.0, 1.0, height, stream, carveout);
        case 6: return launch_mpk_stream<6, false>(start, B, ld, op, face_b, scale, nullptr, 1.0, 0.0, 1.0, height, stream, carveout);
        default: return cudaErrorInvalidValue;
    }
}

/// @brief Newton basis on real-Leja shifts, streamed. Shifts must be in Leja order.
[[nodiscard]] inline cudaError_t gpu_pde_stream_newton_basis(
    const double* start, double* B, int64_t ld, int steps,
    const GpuPdeOperator& op, const double* face_b, double scale,
    const double* shifts, double normalization,
    int height = kMpkStreamHeight, cudaStream_t stream = nullptr,
    MpkSharedCarveout carveout = MpkSharedCarveout::Default)
{
    if (shifts == nullptr || !(normalization > 0.0))
        return cudaErrorInvalidValue;
    const double inverse = 1.0 / normalization;
    switch (steps) {
        case 1: return launch_mpk_stream<1, false>(start, B, ld, op, face_b, scale, shifts, inverse, 0.0, 1.0, height, stream, carveout);
        case 2: return launch_mpk_stream<2, false>(start, B, ld, op, face_b, scale, shifts, inverse, 0.0, 1.0, height, stream, carveout);
        case 3: return launch_mpk_stream<3, false>(start, B, ld, op, face_b, scale, shifts, inverse, 0.0, 1.0, height, stream, carveout);
        case 4: return launch_mpk_stream<4, false>(start, B, ld, op, face_b, scale, shifts, inverse, 0.0, 1.0, height, stream, carveout);
        case 5: return launch_mpk_stream<5, false>(start, B, ld, op, face_b, scale, shifts, inverse, 0.0, 1.0, height, stream, carveout);
        case 6: return launch_mpk_stream<6, false>(start, B, ld, op, face_b, scale, shifts, inverse, 0.0, 1.0, height, stream, carveout);
        default: return cudaErrorInvalidValue;
    }
}

/**
 * @brief Chebyshev basis, streamed, and unlike the full-volume family unchunked.
 *
 * The third live predecessor costs one extra plane per level here rather than a
 * third full tile, so the whole width fits one launch and the re-staging a chunk
 * boundary pays is gone. That difference is a property of the family and is
 * reported, not folded into a timing comparison against the chunked kernel.
 */
[[nodiscard]] inline cudaError_t gpu_pde_stream_chebyshev_basis(
    const double* start, double* B, int64_t ld, int steps,
    const GpuPdeOperator& op, const double* face_b, double scale,
    double center, double half_width, int height = kMpkStreamHeight,
    cudaStream_t stream = nullptr,
    MpkSharedCarveout carveout = MpkSharedCarveout::Default)
{
    if (!(half_width > 0.0)) return cudaErrorInvalidValue;
    switch (steps) {
        case 1: return launch_mpk_stream<1, true>(start, B, ld, op, face_b, scale, nullptr, 1.0, center, half_width, height, stream, carveout);
        case 2: return launch_mpk_stream<2, true>(start, B, ld, op, face_b, scale, nullptr, 1.0, center, half_width, height, stream, carveout);
        case 3: return launch_mpk_stream<3, true>(start, B, ld, op, face_b, scale, nullptr, 1.0, center, half_width, height, stream, carveout);
        case 4: return launch_mpk_stream<4, true>(start, B, ld, op, face_b, scale, nullptr, 1.0, center, half_width, height, stream, carveout);
        case 5: return launch_mpk_stream<5, true>(start, B, ld, op, face_b, scale, nullptr, 1.0, center, half_width, height, stream, carveout);
        case 6: return launch_mpk_stream<6, true>(start, B, ld, op, face_b, scale, nullptr, 1.0, center, half_width, height, stream, carveout);
        default: return cudaErrorInvalidValue;
    }
}

template <int S, bool CHEBYSHEV>
[[nodiscard]] inline cudaError_t mpk_stream_configure(
    MpkSharedCarveout carveout)
{
    cudaError_t error = cudaFuncSetAttribute(
        mpk_stream_kernel<S, CHEBYSHEV>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(mpk_stream_shared_bytes(S, CHEBYSHEV)));
    if (error != cudaSuccess) return error;
    return cudaFuncSetAttribute(
        mpk_stream_kernel<S, CHEBYSHEV>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        mpk_carveout_value(carveout));
}

template <int S, bool CHEBYSHEV>
[[nodiscard]] inline cudaError_t mpk_stream_occupancy(
    int* active_blocks, MpkSharedCarveout carveout)
{
    const cudaError_t configured = mpk_stream_configure<S, CHEBYSHEV>(carveout);
    if (configured != cudaSuccess) return configured;
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        active_blocks, mpk_stream_kernel<S, CHEBYSHEV>, kMpkStreamThreads,
        mpk_stream_shared_bytes(S, CHEBYSHEV));
}

template <int S, bool CHEBYSHEV>
[[nodiscard]] inline cudaError_t mpk_stream_function_attributes(
    cudaFuncAttributes* attributes, MpkSharedCarveout carveout)
{
    const cudaError_t configured = mpk_stream_configure<S, CHEBYSHEV>(carveout);
    if (configured != cudaSuccess) return configured;
    return cudaFuncGetAttributes(
        attributes, mpk_stream_kernel<S, CHEBYSHEV>);
}

/// Blocks the shared-memory request leaves resident per SM, by width and basis.
[[nodiscard]] inline cudaError_t gpu_pde_stream_occupancy(
    int steps, bool chebyshev, int* active_blocks,
    MpkSharedCarveout carveout = MpkSharedCarveout::Default)
{
    if (active_blocks == nullptr || steps < 1 || steps > kMpkMaxS)
        return cudaErrorInvalidValue;
    if (chebyshev) {
        switch (steps) {
            case 1: return mpk_stream_occupancy<1, true>(active_blocks, carveout);
            case 2: return mpk_stream_occupancy<2, true>(active_blocks, carveout);
            case 3: return mpk_stream_occupancy<3, true>(active_blocks, carveout);
            case 4: return mpk_stream_occupancy<4, true>(active_blocks, carveout);
            case 5: return mpk_stream_occupancy<5, true>(active_blocks, carveout);
            case 6: return mpk_stream_occupancy<6, true>(active_blocks, carveout);
            default: return cudaErrorInvalidValue;
        }
    }
    switch (steps) {
        case 1: return mpk_stream_occupancy<1, false>(active_blocks, carveout);
        case 2: return mpk_stream_occupancy<2, false>(active_blocks, carveout);
        case 3: return mpk_stream_occupancy<3, false>(active_blocks, carveout);
        case 4: return mpk_stream_occupancy<4, false>(active_blocks, carveout);
        case 5: return mpk_stream_occupancy<5, false>(active_blocks, carveout);
        case 6: return mpk_stream_occupancy<6, false>(active_blocks, carveout);
        default: return cudaErrorInvalidValue;
    }
}

/// Compiler and function preferences for the selected streamed recurrence.
[[nodiscard]] inline cudaError_t gpu_pde_stream_function_attributes(
    int steps, bool chebyshev, cudaFuncAttributes* attributes,
    MpkSharedCarveout carveout = MpkSharedCarveout::Default)
{
    if (attributes == nullptr || steps < 1 || steps > kMpkMaxS)
        return cudaErrorInvalidValue;
    if (chebyshev) {
        switch (steps) {
            case 1: return mpk_stream_function_attributes<1, true>(attributes, carveout);
            case 2: return mpk_stream_function_attributes<2, true>(attributes, carveout);
            case 3: return mpk_stream_function_attributes<3, true>(attributes, carveout);
            case 4: return mpk_stream_function_attributes<4, true>(attributes, carveout);
            case 5: return mpk_stream_function_attributes<5, true>(attributes, carveout);
            case 6: return mpk_stream_function_attributes<6, true>(attributes, carveout);
            default: return cudaErrorInvalidValue;
        }
    }
    switch (steps) {
        case 1: return mpk_stream_function_attributes<1, false>(attributes, carveout);
        case 2: return mpk_stream_function_attributes<2, false>(attributes, carveout);
        case 3: return mpk_stream_function_attributes<3, false>(attributes, carveout);
        case 4: return mpk_stream_function_attributes<4, false>(attributes, carveout);
        case 5: return mpk_stream_function_attributes<5, false>(attributes, carveout);
        case 6: return mpk_stream_function_attributes<6, false>(attributes, carveout);
        default: return cudaErrorInvalidValue;
    }
}
