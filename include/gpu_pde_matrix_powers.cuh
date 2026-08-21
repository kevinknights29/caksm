/**
 * @file gpu_pde_matrix_powers.cuh
 * @brief Matrix-free, shared-memory matrix-powers kernel for the 3-D pricing operator.
 *
 * @author Kevin Knights
 * @date 2026-07-26
 */
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

struct GpuPdeOperator {
    int n = 0;
    /// Doubles one field column occupies, padding included. This is where the
    /// augmented tail begins, so it is the physical length and not n-cubed
    /// whenever a padded row stride is compiled in.
    int64_t N = 0;
    /// Physical row stride in doubles. Zero means the unpadded stride n, which
    /// keeps an operator built by older code correct.
    int pitch_x = 0;
    int rainbow = 0;
    double reaction = 0.0;
    double drift[3]{};
    double diffusion[3]{};
    double mixed[3]{};
};

// The interior tile one thread block owns. The shared tile it must hold is
// (X+2S)(Y+2S)(Z+2S), so the ghost redundancy of a launch is fixed here rather
// than by the recurrence. The geometry is overridable at compile time so the
// same kernel can be swept over tile shapes without a second implementation;
// the production values are the defaults.
#ifndef CAKSM_MPK_BLOCK_X
#define CAKSM_MPK_BLOCK_X 8
#endif
#ifndef CAKSM_MPK_BLOCK_Y
#define CAKSM_MPK_BLOCK_Y 4
#endif
#ifndef CAKSM_MPK_BLOCK_Z
#define CAKSM_MPK_BLOCK_Z 4
#endif
#ifndef CAKSM_MPK_SHARED_PAD_X
#define CAKSM_MPK_SHARED_PAD_X 0
#endif

// Threads one block launches. Separate from the tile extents on purpose: the
// launch used to be fixed at 128, so changing the tile volume also changed the
// grid points each thread handled and the two effects could not be told apart.
// Sweeping this independently is what separates them.
#ifndef CAKSM_MPK_THREADS
#define CAKSM_MPK_THREADS 128
#endif

// Physical row stride, in doubles, that a field row is rounded up to. One is
// the packed layout, where the physical and logical indices coincide. Four,
// eight and sixteen are the 32, 64 and 128 byte alignments.
#ifndef CAKSM_MPK_PITCH_ALIGN
#define CAKSM_MPK_PITCH_ALIGN 1
#endif

// Drop the barrier between a basis write and the next recurrence step. The
// write only reads the buffer the pointer swap turns into the next step's
// input, and the next step writes the other buffer, so nothing overwrites the
// source while it is being read. Off by default until the arm passes its gates.
#if defined(CAKSM_MPK_ELIDE_BASIS_BARRIER)
inline constexpr bool kMpkElideBasisBarrier = true;
#else
inline constexpr bool kMpkElideBasisBarrier = false;
#endif

#if defined(CAKSM_MPK_USE_RESTRICT)
#define CAKSM_MPK_RESTRICT __restrict__
inline constexpr bool kMpkRestrictEnabled = true;
#else
#define CAKSM_MPK_RESTRICT
inline constexpr bool kMpkRestrictEnabled = false;
#endif

inline constexpr int kMpkBlockX = CAKSM_MPK_BLOCK_X;
inline constexpr int kMpkBlockY = CAKSM_MPK_BLOCK_Y;
inline constexpr int kMpkBlockZ = CAKSM_MPK_BLOCK_Z;
inline constexpr int kMpkSharedPadX = CAKSM_MPK_SHARED_PAD_X;
inline constexpr int kMpkThreadsPerBlock = CAKSM_MPK_THREADS;
inline constexpr int kMpkPitchAlignment = CAKSM_MPK_PITCH_ALIGN;
inline constexpr int kMpkMaxS = 6;
inline constexpr int kMpkPreferredS = 3;

static_assert(kMpkSharedPadX >= 0);
static_assert(kMpkThreadsPerBlock >= 32 && kMpkThreadsPerBlock <= 1024);
static_assert(kMpkThreadsPerBlock % 32 == 0,
              "a partial warp wastes a scheduler slot on every launch");
static_assert(kMpkPitchAlignment >= 1);

/// Physical row length in doubles: the logical extent rounded up to the
/// requested alignment. One returns n, so the packed layout is not a case.
[[nodiscard]] inline constexpr int mpk_pitch_x(int n)
{
    return ((n + kMpkPitchAlignment - 1) / kMpkPitchAlignment)
         * kMpkPitchAlignment;
}

/// Doubles one field occupies, padded rows included.
[[nodiscard]] inline constexpr int64_t mpk_physical_length(int n)
{
    return static_cast<int64_t>(mpk_pitch_x(n))
         * static_cast<int64_t>(n) * static_cast<int64_t>(n);
}

/// Doubles one basis column occupies: the padded field plus the augmented tail.
///
/// Every caller that sizes a basis column goes through this, so a padded layout
/// cannot be half-applied: a solver built on one alignment and a kernel built on
/// another would disagree about where the tail starts.
[[nodiscard]] inline constexpr int64_t mpk_column_length(int n)
{
    return mpk_physical_length(n) + 3;
}

/// Doubles of padding one field carries. Zero for the packed layout.
[[nodiscard]] inline constexpr int64_t mpk_pad_length(int n)
{
    return mpk_physical_length(n)
         - static_cast<int64_t>(n) * static_cast<int64_t>(n)
             * static_cast<int64_t>(n);
}

/// Physical offset of a logical point. The single definition of the layout:
/// conditional pitch arithmetic is not repeated at the call sites.
[[nodiscard]] __host__ __device__ __forceinline__ int64_t mpk_field_index(
    int i, int j, int k, int n, int pitch_x)
{
    const int64_t stride = pitch_x > 0 ? pitch_x : n;
    return (static_cast<int64_t>(k) * static_cast<int64_t>(n)
            + static_cast<int64_t>(j)) * stride
         + static_cast<int64_t>(i);
}

/// The same for an operator that already carries its stride.
[[nodiscard]] __host__ __device__ __forceinline__ int64_t mpk_field_index(
    int i, int j, int k, const GpuPdeOperator& op)
{
    return mpk_field_index(i, j, k, op.n, op.pitch_x);
}

/// Requested L1/shared-memory preference for a matrix-powers launch.
enum class MpkSharedCarveout {
    Default,
    Maximum
};

[[nodiscard]] inline constexpr int mpk_carveout_value(
    MpkSharedCarveout carveout)
{
    return carveout == MpkSharedCarveout::Maximum
        ? cudaSharedmemCarveoutMaxShared
        : cudaSharedmemCarveoutDefault;
}

/// Interior points one thread block writes per launch.
inline constexpr int kMpkInteriorPoints =
    kMpkBlockX * kMpkBlockY * kMpkBlockZ;

/// Doubles one shared tile holds at width S: the interior plus its ghost shell.
[[nodiscard]] inline constexpr int64_t mpk_tile_volume(int steps)
{
    return static_cast<int64_t>(kMpkBlockX + 2 * steps)
         * static_cast<int64_t>(kMpkBlockY + 2 * steps)
         * static_cast<int64_t>(kMpkBlockZ + 2 * steps);
}

/// Allocated shared-memory doubles, including optional row-stride padding.
[[nodiscard]] inline constexpr int64_t mpk_shared_tile_volume(int steps)
{
    return static_cast<int64_t>(kMpkBlockX + 2 * steps + kMpkSharedPadX)
         * static_cast<int64_t>(kMpkBlockY + 2 * steps)
         * static_cast<int64_t>(kMpkBlockZ + 2 * steps);
}

/// Shared tile doubles per interior point: what the tiling reads for what it writes.
[[nodiscard]] inline constexpr double mpk_redundancy(int steps)
{
    return static_cast<double>(mpk_tile_volume(steps))
         / static_cast<double>(kMpkInteriorPoints);
}

/// Thread blocks one launch dispatches over an n-cubed grid.
[[nodiscard]] inline constexpr int64_t mpk_grid_blocks(int n)
{
    return static_cast<int64_t>((n + kMpkBlockX - 1) / kMpkBlockX)
         * static_cast<int64_t>((n + kMpkBlockY - 1) / kMpkBlockY)
         * static_cast<int64_t>((n + kMpkBlockZ - 1) / kMpkBlockZ);
}

/// Ghost points staged per interior point, which is the redundancy the tiling
/// pays. mpk_redundancy above counts staged over interior; this counts the
/// excess alone, which is the quantity the tile filter is stated in.
[[nodiscard]] inline constexpr double mpk_redundant_fraction(int steps)
{
    return static_cast<double>(mpk_tile_volume(steps) - kMpkInteriorPoints)
         / static_cast<double>(kMpkInteriorPoints);
}

/// Interior points one thread writes per launch. Constant for a given tile and
/// thread count, and the term the old fixed-128 launch confounded with shape.
[[nodiscard]] inline constexpr double mpk_points_per_thread()
{
    return static_cast<double>(kMpkInteriorPoints)
         / static_cast<double>(kMpkThreadsPerBlock);
}

/// Block-wide barriers one block executes at this width: one after staging,
/// then one per recurrence step, plus the basis-write barrier when it is kept.
[[nodiscard]] inline constexpr int64_t mpk_barriers_per_block(int steps)
{
    return 1 + static_cast<int64_t>(steps)
             * (kMpkElideBasisBarrier ? 1 : 2);
}

/// Times the grid covers the device at a given residency, blocks over slots.
/// Below one the launch cannot fill the device whatever its occupancy.
[[nodiscard]] inline double mpk_waves_per_device(
    int n, int active_blocks_per_sm, int multiprocessors)
{
    if (active_blocks_per_sm <= 0 || multiprocessors <= 0) return 0.0;
    return static_cast<double>(mpk_grid_blocks(n))
         / static_cast<double>(
               static_cast<int64_t>(active_blocks_per_sm) * multiprocessors);
}

/// Which kernel builds the basis. Reported rather than inferred: a tuning row
/// that does not name its family cannot be compared with one that does.
enum class MpkKernelFamily {
    FullVolume,
    PlaneStreamed
};

[[nodiscard]] inline constexpr const char* mpk_family_name(
    MpkKernelFamily family)
{
    return family == MpkKernelFamily::PlaneStreamed
        ? "plane-streamed" : "full-volume";
}

/**
 * @brief What a launch actually was, for the benchmark and the solver to report.
 *
 * Every field here is a compile-time property of the binary, so a result row can
 * be attributed to a configuration without the runner having to remember which
 * executable it invoked. Kept flat and trivially copyable so printing it is one
 * statement rather than a formatting layer.
 */
struct MpkLaunchRecord {
    const char* family = mpk_family_name(MpkKernelFamily::FullVolume);
    int tile_x = kMpkBlockX;
    int tile_y = kMpkBlockY;
    int tile_z = kMpkBlockZ;
    int stream_height = 0;
    int threads_per_block = kMpkThreadsPerBlock;
    int shared_pad_x = kMpkSharedPadX;
    int pitch_alignment = kMpkPitchAlignment;
    int elide_basis_barrier = kMpkElideBasisBarrier ? 1 : 0;
    int use_restrict = kMpkRestrictEnabled ? 1 : 0;
    int64_t interior_points = kMpkInteriorPoints;
    int64_t staged_points = 0;
    double redundant_fraction = 0.0;
    double points_per_thread = mpk_points_per_thread();
    std::size_t dynamic_shared_bytes = 0;
};

/// The record for the full-volume family at one recurrence width.
[[nodiscard]] inline MpkLaunchRecord mpk_launch_record(
    int steps, std::size_t dynamic_shared_bytes)
{
    MpkLaunchRecord record;
    record.staged_points = steps >= 1 ? mpk_tile_volume(steps) : 0;
    record.redundant_fraction =
        steps >= 1 ? mpk_redundant_fraction(steps) : 0.0;
    record.dynamic_shared_bytes = dynamic_shared_bytes;
    return record;
}

/// Flat offset into a shared tile of extents (ex, ey, ez), x fastest.
__device__ __forceinline__ int mpk_index(int x, int y, int z, int ex, int ey)
{
    return (z * ey + y) * ex + x;
}

/**
 * The barrier that closes a recurrence step, after the basis column is written.
 *
 * Only the pointer swap follows it. The write phase reads the buffer the swap
 * turns into the next step's input, and the next step writes the other buffer,
 * so no thread can overwrite a source another thread is still reading. The
 * barrier is therefore removable, and the arm that removes it is compiled
 * separately rather than assumed correct. The condition is compile-time, so
 * every thread of the block takes the same path and the barrier stays uniform.
 */
__device__ __forceinline__ void mpk_step_barrier()
{
    if constexpr (!kMpkElideBasisBarrier) __syncthreads();
}

/// Flat offset into one n x n boundary face: the two coordinates that are not the face normal.
__device__ __forceinline__ int mpk_face_index(int dir, int i, int j, int k, int n)
{
    if (dir == 0) return j * n + k;
    if (dir == 1) return i * n + k;
    return i * n + j;
}

/**
 * One row of the skew-centered first-derivative operator T, as up to two (offset, weight) pairs.
 *
 * The device form of build_T. Interior rows are the centered [-1, +1]; a Rainbow top row becomes
 * the one-sided [-2, +2] that imposes zero gamma; a Basket top or bottom row simply drops the
 * neighbor that falls outside, which is the Dirichlet condition with the known value carried
 * separately in face_b. Only the mixed derivatives need T as an explicit row, since they apply
 * it twice; the pure drift and diffusion terms are folded into mpk_apply directly.
 */
__device__ __forceinline__ void mpk_t_row(int i, int n, bool rainbow,
                                           int delta[2], double value[2], int& count)
{
    count = 0;
    if (rainbow && i == n - 1) {
        delta[0] = -1;
        value[0] = -2.0;
        delta[1] = 0;
        value[1] = 2.0;
        count = 2;
        return;
    }
    if (i > 0) {
        delta[count] = -1;
        value[count] = -1.0;
        ++count;
    }
    if (i + 1 < n) {
        delta[count] = 1;
        value[count] = 1.0;
        ++count;
    }
}

/**
 * One row of the pricing operator applied at a single point, read from the shared tile.
 *
 * The whole matrix-free stencil: reaction, three drift and diffusion pairs along the axes, three
 * mixed second derivatives as tensor products of two T rows, and for Basket the known Dirichlet
 * contribution at an outer face, taken from face_b against the tail of the previous column.
 * (lx, ly, lz) index the tile; (i, j, k) are the global coordinates and decide which boundary
 * condition applies, so a slab launch can pass a global k that its tile coordinates do not carry.
 */
__device__ __forceinline__ double mpk_apply(
    const double* x, int lx, int ly, int lz, int ex, int ey,
    int i, int j, int k, const GpuPdeOperator& op,
    const double* face_b, const double* tail)
{
    const int coord[3] = {i, j, k};
    const int stride[3] = {1, ex, ex * ey};
    const int center = mpk_index(lx, ly, lz, ex, ey);
    double y = op.reaction * x[center]
             - op.diffusion[0] * 2.0 * x[center]
             - op.diffusion[1] * 2.0 * x[center]
             - op.diffusion[2] * 2.0 * x[center];

    for (int d = 0; d < 3; ++d) {
        const int q = coord[d];
        if (op.rainbow && q == op.n - 1) {
            y += op.drift[d] * (-2.0 * x[center - stride[d]] + 2.0 * x[center]);
            y += 2.0 * op.diffusion[d] * x[center];
        } else {
            if (q > 0)
                y += (op.diffusion[d] - op.drift[d]) * x[center - stride[d]];
            if (q + 1 < op.n)
                y += (op.diffusion[d] + op.drift[d]) * x[center + stride[d]];
        }
    }

    constexpr int pair_d[3] = {0, 0, 1};
    constexpr int pair_e[3] = {1, 2, 2};
    for (int p = 0; p < 3; ++p) {
        const int d = pair_d[p];
        const int e = pair_e[p];
        int dd[2], de[2], nd = 0, ne = 0;
        double td[2], te[2];
        mpk_t_row(coord[d], op.n, op.rainbow != 0, dd, td, nd);
        mpk_t_row(coord[e], op.n, op.rainbow != 0, de, te, ne);
        for (int a = 0; a < nd; ++a)
            for (int b = 0; b < ne; ++b)
                y += op.mixed[p] * td[a] * te[b]
                   * x[center + dd[a] * stride[d] + de[b] * stride[e]];
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
 * The 3-row augmented tail of the same recurrence, monomial or Newton.
 *
 * K is the nilpotent shift [[0,1,0],[0,0,1],[0,0,0]], so applying it is a shift of the three
 * components and needs no grid at all; one thread is cheaper than any parallel form. It runs
 * before the field kernel because mpk_apply reads tail column q-1 while writing field column q,
 * and it carries the same scale, shift and normalization so the two halves of a column stay
 * consistent. Rainbow starts the tail at zero and propagates zeros, so one launch path serves
 * both options.
 */
__global__ void mpk_tail_powers(const double* start, double* B, int64_t N,
                                int64_t ld, int steps, double scale,
                                const double* shifts, double inverse_normalization)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int c = 0; c < 3; ++c) B[N + c] = start[N + c];
    for (int q = 1; q <= steps; ++q) {
        const double* prev = B + static_cast<int64_t>(q - 1) * ld + N;
        double* next = B + static_cast<int64_t>(q) * ld + N;
        const double shift = shifts != nullptr ? shifts[q - 1] : 0.0;
        next[0] =
            inverse_normalization * (scale * prev[1] - shift * prev[0]);
        next[1] =
            inverse_normalization * (scale * prev[2] - shift * prev[1]);
        next[2] = -inverse_normalization * shift * prev[2];
    }
}

/**
 * @brief Build columns 0, ..., S of the basis in one launch, S fixed at compile time.
 *
 * The horizontal mechanism itself: each block stages an interior tile plus its S-deep ghost
 * shell in shared memory and runs the whole recurrence there, so S operator applications cost
 * one pass over device memory instead of S. What that trades away is redundancy, since every
 * ghost point is loaded by each block that borders it; mpk_redundancy above quantifies it and
 * the tile geometry, not the recurrence, is what sets it.
 *
 * The valid region of the tile retreats one plane per side per step, which is exactly consumed
 * by step S. With shifts null and inverse_normalization 1 the recurrence is monomial, A^q v;
 * otherwise it is the Newton form on the supplied shifts.
 */
template <int S>
__global__ void mpk_tiled_kernel(
    const double* CAKSM_MPK_RESTRICT start,
    double* CAKSM_MPK_RESTRICT B, int64_t ld,
    GpuPdeOperator op, const double* CAKSM_MPK_RESTRICT face_b,
    double scale, const double* CAKSM_MPK_RESTRICT shifts,
    double inverse_normalization)
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
    const int oz = static_cast<int>(blockIdx.z) * kMpkBlockZ;

    for (int q = tid; q < logical_volume;
         q += static_cast<int>(blockDim.x)) {
        const int lx = q % logical_ex;
        const int ly = (q / logical_ex) % ey;
        const int lz = q / (logical_ex * ey);
        const int shared_q = mpk_index(lx, ly, lz, ex, ey);
        const int i = ox + lx - S;
        const int j = oy + ly - S;
        const int k = oz + lz - S;
        if (i >= 0 && i < op.n && j >= 0 && j < op.n && k >= 0 && k < op.n) {
            const int64_t gid =
                mpk_field_index(i, j, k, op);
            in[shared_q] = start[gid];
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
        const int k = oz + tz;
        if (i < op.n && j < op.n && k < op.n) {
            const int64_t gid =
                mpk_field_index(i, j, k, op);
            B[gid] = in[mpk_index(tx + S, ty + S, tz + S, ex, ey)];
        }
    }

    for (int step = 1; step <= S; ++step) {
        const int lo = step;
        const int hi_x = logical_ex - step;
        const int hi_y = ey - step;
        const int hi_z = ez - step;
        const double* tail = B + static_cast<int64_t>(step - 1) * ld + op.N;

        for (int q = tid; q < logical_volume;
             q += static_cast<int>(blockDim.x)) {
            const int lx = q % logical_ex;
            const int ly = (q / logical_ex) % ey;
            const int lz = q / (logical_ex * ey);
            const int shared_q = mpk_index(lx, ly, lz, ex, ey);
            if (lx < lo || lx >= hi_x || ly < lo || ly >= hi_y || lz < lo || lz >= hi_z)
                continue;

            const int i = ox + lx - S;
            const int j = oy + ly - S;
            const int k = oz + lz - S;
            if (i >= 0 && i < op.n && j >= 0 && j < op.n && k >= 0 && k < op.n) {
                const double shift =
                    shifts != nullptr ? shifts[step - 1] : 0.0;
                out[shared_q] =
                    inverse_normalization
                    * (scale * mpk_apply(
                           in, lx, ly, lz, ex, ey, i, j, k, op, face_b, tail)
                       - shift * in[shared_q]);
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
            const int k = oz + tz;
            if (i < op.n && j < op.n && k < op.n) {
                const int64_t gid =
                    mpk_field_index(i, j, k, op);
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
 * The augmented tail under the Chebyshev three-term recurrence.
 *
 * Same role as mpk_tail_powers, but the recurrence needs two predecessors, so the chunk is given
 * the two columns preceding it. has_previous is false only for the first chunk of a basis, where
 * T_1 = X v has no T_-1 and the doubling drops out.
 */
__global__ void mpk_chebyshev_tail_chunk(
    const double* previous, const double* current, double* B,
    int64_t N, int64_t ld, int steps, double scale,
    double center, double half_width, int has_previous)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    double prev[3]{};
    double curr[3]{};
    for (int c = 0; c < 3; ++c) {
        if (has_previous != 0) prev[c] = previous[N + c];
        curr[c] = current[N + c];
        B[N + c] = curr[c];
    }

    for (int step = 1; step <= steps; ++step) {
        const double applied[3] = {
            scale * curr[1],
            scale * curr[2],
            0.0
        };
        double next[3]{};
        for (int c = 0; c < 3; ++c) {
            const double x = (applied[c] - center * curr[c]) / half_width;
            next[c] = (has_previous != 0 || step > 1) ? 2.0 * x - prev[c] : x;
            B[static_cast<int64_t>(step) * ld + N + c] = next[c];
        }
        for (int c = 0; c < 3; ++c) {
            prev[c] = curr[c];
            curr[c] = next[c];
        }
    }
}

/**
 * @brief Up to S columns of the Chebyshev basis, T_{k+1} = 2 X T_k - T_{k-1} on the shared tile.
 *
 * The same tiled recurrence as mpk_tiled_kernel with one structural difference: two predecessors
 * are live at once, so the block stages three tiles rather than two. That is why the width is
 * capped at kMpkPreferredS here and a wider request is chunked by gpu_pde_chebyshev_basis
 * instead, at the cost of re-staging the tile once per chunk.
 *
 * These pointers are intentionally not restrict-qualified. After the first
 * chunk, current and B begin at the same column, so qualifying them would make
 * the valid in-place chunk transition undefined.
 */
template <int S>
__global__ void mpk_chebyshev_chunk_kernel(
    const double* previous, const double* current, double* B, int64_t ld,
    GpuPdeOperator op, const double* face_b, double scale,
    double center, double half_width, int has_previous)
{
    static_assert(S >= 1 && S <= kMpkPreferredS);
    constexpr int logical_ex = kMpkBlockX + 2 * S;
    constexpr int ex = logical_ex + kMpkSharedPadX;
    constexpr int ey = kMpkBlockY + 2 * S;
    constexpr int ez = kMpkBlockZ + 2 * S;
    constexpr int logical_volume = logical_ex * ey * ez;
    constexpr int shared_volume = ex * ey * ez;

    extern __shared__ double storage[];
    double* prev = storage;
    double* curr = storage + shared_volume;
    double* next = storage + 2 * shared_volume;

    const int tid = static_cast<int>(threadIdx.x);
    const int ox = static_cast<int>(blockIdx.x) * kMpkBlockX;
    const int oy = static_cast<int>(blockIdx.y) * kMpkBlockY;
    const int oz = static_cast<int>(blockIdx.z) * kMpkBlockZ;

    for (int q = tid; q < logical_volume;
         q += static_cast<int>(blockDim.x)) {
        const int lx = q % logical_ex;
        const int ly = (q / logical_ex) % ey;
        const int lz = q / (logical_ex * ey);
        const int shared_q = mpk_index(lx, ly, lz, ex, ey);
        const int i = ox + lx - S;
        const int j = oy + ly - S;
        const int k = oz + lz - S;
        if (i >= 0 && i < op.n && j >= 0 && j < op.n && k >= 0 && k < op.n) {
            const int64_t gid =
                mpk_field_index(i, j, k, op);
            prev[shared_q] = has_previous != 0 ? previous[gid] : 0.0;
            curr[shared_q] = current[gid];
        } else {
            prev[shared_q] = 0.0;
            curr[shared_q] = 0.0;
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
        const int k = oz + tz;
        if (i < op.n && j < op.n && k < op.n) {
            const int64_t gid =
                mpk_field_index(i, j, k, op);
            B[gid] = curr[mpk_index(tx + S, ty + S, tz + S, ex, ey)];
        }
    }

    for (int step = 1; step <= S; ++step) {
        const int lo = step;
        const int hi_x = logical_ex - step;
        const int hi_y = ey - step;
        const int hi_z = ez - step;
        const double* tail = B + static_cast<int64_t>(step - 1) * ld + op.N;

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
            const int k = oz + lz - S;
            if (i >= 0 && i < op.n && j >= 0 && j < op.n && k >= 0 && k < op.n) {
                const double x =
                    (scale * mpk_apply(
                         curr, lx, ly, lz, ex, ey, i, j, k, op, face_b, tail)
                     - center * curr[shared_q])
                    / half_width;
                next[shared_q] =
                    (has_previous != 0 || step > 1)
                        ? 2.0 * x - prev[shared_q] : x;
            } else {
                next[shared_q] = 0.0;
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
            const int k = oz + tz;
            if (i < op.n && j < op.n && k < op.n) {
                const int64_t gid =
                    mpk_field_index(i, j, k, op);
                B[gid + static_cast<int64_t>(step) * ld] =
                    next[mpk_index(tx + S, ty + S, tz + S, ex, ey)];
            }
        }
        // Three buffers rather than two, but the same argument: the write reads
        // next, the swap makes next the new curr, and the following step writes
        // the buffer that was prev, which the write phase never touched.
        mpk_step_barrier();

        double* swap = prev;
        prev = curr;
        curr = next;
        next = swap;
    }
}

/// Opt in to the shared memory this width needs, then launch the tail and the field.
template <int S>
[[nodiscard]] inline cudaError_t launch_mpk_tiled(
    const double* start, double* B, int64_t ld, const GpuPdeOperator& op,
    const double* face_b, double scale, const double* shifts,
    double inverse_normalization, cudaStream_t stream,
    MpkSharedCarveout carveout)
{
#if defined(CAKSM_MPK_USE_RESTRICT)
    if (start == B || face_b == B || shifts == B)
        return cudaErrorInvalidValue;
#endif
    constexpr std::size_t volume =
        static_cast<std::size_t>(kMpkBlockX + 2 * S + kMpkSharedPadX)
        * static_cast<std::size_t>(kMpkBlockY + 2 * S)
        * static_cast<std::size_t>(kMpkBlockZ + 2 * S);
    constexpr std::size_t shared_bytes = 2 * volume * sizeof(double);

    cudaError_t e = cudaFuncSetAttribute(
        mpk_tiled_kernel<S>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes));
    if (e != cudaSuccess) return e;
    e = cudaFuncSetAttribute(
        mpk_tiled_kernel<S>, cudaFuncAttributePreferredSharedMemoryCarveout,
        mpk_carveout_value(carveout));
    if (e != cudaSuccess) return e;

    mpk_tail_powers<<<1, 1, 0, stream>>>(
        start, B, op.N, ld, S, scale, shifts, inverse_normalization);
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;

    const dim3 grid(
        static_cast<unsigned>((op.n + kMpkBlockX - 1) / kMpkBlockX),
        static_cast<unsigned>((op.n + kMpkBlockY - 1) / kMpkBlockY),
        static_cast<unsigned>((op.n + kMpkBlockZ - 1) / kMpkBlockZ));
    mpk_tiled_kernel<S><<<grid, kMpkThreadsPerBlock, shared_bytes, stream>>>(
        start, B, ld, op, face_b, scale, shifts, inverse_normalization);
    return cudaGetLastError();
}

/// The same for one Chebyshev chunk, which asks for three tiles instead of two.
template <int S>
[[nodiscard]] inline cudaError_t launch_mpk_chebyshev_chunk(
    const double* previous, const double* current, double* B, int64_t ld,
    const GpuPdeOperator& op, const double* face_b, double scale,
    double center, double half_width, bool has_previous,
    cudaStream_t stream, MpkSharedCarveout carveout)
{
    constexpr std::size_t volume =
        static_cast<std::size_t>(kMpkBlockX + 2 * S + kMpkSharedPadX)
        * static_cast<std::size_t>(kMpkBlockY + 2 * S)
        * static_cast<std::size_t>(kMpkBlockZ + 2 * S);
    constexpr std::size_t shared_bytes = 3 * volume * sizeof(double);

    cudaError_t e = cudaFuncSetAttribute(
        mpk_chebyshev_chunk_kernel<S>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes));
    if (e != cudaSuccess) return e;
    e = cudaFuncSetAttribute(
        mpk_chebyshev_chunk_kernel<S>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        mpk_carveout_value(carveout));
    if (e != cudaSuccess) return e;

    mpk_chebyshev_tail_chunk<<<1, 1, 0, stream>>>(
        previous, current, B, op.N, ld, S, scale, center, half_width,
        has_previous ? 1 : 0);
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;

    const dim3 grid(
        static_cast<unsigned>((op.n + kMpkBlockX - 1) / kMpkBlockX),
        static_cast<unsigned>((op.n + kMpkBlockY - 1) / kMpkBlockY),
        static_cast<unsigned>((op.n + kMpkBlockZ - 1) / kMpkBlockZ));
    mpk_chebyshev_chunk_kernel<S><<<grid, kMpkThreadsPerBlock, shared_bytes, stream>>>(
        previous, current, B, ld, op, face_b, scale,
        center, half_width, has_previous ? 1 : 0);
    return cudaGetLastError();
}

/**
 * @brief Monomial basis [v, Av, ..., A^steps v]: steps+1 columns, one launch, zero reductions.
 *
 * The production entry point. The switch exists because the tile extents and recurrence bounds
 * must be compile-time constants, so a run-time width can only be dispatched onto an
 * instantiation; an unsupported width is rejected rather than clamped, since a silently narrowed
 * basis would return fewer columns than the caller sized its workspace for.
 */
[[nodiscard]] inline cudaError_t gpu_pde_matrix_powers(
    const double* start, double* B, int64_t ld, int steps,
    const GpuPdeOperator& op, const double* face_b, double scale,
    cudaStream_t stream = nullptr,
    MpkSharedCarveout carveout = MpkSharedCarveout::Default)
{
    switch (steps) {
        case 1: return launch_mpk_tiled<1>(start, B, ld, op, face_b, scale, nullptr, 1.0, stream, carveout);
        case 2: return launch_mpk_tiled<2>(start, B, ld, op, face_b, scale, nullptr, 1.0, stream, carveout);
        case 3: return launch_mpk_tiled<3>(start, B, ld, op, face_b, scale, nullptr, 1.0, stream, carveout);
        case 4: return launch_mpk_tiled<4>(start, B, ld, op, face_b, scale, nullptr, 1.0, stream, carveout);
        case 5: return launch_mpk_tiled<5>(start, B, ld, op, face_b, scale, nullptr, 1.0, stream, carveout);
        case 6: return launch_mpk_tiled<6>(start, B, ld, op, face_b, scale, nullptr, 1.0, stream, carveout);
        default: return cudaErrorInvalidValue;
    }
}

/**
 * @brief Newton basis on real-Leja shifts: column q+1 is (A - shift_q I) column q, normalized.
 *
 * Same kernel and same cost as the monomial basis, differing only in the per-step shift and the
 * division by the interval half-width, which is what keeps kappa(B) from growing geometrically
 * in the width. The shifts must be in Leja order, as real_leja_shifts returns them.
 */
[[nodiscard]] inline cudaError_t gpu_pde_newton_basis(
    const double* start, double* B, int64_t ld, int steps,
    const GpuPdeOperator& op, const double* face_b, double scale,
    const double* shifts, double normalization,
    cudaStream_t stream = nullptr,
    MpkSharedCarveout carveout = MpkSharedCarveout::Default)
{
    if (shifts == nullptr || !(normalization > 0.0)) return cudaErrorInvalidValue;
    const double inverse_normalization = 1.0 / normalization;
    switch (steps) {
        case 1: return launch_mpk_tiled<1>(start, B, ld, op, face_b, scale, shifts, inverse_normalization, stream, carveout);
        case 2: return launch_mpk_tiled<2>(start, B, ld, op, face_b, scale, shifts, inverse_normalization, stream, carveout);
        case 3: return launch_mpk_tiled<3>(start, B, ld, op, face_b, scale, shifts, inverse_normalization, stream, carveout);
        case 4: return launch_mpk_tiled<4>(start, B, ld, op, face_b, scale, shifts, inverse_normalization, stream, carveout);
        case 5: return launch_mpk_tiled<5>(start, B, ld, op, face_b, scale, shifts, inverse_normalization, stream, carveout);
        case 6: return launch_mpk_tiled<6>(start, B, ld, op, face_b, scale, shifts, inverse_normalization, stream, carveout);
        default: return cudaErrorInvalidValue;
    }
}

/**
 * @brief Chebyshev basis, built in chunks of at most kMpkPreferredS columns.
 *
 * Three live tiles instead of two put a lower ceiling on the width one launch can hold, so wide
 * requests are walked in chunks, each seeded from the two columns the previous one produced.
 * Every chunk boundary re-stages the tile from device memory, so this basis pays more traffic
 * than the monomial and Newton forms at the same width, which is the cost of its conditioning.
 */
[[nodiscard]] inline cudaError_t gpu_pde_chebyshev_basis(
    const double* start, double* B, int64_t ld, int steps,
    const GpuPdeOperator& op, const double* face_b, double scale,
    double center, double half_width, cudaStream_t stream = nullptr,
    MpkSharedCarveout carveout = MpkSharedCarveout::Default)
{
    if (steps < 1 || steps > kMpkMaxS || !(half_width > 0.0))
        return cudaErrorInvalidValue;

    int offset = 0;
    while (offset < steps) {
        const int remaining = steps - offset;
        const int chunk =
            remaining < kMpkPreferredS ? remaining : kMpkPreferredS;
        const bool has_previous = offset > 0;
        const double* previous =
            has_previous ? B + static_cast<int64_t>(offset - 1) * ld : nullptr;
        const double* current =
            has_previous ? B + static_cast<int64_t>(offset) * ld : start;
        double* output = B + static_cast<int64_t>(offset) * ld;

        cudaError_t e = cudaSuccess;
        switch (chunk) {
            case 1:
                e = launch_mpk_chebyshev_chunk<1>(
                    previous, current, output, ld, op, face_b, scale,
                    center, half_width, has_previous, stream, carveout);
                break;
            case 2:
                e = launch_mpk_chebyshev_chunk<2>(
                    previous, current, output, ld, op, face_b, scale,
                    center, half_width, has_previous, stream, carveout);
                break;
            case 3:
                e = launch_mpk_chebyshev_chunk<3>(
                    previous, current, output, ld, op, face_b, scale,
                    center, half_width, has_previous, stream, carveout);
                break;
            default:
                return cudaErrorInvalidValue;
        }
        if (e != cudaSuccess) return e;
        offset += chunk;
    }
    return cudaSuccess;
}

/// Dynamic shared memory one block requests at this width: two tiles, in and out. 0 if invalid.
[[nodiscard]] inline std::size_t gpu_pde_matrix_powers_shared_bytes(int steps)
{
    if (steps < 1 || steps > kMpkMaxS) return 0;
    const std::size_t ex = static_cast<std::size_t>(
        kMpkBlockX + 2 * steps + kMpkSharedPadX);
    const std::size_t ey = static_cast<std::size_t>(kMpkBlockY + 2 * steps);
    const std::size_t ez = static_cast<std::size_t>(kMpkBlockZ + 2 * steps);
    return 2 * ex * ey * ez * sizeof(double);
}

/// The same for Chebyshev: three tiles, sized by the chunk width rather than the full width.
[[nodiscard]] inline std::size_t gpu_pde_chebyshev_shared_bytes(int steps)
{
    if (steps < 1 || steps > kMpkMaxS) return 0;
    const std::size_t chunk =
        static_cast<std::size_t>(steps < kMpkPreferredS ? steps : kMpkPreferredS);
    const std::size_t ex = static_cast<std::size_t>(
        kMpkBlockX + kMpkSharedPadX) + 2 * chunk;
    const std::size_t ey = static_cast<std::size_t>(kMpkBlockY) + 2 * chunk;
    const std::size_t ez = static_cast<std::size_t>(kMpkBlockZ) + 2 * chunk;
    return 3 * ex * ey * ez * sizeof(double);
}

/// Blocks the shared-memory request leaves resident per SM at this width.
template <int S>
[[nodiscard]] inline cudaError_t mpk_configure(
    MpkSharedCarveout carveout)
{
    cudaError_t error = cudaFuncSetAttribute(
        mpk_tiled_kernel<S>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(gpu_pde_matrix_powers_shared_bytes(S)));
    if (error != cudaSuccess) return error;
    return cudaFuncSetAttribute(
        mpk_tiled_kernel<S>, cudaFuncAttributePreferredSharedMemoryCarveout,
        mpk_carveout_value(carveout));
}

template <int S>
[[nodiscard]] inline cudaError_t mpk_occupancy(
    int* active_blocks, MpkSharedCarveout carveout)
{
    const cudaError_t configured = mpk_configure<S>(carveout);
    if (configured != cudaSuccess) return configured;
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        active_blocks, mpk_tiled_kernel<S>, kMpkThreadsPerBlock,
        gpu_pde_matrix_powers_shared_bytes(S));
}

/// The same for the Chebyshev chunk kernel, whose third tile costs it residency.
template <int S>
[[nodiscard]] inline cudaError_t mpk_chebyshev_configure(
    MpkSharedCarveout carveout)
{
    cudaError_t error = cudaFuncSetAttribute(
        mpk_chebyshev_chunk_kernel<S>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(gpu_pde_chebyshev_shared_bytes(S)));
    if (error != cudaSuccess) return error;
    return cudaFuncSetAttribute(
        mpk_chebyshev_chunk_kernel<S>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        mpk_carveout_value(carveout));
}

template <int S>
[[nodiscard]] inline cudaError_t mpk_chebyshev_occupancy(
    int* active_blocks, MpkSharedCarveout carveout)
{
    const cudaError_t configured = mpk_chebyshev_configure<S>(carveout);
    if (configured != cudaSuccess) return configured;
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        active_blocks, mpk_chebyshev_chunk_kernel<S>, kMpkThreadsPerBlock,
        gpu_pde_chebyshev_shared_bytes(S));
}

/**
 * @brief Occupancy for whichever basis is selected, for the vertical account to report.
 *
 * Reported rather than optimized against: the tile sweep measured lower occupancy running
 * faster, so this is one term in the roofline record, not a target.
 */
[[nodiscard]] inline cudaError_t gpu_pde_basis_occupancy(
    int steps, bool chebyshev, int* active_blocks,
    MpkSharedCarveout carveout = MpkSharedCarveout::Default)
{
    if (active_blocks == nullptr || steps < 1 || steps > kMpkMaxS)
        return cudaErrorInvalidValue;
    if (chebyshev) {
        const int chunk =
            steps < kMpkPreferredS ? steps : kMpkPreferredS;
        switch (chunk) {
            case 1: return mpk_chebyshev_occupancy<1>(active_blocks, carveout);
            case 2: return mpk_chebyshev_occupancy<2>(active_blocks, carveout);
            case 3: return mpk_chebyshev_occupancy<3>(active_blocks, carveout);
            default: return cudaErrorInvalidValue;
        }
    }
    switch (steps) {
        case 1: return mpk_occupancy<1>(active_blocks, carveout);
        case 2: return mpk_occupancy<2>(active_blocks, carveout);
        case 3: return mpk_occupancy<3>(active_blocks, carveout);
        case 4: return mpk_occupancy<4>(active_blocks, carveout);
        case 5: return mpk_occupancy<5>(active_blocks, carveout);
        case 6: return mpk_occupancy<6>(active_blocks, carveout);
        default: return cudaErrorInvalidValue;
    }
}

template <int S>
[[nodiscard]] inline cudaError_t mpk_function_attributes(
    cudaFuncAttributes* attributes, MpkSharedCarveout carveout)
{
    const cudaError_t configured = mpk_configure<S>(carveout);
    if (configured != cudaSuccess) return configured;
    return cudaFuncGetAttributes(attributes, mpk_tiled_kernel<S>);
}

template <int S>
[[nodiscard]] inline cudaError_t mpk_chebyshev_function_attributes(
    cudaFuncAttributes* attributes, MpkSharedCarveout carveout)
{
    const cudaError_t configured = mpk_chebyshev_configure<S>(carveout);
    if (configured != cudaSuccess) return configured;
    return cudaFuncGetAttributes(attributes, mpk_chebyshev_chunk_kernel<S>);
}

/// Compiler and function preferences for the selected recurrence kernel.
[[nodiscard]] inline cudaError_t gpu_pde_basis_function_attributes(
    int steps, bool chebyshev, cudaFuncAttributes* attributes,
    MpkSharedCarveout carveout = MpkSharedCarveout::Default)
{
    if (attributes == nullptr || steps < 1 || steps > kMpkMaxS)
        return cudaErrorInvalidValue;
    if (chebyshev) {
        const int chunk =
            steps < kMpkPreferredS ? steps : kMpkPreferredS;
        switch (chunk) {
            case 1:
                return mpk_chebyshev_function_attributes<1>(
                    attributes, carveout);
            case 2:
                return mpk_chebyshev_function_attributes<2>(
                    attributes, carveout);
            case 3:
                return mpk_chebyshev_function_attributes<3>(
                    attributes, carveout);
            default:
                return cudaErrorInvalidValue;
        }
    }
    switch (steps) {
        case 1: return mpk_function_attributes<1>(attributes, carveout);
        case 2: return mpk_function_attributes<2>(attributes, carveout);
        case 3: return mpk_function_attributes<3>(attributes, carveout);
        case 4: return mpk_function_attributes<4>(attributes, carveout);
        case 5: return mpk_function_attributes<5>(attributes, carveout);
        case 6: return mpk_function_attributes<6>(attributes, carveout);
        default: return cudaErrorInvalidValue;
    }
}
