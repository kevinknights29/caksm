/**
 * @file akx.hpp
 * @brief The cache-blocked matrix-powers (akx) kernel: the timed half of the vertical
 *        mechanism.
 *
 * The baseline builds the Krylov basis [v, Av, ..., A^s v] as s separate SpMVs, each
 * streaming the whole operator from DRAM, so above R_v = 1 the basis costs s trips to
 * DRAM. This kernel builds the same basis while streaming the operator once: it tiles
 * the rows into panels sized to the cache (Hoemmen's PA1, recomputing a 2*s*w-row halo
 * locally rather than communicating it) and computes all s powers on a panel while
 * that panel's slice of the operator stays resident, sized by mpk.hpp::mpk_panel_rows.
 * Output is bit-identical to the baseline, since only the memory access order changes,
 * never the arithmetic. s is always the caller-supplied certified block width
 * (mpk.hpp's operative_s), this kernel never picks it.
 *
 * Sequential, single-core, single-level: the minimum viable kernel that exhibits the
 * DRAM-traffic gap. OpenMP and multi-level variants belong to the timed sweep harness,
 * and must reproduce this kernel's output.
 *
 * @author Kevin Knights
 * @date 2026-07-19
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/Sparse>

/// Row-major sparse operator: rows are contiguous, so a row-range SpMV reads a contiguous
/// slice of the operator, the access pattern the cache-blocking depends on.
using SpMatRow = Eigen::SparseMatrix<double, Eigen::RowMajor>;

/// max |row - col| over the nonzeros: the operator's bandwidth, the halo's unit. Measured
/// here so the kernel is self-contained; mpk.hpp::predicted_bandwidth is the a-priori twin.
[[nodiscard]] inline int64_t operator_bandwidth(const SpMatRow& A) noexcept
{
    int64_t w = 0;
    for (int r = 0; r < A.outerSize(); ++r)
        for (SpMatRow::InnerIterator it(A, r); it; ++it)
            w = std::max(w, std::abs(static_cast<int64_t>(it.row())
                                   - static_cast<int64_t>(it.col())));
    return w;
}

/**
 * @brief One row's dot product: the single arithmetic kernel both arms share.
 *
 * Sharing it makes this an experiment about access order rather than about who wrote
 * the better SpMV: flop count and accumulation order are identical between baseline
 * and tiled, so every nanosecond of difference is memory behavior. It uses four
 * independent accumulators rather than one, since a single serial FMA chain is
 * latency-bound regardless of whether its operands come from L1 or DRAM (confirmed on
 * puffin: a no-halo diagnostic left GFLOP/s essentially unchanged, 2.47 vs 2.50),
 * and this changes only the summation order, not the flop count or bytes touched.
 *
 * Reads Eigen's raw CSR arrays directly (outerIndexPtr/innerIndexPtr/valuePtr) rather
 * than through SpMatRow::InnerIterator, avoiding its per-element indirection; the
 * matrix must be compressed (A.makeCompressed() called once when A was built).
 *
 * @param x_offset absolute row index that x[0] corresponds to (0 for a full-length vector,
 *                 the panel's expanded lower bound for a panel-local buffer).
 */
[[nodiscard]] inline double spmv_row_dot(const SpMatRow& A, int64_t i,
                                         const double* x, int64_t x_offset) noexcept
{
    const int* outer = A.outerIndexPtr();
    const int* cols   = A.innerIndexPtr();
    const double* vals = A.valuePtr();
    const int64_t k0 = outer[i];
    const int64_t k1 = outer[i + 1];

    double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0, acc3 = 0.0;
    int64_t k = k0;
    const int64_t k_unrolled = k0 + ((k1 - k0) / 4) * 4;
    for (; k < k_unrolled; k += 4) {
        acc0 += vals[k]     * x[static_cast<int64_t>(cols[k])     - x_offset];
        acc1 += vals[k + 1] * x[static_cast<int64_t>(cols[k + 1]) - x_offset];
        acc2 += vals[k + 2] * x[static_cast<int64_t>(cols[k + 2]) - x_offset];
        acc3 += vals[k + 3] * x[static_cast<int64_t>(cols[k + 3]) - x_offset];
    }
    double acc = (acc0 + acc1) + (acc2 + acc3);
    for (; k < k1; ++k)
        acc += vals[k] * x[static_cast<int64_t>(cols[k]) - x_offset];
    return acc;
}

/**
 * @brief Baseline monomial basis: s separate SpMVs, the operator streamed s times.
 *
 * The vertical experiment's control arm. Row-major, like the tiled kernel, and built
 * on the same spmv_row_dot, so the only difference the timer sees is access order,
 * never the layout or the arithmetic.
 */
[[nodiscard]] inline Eigen::MatrixXd spmv_chain(const SpMatRow& A, const Eigen::VectorXd& v,
                                                int s)
{
    if (s < 0) throw std::invalid_argument("spmv_chain: s must be >= 0");
    const int64_t N = A.rows();
    Eigen::MatrixXd B(N, static_cast<Eigen::Index>(s) + 1);
    B.col(0) = v;
    for (int k = 1; k <= s; ++k) {
        const double* prev = B.col(k - 1).data();
        double* cur = B.col(k).data();
        for (int64_t i = 0; i < N; ++i)
            cur[i] = spmv_row_dot(A, i, prev, 0);
    }
    return B;
}

/**
 * @brief Cache-blocked matrix-powers, writing directly into a caller-owned basis with no
 *        per-call heap allocation. The performance path (see tiled_matrix_powers()) below
 *        for a simple return-by-value wrapper meant for correctness tests, not timing.
 *
 * A version that allocates and returns its own N x (s+1) matrix looks clean, but a
 * multi-block caller (ca_arnoldi's s-step loop, or this file's own sweep harness) pays
 * a fresh allocation and a full-width copy on every block. This is large enough on puffin
 * to make the tiled arm slower than the baseline at every grid size. This function writes
 * straight into the caller's basis and reuses a caller-owned scratch buffer across
 * every block of a cycle, so after the first block nothing is allocated. `out.col(col0)`
 * must already hold a value valid across all N rows (level 0), which a multi-block
 * caller satisfies by construction since the panel loop covers the whole vector.
 *
 * Level 0 is read directly from `out`, levels 1 to s are computed into `scratch`, since
 * their halo is never part of the final basis. The interior [lo, hi) of every level is
 * copied from scratch into `out` once per panel, once per level: the one copy this
 * kernel pays that spmv_chain does not.
 *
 * @param out    the caller's basis matrix, rows() gives N. Columns [col0+1, col0+s] are
 *               written, column col0 is read only.
 * @param scratch caller-owned trapezoid workspace, resized here if too small and
 *               otherwise left alone, so a caller that pre-sizes it once before a
 *               repeats loop pays no allocation on any call.
 */
inline void tiled_matrix_powers_into(const SpMatRow& A, int s, int64_t tile_rows, int64_t w,
                                     Eigen::MatrixXd& out, Eigen::Index col0,
                                     Eigen::MatrixXd& scratch, bool debug_no_halo = false)
{
    if (s < 0) throw std::invalid_argument("tiled_matrix_powers_into: s must be >= 0");
    const int64_t N = out.rows();
    if (s == 0) return;
    if (tile_rows <= 0 || tile_rows > N) tile_rows = N;

    const int64_t max_W = std::min(N, tile_rows + 2 * static_cast<int64_t>(s) * w);
    if (scratch.rows() < max_W || scratch.cols() < static_cast<Eigen::Index>(s) + 1)
        scratch.resize(std::max<Eigen::Index>(scratch.rows(), max_W),
                       std::max<Eigen::Index>(scratch.cols(), static_cast<Eigen::Index>(s) + 1));

    const double* v_full = out.col(col0).data();   // level 0: already valid for every row

    for (int64_t lo = 0; lo < N; lo += tile_rows) {
        const int64_t hi  = std::min(N, lo + tile_rows);
        // elo is always the real halo-aware offset, even in debug_no_halo mode: it keeps
        // every scratch index non-negative and in-bounds. Only the iteration range
        // (klo, khi) below is what debug_no_halo touches.
        const int64_t elo = std::max<int64_t>(0, lo - static_cast<int64_t>(s) * w);

        for (int k = 1; k <= s; ++k) {
            // debug_no_halo is a diagnostic, never used on the reported crossover: it
            // forces every level to compute only the panel's own interior [lo, hi), giving
            // a wrong basis but a row-dot count of exactly N*s (no redundant halo work),
            // which isolates whether the tiled kernel's cost is the halo or the scalar
            // spmv_row_dot loop itself.
            const int64_t klo = debug_no_halo ? lo
                : std::max<int64_t>(0, lo - static_cast<int64_t>(s - k) * w);
            const int64_t khi = debug_no_halo ? hi
                : std::min<int64_t>(N, hi + static_cast<int64_t>(s - k) * w);
            // Level 1 reads the caller's full-length column directly (global indexing);
            // every later level reads the previous level out of this panel's own scratch
            // (elo-relative indexing, since scratch never holds more than one panel at a
            // time and different panels use different elo).
            const double* prev     = (k == 1) ? v_full : scratch.col(k - 1).data();
            const int64_t prev_off = (k == 1) ? 0       : elo;

            double* cur = scratch.col(k).data();
            for (int64_t i = klo; i < khi; ++i)
                cur[i - elo] = spmv_row_dot(A, i, prev, prev_off);
        }

        // The interior [lo, hi) of every level is the finished basis for these rows: the
        // one copy the algorithm cannot avoid (see the function doc above).
        for (int k = 1; k <= s; ++k)
            out.col(col0 + k).segment(lo, hi - lo) =
                scratch.col(k).segment(lo - elo, hi - lo);
    }
}

/**
 * @brief Cache-blocked matrix-powers: B = [v, Av, ..., A^s v], operator streamed once/panel.
 *
 * Bit-identical to spmv_chain(A, v, s); see the file header. A simple return-by-value
 * convenience wrapper around tiled_matrix_powers_into(), correct and what the
 * correctness tests exercise, but not the performance path: it allocates its own
 * output and scratch on every call. A caller building a multi-block basis (the sweep
 * harness, or ca_arnoldi's s-step loop) should call tiled_matrix_powers_into()
 * directly with a persistent scratch buffer, or pay a fresh allocation per block.
 *
 * @param tile_rows interior rows per panel. Size it with mpk.hpp::mpk_panel_rows so the
 *                  panel's operator slice fits the target cache; <= 0 or >= N means one
 *                  panel (no tiling, equivalent to spmv_chain in arithmetic).
 */
[[nodiscard]] inline Eigen::MatrixXd tiled_matrix_powers(const SpMatRow& A,
                                                         const Eigen::VectorXd& v,
                                                         int s, int64_t tile_rows, int64_t w)
{
    if (s < 0) throw std::invalid_argument("tiled_matrix_powers: s must be >= 0");
    Eigen::MatrixXd B(A.rows(), static_cast<Eigen::Index>(s) + 1);
    B.col(0) = v;
    Eigen::MatrixXd scratch;
    tiled_matrix_powers_into(A, s, tile_rows, w, B, 0, scratch);
    return B;
}
