/**
 * @file mpk.hpp
 * @brief A-priori model of the tiled matrix-powers kernel: halo, panel height, and the
 *        two roofs on s.
 *
 * Tiling is what earns the vertical axis: a naive kernel just streams the operator s
 * times, so instead the rows are split into panels and all s powers are computed while
 * a panel stays resident, carrying an s*w-wide halo (w = operator bandwidth) shared
 * within a band, so it costs footprint, not arithmetic. Two independent roofs bound s:
 * a capacity roof (panel stops fitting or halo stops being thin) and a numerical roof
 * (certified s_max), the operative s is the smaller.
 *
 * The pattern switch decides a priori from the scatter knob b whether tiling can pay: a
 * thin halo at b = 1 (banded), a whole-vector halo at b = N (scattered, falls back to
 * naive s-SpMV). Everything here is a predictor, measured_bandwidth() only checks
 * predicted_bandwidth().
 *
 * @author Kevin Knights
 * @date 2026-07-17
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>

#include "machine.hpp"
#include "regime.hpp"
#include "synthetic.hpp"

// Operator bandwidth: the quantity the halo is measured in
/**
 * @brief Widest stencil offset of the unpermuted operator, in rows.
 *
 * The dD Kronecker sum couples each axis at stride n1^d, so the widest axial offset is
 * n1^(dim-1). The correlation cross term couples axes 0 and 1 diagonally, reaching stride
 * 1 + n1, which only matters in 2D; at dim >= 3 the axial stride already dominates it.
 */
[[nodiscard]] inline int64_t stencil_bandwidth(const SyntheticSpec& sp) noexcept
{
    int64_t max_stride = 1;
    for (int d = 1; d < sp.dim; ++d) max_stride *= sp.n1;   // n1^(dim-1)
    if (sp.correlation != 0.0 && sp.dim >= 2)
        max_stride = std::max(max_stride, int64_t{1} + sp.n1);
    return max_stride;
}

/**
 * @brief A-priori bandwidth w of the operator after the scatter knob is applied.
 *
 * The knob shuffles indices within contiguous blocks of b, so a stencil neighbor at
 * offset st lands within st + 2b of its row: w(b) <= max_stride + 2b, capped at n-1. At
 * b = 1 this is the stencil bandwidth. At b = n it saturates at n-1 (the halo becomes the
 * whole vector, which is why the tiled kernel has nothing to tile there). An upper bound,
 * the safe direction for a switch: it can only refuse to tile where tiling was viable,
 * never tile where it cannot pay. measured_bandwidth() checks it.
 */
[[nodiscard]] inline int64_t predicted_bandwidth(const SyntheticSpec& sp) noexcept
{
    const int64_t n  = synthetic_dimension(sp);
    const int64_t w0 = stencil_bandwidth(sp);
    if (sp.scatter_block <= 1) return std::min(w0, n - 1);   // identity permutation
    const int64_t b = std::min(sp.scatter_block, n);
    return std::min(n - 1, w0 + 2 * b);
}

/**
 * @brief Bandwidth measured from an assembled operator: max |row - col| over nonzeros.
 *
 * An outcome, not a placement input: it checks predicted_bandwidth()'s bound against the
 * matrix actually built.
 */
[[nodiscard]] inline int64_t measured_bandwidth(const SpMatS& A) noexcept
{
    int64_t w = 0;
    for (int k = 0; k < A.outerSize(); ++k)
        for (SpMatS::InnerIterator it(A, k); it; ++it)
            w = std::max(w, std::abs(static_cast<int64_t>(it.row())
                                   - static_cast<int64_t>(it.col())));
    return w;
}

// Tiling level: the capacity a panel is sized against
/**
 * @brief Which cache level a panel is tiled to fit.
 *
 * L3 is the coarse boundary R_v's denominator is built from; L2 is the finer one, giving
 * the crossover a second measured point.
 */
enum class TileLevel { L2, L3 };

[[nodiscard]] inline const char* tile_level_name(TileLevel l) noexcept
{
    return l == TileLevel::L2 ? "L2" : "L3";
}

/**
 * @brief Bytes one core may hold at a tiling level.
 *
 * L2 is private, so a core commands the whole thing. An L3 slice is shared by the cores on
 * its CCX, all running panels at once, so a core commands its share, not the slice.
 *
 * @param cores_sharing override for how many cores contend for the slice, pass 1 for a
 *        sequential harness (a lone core owns the whole slice) or 0 for the machine's
 *        full-occupancy default.
 */
[[nodiscard]] inline double tile_capacity_bytes(const Machine& mc, TileLevel level,
                                                int cores_sharing = 0) noexcept
{
    if (level == TileLevel::L2) return static_cast<double>(mc.l2_bytes_core);
    const int share = (cores_sharing > 0) ? cores_sharing : mc.cores_per_llc_slice;
    return static_cast<double>(mc.llc_slice_bytes) / static_cast<double>(share);
}

// The panel's footprint
/**
 * @brief Bytes one panel of `rows` interior rows must hold resident to build s powers.
 *
 * Two terms: the operator over the widest dependency set (interior grown by s*w each side,
 * charged once), and the levels, which shrink as the recurrence climbs so they sum to a
 * trapezoid (s+1)*rows + w*s*(s+1), not (s+1)*expanded. Charging the box would over-count
 * the halo about twofold.
 */
[[nodiscard]] inline double mpk_panel_bytes(int64_t rows, int s, int64_t w, int64_t n,
                                            double nnz_per_row) noexcept
{
    const double ds = static_cast<double>(s);
    const double dw = static_cast<double>(w);
    const double dr = static_cast<double>(std::min(rows, n));
    const double expanded = static_cast<double>(std::min(n, rows + 2 * static_cast<int64_t>(s) * w));

    const double op     = expanded * (nnz_per_row * 12.0 + 4.0);
    const double levels = ((ds + 1.0) * dr + dw * ds * (ds + 1.0)) * 8.0;
    return op + levels;
}

/**
 * @brief Halo rows per interior row: 2*s*w / rows.
 *
 * The thinness measure the pattern switch turns on. Not a redundant-flop multiplier: halos
 * are shared between panels, so they cost footprint. Above 1 the panel is mostly halo.
 */
[[nodiscard]] inline double mpk_halo_ratio(int64_t rows, int s, int64_t w) noexcept
{
    if (rows <= 0) return std::numeric_limits<double>::infinity();
    return (2.0 * static_cast<double>(s) * static_cast<double>(w))
         / static_cast<double>(rows);
}

/**
 * @brief Halo may not exceed the interior it serves.
 *
 * A chosen threshold, reported rather than derived: at a ratio of 1 a panel carries as many
 * halo rows as it writes and the band-boundary barriers stop amortizing.
 */
inline constexpr double kThinHaloRatio = 1.0;

/// Fraction of the tiling level a panel may claim. Sizing to 100% of the cache guarantees
/// thrashing, since the panel then has no room for the streaming basis columns, the output
/// matrix, or associativity conflicts; measured on puffin, a full-capacity panel (16.2 MiB
/// against a 16 MiB slice) ran slower than the baseline. Half is the usual rule of thumb.
inline constexpr double kPanelFillFactor = 0.5;

/**
 * @brief Tallest panel that still fits one core's share of `level`, in interior rows.
 *
 * A taller panel is better on the halo ratio and worse on residency, so the optimum is the
 * tallest that fits; mpk_panel_bytes is linear in the interior, so it inverts in closed form.
 * Returns 0 when the halo alone overflows the capacity: callers must read 0 as "do not tile".
 */
[[nodiscard]] inline int64_t mpk_panel_rows(const Machine& mc, TileLevel level, int s,
                                            int64_t w, int64_t n, double nnz_per_row,
                                            int cores_sharing = 0) noexcept
{
    const double capacity =
        kPanelFillFactor * tile_capacity_bytes(mc, level, cores_sharing);
    const double ds  = static_cast<double>(s);
    const double dw  = static_cast<double>(w);
    const double row = nnz_per_row * 12.0 + 4.0;

    const double coeff = row + 8.0 * (ds + 1.0);
    const double fixed = 2.0 * ds * dw * row + 8.0 * dw * ds * (ds + 1.0);

    const double rows = (capacity - fixed) / coeff;
    if (!(rows >= 1.0)) return 0;                     // the halo alone overflows the level
    return std::min(n, static_cast<int64_t>(rows));
}

// The vertical coordinate of the tiled arm
/**
 * @brief Is a panel resident at its tiling level? Predicted crossover at 1.
 *
 * The baseline's R_v asks whether the whole Arnoldi working set fits; the tiled kernel's
 * reuse window is one panel, so it gets its own coordinate on the same frame. Below 1 the
 * panel is resident and matrix-powers hides the traffic; above 1 the panel spills.
 *
 * mpk_panel_rows() sizes the panel to fit by construction, so a kernel free to choose its
 * height has no crossover; it appears only when the height is fixed (pinned to a band or
 * across an N-sweep), which is why the sweep pins it and reports the height.
 */
[[nodiscard]] inline double mpk_tiled_R_v(const Machine& mc, TileLevel level, int64_t rows,
                                          int s, int64_t w, int64_t n,
                                          double nnz_per_row, int cores_sharing = 0) noexcept
{
    return mpk_panel_bytes(rows, s, w, n, nnz_per_row)
         / tile_capacity_bytes(mc, level, cores_sharing);
}

// The two roofs on s
/**
 * @brief Capacity roof: largest s that still admits a tiling panel with a thin halo.
 *
 * Walks s upward until no panel fits or the halo stops being thin. Independent of the
 * numerical certificate.
 */
[[nodiscard]] inline int s_ghost_max(const Machine& mc, TileLevel level, int64_t w, int64_t n,
                                     double nnz_per_row, int s_ceiling,
                                     int cores_sharing = 0) noexcept
{
    int best = 0;
    for (int s = 1; s <= s_ceiling; ++s) {
        const int64_t rows = mpk_panel_rows(mc, level, s, w, n, nnz_per_row, cores_sharing);
        if (rows <= 0) break;                                     // halo overflows the level
        if (mpk_halo_ratio(rows, s, w) > kThinHaloRatio) break;   // halo no longer thin
        best = s;
    }
    return best;
}

/**
 * @brief The operative block width: min(capacity roof, numerical certificate).
 *
 * Both are upper bounds and neither implies the other, so the kernel obeys the tighter one.
 */
[[nodiscard]] inline int operative_s(int s_capacity, int s_certified) noexcept
{
    return std::max(1, std::min(s_capacity, s_certified));
}

// The pattern switch
/// Which form of the matrix-powers arm a point runs. Decided a priori, from b.
enum class MpkForm {
    TILED,   ///< contiguous row panels, shared halos: the vertical mechanism is available
    NAIVE,   ///< s successive SpMVs: the fallback, and a finding in its own right
};

[[nodiscard]] inline const char* mpk_form_name(MpkForm f) noexcept
{
    return f == MpkForm::TILED ? "tiled" : "naive";
}

/// The switch's inputs and its verdict, recorded per point so the choice is auditable.
struct MpkPlan {
    MpkForm form       = MpkForm::NAIVE;
    TileLevel level    = TileLevel::L3;
    int64_t w          = 0;    ///< predicted operator bandwidth
    int64_t panel_rows = 0;    ///< interior rows per panel; 0 when not tiling
    int     s          = 0;    ///< operative block width actually used
    int     s_capacity = 0;    ///< the capacity roof on s
    int     s_certified = 0;   ///< the numerical roof on s, as supplied
    double  halo_ratio = 0.0;  ///< halo rows per interior row at that panel
    double  tiled_rv   = 0.0;  ///< the tiled arm's vertical coordinate
    bool    capacity_binds = false;  ///< which roof picked s -- a finding either way
    const char* reason = "";   ///< why the switch went the way it did
};

/**
 * @brief Decide, before running, whether the tiled kernel can pay on this operator.
 *
 * Two ways to fall back: "no panel fits" (the halo overflows the tiling level) and "halo
 * not thin" (a panel exists but carries more halo than interior). The scattered arm hits
 * the first at every working-set size, which is the expected verdict there, not a defect.
 *
 * @param s_certified the certified block width from the spectrum/ensemble. The capacity
 *                    roof is computed here; the tighter of the two runs.
 */
[[nodiscard]] inline MpkPlan plan_mpk(const Machine& mc, TileLevel level,
                                      const SyntheticSpec& sp, int s_certified,
                                      double nnz_per_row, int s_ceiling = 64,
                                      int cores_sharing = 0)
{
    const int64_t n = synthetic_dimension(sp);
    MpkPlan p;
    p.level       = level;
    p.w           = predicted_bandwidth(sp);
    p.s_certified = s_certified;
    p.s_capacity  = s_ghost_max(mc, level, p.w, n, nnz_per_row, s_ceiling, cores_sharing);
    p.s           = operative_s(p.s_capacity, s_certified);
    p.capacity_binds = (p.s_capacity < s_certified);

    p.panel_rows = mpk_panel_rows(mc, level, p.s, p.w, n, nnz_per_row, cores_sharing);
    if (p.panel_rows <= 0) {
        p.form = MpkForm::NAIVE;
        p.reason = "no panel fits";
        p.halo_ratio = std::numeric_limits<double>::infinity();
        return p;
    }

    p.halo_ratio = mpk_halo_ratio(p.panel_rows, p.s, p.w);
    p.tiled_rv   = mpk_tiled_R_v(mc, level, p.panel_rows, p.s, p.w, n, nnz_per_row,
                                 cores_sharing);

    if (p.halo_ratio > kThinHaloRatio) {
        p.form = MpkForm::NAIVE;
        p.reason = "halo not thin";
        return p;
    }

    p.form   = MpkForm::TILED;
    p.reason = "panel resident, halo thin";
    return p;
}
