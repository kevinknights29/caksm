/**
 * @file gpu_regime.hpp
 * @brief The dimensionless coordinates on a GPU, the roofline gate in front of them, and the
 *        two Phase 0 results.
 *
 * The GPU twin of regime.hpp. The byte models are not restated: matrix_bytes,
 * arnoldi_working_set_bytes, spmv_dram_bytes, mgs_dram_bytes, mgs_reductions and ca_reductions
 * are properties of the operator and the algorithm, machine-independent by construction, and
 * re-deriving them would risk a discrepancy between the two maps. Only the machine-facing half
 * is rewritten, against GpuMachine rather than Machine, so a CPU constant reaching a GPU
 * coordinate is a compile error rather than a plausible wrong number.
 *
 * Three things here have no CPU counterpart:
 *
 *   - Every placed point carries a roofline verdict per kernel. Both coordinates measure
 *     communication, so a compute-bound point is off-map rather than lower-left.
 *   - regime_invariant() computes R_v * R_h in closed form. On the CPU that product was a
 *     latent property nobody needed to name; here it is both the Phase 0 answer and the width
 *     of the Upper-Right window, so it is a first-class output.
 *   - gram_intensity() prices the tall-skinny Gram matrix that communication-avoiding Arnoldi
 *     introduces and MGS does not have. It is the kernel on which the two-card contrast
 *     discriminates; see docs/regime_gpu_phase0.md.
 *
 * The predictor/outcome separation is inherited unchanged: nothing here sees a timer, and
 * R_h's numerator is a measured machine constant times a tier index, never a measured fraction
 * of the runtime it is supposed to predict.
 *
 * @author Kevin Knights
 * @date 2026-07-21
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "gpu_machine.hpp"
#include "regime.hpp"

// The kernel communication-avoiding Arnoldi adds
/**
 * @brief FLOPs of the tall-skinny Gram matrix G = B^T B for an n x s block.
 *
 * The kernel s-step orthogonalization introduces and MGS does not have. MGS's reductions are
 * a sequence of length-n dot products, intensity ~1/8 FLOP/byte; CholQR's is one n x s block
 * contracted against itself, and its intensity rises with s.
 */
[[nodiscard]] inline double gram_flops(int64_t n, int s) noexcept
{
    const double dn = static_cast<double>(n);
    const double ds = static_cast<double>(s);
    return 2.0 * dn * ds * ds;
}

/// Bytes the Gram matrix moves: the block is read once, the s x s result written once.
[[nodiscard]] inline double gram_bytes(int64_t n, int s) noexcept
{
    const double dn = static_cast<double>(n);
    const double ds = static_cast<double>(s);
    return 8.0 * dn * ds + 8.0 * ds * ds;
}

/**
 * @brief Arithmetic intensity [FLOP/byte] of the Gram matrix: ~s/4 for s << n.
 *
 * The number that decides the two-card contrast. At the certified s_max = 9 it is ~2.25
 * FLOP/byte, below the V100's FP64 ridge (~7.8) and above the 3090's (~0.59). On the
 * datacenter card the CA treatment's own orthogonalization stays memory-bound and the map's
 * mechanism is clear to pay; on consumer silicon it crosses the ridge at s ~ 2.4, far below
 * the certified width, and the advertised win is eaten by a compute roof neither coordinate
 * can see.
 *
 * Unlike SpMV (0.135) and MGS (0.375), whose intensities are below even the throttled ridge,
 * this is the kernel where the FP64 penalty binds. See docs/regime_gpu_phase0.md.
 */
[[nodiscard]] inline double gram_intensity(int64_t n, int s) noexcept
{
    const double b = gram_bytes(n, s);
    return b > 0.0 ? gram_flops(n, s) / b : 0.0;
}

/// The s at which the Gram matrix crosses a machine's ridge: below it CA's orthogonalization
/// is memory-bound and on-map, at or above it the block is compute-bound and off-map.
/// Compare against the certified s_max: a machine whose crossing sits below the certificate
/// cannot spend the full certified block width without leaving the map.
[[nodiscard]] inline double gram_ridge_s(const GpuMachine& gm, Precision p) noexcept
{
    return 4.0 * ridge_ai(gm, p);   // AI ~ s/4, so the crossing is at s = 4 * ridge
}

// Per-kernel cycle pricing
/// Timing of one Arnoldi cycle on a GPU, with the roof and the gate verdict each kernel got.
struct GpuCycleTime {
    double spmv_s        = 0.0;
    double mgs_s         = 0.0;
    double total_s       = 0.0;
    double ai_spmv       = 0.0;
    double ai_mgs        = 0.0;
    double bw_spmv_gbs   = 0.0;   ///< L2 while resident, HBM once spilled. Flat in P either way.
    double bw_mgs_gbs    = 0.0;
    bool   spmv_resident = false;
    bool   mgs_resident  = false;
    RooflineVerdict gate_spmv;
    RooflineVerdict gate_mgs;
};

/**
 * @brief Seconds for one m-step Arnoldi cycle on a GPU, pricing each kernel on its own roof.
 *
 * Same two-kernel structure as the CPU twin, with two changes. Residency is tested against a
 * fixed L2 rather than an aggregate that grows with the team, and each kernel's roofline
 * verdict is recorded rather than assumed. The CPU model could take the memory side for
 * granted because a Zen 2 core's FP64 ridge is far above any Arnoldi intensity, and that
 * assumption is what the two-GPU pair is built to test.
 */
[[nodiscard]] inline GpuCycleTime gpu_arnoldi_cycle_seconds(const GpuMachine& gm, int P,
                                                            int64_t nnz, int64_t n, int m,
                                                            double x_reuse, Precision p)
{
    GpuCycleTime t;
    const int64_t ws_spmv = spmv_working_set_bytes(nnz, n);
    const int64_t ws_mgs  = mgs_working_set_bytes(n, m);

    t.spmv_resident = ws_spmv <= aggregate_l2_bytes(gm, P);
    t.mgs_resident  = ws_mgs  <= aggregate_l2_bytes(gm, P);
    t.bw_spmv_gbs   = memory_bw_gbs(gm, ws_spmv);
    t.bw_mgs_gbs    = memory_bw_gbs(gm, ws_mgs);
    t.ai_spmv       = spmv_intensity(nnz, n, x_reuse);
    t.ai_mgs        = mgs_intensity(n, m);
    t.gate_spmv     = roofline_gate(gm, P, t.ai_spmv, p);
    t.gate_mgs      = roofline_gate(gm, P, t.ai_mgs, p);

    t.spmv_s = spmv_cycle_flops(nnz, m)
             / attainable_flops(gm, P, t.ai_spmv, t.bw_spmv_gbs, p);
    t.mgs_s  = mgs_cycle_flops(n, m)
             / attainable_flops(gm, P, t.ai_mgs, t.bw_mgs_gbs, p);
    t.total_s = t.spmv_s + t.mgs_s;
    return t;
}

// The coordinates
/**
 * @brief Vertical coordinate on a GPU: working set against the device's fixed L2.
 *
 * Independent of P, which is the structural break rather than a simplification. Engaging more
 * SMs adds no cache, so unlike the CPU's R_v ~ 1/P there is no team size that retires the
 * vertical opportunity. On the V100 the denominator is 6 MiB flat, small enough that the
 * production operator clears it by a factor of ~7, so the vertical mechanism is sharper here
 * than on puffin.
 */
[[nodiscard]] inline double R_v(const GpuMachine& gm, int P, int64_t working_set) noexcept
{
    return static_cast<double>(working_set) / static_cast<double>(aggregate_l2_bytes(gm, P));
}

/**
 * @brief Horizontal coordinate on a GPU: one reduction's latency against the compute between two.
 *
 *     numerator   = reduction_cost_s(gm, tier)    [s]  a priori, measured offline, never fitted
 *     denominator = cycle_seconds / R             [s]
 *
 * The numerator is indexed by tier, not by team size. On the CPU the reduction cost grew with
 * P because the tree deepened; here the rung sets the cost and P does not enter, so the
 * horizontal axis is traversed by changing the reduction hardware while holding N and the
 * device fixed. That is what makes the ladder a swept axis rather than a shaded prediction.
 */
[[nodiscard]] inline double R_h(const GpuMachine& gm, ReductionTier tier, double cycle_seconds,
                                int64_t reductions) noexcept
{
    if (reductions <= 0 || !(cycle_seconds > 0.0)) return 0.0;
    const double t_reduce  = reduction_cost_s(gm, tier);
    const double t_compute = cycle_seconds / static_cast<double>(reductions);
    return t_reduce / t_compute;
}

/// A point in the GPU regime plane, with the inputs that placed it and the gate that says
/// whether the plane applies to it at all.
struct GpuRegimePoint {
    int64_t n           = 0;
    int64_t nnz         = 0;
    int     m           = 0;
    int     P           = 0;      ///< SMs engaged
    ReductionTier tier  = ReductionTier::WARP;
    Precision precision = Precision::FP64;
    double  x_reuse     = 1.0;
    double  working_set = 0.0;
    double  l2          = 0.0;    ///< bytes; flat in P, unlike the CPU's llc
    double  ai          = 0.0;    ///< SpMV
    double  W           = 0.0;    ///< FLOPs per Arnoldi cycle
    int64_t reductions  = 0;      ///< R, on the MGS baseline
    double  rv          = 0.0;
    double  rh          = 0.0;
    double  cycle_s     = 0.0;
    double  t_reduce_s  = 0.0;    ///< the numerator, exposed so the tier is auditable
    double  ai_mgs      = 0.0;
    double  rv_spmv     = 0.0;
    double  rv_mgs      = 0.0;
    bool    spmv_resident = false;
    bool    mgs_resident  = false;

    // The gate. A point failing any of these is off-map, not lower-left.
    RooflineVerdict gate_spmv;
    RooflineVerdict gate_mgs;
    RooflineVerdict gate_gram;   ///< the CA-only kernel; s = the certified block width

    /// True when the machine's reachable tiers are all measured. False means rh has a shape
    /// but no trustworthy magnitude, and no horizontal verdict may be printed.
    bool rh_trustworthy = false;
    /// True when an achieved bandwidth backs the gate. False means every verdict above is
    /// provisional, computed against a theoretical roof that biases toward memory-bound.
    bool gate_trustworthy = false;
};

/// Whether this point sits on the map at all, for the baseline method (SpMV + MGS).
[[nodiscard]] inline bool on_map(const GpuRegimePoint& pt) noexcept
{
    return pt.gate_spmv.memory_bound && pt.gate_mgs.memory_bound;
}

/// Whether the CA treatment stays on the map here. Can be false while on_map() is true, and
/// that separation is the sharpened negative arm of the two-card contrast.
[[nodiscard]] inline bool treatment_on_map(const GpuRegimePoint& pt) noexcept
{
    return pt.gate_gram.memory_bound;
}

/**
 * @brief Place an operator in the GPU (R_v, R_h) plane. Pure prediction, plus a gate verdict.
 *
 * @param P  SMs engaged. Kept in the signature although neither coordinate depends on it,
 *           because the compute roof does, and therefore so does whether the gate passes.
 * @param s  certified block width, for the Gram-matrix gate only. Does not affect the
 *           coordinates, which are defined on the MGS baseline so they never depend on the
 *           treatment.
 */
[[nodiscard]] inline GpuRegimePoint place_gpu(const GpuMachine& gm, int P, ReductionTier tier,
                                              int64_t nnz, int64_t n, int m, double x_reuse,
                                              Precision p = Precision::FP64, int s = 8)
{
    GpuRegimePoint pt;
    pt.n           = n;
    pt.nnz         = nnz;
    pt.m           = m;
    pt.P           = P;
    pt.tier        = tier;
    pt.precision   = p;
    pt.x_reuse     = x_reuse;
    pt.working_set = static_cast<double>(arnoldi_working_set_bytes(nnz, n, m));
    pt.l2          = static_cast<double>(aggregate_l2_bytes(gm, P));
    pt.ai          = spmv_intensity(nnz, n, x_reuse);
    pt.W           = arnoldi_cycle_flops(nnz, n, m);
    pt.reductions  = mgs_reductions(m);
    pt.rv          = R_v(gm, P, arnoldi_working_set_bytes(nnz, n, m));

    const GpuCycleTime ct = gpu_arnoldi_cycle_seconds(gm, P, nnz, n, m, x_reuse, p);
    pt.cycle_s       = ct.total_s;
    pt.ai_mgs        = ct.ai_mgs;
    pt.spmv_resident = ct.spmv_resident;
    pt.mgs_resident  = ct.mgs_resident;
    pt.gate_spmv     = ct.gate_spmv;
    pt.gate_mgs      = ct.gate_mgs;
    pt.gate_gram     = roofline_gate(gm, P, gram_intensity(n, s), p);
    pt.rv_spmv       = R_v(gm, P, spmv_working_set_bytes(nnz, n));
    pt.rv_mgs        = R_v(gm, P, mgs_working_set_bytes(n, m));
    pt.t_reduce_s    = reduction_cost_s(gm, tier);
    pt.rh            = R_h(gm, tier, ct.total_s, pt.reductions);

    pt.rh_trustworthy   = gm.reduction_calibrated && gm.tier_calibrated[tier_index(tier)];
    pt.gate_trustworthy = gm.roofline_gated;
    return pt;
}

// The invariant, in closed form
/**
 * @brief The machine number Lambda = t_reduce * BW / C.
 *
 * Dimensionless, [s] * [B/s] / [B], and reads as how many cache-fulls of data the memory
 * system delivers during one reduction. It is the only place the machine enters the
 * R_v * R_h product, so it is the single number deciding whether Upper-Right is reachable on
 * a given rung.
 */
[[nodiscard]] inline double machine_number(const GpuMachine& gm, ReductionTier tier,
                                           double bw_gbs) noexcept
{
    const double c = static_cast<double>(gm.l2_bytes);
    return c > 0.0 ? reduction_cost_s(gm, tier) * bw_gbs * 1e9 / c : 0.0;
}

/// The invariant and the Upper-Right window width, as one object. See regime_invariant().
struct RegimeInvariant {
    double lambda       = 0.0;  ///< t_reduce * BW / C: the machine's contribution
    double g            = 0.0;  ///< working-set bytes / cycle bytes: the method's, N-free
    int64_t reductions  = 0;    ///< R(m): the method's, N-free
    double product      = 0.0;  ///< R_v * R_h = g * R * lambda
    double tau_star_s   = 0.0;  ///< the reduction cost at which product = 1
    bool   upper_right  = false;///< product >= 1: the corner is open on this rung
    bool   trustworthy  = false;///< the tier and the bandwidth were both measured
};

/**
 * @brief R_v * R_h in closed form, and the threshold reduction latency that opens Upper-Right.
 *
 * Derivation in docs/regime_gpu_phase0.md. In the memory-bound branch the modeled cycle time
 * is bytes over bandwidth, so
 *
 *     R_v * R_h = (W/C) * (t_reduce * R * BW / B) = (W/B) * R * (t_reduce * BW / C)
 *
 * and N cancels because W and B are both linear in N. What survives is a machine-independent
 * method factor g * R, times the machine number Lambda.
 *
 * On a GPU the product is independent of P and of N, but for a different reason than on the
 * CPU. There the two coordinates carried reciprocal powers of P that annihilated; here neither
 * coordinate depends on P at all, because L2 does not grow with SMs and the bandwidth roof is
 * device-wide. The axes are decoupled, N moving R_v alone and the reduction tier moving R_h
 * alone, so engaging more of the GPU cannot open the corner and climbing the ladder is the
 * only thing that can.
 *
 * The identity holds where the roofline gate passes, and only there. Engage few enough SMs and
 * the compute roof drops below the memory roof; the cycle time then carries a 1/P, R_h
 * recovers a factor of P, and the product floats. On the V100 in FP64 that happens below about
 * 4 SMs. Those points are off-map by construction, so the escape hatch is closed by the gate
 * rather than by an assumption, which is a second reason to evaluate the gate before reading
 * the coordinates.
 *
 * @param nnz,n,m,x_reuse the operator and method, exactly as place_gpu() takes them. Computed
 *        from the real nnz rather than an assumed nonzeros-per-row, so no interior-limit
 *        approximation enters.
 * @param bw_gbs the roof serving the cycle. Pass the spilled roof: the identity is derived in
 *        the memory-bound branch, and a cache-resident point is below theta_v anyway.
 */
[[nodiscard]] inline RegimeInvariant regime_invariant(const GpuMachine& gm, ReductionTier tier,
                                                      int64_t nnz, int64_t n, int m,
                                                      double x_reuse, double bw_gbs)
{
    RegimeInvariant ri;
    const double w = static_cast<double>(arnoldi_working_set_bytes(nnz, n, m));
    const double b = spmv_dram_bytes(nnz, n, x_reuse) * static_cast<double>(m)
                   + mgs_dram_bytes(n, m);
    if (!(b > 0.0)) return ri;

    ri.g          = w / b;
    ri.reductions = mgs_reductions(m);
    ri.lambda     = machine_number(gm, tier, bw_gbs);
    ri.product    = ri.g * static_cast<double>(ri.reductions) * ri.lambda;
    ri.upper_right = ri.product >= 1.0;

    // tau* : set product = 1 and solve for the reduction cost.
    const double denom = ri.g * static_cast<double>(ri.reductions) * bw_gbs * 1e9;
    ri.tau_star_s = denom > 0.0 ? static_cast<double>(gm.l2_bytes) / denom : 0.0;

    ri.trustworthy = gm.tier_calibrated[tier_index(tier)] && gm.hbm_bw_gbs_achieved > 0.0;
    return ri;
}

// The Upper-Right window, solved jointly
/// The interval in N on which both coordinates exceed 1 and the operator still fits.
struct UpperRightWindow {
    int64_t n_min      = 0;      ///< R_v >= 1 floor: C / c_w
    int64_t n_max      = 0;      ///< the binding ceiling, min of the two below
    int64_t n_max_rh   = 0;      ///< R_h >= 1 ceiling: tau * R * BW / c_b
    int64_t n_max_mem  = 0;      ///< device-memory ceiling: budget / c_w
    double  n1_min     = 0.0;    ///< n_min^(1/dim): the grid resolution
    double  n1_max     = 0.0;
    double  width      = 0.0;    ///< n_max / n_min, equal to R_v * R_h when memory does not bind
    bool    feasible   = false;
    bool    memory_binds = false;///< true if the 16 GB ceiling, not R_h, closes the window
};

/**
 * @brief Solve the three Upper-Right constraints jointly and return the interval in N.
 *
 * A joint solve, because freezing one axis while sliding the other can open a corner that is
 * only a 1%-wide artifact. Written in N the constraints are
 *
 *     R_v >= 1   =>  N >= C / c_w                (a floor)
 *     R_h >= 1   =>  N <= tau * R * BW / c_b     (a ceiling)
 *     footprint  =>  N <= budget / c_w           (a ceiling)
 *
 * so the feasible set is an interval, non-empty exactly when the invariant exceeds 1. The two
 * blocking questions are one question, which is why the derivation came before the search: a
 * numerical sweep over (N, P) would have found the interval without revealing that its width
 * is one dimensionless number.
 *
 * `memory_binds` says whether device memory is what closes the window. On the V100 it is false
 * by two orders of magnitude, since R_h's ceiling binds long before 16 GB does.
 *
 * @param nu    nonzeros per row (3, 5, 7 for a 1/2/3-D stencil). Interior limit: the boundary
 *              rows lower it by O(1/n1), under 5% at production resolution.
 * @param dim   for reporting the window in grid resolution n1 = N^(1/dim).
 * @param budget_bytes  device memory available to the operator and basis. Pass a fraction of
 *              device_memory_bytes, not all of it: workspace, the Hessenberg factor and
 *              cuSOLVER scratch are not modeled here.
 */
[[nodiscard]] inline UpperRightWindow upper_right_window(const GpuMachine& gm,
                                                         ReductionTier tier, int m, double nu,
                                                         int dim, double x_reuse,
                                                         double bw_gbs, int64_t budget_bytes)
{
    UpperRightWindow w;
    const double dm = static_cast<double>(m);

    // Per-row coefficients: both byte models are linear in N, which is what makes the
    // constraints closed-form rather than a search.
    const double c_w = 12.0 * nu + 4.0 + 8.0 * (dm + 2.0);
    const double c_spmv = (12.0 * nu + 4.0) + 8.0
                        + x_reuse * 8.0 + (1.0 - x_reuse) * 8.0 * nu;
    const double c_mgs  = 8.0 * (dm * (dm + 1.0) / 2.0) + 24.0 * dm + 24.0;
    const double c_b    = dm * c_spmv + c_mgs;
    if (!(c_w > 0.0) || !(c_b > 0.0)) return w;

    const double tau = reduction_cost_s(gm, tier);
    const double R   = static_cast<double>(mgs_reductions(m));

    w.n_min     = static_cast<int64_t>(std::ceil(static_cast<double>(gm.l2_bytes) / c_w));
    w.n_max_rh  = static_cast<int64_t>(std::floor(tau * R * bw_gbs * 1e9 / c_b));
    w.n_max_mem = static_cast<int64_t>(std::floor(static_cast<double>(budget_bytes) / c_w));
    w.n_max     = std::min(w.n_max_rh, w.n_max_mem);
    w.memory_binds = w.n_max_mem < w.n_max_rh;
    w.feasible  = w.n_max >= w.n_min && w.n_min > 0;

    const double inv_dim = 1.0 / static_cast<double>(dim);
    w.n1_min = std::pow(static_cast<double>(w.n_min), inv_dim);
    w.n1_max = w.n_max > 0 ? std::pow(static_cast<double>(w.n_max), inv_dim) : 0.0;
    w.width  = w.n_min > 0 ? static_cast<double>(w.n_max) / static_cast<double>(w.n_min) : 0.0;
    return w;
}

/// Grid resolution N = n1^dim, for reading a window back as the thing a pricing problem sets.
[[nodiscard]] inline int64_t grid_points(int n1, int dim) noexcept
{
    int64_t N = 1;
    for (int d = 0; d < dim; ++d) N *= n1;
    return N;
}
