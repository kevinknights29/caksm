/**
 * @file regime_gpu_place.cpp
 * @brief The GPU regime map's predictor: places points, evaluates the roofline gate, and
 *        regenerates the Phase 0 answers from whatever constants the calibration returned.
 *
 * Pure prediction. It links neither a solver nor a timer and never calls CUDA, so it runs on a
 * login node or a laptop. Every number it prints is an a-priori coordinate, and a binary that
 * could see a runtime would put the map's non-circularity at risk, which is the rule
 * src/calibrate_alpha.cpp is built around from the other side.
 *
 * The predict-then-measure firewall is enforced structurally. A preset whose tiers are
 * uncalibrated has t_reduce = 0, which would make every R_h zero and every window empty. The
 * --assume-* flags let the tables be produced from stated priors anyway, but every such figure
 * is marked assumed in the output and in the CSV, and the binary refuses to present an assumed
 * number as a verdict. Tuning constants until a crossover landed at 1 would prove nothing.
 *
 * Usage:
 *   ./regime-gpu-place [--machine v100-pcie-16gb] [--m 12] [--dim 3] [--s 8]
 *                      [--n1-list "31 61 74 89"] [--sm N]
 *                      [--assume-grid-us X] [--assume-p2p-us X] [--assume-node-us X]
 *                      [--assume-bw-gbs X] [--budget-frac F] [--csv PATH]
 *
 * @author Kevin Knights
 * @date 2026-07-21
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <print>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "gpu_machine.hpp"
#include "gpu_regime.hpp"

namespace {

/**
 * @brief Nonzeros of a dim-dimensional (2*dim+1)-point stencil on an n1^dim grid.
 *
 * Exact, not the interior limit 2*dim+1 per row: the boundary planes lose one neighbor each,
 * and at the coarse end of a sweep that is a several-percent correction to every byte model
 * downstream. regime_invariant() takes the real nnz for the same reason.
 */
[[nodiscard]] int64_t stencil_nnz(int n1, int dim)
{
    const int64_t N = grid_points(n1, dim);
    int64_t nnz = N;                                    // the diagonal
    int64_t plane = N / n1;                             // points in one boundary plane
    for (int d = 0; d < dim; ++d) nnz += 2 * (N - plane);
    return nnz;
}

/// A mutable copy of a preset, so --assume-* can fill unmeasured slots without touching the
/// constexpr table. Carries a flag per slot so the output can mark what was assumed.
struct Assumed {
    bool tau[kTierCount] = {false, false, false, false, false};
    bool bandwidth = false;
    [[nodiscard]] bool any() const
    {
        for (bool b : tau) if (b) return true;
        return bandwidth;
    }
};

struct Args {
    std::string_view machine = "v100-pcie-16gb";
    int    m   = 12;
    int    dim = 3;
    int    s   = 8;
    int    sm  = 0;          ///< 0 = all SMs
    double budget_frac = 0.8;
    std::vector<int> n1_list{31, 45, 61, 74, 89};
    double assume_us[kTierCount] = {0.0, 0.0, 0.0, 0.0, 0.0};
    double assume_bw_gbs = 0.0;
    std::string csv_path;
};

[[nodiscard]] std::vector<int> parse_int_list(std::string_view s)
{
    std::vector<int> out;
    std::istringstream in{std::string(s)};
    int v = 0;
    while (in >> v) out.push_back(v);
    if (out.empty()) throw std::invalid_argument("Empty integer list");
    return out;
}

[[nodiscard]] Args parse_args(std::span<const char* const> argv)
{
    Args a;
    for (std::size_t i = 1; i < argv.size(); ++i) {
        const std::string_view arg = argv[i];
        auto next = [&]() -> std::string_view {
            if (++i >= argv.size())
                throw std::invalid_argument("Missing value for " + std::string(arg));
            return argv[i];
        };
        if      (arg == "--machine")     a.machine = next();
        else if (arg == "--m")           a.m   = std::stoi(std::string(next()));
        else if (arg == "--dim")         a.dim = std::stoi(std::string(next()));
        else if (arg == "--s")           a.s   = std::stoi(std::string(next()));
        else if (arg == "--sm")          a.sm  = std::stoi(std::string(next()));
        else if (arg == "--budget-frac") a.budget_frac = std::stod(std::string(next()));
        else if (arg == "--n1-list")     a.n1_list = parse_int_list(next());
        else if (arg == "--assume-warp-us")
            a.assume_us[tier_index(ReductionTier::WARP)] = std::stod(std::string(next()));
        else if (arg == "--assume-block-us")
            a.assume_us[tier_index(ReductionTier::BLOCK)] = std::stod(std::string(next()));
        else if (arg == "--assume-grid-us")
            a.assume_us[tier_index(ReductionTier::GRID)] = std::stod(std::string(next()));
        else if (arg == "--assume-p2p-us")
            a.assume_us[tier_index(ReductionTier::DEVICE_P2P)] = std::stod(std::string(next()));
        else if (arg == "--assume-node-us")
            a.assume_us[tier_index(ReductionTier::NODE)] = std::stod(std::string(next()));
        else if (arg == "--assume-bw-gbs") a.assume_bw_gbs = std::stod(std::string(next()));
        else if (arg == "--csv")         a.csv_path = std::string(next());
        else if (arg == "--help") {
            std::println("Usage: ./regime-gpu-place [--machine KEY] [--m M] [--dim D] [--s S]");
            std::println("                          [--n1-list \"31 61 74\"] [--sm N]");
            std::println("                          [--assume-{{warp,block,grid,p2p,node}}-us X]");
            std::println("                          [--assume-bw-gbs X] [--budget-frac F]");
            std::println("                          [--csv PATH]");
            std::println("");
            std::println("  --machine KEY  v100-pcie-16gb (synge) or rtx-3090 (puffin).");
            std::println("  --m M          Krylov dimension per cycle (default 12, production).");
            std::println("  --dim D        spatial dimensions; sets the stencil (default 3).");
            std::println("  --s S          certified block width, for the Gram-matrix gate only.");
            std::println("                 Does not touch the coordinates: those are defined on");
            std::println("                 the MGS baseline so they never depend on the treatment.");
            std::println("  --sm N         SMs engaged (default: all). Neither coordinate depends");
            std::println("                 on it, which is the invariant's result, but the compute");
            std::println("                 roof does, so the gate verdict can.");
            std::println("  --assume-*-us  cumulative reduction latency for a rung, microseconds,");
            std::println("                 used only where the preset is uncalibrated. Every");
            std::println("                 figure derived from one is marked ASSUMED. These are");
            std::println("                 priors for the Phase 0 tables, never inputs to a");
            std::println("                 verdict: calibrate the rung instead.");
            std::println("  --budget-frac  fraction of device memory the operator and basis may");
            std::println("                 use in the window solve (default 0.8; workspace, the");
            std::println("                 Hessenberg factor and cuSOLVER scratch are unmodeled).");
            std::exit(0);
        }
        else throw std::invalid_argument("Unknown flag: " + std::string(arg));
    }
    if (a.m < 1)   throw std::invalid_argument("--m must be >= 1");
    if (a.dim < 1 || a.dim > 3) throw std::invalid_argument("--dim must be 1, 2 or 3");
    if (a.s < 1)   throw std::invalid_argument("--s must be >= 1");
    if (!(a.budget_frac > 0.0 && a.budget_frac <= 1.0))
        throw std::invalid_argument("--budget-frac must be in (0, 1]");
    return a;
}

/// Fill unmeasured slots from the --assume-* priors. Returns what was assumed, so the report
/// can mark it; an assumed figure that reached a verdict unlabeled would break the firewall.
[[nodiscard]] Assumed apply_assumptions(GpuMachine& gm, const Args& a)
{
    Assumed as;
    // The flags are cumulative latencies, which is what a probe reports; the struct stores
    // increments. Convert here rather than asking the user to subtract.
    double running = 0.0;
    for (int i = 0; i < kTierCount; ++i) {
        const auto t = static_cast<ReductionTier>(i);
        if (gm.tier_calibrated[tier_index(t)]) {
            running = reduction_cost_s(gm, t);
            continue;
        }
        if (a.assume_us[i] > 0.0) {
            const double cumulative = a.assume_us[i] * 1e-6;
            gm.t_reduce_s[tier_index(t)] = std::max(0.0, cumulative - running);
            running = cumulative;
            as.tau[i] = true;
        }
    }
    if (gm.hbm_bw_gbs_achieved <= 0.0 && a.assume_bw_gbs > 0.0) {
        gm.hbm_bw_gbs_achieved = a.assume_bw_gbs;
        as.bandwidth = true;
    }
    return as;
}

/**
 * @brief Row marker: how much of this row is a measurement.
 *
 * Three states, not two. Assumed means a --assume-* prior filled the slot. Uncalibrated means
 * nothing filled it, so the rung's increment is zero and reduction_cost_s() returned the rung
 * below: the row's t_reduce is real but belongs to different hardware, which is more dangerous
 * than an obviously invented number.
 */
[[nodiscard]] const char* mark(bool assumed, bool calibrated)
{
    if (assumed)         return " (ASSUMED)";
    if (!calibrated)     return " (UNCALIBRATED: shows the rung below)";
    return "";
}

}  // namespace

int main(int argc, char* argv[])
{
    try {
        const Args a = parse_args(
            std::span<const char* const>(argv, static_cast<std::size_t>(argc)));
        GpuMachine gm = lookup_gpu_machine(a.machine);
        const Assumed as = apply_assumptions(gm, a);

        const int P = a.sm > 0 ? a.sm : gm.sm_count;
        const double bw = achieved_or_peak_bw_gbs(gm);
        const int64_t budget = static_cast<int64_t>(
            static_cast<double>(gm.device_memory_bytes) * a.budget_frac);

        std::println("GPU regime placement -- a-priori coordinates, gate verdicts, and the");
        std::println("Phase 0 answers recomputed from the preset's current constants.");
        std::println("");
        std::println("  machine = {} ({})", gm.key, gm.name);
        std::println("  SMs={} (P={})  L2={:.1f} MiB  memory={:.0f} GiB  interconnect={} x{}",
                     gm.sm_count, P,
                     static_cast<double>(gm.l2_bytes) / (1024.0 * 1024.0),
                     static_cast<double>(gm.device_memory_bytes) / (1024.0 * 1024.0 * 1024.0),
                     gm.interconnect, gm.gpu_count);
        std::println("  FP64 peak={:.2f} TFLOP/s   BW={:.0f} GB/s{}   FP64 ridge={:.2f} FLOP/B",
                     gm.fp64_flops_peak * 1e-12, bw,
                     gm.hbm_bw_gbs_achieved > 0.0 ? (as.bandwidth ? " (ASSUMED)" : " (measured)")
                                                  : " (datasheet)",
                     ridge_ai(gm, Precision::FP64));
        std::println("  method: m={}  dim={}  s={}  reductions/cycle R={}",
                     a.m, a.dim, a.s, mgs_reductions(a.m));
        std::println("");
        if (!gm.reduction_calibrated || !gm.roofline_gated)
            std::println("  NOTE: preset is not fully calibrated "
                         "(reduction_calibrated={}, roofline_gated={}). Magnitudes below are\n"
                         "        PREDICTIONS from priors, not verdicts. Run "
                         "scripts/regime/calibrate_gpu.sh on the host.",
                         gm.reduction_calibrated, gm.roofline_gated);
        if (as.any())
            std::println("  NOTE: one or more constants came from --assume-*. Every figure they "
                         "touch is marked ASSUMED.");
        std::println("");

        // The invariant, per rung
        std::println("=== R_v * R_h = g(m) * R(m) * Lambda,  Lambda = t_reduce * BW / C ===");
        std::println("");
        std::println("  {:<12} {:>12} {:>12} {:>12} {:>12} {:>8}",
                     "tier", "t_reduce", "Lambda", "R_v*R_h", "tau*", "corner");
        std::println("  {}", std::string(74, '-'));

        // The invariant is N-free, so any representative N does; use the production point.
        const int n1_ref = a.n1_list[a.n1_list.size() / 2];
        const int64_t N_ref   = grid_points(n1_ref, a.dim);
        const int64_t nnz_ref = stencil_nnz(n1_ref, a.dim);

        std::vector<std::pair<ReductionTier, RegimeInvariant>> invariants;
        for (int i = 0; i < kTierCount; ++i) {
            const auto t = static_cast<ReductionTier>(i);
            if (!tier_reachable(gm, t)) continue;
            const RegimeInvariant ri =
                regime_invariant(gm, t, nnz_ref, N_ref, a.m, 1.0, bw);
            invariants.emplace_back(t, ri);
            const bool assumed = as.tau[i] || as.bandwidth;
            std::println("  {:<12} {:>9.2f} us {:>12.4g} {:>12.4g} {:>9.2f} us {:>8}{}",
                         tier_name(t), reduction_cost_s(gm, t) * 1e6, ri.lambda, ri.product,
                         ri.tau_star_s * 1e6, ri.upper_right ? "OPEN" : "closed",
                         mark(assumed, gm.tier_calibrated[tier_index(t)] || assumed));
        }
        std::println("  {}", std::string(74, '-'));
        std::println("  tau* is the reduction cost at which the product reaches 1; it does not");
        std::println("  vary by rung, so the column is a constant and the comparison to read is");
        std::println("  t_reduce against it. Both coordinates are P-free on a GPU, so no choice");
        std::println("  of SM count moves any row: only the rung does.");
        std::println("");

        // The window, per rung
        std::println("=== Upper-Right window: R_v >= 1 and R_h >= 1 and footprint <= budget ===");
        std::println("");
        std::println("  budget = {:.1f} GiB ({:.0f}% of device memory)",
                     static_cast<double>(budget) / (1024.0 * 1024.0 * 1024.0),
                     a.budget_frac * 100.0);
        std::println("");
        std::println("  {:<12} {:>11} {:>11} {:>9} {:>9} {:>9} {:>12}",
                     "tier", "N_min", "N_max", "n1_min", "n1_max", "width", "closed by");
        std::println("  {}", std::string(80, '-'));

        const double nu = static_cast<double>(nnz_ref) / static_cast<double>(N_ref);
        std::vector<std::pair<ReductionTier, UpperRightWindow>> windows;
        for (int i = 0; i < kTierCount; ++i) {
            const auto t = static_cast<ReductionTier>(i);
            if (!tier_reachable(gm, t)) continue;
            const UpperRightWindow w =
                upper_right_window(gm, t, a.m, nu, a.dim, 1.0, bw, budget);
            windows.emplace_back(t, w);
            std::println("  {:<12} {:>11} {:>11} {:>9.1f} {:>9.1f} {:>9.2g} {:>12}{}",
                         tier_name(t), w.n_min, std::max<int64_t>(w.n_max, 0),
                         w.n1_min, w.n1_max, w.width,
                         !w.feasible ? "EMPTY" : (w.memory_binds ? "memory" : "R_h"),
                         mark(as.tau[i] || as.bandwidth,
                              gm.tier_calibrated[tier_index(t)] || as.tau[i]));
        }
        std::println("  {}", std::string(80, '-'));
        std::println("  The window is non-empty exactly when the product above exceeds 1, and its");
        std::println("  width in N is that product: the two blocking questions are one question.");
        std::println("  'closed by' names the binding ceiling. If it never reads 'memory', the");
        std::println("  device-memory ceiling is not the operative constraint.");
        std::println("");

        // The gate, per kernel
        std::println("=== Roofline gate: is the point on the map at all? ===");
        std::println("");
        std::println("  ridge (FP64) = {:.2f} FLOP/B      ridge (FP32) = {:.2f} FLOP/B",
                     ridge_ai(gm, Precision::FP64), ridge_ai(gm, Precision::FP32));
        std::println("  Gram matrix crosses the FP64 ridge at s = {:.1f}  (certified s_max is 9)",
                     gram_ridge_s(gm, Precision::FP64));
        std::println("");
        {
            const GpuRegimePoint pt = place_gpu(gm, P, highest_reachable_tier(gm),
                                                nnz_ref, N_ref, a.m, 1.0, Precision::FP64, a.s);
            std::println("  at n1={} (N={}), FP64:", n1_ref, N_ref);
            auto gate_row = [](const char* k, const RooflineVerdict& v) {
                std::println("    {:<22} AI={:>8.3f}  margin={:>8.2f}x  {}{}",
                             k, v.ai, v.margin,
                             v.memory_bound ? "memory-bound (ON MAP)" : "COMPUTE-BOUND (OFF MAP)",
                             v.provisional ? "  [provisional: roof unmeasured]" : "");
            };
            gate_row("SpMV", pt.gate_spmv);
            gate_row("MGS", pt.gate_mgs);
            gate_row("Gram B^T B (CA only)", pt.gate_gram);
            std::println("");
            std::println("    baseline on map: {}      CA treatment on map: {}",
                         on_map(pt) ? "yes" : "NO", treatment_on_map(pt) ? "yes" : "NO");
            if (on_map(pt) && !treatment_on_map(pt))
                std::println("    READING: the map charts the baseline here, but the kernel CA "
                             "ADDS is compute-bound.\n"
                             "             s-step's advertised win is eaten by a roof neither "
                             "coordinate can see.");
        }
        std::println("");

        // The trajectory
        std::println("=== The operator's trajectory across the grid sweep ===");
        std::println("");
        // The most expensive rung with a measurement behind it, not merely the most expensive
        // the hardware reaches. An uncalibrated rung has a zero increment, so placing on it
        // would report the cost of the rung below under the higher rung's name.
        const ReductionTier top       = highest_calibrated_tier(gm);
        const ReductionTier reachable = highest_reachable_tier(gm);
        std::println("  placed on the '{}' rung (the most expensive one that is CALIBRATED)",
                     tier_name(top));
        if (top != reachable)
            std::println("  NOTE: this machine reaches '{}', but that rung is unmeasured, so it\n"
                         "        is not used. The horizontal axis below is {} rung(s) short of\n"
                         "        the hardware's reach; say so rather than implying otherwise.",
                         tier_name(reachable),
                         static_cast<int>(reachable) - static_cast<int>(top));
        std::println("");
        std::println("  {:>5} {:>11} {:>12} {:>10} {:>10} {:>11} {:>10} {:>8}",
                     "n1", "N", "ws (MiB)", "R_v", "R_h", "cycle (us)", "corner", "on map");
        std::println("  {}", std::string(86, '-'));

        std::vector<GpuRegimePoint> pts;
        for (const int n1 : a.n1_list) {
            const int64_t N   = grid_points(n1, a.dim);
            const int64_t nnz = stencil_nnz(n1, a.dim);
            const GpuRegimePoint pt =
                place_gpu(gm, P, top, nnz, N, a.m, 1.0, Precision::FP64, a.s);
            pts.push_back(pt);

            const char* corner =
                pt.rv >= kThetaV ? (pt.rh >= kThetaH ? "upper-right" : "upper-left")
                                 : (pt.rh >= kThetaH ? "lower-right" : "lower-left");
            std::println("  {:>5} {:>11} {:>12.2f} {:>10.4g} {:>10.4g} {:>11.3f} {:>10} {:>8}",
                         n1, N, pt.working_set / (1024.0 * 1024.0), pt.rv, pt.rh,
                         pt.cycle_s * 1e6, corner, on_map(pt) ? "yes" : "NO");
        }
        std::println("  {}", std::string(86, '-'));
        std::println("  theta_v = theta_h = 1. The thresholds do not move between machines; only");
        std::println("  the coordinates are recomputed. A point that fails the gate is off-map,");
        std::println("  not lower-left, and its corner label is meaningless.");
        std::println("");

        if (!a.csv_path.empty()) {
            std::ofstream out(a.csv_path, std::ios::trunc);
            if (!out) throw std::runtime_error("Cannot open CSV: " + a.csv_path);
            out << "machine,device,sm_count,P,tier,precision,m,dim,s,n1,n,nnz,"
                   "working_set_bytes,l2_bytes,rv,rh,cycle_s,t_reduce_s,reductions,"
                   "ai_spmv,ai_mgs,ai_gram,ridge_fp64,gate_spmv,gate_mgs,gate_gram,"
                   "on_map,treatment_on_map,rh_trustworthy,gate_trustworthy,assumed\n";
            for (std::size_t i = 0; i < pts.size(); ++i) {
                const GpuRegimePoint& pt = pts[i];
                out << gm.key << ',' << '"' << gm.name << '"' << ','
                    << gm.sm_count << ',' << pt.P << ',' << tier_name(pt.tier) << ','
                    << precision_name(pt.precision) << ','
                    << pt.m << ',' << a.dim << ',' << a.s << ',' << a.n1_list[i] << ','
                    << pt.n << ',' << pt.nnz << ','
                    << pt.working_set << ',' << gm.l2_bytes << ','
                    << pt.rv << ',' << pt.rh << ',' << pt.cycle_s << ',' << pt.t_reduce_s << ','
                    << pt.reductions << ','
                    << pt.ai << ',' << pt.ai_mgs << ',' << pt.gate_gram.ai << ','
                    << ridge_ai(gm, Precision::FP64) << ','
                    << (pt.gate_spmv.memory_bound ? 1 : 0) << ','
                    << (pt.gate_mgs.memory_bound ? 1 : 0) << ','
                    << (pt.gate_gram.memory_bound ? 1 : 0) << ','
                    << (on_map(pt) ? 1 : 0) << ',' << (treatment_on_map(pt) ? 1 : 0) << ','
                    << (pt.rh_trustworthy ? 1 : 0) << ',' << (pt.gate_trustworthy ? 1 : 0) << ','
                    << (as.any() ? 1 : 0) << '\n';
            }
            std::println("  [Wrote {} rows to {}]", pts.size(), a.csv_path);
        }

    } catch (const std::exception& e) {
        std::println(std::cerr, "Error: {}", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
