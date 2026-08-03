/**
 * @file regime_sweep.cpp
 * @brief The vertical crossover experiment: does cache-blocked matrix-powers pay above
 *        R_v = 1?
 *
 * The strong-scaling anchor already located theta_v: SpMV falls onto the DRAM roof as R_v
 * crosses 1. This harness demonstrates that cache-blocked matrix-powers pays above it,
 * timing baseline (m separate SpMVs) against tiled (blocks of the certified width s)
 * across an R_v sweep. The claim under test: tiled/baseline is ~1 below R_v = 1 and rises
 * toward the traffic ratio m/ceil(m/s) ~ s above it.
 *
 * s comes from measured_s_max (mpk.hpp's operative_s), never a bandwidth-optimal value,
 * and the panel height from mpk.hpp::mpk_panel_rows so the operator slice fits the target
 * cache. Each point also reports achieved effective bandwidth against the DRAM ceiling, so
 * an underperforming tiled kernel reads as a bug, not a refuted boundary. Single core,
 * single cache level: the minimum viable kernel.
 *
 * Usage:
 *   ./regime-sweep [--n1 N] [--dim D] [--m M] [--pattern banded|scattered]
 *                  [--tile-level l2|l3] [--repeats K] [--machine amd-3960x] [--csv PATH]
 *
 * @author Kevin Knights
 * @date 2026-07-19
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <print>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "akx.hpp"
#include "ca_arnoldi.hpp"
#include "machine.hpp"
#include "mpk.hpp"
#include "regime.hpp"
#include "synthetic.hpp"

using Clock = std::chrono::steady_clock;
using Sec   = std::chrono::duration<double>;

namespace {

// median / IQR over timed repeats, mirroring scaling.cpp so the two harnesses report
// timing the same way.
struct Stat { double median = 0.0, q1 = 0.0, q3 = 0.0; };

double quantile(std::vector<double> v, double q)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    if (v.size() == 1) return v[0];
    const double pos = q * static_cast<double>(v.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(pos));
    const auto hi = static_cast<std::size_t>(std::ceil(pos));
    return v[lo] + (pos - static_cast<double>(lo)) * (v[hi] - v[lo]);
}

Stat summarize(const std::vector<double>& v)
{
    return { quantile(v, 0.5), quantile(v, 0.25), quantile(v, 0.75) };
}

/// Build [v, Av, ..., A^m v] as m separate SpMVs: the baseline, operator streamed m times.
Eigen::MatrixXd build_baseline(const SpMatRow& A, const Eigen::VectorXd& v, int m)
{
    return spmv_chain(A, v, m);
}

/// Build the same basis in blocks of the certified width, each block a cache-blocked
/// matrix-powers call: the operator is streamed once per block, ceil(m/s) times. This is
/// the matrix-powers construction the certified ca_arnoldi runs, minus the orthogonalization
/// (the vertical experiment is about traffic, not the horizontal axis). s is the highest
/// power per block, so a block spans [start, A*start, ..., A^s * start], exactly the width
/// the CholeskyQR certificate admits when s = measured_s_max; the caller never passes
/// anything larger.
///
/// `scratch` is caller-owned so it can persist across repeated timed calls; allocating it
/// fresh per call would still leave one allocation per repeat instead of per block, not
/// the steady state a fair timing harness should measure.
/// @param debug_no_halo diagnostic only, see tiled_matrix_powers_into. Produces a wrong
///                      basis silently. Never pass true when the result is used for
///                      anything but a throughput comparison thrown away immediately.
Eigen::MatrixXd build_tiled(const SpMatRow& A, const Eigen::VectorXd& v, int m, int s,
                            int64_t tile, int64_t w, Eigen::MatrixXd& scratch,
                            bool debug_no_halo = false)
{
    Eigen::MatrixXd B(A.rows(), m + 1);
    B.col(0) = v;
    int filled = 0;                        // highest power built so far; also this block's
                                            // level-0 column index (the _into contract)
    while (filled < m) {
        const int blk = std::min(s, m - filled);   // this block builds `blk` more powers
        tiled_matrix_powers_into(A, blk, tile, w, B, filled, scratch, debug_no_halo);
        filled += blk;
    }
    return B;
}

/// Median wall-clock of building a basis, warm-up discarded, result kept observable.
double time_build(auto&& build, int repeats, double& checksum)
{
    (void)build();
    std::vector<double> t;
    t.reserve(static_cast<std::size_t>(repeats));
    for (int r = 0; r < repeats; ++r) {
        const auto t0 = Clock::now();
        const Eigen::MatrixXd B = build();
        t.push_back(Sec(Clock::now() - t0).count());
        checksum += B(B.rows() / 2, B.cols() - 1);
    }
    return summarize(t).median;
}

/// One row of the s-width sweep.
struct SRow {
    int    s = 0;
    double kappa = 0.0;      ///< kappa_2([v, ..., A^s v]): the numerical roof's source
    bool   cert_ok = false;  ///< kappa < u^(-1/2): inside the CholeskyQR certificate
    int64_t panel = 0;       ///< mpk panel height at this s; 0 = no panel fits
    double halo = 0.0;       ///< halo rows per interior row
    bool   fits = false;     ///< panel fits and halo thin: the capacity roof's source
    double t_base = 0.0, t_tiled = 0.0, reuse = 0.0, eff_bw = 0.0;
    int64_t red_mgs = 0, red_ca = 0;   ///< reductions/cycle at a nominal m, MGS vs s-step
};

struct Args {
    int         n1        = 200;
    int         dim       = 2;
    int         m         = 8;
    int         repeats   = 5;
    int         swidth    = 0;    ///< > 0: run the s-width sweep to this ceiling, not a point
    Pattern     pattern   = Pattern::BANDED;
    TileLevel   level     = TileLevel::L3;
    std::string machine   = "amd-3960x";
    std::string csv_path;
    std::string debug_csv_path;   ///< optional: structured landing spot for the no-halo
                                  ///< diagnostic, kept separate from csv_path on purpose;
                                  ///< see the write site for why.
};

Args parse(std::span<const char* const> argv)
{
    Args a;
    for (std::size_t i = 1; i < argv.size(); ++i) {
        const std::string_view arg = argv[i];
        auto next = [&]() -> std::string_view {
            if (++i >= argv.size())
                throw std::invalid_argument("Missing value for " + std::string(arg));
            return argv[i];
        };
        if      (arg == "--n1")       a.n1 = std::stoi(std::string(next()));
        else if (arg == "--dim")      a.dim = std::stoi(std::string(next()));
        else if (arg == "--m")        a.m = std::stoi(std::string(next()));
        else if (arg == "--swidth")   a.swidth = std::stoi(std::string(next()));
        else if (arg == "--repeats")  a.repeats = std::stoi(std::string(next()));
        else if (arg == "--machine")  a.machine = std::string(next());
        else if (arg == "--csv")      a.csv_path = std::string(next());
        else if (arg == "--debug-csv") a.debug_csv_path = std::string(next());
        else if (arg == "--pattern") {
            const auto v = next();
            if      (v == "banded")    a.pattern = Pattern::BANDED;
            else if (v == "scattered") a.pattern = Pattern::SCATTERED;
            else throw std::invalid_argument("--pattern must be banded|scattered");
        }
        else if (arg == "--tile-level") {
            const auto v = next();
            if      (v == "l2") a.level = TileLevel::L2;
            else if (v == "l3") a.level = TileLevel::L3;
            else throw std::invalid_argument("--tile-level must be l2|l3");
        }
        else if (arg == "--help") {
            std::println("Usage: ./regime-sweep [--n1 N] [--dim D] [--m M]");
            std::println("                      [--pattern banded|scattered] [--tile-level l2|l3]");
            std::println("                      [--repeats K] [--machine KEY] [--csv PATH]");
            std::println("                      [--debug-csv PATH]");
            std::println("");
            std::println("  Times baseline (m SpMVs) vs tiled matrix-powers (blocks of the");
            std::println("  certified width s) on ONE core, and reports the tiled/baseline");
            std::println("  speedup against R_v. Sweep --n1 to move R_v through 1.");
            std::println("");
            std::println("  --debug-csv PATH  structured landing spot for the no-halo diagnostic");
            std::println("                 (spilled points only): baseline/tiled/no-halo GFLOP/s");
            std::println("                 and a cache-residency verdict. DELIBERATELY separate");
            std::println("                 from --csv: the no-halo basis is known wrong, so it must");
            std::println("                 never land in the file a plot might treat as real points.");
            std::println("");
            std::println("  --swidth CEIL  instead of one crossover point, sweep the block width");
            std::println("                 s = 1..CEIL at this n1 and report BOTH roofs: the");
            std::println("                 numerical one (kappa(B_s) leaving the CholeskyQR");
            std::println("                 certificate) and the capacity one (the halo growing");
            std::println("                 until no panel fits). Which binds first is the finding;");
            std::println("                 run it at an n1 whose R_v > 1 so reuse actually matters.");
            std::exit(0);
        }
        else throw std::invalid_argument("Unknown flag: " + std::string(arg));
    }
    if (a.m < 1) throw std::invalid_argument("--m must be >= 1");
    return a;
}

const char* kCsvHeader =
    "machine,pattern,tile_level,n1,dim,N,nnz,m,s_cert,tile_rows,w,repeats,"
    "working_set_mib,tile_capacity_mib,rv_1core,rv_blocked,mpk_form,"
    "t_baseline_ms,t_baseline_q1,t_baseline_q3,t_tiled_ms,t_tiled_q1,t_tiled_q3,"
    "speedup,traffic_ratio,gflops_baseline,gflops_tiled,"
    "eff_bw_baseline_gbs,eff_bw_tiled_gbs,ai_gate_pass\n";

}  // namespace

int main(int argc, char* argv[])
{
    try {
        const Args a = parse(std::span<const char* const>(argv, static_cast<std::size_t>(argc)));
        const Machine& mc = lookup_machine(a.machine);

        // Banded operator by default; scattered for the "SpMV genuinely on the DRAM roof"
        // control (where the kernel does not tile and the tiled arm degenerates to baseline).
        SyntheticSpec sp;
        sp.n1 = a.n1;
        sp.dim = a.dim;
        sp.scatter_block = (a.pattern == Pattern::BANDED) ? 1 : synthetic_dimension(sp);
        const SyntheticOperator op = build_synthetic(sp);
        // spmv_row_dot reads the raw CSR arrays directly, which requires a compressed
        // matrix; the row-major conversion produces one in practice, but this makes it
        // explicit rather than relying on that staying true.
        SpMatRow A_mut(op.A);
        A_mut.makeCompressed();
        const SpMatRow& A = A_mut;
        const int64_t N   = op.n;
        const int64_t nnz = op.nnz;
        const double  nnz_per_row = static_cast<double>(nnz) / static_cast<double>(N);

        const Eigen::VectorXd v0 = [&] {
            Eigen::VectorXd v(N);
            for (int64_t i = 0; i < N; ++i)
                v(i) = std::sin(0.7 * static_cast<double>(i) + 1.0);
            return v.normalized();
        }();

        const int64_t w = operator_bandwidth(A);

        // s-width sweep: do not pin s at the certified width, sweep it and let the two
        // independent roofs pick it. Larger s means fewer reductions (horizontal) and more
        // reuse per operator-stream (vertical), but a fatter halo (the capacity roof: the
        // panel stops fitting) and a worse-conditioned monomial basis (the numerical roof:
        // it leaves the CholeskyQR certificate). Which roof binds first is the finding, and
        // it is a property of the operator and machine jointly, not a value to assume.
        if (a.swidth > 0) {
            std::vector<SRow> sweep;
            double checksum = 0.0;
            for (int s = 1; s <= a.swidth; ++s) {
                SRow r;
                r.s = s;
                r.kappa   = condition_number(matrix_powers(op.A, v0, s));
                r.cert_ok = r.kappa < cholqr_kappa_limit();
                r.panel   = mpk_panel_rows(mc, a.level, s, w, N, nnz_per_row, /*cores_sharing=*/1);
                r.halo    = (r.panel > 0) ? mpk_halo_ratio(r.panel, s, w)
                                          : std::numeric_limits<double>::infinity();
                r.fits    = (r.panel > 0) && (r.halo <= kThinHaloRatio);
                r.red_mgs = mgs_reductions(a.m);
                r.red_ca  = ca_reductions(a.m, s, /*reorth=*/true);

                const int64_t stile = r.fits ? r.panel : N;
                r.t_base  = time_build([&] { return spmv_chain(A, v0, s); }, a.repeats, checksum);
                r.t_tiled = time_build([&] { return tiled_matrix_powers(A, v0, s, stile, w); },
                                       a.repeats, checksum);
                r.reuse   = r.t_base / r.t_tiled;
                r.eff_bw  = static_cast<double>(matrix_bytes(nnz, N)) / r.t_tiled / 1e9;
                sweep.push_back(r);
            }

            int s_certified = 0, s_ghost = 0;
            for (const SRow& r : sweep) {
                if (r.cert_ok) s_certified = r.s;
                if (r.fits)    s_ghost     = r.s;
            }
            const char* binds = (s_certified <= s_ghost) ? "certificate (numerical)"
                                                         : "ghost (capacity)";

            std::println("regime-sweep -- s-width sweep, one core");
            std::println("  machine={}  tile-level={}  n1={} dim={}  N={}  nnz={}  w={}  m={}",
                         mc.key, tile_level_name(a.level), a.n1, a.dim, N, nnz, w, a.m);
            std::println("  R_v(1 core) = {:.3f}   (want > 1 so the operator spills and reuse matters)",
                         static_cast<double>(arnoldi_working_set_bytes(nnz, N, a.m))
                             / tile_capacity_bytes(mc, a.level, /*cores_sharing=*/1));
            std::println("");
            std::println("  {:>3} {:>10} {:>5} {:>9} {:>7} {:>5} {:>8} {:>7} {:>8} {:>8}",
                         "s", "kappa(Bs)", "cert", "panel", "halo", "fit", "reuse",
                         "eff GB/s", "red_mgs", "red_ca");
            for (const SRow& r : sweep)
                std::println("  {:>3} {:>10.2e} {:>5} {:>9} {:>7.2f} {:>5} {:>8.2f} {:>7.1f} "
                             "{:>8} {:>8}",
                             r.s, r.kappa, r.cert_ok ? "ok" : "OUT", r.panel, r.halo,
                             r.fits ? "yes" : "no", r.reuse, r.eff_bw, r.red_mgs, r.red_ca);
            std::println("");
            std::println("=== s_certified = {} (numerical roof)   s_ghost_max = {} (capacity roof) ===",
                         s_certified, s_ghost);
            std::println("=== BINDING ROOF: {}  ->  operative s = {} ===",
                         binds, std::min(std::max(1, s_certified), std::max(1, s_ghost)));

            if (!a.csv_path.empty()) {
                const bool exists = std::filesystem::exists(a.csv_path)
                                 && std::filesystem::file_size(a.csv_path) > 0;
                std::ofstream f(a.csv_path, std::ios::app);
                if (!f) throw std::runtime_error("Cannot open CSV: " + a.csv_path);
                if (!exists)
                    f << "machine,tile_level,n1,dim,N,nnz,w,m,repeats,s,kappa_Bs,cert_ok,"
                         "panel_rows,halo_ratio,fits,t_base_ms,t_tiled_ms,reuse,eff_bw_gbs,"
                         "red_mgs,red_ca,s_certified,s_ghost_max\n";
                for (const SRow& r : sweep)
                    f << mc.key << ',' << tile_level_name(a.level) << ',' << a.n1 << ','
                      << a.dim << ',' << N << ',' << nnz << ',' << w << ',' << a.m << ','
                      << a.repeats << ',' << r.s << ',' << r.kappa << ','
                      << (r.cert_ok ? 1 : 0) << ',' << r.panel << ',' << r.halo << ','
                      << (r.fits ? 1 : 0) << ',' << r.t_base * 1e3 << ',' << r.t_tiled * 1e3 << ','
                      << r.reuse << ',' << r.eff_bw << ',' << r.red_mgs << ',' << r.red_ca << ','
                      << s_certified << ',' << s_ghost << '\n';
                std::println("  [appended {} rows to {}]", sweep.size(), a.csv_path);
            }
            if (checksum == 0.0) std::println(std::cerr, "  (checksum {})", checksum);
            return EXIT_SUCCESS;
        }

        // The certified block width, measured on this start vector (the supply half of the
        // certificate): the s the kernel is allowed, never a fatter one.
        const int s_cert = std::max(1, measured_s_max(op.A, v0, /*ceiling=*/32));

        // Panel height and the pattern switch, both a priori (mpk.hpp). On the scattered arm
        // the switch refuses to tile, and the "tiled" arm is then the baseline by fiat.
        const MpkPlan plan = plan_mpk(mc, a.level, sp, s_cert, nnz_per_row, /*s_ceiling=*/64, /*cores_sharing=*/1);
        const int64_t tile = (plan.form == MpkForm::TILED && plan.panel_rows > 0)
                           ? plan.panel_rows : N;

        // Coordinates. rv_1core places the whole Arnoldi working set against one core's
        // share of the tiling level: the single-core vertical coordinate. rv_blocked is
        // the tiled arm's own coordinate (mpk.hpp): is a panel resident?
        const int64_t ws = arnoldi_working_set_bytes(nnz, N, a.m);
        const double cap = tile_capacity_bytes(mc, a.level, /*cores_sharing=*/1);
        const double rv_1core = static_cast<double>(ws) / cap;
        const double rv_blocked = mpk_tiled_R_v(mc, a.level, tile, s_cert, w, N, nnz_per_row, /*cores_sharing=*/1);

        std::println("regime-sweep -- vertical crossover, one core");
        std::println("  machine={}  pattern={}  tile-level={}", mc.key,
                     pattern_name(a.pattern), tile_level_name(a.level));
        std::println("  n1={} dim={}  N={}  nnz={}  m={}  s_cert={}  w={}",
                     a.n1, a.dim, N, nnz, a.m, s_cert, w);
        std::println("  working set={:.2f} MiB   tile capacity={:.2f} MiB   R_v(1 core)={:.3f}",
                     static_cast<double>(ws) / (1 << 20), cap / (1 << 20), rv_1core);
        std::println("  MPK form={} ({})  tile_rows={}  R_v(blocked)={:.3f}",
                     mpk_form_name(plan.form), plan.reason, tile, rv_blocked);

        double checksum = 0.0;
        const Stat t_base = [&] {
            std::vector<double> t;
            for (int r = 0; r < a.repeats + 1; ++r) {
                const auto t0 = Clock::now();
                const Eigen::MatrixXd B = build_baseline(A, v0, a.m);
                if (r > 0) t.push_back(Sec(Clock::now() - t0).count());
                checksum += B(N / 2, a.m);
            }
            return summarize(t);
        }();
        // Allocated once, before the timed repeats, and reused by every call: a real
        // solver preallocates its workspace once and reuses it across time-steps, and a
        // fair comparison against the baseline (which pays no comparable scratch at all)
        // must not let per-repeat allocation into the timed region either.
        Eigen::MatrixXd tile_scratch;
        const Stat t_tile = [&] {
            std::vector<double> t;
            for (int r = 0; r < a.repeats + 1; ++r) {
                const auto t0 = Clock::now();
                const Eigen::MatrixXd B =
                    (plan.form == MpkForm::TILED)
                        ? build_tiled(A, v0, a.m, s_cert, tile, w, tile_scratch)
                        : build_baseline(A, v0, a.m);   // scattered: no tiling, = baseline
                if (r > 0) t.push_back(Sec(Clock::now() - t0).count());
                checksum += B(N / 2, a.m);
            }
            return summarize(t);
        }();

        // Traffic model. Baseline reads the operator m times, tiled reads it once per block.
        const double op_bytes = static_cast<double>(matrix_bytes(nnz, N));
        const int    blocks   = (a.m + s_cert - 1) / s_cert;
        const double traffic_ratio = static_cast<double>(a.m) / static_cast<double>(blocks);

        const double flops = 2.0 * static_cast<double>(nnz) * static_cast<double>(a.m);
        const double gf_base = flops / t_base.median / 1e9;
        const double gf_tile = flops / t_tile.median / 1e9;
        // Effective DRAM bandwidth = operator bytes that must be fetched / time. Baseline
        // fetches op_bytes*m; tiled fetches op_bytes*blocks (once per block) if the reuse
        // worked. Comparing the achieved rate to the ceiling is the AI gate.
        const double eff_bw_base = op_bytes * a.m / t_base.median / 1e9;
        const double eff_bw_tile = op_bytes * blocks / t_tile.median / 1e9;
        const double speedup = t_base.median / t_tile.median;

        // AI gate: in the spilled regime the tiled kernel should approach the traffic ratio.
        // We pass the gate if either the point is cache-resident (nothing to gain, speedup
        // ~1 expected) or the tiled kernel realized most of the modeled reuse.
        const bool resident = rv_1core < 1.0;
        const bool ai_gate = (plan.form != MpkForm::TILED) || resident
                           || (speedup >= 0.6 * traffic_ratio);

        std::println("");
        std::println("  {:<12} {:>10} {:>10} {:>10}", "arm", "median(ms)", "GFLOP/s", "eff GB/s");
        std::println("  {:<12} {:>10.3f} {:>10.2f} {:>10.2f}",
                     "baseline", t_base.median * 1e3, gf_base, eff_bw_base);
        std::println("  {:<12} {:>10.3f} {:>10.2f} {:>10.2f}",
                     "tiled", t_tile.median * 1e3, gf_tile, eff_bw_tile);
        std::println("  speedup = {:.2f}x   (traffic ratio m/blocks = {:.2f}x, blocks={})",
                     speedup, traffic_ratio, blocks);
        if (plan.form == MpkForm::TILED && !resident)
            std::println("  AI gate: {} (tiled realized {:.0f}% of the modeled traffic cut)",
                         ai_gate ? "PASS" : "FAIL -- KERNEL BUG, not a finding",
                         100.0 * speedup / traffic_ratio);
        else
            std::println("  AI gate: n/a ({})",
                         resident ? "cache-resident, ~1x expected" : "scattered, no tiling");

        // Diagnostic, not a result: isolates whether the real tiled arm's flat GFLOP/s (vs
        // baseline) is the halo's redundant flops or the row-dot loop's own ceiling. Runs
        // only in the spilled regime, where there is something to explain.
        if (plan.form == MpkForm::TILED && !resident) {
            Eigen::MatrixXd nohalo_scratch;
            const Stat t_nohalo = [&] {
                std::vector<double> t;
                for (int r = 0; r < a.repeats + 1; ++r) {
                    const auto t0 = Clock::now();
                    const Eigen::MatrixXd B = build_tiled(A, v0, a.m, s_cert, tile, w,
                                                          nohalo_scratch, /*debug_no_halo=*/true);
                    if (r > 0) t.push_back(Sec(Clock::now() - t0).count());
                    checksum += B(N / 2, a.m);
                }
                return summarize(t);
            }();
            const double gf_nohalo = flops / t_nohalo.median / 1e9;
            std::println("");
            std::println("  [DEBUG no-halo, WRONG basis, throughput only]  {:.3f} ms   "
                         "{:.2f} GFLOP/s   (real tiled: {:.2f}, baseline: {:.2f})",
                         t_nohalo.median * 1e3, gf_nohalo, gf_tile, gf_base);

            // Structured landing spot for the diagnostic, kept in a separate file from
            // csv_path rather than extra columns on it: csv_path is the trusted result
            // (ai_gate_pass, speedup, everything a plot may treat as a placed point), and
            // mixing a column whose basis is known wrong into it risks a future script
            // reading that column as one more legitimate arm.
            if (!a.debug_csv_path.empty()) {
                const bool resident_verdict = gf_nohalo > 1.3 * gf_tile;
                const bool exists = std::filesystem::exists(a.debug_csv_path)
                                 && std::filesystem::file_size(a.debug_csv_path) > 0;
                std::ofstream f(a.debug_csv_path, std::ios::app);
                if (!f) throw std::runtime_error("Cannot open debug CSV: " + a.debug_csv_path);
                if (!exists)
                    f << "machine,pattern,tile_level,n1,N,rv_1core,gflops_baseline,"
                         "gflops_tiled,gflops_nohalo,cache_resident_verdict\n";
                f << mc.key << ',' << pattern_name(a.pattern) << ',' << tile_level_name(a.level)
                  << ',' << a.n1 << ',' << N << ',' << rv_1core << ',' << gf_base << ','
                  << gf_tile << ',' << gf_nohalo << ',' << (resident_verdict ? 1 : 0) << '\n';
            }
        }

        if (!a.csv_path.empty()) {
            const bool exists = std::filesystem::exists(a.csv_path)
                             && std::filesystem::file_size(a.csv_path) > 0;
            std::ofstream f(a.csv_path, std::ios::app);
            if (!f) throw std::runtime_error("Cannot open CSV: " + a.csv_path);
            if (!exists) f << kCsvHeader;
            f << mc.key << ',' << pattern_name(a.pattern) << ',' << tile_level_name(a.level) << ','
              << a.n1 << ',' << a.dim << ',' << N << ',' << nnz << ',' << a.m << ','
              << s_cert << ',' << tile << ',' << w << ',' << a.repeats << ','
              << static_cast<double>(ws) / (1 << 20) << ',' << cap / (1 << 20) << ','
              << rv_1core << ',' << rv_blocked << ',' << mpk_form_name(plan.form) << ','
              << t_base.median << ',' << t_base.q1 << ',' << t_base.q3 << ','
              << t_tile.median << ',' << t_tile.q1 << ',' << t_tile.q3 << ','
              << speedup << ',' << traffic_ratio << ',' << gf_base << ',' << gf_tile << ','
              << eff_bw_base << ',' << eff_bw_tile << ',' << (ai_gate ? 1 : 0) << '\n';
            std::println("  [appended to {}]", a.csv_path);
        }

        if (checksum == 0.0) std::println(std::cerr, "  (checksum {})", checksum);
    } catch (const std::exception& e) {
        std::println(std::cerr, "Error: {}", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
