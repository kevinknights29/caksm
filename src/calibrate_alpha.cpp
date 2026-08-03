/**
 * @file calibrate_alpha.cpp
 * @brief Offline measurement of the reduction cost model: the only empirically-sourced
 *        input in machine.hpp, and the gate on every horizontal-axis verdict.
 *
 * R_h's horizontal verdicts are gated on the magnitude this binary supplies: alpha comes
 * from a bare scalar reduction over an empty team, offline, with the solver not linked, so
 * it is known before the run it places. It shares include/reduction.hpp with
 * src/scaling.cpp, so alpha is the slope of the code that actually runs.
 *
 * The model replaced an earlier single-alpha fit to t = alpha*log2(P): the linear
 * primitive trended 2.2x in implied alpha, and a corrected tree still didn't rescue
 * alpha*log2(P), since levels are not equal (crossings) and eventually go free
 * (saturation). Only the crossing model is recorded; alpha is a property of one machine,
 * so --machine is checked against the host core count.
 *
 * Usage:
 *   OMP_PLACES=cores OMP_PROC_BIND=close \
 *     ./calibrate-alpha [--machine amd-3960x] [--reduce tree|linear]
 *                       [--reduces R] [--repeats K] [--layout shared|padded] [--csv PATH]
 *
 * @author Kevin Knights
 * @date 2026-07-17
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <print>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <omp.h>

#include "machine.hpp"
#include "reduction.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using Sec   = std::chrono::duration<double>;

[[nodiscard]] ReduceKind parse_reduce(std::string_view s)
{
    if (s == "tree")   return ReduceKind::TREE;
    if (s == "linear") return ReduceKind::LINEAR;
    throw std::invalid_argument("Unknown --reduce: " + std::string(s) + " (tree|linear)");
}

[[nodiscard]] Layout parse_layout(std::string_view s)
{
    if (s == "shared") return Layout::SHARED;
    if (s == "padded") return Layout::PADDED;
    throw std::invalid_argument("Unknown --layout: " + std::string(s) + " (shared|padded)");
}

/**
 * @brief Time `reduces` back-to-back scalar all-reduces over a bound team of P threads.
 *
 * One persistent parallel region for the whole batch, so fork/join amortizes to nothing.
 * The reduced value is carried into the next contribution (scaled to 1e-18) so the
 * dependency chain is real and the compiler cannot hoist the combine out of the loop.
 *
 * @return seconds per single reduction.
 */
[[nodiscard]] double time_reduces(int P, int64_t reduces, ReduceKind kind, Layout layout,
                                  double& sink)
{
    TeamReducer red(P, kind, layout);
    double total = 0.0;
    Clock::time_point t0;

    #pragma omp parallel num_threads(P)
    {
        const int t = omp_get_thread_num();
        double contrib = 1.0 + static_cast<double>(t);

        #pragma omp barrier          // every thread bound and warm before the clock starts
        #pragma omp masked
        { t0 = Clock::now(); }
        #pragma omp barrier

        for (int64_t it = 0; it < reduces; ++it) {
            const double acc = red.reduce(t, contrib);
            contrib = 1.0 + static_cast<double>(t) + acc * 1e-18;
        }

        #pragma omp barrier          // every thread done before the clock stops
        #pragma omp masked
        { total = Sec(Clock::now() - t0).count(); }

        #pragma omp critical
        { sink += contrib; }
    }

    return total / static_cast<double>(reduces);
}

// median / IQR over the timed repeats, matching src/scaling.cpp's summary
struct Stat {
    double median = 0.0;
    double q1     = 0.0;
    double q3     = 0.0;
};

[[nodiscard]] double quantile(std::vector<double> v, double q)  // by value: sorted here
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    if (v.size() == 1) return v[0];
    const double pos = q * static_cast<double>(v.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
    return v[lo] + (pos - static_cast<double>(lo)) * (v[hi] - v[lo]);
}

[[nodiscard]] Stat summarize(const std::vector<double>& v)
{
    return { quantile(v, 0.5), quantile(v, 0.25), quantile(v, 0.75) };
}

/// One calibration point: P, the measured per-reduction time, and the alpha it implies.
struct Point {
    int    P             = 0;
    Stat   t_reduce;               ///< seconds per reduction
    double alpha_implied = 0.0;    ///< t_median / log2(P); NaN at P=1 (no tree)
};

/// The models compared on every run. See fit_models() for what each is for.
struct Fit {
    double alpha_pinned = 0.0;  ///< t = alpha * log2(P): the model that was refuted
    double r2_pinned    = 0.0;
    double sse_pinned   = 0.0;
    double floor_affine = 0.0;  ///< t = floor + alpha * log2(P): floor diagnostic
    double alpha_affine = 0.0;
    double r2_affine    = 0.0;
    // The crossing model machine.hpp implements:
    //   t = t_intra * intra_levels(P) + t_cross * min(cross_levels(P), L*)
    double t_intra      = 0.0;
    double t_cross      = 0.0;
    double l_star       = 0.0;
    double r2_cross     = 0.0;
    double sse_cross    = 0.0;
};

/// Reduction-tree levels at P, split by whether the partner crosses a cache domain.
/// Mirrors intra_domain_levels/cross_domain_levels in machine.hpp exactly.
struct Levels { double intra = 0.0; double cross = 0.0; };

[[nodiscard]] Levels levels_at(const Machine& mc, int P)
{
    return { static_cast<double>(intra_domain_levels(mc, P)),
             static_cast<double>(cross_domain_levels(mc, P)) };
}

/**
 * @brief Fit all models over the P >= 2 points.
 *
 * P=1 is excluded from the log models: log2(1) = 0 carries no slope, and a P=1 "reduction"
 * is a thread reading its own deposit. The pinned R^2 is measured against sum t^2 (an
 * intercept-free model has no mean to regress toward), the affine against the mean, so the
 * two R^2 values are not comparable, compare the floor via floor_share() instead.
 */
[[nodiscard]] Fit fit_models(const std::vector<Point>& pts, const Machine& mc)
{
    Fit f;

    // Pinned: one parameter, through the origin.
    double num = 0.0, den = 0.0;
    for (const Point& p : pts) {
        if (p.P <= 1) continue;
        const double x = std::log2(static_cast<double>(p.P));
        num += p.t_reduce.median * x;
        den += x * x;
    }
    f.alpha_pinned = den > 0.0 ? num / den : 0.0;

    double ss_res = 0.0, ss_tot = 0.0;
    for (const Point& p : pts) {
        if (p.P <= 1) continue;
        const double pred = f.alpha_pinned * std::log2(static_cast<double>(p.P));
        ss_res += (p.t_reduce.median - pred) * (p.t_reduce.median - pred);
        ss_tot += p.t_reduce.median * p.t_reduce.median;
    }
    f.sse_pinned = ss_res;
    f.r2_pinned  = ss_tot > 0.0 ? 1.0 - ss_res / ss_tot : 0.0;

    // Crossing model: two costs and a saturation level, no intercept. Every P including P=1
    // is fitted (it is defined there: no levels, no cost). L* is swept rather than solved:
    // min() is non-linear in L* but linear in the two costs at fixed L*, so a sweep with a
    // least-squares solve inside is exact.
    double best_sse = std::numeric_limits<double>::infinity();
    for (double ls = 0.5; ls <= 4.0001; ls += 0.05) {
        // Normal equations for t ~ a*intra + b*cross_sat, both through the origin.
        double saa = 0.0, sab = 0.0, sbb = 0.0, sat = 0.0, sbt = 0.0;
        for (const Point& p : pts) {
            const Levels lv = levels_at(mc, p.P);
            const double a = lv.intra;
            const double b = std::min(lv.cross, ls);
            const double y = p.t_reduce.median;
            saa += a * a; sab += a * b; sbb += b * b; sat += a * y; sbt += b * y;
        }
        const double det = saa * sbb - sab * sab;
        if (std::abs(det) < 1e-30) continue;
        const double a_hat = (sat * sbb - sbt * sab) / det;
        const double b_hat = (sbt * saa - sat * sab) / det;

        double sse = 0.0;
        for (const Point& p : pts) {
            const Levels lv = levels_at(mc, p.P);
            const double pred = a_hat * lv.intra + b_hat * std::min(lv.cross, ls);
            sse += (p.t_reduce.median - pred) * (p.t_reduce.median - pred);
        }
        if (sse < best_sse) {
            best_sse = sse;
            f.t_intra = a_hat; f.t_cross = b_hat; f.l_star = ls; f.sse_cross = sse;
        }
    }
    {
        double mean = 0.0, n = 0.0;
        for (const Point& p : pts) { mean += p.t_reduce.median; n += 1.0; }
        mean = n > 0.0 ? mean / n : 0.0;
        double tot = 0.0;
        for (const Point& p : pts)
            tot += (p.t_reduce.median - mean) * (p.t_reduce.median - mean);
        f.r2_cross = tot > 0.0 ? 1.0 - f.sse_cross / tot : 0.0;
    }

    // Affine: ordinary least squares of t on log2(P).
    double n = 0.0, sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    for (const Point& p : pts) {
        if (p.P <= 1) continue;
        const double x = std::log2(static_cast<double>(p.P));
        const double y = p.t_reduce.median;
        n += 1.0; sx += x; sy += y; sxx += x * x; sxy += x * y;
    }
    const double d = n * sxx - sx * sx;
    if (n >= 2.0 && std::abs(d) > 0.0) {
        f.alpha_affine = (n * sxy - sx * sy) / d;
        f.floor_affine = (sy - f.alpha_affine * sx) / n;

        const double mean = sy / n;
        double res = 0.0, tot = 0.0;
        for (const Point& p : pts) {
            if (p.P <= 1) continue;
            const double x = std::log2(static_cast<double>(p.P));
            const double pred = f.floor_affine + f.alpha_affine * x;
            res += (p.t_reduce.median - pred) * (p.t_reduce.median - pred);
            tot += (p.t_reduce.median - mean) * (p.t_reduce.median - mean);
        }
        f.r2_affine = tot > 0.0 ? 1.0 - res / tot : 0.0;
    }
    return f;
}

/**
 * @brief Fraction of the reduction's cost that is the flat floor, at the widest P.
 *
 * Near 0 the cost is tree-bound and R_h's log2(P) numerator is real; near 1 it is
 * floor-bound and the numerator is nearly constant over the accessible range.
 *
 * Only meaningful when the floor is physical: a fitted floor below zero is an artifact of
 * fitting a straight line to data more concave than a logarithm, not a small floor. t(P=1)
 * is measured, so the honest floor is never negative. is_floor_physical() gates the reading.
 */
[[nodiscard]] double floor_share(const Fit& f, int P_max)
{
    const double tree = f.alpha_affine * std::log2(static_cast<double>(P_max));
    const double tot  = f.floor_affine + tree;
    return std::abs(tot) > 0.0 ? f.floor_affine / tot : 0.0;
}

/// Whether the affine fit's intercept can be read as a floor at all. See floor_share().
[[nodiscard]] bool is_floor_physical(const Fit& f) noexcept
{
    return f.floor_affine >= 0.0;
}

/**
 * @brief Does an extra tree level still cost anything at the top of the P range?
 *
 * Compares the measured cost across the widest pair of points differing by a full level of
 * depth. If a whole extra level is free the reduction has saturated (subtrees complete
 * concurrently), so alpha is the slope of a shape that is not a line in log2(P).
 *
 * @return measured cost of the last full level of depth, in seconds. Near zero or negative
 *         means saturated.
 */
[[nodiscard]] double top_level_cost(const std::vector<Point>& pts)
{
    // Walk back from the widest P to the last point whose tree is one level shallower.
    if (pts.size() < 2) return 0.0;
    const Point& top = pts.back();
    const double d_top = std::ceil(std::log2(static_cast<double>(top.P)));
    for (auto it = pts.rbegin(); it != pts.rend(); ++it) {
        if (it->P <= 1) continue;
        const double d = std::ceil(std::log2(static_cast<double>(it->P)));
        if (d < d_top) return top.t_reduce.median - it->t_reduce.median;
    }
    return 0.0;
}

// CLI
struct Args {
    std::string_view machine = "amd-3960x";
    ReduceKind reduce  = ReduceKind::TREE;
    int64_t    reduces = 20000;
    int        repeats = 7;
    Layout     layout  = Layout::SHARED;
    std::string csv_path;
};

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
        if      (arg == "--machine") a.machine = next();
        else if (arg == "--reduce")  a.reduce  = parse_reduce(next());
        else if (arg == "--reduces") a.reduces = std::stoll(std::string(next()));
        else if (arg == "--repeats") a.repeats = std::stoi(std::string(next()));
        else if (arg == "--layout")  a.layout  = parse_layout(next());
        else if (arg == "--csv")     a.csv_path = std::string(next());
        else if (arg == "--help") {
            std::println("Usage: OMP_PLACES=cores OMP_PROC_BIND=close \\");
            std::println("         ./calibrate-alpha [--machine KEY] [--reduce tree|linear]");
            std::println("                           [--reduces R] [--repeats K]");
            std::println("                           [--layout shared|padded] [--csv PATH]");
            std::println("");
            std::println("  --machine KEY  amd-3960x (default, and the only preset). Core count");
            std::println("                 and CCX geometry come from it and are checked against");
            std::println("                 the host: these costs do not transfer between machines.");
            std::println("  --reduce KIND  tree (default) = binary tree, point-to-point flag sync.");
            std::println("                 The primitive to calibrate against, and the one");
            std::println("                 src/scaling.cpp runs. linear = barrier + O(P) redundant");
            std::println("                 scan; the old artifact, kept only so the correction can");
            std::println("                 be measured. Do NOT record a linear alpha.");
            std::println("  --reduces R    reductions per timed repeat (default 20000). Raise until");
            std::println("                 the per-P IQR is small against the median.");
            std::println("  --repeats K    timed repeats after 1 warm-up (default 7)");
            std::println("  --layout L     LINEAR only: shared (default) packs the partials,");
            std::println("                 padded gives each thread a line. The tree ignores this.");
            std::println("  --csv PATH     write one row per P (file truncated and re-headed)");
            std::exit(0);
        }
        else throw std::invalid_argument("Unknown flag: " + std::string(arg));
    }
    if (a.reduces < 1)  throw std::invalid_argument("--reduces must be >= 1");
    if (a.repeats < 1)  throw std::invalid_argument("--repeats must be >= 1");
    return a;
}

/**
 * @brief The P grid: 1, then every point that engages one more whole LLC slice.
 *
 * Under OMP_PROC_BIND=close a team fills one slice before engaging the next, so on puffin
 * these are P = 3, 6, ..., 24. Off-grid points straddle a slice boundary, which is noise in
 * a slope fit. P=1 is measured but not fitted: the zero-communication anchor.
 */
[[nodiscard]] std::vector<int> ccx_aligned_grid(const Machine& mc)
{
    std::vector<int> grid{1};
    for (int P = mc.cores_per_llc_slice; P <= mc.cores; P += mc.cores_per_llc_slice)
        grid.push_back(P);
    return grid;
}

const char* kCsvHeader =
    "machine,reduce,layout,P,slices,reduces,repeats,"
    "t_reduce_s,t_reduce_q1,t_reduce_q3,alpha_implied_s,"
    "intra_levels,cross_levels,"
    "t_intra_s,t_cross_s,l_star,r2_cross,sse_cross,crossing_multiplier,"
    "alpha_pinned_s,r2_pinned,sse_pinned,floor_affine_s,alpha_affine_s,r2_affine,"
    "floor_share,floor_physical,top_level_s,saturated,places,bind\n";

}  // namespace

int main(int argc, char* argv[])
{
    try {
        const Args a = parse_args(
            std::span<const char* const>(argv, static_cast<std::size_t>(argc)));
        const Machine& mc = lookup_machine(a.machine);

        const char* places = std::getenv("OMP_PLACES");
        const char* bind   = std::getenv("OMP_PROC_BIND");
        const int   avail  = omp_get_max_threads();

        std::println("alpha calibration -- reduction latency per tree level");
        std::println("  machine={} ({})", mc.key, mc.name);
        std::println("  cores={}  cores/slice={}  reduce={}  reduces={}  repeats={}  layout={}",
                     mc.cores, mc.cores_per_llc_slice, reduce_kind_name(a.reduce),
                     a.reduces, a.repeats, layout_name(a.layout));
        std::println("  OMP_PLACES={}  OMP_PROC_BIND={}",
                     places ? places : "(unset)", bind ? bind : "(unset)");

        if (avail < mc.cores)
            throw std::runtime_error(
                std::format("Host offers {} threads but preset '{}' expects {} cores. "
                            "alpha does not transfer between machines: run this on {}.",
                            avail, mc.key, mc.cores, mc.name));
        if (!places || !bind)
            std::println("  WARNING: unbound team. Set OMP_PLACES=cores OMP_PROC_BIND=close; "
                         "an unbound tree's depth is not log2(P), and the tree spins rather "
                         "than blocking, so an oversubscribed team can livelock.");

        // Verify before timing: a subtly wrong tree returns a plausible number slightly
        // too small, and every downstream figure inherits it.
        std::println("");
        std::print("  [Verifying the {} reduction at every P...] ", reduce_kind_name(a.reduce));
        for (const int P : ccx_aligned_grid(mc)) {
            const std::string err = verify_reducer(P, a.reduce, a.layout);
            if (!err.empty()) throw std::runtime_error(err);
        }
        std::println("ok");

        std::vector<Point> pts;
        double sink = 0.0;

        std::println("");
        std::println("  {:>3}  {:>6}  {:>12}  {:>12}  {:>12}  {:>12}",
                     "P", "slices", "t_red (ns)", "q1 (ns)", "q3 (ns)", "alpha (ns)");
        std::println("  {}", std::string(68, '-'));

        for (const int P : ccx_aligned_grid(mc)) {
            // Warm-up, discarded: first-touch of the slots, threads settling onto their
            // places, and the boost ramp.
            (void)time_reduces(P, a.reduces, a.reduce, a.layout, sink);

            std::vector<double> samples;
            samples.reserve(static_cast<std::size_t>(a.repeats));
            for (int k = 0; k < a.repeats; ++k)
                samples.push_back(time_reduces(P, a.reduces, a.reduce, a.layout, sink));

            Point pt;
            pt.P        = P;
            pt.t_reduce = summarize(samples);
            pt.alpha_implied = (P > 1)
                ? pt.t_reduce.median / std::log2(static_cast<double>(P))
                : std::numeric_limits<double>::quiet_NaN();
            pts.push_back(pt);

            std::println("  {:>3}  {:>6}  {:>12.1f}  {:>12.1f}  {:>12.1f}  {:>12.1f}",
                         P, engaged_llc_slices(mc, P),
                         pt.t_reduce.median * 1e9, pt.t_reduce.q1 * 1e9,
                         pt.t_reduce.q3 * 1e9, pt.alpha_implied * 1e9);
        }

        const Fit f = fit_models(pts, mc);
        const double share = floor_share(f, mc.cores);

        std::println("  {}", std::string(68, '-'));
        std::println("");
        std::println("=== crossing: t = t_intra*intra + t_cross*min(cross, L*) ===");
        std::println("||  t_intra = {:.1f} ns/level    t_cross = {:.1f} ns/level    crossing x{:.1f}",
                     f.t_intra * 1e9, f.t_cross * 1e9,
                     f.t_intra > 0.0 ? f.t_cross / f.t_intra : 0.0);
        std::println("||  L* = {:.2f}    SSE = {:.3e}    R^2 = {:.4f}",
                     f.l_star, f.sse_cross, f.r2_cross);
        std::println("");
        std::println("=== pinned: t(P) = alpha * log2(P) ===");
        std::println("||  alpha = {:.4e} s ({:.1f} ns/level)    SSE = {:.3e}    R^2 = {:.4f}",
                     f.alpha_pinned, f.alpha_pinned * 1e9, f.sse_pinned, f.r2_pinned);
        std::println("");
        std::println("=== affine: t(P) = floor + alpha * log2(P) ===");
        std::println("||  floor = {:.1f} ns    alpha = {:.1f} ns/level    R^2 = {:.4f}",
                     f.floor_affine * 1e9, f.alpha_affine * 1e9, f.r2_affine);
        std::println("||  floor physical: {}", is_floor_physical(f) ? "yes" : "no");
        if (is_floor_physical(f))
            std::println("||  floor share at P={}: {:.0f}%", mc.cores, share * 100.0);
        std::println("===");
        std::println("");

        // Saturation check: whether an extra tree level still costs anything at the top of
        // the P range, recorded to the CSV alongside the fit.
        const double top = top_level_cost(pts);
        const bool saturated = top < 0.25 * f.alpha_pinned;

        if (a.reduce == ReduceKind::TREE) {
            std::println("To record, set these three on the {} preset in include/machine.hpp:",
                         mc.key);
            std::println("  t_level_intra_s  = {:.4e}", f.t_intra);
            std::println("  t_level_cross_s  = {:.4e}", f.t_cross);
            std::println("  cross_saturation = {:.2f}", f.l_star);
            std::println("  reduction_calibrated = true");
            std::println("");
            std::println("The pinned alpha is printed for comparison only.");
        } else {
            std::println("--reduce linear is the artifact, measured only to size the correction.");
        }

        if (!a.csv_path.empty()) {
            std::ofstream out(a.csv_path, std::ios::trunc);
            if (!out) throw std::runtime_error("Cannot open CSV: " + a.csv_path);
            out << kCsvHeader;
            for (const Point& p : pts) {
                out << mc.key << ',' << reduce_kind_name(a.reduce) << ','
                    << layout_name(a.layout) << ','
                    << p.P << ',' << engaged_llc_slices(mc, p.P) << ','
                    << a.reduces << ',' << a.repeats << ','
                    << p.t_reduce.median << ',' << p.t_reduce.q1 << ',' << p.t_reduce.q3 << ','
                    << p.alpha_implied << ','
                    << intra_domain_levels(mc, p.P) << ','
                    << cross_domain_levels(mc, p.P) << ','
                    << f.t_intra << ',' << f.t_cross << ',' << f.l_star << ','
                    << f.r2_cross << ',' << f.sse_cross << ','
                    << (f.t_intra > 0.0 ? f.t_cross / f.t_intra : 0.0) << ','
                    << f.alpha_pinned << ',' << f.r2_pinned << ',' << f.sse_pinned << ','
                    << f.floor_affine << ',' << f.alpha_affine << ',' << f.r2_affine << ','
                    << share << ','
                    << (is_floor_physical(f) ? 1 : 0) << ',' << top << ','
                    << (saturated ? 1 : 0) << ','
                    << (places ? places : "") << ',' << (bind ? bind : "") << '\n';
            }
            std::println("");
            std::println("  [Wrote {} rows to {}]", pts.size(), a.csv_path);
        }

        // Keep the reduction chain observable so it cannot be optimized away.
        if (sink == 0.0) std::println(std::cerr, "  (sink {})", sink);

    } catch (const std::exception& e) {
        std::println(std::cerr, "Error: {}", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
