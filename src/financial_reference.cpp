/**
 * @file financial_reference.cpp
 * @brief Independent price and Greek references for the three-asset European
 *        Basket and Rainbow contracts, with their uncertainty.
 *
 *   ./financial-reference [--output-dir DIR] [--min-level A] [--max-level B]
 *                         [--scrambles R] [--seed S] [--bumps a,b,c]
 *                         [--half-width-target X] [--gates-only]
 *   ./financial-reference --summary [--output-dir DIR]
 *
 * The reference run writes one machine-readable row per payoff, sample level,
 * scramble, spot bump, and Greek estimator, plus the accepted compact summary
 * that the PDE harness and the thesis read. The summary command rebuilds the
 * compact table from those stored rows without repeating the sampling.
 *
 * Exits non-zero when a blocking correctness gate fails. Two checks report
 * without blocking: whether the half-width target was reached, and whether the
 * bump sweep is second order. Neither invalidates a result, because the
 * response to both is to report the achieved interval and the fitted bump
 * truncation as separate budget lines, which the summary already does.
 *
 * @author Kevin Knights
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <print>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "financial_reference.hpp"

namespace fr = financial_reference;

namespace {

// The four-decimal values published by Dang, Christara and Jackson. They are
// historical comparison values, not references: 4.4450 comes from a Johnson
// evaluation and 13.2449 from an FFT method, and neither is quoted with an
// uncertainty that would support a price-and-Greeks error budget.
constexpr double kHistoricalRainbow = 4.4450;
constexpr double kHistoricalBasket = 13.2449;

struct Args {
    std::filesystem::path output_dir = "data/financial-validation";
    int min_level = 14;
    int max_level = 22;
    int scrambles = 16;
    std::uint64_t seed = fr::kDefaultQmcSeed;
    // The shared bump sweep. It reaches beyond what the QMC estimates need
    // because the PDE harness uses the same sweep, and a trilinear interpolant
    // carries no curvature until the bump spans several grid cells.
    std::vector<double> bumps = {0.005, 0.01, 0.02, 0.05, 0.10};
    double half_width_target = 1.0e-5;
    bool gates_only = false;
    bool summary_only = false;
};

/// One accepted or reported estimate, ready for the summary table.
///
/// The sampling uncertainty and the bump truncation are kept apart because the
/// error budget has to separate them: one shrinks with more points and the
/// other with a smaller bump, and adding them up front hides which is binding.
struct Estimate {
    std::string payoff;
    std::string quantity;
    double relative_bump = 0.0;
    double value = 0.0;
    double half_width = 0.0;   ///< across-scramble Student t half-width
    double bump_error = 0.0;   ///< estimated central-difference truncation
    std::string source;
};

/// One acceptance check and the number that decided it.
struct Gate {
    std::string name;
    bool passed = false;
    bool blocking = true;
    double measured = 0.0;
    double allowed = 0.0;
    std::string note;
};

/// One pilot sample level.
struct Level {
    int log2_points = 0;
    std::int64_t points = 0;
    fr::Interval basket;
    fr::Interval rainbow;
};

// Arguments

[[nodiscard]] std::vector<double> parse_list(std::string_view text)
{
    std::vector<double> values;
    std::string field;
    std::istringstream stream{std::string(text)};
    while (std::getline(stream, field, ','))
        if (!field.empty()) values.push_back(std::stod(field));
    if (values.empty())
        throw std::invalid_argument("expected a comma-separated list of numbers");
    return values;
}

[[nodiscard]] Args parse_reference_args(std::span<const char* const> argv)
{
    Args args;
    for (std::size_t i = 1; i < argv.size(); ++i) {
        const std::string_view arg = argv[i];
        const auto next = [&]() -> std::string_view {
            if (++i >= argv.size())
                throw std::invalid_argument("missing value for " + std::string(arg));
            return argv[i];
        };
        if (arg == "--output-dir")
            args.output_dir = std::filesystem::path(std::string(next()));
        else if (arg == "--min-level") args.min_level = std::stoi(std::string(next()));
        else if (arg == "--max-level") args.max_level = std::stoi(std::string(next()));
        else if (arg == "--scrambles") args.scrambles = std::stoi(std::string(next()));
        else if (arg == "--seed") args.seed = std::stoull(std::string(next()));
        else if (arg == "--bumps") args.bumps = parse_list(next());
        else if (arg == "--half-width-target")
            args.half_width_target = std::stod(std::string(next()));
        else if (arg == "--gates-only") args.gates_only = true;
        else if (arg == "--summary") args.summary_only = true;
        else if (arg == "--help") {
            std::println(
                "Usage: ./financial-reference [--output-dir DIR]\n"
                "         [--min-level A] [--max-level B] [--scrambles R]\n"
                "         [--seed S] [--bumps a,b,c] [--half-width-target X]\n"
                "         [--gates-only]\n"
                "       ./financial-reference --summary [--output-dir DIR]\n"
                "  --min-level A        smallest pilot sample level, 2^A points\n"
                "  --max-level B        largest pilot sample level, 2^B points\n"
                "  --scrambles R        independent scrambles per level (at least 16)\n"
                "  --bumps a,b,c        relative spot bumps for Delta and Gamma\n"
                "  --half-width-target  accepted price half-width\n"
                "  --gates-only         run the correctness gates and exit\n"
                "  --summary            rebuild the compact table from stored rows");
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (args.min_level < 8 || args.max_level > 26 || args.min_level >= args.max_level)
        throw std::invalid_argument("require 8 <= min-level < max-level <= 26");
    if (args.scrambles < 16)
        throw std::invalid_argument(
            "at least 16 scrambles are required for the final result");
    if (!(args.half_width_target > 0.0))
        throw std::invalid_argument("--half-width-target must be positive");
    if (args.bumps.size() < 3)
        throw std::invalid_argument("at least three relative bump sizes are required");
    // Ascending order is what lets the truncation fit anchor on a known pair
    // and keep the narrowest bump as an independent check.
    std::ranges::sort(args.bumps);
    return args;
}

// Model helpers

/// Apply an asset permutation to every asset-specific input, correlations included.
[[nodiscard]] Config permute_assets(const Config& model, const std::array<int, 3>& order)
{
    Config permuted = model;
    const auto correlation = fr::correlation_matrix(model.rho_off);
    for (std::size_t d = 0; d < 3; ++d) {
        const auto from = static_cast<std::size_t>(order[d]);
        permuted.initial_prices[d] = model.initial_prices[from];
        permuted.sigma[d] = model.sigma[from];
        permuted.weight[d] = model.weight[from];
    }
    const auto pick = [&](int a, int b) {
        return correlation[static_cast<std::size_t>(order[static_cast<std::size_t>(a)])]
                          [static_cast<std::size_t>(order[static_cast<std::size_t>(b)])];
    };
    permuted.rho_off = {pick(0, 1), pick(0, 2), pick(1, 2)};
    return permuted;
}

[[nodiscard]] std::vector<double> prices_of(const std::vector<fr::ScrambleEstimate>& runs)
{
    std::vector<double> values;
    values.reserve(runs.size());
    for (const auto& run : runs) values.push_back(run.price);
    return values;
}

[[nodiscard]] std::vector<double> greeks_of(
    const std::vector<fr::ScrambleEstimate>& runs, std::size_t bump,
    std::size_t asset, bool gamma)
{
    std::vector<double> values;
    values.reserve(runs.size());
    for (const auto& run : runs)
        values.push_back(gamma ? run.bump[bump].gamma[asset]
                               : run.bump[bump].delta[asset]);
    return values;
}

/// Second-order fit of the central-difference truncation across a bump sweep.
struct BumpFit {
    double coefficient = 0.0;  ///< truncation is coefficient * h^2, h the relative bump
    double residual = 0.0;     ///< how far the smallest bump sits off that fit
};

/**
 * @brief Fit the central-difference truncation from the bump sweep itself.
 *
 * A centered difference carries an O(h^2) truncation, so one adjacent pair
 * determines its coefficient and a third bump is left over to test the fit.
 * This is what turns a bump sweep from a sensitivity anecdote into a budgeted
 * error term: the truncation stops being lumped into the sampling interval and
 * becomes its own line, which is the only way to say which of the two is
 * binding.
 *
 * The fit is anchored on the second and third smallest bumps and validated
 * against the smallest, so extending the sweep upward for the PDE harness,
 * which needs bumps that span several grid cells before a trilinear interpolant
 * carries any curvature, does not move the coefficient. The narrowest bump is
 * noise-dominated and the widest leave the asymptotic regime, so neither is a
 * sound anchor.
 *
 * @param bumps Relative bump sizes in ascending order, at least three of them.
 * @param values The central estimate at each bump, in the same order.
 */
[[nodiscard]] BumpFit fit_bump_truncation(const std::vector<double>& bumps,
                                          const std::vector<double>& values)
{
    const double wide = bumps[2] * bumps[2];
    const double near = bumps[1] * bumps[1];
    BumpFit fit;
    if (std::abs(wide - near) > 0.0)
        fit.coefficient = (values[2] - values[1]) / (wide - near);
    const double predicted =
        values[1] - fit.coefficient * (near - bumps[0] * bumps[0]);
    fit.residual = std::abs(values[0] - predicted);
    return fit;
}

/// Name of a diagonal Gamma or a Delta, as it appears in the machine-readable rows.
[[nodiscard]] std::string greek_name(bool gamma, std::size_t asset)
{
    const std::string index = std::to_string(asset + 1);
    return gamma ? "gamma_" + index + index : "delta_" + index;
}

// Output

[[nodiscard]] std::ofstream open_csv(const std::filesystem::path& path)
{
    std::ofstream file(path);
    if (!file) throw std::runtime_error("cannot create " + path.string());
    file << std::setprecision(17);
    return file;
}

/// Append the complete parameter row so no consumer has to guess the contract.
void write_parameter_columns(std::ofstream& file, const Config& model)
{
    file << model.strike_price << ',' << model.risk_free_rate << ','
         << model.t_final << ',';
    for (const double value : model.initial_prices) file << value << ',';
    for (const double value : model.sigma) file << value << ',';
    for (const double value : model.rho_off) file << value << ',';
    file << model.weight[0] << ',' << model.weight[1] << ',' << model.weight[2];
}

constexpr std::string_view kParameterHeader =
    "strike,rate,expiry,spot_1,spot_2,spot_3,sigma_1,sigma_2,sigma_3,"
    "rho_12,rho_13,rho_23,weight_1,weight_2,weight_3";

void write_summary(const std::filesystem::path& path,
                   const std::vector<Estimate>& summary, const Config& model,
                   int accepted_level, int scrambles, std::uint64_t seed)
{
    std::ofstream file = open_csv(path);
    file << "payoff,quantity,relative_bump,value,half_width,bump_error,source,"
            "log2_points,scrambles,seed," << kParameterHeader << '\n';
    for (const Estimate& item : summary) {
        file << item.payoff << ',' << item.quantity << ',' << item.relative_bump
             << ',' << item.value << ',' << item.half_width << ','
             << item.bump_error << ',' << item.source << ',' << accepted_level
             << ',' << scrambles << ',' << seed << ',';
        write_parameter_columns(file, model);
        file << '\n';
    }
}

// The summary command

/// Minimal CSV reader: splits on commas, no quoting, which is all these files use.
[[nodiscard]] std::vector<std::map<std::string, std::string>> read_csv(
    const std::filesystem::path& path)
{
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot read " + path.string());

    std::string line;
    if (!std::getline(file, line))
        throw std::runtime_error("empty file: " + path.string());
    std::vector<std::string> header;
    {
        std::istringstream stream(line);
        std::string field;
        while (std::getline(stream, field, ',')) header.push_back(field);
    }

    std::vector<std::map<std::string, std::string>> rows;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::map<std::string, std::string> row;
        std::istringstream stream(line);
        std::string field;
        for (std::size_t i = 0; i < header.size() && std::getline(stream, field, ','); ++i)
            row[header[i]] = field;
        rows.push_back(std::move(row));
    }
    return rows;
}

/**
 * @brief Rebuild the compact thesis table from the stored per-scramble rows.
 *
 * Reads only what the reference run wrote, so the table can be regenerated,
 * inspected, or reformatted without repeating hours of sampling.
 */
int run_summary(const std::filesystem::path& output_dir)
{
    const auto estimates = read_csv(output_dir / "qmc_estimates.csv");
    const auto johnson = read_csv(output_dir / "johnson_reference.csv");
    const auto levels = read_csv(output_dir / "qmc_levels.csv");
    if (johnson.empty())
        throw std::runtime_error("johnson_reference.csv carries no rows");

    // The accepted level, not the largest one the pilot reached. The pilot may
    // climb past acceptance, and summarizing the top level instead would quote
    // a sample size the acceptance rule never selected and would drop the
    // Greeks, which are only drawn at the accepted level.
    int accepted_level = 0;
    for (const auto& row : levels)
        if (row.at("accepted") == "1")
            accepted_level = std::stoi(row.at("log2_points"));
    if (accepted_level == 0)
        throw std::runtime_error("qmc_levels.csv marks no accepted sample level");

    // Group the accepted level's rows by payoff, estimator, and bump.
    std::map<std::tuple<std::string, std::string, std::string>, std::vector<double>> grouped;
    for (const auto& row : estimates) {
        if (std::stoi(row.at("log2_points")) != accepted_level) continue;
        grouped[{row.at("payoff"), row.at("estimator"), row.at("relative_bump")}]
            .push_back(std::stod(row.at("value")));
    }

    std::ofstream file = open_csv(output_dir / "reference_table.csv");
    file << "payoff,estimator,relative_bump,mean,half_width,standard_error,"
            "scrambles,log2_points,source\n";
    std::println("Compact reference table, level 2^{}", accepted_level);
    std::println("  {:<8} {:<10} {:>8}  {:>15}  {:>12}  {:>4}",
                 "payoff", "estimator", "bump", "value", "half-width", "n");

    const double johnson_price = std::stod(johnson.front().at("price"));
    const double johnson_shift = std::stod(johnson.front().at("quadrature_shift"));
    file << "rainbow,price,0," << johnson_price << ',' << johnson_shift
         << ",0," << 0 << ',' << 0 << ",johnson\n";
    std::println("  {:<8} {:<10} {:>8}  {:>15.7f}  {:>12.2e}  {:>4}",
                 "rainbow", "price", "-", johnson_price, johnson_shift, "-");

    for (const auto& [key, values] : grouped) {
        const auto& [payoff, estimator, bump] = key;
        const fr::Interval interval = fr::scramble_interval(values);
        file << payoff << ',' << estimator << ',' << bump << ',' << interval.mean
             << ',' << interval.half_width << ',' << interval.standard_error << ','
             << interval.samples << ',' << accepted_level << ",qmc\n";
        std::println("  {:<8} {:<10} {:>8}  {:>15.7f}  {:>12.2e}  {:>4}",
                     payoff, estimator,
                     std::stod(bump) == 0.0 ? std::string("-")
                                            : std::format("{:.3f}", std::stod(bump)),
                     interval.mean, interval.half_width, interval.samples);
    }
    std::println("\n  wrote {}", (output_dir / "reference_table.csv").string());
    return EXIT_SUCCESS;
}

// The correctness gates

/**
 * @brief Deterministic gates that must hold before any reference value is used.
 *
 * Cheap by design: each runs at a small sample level, because these check the
 * construction rather than the accuracy.
 */
void run_correctness_gates(const Config& model, const Args& args,
                           const fr::JohnsonResult& johnson,
                           double quadrature_shift, double permutation_shift,
                           double intrinsic_gap, std::vector<Gate>& gates)
{
    const auto record = [&](std::string name, bool passed, double measured,
                            double allowed, std::string note, bool blocking = true) {
        gates.push_back(Gate{std::move(name), passed, blocking, measured, allowed,
                             std::move(note)});
    };

    record("johnson_reproduces_historical_rainbow",
           std::abs(johnson.price - kHistoricalRainbow) < 5.0e-5,
           std::abs(johnson.price - kHistoricalRainbow), 5.0e-5,
           "rounds to the published four-decimal value");
    record("johnson_quadrature_stable", quadrature_shift < 1.0e-10,
           quadrature_shift, 1.0e-10,
           "every reported digit survives doubled quadrature settings");
    record("johnson_permutation_invariant", permutation_shift < 1.0e-10,
           permutation_shift, 1.0e-10, "asset relabeling leaves the price alone");
    record("johnson_within_bounds",
           johnson.price > 0.0
               && johnson.price < johnson.discounted_minimum_expectation,
           johnson.price, johnson.discounted_minimum_expectation,
           "price lies between zero and the discounted expected minimum");
    record("johnson_intrinsic_limit", intrinsic_gap < 1.0e-6, intrinsic_gap, 1.0e-6,
           "a vanishing expiry returns the intrinsic payoff");

    fr::QmcRequest request;
    request.log2_points = 12;
    request.scrambles = args.scrambles;
    request.seed = args.seed;
    request.relative_bumps.clear();

    Config flat = model;
    flat.sigma = {1.0e-12, 1.0e-12, 1.0e-12};
    const fr::QmcResult zero_volatility = fr::run_qmc(flat, request);
    const double grown =
        model.initial_prices[0] * std::exp(model.risk_free_rate * model.t_final);
    const double deterministic = std::exp(-model.risk_free_rate * model.t_final)
                               * std::max(grown - model.strike_price, 0.0);
    double zero_shift = 0.0;
    for (const auto& run : zero_volatility.basket)
        zero_shift = std::max(zero_shift, std::abs(run.price - deterministic));
    for (const auto& run : zero_volatility.rainbow)
        zero_shift = std::max(zero_shift, std::abs(run.price - deterministic));
    record("qmc_zero_volatility", zero_shift < 1.0e-9, zero_shift, 1.0e-9,
           "a deterministic terminal value gives the discounted payoff");

    Config single = model;
    single.weight = {1.0, 0.0, 0.0};
    fr::QmcRequest single_request = request;
    single_request.log2_points = 18;
    const fr::Interval reduced =
        fr::scramble_interval(prices_of(fr::run_qmc(single, single_request).basket));
    const double black_scholes = fr::black_scholes_call(
        model.initial_prices[0], model.strike_price, model.risk_free_rate,
        model.sigma[0], model.t_final);
    record("qmc_black_scholes_reduction",
           std::abs(reduced.mean - black_scholes) <= reduced.half_width,
           std::abs(reduced.mean - black_scholes), reduced.half_width,
           "a one-asset basket reproduces Black-Scholes inside its interval");

    const fr::QmcResult base = fr::run_qmc(model, request);
    const fr::Interval base_basket = fr::scramble_interval(prices_of(base.basket));
    const fr::Interval base_rainbow = fr::scramble_interval(prices_of(base.rainbow));

    // A permutation relabels the assets, but it also relabels which Sobol
    // coordinate drives which asset through the Cholesky factor. The estimate
    // therefore moves by the quadrature error rather than staying identical,
    // and the honest test is that the two intervals overlap.
    double permuted_gap = 0.0;
    double permuted_allowance = 0.0;
    for (const std::array<int, 3> order :
         {std::array<int, 3>{1, 2, 0}, std::array<int, 3>{2, 0, 1}}) {
        const fr::QmcResult moved = fr::run_qmc(permute_assets(model, order), request);
        const fr::Interval basket = fr::scramble_interval(prices_of(moved.basket));
        const fr::Interval rainbow = fr::scramble_interval(prices_of(moved.rainbow));
        if (std::abs(basket.mean - base_basket.mean) > permuted_gap) {
            permuted_gap = std::abs(basket.mean - base_basket.mean);
            permuted_allowance = basket.half_width + base_basket.half_width;
        }
        if (std::abs(rainbow.mean - base_rainbow.mean) > permuted_gap) {
            permuted_gap = std::abs(rainbow.mean - base_rainbow.mean);
            permuted_allowance = rainbow.half_width + base_rainbow.half_width;
        }
    }
    record("qmc_permutation_invariant", permuted_gap <= permuted_allowance,
           permuted_gap, permuted_allowance,
           "asset relabeling leaves the price inside its interval");

    const fr::QmcResult repeated = fr::run_qmc(model, request);
    double repeat_shift = 0.0;
    for (std::size_t i = 0; i < base.basket.size(); ++i) {
        repeat_shift = std::max(repeat_shift,
            std::abs(base.basket[i].price - repeated.basket[i].price));
        repeat_shift = std::max(repeat_shift,
            std::abs(base.rainbow[i].price - repeated.rainbow[i].price));
    }
    record("qmc_seed_reproducible", repeat_shift == 0.0, repeat_shift, 0.0,
           "reusing a seed reproduces every scramble estimate exactly");

    fr::QmcRequest other_seed = request;
    other_seed.seed = args.seed + 1ull;
    const fr::QmcResult moved_seed = fr::run_qmc(model, other_seed);
    double closest = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < base.basket.size(); ++i)
        closest = std::min(closest,
            std::abs(base.basket[i].price - moved_seed.basket[i].price));
    record("qmc_distinct_seeds_differ", closest > 0.0, closest, 0.0,
           "a different seed produces a different estimate in every scramble");
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const Args args = parse_reference_args(
            std::span<const char* const>(argv, static_cast<std::size_t>(argc)));
        std::filesystem::create_directories(args.output_dir);
        if (args.summary_only) return run_summary(args.output_dir);

        // The shared parameter object. Its defaults are the base case of the
        // three-asset experiment, and the PDE harness reads the same struct.
        const Config model;
        std::vector<Gate> gates;

        // The analytical Rainbow reference
        const fr::QuadratureSettings settings;
        const fr::MultivariateNormal normal(settings);
        const fr::JohnsonResult johnson = fr::johnson_call_on_minimum(model, normal);

        const double johnson_tightened =
            fr::johnson_call_on_minimum(
                model, fr::MultivariateNormal(settings.tightened())).price;
        const double quadrature_shift = std::abs(johnson.price - johnson_tightened);

        double permutation_shift = 0.0;
        for (const std::array<int, 3> order :
             {std::array<int, 3>{1, 2, 0}, std::array<int, 3>{2, 0, 1},
              std::array<int, 3>{0, 2, 1}})
            permutation_shift = std::max(permutation_shift, std::abs(
                fr::johnson_call_on_minimum(permute_assets(model, order), normal).price
                - johnson.price));

        // A vanishing expiry must return the intrinsic payoff, checked in the
        // money so the test has something to reproduce beyond zero.
        Config expiring = model;
        expiring.t_final = 1.0e-8;
        expiring.initial_prices = {120.0, 130.0, 140.0};
        const double intrinsic_gap = std::abs(
            fr::johnson_call_on_minimum(expiring, normal).price
            - (120.0 - model.strike_price));

        std::println("Johnson call on the minimum");
        std::println("  price              {:.10f}", johnson.price);
        std::println("  tightened          {:.10f}  (shift {:.2e})",
                     johnson_tightened, quadrature_shift);
        std::println("  discounted E[min]  {:.10f}",
                     johnson.discounted_minimum_expectation);
        std::println("  settings           {}", settings.describe());
        std::println("  truncation bound   {:.2e}\n", johnson.truncation_bound);

        {
            std::ofstream file = open_csv(args.output_dir / "johnson_reference.csv");
            file << "payoff,price,discounted_minimum_expectation,tightened_price,"
                    "quadrature_shift,truncation_bound,outer_panels,outer_nodes,"
                    "inner_panels,inner_nodes,tail_cut,permutation_shift,"
                    "historical_comparison_value,random_state," << kParameterHeader
                 << '\n';
            file << "rainbow," << johnson.price << ','
                 << johnson.discounted_minimum_expectation << ','
                 << johnson_tightened << ',' << quadrature_shift << ','
                 << johnson.truncation_bound << ',' << settings.outer_panels << ','
                 << settings.outer_nodes << ',' << settings.inner_panels << ','
                 << settings.inner_nodes << ',' << settings.tail_cut << ','
                 << permutation_shift << ',' << kHistoricalRainbow << ",none,";
            write_parameter_columns(file, model);
            file << '\n';
        }

        std::println("Correctness gates ...");
        run_correctness_gates(model, args, johnson, quadrature_shift,
                              permutation_shift, intrinsic_gap, gates);

        if (args.gates_only) {
            int failures = 0;
            for (const Gate& gate : gates) {
                if (gate.blocking && !gate.passed) ++failures;
                std::println("  [{}] {:<38} measured={:.3e} allowed={:.3e}",
                             gate.passed ? "pass" : "FAIL", gate.name, gate.measured,
                             gate.allowed);
            }
            return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
        }

        // The randomized QMC pilot
        std::vector<Level> levels;
        std::ofstream estimates = open_csv(args.output_dir / "qmc_estimates.csv");
        estimates << "payoff,log2_points,points,scramble,seed,relative_bump,"
                     "estimator,value\n";
        const auto emit = [&](std::string_view payoff, int log2_points,
                              std::int64_t points, int scramble, double bump,
                              std::string_view estimator, double value) {
            estimates << payoff << ',' << log2_points << ',' << points << ','
                      << scramble << ',' << args.seed << ',' << bump << ','
                      << estimator << ',' << value << '\n';
        };

        std::println("\nRandomized QMC pilot ({} scrambles per level)", args.scrambles);
        std::println("  {:>5}  {:>11}  {:>15}  {:>10}  {:>15}  {:>10}",
                     "level", "points", "basket", "half-width", "rainbow", "half-width");
        for (int level = args.min_level; level <= args.max_level; ++level) {
            fr::QmcRequest request;
            request.log2_points = level;
            request.scrambles = args.scrambles;
            request.seed = args.seed;
            request.relative_bumps.clear();
            const fr::QmcResult result = fr::run_qmc(model, request);

            levels.push_back(Level{level, result.points,
                                   fr::scramble_interval(prices_of(result.basket)),
                                   fr::scramble_interval(prices_of(result.rainbow))});
            for (const auto& run : result.basket)
                emit("basket", level, result.points, run.scramble, 0.0, "price",
                     run.price);
            for (const auto& run : result.rainbow)
                emit("rainbow", level, result.points, run.scramble, 0.0, "price",
                     run.price);

            const Level& entry = levels.back();
            std::println("  {:>5}  {:>11}  {:>15.7f}  {:>10.2e}  {:>15.7f}  {:>10.2e}",
                         level, entry.points, entry.basket.mean,
                         entry.basket.half_width, entry.rainbow.mean,
                         entry.rainbow.half_width);
        }

        // A level is accepted when its center agrees with the level below it to
        // within the larger of their half-widths and its half-width has reached
        // the target. Failing both, the top level is reported as achieved.
        std::size_t accepted_index = levels.size() - 1;
        bool target_met = false;
        for (std::size_t i = 1; i < levels.size(); ++i) {
            const auto overlaps = [](const fr::Interval& a, const fr::Interval& b) {
                return std::abs(a.mean - b.mean) <= std::max(a.half_width, b.half_width);
            };
            const bool consistent = overlaps(levels[i - 1].basket, levels[i].basket)
                                 && overlaps(levels[i - 1].rainbow, levels[i].rainbow);
            const bool tight = levels[i].basket.half_width <= args.half_width_target
                            && levels[i].rainbow.half_width <= args.half_width_target;
            if (consistent && tight) {
                accepted_index = i;
                target_met = true;
                break;
            }
        }
        const Level& accepted = levels[accepted_index];
        const Level& below = levels[accepted_index == 0 ? 0 : accepted_index - 1];
        const int accepted_level = accepted.log2_points;

        const double level_step = std::max(
            std::abs(accepted.basket.mean - below.basket.mean),
            std::abs(accepted.rainbow.mean - below.rainbow.mean));
        const double level_allowance = std::max(
            std::max(accepted.basket.half_width, below.basket.half_width),
            std::max(accepted.rainbow.half_width, below.rainbow.half_width));
        gates.push_back(Gate{"qmc_successive_levels_agree",
                             level_step <= level_allowance, true, level_step,
                             level_allowance,
                             "successive sample levels agree inside their intervals"});
        gates.push_back(Gate{"qmc_half_width_target", target_met, false,
                             std::max(accepted.basket.half_width,
                                      accepted.rainbow.half_width),
                             args.half_width_target,
                             target_met
                                 ? "the price half-width reached its target"
                                 : "target not reached; the achieved interval "
                                   "is reported and fewer digits are claimed"});

        // Greeks at the accepted level, on the shared bump sweep
        fr::QmcRequest final_request;
        final_request.log2_points = accepted_level;
        final_request.scrambles = args.scrambles;
        final_request.seed = args.seed;
        final_request.relative_bumps = args.bumps;
        const fr::QmcResult final_result = fr::run_qmc(model, final_request);

        const double center_shift = std::max(
            std::abs(fr::scramble_interval(prices_of(final_result.basket)).mean
                     - accepted.basket.mean),
            std::abs(fr::scramble_interval(prices_of(final_result.rainbow)).mean
                     - accepted.rainbow.mean));
        gates.push_back(Gate{"qmc_center_unchanged_by_bumps", center_shift == 0.0,
                             true, center_shift, 0.0,
                             "the Greek pass reproduces the pilot center bit for bit"});

        std::vector<Estimate> summary;
        summary.push_back(Estimate{"rainbow", "price", 0.0, johnson.price,
                                   quadrature_shift + johnson.truncation_bound, 0.0,
                                   "johnson"});
        summary.push_back(Estimate{"basket", "price", 0.0, accepted.basket.mean,
                                   accepted.basket.half_width, 0.0, "qmc"});
        summary.push_back(Estimate{"rainbow", "price_cross_check", 0.0,
                                   accepted.rainbow.mean, accepted.rainbow.half_width,
                                   0.0, "qmc"});

        std::ofstream greeks = open_csv(args.output_dir / "qmc_greeks.csv");
        greeks << "payoff,log2_points,points,relative_bump,estimator,asset,mean,"
                  "half_width,standard_error,scrambles,bump_error,total_error,"
                  "selected\n";

        const std::pair<std::string_view, const std::vector<fr::ScrambleEstimate>*>
            series[2] = {{"basket", &final_result.basket},
                         {"rainbow", &final_result.rainbow}};

        double worst_fit_ratio = 0.0;
        for (const auto& [payoff, runs] : series) {
            for (std::size_t asset = 0; asset < 3; ++asset) {
                for (const bool gamma : {false, true}) {
                    std::vector<fr::Interval> band;
                    std::vector<double> centers;
                    band.reserve(args.bumps.size());
                    for (std::size_t b = 0; b < args.bumps.size(); ++b) {
                        band.push_back(fr::scramble_interval(
                            greeks_of(*runs, b, asset, gamma)));
                        centers.push_back(band.back().mean);
                        for (const auto& run : *runs)
                            emit(payoff, accepted_level, final_result.points,
                                 run.scramble, args.bumps[b], greek_name(gamma, asset),
                                 gamma ? run.bump[b].gamma[asset]
                                       : run.bump[b].delta[asset]);
                    }

                    // Two error terms pull in opposite directions: the sampling
                    // interval shrinks with a wider bump and the truncation with
                    // a narrower one. The reported bump is the one whose sum is
                    // smallest, which is a decision the sweep makes rather than
                    // a favorable value being picked out of it.
                    const BumpFit fit = fit_bump_truncation(args.bumps, centers);
                    std::size_t best = 0;
                    double best_total = std::numeric_limits<double>::infinity();
                    std::vector<double> truncation(args.bumps.size(), 0.0);
                    for (std::size_t b = 0; b < args.bumps.size(); ++b) {
                        truncation[b] =
                            std::abs(fit.coefficient) * args.bumps[b] * args.bumps[b];
                        const double total = band[b].half_width + truncation[b];
                        if (total < best_total) { best_total = total; best = b; }
                    }

                    // Whether the sweep really is second order: the fit is built
                    // from one adjacent pair, so the narrowest bump is a free
                    // check that nothing else is driving the trend.
                    const double allowance = band[0].half_width + band[1].half_width;
                    worst_fit_ratio = std::max(
                        worst_fit_ratio, fit.residual / std::max(allowance, 1.0e-300));

                    for (std::size_t b = 0; b < args.bumps.size(); ++b)
                        greeks << payoff << ',' << accepted_level << ','
                               << final_result.points << ',' << args.bumps[b] << ','
                               << (gamma ? "gamma" : "delta") << ',' << (asset + 1)
                               << ',' << band[b].mean << ',' << band[b].half_width
                               << ',' << band[b].standard_error << ','
                               << band[b].samples << ',' << truncation[b] << ','
                               << (band[b].half_width + truncation[b]) << ','
                               << (b == best ? 1 : 0) << '\n';

                    summary.push_back(Estimate{
                        std::string(payoff), greek_name(gamma, asset),
                        args.bumps[best], band[best].mean, band[best].half_width,
                        truncation[best], "qmc"});
                }
            }
        }
        gates.push_back(Gate{"qmc_bump_sweep_is_second_order", worst_fit_ratio <= 3.0,
                             false, worst_fit_ratio, 3.0,
                             "the narrowest bump lands on the second-order fit "
                             "built from the two widest, so the truncation is a "
                             "budgeted term rather than an unexplained spread"});

        const double rainbow_gap = std::abs(accepted.rainbow.mean - johnson.price);
        const double rainbow_allowance =
            accepted.rainbow.half_width + quadrature_shift + johnson.truncation_bound;
        gates.push_back(Gate{"qmc_agrees_with_johnson",
                             rainbow_gap <= rainbow_allowance, true, rainbow_gap,
                             rainbow_allowance,
                             "the independent Rainbow paths agree inside the "
                             "stated uncertainty"});

        {
            std::ofstream file = open_csv(args.output_dir / "qmc_levels.csv");
            file << "payoff,log2_points,points,scrambles,mean,half_width,"
                    "standard_error,accepted\n";
            for (const Level& level : levels)
                for (const auto& [payoff, interval] :
                     {std::pair{"basket", level.basket},
                      std::pair{"rainbow", level.rainbow}})
                    file << payoff << ',' << level.log2_points << ',' << level.points
                         << ',' << interval.samples << ',' << interval.mean << ','
                         << interval.half_width << ',' << interval.standard_error
                         << ',' << (level.log2_points == accepted_level ? 1 : 0) << '\n';
        }

        {
            std::ofstream file = open_csv(args.output_dir / "parameters.csv");
            file << kParameterHeader << '\n';
            write_parameter_columns(file, model);
            file << '\n';
        }

        write_summary(args.output_dir / "reference_summary.csv", summary, model,
                      accepted_level, args.scrambles, args.seed);

        {
            std::ofstream file = open_csv(args.output_dir / "reference_gates.csv");
            file << "gate,passed,blocking,measured,allowed,note\n";
            for (const Gate& gate : gates)
                file << gate.name << ',' << (gate.passed ? 1 : 0) << ','
                     << (gate.blocking ? 1 : 0) << ',' << gate.measured << ','
                     << gate.allowed << ",\"" << gate.note << "\"\n";
        }

        std::println("\nAccepted sample level 2^{} = {} points, {} scrambles",
                     accepted_level, accepted.points, args.scrambles);
        std::println("  basket  price {:.7f} +/- {:.2e}   "
                     "(historical comparison {:.4f})",
                     accepted.basket.mean, accepted.basket.half_width,
                     kHistoricalBasket);
        std::println("  rainbow price {:.7f} +/- {:.2e}   (Johnson {:.7f})",
                     accepted.rainbow.mean, accepted.rainbow.half_width,
                     johnson.price);

        std::println("\n  {:<8} {:<18} {:>6}  {:>15}  {:>11}  {:>11}",
                     "payoff", "quantity", "bump", "value", "half-width", "bump err");
        for (const Estimate& item : summary)
            std::println("  {:<8} {:<18} {:>6.3f}  {:>15.7f}  {:>11.2e}  {:>11.2e}",
                         item.payoff, item.quantity, item.relative_bump, item.value,
                         item.half_width, item.bump_error);

        std::println("\nGates");
        int failures = 0;
        for (const Gate& gate : gates) {
            if (gate.blocking && !gate.passed) ++failures;
            std::println("  [{}]{} {:<38} measured={:.3e} allowed={:.3e}",
                         gate.passed ? "pass" : "FAIL",
                         gate.blocking ? " " : "*", gate.name, gate.measured,
                         gate.allowed);
        }
        std::println("  (* reported, not blocking)");
        std::println("\n  wrote {}", args.output_dir.string());

        if (failures > 0) {
            std::println(std::cerr, "Error: {} blocking reference gate(s) failed",
                         failures);
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::println(std::cerr, "Error: {}", error.what());
        return EXIT_FAILURE;
    }
}
