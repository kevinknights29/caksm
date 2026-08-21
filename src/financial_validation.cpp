/**
 * @file financial_validation.cpp
 * @brief PDE-side financial validation: refinement, Greeks, error budget, and
 *        the equal-accuracy CPU comparison of KSM-EI against smoothed HV-ADI.
 *
 *   ./financial-validation [--stage all|domain|space|time|greeks|equal-accuracy]
 *                          [--reference-dir DIR] [--output-dir DIR]
 *                          [--base-n N] [--base-alpha A] [--base-steps M]
 *                          [--bumps a,b,c] [--repeats R] [--error-target X]
 *
 * Reads the accepted reference row written by ./financial-reference and
 * refuses to run if its contract does not match the one compiled in, so a
 * price can never be judged against a reference for a different option.
 *
 * @author Kevin Knights
 */

#include <algorithm>
#include <array>
#include <chrono>
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
#include <vector>

#include <Eigen/Core>

#include "config.hpp"
#include "pde_operators.hpp"
#include "solvers.hpp"

namespace {

struct Args {
    std::string stage = "all";
    std::filesystem::path reference_dir = "data/financial-validation";
    std::filesystem::path output_dir = "data/financial-validation";
    // The accepted working point. alpha = 4.275 is the domain the sweep
    // accepts, not the 2.85 the historical benchmark uses: at 2.85 the Basket
    // price still carries about 1.6e-2 of domain truncation, which no amount of
    // grid refinement removes.
    int base_n = 61;
    double base_alpha = 4.275;
    int base_steps = 200;
    int repeats = 7;
    double error_target = 0.0;  ///< zero selects the target from the spatial ladder
    /// The same sweep the reference program uses, so the two are comparable
    /// bump by bump. It reaches to 10 percent because a trilinear interpolant
    /// only carries curvature once the bump spans several grid cells.
    std::vector<double> bumps = {0.005, 0.01, 0.02, 0.05, 0.10};
    std::vector<int> spatial_ladder = {31, 61, 121};
    // Each domain is the base pair rescaled to an exactly odd grid size, so the
    // central spacing is matched to roundoff and the comparison is not reading
    // a spacing drift as a domain effect.
    std::vector<double> domain_ladder = {2.1375, 2.85, 3.5625, 4.275, 4.9875};
    std::vector<int> temporal_ladder = {25, 50, 100, 200, 400};
    /// Reaches further down than the refinement ladder because the
    /// equal-accuracy question is how little temporal work each method needs,
    /// and both are cheap enough at coarse step counts to be worth asking. It
    /// starts at one so that "cheapest" is not an artifact of where the ladder
    /// happens to begin: an exponential integrator that is time-exact to its
    /// tolerance would otherwise be charged for steps it does not need.
    std::vector<int> equal_accuracy_ladder = {1, 2, 5, 10, 25, 50, 100, 200, 400};
};

/// One reference quantity as the reference program accepted it.
struct Reference {
    double value = 0.0;
    double half_width = 0.0;
    double bump_error = 0.0;
    double relative_bump = 0.0;
    std::string source;

    /// Total stated uncertainty: sampling plus bump truncation.
    [[nodiscard]] double uncertainty() const { return half_width + bump_error; }
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

/// One solved field, with everything the tables read off it.
struct Solution {
    double price = 0.0;
    std::array<double, 3> delta{};
    std::array<double, 3> gamma{};
    double seconds = 0.0;
    bool finite = true;
};

using SolverFunction = VecXd (*)(const PDESystem&, const Config&);

// Arguments

[[nodiscard]] std::vector<double> parse_doubles(std::string_view text)
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

[[nodiscard]] std::vector<int> parse_ints(std::string_view text)
{
    std::vector<int> values;
    for (const double value : parse_doubles(text))
        values.push_back(static_cast<int>(value));
    return values;
}

[[nodiscard]] Args parse_validation_args(std::span<const char* const> argv)
{
    Args args;
    for (std::size_t i = 1; i < argv.size(); ++i) {
        const std::string_view arg = argv[i];
        const auto next = [&]() -> std::string_view {
            if (++i >= argv.size())
                throw std::invalid_argument("missing value for " + std::string(arg));
            return argv[i];
        };
        if (arg == "--stage") args.stage = std::string(next());
        else if (arg == "--reference-dir")
            args.reference_dir = std::filesystem::path(std::string(next()));
        else if (arg == "--output-dir")
            args.output_dir = std::filesystem::path(std::string(next()));
        else if (arg == "--base-n") args.base_n = std::stoi(std::string(next()));
        else if (arg == "--base-alpha") args.base_alpha = std::stod(std::string(next()));
        else if (arg == "--base-steps") args.base_steps = std::stoi(std::string(next()));
        else if (arg == "--repeats") args.repeats = std::stoi(std::string(next()));
        else if (arg == "--error-target")
            args.error_target = std::stod(std::string(next()));
        else if (arg == "--bumps") args.bumps = parse_doubles(next());
        else if (arg == "--grids") args.spatial_ladder = parse_ints(next());
        else if (arg == "--alphas") args.domain_ladder = parse_doubles(next());
        else if (arg == "--step-ladder") args.temporal_ladder = parse_ints(next());
        else if (arg == "--help") {
            std::println(
                "Usage: ./financial-validation [--stage STAGE] [--reference-dir DIR]\n"
                "         [--output-dir DIR] [--base-n N] [--base-alpha A]\n"
                "         [--base-steps M] [--bumps a,b,c] [--grids n1,n2,n3]\n"
                "         [--alphas a1,a2,a3] [--step-ladder m1,m2] [--repeats R]\n"
                "         [--error-target X]\n"
                "  --stage   all | domain | space | time | greeks | equal-accuracy\n"
                "  --grids   odd grid sizes for the spatial ladder\n"
                "  --alphas  domain half-widths, rescaled to matched spacing\n"
                "  --repeats measured repetitions in the equal-accuracy table");
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    for (const int n : args.spatial_ladder)
        if (n < 5 || n % 2 == 0)
            throw std::invalid_argument("every grid size must be odd and at least 5");
    if (args.repeats < 7)
        throw std::invalid_argument("at least seven measured repetitions are required");
    std::ranges::sort(args.bumps);
    std::ranges::sort(args.spatial_ladder);
    std::ranges::sort(args.domain_ladder);
    std::ranges::sort(args.temporal_ladder);
    return args;
}

// The accepted reference row

[[nodiscard]] std::vector<std::map<std::string, std::string>> read_csv(
    const std::filesystem::path& path)
{
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot read " + path.string()
                                        + "; run ./financial-reference first");
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
 * @brief Load the accepted reference values and check they price this contract.
 *
 * The reference row carries the complete parameter set, so a mismatch is a hard
 * error rather than a silent comparison against a different option.
 */
[[nodiscard]] std::map<std::string, Reference> load_reference(
    const std::filesystem::path& path, const Config& model)
{
    const auto rows = read_csv(path);
    if (rows.empty()) throw std::runtime_error("no reference rows in " + path.string());

    const auto agrees = [&](const std::map<std::string, std::string>& row,
                            std::string_view column, double expected) {
        const auto found = row.find(std::string(column));
        if (found == row.end())
            throw std::runtime_error("reference row is missing column "
                                     + std::string(column));
        return std::abs(std::stod(found->second) - expected)
            <= 1.0e-12 * std::max(1.0, std::abs(expected));
    };

    std::map<std::string, Reference> reference;
    for (const auto& row : rows) {
        const bool same =
            agrees(row, "strike", model.strike_price)
            && agrees(row, "rate", model.risk_free_rate)
            && agrees(row, "expiry", model.t_final)
            && agrees(row, "spot_1", model.initial_prices[0])
            && agrees(row, "spot_2", model.initial_prices[1])
            && agrees(row, "spot_3", model.initial_prices[2])
            && agrees(row, "sigma_1", model.sigma[0])
            && agrees(row, "sigma_2", model.sigma[1])
            && agrees(row, "sigma_3", model.sigma[2])
            && agrees(row, "rho_12", model.rho_off[0])
            && agrees(row, "rho_13", model.rho_off[1])
            && agrees(row, "rho_23", model.rho_off[2])
            && agrees(row, "weight_1", model.weight[0])
            && agrees(row, "weight_2", model.weight[1])
            && agrees(row, "weight_3", model.weight[2]);
        if (!same)
            throw std::runtime_error(
                "the stored reference prices a different contract than this build");
        Reference entry;
        entry.value = std::stod(row.at("value"));
        entry.half_width = std::stod(row.at("half_width"));
        entry.bump_error = std::stod(row.at("bump_error"));
        entry.relative_bump = std::stod(row.at("relative_bump"));
        entry.source = row.at("source");
        reference[row.at("payoff") + "." + row.at("quantity")] = entry;
    }
    return reference;
}

// Solving

/// Assemble the system for one payoff at one grid and domain half-width.
[[nodiscard]] PDESystem assemble(const Config& model, bool rainbow)
{
    return build_pde_system(model.n, model.strike_price, model.risk_free_rate,
                            model.t_final, model.sigma, model.rho_off, model.weight,
                            model.initial_prices, model.alpha, rainbow);
}

/**
 * @brief Read the price and Greeks off one already-solved field.
 *
 * Every bumped spot is interpolated from the same field. Rebuilding a grid
 * centered on each bump would change the discretization between the terms of a
 * difference, which would leave the Greek measuring the regridding rather than
 * the sensitivity. It also means the whole bump sweep costs one solve.
 */
[[nodiscard]] Solution extract(const VecXd& field, const Grid& grid,
                                const std::array<double, 3>& spots,
                                double relative_bump)
{
    Solution result;
    result.price = interpolate_price_trilinear(field, grid, spots);
    for (std::size_t d = 0; d < 3; ++d) {
        const double step = relative_bump * spots[d];
        std::array<double, 3> up = spots;
        std::array<double, 3> down = spots;
        up[d] += step;
        down[d] -= step;
        const double high = interpolate_price_trilinear(field, grid, up);
        const double low = interpolate_price_trilinear(field, grid, down);
        result.delta[d] = (high - low) / (2.0 * step);
        result.gamma[d] = (high - 2.0 * result.price + low) / (step * step);
    }

    result.finite = std::isfinite(result.price);
    for (std::size_t d = 0; d < 3; ++d)
        result.finite = result.finite && std::isfinite(result.delta[d])
                     && std::isfinite(result.gamma[d]);
    return result;
}

/**
 * @brief Delta and diagonal Gamma from the grid stencil at the base spot.
 *
 * Only valid where the spot is a node, which odd centered grids arrange. This
 * is not the reported estimator; it is the control that separates two effects
 * the bump sweep otherwise confounds.
 *
 * A trilinear interpolant is linear inside a cell, so its second difference
 * collapses when the bump is smaller than the spacing: the estimate degenerates
 * to the grid's own curvature scaled by spacing over bump, which inflates Gamma
 * without saying anything about the discretization. Reading the same curvature
 * straight off the stencil says which of the two a Gamma discrepancy is.
 *
 * In log price x = log(S), the chain rule gives
 * S dV/dS = V_x and S^2 d2V/dS^2 = V_xx - V_x.
 */
[[nodiscard]] Solution node_greeks(const VecXd& field, const Grid& grid,
                                    const std::array<double, 3>& spots)
{
    const int n = grid.n;
    int node[3];
    for (int d = 0; d < 3; ++d) {
        Eigen::Index index = 0;
        (grid.x[d].array() - std::log(spots[d])).cwiseAbs().minCoeff(&index);
        node[d] = static_cast<int>(index);
        if (node[d] < 1 || node[d] > n - 2)
            throw std::runtime_error("node_greeks: the spot sits on the boundary");
    }
    const int stride[3] = {1, n, n * n};
    const int center = node[2] * n * n + node[1] * n + node[0];

    Solution result;
    result.price = field[center];
    for (std::size_t d = 0; d < 3; ++d) {
        const double dx = grid.dx[d];
        const double high = field[center + stride[d]];
        const double low = field[center - stride[d]];
        const double first = (high - low) / (2.0 * dx);
        const double second = (high - 2.0 * result.price + low) / (dx * dx);
        result.delta[d] = first / spots[d];
        result.gamma[d] = (second - first) / (spots[d] * spots[d]);
    }
    result.finite = std::isfinite(result.price);
    for (std::size_t d = 0; d < 3; ++d)
        result.finite = result.finite && std::isfinite(result.delta[d])
                     && std::isfinite(result.gamma[d]);
    return result;
}

/// Solve, then extract. The timing covers method-specific setup and the solve.
[[nodiscard]] Solution solve_and_extract(const PDESystem& system, const Config& model,
                                          SolverFunction solver, double relative_bump)
{
    const auto start = std::chrono::steady_clock::now();
    const VecXd field = solver(system, model);
    const auto stop = std::chrono::steady_clock::now();
    Solution result = extract(field, system.grid, model.initial_prices, relative_bump);
    result.seconds = std::chrono::duration<double>(stop - start).count();
    return result;
}

/// Odd grid size that keeps the central spacing fixed as the domain widens.
[[nodiscard]] int matched_grid_size(int base_n, double base_alpha, double alpha)
{
    const double scaled =
        1.0 + static_cast<double>(base_n - 1) * alpha / base_alpha;
    int rounded = static_cast<int>(std::lround(scaled));
    if (rounded % 2 == 0) ++rounded;
    return rounded;
}

/// Observed order from three successively refined values.
[[nodiscard]] double observed_order(double coarse, double medium, double fine)
{
    const double first = coarse - medium;
    const double second = medium - fine;
    if (!(std::abs(second) > 0.0) || first / second <= 0.0)
        return std::numeric_limits<double>::quiet_NaN();
    return std::log2(first / second);
}

// Output

[[nodiscard]] std::ofstream open_csv(const std::filesystem::path& path)
{
    std::ofstream file(path);
    if (!file) throw std::runtime_error("cannot create " + path.string());
    file << std::setprecision(17);
    return file;
}

/// Column block shared by every refinement table.
void write_solution_columns(std::ofstream& file, const Solution& solution)
{
    file << solution.price;
    for (const double value : solution.delta) file << ',' << value;
    for (const double value : solution.gamma) file << ',' << value;
    file << ',' << solution.seconds;
}

constexpr std::string_view kSolutionHeader =
    "price,delta_1,delta_2,delta_3,gamma_11,gamma_22,gamma_33,seconds";

const std::pair<std::string_view, bool> kPayoffs[2] = {{"basket", false},
                                                       {"rainbow", true}};

/// The reference price of one payoff, whichever calculation is primary for it.
[[nodiscard]] const Reference& primary_price(
    const std::map<std::string, Reference>& reference, std::string_view payoff)
{
    return reference.at(std::string(payoff) + ".price");
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const Args args = parse_validation_args(
            std::span<const char* const>(argv, static_cast<std::size_t>(argc)));
        std::filesystem::create_directories(args.output_dir);

        // One thread throughout, so the equal-accuracy table is a method
        // comparison and not an accident of how each solver happens to thread.
        Eigen::setNbThreads(1);

        const Config base;  // the shared parameter object
        const auto reference =
            load_reference(args.reference_dir / "reference_summary.csv", base);
        std::vector<Gate> gates;

        const bool run_domain = args.stage == "all" || args.stage == "domain";
        const bool run_space = args.stage == "all" || args.stage == "space";
        const bool run_time = args.stage == "all" || args.stage == "time";
        const bool run_greeks = args.stage == "all" || args.stage == "greeks";
        const bool run_equal = args.stage == "all" || args.stage == "equal-accuracy";
        if (!run_domain && !run_space && !run_time && !run_greeks && !run_equal)
            throw std::invalid_argument("unknown stage: " + args.stage);

        std::println("Financial validation, smoothed HV-ADI and KSM-EI on the CPU");
        std::println("  reference {}",
                     (args.reference_dir / "reference_summary.csv").string());
        for (const auto& [payoff, rainbow] : kPayoffs) {
            const Reference& price = primary_price(reference, payoff);
            std::println("  {:<8} price {:.7f} +/- {:.2e}  [{}]", payoff,
                         price.value, price.uncertainty(), price.source);
        }
        std::println("  start-up smoothing steps {}, theta = 1 then 0.5\n",
                     base.hv_smoothing_steps);

        // A settled temporal configuration for the spatial and domain stages,
        // so what those tables move is the grid and nothing else.
        const auto settled = [&](int n, double alpha, int steps) {
            Config model = base;
            model.n = n;
            model.alpha = alpha;
            model.temporal_steps = steps;
            model.ei_steps = steps;
            model.tol_ei = 1.0e-12;
            return model;
        };
        const double reported_bump = args.bumps[args.bumps.size() / 2];

        int accepted_n = args.base_n;
        std::map<std::string, double> domain_effect;

        // Domain truncation
        if (run_domain) {
            std::ofstream file = open_csv(args.output_dir / "pde_domain.csv");
            file << "payoff,method,alpha,n,dx_1,dx_2,dx_3,spacing_mismatch,"
                 << kSolutionHeader << ",reference,error\n";
            std::println("Domain truncation at matched central spacing");
            std::println("  {:<8} {:>6} {:>5} {:>12} {:>13} {:>12}",
                         "payoff", "alpha", "n", "dx mismatch", "price", "error");

            std::map<std::string, std::vector<std::pair<double, double>>> by_payoff;
            for (const auto& [payoff, rainbow] : kPayoffs) {
                const Reference& price = primary_price(reference, payoff);
                double base_spacing = 0.0;
                for (const double alpha : args.domain_ladder) {
                    const int n = matched_grid_size(args.base_n, args.base_alpha, alpha);
                    const Config model = settled(n, alpha, args.base_steps);
                    const PDESystem system = assemble(model, rainbow);
                    if (!spots_are_grid_nodes(system.grid, model.initial_prices))
                        throw std::runtime_error(
                            "the base spots do not lie on grid nodes at n="
                            + std::to_string(n));
                    const Solution solution = solve_and_extract(
                        system, model, solve_adi_hv_s, reported_bump);
                    if (!solution.finite)
                        throw std::runtime_error("domain stage produced a non-finite value");

                    if (base_spacing == 0.0) base_spacing = system.grid.dx[0];
                    const double mismatch =
                        std::abs(system.grid.dx[0] - base_spacing) / base_spacing;
                    const double error = solution.price - price.value;
                    by_payoff[std::string(payoff)].emplace_back(alpha, solution.price);

                    file << payoff << ",ADI-HV-S," << alpha << ',' << n << ','
                         << system.grid.dx[0] << ',' << system.grid.dx[1] << ','
                         << system.grid.dx[2] << ',' << mismatch << ',';
                    write_solution_columns(file, solution);
                    file << ',' << price.value << ',' << error << '\n';
                    std::println("  {:<8} {:>6.2f} {:>5} {:>12.2e} {:>13.6f} {:>12.2e}",
                                 payoff, alpha, n, mismatch, solution.price, error);

                    // A comparison whose spacings drifted cannot separate the
                    // domain effect from a spatial one.
                    gates.push_back(Gate{
                        "domain_spacing_matched_" + std::string(payoff) + "_alpha"
                            + std::to_string(alpha).substr(0, 4),
                        mismatch < 0.02, true, mismatch, 0.02,
                        "the rescaled grid keeps the central spacing"});
                }
            }

            // The remaining domain effect is what the two widest domains disagree by.
            std::ofstream residual = open_csv(args.output_dir / "pde_domain_residual.csv");
            residual << "payoff,alpha_low,alpha_high,price_low,price_high,"
                        "domain_effect\n";
            for (const auto& [payoff, points] : by_payoff) {
                const auto& low = points[points.size() - 2];
                const auto& high = points.back();
                const double effect = std::abs(high.second - low.second);
                domain_effect[payoff] = effect;
                residual << payoff << ',' << low.first << ',' << high.first << ','
                         << low.second << ',' << high.second << ',' << effect << '\n';
                std::println("  {:<8} remaining domain effect between alpha {:.2f} "
                             "and {:.2f}: {:.2e}",
                             payoff, low.first, high.first, effect);
            }
            std::println("");
        }

        // Spatial refinement
        std::map<std::string, std::vector<std::pair<int, double>>> spatial;
        if (run_space) {
            std::ofstream file = open_csv(args.output_dir / "pde_space.csv");
            file << "payoff,method,alpha,n,dx_1," << kSolutionHeader
                 << ",reference,error,observed_order\n";
            std::println("Spatial refinement at alpha = {:.2f}", args.base_alpha);
            std::println("  {:<8} {:<9} {:>5} {:>13} {:>12} {:>8}",
                         "payoff", "method", "n", "price", "error", "order");

            const std::pair<std::string_view, SolverFunction> methods[2] = {
                {"ADI-HV-S", solve_adi_hv_s}, {"KSM-EI", solve_ksm_ei}};

            for (const auto& [payoff, rainbow] : kPayoffs) {
                const Reference& price = primary_price(reference, payoff);
                for (const auto& [label, solver] : methods) {
                    std::vector<double> prices;
                    for (const int n : args.spatial_ladder) {
                        const Config model =
                            settled(n, args.base_alpha, args.base_steps);
                        const PDESystem system = assemble(model, rainbow);
                        if (!spots_are_grid_nodes(system.grid, model.initial_prices))
                            throw std::runtime_error(
                                "the base spots do not lie on grid nodes at n="
                                + std::to_string(n));
                        const Solution solution =
                            solve_and_extract(system, model, solver, reported_bump);
                        if (!solution.finite)
                            throw std::runtime_error(
                                "spatial stage produced a non-finite value");
                        prices.push_back(solution.price);

                        const double order = prices.size() >= 3
                            ? observed_order(prices[prices.size() - 3],
                                             prices[prices.size() - 2], prices.back())
                            : std::numeric_limits<double>::quiet_NaN();
                        const double error = solution.price - price.value;

                        file << payoff << ',' << label << ',' << args.base_alpha << ','
                             << n << ',' << system.grid.dx[0] << ',';
                        write_solution_columns(file, solution);
                        file << ',' << price.value << ',' << error << ',' << order
                             << '\n';
                        std::println("  {:<8} {:<9} {:>5} {:>13.6f} {:>12.2e} {:>8.2f}",
                                     payoff, label, n, solution.price, error, order);

                        if (label == "ADI-HV-S")
                            spatial[std::string(payoff)].emplace_back(n, solution.price);
                    }

                    // Dang reports the spot price rising toward the reference as
                    // the grid is refined. The log grid and the boundary closure
                    // differ, so the magnitude need not match, but a refinement
                    // that walked away from the reference would mean something
                    // other than discretization is moving.
                    const double coarse = std::abs(prices.front() - price.value);
                    const double fine = std::abs(prices.back() - price.value);
                    gates.push_back(Gate{
                        "spatial_error_decreases_" + std::string(payoff) + "_"
                            + std::string(label),
                        fine < coarse, true, fine, coarse,
                        "refinement moves the price toward the reference"});
                }
            }
            accepted_n = args.spatial_ladder.back();

            // The working domain is only accepted if what it still truncates is
            // smaller than what the grid still discretizes. Otherwise the
            // spatial table would be reporting a domain effect.
            for (const auto& [payoff, effect] : domain_effect) {
                const auto found = spatial.find(payoff);
                if (found == spatial.end() || found->second.size() < 2) continue;
                const double spatial_error =
                    std::abs(found->second.back().second
                             - found->second[found->second.size() - 2].second);
                gates.push_back(Gate{
                    "domain_effect_below_spatial_error_" + payoff,
                    effect < spatial_error, true, effect, spatial_error,
                    "the remaining domain truncation is smaller than the "
                    "spatial discretization it is measured beside"});
            }
            std::println("");
        }

        // Temporal refinement
        if (run_time) {
            std::ofstream file = open_csv(args.output_dir / "pde_time.csv");
            file << "payoff,method,n,steps,tolerance,smoothing_steps," << kSolutionHeader
                 << ",reference,error,change_from_previous\n";
            std::println("Temporal refinement at n = {}, alpha = {:.2f}",
                         args.base_n, args.base_alpha);
            std::println("  {:<8} {:<9} {:>6} {:>13} {:>12} {:>12}",
                         "payoff", "method", "steps", "price", "error", "change");

            for (const auto& [payoff, rainbow] : kPayoffs) {
                const Reference& price = primary_price(reference, payoff);
                const Config assembly = settled(args.base_n, args.base_alpha,
                                                args.temporal_ladder.front());
                const PDESystem system = assemble(assembly, rainbow);

                double previous = std::numeric_limits<double>::quiet_NaN();
                for (const int steps : args.temporal_ladder) {
                    Config model = settled(args.base_n, args.base_alpha, steps);
                    const Solution solution = solve_and_extract(
                        system, model, solve_adi_hv_s, reported_bump);
                    if (!solution.finite)
                        throw std::runtime_error("time stage produced a non-finite value");
                    const double change = std::abs(solution.price - previous);
                    file << payoff << ",ADI-HV-S," << args.base_n << ',' << steps
                         << ",0," << model.hv_smoothing_steps << ',';
                    write_solution_columns(file, solution);
                    file << ',' << price.value << ',' << (solution.price - price.value)
                         << ',' << change << '\n';
                    std::println("  {:<8} {:<9} {:>6} {:>13.6f} {:>12.2e} {:>12.2e}",
                                 payoff, "ADI-HV-S", steps, solution.price,
                                 solution.price - price.value, change);
                    previous = solution.price;

                    // The smoothing must still be active at every level, which
                    // it can only be if the step count exceeds the start-up run.
                    gates.push_back(Gate{
                        "hv_smoothing_active_" + std::string(payoff) + "_steps"
                            + std::to_string(steps),
                        steps > model.hv_smoothing_steps, true,
                        static_cast<double>(steps),
                        static_cast<double>(model.hv_smoothing_steps),
                        "the start-up smoothing region fits inside the step count"});
                }

                previous = std::numeric_limits<double>::quiet_NaN();
                for (const double tolerance :
                     {1.0e-4, 1.0e-6, 1.0e-8, 1.0e-10, 1.0e-12}) {
                    Config model = settled(args.base_n, args.base_alpha,
                                           args.temporal_ladder.back());
                    model.tol_ei = tolerance;
                    model.ei_steps = args.base_steps;
                    const Solution solution =
                        solve_and_extract(system, model, solve_ksm_ei, reported_bump);
                    if (!solution.finite)
                        throw std::runtime_error("time stage produced a non-finite value");
                    const double change = std::abs(solution.price - previous);
                    file << payoff << ",KSM-EI," << args.base_n << ','
                         << model.ei_steps << ',' << tolerance << ",0,";
                    write_solution_columns(file, solution);
                    file << ',' << price.value << ',' << (solution.price - price.value)
                         << ',' << change << '\n';
                    std::println("  {:<8} {:<9} {:>6.0e} {:>13.6f} {:>12.2e} {:>12.2e}",
                                 payoff, "KSM-EI", tolerance, solution.price,
                                 solution.price - price.value, change);
                    previous = solution.price;
                }
            }
            std::println("");
        }

        // Greeks on the shared bump sweep
        if (run_greeks) {
            std::ofstream file = open_csv(args.output_dir / "pde_greeks.csv");
            file << "payoff,method,n,alpha,steps,relative_bump,estimator,asset,"
                    "pde_value,node_value,reference_value,reference_half_width,"
                    "reference_bump_error,difference,relative_difference,"
                    "node_difference,node_relative_difference,bump_over_spacing,"
                    "within_reference_band\n";
            std::println("Greeks at n = {}, alpha = {:.2f}, on the shared bump sweep",
                         accepted_n, args.base_alpha);
            std::println("  {:<8} {:<9} {:>6} {:>7} {:>12} {:>12} {:>12} {:>10} {:>10}",
                         "payoff", "estimator", "bump", "h/dx", "PDE", "reference",
                         "rel diff", "node value", "node rel");

            const std::pair<std::string_view, SolverFunction> methods[2] = {
                {"ADI-HV-S", solve_adi_hv_s}, {"KSM-EI", solve_ksm_ei}};

            for (const auto& [payoff, rainbow] : kPayoffs) {
                const Config model = settled(accepted_n, args.base_alpha, args.base_steps);
                const PDESystem system = assemble(model, rainbow);
                for (const auto& [label, solver] : methods) {
                    // A node lookup and a trilinear interpolation must agree at
                    // the base spot, which is exactly on a node. Any gap there
                    // is interpolation error, not discretization error.
                    const VecXd field = solver(system, model);
                    const double node_price =
                        extract_price(field, system.grid, model.initial_prices);
                    const double interpolated = interpolate_price_trilinear(
                        field, system.grid, model.initial_prices);
                    gates.push_back(Gate{
                        "interpolation_exact_at_node_" + std::string(payoff) + "_"
                            + std::string(label),
                        std::abs(node_price - interpolated)
                            <= 1.0e-12 * std::max(1.0, std::abs(node_price)),
                        true, std::abs(node_price - interpolated),
                        1.0e-12 * std::max(1.0, std::abs(node_price)),
                        "trilinear interpolation returns the stored value at a node"});

                    const Solution node = node_greeks(field, system.grid,
                                                      model.initial_prices);
                    for (const double bump : args.bumps) {
                        const Solution solution =
                            extract(field, system.grid, model.initial_prices, bump);
                        if (!solution.finite || !node.finite)
                            throw std::runtime_error(
                                "greek stage produced a non-finite value");
                        for (std::size_t asset = 0; asset < 3; ++asset) {
                            // How many cells the bump spans in the coordinate the
                            // interpolant is linear in. Below one, a trilinear
                            // second difference is reading the grid curvature
                            // divided by the bump instead of the true one.
                            const double offset = std::log1p(bump);
                            const double spacing = offset / system.grid.dx[asset];
                            for (const bool gamma : {false, true}) {
                                const std::string index = std::to_string(asset + 1);
                                const std::string quantity = gamma
                                    ? "gamma_" + index + index : "delta_" + index;
                                const auto found = reference.find(
                                    std::string(payoff) + "." + quantity);
                                if (found == reference.end()) continue;
                                const double expected = found->second.value;
                                const double scale = std::max(std::abs(expected),
                                                              1.0e-300);
                                const double value = gamma ? solution.gamma[asset]
                                                           : solution.delta[asset];
                                const double bare = gamma ? node.gamma[asset]
                                                          : node.delta[asset];
                                const double difference = value - expected;
                                const bool within =
                                    std::abs(difference) <= found->second.uncertainty();
                                file << payoff << ',' << label << ',' << accepted_n
                                     << ',' << args.base_alpha << ','
                                     << args.base_steps << ',' << bump << ','
                                     << (gamma ? "gamma" : "delta") << ','
                                     << (asset + 1) << ',' << value << ',' << bare
                                     << ',' << expected << ','
                                     << found->second.half_width << ','
                                     << found->second.bump_error << ',' << difference
                                     << ',' << (difference / scale) << ','
                                     << (bare - expected) << ','
                                     << ((bare - expected) / scale) << ','
                                     << spacing << ',' << (within ? 1 : 0) << '\n';
                                if (label == "ADI-HV-S"
                                    && std::abs(bump - found->second.relative_bump)
                                        < 1.0e-15)
                                    std::println(
                                        "  {:<8} {:<9} {:>6.3f} {:>7.2f} {:>12.6f} "
                                        "{:>12.6f} {:>12.2e} {:>10.6f} {:>10.2e}",
                                        payoff, quantity, bump, spacing, value,
                                        expected, difference / scale, bare,
                                        (bare - expected) / scale);
                            }

                            // Delta inherits the price's relative discretization
                            // error and must land inside it. Gamma is reported
                            // rather than gated: at the reference's own bump the
                            // trilinear estimator is resolution-limited, and the
                            // node control beside it is what makes the size of
                            // that limitation legible.
                            const auto delta_reference = reference.find(
                                std::string(payoff) + ".delta_"
                                + std::to_string(asset + 1));
                            if (delta_reference != reference.end()
                                && std::abs(bump - delta_reference->second.relative_bump)
                                    < 1.0e-15)
                                gates.push_back(Gate{
                                    "delta_matches_reference_" + std::string(payoff)
                                        + "_" + std::string(label) + "_asset"
                                        + std::to_string(asset + 1),
                                    std::abs(solution.delta[asset]
                                             - delta_reference->second.value)
                                        <= 5.0e-3 * std::abs(delta_reference->second.value),
                                    true,
                                    std::abs(solution.delta[asset]
                                             - delta_reference->second.value),
                                    5.0e-3 * std::abs(delta_reference->second.value),
                                    "the PDE Delta lands inside the price's own "
                                    "relative discretization error"});
                        }
                    }
                }
            }
            std::println("");
        }

        // Equal-accuracy CPU comparison
        if (run_equal) {
            std::println("Equal-accuracy comparison, one thread, "
                         "{} measured repetitions", args.repeats);
            std::ofstream sweep = open_csv(args.output_dir / "equal_accuracy_sweep.csv");
            sweep << "payoff,method,n,steps,tolerance,price,error,target,meets_target,"
                     "converged_price,temporal_error,headroom,admissible,"
                     "probe_seconds\n";
            std::ofstream table = open_csv(args.output_dir / "equal_accuracy.csv");
            table << "payoff,method,n,steps,tolerance,price,error,target,"
                     "temporal_error,headroom,threads,"
                     "repeats,median_seconds,min_seconds,max_seconds,"
                     "lower_quartile_seconds,upper_quartile_seconds\n";
            std::println("  {:<8} {:<9} {:>5} {:>7} {:>13} {:>11} {:>11} {:>11}",
                         "payoff", "method", "n", "steps", "price", "error",
                         "median s", "range s");

            struct Candidate {
                int n = 0;
                int steps = 0;
                double tolerance = 0.0;
                double price = 0.0;
                double error = 0.0;          ///< distance from the reference
                double temporal_error = 0.0; ///< distance from the converged price
                double headroom = 0.0;       ///< target less the converged error
                double probe_seconds = 0.0;
            };

            for (const auto& [payoff, rainbow] : kPayoffs) {
                const Reference& price = primary_price(reference, payoff);

                // The comparison is run on one grid, the accepted one. Both
                // methods converge to the same semi-discrete answer there, so
                // sweeping grids as well would only re-measure a spatial error
                // the two share, and would bury the question the table is asked
                // to settle: how much temporal work each method needs.
                //
                // The target is that grid's own converged error with a small
                // margin, which is what keeps the temporal setting a live
                // constraint. A looser target would be met by every step count
                // and the table would report nothing. The floor is four times
                // the reference uncertainty, below which the comparison would
                // be measuring the reference and not the methods.
                double target = args.error_target;
                if (!(target > 0.0)) {
                    const Config probe = settled(args.base_n, args.base_alpha,
                                                 args.temporal_ladder.back());
                    const PDESystem system = assemble(probe, rainbow);
                    const Solution solution = solve_and_extract(
                        system, probe, solve_adi_hv_s, reported_bump);
                    target = 1.05 * std::abs(solution.price - price.value);
                }
                if (!(target > 4.0 * price.uncertainty()))
                    target = 4.0 * price.uncertainty();

                const std::pair<std::string_view, SolverFunction> methods[2] = {
                    {"ADI-HV-S", solve_adi_hv_s}, {"KSM-EI", solve_ksm_ei}};

                for (const auto& [label, solver] : methods) {
                    std::vector<Candidate> passing;
                    for (const int n : {args.base_n}) {
                        const Config assembly = settled(n, args.base_alpha,
                                                        args.temporal_ladder.front());
                        const PDESystem system = assemble(assembly, rainbow);
                        std::vector<Candidate> row;
                        for (const int steps : args.equal_accuracy_ladder) {
                            Config model = settled(n, args.base_alpha, steps);
                            if (label == "KSM-EI") model.tol_ei = 1.0e-10;
                            const Solution solution =
                                solve_and_extract(system, model, solver, reported_bump);
                            if (!solution.finite) continue;
                            row.push_back(Candidate{
                                n, steps, label == "KSM-EI" ? model.tol_ei : 0.0,
                                solution.price, std::abs(solution.price - price.value),
                                0.0, 0.0, solution.seconds});
                        }
                        if (row.empty()) continue;

                        // A coarse step count can meet the target because its
                        // temporal error happens to cancel part of the spatial
                        // one, and a cancellation is not accuracy: refine the
                        // same configuration and it drifts back to the grid's
                        // converged answer rather than improving. Admissibility
                        // is therefore measured against that answer.
                        //
                        // The finest temporal setting in the ladder is what the
                        // grid converges to, and the error it still carries is
                        // spatial, so it is already spent. What the target does
                        // not spend on it is the whole budget left for temporal
                        // error, and a candidate is accepted only if its
                        // distance from the converged price fits inside that
                        // budget. Testing the total error alone would let a
                        // configuration buy its way in with the wrong sign.
                        const Candidate& converged = row.back();
                        const double headroom = target - converged.error;
                        for (Candidate candidate : row) {
                            const double temporal =
                                std::abs(candidate.price - converged.price);
                            candidate.temporal_error = temporal;
                            candidate.headroom = headroom;
                            const bool admissible =
                                candidate.error <= target && temporal <= headroom;
                            sweep << payoff << ',' << label << ',' << n << ','
                                  << candidate.steps << ',' << candidate.tolerance
                                  << ',' << candidate.price << ',' << candidate.error
                                  << ',' << target << ','
                                  << (candidate.error <= target ? 1 : 0) << ','
                                  << converged.price << ',' << temporal << ','
                                  << headroom << ',' << (admissible ? 1 : 0) << ','
                                  << candidate.probe_seconds << '\n';
                            if (admissible) passing.push_back(candidate);
                        }

                        // Cost rises with the grid and the error falls, so once
                        // a grid can meet the target no finer one can be cheaper.
                        if (!passing.empty()) break;
                    }
                    if (passing.empty()) {
                        gates.push_back(Gate{
                            "equal_accuracy_reachable_" + std::string(payoff) + "_"
                                + std::string(label),
                            false, true, target, target,
                            "no configuration in the sweep reached the target"});
                        continue;
                    }

                    // Cheapest is decided by the sweep's own measured cost, then
                    // that one configuration is timed properly.
                    const Candidate& choice = *std::ranges::min_element(
                        passing, {}, &Candidate::probe_seconds);
                    Config model = settled(choice.n, args.base_alpha, choice.steps);
                    if (label == "KSM-EI") model.tol_ei = choice.tolerance;
                    const PDESystem system = assemble(model, rainbow);

                    // One warm-up, discarded: it pays for the first-touch page
                    // faults that the assembled system has not yet triggered.
                    (void)solve_and_extract(system, model, solver, reported_bump);
                    std::vector<double> seconds;
                    double measured_price = 0.0;
                    for (int repeat = 0; repeat < args.repeats; ++repeat) {
                        const Solution solution =
                            solve_and_extract(system, model, solver, reported_bump);
                        seconds.push_back(solution.seconds);
                        measured_price = solution.price;
                    }
                    std::ranges::sort(seconds);
                    const auto quantile = [&](double fraction) {
                        const auto index = static_cast<std::size_t>(
                            fraction * static_cast<double>(seconds.size() - 1));
                        return seconds[index];
                    };
                    const double median = seconds[seconds.size() / 2];
                    const double error = std::abs(measured_price - price.value);

                    table << payoff << ',' << label << ',' << choice.n << ','
                          << choice.steps << ',' << choice.tolerance << ','
                          << measured_price << ',' << error << ',' << target << ','
                          << choice.temporal_error << ',' << choice.headroom
                          << ",1," << args.repeats << ',' << median << ','
                          << seconds.front() << ',' << seconds.back() << ','
                          << quantile(0.25) << ',' << quantile(0.75) << '\n';
                    std::println("  {:<8} {:<9} {:>5} {:>7} {:>13.6f} {:>11.2e} "
                                 "{:>11.4f} {:>11}",
                                 payoff, label, choice.n, choice.steps, measured_price,
                                 error,
                                 median,
                                 std::format("{:.4f}-{:.4f}", seconds.front(),
                                             seconds.back()));

                    gates.push_back(Gate{
                        "equal_accuracy_target_met_" + std::string(payoff) + "_"
                            + std::string(label),
                        error <= target, true, error, target,
                        "the timed configuration still meets the declared target"});
                }
                std::println("  {:<8} declared total price-error target {:.3e} "
                             "(reference floor {:.2e})",
                             payoff, target, price.uncertainty());
            }
            std::println("");
        }

        // Error budget
        if (args.stage == "all") {
            std::ofstream file = open_csv(args.output_dir / "error_budget.csv");
            file << "payoff,component,magnitude,evidence\n";
            std::println("Error budget");
            std::println("  {:<8} {:<26} {:>12}  {}",
                         "payoff", "component", "magnitude", "evidence");

            const auto domain_rows =
                std::filesystem::exists(args.output_dir / "pde_domain_residual.csv")
                    ? read_csv(args.output_dir / "pde_domain_residual.csv")
                    : std::vector<std::map<std::string, std::string>>{};
            const auto time_rows =
                std::filesystem::exists(args.output_dir / "pde_time.csv")
                    ? read_csv(args.output_dir / "pde_time.csv")
                    : std::vector<std::map<std::string, std::string>>{};

            for (const auto& [payoff, rainbow] : kPayoffs) {
                const Reference& price = primary_price(reference, payoff);
                const auto emit = [&](std::string_view component, double magnitude,
                                      std::string_view evidence) {
                    file << payoff << ',' << component << ',' << magnitude << ",\""
                         << evidence << "\"\n";
                    std::println("  {:<8} {:<26} {:>12.2e}  {}", payoff, component,
                                 magnitude, evidence);
                };

                emit("independent reference", price.uncertainty(),
                     price.source == "johnson"
                         ? "Johnson quadrature stability plus truncation"
                         : "randomized QMC 95 percent half-width");

                for (const auto& row : domain_rows)
                    if (row.at("payoff") == payoff)
                        emit("domain truncation", std::stod(row.at("domain_effect")),
                             "change between the two widest matched-spacing domains");

                if (const auto found = spatial.find(std::string(payoff));
                    found != spatial.end() && found->second.size() >= 2)
                    emit("spatial discretization",
                         std::abs(found->second.back().second
                                  - found->second[found->second.size() - 2].second),
                         "odd-grid refinement at the accepted domain");

                double adi_change = 0.0;
                double ksm_change = 0.0;
                for (const auto& row : time_rows) {
                    if (row.at("payoff") != payoff) continue;
                    const double change = std::stod(row.at("change_from_previous"));
                    if (!std::isfinite(change)) continue;
                    if (row.at("method") == "ADI-HV-S") adi_change = change;
                    else ksm_change = change;
                }
                if (adi_change > 0.0)
                    emit("ADI temporal", adi_change,
                         "smoothed HV time-step refinement, finest pair");
                if (ksm_change > 0.0)
                    emit("KSM temporal", ksm_change,
                         "tolerance refinement and the endpoint defect calibration");

                const auto delta = reference.find(std::string(payoff) + ".delta_1");
                if (delta != reference.end())
                    emit("interpolation and bump", delta->second.bump_error,
                         "exact-node test, trilinear tests, and the bump sweep");
            }
            std::println("");
        }

        {
            std::ofstream file = open_csv(args.output_dir / "validation_gates.csv");
            file << "gate,passed,blocking,measured,allowed,note\n";
            for (const Gate& gate : gates)
                file << gate.name << ',' << (gate.passed ? 1 : 0) << ','
                     << (gate.blocking ? 1 : 0) << ',' << gate.measured << ','
                     << gate.allowed << ",\"" << gate.note << "\"\n";
        }

        int failures = 0;
        std::println("Gates");
        for (const Gate& gate : gates) {
            if (gate.blocking && !gate.passed) ++failures;
            if (!gate.passed)
                std::println("  [FAIL] {:<44} measured={:.3e} allowed={:.3e}",
                             gate.name, gate.measured, gate.allowed);
        }
        std::println("  {} of {} gates passed",
                     gates.size() - static_cast<std::size_t>(failures), gates.size());
        std::println("\n  wrote {}", args.output_dir.string());

        if (failures > 0) {
            std::println(std::cerr, "Error: {} blocking validation gate(s) failed",
                         failures);
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::println(std::cerr, "Error: {}", error.what());
        return EXIT_FAILURE;
    }
}
