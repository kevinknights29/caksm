/**
 * @file krylov_indicator_calibration.cpp
 * @brief Compare the production endpoint defect indicator with true action error.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <print>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include "ca_arnoldi.hpp"
#include "config.hpp"
#include "gpu_ca_config.hpp"
#include "pde_operators.hpp"

namespace {

struct Args {
    int n = 5;
    int m_max = kGpuCaMaxM;
    int steps = 100;
    double expiry = 0.5;
    std::filesystem::path output_dir =
        "data/krylov-indicator-calibration";
};

struct Measurement {
    std::string option;
    double tau = 0.0;
    double h = 0.0;
    int width = 0;
    int m = 0;
    double indicator = 0.0;
    double true_error_2 = 0.0;
    double true_error_inf = 0.0;
    double relative_error_2 = 0.0;
    double indicator_over_error = 0.0;
    double arnoldi_relation = 0.0;
    double relative_arnoldi_relation = 0.0;
    double orthogonality = 0.0;
    double max_block_kappa = 0.0;
};

/// One representative action, reduced to the block-stability appendix row.
struct BlockStability {
    std::string option;
    double tau = 0.0;
    int width = 0;
    int blocks = 0;
    int cholesky_failures = 0;
    double max_block_residual = 0.0;
    double max_block_kappa = 0.0;
    double orthogonality = 0.0;
    double relative_arnoldi_relation = 0.0;
    int selected_m = 0;
    double indicator = 0.0;
    double referee_agreement = 0.0;
};

[[nodiscard]] Args parse_calibration_args(
    std::span<const char* const> argv)
{
    Args args;
    for (std::size_t i = 1; i < argv.size(); ++i) {
        const std::string_view arg = argv[i];
        const auto next = [&]() -> std::string_view {
            if (++i >= argv.size())
                throw std::invalid_argument(
                    "missing value for " + std::string(arg));
            return argv[i];
        };
        if (arg == "--n") args.n = std::stoi(std::string(next()));
        else if (arg == "--m-max")
            args.m_max = std::stoi(std::string(next()));
        else if (arg == "--steps")
            args.steps = std::stoi(std::string(next()));
        else if (arg == "--expiry")
            args.expiry = std::stod(std::string(next()));
        else if (arg == "--output-dir")
            args.output_dir = std::filesystem::path(std::string(next()));
        else if (arg == "--help") {
            std::println(
                "Usage: ./krylov-indicator-calibration [--n N] "
                "[--m-max M]\n"
                "       [--steps K] [--expiry T] [--output-dir DIR]");
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::invalid_argument(
                "unknown argument: " + std::string(arg));
        }
    }
    if (args.n < 3) throw std::invalid_argument("--n must be at least 3");
    if (args.m_max < 2)
        throw std::invalid_argument("--m-max must be at least 2");
    if (args.steps < 1)
        throw std::invalid_argument("--steps must be positive");
    if (!(args.expiry > 0.0) || !std::isfinite(args.expiry))
        throw std::invalid_argument("--expiry must be finite and positive");
    return args;
}

[[nodiscard]] SpMat production_operator(
    const PDESystem& system, bool rainbow)
{
    const MatXd forcing = rainbow
        ? MatXd::Zero(system.N, 3)
        : system.B;
    return build_A_tilde(system.A, forcing, system.N);
}

[[nodiscard]] Eigen::VectorXd production_initial_state(
    const PDESystem& system, bool rainbow)
{
    Eigen::VectorXd state = Eigen::VectorXd::Zero(system.N + 3);
    state.head(system.N) = system.u0;
    if (!rainbow) state.tail(3) = make_s_vec(0.0);
    return state;
}

[[nodiscard]] std::vector<Measurement> measure_case(
    const Args& args, bool rainbow, double tau, int width,
    BlockStability* stability = nullptr)
{
    Config model;
    model.n = args.n;
    model.t_final = args.expiry;
    const PDESystem system = build_pde_system(
        model.n, model.strike_price, model.risk_free_rate, model.t_final,
        model.sigma, model.rho_off, model.weight, model.initial_prices,
        model.alpha, rainbow);
    const SpMat op = production_operator(system, rainbow);
    const Eigen::MatrixXd dense_op = Eigen::MatrixXd(op);
    const Eigen::VectorXd initial =
        production_initial_state(system, rainbow);
    const double h = args.expiry / static_cast<double>(args.steps);
    const Eigen::VectorXd start =
        tau == 0.0 ? initial : (tau * dense_op).exp() * initial;
    const Eigen::VectorXd reference = (h * dense_op).exp() * start;
    const SpMat scaled_op = h * op;

    const int m_max = std::min(
        args.m_max, static_cast<int>(scaled_op.rows()) - 1);
    // The block residual is formed only where the appendix table asks for it,
    // so the calibration measurements themselves stay on the production path.
    const CaArnoldiResult arnoldi = ca_arnoldi(
        scaled_op, start, m_max, width, true, CaPolynomialBasis::Monomial, 0.0, 1.0,
        stability != nullptr);
    if (arnoldi.broke_down || arnoldi.m_used < m_max)
        throw std::runtime_error(
            "CA Arnoldi did not complete the calibration basis");

    std::vector<Measurement> rows;
    rows.reserve(static_cast<std::size_t>(m_max));
    const double reference_norm = reference.norm();
    for (int m = 1; m <= m_max; ++m) {
        const Eigen::MatrixXd hessenberg =
            arnoldi.H.topLeftCorner(m, m);
        const Eigen::VectorXd projected = hessenberg.exp().col(0);
        const Eigen::VectorXd approximation =
            arnoldi.beta * arnoldi.V.leftCols(m) * projected;
        const Eigen::VectorXd error = reference - approximation;
        const double true_error_2 = error.norm();
        const double indicator =
            arnoldi.beta * std::abs(arnoldi.H(m, m - 1))
            * std::abs(projected[m - 1]);

        const Eigen::MatrixXd image =
            scaled_op * arnoldi.V.leftCols(m);
        const Eigen::MatrixXd relation_error =
            image
            - arnoldi.V_extended.leftCols(m + 1)
                * arnoldi.H.topLeftCorner(m + 1, m);

        rows.push_back(Measurement{
            .option = rainbow ? "rainbow" : "basket",
            .tau = tau,
            .h = h,
            .width = width,
            .m = m,
            .indicator = indicator,
            .true_error_2 = true_error_2,
            .true_error_inf = error.lpNorm<Eigen::Infinity>(),
            .relative_error_2 = true_error_2
                / std::max(reference_norm, std::numeric_limits<double>::min()),
            .indicator_over_error = true_error_2 > 0.0
                ? indicator / true_error_2
                : std::numeric_limits<double>::quiet_NaN(),
            .arnoldi_relation = relation_error.norm(),
            .relative_arnoldi_relation = relation_error.norm()
                / std::max(image.norm(), std::numeric_limits<double>::min()),
            .orthogonality =
                orthogonality_loss(arnoldi.V.leftCols(m)),
            .max_block_kappa = arnoldi.max_kappa,
        });
    }

    if (stability != nullptr && !rows.empty()) {
        // The production stopping tolerance decides which Krylov dimension this
        // action would actually have used, and the referee agreement is that
        // dimension's distance from the dense exponential action.
        constexpr double production_tolerance = 1.0e-8;
        const auto selected = std::ranges::find_if(
            rows, [&](const Measurement& row) {
                return row.indicator < production_tolerance;
            });
        const Measurement& chosen = selected == rows.end() ? rows.back() : *selected;
        *stability = BlockStability{
            .option = rainbow ? "rainbow" : "basket",
            .tau = tau,
            .width = width,
            .blocks = arnoldi.blocks,
            .cholesky_failures = arnoldi.chol_failed,
            .max_block_residual = arnoldi.max_block_residual,
            .max_block_kappa = arnoldi.max_kappa,
            .orthogonality = chosen.orthogonality,
            .relative_arnoldi_relation = chosen.relative_arnoldi_relation,
            .selected_m = chosen.m,
            .indicator = chosen.indicator,
            .referee_agreement = chosen.true_error_2,
        };
    }
    return rows;
}

/**
 * @brief Write the compact block-stability appendix table.
 *
 * Empirical evidence about the blocks these production actions actually built,
 * not a proof and not a dimension-free certificate. It reuses the conditioning
 * campaign's numbers and adds only what was missing from them: whether every
 * Cholesky succeeded, how far each block factorization sat from its input, and
 * how far the selected Krylov dimension sat from the exponential-action referee.
 */
void write_block_stability(
    const std::filesystem::path& path,
    const std::vector<BlockStability>& rows)
{
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot create " + path.string());
    output << std::setprecision(17);
    output
        << "option,tau,s,blocks,cholesky_failures,cholesky_success_rate,"
           "max_block_residual,max_block_kappa,orthogonality,"
           "relative_arnoldi_relation,selected_m,indicator,referee_agreement\n";
    for (const BlockStability& row : rows) {
        const double success = row.blocks > 0
            ? static_cast<double>(row.blocks - row.cholesky_failures)
                / static_cast<double>(row.blocks)
            : std::numeric_limits<double>::quiet_NaN();
        output
            << row.option << ',' << row.tau << ',' << row.width << ','
            << row.blocks << ',' << row.cholesky_failures << ',' << success << ','
            << row.max_block_residual << ',' << row.max_block_kappa << ','
            << row.orthogonality << ',' << row.relative_arnoldi_relation << ','
            << row.selected_m << ',' << row.indicator << ','
            << row.referee_agreement << '\n';
    }
}

void write_measurements(
    const std::filesystem::path& path,
    const std::vector<Measurement>& rows)
{
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot create " + path.string());
    output << std::setprecision(17);
    output
        << "option,tau,h,s,m,indicator,true_error_2,true_error_inf,"
           "relative_error_2,indicator_over_error,arnoldi_relation,"
           "relative_arnoldi_relation,orthogonality,max_block_kappa\n";
    for (const Measurement& row : rows) {
        output
            << row.option << ',' << row.tau << ',' << row.h << ','
            << row.width << ',' << row.m << ',' << row.indicator << ','
            << row.true_error_2 << ',' << row.true_error_inf << ','
            << row.relative_error_2 << ',' << row.indicator_over_error << ','
            << row.arnoldi_relation << ','
            << row.relative_arnoldi_relation << ','
            << row.orthogonality << ',' << row.max_block_kappa << '\n';
    }
}

void write_stopping_summary(
    const std::filesystem::path& path,
    const std::vector<Measurement>& rows,
    double expiry)
{
    constexpr std::array tolerances{
        1.0e-4, 1.0e-6, 1.0e-8, 1.0e-10, 1.0e-12};
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot create " + path.string());
    output << std::setprecision(17);
    output
        << "option,tau,h,s,tolerance,selected_m,indicator,true_error_2,"
           "true_error_over_tolerance\n";

    for (const std::string_view option : {"basket", "rainbow"}) {
        for (const double tau : {0.0, 0.5 * expiry}) {
            for (const int width : {1, 4}) {
                for (const double tolerance : tolerances) {
                    const auto selected = std::ranges::find_if(
                        rows,
                        [&](const Measurement& row) {
                            return row.option == option && row.tau == tau
                                && row.width == width
                                && row.indicator < tolerance;
                        });
                    output << option << ',' << tau << ',';
                    if (selected == rows.end()) {
                        output
                            << "nan," << width << ',' << tolerance
                            << ",0,nan,nan,nan\n";
                    } else {
                        output
                            << selected->h << ',' << width << ',' << tolerance
                            << ',' << selected->m << ',' << selected->indicator
                            << ',' << selected->true_error_2 << ','
                            << selected->true_error_2 / tolerance << '\n';
                    }
                }
            }
        }
    }
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const Args args = parse_calibration_args(
            std::span<const char* const>(
                argv, static_cast<std::size_t>(argc)));
        std::filesystem::create_directories(args.output_dir);

        std::vector<Measurement> rows;
        std::vector<BlockStability> stability;
        for (const bool rainbow : {false, true}) {
            for (const double tau : {0.0, 0.5 * args.expiry}) {
                for (const int width : {1, 4}) {
                    BlockStability block;
                    std::vector<Measurement> measured =
                        measure_case(args, rainbow, tau, width, &block);
                    rows.insert(rows.end(), measured.begin(), measured.end());
                    stability.push_back(block);
                }
            }
        }

        write_measurements(args.output_dir / "measurements.csv", rows);
        write_stopping_summary(
            args.output_dir / "stopping.csv", rows, args.expiry);
        write_block_stability(args.output_dir / "block_stability.csv", stability);

        const auto max_relation = std::ranges::max_element(
            rows, {}, &Measurement::relative_arnoldi_relation);
        const auto max_orthogonality = std::ranges::max_element(
            rows, {}, &Measurement::orthogonality);
        std::println(
            "Krylov indicator calibration: {} rows, "
            "max relative Arnoldi relation={:.3e}, "
            "max orthogonality loss={:.3e}",
            rows.size(), max_relation->relative_arnoldi_relation,
            max_orthogonality->orthogonality);
        std::println("  wrote {}", args.output_dir.string());

        if (max_relation->relative_arnoldi_relation > 1.0e-10
            || max_orthogonality->orthogonality > 1.0e-10) {
            std::println(
                std::cerr,
                "Error: the calibration basis failed its numerical checks");
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::println(std::cerr, "Error: {}", error.what());
        return EXIT_FAILURE;
    }
}
