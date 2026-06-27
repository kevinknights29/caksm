/**
 * @file main.cpp
 * @brief Benchmark runner for the European option pricer.
 *
 *   ./pricer --benchmark [--option basket|rainbow] [--n N] [--steps M]
 *            [--tol T] [--export] [--referee-dir DIR] [--skip METHOD]
 *
 * @author Kevin Knights
 * @version 1.2
 * @date 2026-06-26
 */

#include <algorithm>
#include <chrono>
#include <ctime>
#include <format>
#include <fstream>
#include <iostream>
#include <optional>
#include <print>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

#include "config.hpp"
#include "pde_operators.hpp"
#include "solvers.hpp"

// Benchmark runner
void run_benchmark(const Config& cfg, EuropeanOptionType option_type,
                   std::ofstream* csv = nullptr,
                   const VecXd* preloaded_ref = nullptr)
{
    const bool rainbow = (option_type == EuropeanOptionType::CALL_MIN_RAINBOW);
    const std::string_view opt_label = rainbow ? "Rainbow min-call" : "Basket call";
    const double reference           = rainbow ? 4.4450 : 13.2449;

    std::println("\n {}", opt_label);
    std::println("  Parameters: n={}, steps={}, K={:.0f}, r={:.2f}, T={:.1f}",
                 cfg.n, cfg.temporal_steps,
                 cfg.strike_price, cfg.risk_free_rate, cfg.t_final);
    std::println("  sigma=[{:.2f},{:.2f},{:.2f}]  rho_off=[{:.1f},{:.1f},{:.1f}]",
                 cfg.sigma[0], cfg.sigma[1], cfg.sigma[2],
                 cfg.rho_off[0], cfg.rho_off[1], cfg.rho_off[2]);
    std::println("  KSM-EI: tol={:.2e}, steps={}", cfg.tol_ei, cfg.ei_steps);
    if (!cfg.skip_methods.empty()) {
        std::print("  Skipping:");
        for (const auto& m : cfg.skip_methods) std::print(" {}", m);
        std::println("");
    }
    std::println("  Reference price: {:.4f}\n", reference);

    std::println("  [Building PDE system...]");
    const PDESystem sys = build_pde_system(
        cfg.n, cfg.strike_price, cfg.risk_free_rate, cfg.t_final,
        cfg.sigma, cfg.rho_off, cfg.weight, cfg.initial_prices,
        cfg.alpha, rainbow);

    VecXd ref_u;
    if (preloaded_ref && preloaded_ref->size() > 0) {
        std::println("  [ME referee: loaded from file]");
        ref_u = *preloaded_ref;
    } else {
        std::println("  [Computing ME referee...]");
        ref_u = compute_me_referee(sys, cfg);
    }
    const VecXd ref_cube = extract_cube(ref_u, sys.grid, cfg.initial_prices);

    std::println("  {:<12} {:>10}  {:>10}  {:>10}  {:>10}",
                 "Method", "Price", "PDE Err", "ODE Err", "Time(ms)");
    std::println("  {}", std::string(60, '-'));

    struct Run {
        std::string_view name;
        VecXd (*fn)(const PDESystem&, const Config&);
    };
    const Run runs[] = {
        {"CN",      solve_cn},
        {"ADI-DR",  solve_adi_dr},
        {"ADI-HV",  solve_adi_hv},
        {"ME",      solve_me},
        {"KSM-EI",  solve_ksm_ei},
    };

    for (const auto& run : runs) {
        if (std::ranges::any_of(cfg.skip_methods,
                [&](const std::string& s) { return s == run.name; }))
            continue;

        const auto t0 = std::chrono::steady_clock::now();
        const VecXd u = run.fn(sys, cfg);
        const auto t1 = std::chrono::steady_clock::now();

        const double ms      = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double price   = extract_price(u, sys.grid, cfg.initial_prices);
        const double pde_err = std::abs(price - reference);
        const double ode_err = (extract_cube(u, sys.grid, cfg.initial_prices) - ref_cube).norm();

        std::println("  {:<12} {:>10.4f}  {:>10.4f}  {:>10.3e}  {:>10.1f}",
                     run.name, price, pde_err, ode_err, ms);

        if (csv) {
            *csv << std::format("{},{},{},{},{:.6e},{},{:.6f},{:.6f},{:.6e},{:.3f}\n",
                                (rainbow ? "rainbow" : "basket"),
                                cfg.n, cfg.temporal_steps, cfg.ei_steps,
                                cfg.tol_ei, run.name,
                                price, pde_err, ode_err, ms);
        }
    }
    std::println("");
}

int main(const int argc, char* argv[])
{
    try {
        const Config cfg = parse_args(
            std::span<const char* const>(argv, static_cast<std::size_t>(argc)));

        if (cfg.save_referee) {
            if (cfg.referee_dir.empty())
                throw std::invalid_argument("--save-referee requires --referee-dir <path>");
            std::println("European Option PDE Pricer [Save Referee]");
            std::println("  n={}, saving to {}/", cfg.n, cfg.referee_dir);
            for (const bool rainbow : {false, true}) {
                const std::string label = rainbow ? "rainbow" : "basket";
                std::println("\n  Building PDE system ({})...", label);
                const PDESystem sys = build_pde_system(
                    cfg.n, cfg.strike_price, cfg.risk_free_rate, cfg.t_final,
                    cfg.sigma, cfg.rho_off, cfg.weight, cfg.initial_prices,
                    cfg.alpha, rainbow);
                std::println("  Computing ME referee ({})...", label);
                const VecXd ref = compute_me_referee(sys, cfg);
                const std::string path = referee_path(cfg.referee_dir, cfg.n, rainbow);
                save_referee_file(ref, path);
                std::println("  Saved: {}", path);
            }
            return EXIT_SUCCESS;
        }

        if (!cfg.benchmark) {
            std::println("No action specified. Use --benchmark or --save-referee.");
            std::println("Run with --help for usage information.");
            return EXIT_SUCCESS;
        }

        std::println("European Option PDE Pricer [Benchmark]");

        std::optional<VecXd> basket_ref, rainbow_ref;
        if (!cfg.referee_dir.empty()) {
            auto try_load = [&](bool rainbow) -> std::optional<VecXd> {
                const std::string path = referee_path(cfg.referee_dir, cfg.n, rainbow);
                VecXd v = load_referee_file(path);
                if (v.size() > 0) {
                    std::println("  Referee loaded: {}", path);
                    return v;
                }
                std::println("  Warning: referee not found at {}, will compute", path);
                return std::nullopt;
            };
            basket_ref  = try_load(false);
            rainbow_ref = try_load(true);
        }

        std::ofstream csv;
        if (cfg.export_csv) {
            std::time_t now = std::time(nullptr);
            char ts[20];
            std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", std::localtime(&now));
            const std::string csv_path = std::format("caksm_export_n{}_{}.csv", cfg.n, ts);
            csv.open(csv_path);
            if (!csv)
                throw std::runtime_error("Cannot open CSV file: " + csv_path);
            csv << "option_type,n,temporal_steps,ei_steps,tol_ei,"
                   "method,price,pde_err,ode_err,time_ms\n";
            std::println("  Exporting to {}\n", csv_path);
        }

        std::ofstream* csv_ptr = cfg.export_csv ? &csv : nullptr;
        run_benchmark(cfg, EuropeanOptionType::CALL_BASKET,
                      csv_ptr,
                      basket_ref.has_value()  ? &basket_ref.value()  : nullptr);
        run_benchmark(cfg, EuropeanOptionType::CALL_MIN_RAINBOW,
                      csv_ptr,
                      rainbow_ref.has_value() ? &rainbow_ref.value() : nullptr);

    } catch (const std::exception& e) {
        std::println(std::cerr, "Error: {}", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
