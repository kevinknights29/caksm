/**
 * @file config.hpp
 * @brief Simulation configuration and CLI argument parsing.
 * @author Kevin Knights
 */
#pragma once

#include <array>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <print>

enum class EuropeanOptionType {
    CALL_BASKET,
    CALL_MIN_RAINBOW,
};

struct Config {
    int n = 15;
    int temporal_steps = 100;
    int ei_steps = 100;
    double tol_ei = 1e-8;
    double strike_price = 100.0;
    double risk_free_rate = 0.04;
    double t_final = 1.0;
    std::array<double, 3> sigma = {0.30, 0.35, 0.40};
    std::array<double, 3> rho_off = {0.50, 0.50, 0.50};
    std::array<double, 3> weight = {1.0/3, 1.0/3, 1.0/3};
    std::array<double, 3> initial_prices = {100.0, 100.0, 100.0};
    double alpha = 2.85;
    EuropeanOptionType option_type = EuropeanOptionType::CALL_BASKET;
    bool benchmark = false;
    bool export_csv = false;
    bool save_referee = false;
    std::string referee_dir = "";
    std::vector<std::string> skip_methods = {};
};

inline Config parse_args(std::span<const char* const> args)
{
    Config cfg;
    bool tol_given = false;
    for (std::size_t i = 1; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        auto next = [&]() -> std::string_view {
            if (++i >= args.size())
                throw std::invalid_argument("Missing value for " + std::string(arg));
            return args[i];
        };
        if      (arg == "--benchmark")    cfg.benchmark     = true;
        else if (arg == "--export")       cfg.export_csv    = true;
        else if (arg == "--save-referee") cfg.save_referee  = true;
        else if (arg == "--referee-dir")  cfg.referee_dir   = std::string(next());
        else if (arg == "--skip")         cfg.skip_methods.emplace_back(next());
        else if (arg == "--n")            cfg.n              = std::stoi(std::string(next()));
        else if (arg == "--steps")        cfg.temporal_steps = std::stoi(std::string(next()));
        else if (arg == "--tol") {
            cfg.tol_ei = std::stod(std::string(next()));
            tol_given  = true;
        }
        else if (arg == "--option") {
            const auto opt = next();
            if      (opt == "basket")  cfg.option_type = EuropeanOptionType::CALL_BASKET;
            else if (opt == "rainbow") cfg.option_type = EuropeanOptionType::CALL_MIN_RAINBOW;
            else throw std::invalid_argument("Unknown option type: " + std::string(opt));
        }
        else if (arg == "--help") {
            std::println("Usage: ./pricer --benchmark [--option basket|rainbow]");
            std::println("               [--n N] [--steps M] [--tol T] [--export]");
            std::println("               [--referee-dir DIR]");
            std::println("       ./pricer --save-referee --n N --referee-dir DIR");
            std::println("  --steps M          temporal steps for CN, ADI-DR, ADI-HV, KSM-EI");
            std::println("  --tol T            KSM-EI convergence tolerance (default 1e-8)");
            std::println("  --export           write results to caksm_export_<timestamp>.csv");
            std::println("  --save-referee     compute ME referee and save to DIR; then exit");
            std::println("  --referee-dir DIR  load pre-computed referees from DIR");
            std::println("  --skip METHOD      omit METHOD (CN, ADI-DR, ADI-HV, ME, KSM-EI)");
            std::exit(0);
        }
        else throw std::invalid_argument("Unknown flag: " + std::string(arg));
    }
    cfg.ei_steps = tol_given ? 100 : cfg.temporal_steps;
    return cfg;
}
