/**
 * @file test_financial_validation.cpp
 * @brief Gates for the independent references, the log-grid interpolator, and
 *        the Hundsdorfer-Verwer start-up smoothing.
 */

#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "financial_reference.hpp"
#include "pde_operators.hpp"
#include "solvers.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace fr = financial_reference;

namespace {

/// The base case: the three-asset experiment of Dang, Christara and Jackson.
[[nodiscard]] Config base_model()
{
    return Config{};
}

/// A grid small enough to solve densely, large enough to have interior cells.
[[nodiscard]] PDESystem small_system(int n, bool rainbow, int steps)
{
    Config model = base_model();
    model.n = n;
    model.temporal_steps = steps;
    return build_pde_system(model.n, model.strike_price, model.risk_free_rate,
                            model.t_final, model.sigma, model.rho_off, model.weight,
                            model.initial_prices, model.alpha, rainbow);
}

} // namespace

// The analytical Rainbow reference

/// The quantile has to invert the CDF, not merely approximate it, because every
/// Sobol point is pushed through it and a biased tail would bias the price.
///
/// Expected: the round trip is exact to 1e-13 relative across twelve decades of
/// tail probability, which is what the Halley correction buys over the bare
/// rational approximation.
TEST_CASE("the normal quantile inverts the normal CDF to roundoff",
          "[financial][reference][normal]")
{
    for (double p = 1.0e-12; p < 1.0; p *= 1.7) {
        const double x = fr::normal_quantile(p);
        REQUIRE_THAT(fr::normal_cdf(x), WithinRel(p, 1.0e-13));
    }
    REQUIRE_THAT(fr::normal_quantile(0.5), WithinAbs(0.0, 1.0e-15));
    REQUIRE_THROWS(fr::normal_quantile(0.0));
    REQUIRE_THROWS(fr::normal_quantile(1.0));
}

/// The multivariate normal CDFs are checked against closed forms rather than
/// against another quadrature, so agreement cannot come from a shared mistake.
///
/// Expected: independence factorizes exactly; the equicorrelated trivariate
/// orthant at the origin matches 1/8 + 3 asin(rho) / (4 pi); and an infinite
/// threshold reduces each to the lower-dimensional case.
TEST_CASE("the multivariate normal CDFs match their closed forms",
          "[financial][reference][mvn]")
{
    const fr::MultivariateNormal normal{fr::QuadratureSettings{}};
    constexpr double infinity = std::numeric_limits<double>::infinity();

    REQUIRE_THAT(normal.bivariate(1.0, -0.4, 0.0),
                 WithinRel(fr::normal_cdf(1.0) * fr::normal_cdf(-0.4), 1.0e-14));
    REQUIRE_THAT(normal.bivariate(0.0, 0.0, 0.5), WithinRel(1.0 / 3.0, 1.0e-13));
    REQUIRE_THAT(normal.bivariate(infinity, 0.3, 0.7),
                 WithinRel(fr::normal_cdf(0.3), 1.0e-14));

    const fr::Matrix3 independent = fr::correlation_matrix({0.0, 0.0, 0.0});
    REQUIRE_THAT(normal.trivariate({0.3, -0.7, 1.1}, independent),
                 WithinRel(fr::normal_cdf(0.3) * fr::normal_cdf(-0.7)
                           * fr::normal_cdf(1.1), 1.0e-13));

    for (const double rho : {-0.4, 0.25, 0.5, 0.8}) {
        const double closed =
            0.125 + 3.0 * std::asin(rho) / (4.0 * std::numbers::pi);
        REQUIRE_THAT(normal.trivariate({0.0, 0.0, 0.0},
                                       fr::correlation_matrix({rho, rho, rho})),
                     WithinRel(closed, 1.0e-12));
    }
    REQUIRE_THAT(normal.trivariate({0.6, infinity, infinity}, independent),
                 WithinRel(fr::normal_cdf(0.6), 1.0e-13));
}

/// A correlation matrix that is not one must be refused before it reaches the
/// Cholesky, so an inadmissible model gives a description and not a NaN price.
TEST_CASE("inadmissible correlation matrices are rejected",
          "[financial][reference][correlation]")
{
    REQUIRE_NOTHROW(fr::correlation_cholesky(fr::correlation_matrix({0.5, 0.5, 0.5})));
    // Pairwise admissible, jointly indefinite.
    REQUIRE_THROWS(fr::correlation_cholesky(
        fr::correlation_matrix({0.9, 0.9, -0.9})));
    REQUIRE_THROWS(fr::correlation_cholesky(fr::correlation_matrix({1.0, 0.5, 0.5})));

    fr::Matrix3 asymmetric = fr::correlation_matrix({0.5, 0.5, 0.5});
    asymmetric[0][1] = 0.4;
    REQUIRE_THROWS(fr::correlation_cholesky(asymmetric));
}

/// The Johnson formula must be the published one, not an expression tuned to
/// reproduce a known number.
///
/// Expected: the base case rounds to the published 4.4450; doubling every
/// quadrature setting leaves the value alone; relabeling the assets leaves it
/// alone; and it stays inside its own bounds. Pushing the two companion assets
/// far above the strike leaves a Black-Scholes call on the remaining one, which
/// is an independent closed form the formula must reduce to.
TEST_CASE("the Johnson call on the minimum reproduces its published value",
          "[financial][reference][johnson][critical]")
{
    const fr::QuadratureSettings settings;
    const fr::MultivariateNormal normal{settings};
    const Config model = base_model();
    const fr::JohnsonResult result = fr::johnson_call_on_minimum(model, normal);

    REQUIRE_THAT(result.price, WithinAbs(4.4450, 5.0e-5));
    REQUIRE(result.price > 0.0);
    REQUIRE(result.price < result.discounted_minimum_expectation);

    const fr::MultivariateNormal tightened{settings.tightened()};
    REQUIRE_THAT(fr::johnson_call_on_minimum(model, tightened).price,
                 WithinAbs(result.price, 1.0e-10));

    Config relabeled = model;
    relabeled.sigma = {model.sigma[2], model.sigma[0], model.sigma[1]};
    REQUIRE_THAT(fr::johnson_call_on_minimum(relabeled, normal).price,
                 WithinAbs(result.price, 1.0e-10));

    // Two assets far out of reach leave the minimum on the third one alone.
    Config single = model;
    single.initial_prices = {100.0, 1.0e7, 1.0e7};
    single.sigma = {0.30, 0.01, 0.01};
    REQUIRE_THAT(fr::johnson_call_on_minimum(single, normal).price,
                 WithinAbs(fr::black_scholes_call(100.0, model.strike_price,
                                                  model.risk_free_rate, 0.30,
                                                  model.t_final), 1.0e-6));
}

/// Where the formula is defined at a vanishing expiry it must return the
/// intrinsic payoff, in and out of the money.
TEST_CASE("the Johnson formula reduces to the intrinsic payoff at expiry",
          "[financial][reference][johnson]")
{
    const fr::MultivariateNormal normal{fr::QuadratureSettings{}};
    Config model = base_model();
    model.t_final = 1.0e-10;

    model.initial_prices = {120.0, 130.0, 140.0};
    REQUIRE_THAT(fr::johnson_call_on_minimum(model, normal).price,
                 WithinAbs(20.0, 1.0e-6));

    model.initial_prices = {80.0, 90.0, 95.0};
    REQUIRE_THAT(fr::johnson_call_on_minimum(model, normal).price,
                 WithinAbs(0.0, 1.0e-6));
}

// The randomized QMC reference

/// The interval is built across scrambles, so the Student t quantile that
/// scales it has to be the published one.
TEST_CASE("the Student t quantile matches published tables",
          "[financial][reference][uncertainty]")
{
    REQUIRE_THAT(fr::student_t_two_sided_quantile(0.95, 1), WithinAbs(12.70620, 1.0e-5));
    REQUIRE_THAT(fr::student_t_two_sided_quantile(0.95, 15), WithinAbs(2.13145, 1.0e-5));
    REQUIRE_THAT(fr::student_t_two_sided_quantile(0.95, 30), WithinAbs(2.04227, 1.0e-5));
    REQUIRE_THAT(fr::student_t_two_sided_quantile(0.99, 15), WithinAbs(2.94671, 1.0e-5));
    REQUIRE_THROWS(fr::student_t_two_sided_quantile(0.95, 0));
}

/// Scrambled Sobol draws must be a randomization of the same net, not fresh
/// random numbers: the sequence has to reproduce under its seed, differ under
/// another, and stay strictly inside the open unit cube.
TEST_CASE("scrambled Sobol points reproduce, differ, and stay interior",
          "[financial][reference][sobol]")
{
    STATIC_REQUIRE(fr::kDefaultQmcSeed == 25'378'095ull);
    STATIC_REQUIRE(fr::kDefaultQmcSeed
                   < std::numeric_limits<std::uint64_t>::max());

    std::array<double, 3> point{};
    std::vector<double> first;
    fr::ScrambledSobol a(1234ull, 0);
    for (int i = 0; i < 4096; ++i) {
        a.next(point);
        for (const double value : point) {
            REQUIRE(value > 0.0);
            REQUIRE(value < 1.0);
            first.push_back(value);
        }
    }

    fr::ScrambledSobol b(1234ull, 0);
    for (std::size_t i = 0; i < first.size(); i += 3) {
        b.next(point);
        for (std::size_t d = 0; d < 3; ++d) REQUIRE(point[d] == first[i + d]);
    }

    // A different scramble index and a different seed must both move the points.
    fr::ScrambledSobol c(1234ull, 1);
    c.next(point);
    REQUIRE(point[0] != first[0]);
    fr::ScrambledSobol d(1235ull, 0);
    d.next(point);
    REQUIRE(point[0] != first[0]);

    // Each coordinate must be equidistributed, which is the property the whole
    // construction rests on and the first thing a broken scramble destroys.
    fr::ScrambledSobol e(99ull, 3);
    std::array<int, 8> occupancy{};
    constexpr int draws = 4096;
    for (int i = 0; i < draws; ++i) {
        e.next(point);
        occupancy[static_cast<std::size_t>(point[2] * 8.0)] += 1;
    }
    for (const int count : occupancy) REQUIRE(count == draws / 8);
}

/// Limiting cases the estimator must reproduce exactly or inside its interval.
///
/// Expected: a deterministic terminal value gives the discounted payoff with no
/// sampling error at all; a one-asset basket is Black-Scholes; and the Rainbow
/// estimate agrees with the Johnson value inside the reported band.
TEST_CASE("randomized QMC reproduces its limiting cases",
          "[financial][reference][qmc][critical]")
{
    const Config model = base_model();
    fr::QmcRequest request;
    request.log2_points = 12;
    request.scrambles = 16;
    request.relative_bumps.clear();

    Config flat = model;
    flat.sigma = {1.0e-12, 1.0e-12, 1.0e-12};
    const double deterministic =
        std::exp(-model.risk_free_rate * model.t_final)
        * std::max(model.initial_prices[0]
                       * std::exp(model.risk_free_rate * model.t_final)
                   - model.strike_price, 0.0);
    for (const auto& run : fr::run_qmc(flat, request).basket)
        REQUIRE_THAT(run.price, WithinAbs(deterministic, 1.0e-9));

    Config single = model;
    single.weight = {1.0, 0.0, 0.0};
    fr::QmcRequest larger = request;
    larger.log2_points = 16;
    const fr::Interval reduced = fr::scramble_interval([&] {
        std::vector<double> values;
        for (const auto& run : fr::run_qmc(single, larger).basket)
            values.push_back(run.price);
        return values;
    }());
    REQUIRE_THAT(reduced.mean,
                 WithinAbs(fr::black_scholes_call(model.initial_prices[0],
                                                  model.strike_price,
                                                  model.risk_free_rate,
                                                  model.sigma[0], model.t_final),
                           reduced.half_width));

    const fr::MultivariateNormal normal{fr::QuadratureSettings{}};
    const double johnson = fr::johnson_call_on_minimum(model, normal).price;
    const fr::Interval rainbow = fr::scramble_interval([&] {
        std::vector<double> values;
        for (const auto& run : fr::run_qmc(model, larger).rainbow)
            values.push_back(run.price);
        return values;
    }());
    REQUIRE_THAT(rainbow.mean, WithinAbs(johnson, rainbow.half_width));
}

/// Common points are what make the Greek estimates usable, and their signature
/// is that the center is untouched by the bump machinery.
///
/// Expected: adding bumps leaves every scramble's price bit for bit identical,
/// and Delta lands near the analytical Black-Scholes Delta in the one-asset
/// reduction where a closed form exists.
TEST_CASE("common-point Greeks leave the center alone",
          "[financial][reference][qmc][greeks]")
{
    Config model = base_model();
    model.weight = {1.0, 0.0, 0.0};

    fr::QmcRequest plain;
    plain.log2_points = 14;
    plain.scrambles = 16;
    plain.relative_bumps.clear();
    fr::QmcRequest bumped = plain;
    bumped.relative_bumps = {0.005, 0.01, 0.02};

    const fr::QmcResult without = fr::run_qmc(model, plain);
    const fr::QmcResult with = fr::run_qmc(model, bumped);
    for (std::size_t i = 0; i < without.basket.size(); ++i)
        REQUIRE(without.basket[i].price == with.basket[i].price);

    const double root = model.sigma[0] * std::sqrt(model.t_final);
    const double d1 = (std::log(model.initial_prices[0] / model.strike_price)
                       + (model.risk_free_rate + 0.5 * model.sigma[0] * model.sigma[0])
                             * model.t_final) / root;
    const fr::Interval delta = fr::scramble_interval([&] {
        std::vector<double> values;
        for (const auto& run : with.basket) values.push_back(run.bump[1].delta[0]);
        return values;
    }());
    REQUIRE_THAT(delta.mean,
                 WithinAbs(fr::normal_cdf(d1), 10.0 * delta.half_width + 1.0e-4));
}

// Trilinear log-grid interpolation

/// The interpolator has to return the stored value at a node, reproduce a field
/// that is trilinear in log price to roundoff inside a cell, and refuse a point
/// outside the domain rather than extrapolating past the boundary closure.
TEST_CASE("trilinear log-grid interpolation meets its gates",
          "[financial][interpolation][critical]")
{
    const Config model = base_model();
    const Grid grid = build_grid(15, model.initial_prices, model.sigma, model.alpha,
                                 model.t_final);
    const int n = grid.n;

    // A field that is exactly trilinear in the log coordinates.
    VecXd field(n * n * n);
    const auto exact = [&](double x0, double x1, double x2) {
        return 3.0 + 2.0 * x0 - 1.5 * x1 + 0.75 * x2
             + 0.5 * x0 * x1 - 0.25 * x0 * x2 + 0.125 * x1 * x2
             + 0.0625 * x0 * x1 * x2;
    };
    for (int i3 = 0; i3 < n; ++i3)
        for (int i2 = 0; i2 < n; ++i2)
            for (int i1 = 0; i1 < n; ++i1)
                field[i3 * n * n + i2 * n + i1] =
                    exact(grid.x[0][i1], grid.x[1][i2], grid.x[2][i3]);

    // Exact at every node, and with the index order the rest of the code uses.
    for (const int i3 : {0, 4, n / 2, n - 1})
        for (const int i2 : {0, 7, n - 1})
            for (const int i1 : {1, n / 2, n - 1}) {
                const std::array<double, 3> spot{std::exp(grid.x[0][i1]),
                                                 std::exp(grid.x[1][i2]),
                                                 std::exp(grid.x[2][i3])};
                REQUIRE_THAT(interpolate_price_trilinear(field, grid, spot),
                             WithinRel(field[i3 * n * n + i2 * n + i1], 1.0e-13));
            }

    // Exact inside a cell, which a nearest-node lookup could not be.
    for (const double fraction : {0.13, 0.5, 0.87}) {
        std::array<double, 3> point{};
        for (int d = 0; d < 3; ++d)
            point[static_cast<std::size_t>(d)] =
                grid.x[d][n / 2] + fraction * grid.dx[d];
        const std::array<double, 3> spot{std::exp(point[0]), std::exp(point[1]),
                                         std::exp(point[2])};
        REQUIRE_THAT(interpolate_price_trilinear(field, grid, spot),
                     WithinRel(exact(point[0], point[1], point[2]), 1.0e-12));
    }

    // Outside the domain is refused rather than silently extrapolated.
    const std::array<double, 3> inside{std::exp(grid.x[0][n / 2]),
                                       std::exp(grid.x[1][n / 2]),
                                       std::exp(grid.x[2][n / 2])};
    REQUIRE_NOTHROW(interpolate_price_trilinear(field, grid, inside));
    for (int d = 0; d < 3; ++d) {
        std::array<double, 3> beyond = inside;
        beyond[static_cast<std::size_t>(d)] =
            std::exp(grid.x[d][n - 1] + 0.5 * grid.dx[d]);
        REQUIRE_THROWS_AS(interpolate_price_trilinear(field, grid, beyond),
                          std::out_of_range);
        beyond[static_cast<std::size_t>(d)] =
            std::exp(grid.x[d][0] - 0.5 * grid.dx[d]);
        REQUIRE_THROWS_AS(interpolate_price_trilinear(field, grid, beyond),
                          std::out_of_range);
    }
    REQUIRE_THROWS_AS(interpolate_price_trilinear(field, grid, {0.0, 100.0, 100.0}),
                      std::out_of_range);

    // Odd centered grids put the base spots on nodes, which is what lets the
    // node lookup and the interpolator be compared at all.
    REQUIRE(spots_are_grid_nodes(grid, model.initial_prices));
    REQUIRE_THAT(interpolate_price_trilinear(field, grid, model.initial_prices),
                 WithinRel(extract_price(field, grid, model.initial_prices), 1.0e-13));
}

// Hundsdorfer-Verwer start-up smoothing

/// Zero smoothing steps must be the existing solver, or the historical ADI-HV
/// results would quietly change meaning when the new entry point is adopted.
TEST_CASE("zero smoothing steps reproduce the unsmoothed HV solver",
          "[financial][adi][smoothing][critical]")
{
    for (const bool rainbow : {false, true}) {
        Config model = base_model();
        model.n = 11;
        model.temporal_steps = 12;
        const PDESystem system = small_system(model.n, rainbow, model.temporal_steps);

        const VecXd unsmoothed = solve_adi_hv(system, model);
        const VecXd zero = solve_adi_hv_smoothed(system, model, 0);
        REQUIRE(zero.size() == unsmoothed.size());
        REQUIRE((zero - unsmoothed).lpNorm<Eigen::Infinity>()
                <= 1.0e-12 * unsmoothed.lpNorm<Eigen::Infinity>());
    }
}

/// The switch from theta = 1 to theta = 0.5 must land exactly at the requested
/// step index, which is what pins down "the first two steps" as a claim.
///
/// Expected: each extra smoothed step changes the answer while it still falls
/// inside the run, and asking for more smoothing than there are steps changes
/// nothing further. Saturating at exactly temporal_steps is what shows the
/// switch is counted in steps taken and not in elapsed time or stage index.
TEST_CASE("start-up smoothing switches at the requested step index",
          "[financial][adi][smoothing][critical]")
{
    Config model = base_model();
    model.n = 11;
    model.temporal_steps = 5;
    const PDESystem system = small_system(model.n, false, model.temporal_steps);

    std::vector<VecXd> runs;
    for (int smoothing = 0; smoothing <= model.temporal_steps + 2; ++smoothing)
        runs.push_back(solve_adi_hv_smoothed(system, model, smoothing));

    // Every additional smoothed step inside the run moves the answer.
    for (int smoothing = 1; smoothing <= model.temporal_steps; ++smoothing) {
        const auto here = static_cast<std::size_t>(smoothing);
        REQUIRE((runs[here] - runs[here - 1]).lpNorm<Eigen::Infinity>() > 1.0e-8);
    }
    // Beyond the last step there is nothing left to smooth.
    for (int smoothing = model.temporal_steps + 1;
         smoothing <= model.temporal_steps + 2; ++smoothing) {
        const auto here = static_cast<std::size_t>(smoothing);
        REQUIRE((runs[here] - runs[here - 1]).lpNorm<Eigen::Infinity>() == 0.0);
    }

    // The configured entry point is the two-step default of the paper.
    REQUIRE(model.hv_smoothing_steps == 2);
    REQUIRE((solve_adi_hv_s(system, model) - runs[2]).lpNorm<Eigen::Infinity>() == 0.0);
    REQUIRE_THROWS(solve_adi_hv_smoothed(system, model, -1));
}

/// The smoothed scheme must still be second order in time away from the
/// start-up region, for both the homogeneous Rainbow system and the Basket
/// system with its time-dependent boundary forcing.
///
/// Expected: halving the step size roughly quarters the change in the spot
/// price, giving an observed order near two. The check is against the solver's
/// own finest run, so it is a temporal statement and carries no spatial error.
TEST_CASE("smoothed HV is second order in time after the start-up region",
          "[financial][adi][smoothing][order]")
{
    for (const bool rainbow : {false, true}) {
        Config model = base_model();
        model.n = 11;
        const PDESystem system = small_system(model.n, rainbow, 16);

        std::vector<double> prices;
        for (const int steps : {16, 32, 64, 128}) {
            Config level = model;
            level.temporal_steps = steps;
            const VecXd field = solve_adi_hv_smoothed(system, level, 2);
            prices.push_back(
                interpolate_price_trilinear(field, system.grid, level.initial_prices));
        }
        const double first = prices[0] - prices[1];
        const double second = prices[1] - prices[2];
        const double third = prices[2] - prices[3];
        REQUIRE(std::abs(second) > 0.0);
        REQUIRE(std::abs(third) > 0.0);
        REQUIRE_THAT(std::log2(first / second), WithinAbs(2.0, 0.45));
        REQUIRE_THAT(std::log2(second / third), WithinAbs(2.0, 0.45));
    }
}

/// The Basket forcing path is the one the stage equations treat differently, so
/// it is checked against the high-accuracy exponential action of the same
/// semi-discrete system as the ADI step is refined.
///
/// Expected: the gap to the referee shrinks with the step size and reaches
/// second order, which it could not do if the forcing entered the wrong stage
/// or at the wrong time level.
TEST_CASE("the Basket forcing path converges to the exponential-action referee",
          "[financial][adi][forcing][critical]")
{
    Config model = base_model();
    model.n = 9;
    const PDESystem system = small_system(model.n, false, 16);
    REQUIRE(system.has_forcing);

    const VecXd referee = compute_me_referee(system, model);
    const double exact =
        interpolate_price_trilinear(referee, system.grid, model.initial_prices);

    std::vector<double> errors;
    for (const int steps : {32, 64, 128}) {
        Config level = model;
        level.temporal_steps = steps;
        const VecXd field = solve_adi_hv_smoothed(system, level, 2);
        errors.push_back(std::abs(
            interpolate_price_trilinear(field, system.grid, level.initial_prices)
            - exact));
    }
    REQUIRE(errors[1] < errors[0]);
    REQUIRE(errors[2] < errors[1]);
    REQUIRE_THAT(std::log2(errors[0] / errors[1]), WithinAbs(2.0, 0.6));
    REQUIRE_THAT(std::log2(errors[1] / errors[2]), WithinAbs(2.0, 0.6));
}
