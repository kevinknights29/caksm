/**
 * @file financial_reference.hpp
 * @brief Independent terminal-distribution references for the three-asset
 *        European Basket and Rainbow contracts.
 *
 * The Johnson formula supplies the primary Rainbow price. Randomized
 * quasi-Monte Carlo supplies the primary Basket price, an independent Rainbow
 * cross-check, and the Delta and diagonal Gamma estimates for both payoffs.
 *
 * Deliberately self-contained. Nothing here includes the PDE assembly, the
 * boundary closure, the grid interpolation, the Krylov code, or the ADI
 * solvers, so a reference value cannot inherit a discretization error from the
 * method it is used to judge. The numerical stack is the standard library
 * alone: this file carries its own normal quantile, its own bivariate and
 * trivariate normal CDFs, its own Sobol generator and scramble, and its own
 * Student t quantile.
 *
 * The one shared header is config.hpp, which holds the model parameters and no
 * numerics. Sharing it is deliberate: the reference and the PDE harness must
 * price the same contract, and a second parameter struct is a second place for
 * them to disagree.
 *
 * @author Kevin Knights
 */
#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.hpp"

namespace financial_reference {

// Model parameters

/// Full correlation matrix, row-major, built from the three off-diagonals.
using Matrix3 = std::array<std::array<double, 3>, 3>;

/**
 * @brief Assemble the 3 x 3 correlation matrix from [rho_01, rho_02, rho_12].
 */
[[nodiscard]] inline Matrix3 correlation_matrix(
    const std::array<double, 3>& rho_off)
{
    return Matrix3{{
        {1.0,        rho_off[0], rho_off[1]},
        {rho_off[0], 1.0,        rho_off[2]},
        {rho_off[1], rho_off[2], 1.0       },
    }};
}

/**
 * @brief Lower Cholesky factor of a correlation matrix.
 *
 * Symmetry, unit diagonal, and positive definiteness are checked before the
 * factorization rather than inferred from its success, so an inadmissible
 * correlation is rejected with a description instead of producing a NaN price.
 *
 * @throws std::invalid_argument if the matrix is not a valid correlation matrix.
 */
[[nodiscard]] inline Matrix3 correlation_cholesky(const Matrix3& correlation)
{
    constexpr double symmetry_tolerance = 1.0e-14;
    for (int i = 0; i < 3; ++i) {
        if (std::abs(correlation[static_cast<std::size_t>(i)]
                                [static_cast<std::size_t>(i)] - 1.0)
            > symmetry_tolerance)
            throw std::invalid_argument(
                "correlation_cholesky: diagonal is not unity");
        for (int j = 0; j < i; ++j) {
            const double upper = correlation[static_cast<std::size_t>(j)]
                                            [static_cast<std::size_t>(i)];
            const double lower = correlation[static_cast<std::size_t>(i)]
                                            [static_cast<std::size_t>(j)];
            if (std::abs(upper - lower) > symmetry_tolerance)
                throw std::invalid_argument(
                    "correlation_cholesky: matrix is not symmetric");
            if (!(std::abs(lower) < 1.0))
                throw std::invalid_argument(
                    "correlation_cholesky: off-diagonal is not inside (-1, 1)");
        }
    }

    Matrix3 factor{};
    for (int i = 0; i < 3; ++i) {
        const auto row = static_cast<std::size_t>(i);
        for (int j = 0; j <= i; ++j) {
            const auto col = static_cast<std::size_t>(j);
            double sum = correlation[row][col];
            for (int k = 0; k < j; ++k)
                sum -= factor[row][static_cast<std::size_t>(k)]
                     * factor[col][static_cast<std::size_t>(k)];
            if (i == j) {
                if (!(sum > 0.0))
                    throw std::invalid_argument(
                        "correlation_cholesky: matrix is not positive definite");
                factor[row][col] = std::sqrt(sum);
            } else {
                factor[row][col] = sum / factor[col][col];
            }
        }
    }
    return factor;
}

// Univariate normal

/// Standard normal cumulative distribution function.
[[nodiscard]] inline double normal_cdf(double x)
{
    return 0.5 * std::erfc(-x * std::numbers::sqrt2 / 2.0);
}

/// Standard normal probability density function.
[[nodiscard]] inline double normal_pdf(double x)
{
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * std::numbers::pi);
}

/**
 * @brief Standard normal quantile, accurate to full double precision.
 *
 * Acklam's rational approximation followed by one Halley correction against
 * erfc. The correction is what makes the round trip normal_cdf(quantile(p)) = p
 * hold to roundoff, which is checked as a unit test rather than assumed.
 *
 * The argument must lie strictly inside (0, 1). Sobol points on the closed
 * boundary are moved to the nearest representable interior value by the caller.
 */
[[nodiscard]] inline double normal_quantile(double p)
{
    if (!(p > 0.0) || !(p < 1.0))
        throw std::invalid_argument(
            "normal_quantile: argument must lie inside the open unit interval");

    constexpr double a[6] = {
        -3.969683028665376e+01,  2.209460984245205e+02,
        -2.759285104469687e+02,  1.383577518672690e+02,
        -3.066479806614716e+01,  2.506628277459239e+00};
    constexpr double b[5] = {
        -5.447609879822406e+01,  1.615858368580409e+02,
        -1.556989798598866e+02,  6.680131188771972e+01,
        -1.328068155288572e+01};
    constexpr double c[6] = {
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00};
    constexpr double d[4] = {
         7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00};
    constexpr double low = 0.02425;

    double x = 0.0;
    if (p < low) {
        const double q = std::sqrt(-2.0 * std::log(p));
        x = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
          / ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0);
    } else if (p <= 1.0 - low) {
        const double q = p - 0.5;
        const double r = q * q;
        x = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
          / (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0);
    } else {
        const double q = std::sqrt(-2.0 * std::log(1.0 - p));
        x = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
          / ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0);
    }

    // Halley step on normal_cdf(x) - p = 0. In the extreme tail the density
    // underflows and the correction carries no information, so it is rejected
    // rather than applied: the starting approximation is already good to about
    // 1.2e-9 relative, and a Sobol point that far out contributes a payoff of
    // essentially zero.
    const double density = normal_pdf(x);
    if (density > 1.0e-300) {
        const double step = (normal_cdf(x) - p) / density;
        const double correction = step / (1.0 + 0.5 * x * step);
        if (std::isfinite(correction) && std::abs(correction) < 0.01 * (std::abs(x) + 1.0))
            x -= correction;
    }
    return x;
}

// Quadrature

/**
 * @brief Composite Gauss-Legendre settings for the multivariate normal CDFs.
 *
 * Every reported reference value carries the settings that produced it, and
 * acceptance requires that tightening them leaves the reported digits alone.
 */
struct QuadratureSettings {
    int outer_panels = 24;   ///< panels for the conditioning integral of Phi_3
    int outer_nodes = 24;    ///< Gauss-Legendre nodes per outer panel
    int inner_panels = 16;   ///< panels for the Genz integral of Phi_2
    int inner_nodes = 24;    ///< Gauss-Legendre nodes per inner panel
    double tail_cut = 8.75;  ///< outer integration is truncated at this |z|

    /// Every count doubled: the tightened settings used by the stability gate.
    [[nodiscard]] QuadratureSettings tightened() const
    {
        QuadratureSettings tighter = *this;
        tighter.outer_panels *= 2;
        tighter.outer_nodes *= 2;
        tighter.inner_panels *= 2;
        tighter.inner_nodes *= 2;
        tighter.tail_cut += 0.75;
        return tighter;
    }

    /// Upper bound on the normal mass discarded by the outer truncation.
    [[nodiscard]] double truncation_bound() const
    {
        return 2.0 * normal_cdf(-tail_cut);
    }

    [[nodiscard]] std::string describe() const
    {
        return "outer_panels=" + std::to_string(outer_panels)
             + " outer_nodes=" + std::to_string(outer_nodes)
             + " inner_panels=" + std::to_string(inner_panels)
             + " inner_nodes=" + std::to_string(inner_nodes)
             + " tail_cut=" + std::to_string(tail_cut);
    }
};

/// Gauss-Legendre nodes and weights on [-1, 1].
struct GaussLegendreRule {
    std::vector<double> node;
    std::vector<double> weight;
};

/**
 * @brief Build a Gauss-Legendre rule of the requested order by Newton iteration.
 *
 * Generated rather than tabulated so the order is a free parameter and the
 * tightening gate can double it.
 */
[[nodiscard]] inline GaussLegendreRule gauss_legendre_rule(int order)
{
    if (order < 1)
        throw std::invalid_argument("gauss_legendre_rule: order must be positive");

    GaussLegendreRule rule;
    const auto count = static_cast<std::size_t>(order);
    rule.node.assign(count, 0.0);
    rule.weight.assign(count, 0.0);

    const double n = static_cast<double>(order);
    for (int i = 0; i < order; ++i) {
        double x = std::cos(std::numbers::pi
                            * (static_cast<double>(i) + 0.75) / (n + 0.5));
        double derivative = 1.0;
        for (int sweep = 0; sweep < 100; ++sweep) {
            double current = 1.0;
            double previous = 0.0;
            for (int k = 0; k < order; ++k) {
                const double older = previous;
                previous = current;
                current = ((2.0 * static_cast<double>(k) + 1.0) * x * previous
                           - static_cast<double>(k) * older)
                        / static_cast<double>(k + 1);
            }
            derivative = n * (x * current - previous) / (x * x - 1.0);
            const double step = current / derivative;
            x -= step;
            if (std::abs(step) < 1.0e-16) break;
        }
        rule.node[static_cast<std::size_t>(i)] = x;
        rule.weight[static_cast<std::size_t>(i)] =
            2.0 / ((1.0 - x * x) * derivative * derivative);
    }
    return rule;
}

/**
 * @brief Bivariate and trivariate standard normal CDFs by composite quadrature.
 *
 * Deterministic throughout: there is no random state, so a reported price
 * depends only on the settings that are recorded beside it. Phi_2 uses the Genz
 * form of Plackett's identity, whose integrand stays bounded as the correlation
 * approaches unity. Phi_3 conditions on the first coordinate through the
 * Cholesky factor and integrates a Phi_2 against the normal density.
 */
class MultivariateNormal {
public:
    explicit MultivariateNormal(const QuadratureSettings& settings)
        : settings_(settings),
          outer_(gauss_legendre_rule(settings.outer_nodes)),
          inner_(gauss_legendre_rule(settings.inner_nodes))
    {
        if (settings.outer_panels < 1 || settings.inner_panels < 1)
            throw std::invalid_argument(
                "MultivariateNormal: panel counts must be positive");
        if (!(settings.tail_cut > 0.0))
            throw std::invalid_argument(
                "MultivariateNormal: tail_cut must be positive");
    }

    [[nodiscard]] const QuadratureSettings& settings() const { return settings_; }

    /**
     * @brief P(X <= a, Y <= b) for a standard bivariate normal of correlation rho.
     *
     * Infinite thresholds are accepted and reduce to the marginal cases.
     */
    [[nodiscard]] double bivariate(double a, double b, double rho) const
    {
        if (!(std::abs(rho) < 1.0))
            throw std::invalid_argument(
                "MultivariateNormal::bivariate: |rho| must be below one");
        if (a == -std::numeric_limits<double>::infinity()
            || b == -std::numeric_limits<double>::infinity())
            return 0.0;
        if (a == std::numeric_limits<double>::infinity())
            return normal_cdf(b);
        if (b == std::numeric_limits<double>::infinity())
            return normal_cdf(a);
        if (rho == 0.0)
            return normal_cdf(a) * normal_cdf(b);

        // Phi_2(a, b; rho) = Phi(a) Phi(b)
        //   + (1 / 2 pi) int_0^{asin(rho)} exp(-(a^2 - 2 a b sin t + b^2)
        //                                       / (2 cos^2 t)) dt
        const double limit = std::asin(rho);
        const double panel_width = limit / static_cast<double>(settings_.inner_panels);
        double integral = 0.0;
        for (int panel = 0; panel < settings_.inner_panels; ++panel) {
            const double lo = static_cast<double>(panel) * panel_width;
            const double half = 0.5 * panel_width;
            const double center = lo + half;
            for (std::size_t k = 0; k < inner_.node.size(); ++k) {
                const double t = center + half * inner_.node[k];
                const double sine = std::sin(t);
                const double cosine_squared = 1.0 - sine * sine;
                if (!(cosine_squared > 0.0)) continue;
                const double exponent =
                    -(a * a - 2.0 * a * b * sine + b * b) / (2.0 * cosine_squared);
                integral += inner_.weight[k] * half * std::exp(exponent);
            }
        }
        const double value = normal_cdf(a) * normal_cdf(b)
                           + integral / (2.0 * std::numbers::pi);
        return std::min(std::max(value, 0.0), 1.0);
    }

    /**
     * @brief P(X_1 <= h_1, X_2 <= h_2, X_3 <= h_3) for a standard trivariate normal.
     *
     * @param threshold Upper limits, any of which may be positive infinity.
     * @param correlation Correlation matrix, checked and factorized here.
     */
    [[nodiscard]] double trivariate(const std::array<double, 3>& threshold,
                                    const Matrix3& correlation) const
    {
        for (const double limit : threshold)
            if (limit == -std::numeric_limits<double>::infinity()) return 0.0;

        const Matrix3 factor = correlation_cholesky(correlation);
        const double l21 = factor[1][0];
        const double l22 = factor[1][1];
        const double l31 = factor[2][0];
        const double l32 = factor[2][1];
        const double l33 = factor[2][2];
        const double third_scale = std::hypot(l32, l33);
        if (!(l22 > 0.0) || !(third_scale > 0.0))
            throw std::invalid_argument(
                "MultivariateNormal::trivariate: degenerate correlation");
        const double inner_rho = l32 / third_scale;

        const double lo = -settings_.tail_cut;
        const double hi = std::min(threshold[0], settings_.tail_cut);
        if (!(hi > lo)) return 0.0;

        const double panel_width =
            (hi - lo) / static_cast<double>(settings_.outer_panels);
        double integral = 0.0;
        for (int panel = 0; panel < settings_.outer_panels; ++panel) {
            const double half = 0.5 * panel_width;
            const double center = lo + (static_cast<double>(panel) + 0.5) * panel_width;
            for (std::size_t k = 0; k < outer_.node.size(); ++k) {
                const double z = center + half * outer_.node[k];
                const double second = (threshold[1] - l21 * z) / l22;
                const double third = (threshold[2] - l31 * z) / third_scale;
                integral += outer_.weight[k] * half * normal_pdf(z)
                          * bivariate(second, third, inner_rho);
            }
        }
        return std::min(std::max(integral, 0.0), 1.0);
    }

private:
    QuadratureSettings settings_;
    GaussLegendreRule outer_;
    GaussLegendreRule inner_;
};

// Johnson analytical Rainbow reference

/// One evaluation of the Johnson formula, with the evidence for its accuracy.
struct JohnsonResult {
    double price = 0.0;
    /// Discounted expectation of the minimum asset: the upper bound on the price.
    double discounted_minimum_expectation = 0.0;
    /// Upper bound on the normal mass discarded by the outer truncation.
    double truncation_bound = 0.0;
};

/**
 * @brief European call on the minimum of three lognormal assets, Johnson (1987).
 *
 * Derived rather than reconstructed around a published number. With
 * S_i(T) = S_i(0) exp((r - sigma_i^2 / 2) T + sigma_i sqrt(T) W_i) and W ~ N(0, R),
 *
 *   C = sum_j S_j(0) Phi_3(d_j1, {d_ij}_{i != j}; R^(j))
 *       - K exp(-rT) Phi_3(d_12, d_22, d_32; R),
 *
 * where d_j2 = (log(S_j(0)/K) + (r - sigma_j^2 / 2) T) / (sigma_j sqrt(T)),
 * d_j1 = d_j2 + sigma_j sqrt(T), and, writing
 * sigma_ij^2 = sigma_i^2 + sigma_j^2 - 2 rho_ij sigma_i sigma_j,
 *
 *   d_ij = (log(S_i(0)/S_j(0)) - sigma_ij^2 T / 2) / (sigma_ij sqrt(T)).
 *
 * The j-th correlation matrix is that of the vector (-Z_j, {-U_ij}), the
 * standardized constraints "asset j finishes above the strike" and "asset j
 * finishes below asset i", under the measure in which asset j is the numeraire.
 *
 * A strike of zero is admissible and reduces the formula to the discounted
 * expectation of the minimum asset, which is what supplies the upper bound.
 */
[[nodiscard]] inline JohnsonResult johnson_call_on_minimum(
    const Config& model, const MultivariateNormal& normal)
{
    constexpr double infinity = std::numeric_limits<double>::infinity();
    const double expiry = model.t_final;
    const double rate = model.risk_free_rate;
    if (!(expiry > 0.0))
        throw std::invalid_argument(
            "johnson_call_on_minimum: expiry must be positive");
    for (const double volatility : model.sigma)
        if (!(volatility > 0.0))
            throw std::invalid_argument(
                "johnson_call_on_minimum: every volatility must be positive");
    if (model.strike_price < 0.0)
        throw std::invalid_argument(
            "johnson_call_on_minimum: strike must not be negative");

    const Matrix3 correlation = correlation_matrix(model.rho_off);
    const double root_expiry = std::sqrt(expiry);

    // Pairwise spread volatilities and the exchange-option thresholds.
    Matrix3 spread{};
    Matrix3 exchange{};
    for (int i = 0; i < 3; ++i) {
        const auto row = static_cast<std::size_t>(i);
        for (int j = 0; j < 3; ++j) {
            if (i == j) continue;
            const auto col = static_cast<std::size_t>(j);
            const double variance =
                model.sigma[row] * model.sigma[row]
                + model.sigma[col] * model.sigma[col]
                - 2.0 * correlation[row][col] * model.sigma[row] * model.sigma[col];
            if (!(variance > 0.0))
                throw std::invalid_argument(
                    "johnson_call_on_minimum: a pairwise spread volatility vanishes");
            spread[row][col] = std::sqrt(variance);
            exchange[row][col] =
                (std::log(model.initial_prices[row] / model.initial_prices[col])
                 - 0.5 * variance * expiry)
                / (spread[row][col] * root_expiry);
        }
    }

    const auto strike_threshold = [&](int asset) {
        const auto index = static_cast<std::size_t>(asset);
        if (model.strike_price == 0.0) return infinity;
        return (std::log(model.initial_prices[index] / model.strike_price)
                + (rate - 0.5 * model.sigma[index] * model.sigma[index]) * expiry)
             / (model.sigma[index] * root_expiry);
    };

    // Correlation of the constraint vector under the measure in which asset
    // "owner" is the numeraire. Position 0 is the strike constraint and
    // positions 1 and 2 are the two "owner finishes below asset i" constraints.
    const auto owner_correlation = [&](int owner, const std::array<int, 2>& others) {
        const auto own = static_cast<std::size_t>(owner);
        Matrix3 block = correlation_matrix({0.0, 0.0, 0.0});
        for (int slot = 0; slot < 2; ++slot) {
            const auto other = static_cast<std::size_t>(others[static_cast<std::size_t>(slot)]);
            const double value =
                (correlation[other][own] * model.sigma[other] - model.sigma[own])
                / spread[other][own];
            block[0][static_cast<std::size_t>(slot + 1)] = value;
            block[static_cast<std::size_t>(slot + 1)][0] = value;
        }
        const auto first = static_cast<std::size_t>(others[0]);
        const auto second = static_cast<std::size_t>(others[1]);
        const double covariance =
            model.sigma[first] * model.sigma[second] * correlation[first][second]
            - model.sigma[first] * model.sigma[own] * correlation[first][own]
            - model.sigma[second] * model.sigma[own] * correlation[second][own]
            + model.sigma[own] * model.sigma[own];
        const double value =
            covariance / (spread[first][own] * spread[second][own]);
        block[1][2] = value;
        block[2][1] = value;
        return block;
    };

    JohnsonResult result;
    result.truncation_bound = normal.settings().truncation_bound();

    double asset_terms = 0.0;
    double unstruck_terms = 0.0;
    for (int owner = 0; owner < 3; ++owner) {
        const auto own = static_cast<std::size_t>(owner);
        std::array<int, 2> others{};
        int slot = 0;
        for (int other = 0; other < 3; ++other)
            if (other != owner) others[static_cast<std::size_t>(slot++)] = other;

        const Matrix3 block = owner_correlation(owner, others);
        const double in_the_money =
            strike_threshold(owner) + model.sigma[own] * root_expiry;
        const std::array<double, 3> threshold{
            in_the_money,
            exchange[static_cast<std::size_t>(others[0])][own],
            exchange[static_cast<std::size_t>(others[1])][own]};
        const std::array<double, 3> unstruck{
            infinity, threshold[1], threshold[2]};

        asset_terms += model.initial_prices[own] * normal.trivariate(threshold, block);
        unstruck_terms += model.initial_prices[own] * normal.trivariate(unstruck, block);
    }

    const std::array<double, 3> strike_thresholds{
        strike_threshold(0), strike_threshold(1), strike_threshold(2)};
    const double strike_term = model.strike_price * std::exp(-rate * expiry)
                             * normal.trivariate(strike_thresholds, correlation);

    result.price = asset_terms - strike_term;
    result.discounted_minimum_expectation = unstruck_terms;
    return result;
}

/// Black-Scholes European call, used only by the limiting-case gates.
[[nodiscard]] inline double black_scholes_call(
    double spot, double strike, double rate, double volatility, double expiry)
{
    if (!(expiry > 0.0) || !(volatility > 0.0))
        return std::max(spot - strike, 0.0);
    const double d1 = (std::log(spot / strike)
                       + (rate + 0.5 * volatility * volatility) * expiry)
                    / (volatility * std::sqrt(expiry));
    const double d2 = d1 - volatility * std::sqrt(expiry);
    return spot * normal_cdf(d1)
         - strike * std::exp(-rate * expiry) * normal_cdf(d2);
}

// Scrambled Sobol points

/// SplitMix64, used only to derive the scramble matrices and digital shifts.
[[nodiscard]] inline std::uint64_t splitmix64(std::uint64_t& state)
{
    state += 0x9E3779B97F4A7C15ull;
    std::uint64_t value = state;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

/**
 * @brief Three-dimensional Sobol sequence with a Matousek random linear scramble.
 *
 * Direction numbers come from the primitive polynomials of the first three
 * Sobol dimensions: the identity for dimension one, x + 1 for dimension two,
 * and x^2 + x + 1 for dimension three, with the standard initial values.
 *
 * The randomization is a random lower-triangular unit-diagonal binary matrix
 * applied to each dimension's generator columns, followed by a uniform digital
 * shift. Both are bijections of the dyadic digits, so every scramble is a
 * measure-preserving transformation of the unit cube and each scramble's mean
 * is an unbiased estimate of the integral. That is what makes the across-
 * scramble Student t interval legitimate; the individual points are not
 * independent samples and are never treated as such.
 *
 * Points are drawn only in powers of two, starting from index zero, because a
 * truncated or concatenated Sobol run loses the balance properties the whole
 * construction rests on.
 */
class ScrambledSobol {
public:
    static constexpr int kDimension = 3;
    static constexpr int kBits = 32;

    /**
     * @param seed Base seed. The same seed reproduces the same points exactly.
     * @param scramble Index of this randomization within the batch.
     */
    ScrambledSobol(std::uint64_t seed, int scramble)
    {
        build_direction_numbers();
        apply_random_linear_scramble(seed, scramble);
        reset();
    }

    /// Return to the start of the sequence without rebuilding the scramble.
    void reset()
    {
        state_.fill(0u);
        index_ = 0;
    }

    /**
     * @brief Draw the next point of the sequence into @p point.
     *
     * Coordinates that land on the closed boundary are moved to the nearest
     * representable value inside the open unit interval, so the normal quantile
     * is never handed a zero or a one.
     */
    void next(std::array<double, kDimension>& point)
    {
        if (index_ > 0) {
            const auto column = static_cast<std::size_t>(
                std::countr_one(static_cast<std::uint32_t>(index_ - 1)));
            for (std::size_t d = 0; d < kDimension; ++d)
                state_[d] ^= direction_[d][column];
        }
        ++index_;

        constexpr double scale = 1.0 / 4294967296.0;  // 2^-32
        for (std::size_t d = 0; d < kDimension; ++d) {
            const double raw = static_cast<double>(state_[d] ^ shift_[d]) * scale;
            point[d] = raw <= 0.0 ? std::nextafter(0.0, 1.0)
                     : raw >= 1.0 ? std::nextafter(1.0, 0.0)
                                  : raw;
        }
    }

private:
    void build_direction_numbers()
    {
        // Initial values and polynomial coefficients per dimension. Dimension
        // zero is the van der Corput sequence and has no recurrence.
        constexpr int degree[kDimension] = {0, 1, 2};
        constexpr std::uint32_t coefficient[kDimension] = {0u, 0u, 1u};
        constexpr std::uint32_t initial[kDimension][2] = {{1u, 0u}, {1u, 0u}, {1u, 3u}};

        for (std::size_t d = 0; d < kDimension; ++d) {
            std::array<std::uint32_t, kBits> odd{};
            if (degree[d] == 0) {
                odd.fill(1u);
            } else {
                for (int j = 0; j < degree[d]; ++j)
                    odd[static_cast<std::size_t>(j)] = initial[d][static_cast<std::size_t>(j)];
                for (int j = degree[d]; j < kBits; ++j) {
                    const auto slot = static_cast<std::size_t>(j);
                    const auto back = static_cast<std::size_t>(j - degree[d]);
                    std::uint32_t value = odd[back] ^ (odd[back] << degree[d]);
                    for (int k = 1; k < degree[d]; ++k)
                        if (((coefficient[d] >> (degree[d] - 1 - k)) & 1u) != 0u)
                            value ^= odd[static_cast<std::size_t>(j - k)]
                                   << static_cast<std::uint32_t>(k);
                    odd[slot] = value;
                }
            }
            for (int j = 0; j < kBits; ++j)
                direction_[d][static_cast<std::size_t>(j)] =
                    odd[static_cast<std::size_t>(j)]
                    << static_cast<std::uint32_t>(kBits - 1 - j);
        }
    }

    void apply_random_linear_scramble(std::uint64_t seed, int scramble)
    {
        for (std::size_t d = 0; d < kDimension; ++d) {
            std::uint64_t state = seed
                ^ (static_cast<std::uint64_t>(scramble) * 0x9E3779B97F4A7C15ull)
                ^ (static_cast<std::uint64_t>(d + 1) * 0xC2B2AE3D27D4EB4Full);
            (void)splitmix64(state);

            // Row i has a unit diagonal and uniform random bits strictly below it.
            std::array<std::uint32_t, kBits> row{};
            for (int i = 0; i < kBits; ++i) {
                const std::uint32_t bits =
                    static_cast<std::uint32_t>(splitmix64(state) >> 32);
                const std::uint32_t strict_lower =
                    i == 0 ? 0u
                           : static_cast<std::uint32_t>(
                                 0xFFFFFFFFu << static_cast<std::uint32_t>(kBits - i));
                row[static_cast<std::size_t>(i)] =
                    (bits & strict_lower)
                    | (1u << static_cast<std::uint32_t>(kBits - 1 - i));
            }
            for (std::size_t j = 0; j < kBits; ++j) {
                const std::uint32_t column = direction_[d][j];
                std::uint32_t scrambled = 0u;
                for (int i = 0; i < kBits; ++i) {
                    const int parity = std::popcount(
                        row[static_cast<std::size_t>(i)] & column) & 1;
                    scrambled |= static_cast<std::uint32_t>(parity)
                               << static_cast<std::uint32_t>(kBits - 1 - i);
                }
                direction_[d][j] = scrambled;
            }
            shift_[d] = static_cast<std::uint32_t>(splitmix64(state) >> 32);
        }
    }

    std::array<std::array<std::uint32_t, kBits>, kDimension> direction_{};
    std::array<std::uint32_t, kDimension> shift_{};
    std::array<std::uint32_t, kDimension> state_{};
    std::uint64_t index_ = 0;
};

// Randomized quasi-Monte Carlo estimator

/// Default randomized-QMC seed: the author's MSc student identifier.
inline constexpr std::uint64_t kDefaultQmcSeed = 25'378'095ull;
static_assert(kDefaultQmcSeed < std::numeric_limits<std::uint64_t>::max(),
              "the companion seed used by the reference must not overflow");

/// Delta and diagonal Gamma at one relative spot bump, from one scramble.
struct BumpEstimate {
    double relative_bump = 0.0;
    std::array<double, 3> delta{};
    std::array<double, 3> gamma{};
};

/// Everything one scramble contributes for one payoff.
struct ScrambleEstimate {
    int scramble = 0;
    double price = 0.0;
    std::vector<BumpEstimate> bump;
};

/// What to draw. The sample size is always an exact power of two.
struct QmcRequest {
    int log2_points = 20;
    int scrambles = 16;
    std::uint64_t seed = kDefaultQmcSeed;
    std::vector<double> relative_bumps = {0.005, 0.01, 0.02};
};

/// Per-scramble estimates for both payoffs, drawn from the same points.
struct QmcResult {
    std::int64_t points = 0;
    std::vector<ScrambleEstimate> basket;
    std::vector<ScrambleEstimate> rainbow;
};

/// Neumaier compensated accumulator, so a long sum does not eat the Gamma digits.
class CompensatedSum {
public:
    void add(double value)
    {
        const double next = sum_ + value;
        compensation_ += std::abs(sum_) >= std::abs(value)
            ? (sum_ - next) + value
            : (value - next) + sum_;
        sum_ = next;
    }

    [[nodiscard]] double value() const { return sum_ + compensation_; }

private:
    double sum_ = 0.0;
    double compensation_ = 0.0;
};

/**
 * @brief Run the randomized QMC reference for both payoffs.
 *
 * The two payoffs share the point set. That is a cost decision, not a
 * statistical one: each payoff's interval is formed from its own scramble
 * estimates, and neither is used to judge the other.
 *
 * Delta and Gamma reuse the center's Sobol points, scramble, normal draws, and
 * lognormal factors. Because the terminal value is proportional to the initial
 * spot, a bumped path is the center path rescaled, so the common-point
 * construction is exact rather than approximate. The derivative is formed
 * inside each scramble before any averaging, which is what preserves the
 * covariance that makes the estimates usable.
 */
[[nodiscard]] inline QmcResult run_qmc(const Config& model, const QmcRequest& request)
{
    if (request.log2_points < 0 || request.log2_points > 30)
        throw std::invalid_argument("run_qmc: log2_points must lie in [0, 30]");
    if (request.scrambles < 2)
        throw std::invalid_argument("run_qmc: at least two scrambles are required");
    for (const double bump : request.relative_bumps)
        if (!(bump > 0.0) || !(bump < 1.0))
            throw std::invalid_argument(
                "run_qmc: every relative bump must lie inside (0, 1)");

    const Matrix3 factor = correlation_cholesky(correlation_matrix(model.rho_off));
    const std::int64_t points = std::int64_t{1} << request.log2_points;
    const double discount = std::exp(-model.risk_free_rate * model.t_final);
    const double root_expiry = std::sqrt(model.t_final);
    const std::size_t bumps = request.relative_bumps.size();

    std::array<double, 3> drift{};
    for (std::size_t d = 0; d < 3; ++d)
        drift[d] = (model.risk_free_rate - 0.5 * model.sigma[d] * model.sigma[d])
                 * model.t_final;

    QmcResult result;
    result.points = points;
    result.basket.reserve(static_cast<std::size_t>(request.scrambles));
    result.rainbow.reserve(static_cast<std::size_t>(request.scrambles));

    for (int scramble = 0; scramble < request.scrambles; ++scramble) {
        ScrambledSobol sobol(request.seed, scramble);

        CompensatedSum basket_center;
        CompensatedSum rainbow_center;
        // Indexed [bump][asset][side], side 0 is the up bump.
        std::vector<std::array<std::array<CompensatedSum, 2>, 3>> basket_bumped(bumps);
        std::vector<std::array<std::array<CompensatedSum, 2>, 3>> rainbow_bumped(bumps);

        std::array<double, 3> point{};
        std::array<double, 3> normal{};
        std::array<double, 3> growth{};
        std::array<double, 3> terminal{};

        for (std::int64_t draw = 0; draw < points; ++draw) {
            sobol.next(point);
            for (std::size_t d = 0; d < 3; ++d)
                normal[d] = normal_quantile(point[d]);

            double basket_sum = 0.0;
            double rainbow_min = std::numeric_limits<double>::infinity();
            for (std::size_t d = 0; d < 3; ++d) {
                double correlated = 0.0;
                for (std::size_t k = 0; k <= d; ++k)
                    correlated += factor[d][k] * normal[k];
                growth[d] = std::exp(drift[d] + model.sigma[d] * root_expiry * correlated);
                terminal[d] = model.initial_prices[d] * growth[d];
                basket_sum += model.weight[d] * terminal[d];
                rainbow_min = std::min(rainbow_min, terminal[d]);
            }
            basket_center.add(std::max(basket_sum - model.strike_price, 0.0));
            rainbow_center.add(std::max(rainbow_min - model.strike_price, 0.0));

            for (std::size_t b = 0; b < bumps; ++b) {
                for (std::size_t d = 0; d < 3; ++d) {
                    const double step = request.relative_bumps[b] * model.initial_prices[d];
                    for (std::size_t side = 0; side < 2; ++side) {
                        const double signed_step = side == 0 ? step : -step;
                        const double moved = terminal[d] + signed_step * growth[d];
                        basket_bumped[b][d][side].add(std::max(
                            basket_sum + model.weight[d] * signed_step * growth[d]
                            - model.strike_price, 0.0));
                        double minimum = moved;
                        for (std::size_t other = 0; other < 3; ++other)
                            if (other != d) minimum = std::min(minimum, terminal[other]);
                        rainbow_bumped[b][d][side].add(std::max(
                            minimum - model.strike_price, 0.0));
                    }
                }
            }
        }

        const double inverse = discount / static_cast<double>(points);
        const auto finish = [&](const CompensatedSum& center,
                                const std::vector<std::array<std::array<CompensatedSum, 2>, 3>>& bumped) {
            ScrambleEstimate estimate;
            estimate.scramble = scramble;
            estimate.price = center.value() * inverse;
            estimate.bump.reserve(bumps);
            for (std::size_t b = 0; b < bumps; ++b) {
                BumpEstimate entry;
                entry.relative_bump = request.relative_bumps[b];
                for (std::size_t d = 0; d < 3; ++d) {
                    const double step = request.relative_bumps[b] * model.initial_prices[d];
                    const double up = bumped[b][d][0].value() * inverse;
                    const double down = bumped[b][d][1].value() * inverse;
                    entry.delta[d] = (up - down) / (2.0 * step);
                    entry.gamma[d] = (up - 2.0 * estimate.price + down) / (step * step);
                }
                estimate.bump.push_back(entry);
            }
            return estimate;
        };

        result.basket.push_back(finish(basket_center, basket_bumped));
        result.rainbow.push_back(finish(rainbow_center, rainbow_bumped));
    }
    return result;
}

// Uncertainty

/// Regularized incomplete beta function, by the Lentz continued fraction.
[[nodiscard]] inline double regularized_incomplete_beta(double a, double b, double x)
{
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;

    // The continued fraction converges quickly only on the near side of the
    // mode. A strict comparison is what stops the reflection from bouncing
    // back on the boundary case.
    if (x > (a + 1.0) / (a + b + 2.0))
        return 1.0 - regularized_incomplete_beta(b, a, 1.0 - x);

    const double front = std::exp(
        std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b)
        + a * std::log(x) + b * std::log1p(-x));

    constexpr double tiny = 1.0e-300;
    double c = 1.0;
    double d = 1.0 - (a + b) * x / (a + 1.0);
    if (std::abs(d) < tiny) d = tiny;
    d = 1.0 / d;
    double fraction = d;

    for (int m = 1; m <= 300; ++m) {
        const double index = static_cast<double>(m);
        const double even = index * (b - index) * x
                          / ((a + 2.0 * index - 1.0) * (a + 2.0 * index));
        const double odd = -(a + index) * (a + b + index) * x
                         / ((a + 2.0 * index) * (a + 2.0 * index + 1.0));
        for (const double numerator : {even, odd}) {
            d = 1.0 + numerator * d;
            if (std::abs(d) < tiny) d = tiny;
            d = 1.0 / d;
            c = 1.0 + numerator / c;
            if (std::abs(c) < tiny) c = tiny;
            const double delta = c * d;
            fraction *= delta;
            if (std::abs(delta - 1.0) < 1.0e-15) return front * fraction / a;
        }
    }
    return front * fraction / a;
}

/// P(|T| <= t) for a Student t variate with @p dof degrees of freedom.
[[nodiscard]] inline double student_t_two_sided_cdf(double t, int dof)
{
    const double freedom = static_cast<double>(dof);
    return 1.0 - regularized_incomplete_beta(
        0.5 * freedom, 0.5, freedom / (freedom + t * t));
}

/**
 * @brief Two-sided Student t quantile, found by bisection on the CDF.
 *
 * Bisection rather than a table so the number of scrambles stays a free
 * parameter, and so the value can be checked against published quantiles.
 */
[[nodiscard]] inline double student_t_two_sided_quantile(double confidence, int dof)
{
    if (dof < 1)
        throw std::invalid_argument(
            "student_t_two_sided_quantile: at least one degree of freedom is required");
    if (!(confidence > 0.0) || !(confidence < 1.0))
        throw std::invalid_argument(
            "student_t_two_sided_quantile: confidence must lie inside (0, 1)");

    double low = 0.0;
    double high = 1.0e4;
    for (int sweep = 0; sweep < 200; ++sweep) {
        const double middle = 0.5 * (low + high);
        if (student_t_two_sided_cdf(middle, dof) < confidence) low = middle;
        else high = middle;
    }
    return 0.5 * (low + high);
}

/// A center and its two-sided confidence interval across scramble estimates.
struct Interval {
    double mean = 0.0;
    double standard_error = 0.0;
    double half_width = 0.0;
    int samples = 0;
};

/**
 * @brief Mean and two-sided Student t interval across independent scrambles.
 *
 * The population is the scramble estimates, never the individual Sobol points.
 */
[[nodiscard]] inline Interval scramble_interval(
    const std::vector<double>& estimates, double confidence = 0.95)
{
    Interval interval;
    interval.samples = static_cast<int>(estimates.size());
    if (interval.samples < 2)
        throw std::invalid_argument(
            "scramble_interval: at least two scramble estimates are required");

    CompensatedSum total;
    for (const double value : estimates) total.add(value);
    interval.mean = total.value() / static_cast<double>(interval.samples);

    CompensatedSum squared;
    for (const double value : estimates) {
        const double deviation = value - interval.mean;
        squared.add(deviation * deviation);
    }
    const double variance = squared.value() / static_cast<double>(interval.samples - 1);
    interval.standard_error =
        std::sqrt(std::max(variance, 0.0) / static_cast<double>(interval.samples));
    interval.half_width = student_t_two_sided_quantile(confidence, interval.samples - 1)
                        * interval.standard_error;
    return interval;
}

} // namespace financial_reference
