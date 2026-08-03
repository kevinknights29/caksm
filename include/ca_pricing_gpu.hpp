/**
 * @file ca_pricing_gpu.hpp
 * @brief Host construction of the matrix-free GPU pricing operator.
 *
 * @author Kevin Knights
 * @date 2026-07-26
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "gpu_pde_matrix_powers.cuh"
#include "pde_operators.hpp"

struct CaPricingModel {
    double strike = 100.0;
    double rate = 0.04;
    double expiry = 1.0;
    std::array<double, 3> sigma{0.30, 0.35, 0.40};
    std::array<double, 3> rho{0.50, 0.50, 0.50};
    std::array<double, 3> weight{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
    std::array<double, 3> spot{100.0, 100.0, 100.0};
    double alpha = 2.85;
};

/**
 * Build only the host data consumed by the matrix-free GPU integrators.
 *
 * The general PDESystem builder also assembles A, all ADI splits, and dense
 * boundary matrices. None of those objects is used by the production
 * matrix-free path, and assembling them would make a device-memory study
 * measure host sparse-matrix construction instead.
 */
[[nodiscard]] inline PDESystem build_gpu_pde_system(
    int n, const CaPricingModel& model, bool rainbow)
{
    const int64_t rows =
        static_cast<int64_t>(n) * static_cast<int64_t>(n)
        * static_cast<int64_t>(n);
    if (rows > std::numeric_limits<int>::max())
        throw std::overflow_error("GPU PDE grid exceeds PDESystem::N");

    PDESystem system;
    system.grid = build_grid(
        n, model.spot, model.sigma, model.alpha, model.expiry);
    system.N = static_cast<int>(rows);
    system.has_forcing = !rainbow;
    system.u0.resize(system.N);

    const Grid& grid = system.grid;
    for (int i3 = 0; i3 < n; ++i3) {
        const double s2 = std::exp(grid.x[2][i3]);
        for (int i2 = 0; i2 < n; ++i2) {
            const double s1 = std::exp(grid.x[1][i2]);
            for (int i1 = 0; i1 < n; ++i1) {
                const double s0 = std::exp(grid.x[0][i1]);
                const int gid = i3 * n * n + i2 * n + i1;
                system.u0[gid] = rainbow
                    ? std::max(std::min({s0, s1, s2}) - model.strike, 0.0)
                    : std::max(
                        model.weight[0] * s0
                        + model.weight[1] * s1
                        + model.weight[2] * s2
                        - model.strike,
                        0.0);
            }
        }
    }
    return system;
}

struct GershgorinInterval {
    double lower = 0.0;
    double upper = 0.0;
};

/**
 * @brief Real enclosure of the spectrum of scale * A, from Gershgorin discs.
 *
 * The Newton and Chebyshev bases need a center and a half-width of an interval containing the
 * spectrum, and Gershgorin supplies one from diagonal and row sums alone, with no eigensolve on
 * an operator this size. The scale is the same factor the matrix-powers kernel applies, so the
 * enclosure describes the operator actually being iterated; the min/max pairing keeps the
 * interval ordered if that scale is negative. include_zero widens the interval to the origin,
 * which is where the nilpotent boundary-forcing tail sits, so one interval covers the whole
 * augmented vector the recurrence runs on.
 */
[[nodiscard]] inline GershgorinInterval scaled_gershgorin_interval(
    const SpMat& A, double scale, bool include_zero)
{
    std::vector<double> diagonal(static_cast<std::size_t>(A.rows()), 0.0);
    std::vector<double> radius(static_cast<std::size_t>(A.rows()), 0.0);
    for (int outer = 0; outer < A.outerSize(); ++outer) {
        for (SpMat::InnerIterator it(A, outer); it; ++it) {
            if (it.row() == it.col())
                diagonal[static_cast<std::size_t>(it.row())] = it.value();
            else
                radius[static_cast<std::size_t>(it.row())] += std::abs(it.value());
        }
    }

    double lower = std::numeric_limits<double>::infinity();
    double upper = -std::numeric_limits<double>::infinity();
    for (Eigen::Index row = 0; row < A.rows(); ++row) {
        const std::size_t q = static_cast<std::size_t>(row);
        lower = std::min(lower, diagonal[q] - radius[q]);
        upper = std::max(upper, diagonal[q] + radius[q]);
    }
    const double scaled_lower = std::min(scale * lower, scale * upper);
    const double scaled_upper = std::max(scale * lower, scale * upper);
    return {
        include_zero ? std::min(0.0, scaled_lower) : scaled_lower,
        include_zero ? std::max(0.0, scaled_upper) : scaled_upper
    };
}

/**
 * @brief Real Leja points of the enclosing interval, in Leja order.
 *
 * The Newton basis prod_k (A - shift_k I) v is conditioned by where the shifts sit and in which
 * order they are consumed. Leja points are grown greedily, each one placed where the product of
 * distances to the points already chosen is largest. This keeps the nodal polynomial near its
 * minimax size over the interval and the basis far better conditioned than the monomial one at
 * the same width. Candidates are Chebyshev nodes of the interval, and the greedy score is
 * accumulated as a sum of logarithms so a long product cannot underflow. Following the real-Leja
 * construction used by LeXInt (Deka, Moriggl & Einkemmer, "LeXInt: GPU-accelerated exponential
 * integrators package", 2023, arXiv:2310.08344).
 */
[[nodiscard]] inline std::vector<double> real_leja_shifts(
    GershgorinInterval interval, int count)
{
    if (count < 0) throw std::invalid_argument("real_leja_shifts: count must be nonnegative");
    std::vector<double> shifts;
    shifts.reserve(static_cast<std::size_t>(count));
    if (count == 0) return shifts;

    const double center = 0.5 * (interval.lower + interval.upper);
    const double half_width = 0.5 * (interval.upper - interval.lower);
    if (!(half_width > 0.0)) {
        shifts.assign(static_cast<std::size_t>(count), center);
        return shifts;
    }

    constexpr int candidates = 256;
    const double pi = std::acos(-1.0);
    std::array<double, candidates + 1> nodes{};
    for (int q = 0; q <= candidates; ++q)
        nodes[static_cast<std::size_t>(q)] =
            center + half_width * std::cos(pi * static_cast<double>(q) / candidates);

    shifts.push_back(interval.upper);
    while (static_cast<int>(shifts.size()) < count) {
        double best_node = center;
        double best_score = -std::numeric_limits<double>::infinity();
        for (double node : nodes) {
            double score = 0.0;
            bool duplicate = false;
            for (double selected : shifts) {
                const double distance = std::abs(node - selected);
                if (distance <= 1.0e-14 * half_width) {
                    duplicate = true;
                    break;
                }
                score += std::log(distance);
            }
            if (!duplicate && score > best_score) {
                best_score = score;
                best_node = node;
            }
        }
        shifts.push_back(best_node);
    }
    return shifts;
}

/**
 * @brief Collapse the model and the grid into the stencil coefficients the kernel reads.
 *
 * The matrix-free kernel never sees the PDE. It sees one reaction term, three drift and three
 * diffusion coefficients, and three mixed-derivative coefficients, each with the log-grid
 * spacing already divided in. Folding the spacings in here keeps the divisions out of the inner
 * loop and leaves GpuPdeOperator small enough to pass by value as a kernel argument.
 * This is what lets the stencil run without touching an assembled matrix at all.
 */
[[nodiscard]] inline GpuPdeOperator make_gpu_pde_operator(
    const PDESystem& sys, const CaPricingModel& model, bool rainbow)
{
    GpuPdeOperator op;
    op.n = sys.grid.n;
    op.N = sys.N;
    op.rainbow = rainbow ? 1 : 0;
    op.reaction = -model.rate;
    for (int d = 0; d < 3; ++d) {
        const double dx = sys.grid.dx[d];
        op.drift[d] =
            (model.rate - 0.5 * model.sigma[d] * model.sigma[d]) / (2.0 * dx);
        op.diffusion[d] =
            0.5 * model.sigma[d] * model.sigma[d] / (dx * dx);
    }
    op.mixed[0] =
        model.rho[0] * model.sigma[0] * model.sigma[1]
        / (4.0 * sys.grid.dx[0] * sys.grid.dx[1]);
    op.mixed[1] =
        model.rho[1] * model.sigma[0] * model.sigma[2]
        / (4.0 * sys.grid.dx[0] * sys.grid.dx[2]);
    op.mixed[2] =
        model.rho[2] * model.sigma[1] * model.sigma[2]
        / (4.0 * sys.grid.dx[1] * sys.grid.dx[2]);
    return op;
}

/**
 * @brief The Basket far-boundary forcing, as nine face planes.
 *
 * At the upper face of each axis the Dirichlet value is known in closed form, so the neighbor
 * the stencil is missing contributes a known term rather than an unknown. That term is a
 * quadratic in time, so it factors into three coefficient planes per axis multiplying the
 * 3-vector tail that the nilpotent K block propagates inside the exponential. Layout is
 * [direction][coefficient][n*n] to match mpk_face_index, and the values mirror the assembled
 * PDESystem::B_adi columns so the matrix-free and assembled operators agree. Rainbow imposes a
 * linearity condition instead and never reads this buffer.
 */
[[nodiscard]] inline std::vector<double> make_gpu_face_b(
    const PDESystem& sys, const CaPricingModel& model)
{
    const int n = sys.grid.n;
    const int n2 = n * n;
    std::vector<double> b(static_cast<std::size_t>(9 * n2), 0.0);

    for (int dir = 0; dir < 3; ++dir) {
        const double dx = sys.grid.dx[dir];
        const double fwd =
            0.5 * model.sigma[dir] * model.sigma[dir] / (dx * dx)
            + (model.rate - 0.5 * model.sigma[dir] * model.sigma[dir]) / (2.0 * dx);
        const double x_virtual = sys.grid.x[dir][n - 1] + dx;

        int free_axis[2];
        int next = 0;
        for (int d = 0; d < 3; ++d)
            if (d != dir) free_axis[next++] = d;

        for (int u = 0; u < n; ++u) {
            for (int v = 0; v < n; ++v) {
                int ijk[3] = {0, 0, 0};
                ijk[dir] = n - 1;
                ijk[free_axis[0]] = u;
                ijk[free_axis[1]] = v;
                double coord[3] = {
                    sys.grid.x[0][ijk[0]],
                    sys.grid.x[1][ijk[1]],
                    sys.grid.x[2][ijk[2]]
                };
                coord[dir] = x_virtual;
                const double payoff =
                    model.weight[0] * std::exp(coord[0])
                    + model.weight[1] * std::exp(coord[1])
                    + model.weight[2] * std::exp(coord[2]) - model.strike;
                const int f = u * n + v;
                b[static_cast<std::size_t>((dir * 3 + 2) * n2 + f)] = fwd * payoff;
                b[static_cast<std::size_t>((dir * 3 + 1) * n2 + f)] =
                    fwd * model.strike * model.rate;
                b[static_cast<std::size_t>((dir * 3 + 0) * n2 + f)] =
                    -fwd * model.strike * model.rate * model.rate;
            }
        }
    }
    return b;
}
