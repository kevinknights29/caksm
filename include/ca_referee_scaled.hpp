/**
 * @file ca_referee_scaled.hpp
 * @brief Baseline power-of-two scaling for the Basket augmented referee.
 *
 * The referee prices independently of the production stencil by applying the Al-Mohy and Higham
 * Taylor action to the augmented operator A_tilde = [[A, B], [0, K]], whose three-row tail
 * carries the Basket boundary forcing. That augmentation is what makes the fixed method
 * impractical on a large grid: its forcing columns dominate the augmented 1-norm, by a factor
 * of about 14,553 at n=31, and since the substep count follows that norm, n=31 needed 305,449
 * substeps and still missed its restored-tail gate. Scaling the forcing block by an exact power
 * of two is an exact similarity, D = diag(I_N, eta I_3), so it moves the representation only:
 * the spectrum, the Taylor constants, and the original-coordinate norms the stopping test
 * compares are all unchanged. The same n=31 case then takes 21 substeps and restores its tail
 * to machine precision, which is why scaled augmentation was promoted to the Basket baseline
 * on 2026-07-29. Rainbow has no forcing tail to scale and keeps the fixed compatibility method,
 * which also remains the Basket reproduction control.
 *
 * Al-Mohy & Higham (2011), "Computing the action of the matrix exponential, with an application
 * to exponential integrators", SIAM J. Sci. Comput. 33(2):488-511, doi:10.1137/100788860.
 *
 * @author Kevin Knights
 * @date 2026-07-28
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pde_operators.hpp"

namespace ca_referee {

inline constexpr int kTaylorMax = 55;
inline constexpr double kThetaRef = 9.9;
inline constexpr double kToleranceRef = 1.1e-16;
inline constexpr double kRestoredTailTolerance = 1.0e-12;

/**
 * One method-independent write gate for every referee implementation.
 *
 * Keeping this predicate host-testable prevents the fixed-compatibility and
 * scaled-augmentation paths from drifting apart.
 */
[[nodiscard]] inline bool referee_output_admissible(
    bool operator_passed, bool result_finite, int unconverged_substeps,
    double restored_tail_error, bool verification_requested,
    bool verification_passed) noexcept
{
    return operator_passed
        && result_finite
        && unconverged_substeps == 0
        && std::isfinite(restored_tail_error)
        && restored_tail_error >= 0.0
        && restored_tail_error <= kRestoredTailTolerance
        && (!verification_requested || verification_passed);
}

struct AugmentationScaling {
    int exponent = 0;
    double eta = 1.0;
    double eta_inverse = 1.0;
    double forcing_norm_1 = 0.0;
    double scaled_forcing_norm_1 = 0.0;
};

struct ScaledActionStats {
    std::vector<int> degrees;
    int64_t spmv_count = 0;
    int unconverged_substeps = 0;
    double max_weighted_previous = 0.0;
    double max_weighted_current = 0.0;
    double max_weighted_sum = 0.0;
};

struct ScaledAugmentationResult {
    VecXd physical;
    Eigen::Vector3d restored_tail = Eigen::Vector3d::Zero();
    AugmentationScaling scaling;
    ScaledActionStats stats;
    double original_norm_1 = 0.0;
    double scaled_norm_1 = 0.0;
    int original_scaling_steps = 0;
    int scaled_scaling_steps = 0;
};

/**
 * Compute ||B||_1 with a fixed scalar accumulation order.
 *
 * This norm alone chooses eta, and eta is part of the frozen method identity, so the order is
 * pinned rather than left to a vectorizer: a reassociated sum could land on the far side of a
 * power of two and select a different, equally valid scaling that no longer reproduces the
 * recorded checksums.
 */
[[nodiscard]] inline double forcing_1norm(const MatXd& forcing)
{
    double norm = 0.0;
    for (Eigen::Index column = 0; column < forcing.cols(); ++column) {
        double column_sum = 0.0;
        for (Eigen::Index row = 0; row < forcing.rows(); ++row)
            column_sum += std::abs(forcing(row, column));
        norm = std::max(norm, column_sum);
    }
    return norm;
}

[[nodiscard]] inline double augmented_forcing_1norm(
    const MatXd& forcing, double scale = 1.0)
{
    double norm = 0.0;
    for (Eigen::Index column = 0; column < forcing.cols(); ++column) {
        double column_sum = column == 0 ? 0.0 : 1.0;
        for (Eigen::Index row = 0; row < forcing.rows(); ++row)
            column_sum += scale * std::abs(forcing(row, column));
        norm = std::max(norm, column_sum);
    }
    return norm;
}

/**
 * Choose eta = 2^-ceil(log2(||B||_1)) without materializing a rounded
 * logarithm.  frexp returns norm = fraction * 2^exponent with
 * fraction in [0.5, 1), so exact powers of two are the one special case.
 *
 * A power of two is the point: ldexp scales and restores by an exponent change alone, so the
 * similarity introduces no rounding of its own and the restored tail can be compared against a
 * gate at machine precision. The exponent is clamped at zero so a forcing block already at or
 * below unit norm is left untouched rather than amplified.
 */
[[nodiscard]] inline AugmentationScaling make_augmentation_scaling(
    const MatXd& forcing)
{
    AugmentationScaling scaling;
    scaling.forcing_norm_1 = forcing_1norm(forcing);
    if (scaling.forcing_norm_1 == 0.0)
        return scaling;
    if (!std::isfinite(scaling.forcing_norm_1))
        throw std::runtime_error("non-finite Basket forcing 1-norm");

    int binary_exponent = 0;
    const double fraction =
        std::frexp(scaling.forcing_norm_1, &binary_exponent);
    scaling.exponent =
        fraction == 0.5 ? binary_exponent - 1 : binary_exponent;
    if (scaling.exponent < 0)
        scaling.exponent = 0;
    if (scaling.exponent > std::numeric_limits<double>::max_exponent - 1)
        throw std::overflow_error("Basket augmentation scale exceeds FP64 range");

    scaling.eta = std::ldexp(1.0, -scaling.exponent);
    scaling.eta_inverse = std::ldexp(1.0, scaling.exponent);
    scaling.scaled_forcing_norm_1 =
        scaling.eta * scaling.forcing_norm_1;
    return scaling;
}

/**
 * Build D^-1 A_tilde D = [[A, eta B], [0, K]] for
 * D = diag(I_N, eta I_3).
 *
 * A similarity, so the spectrum and exp(t A_tilde) are unchanged up to the same transform; only
 * the 1-norm moves, and with it the substep count. K is untouched because D is a multiple of the
 * identity on the tail, which is why the nilpotent structure that propagates the boundary
 * polynomial survives the scaling exactly.
 */
[[nodiscard]] inline SpMat build_scaled_A_tilde(
    const SpMat& physical, const MatXd& forcing, int physical_rows,
    double eta)
{
    if (!(eta > 0.0) || !std::isfinite(eta))
        throw std::invalid_argument(
            "augmentation scale must be positive and finite");
    MatXd scaled_forcing = forcing;
    scaled_forcing *= eta;
    return build_A_tilde(physical, scaled_forcing, physical_rows);
}

/**
 * Infinity norm after mapping a transformed vector back through D.
 *
 * The stopping test has to see the vector the fixed method would have seen, so the scaled tail
 * is weighted by eta before it is compared. Measuring in transformed coordinates would let the
 * similarity move the stopping decision, and the two methods would no longer be the same
 * algorithm on two representations.
 */
[[nodiscard]] inline double weighted_infinity_norm(
    const VecXd& transformed, int physical_rows, double eta)
{
    if (transformed.size() != physical_rows + 3)
        throw std::invalid_argument(
            "weighted norm expects an N+3 augmented vector");
    const double physical_norm =
        transformed.head(physical_rows).lpNorm<Eigen::Infinity>();
    const double tail_norm =
        transformed.tail(3).lpNorm<Eigen::Infinity>();
    return std::max(physical_norm, eta * tail_norm);
}

/**
 * Substep count s = ceil(t ||A||_1 / theta_ref) for the Taylor action.
 *
 * The Al-Mohy & Higham (2011) scaling parameter held at a fixed theta rather than selected per
 * degree: each substep advances t/s, which keeps the argument norm below theta_ref where the
 * truncated series converges inside kTaylorMax terms. Evaluated on both the original and the
 * scaled 1-norm, so the work the similarity avoids is recorded rather than asserted.
 */
[[nodiscard]] inline int select_scaling_steps(double time, double norm_1)
{
    if (!(time >= 0.0) || !std::isfinite(time)
        || !(norm_1 >= 0.0) || !std::isfinite(norm_1))
        throw std::invalid_argument(
            "matrix-exponential time and norm must be finite and nonnegative");
    const double selected = std::ceil(time * norm_1 / kThetaRef);
    if (selected > static_cast<double>(std::numeric_limits<int>::max()))
        throw std::overflow_error("matrix-exponential scaling count exceeds int");
    return std::max(1, static_cast<int>(selected));
}

/**
 * @brief Reference CPU implementation of the scaled-augmentation action.
 *
 * The Taylor recurrence and strict stopping inequality are intentionally the
 * fixed referee's rules. Only the exactly similar representation and the
 * corresponding original-coordinate norm evaluation differ.
 *
 * The production generator is the GPU referee. This is the independent
 * implementation the unit tests check the algebra against on a dense control,
 * so a change to the scaling cannot pass unnoticed.
 */
[[nodiscard]] inline ScaledAugmentationResult
compute_scaled_augmentation_referee(
    const PDESystem& system, double time)
{
    if (!system.has_forcing || system.B.cols() != 3)
        throw std::invalid_argument(
            "scaled-augmentation is defined only for Basket forcing");
    if (system.A.rows() != system.N || system.A.cols() != system.N
        || system.B.rows() != system.N || system.u0.size() != system.N)
        throw std::invalid_argument("inconsistent Basket PDE dimensions");

    ScaledAugmentationResult result;
    result.scaling = make_augmentation_scaling(system.B);

    const SpMat scaled_operator = build_scaled_A_tilde(
        system.A, system.B, system.N, result.scaling.eta);
    result.scaled_norm_1 = sparse_1norm(scaled_operator);
    result.scaled_scaling_steps =
        select_scaling_steps(time, result.scaled_norm_1);

    // The original augmented 1-norm is the maximum of the physical columns
    // and the three forcing/K columns.  The latter add one K entry to
    // columns one and two.
    result.original_norm_1 =
        std::max(
            sparse_1norm(system.A),
            augmented_forcing_1norm(system.B));
    result.original_scaling_steps =
        select_scaling_steps(time, result.original_norm_1);

    VecXd state(system.N + 3);
    state.head(system.N) = system.u0;
    state.tail(3) =
        result.scaling.eta_inverse * make_s_vec(0.0);

    const double action_scale =
        time / static_cast<double>(result.scaled_scaling_steps);
    result.stats.degrees.reserve(
        static_cast<std::size_t>(result.scaled_scaling_steps));
    for (int substep = 0;
         substep < result.scaled_scaling_steps; ++substep) {
        VecXd previous = state;
        VecXd sum = state;
        int degree = kTaylorMax;
        bool converged = false;
        for (int term = 1; term <= kTaylorMax; ++term) {
            const VecXd current =
                (action_scale / static_cast<double>(term))
                * (scaled_operator * previous);
            sum += current;

            const double previous_norm = weighted_infinity_norm(
                previous, system.N, result.scaling.eta);
            const double current_norm = weighted_infinity_norm(
                current, system.N, result.scaling.eta);
            const double sum_norm = weighted_infinity_norm(
                sum, system.N, result.scaling.eta);
            if (!std::isfinite(previous_norm)
                || !std::isfinite(current_norm)
                || !std::isfinite(sum_norm)
                || !current.allFinite() || !sum.allFinite())
                throw std::runtime_error(
                    "non-finite scaled-augmentation recurrence");

            result.stats.max_weighted_previous = std::max(
                result.stats.max_weighted_previous, previous_norm);
            result.stats.max_weighted_current = std::max(
                result.stats.max_weighted_current, current_norm);
            result.stats.max_weighted_sum = std::max(
                result.stats.max_weighted_sum, sum_norm);
            ++result.stats.spmv_count;

            if (sum_norm > 0.0
                && previous_norm + current_norm
                    < kToleranceRef * sum_norm) {
                degree = term;
                converged = true;
                break;
            }
            previous = current;
        }
        if (!converged)
            ++result.stats.unconverged_substeps;
        result.stats.degrees.push_back(degree);
        state = std::move(sum);
    }

    result.physical = state.head(system.N);
    result.restored_tail =
        result.scaling.eta * state.tail(3);
    return result;
}

} // namespace ca_referee
