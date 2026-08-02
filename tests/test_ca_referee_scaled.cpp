/**
 * @file test_ca_referee_scaled.cpp
 * @brief Algebraic gates for the scaled-augmentation Basket baseline.
 */

#include <bit>
#include <cmath>
#include <limits>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <unsupported/Eigen/MatrixFunctions>

#include "ca_referee_scaled.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

[[nodiscard]] Eigen::MatrixXd dense_augmented(
    const Eigen::MatrixXd& physical,
    const Eigen::MatrixXd& forcing)
{
    const Eigen::Index rows = physical.rows();
    Eigen::MatrixXd augmented =
        Eigen::MatrixXd::Zero(rows + 3, rows + 3);
    augmented.topLeftCorner(rows, rows) = physical;
    augmented.topRightCorner(rows, 3) = forcing;
    augmented(rows, rows + 1) = 1.0;
    augmented(rows + 1, rows + 2) = 1.0;
    return augmented;
}

[[nodiscard]] PDESystem small_system()
{
    PDESystem system;
    system.N = 5;
    system.has_forcing = true;

    Eigen::MatrixXd physical = Eigen::MatrixXd::Zero(5, 5);
    physical.diagonal() << -1.1, -0.8, -1.3, -0.7, -1.0;
    physical(0, 1) = 0.2;
    physical(1, 2) = -0.1;
    physical(2, 3) = 0.15;
    physical(3, 4) = 0.08;
    system.A = physical.sparseView();

    system.B.resize(5, 3);
    system.B <<
        100.0, -20.0, 11.0,
        -45.0, 80.0, 7.0,
        30.0, 5.0, -13.0,
        15.0, -9.0, 6.0,
        -4.0, 3.0, 2.0;
    system.u0 =
        (Eigen::VectorXd(5) << 2.0, 1.5, 0.5, 3.0, 1.0).finished();
    return system;
}

} // namespace

/// eta must be an exact power of two, and must be the tightest one that brings
/// the forcing norm into (0.5, 1].
///
/// Expected: eta_inverse has a single set bit, so scaling and restoring are
/// exponent changes with no rounding of their own; eta * eta_inverse is exactly
/// 1; and the scaled norm lands in the half-open interval, which is what makes
/// this the tightest admissible choice rather than merely a safe one.
TEST_CASE(
    "power-of-two augmentation scaling brackets the forcing norm",
    "[referee][scaled-augmentation][algebra]")
{
    const PDESystem system = small_system();
    const auto scaling =
        ca_referee::make_augmentation_scaling(system.B);

    REQUIRE(std::has_single_bit(
        static_cast<unsigned long long>(scaling.eta_inverse)));
    REQUIRE_THAT(
        scaling.eta * scaling.eta_inverse,
        WithinAbs(1.0, 0.0));
    REQUIRE(scaling.scaled_forcing_norm_1 <= 1.0);
    REQUIRE(scaling.scaled_forcing_norm_1 > 0.5);
}

/// The stopping test must see the vector the unscaled method would have seen.
///
/// Expected: weighting the transformed tail by eta reproduces the plain
/// infinity norm of the same vector in original coordinates, exactly. If it did
/// not, the similarity would move the stopping decision and the two methods
/// would stop being the same algorithm on two representations.
TEST_CASE(
    "weighted infinity norm equals the original-coordinate norm",
    "[referee][scaled-augmentation][norm]")
{
    const PDESystem system = small_system();
    const auto scaling =
        ca_referee::make_augmentation_scaling(system.B);
    Eigen::VectorXd transformed(system.N + 3);
    transformed <<
        -3.0, 2.0, 4.0, -1.0, 0.5,
        7.0 * scaling.eta_inverse,
        -2.0 * scaling.eta_inverse,
        5.0 * scaling.eta_inverse;

    Eigen::VectorXd original = transformed;
    original.tail(3) *= scaling.eta;
    REQUIRE_THAT(
        ca_referee::weighted_infinity_norm(
            transformed, system.N, scaling.eta),
        WithinAbs(
            original.lpNorm<Eigen::Infinity>(), 0.0));
}

/// The scaling must be a similarity, so the exponential action it produces is
/// the same one up to the transform. Checked against a dense Eigen exp() rather
/// than against another Taylor implementation, so the reference shares no code
/// with the thing under test.
///
/// Expected: for eta at the selected value and at a factor of two either side,
/// scaling the forcing, transforming the start, exponentiating and restoring
/// reproduces the untransformed answer to 2e-13 in both the physical field and
/// the tail. Sweeping three eta values is the point: agreement at only the
/// selected one could be a coincidence of that particular scale.
TEST_CASE(
    "similarity scaling preserves the dense augmented exponential",
    "[referee][scaled-augmentation][dense-exp][critical]")
{
    const PDESystem system = small_system();
    const Eigen::MatrixXd physical = system.A;
    const auto selected =
        ca_referee::make_augmentation_scaling(system.B);

    Eigen::VectorXd original_start(system.N + 3);
    original_start.head(system.N) = system.u0;
    original_start.tail(3) = make_s_vec(0.0);
    const Eigen::VectorXd expected =
        dense_augmented(physical, system.B).exp() * original_start;

    for (const double eta :
         {0.5 * selected.eta, selected.eta, 2.0 * selected.eta}) {
        Eigen::VectorXd transformed_start = original_start;
        transformed_start.tail(3) /= eta;
        Eigen::VectorXd measured =
            dense_augmented(physical, eta * system.B).exp()
            * transformed_start;
        measured.tail(3) *= eta;

        REQUIRE(
            (measured.head(system.N) - expected.head(system.N))
                .lpNorm<Eigen::Infinity>()
            < 2.0e-13);
        REQUIRE(
            (measured.tail(3) - expected.tail(3))
                .lpNorm<Eigen::Infinity>()
            < 2.0e-13);
    }
}

/// The production CPU recurrence, end to end, against a dense control.
///
/// Expected: every substep converges inside the degree cap; the field matches
/// the dense exponential to 2e-12 and the restored tail to 2e-13. The tail is
/// additionally checked against its closed form at t=1, [t^2/2, t, 1] =
/// [0.5, 1, 1], because the nilpotent block propagates it exactly and a wrong
/// scaling would show up there first.
TEST_CASE(
    "CPU scaled recurrence passes a dense small-system control",
    "[referee][scaled-augmentation][recurrence][critical]")
{
    const PDESystem system = small_system();
    const Eigen::MatrixXd physical = system.A;
    Eigen::VectorXd original_start(system.N + 3);
    original_start.head(system.N) = system.u0;
    original_start.tail(3) = make_s_vec(0.0);
    const Eigen::VectorXd expected =
        dense_augmented(physical, system.B).exp() * original_start;

    const auto measured =
        ca_referee::compute_scaled_augmentation_referee(system, 1.0);

    REQUIRE(measured.stats.unconverged_substeps == 0);
    REQUIRE(
        (measured.physical - expected.head(system.N))
            .lpNorm<Eigen::Infinity>()
        < 2.0e-12);
    REQUIRE(
        (measured.restored_tail - expected.tail(3))
            .lpNorm<Eigen::Infinity>()
        < 2.0e-13);
    REQUIRE_THAT(
        measured.restored_tail[0], WithinAbs(0.5, 2.0e-13));
    REQUIRE_THAT(
        measured.restored_tail[1], WithinAbs(1.0, 2.0e-13));
    REQUIRE_THAT(
        measured.restored_tail[2], WithinAbs(1.0, 2.0e-13));
}

/// The shared write gate must reject on every individual reason, not just in
/// combination, so a canonical file cannot be written past a single failure.
///
/// Expected: an all-clear run is admissible; each of an unconverged substep, a
/// failed operator check, a tail error above tolerance, a requested-but-failed
/// verification, and a non-finite result is independently sufficient to refuse.
/// A NaN, an infinity, and a negative tail error are all refused rather than
/// compared, since each would otherwise pass a naive upper-bound test.
TEST_CASE(
    "referee output gate fails closed for every method",
    "[referee][write-gate][critical]")
{
    REQUIRE(ca_referee::referee_output_admissible(
        true, true, 0, 0.0, false, false));
    REQUIRE_FALSE(ca_referee::referee_output_admissible(
        true, true, 1, 0.0, false, false));
    REQUIRE_FALSE(ca_referee::referee_output_admissible(
        false, true, 0, 0.0, false, false));
    REQUIRE_FALSE(ca_referee::referee_output_admissible(
        true, true, 0,
        2.0 * ca_referee::kRestoredTailTolerance, false, false));
    REQUIRE_FALSE(ca_referee::referee_output_admissible(
        true, true, 0, 0.0, true, false));
    REQUIRE(ca_referee::referee_output_admissible(
        true, true, 0, 0.0, true, true));
    REQUIRE_FALSE(ca_referee::referee_output_admissible(
        true, false, 0, 0.0, false, false));
    REQUIRE_FALSE(ca_referee::referee_output_admissible(
        true, true, 0, std::numeric_limits<double>::quiet_NaN(),
        false, false));
    REQUIRE_FALSE(ca_referee::referee_output_admissible(
        true, true, 0, std::numeric_limits<double>::infinity(),
        false, false));
    REQUIRE_FALSE(ca_referee::referee_output_admissible(
        true, true, 0, -1.0, false, false));
}
