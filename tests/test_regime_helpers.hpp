/**
 * @file test_regime_helpers.hpp
 * @brief Shared fixtures for the regime-instrument test suite.
 *
 * These tests pin the invariants the study rests on: the scatter knob must be a similarity
 * transform, or there is no pure R_v axis; R_v's accounting frame must be consistent, or P
 * is double-counted onto a spurious 1/P^2 scaling; and mgs_reductions() must match the loop
 * in src/scaling.cpp, or R_h describes an algorithm nobody is running.
 *
 * The suite is split by subject across the test_regime_*.cpp files:
 *
 *   test_regime_coordinates.cpp  R_v/R_h accounting, reduction counts, the roofline
 *                                split, reachability, and the reduction cost model
 *   test_regime_synthetic.cpp    the synthetic operator's three independent knobs, and
 *                                basis conditioning predicted from the spectrum alone
 *   test_regime_krylov.cpp       the two orthogonalization arms, the Krylov-dimension
 *                                prediction, and the per-vector CholQR certificate
 *   test_regime_nonnormal.cpp    non-normality: the transferability bridge to the real
 *                                Black-Scholes operator
 *   test_regime_kernels.cpp      the tiled matrix-powers halo model (mpk.hpp) and the
 *                                cache-blocked kernel itself (akx.hpp)
 *
 * @author Kevin Knights
 * @date 2026-07-10
 */

#pragma once

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <complex>
#include <numbers>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "akx.hpp"
#include "ca_arnoldi.hpp"
#include "machine.hpp"
#include "mpk.hpp"
#include "regime.hpp"
#include "synthetic.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace regime_test {

/// A reproducible unit vector, so a failing case can be re-run exactly.
/// Deliberately not random: a conditioning result that only holds for some
/// draws is not a result, and a flaky test here would be indistinguishable
/// from a real regression.
inline Eigen::VectorXd deterministic_unit_vector(Eigen::Index n)
{
    Eigen::VectorXd v(n);
    for (Eigen::Index i = 0; i < n; ++i)
        v(i) = std::sin(0.7 * static_cast<double>(i) + 1.0);
    v.normalize();
    return v;
}

/// Move a vector into the scattered operator's ordering.
/// The scatter knob is a symmetric permutation P A P^T, so a comparison
/// against the banded arm must permute the start vector too; skipping this
/// would compare the two operators on different vectors.
inline Eigen::VectorXd apply_perm(const std::vector<int64_t>& perm, const Eigen::VectorXd& v)
{
    Eigen::VectorXd out(v.size());
    for (Eigen::Index i = 0; i < v.size(); ++i)
        out(static_cast<Eigen::Index>(perm[static_cast<std::size_t>(i)])) = v(i);
    return out;
}

/// Ascending eigenvalues from a dense symmetric solver, for checking the
/// analytic spectrum against something that shares no code with it.
inline Eigen::VectorXd sorted_spectrum(const SpMatS& A)
{
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Eigen::MatrixXd(A),
                                                      Eigen::EigenvaluesOnly);
    return es.eigenvalues();
}

}  // namespace regime_test
