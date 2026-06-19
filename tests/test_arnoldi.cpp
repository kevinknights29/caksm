//
// Created by Kevin Knights on 2/27/26.
//
// tests/test_arnoldi.cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../include/arnoldi.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// Helpers
// Checks that all columns of V are mutually orthonormal
static bool isOrthonormal(const MatrixXd& V, double tol = 1e-9) {
    const MatrixXd gram = V.transpose() * V;
    return (gram - MatrixXd::Identity(V.cols(), V.cols())).norm() < tol;
}

// Input Validation

TEST_CASE("arnoldi throws on non-square matrix", "[arnoldi][validation]") {
    MatrixXd A(3, 4);
    VectorXd v0 = VectorXd::Random(3);
    REQUIRE_THROWS_AS(arnoldi(A, v0, 2), std::invalid_argument);
}

TEST_CASE("arnoldi throws when v0 size mismatches A", "[arnoldi][validation]") {
    MatrixXd A = MatrixXd::Random(4, 4);
    VectorXd v0 = VectorXd::Random(3);  // wrong size
    REQUIRE_THROWS_AS(arnoldi(A, v0, 2), std::invalid_argument);
}

TEST_CASE("arnoldi throws when m is out of range", "[arnoldi][validation]") {
    MatrixXd A = MatrixXd::Random(4, 4);
    VectorXd v0 = VectorXd::Random(4);
    REQUIRE_THROWS_AS(arnoldi(A, v0, 0),  std::invalid_argument);
    REQUIRE_THROWS_AS(arnoldi(A, v0, 5),  std::invalid_argument); // m > n
    REQUIRE_THROWS_AS(arnoldi(A, v0, -1), std::invalid_argument);
}

// Output Shape

TEST_CASE("arnoldi returns correctly shaped matrices", "[arnoldi][shape]") {
    constexpr long n = 8, m = 4;
    MatrixXd A  = MatrixXd::Random(n, n);
    VectorXd v0 = VectorXd::Random(n);

    auto [V, H] = arnoldi(A, v0, m);

    REQUIRE(V.rows() == n);
    REQUIRE(V.cols() == m);
    REQUIRE(H.rows() == m + 1);
    REQUIRE(H.cols() == m);
}

// Mathematical Properties

TEST_CASE("V columns are orthonormal", "[arnoldi][orthonormality]") {
    // Run for several random matrices to increase confidence
    for (int trial = 0; trial < 5; ++trial) {
        constexpr long n = 20, m = 6;
        MatrixXd A  = MatrixXd::Random(n, n);
        VectorXd v0 = VectorXd::Random(n);

        auto [V, H] = arnoldi(A, v0, m);

        INFO("Trial " << trial << ": V^T * V should be identity");
        REQUIRE(isOrthonormal(V));
    }
}

TEST_CASE("Arnoldi relation: A*V = V*H_m + h_{m+1,m} * v_{m+1} * e_m^T",
          "[arnoldi][relation]")
{
    // The core invariant: A * V_m = V_{m+1} * H̃  (where H̃ is the (m+1 x m) matrix)
    constexpr long n = 15, m = 5;
    MatrixXd A  = MatrixXd::Random(n, n);
    VectorXd v0 = VectorXd::Random(n);

    auto [V, H] = arnoldi(A, v0, m);

    // Reconstruct V_{m+1} by appending the next Krylov vector
    // We verify: A * V_m ≈ V_m * H_m(top m rows) + h_{m+1,m}*v_{m+1}*e_m^T
    // Equivalently: || A*V - V*H.topRows(m) ||_F should be small
    // (only checks top-m rows of the Hessenberg relation)
    MatrixXd lhs = A * V;                           // n x m
    MatrixXd rhs = V * H.topRows(m);               // n x m
    double residual = (lhs - rhs).norm();

    // The residual is exactly h_{m+1,m} * ||v_{m+1}||, which isn't zero,
    // so instead we verify the full (m+1) form with an extended V
    // Build V_extended: [V | v_{m+1}]
    MatrixXd V_ext(n, m + 1);
    V_ext.leftCols(m) = V;

    // Recover v_{m+1} from the last column of the Hessenberg relation
    VectorXd w = A * V.col(m - 1);
    for (long i = 0; i < m; ++i)
        w -= H(i, m - 1) * V.col(i);
    if (H(m, m - 1) > 1e-12)
        V_ext.col(m) = w / H(m, m - 1);

    // Now: A * V = V_ext * H  should hold
    MatrixXd full_residual = A * V - V_ext * H;
    REQUIRE_THAT(full_residual.norm(), WithinAbs(0.0, 1e-9));
}

TEST_CASE("H is upper Hessenberg", "[arnoldi][hessenberg]") {
    constexpr long n = 10, m = 5;
    MatrixXd A  = MatrixXd::Random(n, n);
    VectorXd v0 = VectorXd::Random(n);

    auto [V, H] = arnoldi(A, v0, m);

    // All entries below the first subdiagonal must be zero
    for (long col = 0; col < m; ++col) {
        for (long row = col + 2; row < m + 1; ++row) {
            INFO("H(" << row << ", " << col << ") = " << H(row, col));
            REQUIRE_THAT(H(row, col), WithinAbs(0.0, 1e-12));
        }
    }
}

TEST_CASE("Subdiagonal entries of H are non-negative", "[arnoldi][hessenberg]") {
    constexpr long n = 10, m = 4;
    MatrixXd A  = MatrixXd::Random(n, n);
    VectorXd v0 = VectorXd::Random(n);

    auto [V, H] = arnoldi(A, v0, m);

    for (long j = 0; j < m; ++j) {
        INFO("H(" << j+1 << ", " << j << ") = " << H(j+1, j));
        REQUIRE(H(j + 1, j) >= 0.0);
    }
}

TEST_CASE("First column of V matches normalized v0", "[arnoldi][initialization]") {
    constexpr long n = 8, m = 3;
    MatrixXd A  = MatrixXd::Random(n, n);
    VectorXd v0 = VectorXd::Random(n);

    auto [V, H] = arnoldi(A, v0, m);

    REQUIRE_THAT((V.col(0) - v0.normalized()).norm(), WithinAbs(0.0, 1e-12));
}

// Edge Cases

TEST_CASE("Works with m = 1", "[arnoldi][edge]") {
    constexpr long n = 5;
    MatrixXd A  = MatrixXd::Random(n, n);
    VectorXd v0 = VectorXd::Random(n);

    auto [V, H] = arnoldi(A, v0, 1);

    REQUIRE(V.cols() == 1);
    REQUIRE_THAT(V.col(0).norm(), WithinRel(1.0, 1e-12));
}

TEST_CASE("Works with symmetric positive definite matrix", "[arnoldi][spd]") {
    constexpr long n = 10, m = 5;
    MatrixXd B  = MatrixXd::Random(n, n);
    MatrixXd A  = B.transpose() * B + MatrixXd::Identity(n, n); // SPD
    VectorXd v0 = VectorXd::Random(n);

    REQUIRE_NOTHROW([&]{ auto [V, H] = arnoldi(A, v0, m); }());
}