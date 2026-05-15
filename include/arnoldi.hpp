//
// Implementation of Arnoldi's Algorithm
// Created by Kevin Knights on 2/26/26.
//
#pragma once

#include <print>
#include "../eigen/Eigen/Dense"


using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * @brief Arnoldi's procedure is an algorithm for building an orthogonal
 *        basis of the Krylov subspace \mathcal{K}_m(A, v_0)
 *        and the corresponding upper Hessenberg matrix.
 * @param A         Square input matrix (n x n)
 * @param v0        Initial vector (n x 1), must be non-zero
 * @param m         Dimension of subspace \mathcal{K}
 * @param tolerance double precision comparison tolerance
 * @return tuple of matrix V, H.
 *          Where V is n x m matrix whose column-vectors form a basis of \mathcal{K},
 *          and H is a (m + 1) x m Hessenberg matrix.
 */
[[nodiscard]]
std::tuple<MatrixXd, MatrixXd> arnoldi(
    const MatrixXd& A,
    const VectorXd& v0,
    long m,
    double tolerance = std::numeric_limits<double>::epsilon()
    );