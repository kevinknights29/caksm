//
// Created by Kevin Knights on 3/13/26.
//
#include "../include/arnoldi.hpp"
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
    const long m,
    const double tolerance
    ) {
    // Checks
    if (A.rows() != A.cols())
        throw std::invalid_argument("Matrix A must be square.");
    if (v0.size() != A.rows())
        throw std::invalid_argument("v0 size must match A dimensions.");
    if (m <= 0 || m > A.rows())
        throw std::invalid_argument("m must satisfy 0 < m <= n.");

    // Algorithm
    const long n = A.rows();
    MatrixXd V(n, m); // n x m matrix whose column-vectors form a basis of \mathcal{K}
    MatrixXd H = MatrixXd::Zero(m + 1, m); // (m + 1) x m Hessenberg matrix

    V.col(0) = v0.normalized();

    for (long j = 0; j < m; ++j) {
        VectorXd w_j = A * V.col(j); // column vector of basis \mathcal{L}
        for (long i = 0; i <= j; ++i) {
            H(i, j) = w_j.dot(V.col(i));
            w_j -= H(i, j) * V.col(i);
        }
        H(j + 1, j) =  w_j.norm();
        if (H(j + 1, j) <= tolerance) {
            std::println("[WARN] Arnoldi breakdown at j={}: h_{{j+1,j}} approx 0.0", j);
            break;
        }

        if (j + 1 < m) {
            V.col(j + 1) = w_j / H(j + 1, j);
        }
    }
    return {V, H};
}