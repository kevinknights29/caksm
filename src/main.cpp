//
// Created by Kevin Knights on 2/27/26.
//
#include <cmath>
#include <iostream>
#include <limits>
#include <print>
#include <ranges>
#include "../eigen/Eigen/Dense"
#include "../eigen/unsupported/Eigen/MatrixFunctions"
#include "../include/arnoldi.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;


template<>
struct std::formatter<MatrixXd> : std::formatter<std::string> {
    auto format(const MatrixXd& m, std::format_context& ctx) const {
        std::ostringstream oss;
        oss << m;
        return std::formatter<std::string>::format(oss.str(), ctx);
    }
};


int main()
{
    /// 1-D Heat Equation
    /// Problem Parameters
    constexpr long l {1};
    constexpr double alpha {0.1};
    constexpr double t_init {0.0};
    constexpr double t_final {1.0};
    // Initial Conditions
    // Gaussian pulse
    auto gaussian_pulse = [](const double x) {
        constexpr double center = 0.5;  // Center in [0,1] domain!
        constexpr double width = 0.1;
        return std::exp(-std::pow(x - center, 2) / (2 * width * width));
    };
    // Boundary Conditions
    // Insulated bar at both ends
    constexpr double left_boundary {0.0};
    constexpr double right_boundary {0.0};

    /// Spatial Parameters
    constexpr long spatial_steps {10};
    constexpr double delta_x {l / static_cast<double>(spatial_steps - 1)};
    const VectorXd x {VectorXd::LinSpaced(spatial_steps, double{0}, double{l})};

    /// Temporal Parameters
    constexpr long temporal_steps {10};
    constexpr double delta_t {t_final / static_cast<double>(temporal_steps)};
    const VectorXd t {VectorXd::LinSpaced(temporal_steps, t_init, t_final)};

    /// Stability Check
    constexpr double r {alpha * delta_t / (delta_x * delta_x)};
    std::println("Stability parameter r = alpha delta_t / (delta_x)^2 = {:.6f}", r);

    /// Constructing Matrix A
    // Number of interior points (excluding boundaries)
    constexpr long n_interior {spatial_steps - 2};
    MatrixXd A {MatrixXd::Zero(n_interior, n_interior)};
    // Triangular matrix produced from centered difference discretization
    for (auto i {0}; i < A.rows(); i++) {
        for (auto j {0}; j < A.cols(); j++) {
            if (i == j) {
                A(i, j) = -2.0;
            }
            if (j - i == 1) {
                A(i, j) = 1.0;
            }
            if (i - j == 1) {
                A(i, j) = 1.0;
            }
        }
    }
    A = alpha / (delta_x * delta_x) * A;

    /// Constructing b vector
    VectorXd b {VectorXd::Zero(n_interior)};
    b(0) = left_boundary;
    b(n_interior - 1) = right_boundary;

    /// Constructing Matrix U
    MatrixXd U {MatrixXd::Zero(temporal_steps + 1, n_interior)};
    U.row(0) = x.segment(1, n_interior).unaryExpr(gaussian_pulse).transpose();

    // Krylov Exponential Integrator
    double t_current = t_init;
    for (long time_step = 0; time_step < temporal_steps; ++time_step) {
        constexpr double tolerance = 1e-6;
        std::println("=== Time step {} (t = {:.4f}) ===", time_step, t_current);
        // Step 1: Define time step, m and err
        const double h = std::min(delta_t, t_final - t_current);
        long m {1};
        double err {std::numeric_limits<double>::max()};

        // Step 2: Form the extended system (to handle b):
        MatrixXd A_tilde = MatrixXd::Zero(n_interior + 1, n_interior + 1);
        A_tilde.block(0, 0, n_interior, n_interior) = A;
        A_tilde.block(0, n_interior, n_interior, 1) = b;
        VectorXd v_tilde {VectorXd::Ones(n_interior + 1)};
        v_tilde.segment(0, n_interior) = U.row(time_step).transpose();

        // Adaptive loop
        VectorXd w {VectorXd::Zero(m)};
        while (m <= n_interior + 1) {
            // Step 3: Run Arnoldi iteration
            auto [V, H] = arnoldi(A_tilde, v_tilde, m, tolerance);
            const double subdiagonal_entry_H {H(H.rows() - 1, H.cols() - 1)}; // storing for error estimate
            H.conservativeResize(H.rows() - 1, H.cols()); // drops last row of \overline{H}_m

            // Step 4: Compute approximation
            const double beta = v_tilde.norm();
            VectorXd mat_exp_vector {(h * H).exp().col(0)};
            w = beta * V * mat_exp_vector;

            // Step 5: Error estimation
            const double scalar {VectorXd::Unit(m, m - 1).transpose() * mat_exp_vector};
            err = std::abs(subdiagonal_entry_H) * std::abs(scalar);
            if (err < tolerance) {
                std::println("[INFO] Error = {:.4e} ({:.4e}) with subspace K_m = {}!",
                err, tolerance, m);
            };
            m += 1; // Increase Krylov dimension
        }
        if (err >= tolerance) {
            std::println("[WARN] Did not converge! Error = {:.4e} ({:.4e})", err, tolerance);
        }

        // Step 6: Store results
        U.row(time_step + 1) = w(Eigen::seqN(0, n_interior)).transpose();
        t_current += h;
    }

    // Check results
    std::println("[INFO] matrix U:\n{}", U);

    return EXIT_SUCCESS;
}