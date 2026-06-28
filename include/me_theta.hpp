#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

// Single-precision (eps = 2^{-24}) theta_m values for the Al-Mohy & Higham (2011)
// scaling-and-squaring matrix exponential algorithm.
// me_theta[m-1] is the largest ||A||_1 such that the degree-m Taylor polynomial
// T_m(A) approximates e^A to within single-precision relative error.
// Precomputed symbolically via reference/ME_European_Option_Pricing_Basket.ipynb.
inline constexpr std::array<double, 55> me_theta = {{
    1.1920928955064624e-07,  // m= 1
    5.9788588938051650e-04,  // m= 2
    1.1233864735324834e-02,  // m= 3
    5.1166193634450866e-02,  // m= 4
    1.3084871645994234e-01,  // m= 5
    2.4952893228466938e-01,  // m= 6
    4.0145824235104810e-01,  // m= 7
    5.8005246276887690e-01,  // m= 8
    7.7951133743579970e-01,  // m= 9
    9.9518407900044570e-01,  // m=10
    1.2234795424238873e+00,  // m=11
    1.4616615072090255e+00,  // m=12
    1.7076485296090935e+00,  // m=13
    1.9598505859598980e+00,  // m=14
    2.2170443949747205e+00,  // m=15
    2.4782808775219713e+00,  // m=16
    2.7428171126987797e+00,  // m=17
    3.0100663628176343e+00,  // m=18
    3.2795612126359970e+00,  // m=19
    3.5509262147064953e+00,  // m=20
    3.8238574254509228e+00,  // m=21
    4.0981069721915060e+00,  // m=22
    4.3734713118401830e+00,  // m=23
    4.6497822241007580e+00,  // m=24
    4.9268998437559110e+00,  // m=25
    5.2047072280123600e+00,  // m=26
    5.4831060876586340e+00,  // m=27
    5.7620134084476740e+00,  // m=28
    6.0413587581925710e+00,  // m=29
    6.3210821263019610e+00,  // m=30
    6.6011321795011610e+00,  // m=31
    6.8814648452096480e+00,  // m=32
    7.1620421544877610e+00,  // m=33
    7.4428312919366010e+00,  // m=34
    7.7238038115539920e+00,  // m=35
    8.0049349864362980e+00,  // m=36
    8.2862032670021790e+00,  // m=37
    8.5675898276624520e+00,  // m=38
    8.8490781859239630e+00,  // m=39
    9.1306538810901190e+00,  // m=40
    9.4123042022194380e+00,  // m=41
    9.6940179569632450e+00,  // m=42
    9.9757852744711250e+00,  // m=43
    1.0257597436801165e+01,  // m=44
    1.0539446734267605e+01,  // m=45
    1.0821326340991204e+01,  // m=46
    1.1103230207589027e+01,  // m=47
    1.1385152968446757e+01,  // m=48
    1.1667089861242232e+01,  // m=49
    1.1949036656413195e+01,  // m=50
    1.2230989594735464e+01,  // m=51
    1.2512945332603945e+01,  // m=52
    1.2794900895640510e+01,  // m=53
    1.3076853639912011e+01,  // m=54
    1.3358801217693320e+01,  // m=55
}};

// Higham & Tisseur (2000) Algorithm 2.4
// Estimates ||A||_1 for the linear operator defined by matvec / rmatvec
// without forming A explicitly.  Block size t=2 matches scipy's default.
//
// Convergence criterion: the row of A^T*S with the largest inf-norm no longer
// exceeds the current best column 1-norm of A*X (i.e. no new column of A can
// improve our estimate).
inline double onenorm_est(
    const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& matvec,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& rmatvec,
    int n, int t = 2, int itmax = 5)
{
    // Initialise X: first column = uniform 1/n, remaining columns alternate ±1/n
    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(n, t);
    X.col(0).setConstant(1.0 / n);
    for (int j = 1; j < t; ++j)
        for (int i = 0; i < n; ++i)
            X(i, j) = ((i + j) % 2 == 0) ? 1.0 / n : -1.0 / n;

    double est     = 0.0;
    double est_old = 0.0;

    // Track which unit-vector columns have already been tried
    std::vector<int> ind_hist;
    ind_hist.reserve(static_cast<std::size_t>(t * itmax));

    for (int k = 0; k < itmax; ++k) {
        // Y = A * X  (apply matvec to each column)
        Eigen::MatrixXd Y(n, t);
        for (int j = 0; j < t; ++j)
            Y.col(j) = matvec(X.col(j));

        // Estimate = max column 1-norm of Y
        int j_best = 0;
        est = 0.0;
        for (int j = 0; j < t; ++j) {
            const double cn = Y.col(j).cwiseAbs().sum();
            if (cn > est) { est = cn; j_best = j; }
        }

        if (est <= est_old && k > 0) break;
        est_old = est;

        // S = sign(Y), with sign(0) = +1
        Eigen::MatrixXd S =
            Y.unaryExpr([](double v) { return v >= 0.0 ? 1.0 : -1.0; });

        // Z = A^T * S
        Eigen::MatrixXd Z(n, t);
        for (int j = 0; j < t; ++j)
            Z.col(j) = rmatvec(S.col(j));

        // h[i] = ||Z[i, :]||_inf  (inf-norm of each row of Z)
        Eigen::VectorXd h = Z.cwiseAbs().rowwise().maxCoeff();

        // Convergence: no row of A^T*S exceeds the current best column 1-norm
        if (h.maxCoeff() <= est) break;

        // Select the t rows with the largest h not yet in ind_hist
        std::vector<int> order(static_cast<std::size_t>(n));
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return h[a] > h[b]; });

        int added = 0;
        X.setZero();
        for (int idx : order) {
            if (added >= t) break;
            if (std::find(ind_hist.begin(), ind_hist.end(), idx) == ind_hist.end()) {
                ind_hist.push_back(idx);
                X(idx, added) = 1.0;  // unit vector e_idx
                ++added;
            }
        }
        if (added == 0) break;  // all promising rows already visited

        // Pad unused columns with the first selected unit vector
        for (int j = added; j < t; ++j)
            X.col(j) = X.col(0);

        (void)j_best;  // used above; silences unused-variable warning
    }

    return est;
}

// d_p = ||A^p||_1^{1/p} via onenorm_est on the power operator
// matvec:  x -> A^p * x     (p sequential SpMVs with A)
// rmatvec: x -> (A^T)^p * x (p sequential SpMVs with A^T)
inline double d_p_est(const Eigen::SparseMatrix<double>& A, int p)
{
    const int n = static_cast<int>(A.rows());

    auto matvec = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        Eigen::VectorXd y = x;
        for (int i = 0; i < p; ++i) y = A * y;
        return y;
    };

    auto rmatvec = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        Eigen::VectorXd y = x;
        for (int i = 0; i < p; ++i) y = A.transpose() * y;
        return y;
    };

    return std::pow(onenorm_est(matvec, rmatvec, n), 1.0 / static_cast<double>(p));
}

// Exact 1-norm of a column-major sparse matrix: max absolute column sum
inline double me_sparse_1norm(const Eigen::SparseMatrix<double>& A)
{
    double norm = 0.0;
    for (int k = 0; k < A.outerSize(); ++k) {
        double col_sum = 0.0;
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it)
            col_sum += std::abs(it.value());
        norm = std::max(norm, col_sum);
    }
    return norm;
}

// Al-Mohy & Higham (2011) m* / s selector
// Returns {m_star, s} minimizing total Taylor work m*s for exp(t_final * A_op).
//
// Main branch (eq. 3.11): direct search over m when ||t_final * A_op||_1 is small.
// Else branch: joint search over p and m using
//   alpha_p = max(d_p, d_{p+1}) * t_final
// where d_p = ||A_op^p||_1^{1/p} is estimated via onenorm_est on the power operator,
// matching the Python reference (d_p / alpha dict in the notebook).
inline std::pair<int, int> me_select_m_s(
    const Eigen::SparseMatrix<double>& A_op, double t_final)
{
    constexpr int m_max = 55;
    constexpr int p_max = 8;
    // Maximum physically expected total work (m*s). Configurations requiring more
    // substeps than this are never the optimum and are skipped to prevent overflow.
    constexpr int64_t cost_cap = 10'000'000LL;

    // Validate in the double domain first, then narrow — never narrow then validate.
    auto ceil_s = [](double a, double b) -> int64_t {
        const double ratio = a / b;
        if (ratio > static_cast<double>(cost_cap)) return cost_cap + 1;
        return static_cast<int64_t>(std::ceil(ratio));
    };

    // Exact scaled 1-norm (||T * A_op||_1 = T * ||A_op||_1)
    const double A_op_1norm = me_sparse_1norm(A_op);
    const double norm_1 = A_op_1norm * t_final;

    // Main-branch threshold: Al-Mohy & Higham (2011) eq. (3.11), l=1, n_0=1
    const double threshold =
        2.0 * (me_theta[m_max - 1] / m_max) * p_max * (p_max + 3);

    auto theta_at = [](int m) -> double {
        return me_theta[static_cast<std::size_t>(m - 1)];
    };

    if (norm_1 < threshold) {
        int64_t min_cost = std::numeric_limits<int64_t>::max();
        int m_star = m_max;
        for (int m = 1; m <= m_max; ++m) {
            const int64_t cost = static_cast<int64_t>(m) * ceil_s(norm_1, theta_at(m));
            if (cost > cost_cap) continue;
            if (cost < min_cost) { min_cost = cost; m_star = m; }
        }
        const int s = static_cast<int>(std::max(int64_t{1}, ceil_s(norm_1, theta_at(m_star))));
        return {m_star, s};
    }

    // Else branch: compute d[p] = ||A_op^p||_1^{1/p} for p = 2..p_max+1
    // alpha[p] = max(d[p], d[p+1]) * t_final  (scaled to the full time horizon)
    std::array<double, p_max + 2> d{};
    for (int p = 2; p <= p_max + 1; ++p) {
        d[static_cast<std::size_t>(p)] = d_p_est(A_op, p) * t_final;
    }

    int64_t min_cost = std::numeric_limits<int64_t>::max();
    int m_star = m_max;
    for (int p = 2; p <= p_max; ++p) {
        const double alpha =
            std::max(d[static_cast<std::size_t>(p)],
                     d[static_cast<std::size_t>(p + 1)]);
        const int m_lo = p * (p - 1) - 1;
        for (int m = m_lo; m <= m_max; ++m) {
            const int64_t cost = static_cast<int64_t>(m) * ceil_s(alpha, theta_at(m));
            if (cost > cost_cap) continue;
            if (cost < min_cost) { min_cost = cost; m_star = m; }
        }
    }
    const int s = static_cast<int>(std::max(int64_t{1}, min_cost / static_cast<int64_t>(m_star)));
    return {m_star, s};
}
