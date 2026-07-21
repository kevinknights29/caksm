/**
 * @file regime_control.cpp
 * @brief Controls on the synthetic scaffold (numerics gate).
 *
 * Verifies that the instrument's a-priori predictions about arithmetic hold before any
 * claim about communication. Exits non-zero on failure, so the sweep scripts can depend on
 * it. Nothing here is a timing measurement.
 *
 * Five gating checks, each a predicted-then-tested boundary:
 *   C1  analytic spectrum       Kronecker-sum eigenvalues match a dense eigensolver
 *   C2  permutation invariance  the scatter knob moves bytes and nothing else
 *   C3  Krylov dimension m      measured m obeys the Hochbruck-Lubich bound (Hermitian PSD)
 *   C4  basis conditioning      measured kappa matches the row-scaled Vandermonde (normal)
 *   C5  CholeskyQR certificate  CholQR2 attains O(u) at every certified s, over a physical
 *                               starting-vector ensemble
 *
 * plus three non-gating findings:
 *   C3b shift-invariance of m   a shift leaves m fixed
 *   C5b small-m safety          the diffusive cycle fits in few certified blocks
 *   C6  non-normality           non-normality is measured (Henrici departure); normal
 *                               operators predict exactly, non-normal ones by mechanism
 *
 * C2 is load-bearing: if a symmetric permutation moved the spectrum, the "pure R_v line"
 * would not exist.
 *
 * Usage:
 *   ./regime-control [--n1 N] [--dim D] [--h H] [--tol T] [--scale S] [--shift MU]
 *                    [--s-ceiling S] [--m-ceiling M]
 *                    [--machine amd-3960x] [--P P] [--csv PATH]
 *
 * @author Kevin Knights
 * @date 2026-07-10
 */

#include <filesystem>
#include <fstream>
#include <iostream>

#include "regime_control_support.hpp"

using namespace regime_control;   // the support layer: operators, decomposition, ledger

// measured_s_max lives in ca_arnoldi.hpp (shared with the tests).

namespace {

// C1: the analytic spectrum is the real spectrum
void check_spectrum(const SyntheticSpec& spec, const SyntheticOperator& op)
{
    // No analytic spectrum to compare against (correlation broke separability): C1 is a
    // passing finding, which is C6's premise.
    if (!op.has_analytic_spectrum) {
        record("C1 spectrum",
               "the Kronecker sum's eigenvalues are the sum-set of the 1D spectra",
               true, "N/A: this operator has no closed-form (separable) spectrum -- the "
                     "spectrum is obtained densely instead (finding; see C6)",
               /*gating=*/false);
        return;
    }

    constexpr int64_t kDenseLimit = 2048;
    if (op.n > kDenseLimit) {
        record("C1 spectrum", "Kronecker-sum eigenvalues equal a dense eigensolver's",
               true, std::format("skipped: N={} exceeds dense limit {}", op.n, kDenseLimit));
        return;
    }

    const Eigen::MatrixXd dense = Eigen::MatrixXd(op.A);

    // A non-normal operator is not self-adjoint, so its eigenvalues come from the general
    // solver. The convection-diffusion spectrum is real for |gamma| < 1, so take real parts
    // and check the imaginary parts are noise.
    Eigen::VectorXd numeric;
    double max_imag = 0.0;
    if (op.is_normal) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(dense, Eigen::EigenvaluesOnly);
        numeric = es.eigenvalues();
    } else {
        Eigen::EigenSolver<Eigen::MatrixXd> es(dense, /*computeEigenvectors=*/false);
        numeric  = es.eigenvalues().real();
        max_imag = es.eigenvalues().imag().cwiseAbs().maxCoeff();
        std::sort(numeric.data(), numeric.data() + numeric.size());
    }

    Eigen::VectorXd analytic = op.lambda;
    std::sort(analytic.data(), analytic.data() + analytic.size());

    const double spread   = op.lambda_max - op.lambda_min;
    const double max_err  = (analytic - numeric).cwiseAbs().maxCoeff();
    const double rel_err  = (spread > 0.0) ? max_err / spread : max_err;

    // The analytic formula is exact but the dense reference is not: a non-normal matrix has
    // eigenvalue forward error O(kappa(X) u) (Bauer-Fike), so scale the tolerance by
    // kappa(X). It stays 1e-10 for normal operators.
    const double eig_tol  = std::max(1e-10, 100.0 * op.eigvec_condition
                                            * std::numeric_limits<double>::epsilon());
    const bool   real_ok  = (max_imag < 1e-9 * std::max(1.0, spread) * op.eigvec_condition);
    const bool   ok       = (rel_err < eig_tol) && real_ok;

    record("C1 spectrum",
           "the Kronecker sum's eigenvalues are the sum-set of the 1D spectra, exactly "
           "(real even when non-normal, |gamma|<1)",
           ok, std::format("max relative error {:.3e} (tol {:.1e}, kappa(X)-scaled), "
                           "max|imag|={:.2e}, N={}, dim={}",
                           rel_err, eig_tol, max_imag, op.n, spec.dim));
}

// C2: the scatter knob is a similarity transform.
// The load-bearing control: A' = P A P^T for every b, so spectrum, Krylov dimension, and
// basis conditioning are all invariant under b; only bytes-from-DRAM move.
void check_permutation_invariance(const Args& a, const SyntheticOperator& banded)
{
    SyntheticSpec scattered = a.spec;
    scattered.scatter_block = banded.n;                  // full global shuffle
    const SyntheticOperator sc = build_synthetic(scattered);

    // Same starting vector, carried into each operator's index space.
    Eigen::VectorXd v0(banded.n);
    for (Eigen::Index i = 0; i < v0.size(); ++i)
        v0(i) = std::sin(0.7 * static_cast<double>(i) + 1.0);
    v0.normalize();

    const Eigen::VectorXd vb = permute(banded.perm, v0);
    const Eigen::VectorXd vs = permute(sc.perm, v0);

    const bool same_nnz = (banded.nnz == sc.nnz);

    // The exponential integrator's sign: exp(-h A) for a positive-definite scaffold,
    // matching the decaying (heat-equation) sign of the Black-Scholes operator.
    const int m_banded = measured_krylov_dim(banded.A, vb, -a.h, a.tol, a.m_ceiling);
    const int m_scatt  = measured_krylov_dim(sc.A,     vs, -a.h, a.tol, a.m_ceiling);
    const bool same_m  = (m_banded == m_scatt);

    // Probe kappa at the largest s below the roundoff horizon: the two operators agree in
    // exact arithmetic, but the floating-point disagreement is ~kappa*u, so comparing where
    // kappa is large would fail spuriously. Grow s until kappa would exceed 1e8.
    int s_probe = 1;
    for (int s = 2; s <= std::min<int>(8, static_cast<int>(banded.n) - 1); ++s) {
        if (condition_number(matrix_powers(banded.A, vb, s)) > 1e8) break;
        s_probe = s;
    }
    const double k_banded = condition_number(matrix_powers(banded.A, vb, s_probe));
    const double k_scatt  = condition_number(matrix_powers(sc.A,     vs, s_probe));
    const double k_rel = std::abs(k_banded - k_scatt) / std::max(k_banded, k_scatt);
    const bool same_kappa = k_rel < 1e-6;

    const bool ok = same_nnz && same_m && same_kappa;
    record("C2 permutation invariance",
           "a symmetric permutation moves the gather window and nothing else: "
           "nnz, m, and kappa are invariant in b",
           ok,
           std::format("nnz {}=={} : {} | m {}=={} : {} | kappa {:.6e} vs {:.6e} "
                       "(rel {:.2e}) : {}",
                       banded.nnz, sc.nnz, same_nnz ? "ok" : "FAIL",
                       m_banded, m_scatt, same_m ? "ok" : "FAIL",
                       k_banded, k_scatt, k_rel, same_kappa ? "ok" : "FAIL"));
}

// C3: measured m obeys the a-priori spectral bound.
// Hochbruck-Lubich is a pessimistic upper bound, so the control is two-sided: it must hold
// (measured <= predicted) and be useful (predicted not wild). It is a Hermitian PSD theorem,
// so on a shifted-indefinite or non-normal operator it is reported but not gated (gating := PSD).
void check_krylov_dimension(const Args& a, const SyntheticOperator& op,
                            const Eigen::VectorXd& v0_perm, int& m_measured_out)
{
    const int ceiling = std::min(a.m_ceiling, static_cast<int>(op.n));
    const int m_meas = measured_krylov_dim(op.A, v0_perm, -a.h, a.tol, ceiling);
    m_measured_out = m_meas;

    // No analytic spectrum (correlation != 0) means the H-L bound has no input; measure m.
    // This is C6's whole point.
    if (!op.has_analytic_spectrum) {
        record("C3 Krylov dimension",
               "measured m for exp(-hA)v is bounded by Hochbruck-Lubich (Hermitian PSD)",
               true,
               std::format("measured m={} | NO ANALYTIC SPECTRUM (correlation!=0): the "
                           "H-L bound has no input; m must be measured (finding)", m_meas),
               /*gating=*/false);
        return;
    }

    const int m_pred = predict_krylov_dim(op.lambda_min, op.lambda_max, a.h, a.tol);

    // A measured m pinned at the search ceiling has not converged; it would satisfy
    // "m_meas <= m_pred" for the wrong reason. Refuse to let that pass silently.
    const bool saturated   = (m_meas >= ceiling);
    const bool bound_holds = (m_meas <= m_pred);
    const bool bound_useful = (m_pred <= 4 * std::max(1, m_meas));
    const bool psd = is_hermitian_psd(op);

    record("C3 Krylov dimension",
           "measured m for exp(-hA)v is bounded by Hochbruck-Lubich on the "
           "analytic spectral spread (Hermitian PSD only)",
           bound_holds && bound_useful && !saturated,
           std::format("measured m={}  predicted m<={}  (spread={:.4f}, h={:.2e}, "
                       "tol={:.1e}) | bound holds: {} | useful (<=4x): {}{}{}",
                       m_meas, m_pred, op.lambda_max - op.lambda_min, a.h, a.tol,
                       bound_holds ? "yes" : "NO", bound_useful ? "yes" : "NO",
                       saturated ? std::format(" | SATURATED at --m-ceiling={}", ceiling)
                                 : std::string{},
                       psd ? std::string{}
                           : " | operator not Hermitian-PSD: bound N/A here, finding only"),
           /*gating=*/psd);
}

// C3b: a shift leaves the Krylov dimension m fixed (a tested claim, not an assumption).
// The Krylov subspace is exactly shift-invariant, but measured_krylov_dim stops on an
// absolute residual, which under a shift picks up a factor e^{h mu}. For small h mu the
// extra step count is usually zero, so we measure both and report. The reference m used
// downstream for a shifted operator is the unshifted twin's.
void check_shift_invariance(const Args& a, const SyntheticOperator& op)
{
    SyntheticSpec unshifted = a.spec;
    unshifted.spec_shift    = 0.0;
    unshifted.scatter_block = 1;
    const SyntheticOperator twin = build_synthetic(unshifted);

    const int ceiling = std::min(a.m_ceiling, static_cast<int>(op.n));
    const Eigen::VectorXd v_op   = permute(op.perm,   sine_vector(op.n));
    const Eigen::VectorXd v_twin = permute(twin.perm, sine_vector(twin.n));

    const int m_shift  = measured_krylov_dim(op.A,   v_op,   -a.h, a.tol, ceiling);
    const int m_ref    = measured_krylov_dim(twin.A, v_twin, -a.h, a.tol, ceiling);

    const bool invariant = (m_shift == m_ref);
    record("C3b shift-invariance of m (finding)",
           std::format("a shift of mu={:g} leaves m fixed (Krylov is shift-invariant); "
                       "residual e^{{h mu}} factor = {:.4f}",
                       a.spec.spec_shift, std::exp(a.h * a.spec.spec_shift)),
           invariant,
           std::format("m(shifted)={} vs m(unshifted twin)={}: {}",
                       m_shift, m_ref, invariant ? "invariant"
                                                 : "moved by the absolute-residual factor"),
           /*gating=*/false);
}

// C4: basis conditioning tracks the spectrum.
// kappa([v, ..., A^s v]) is predicted from lambda and c = X^{-1} v alone, a normal-operator
// theorem (kappa(basis) = kappa(M), the row-scaled Vandermonde). For a non-normal operator
// the true kappa gains a kappa(X) factor, so C4 is gated only when normal; otherwise it is a
// finding and its gap is what C6 quantifies. Tolerance is tight (0.1 decades).
void check_basis_conditioning(const Args& a, const SyntheticOperator& op,
                              const EigenData& eig,
                              const Eigen::VectorXd& v0_perm,
                              std::vector<KappaRow>& kappa_curve)
{
    constexpr double kTrustable = 1e12;    // above this, measured kappa is noise
    constexpr double kTolDecades = 0.1;    // ~1.26x; near-exact is expected
    double worst_decades = 0.0;
    int    compared = 0;

    for (int s = 1; s <= std::min(a.s_ceiling, static_cast<int>(op.n) - 1); ++s) {
        const double k_meas = condition_number(matrix_powers(op.A, v0_perm, s));
        const double k_pred = predicted_kappa(eig, s);

        // Both are recorded at every s, including past kTrustable, so the plot can
        // show where the measured curve stops being trustworthy rather than truncating.
        const bool trusted = (k_meas <= kTrustable);
        kappa_curve.push_back({s, k_meas, k_pred, trusted});
        if (!trusted) break;

        const double decades = std::abs(std::log10(k_meas) - std::log10(k_pred));
        worst_decades = std::max(worst_decades, decades);
        ++compared;
    }

    // Gate on normality, not PSD or separability: the Vandermonde identity holds for any
    // normal operator (indefinite or non-separable both satisfy it exactly). It fails only
    // for a genuinely non-normal operator, where the gap is bounded by kappa(X) (C6).
    const bool normal = op.is_normal;
    const bool agrees = (compared > 0) && (worst_decades < kTolDecades);
    record("C4 basis conditioning",
           "kappa of the monomial basis equals kappa of the row-scaled Vandermonde "
           "diag(c)*V(lambda) -- for ANY normal operator, closed-form or not",
           agrees,
           std::format("[{}] compared s=1..{} below kappa<{:.0e}; worst disagreement "
                       "{:.3f} decades (tol {:.1f}){}",
                       eig.source, compared, kTrustable, worst_decades, kTolDecades,
                       normal ? std::string{}
                              : std::format(" | GENUINELY NON-NORMAL (kappa(X)={:.2e}): "
                                            "gap expected, finding only, see C6", eig.kappa_X)),
           /*gating=*/normal);
}

// C5: the CholeskyQR certificate is sound, stated as a per-vector ledger.
// The certificate is demand <= supply, but both depend on v (an eigenvector start gives
// s_max=0 but m=1). So for each v we compute blocks = ceil(m(v)/(s_max(v)+1)) and take the
// worst case over the physical ensemble. Not gated: plain-CholQR LLT survival past the
// certificate (a finding) and tightness. The soundness gate is: every vector, every
// certified s, CholeskyQR2 attains O(u).
void check_cholqr_certificate(const Args& a, const SyntheticOperator& op,
                              const Eigen::VectorXd& base, int m_sine,
                              int& s_pred_out, int& s_max_worst_out,
                              int& blocks_worst_out, int& s_llt_out)
{
    const int ceiling = std::min(a.s_ceiling, static_cast<int>(op.n) - 1);

    // Spectrum-alone prediction from the sine vector (analytic operators only): the
    // a-priori value the confound study plots. NaN-safe: skipped without a spectrum.
    const int s_pred = op.has_analytic_spectrum
        ? predicted_s_max(op.lambda, spectral_coefficients(a.spec, base), ceiling)
        : 0;

    // Per-vector certificate over the physical ensemble (smooth, solver-like vectors).
    const std::vector<Eigen::VectorXd> ens = physical_ensemble(op, base);
    std::vector<VectorCert> certs;
    int s_max_worst = ceiling, blocks_worst = 1;
    for (std::size_t i = 0; i < ens.size(); ++i) {
        certs.push_back(certify_vector(op, a, ens[i], std::format("v{}", i)));
        s_max_worst  = std::min(s_max_worst, certs.back().s_max);
        blocks_worst = std::max(blocks_worst, certs.back().blocks);
    }
    // The dominant-mode stress case, reported but not part of the certificate.
    const VectorCert stress = certify_vector(op, a, stress_vector(op), "stress(dominant)");

    // Soundness gate: every vector, every certified s, CholQR2 is O(u).
    constexpr double kOrthoTarget = 1e-13;
    double worst_ortho = 0.0;
    bool   sound = true;
    for (const Eigen::VectorXd& v : ens) {
        const Eigen::VectorXd vp = permute(op.perm, v);
        const int s_here = measured_s_max(op.A, vp, ceiling);
        if (s_here < 1) continue;   // 1-D Krylov space: nothing to orthogonalize beyond v
        for (int s = 1; s <= s_here; ++s) {
            const CholQrResult qr = cholesky_qr2(matrix_powers(op.A, vp, s));
            if (qr.llt_failed) { sound = false; break; }
            worst_ortho = std::max(worst_ortho, orthogonality_loss(qr.Q));
        }
    }
    sound = sound && (worst_ortho < kOrthoTarget);

    // Conservatism margin (finding): where plain CholQR's Cholesky gives up, sine vector.
    int s_llt = 0;
    for (int s = 1; s <= ceiling; ++s) {
        if (cholesky_qr(matrix_powers(op.A, permute(op.perm, sine_vector(op.n)), s)).llt_failed)
            break;
        s_llt = s;
    }

    s_pred_out       = s_pred;
    s_max_worst_out  = s_max_worst;
    blocks_worst_out = blocks_worst;
    s_llt_out        = s_llt;

    // Compact per-vector ledger in the detail line.
    std::string ledger;
    for (const VectorCert& c : certs)
        ledger += std::format("{}(m{} s{} b{}) ", c.label, c.m, c.s_max, c.blocks);

    record("C5 CholeskyQR certificate",
           std::format("kappa(B) <= u^(-1/2) = {:.2e} certifies that CholeskyQR2 "
                       "attains O(u) orthogonality", cholqr_kappa_limit()),
           sound,
           std::format("per-vector [{}]| worst ||I-Q^T Q||={:.2e} (< {:.0e}): {} | "
                       "plain-CholQR LLT survives to s={} (finding)",
                       ledger, worst_ortho, kOrthoTarget,
                       sound ? "sound" : "UNSOUND", s_llt));

    // C5b: the worst-case blocks needed over the physical ensemble (m and s_max paired per
    // vector), plus the stress case. blocks_worst is a result about the OPERATOR, so the
    // verdict thresholds on the value proposition (does s-step still cut reductions?), not on
    // the block count. nnz is printed to tie the tuples to this operator.
    const int    s_block  = s_max_worst + 1;   // largest certified block width
    const int64_t red_ca  = ca_reductions(m_sine, s_block, true);
    const double  cut     = static_cast<double>(mgs_reductions(m_sine))
                          / static_cast<double>(red_ca);
    record("C5b small-m safety (finding, not a gate)",
           "the diffusive solver's cycle needs blocks = ceil(m(v)/(s_max(v)+1)) certified "
           "matrix-powers blocks (worst over the physical ensemble); s-step still cuts the "
           "reduction count",
           cut > 1.0,
           std::format("op nnz={} | worst blocks needed={} (s_max>={}, block width<={}) | "
                       "stress(dominant-mode): m{} s_max{} blocks{} | reductions/cycle at "
                       "m={}: MGS {} vs s-step {} ({:.1f}x)",
                       op.nnz, blocks_worst, s_max_worst, s_block,
                       stress.m, stress.s_max, stress.blocks,
                       m_sine, mgs_reductions(m_sine), red_ca, cut),
           /*gating=*/false);
}

// C6: normality is load-bearing; where the operator is non-normal, whether the spectral
// prediction survives is mechanism-dependent, not a universal law. Non-normality is measured
// (Henrici's departure), never inferred from a knob.
//
//   normal, any variant:  prediction exact (gap ~ 0): closed form, indefinite, or dense.
//   non-normal:           the error in s-step units (one step ~ 1.2 decades of kappa(B_s))
//     depends on the mechanism: constant-coefficient convection-diffusion degrades slowly and
//     stays under one s-step even at kappa(X)=3e8 (certificate survives), while variable-
//     coefficient advection exceeds one s-step at kappa(X)~1e5 (certificate does not). So the
//     gap-vs-kappa(X) "law" is scoped to the constant-coefficient family. The real
//     Black-Scholes operator has variable coefficients, so its s_max must be measured. Both
//     certificates are reported (conservative and measured s_max).
//
// kappa(X) uses unit-2-norm eigenvector columns; near defectiveness makes it large but not
// unreliable (it reproduces across eigensolvers, see the real_bs_operator note). Always a
// finding, never gated.
void check_nonnormality(const Args& a, const SyntheticOperator& op, const EigenData& eig,
                        const Eigen::VectorXd& v_base, const Eigen::VectorXd& v_perm,
                        int s_max_worst, NonNormalReport& rep)
{
    const double henrici = op.henrici;
    const double a_fro   = op.A.norm();
    const double rel_dep = henrici / std::max(1e-30, a_fro * a_fro);
    const int    s_probe = std::max(2, std::min(6, static_cast<int>(op.n) - 1));
    const double k_meas  = condition_number(matrix_powers(op.A, v_perm, s_probe));
    const double k_pred  = predicted_kappa(eig, s_probe);
    const double gap_dec = std::abs(std::log10(k_meas) - std::log10(k_pred));

    rep.kappa_X     = eig.kappa_X;
    rep.henrici_rel = rel_dep;
    rep.gap_dec     = gap_dec;
    // The real BS operator couples all three asset pairs and is not a Kronecker sum of the
    // synthetic factors, so the tensor identity does not apply to it.
    rep.kappa_X_struct = a.real_bs > 0 ? std::numeric_limits<double>::quiet_NaN()
                                       : structured_kappa_X(a.spec);

    if (op.is_normal) {
        const std::string kind =
            op.has_analytic_spectrum
                ? (is_hermitian_psd(op) ? "normal + separable (closed-form spectrum)"
                                        : "normal but INDEFINITE (kappa(X)=1; a shift, not "
                                          "non-normality)")
                : "normal but NON-SEPARABLE (spectrum from a dense eigensolver)";
        record("C6 normality vs separability (finding)",
               "the spectral prediction needs NORMALITY, not a closed form: it is exact "
               "for any normal operator, indefinite or non-separable included",
               gap_dec < 0.05,
               std::format("{} | henrici={:.2e} (rel {:.2e}) ~ 0 | prediction gap at s={} "
                           "= {:.3f} decades (exact) | source: {}",
                           kind, henrici, rel_dep, s_probe, gap_dec, eig.source),
               /*gating=*/false);
        return;
    }

    // Genuinely non-normal. Convert the measured gap into s-step units. The conversion slope
    // (decades of kappa(B_s) per unit s) is spectrum-dependent (~0.95-1.2); use the shallowest
    // observed slope, a fixed constant, so gap_steps stays conservative (safely larger).
    constexpr double kConversionSlope = 0.95;   // shallowest dec-per-step across the battery
    const double kX        = eig.kappa_X;
    const double gap_steps = gap_dec / kConversionSlope;
    const bool   survives  = gap_steps < 1.0;   // cannot move the integer s_max
    const bool   kX_reliable = kX < 1e10;       // beyond this, X nears defective

    // The two certificates: conservative (rigorous) vs measured (empirical).
    const int ceiling = std::min(a.s_ceiling, static_cast<int>(op.n) - 1);
    const int s_cons  = conservative_s_max(eig, ceiling);

    // Name the mechanism: the point of C6 is that survival depends on which. For
    // constant-coefficient advection, prove diagonal removability (D^-1 A D symmetric).
    std::string mechanism;
    if (a.real_bs > 0)
        mechanism = "REAL Black-Scholes (log coords: CONSTANT-coefficient + cross terms)";
    else if (a.spec.var_advection != 0.0)
        mechanism = "variable-coefficient advection (non-Toeplitz, no single diagonal similarity)";
    else if (a.spec.correlation != 0.0)
        mechanism = "correlation +/- advection (mixed derivative; symmetric rho commutes w/ drift)";
    else
        mechanism = std::format("constant-coefficient advection (diagonally removable: "
                                "D^-1 A D departure {:.1e})",
                                departure_after_symmetrizer(op, advection_symmetrizer(a.spec)));

    // Two axes carrying neither the cross term nor the advection ramp are interchangeable,
    // an exact symmetry of A, so the spectrum repeats and any basis of a degenerate
    // eigenspace will do: the dense kappa(X) is then the eigensolver's choice rather than the
    // operator's. Flag it and give the canonical tensor-basis value alongside.
    const int excluded_axes = a.spec.correlation != 0.0 ? 2                       // axes 0, 1
                            : (a.spec.var_advection != 0.0 ? 1 : 0);              // axis 0
    const int clean_axes    = a.real_bs > 0 ? 0 : a.spec.dim - excluded_axes;
    std::string kappa_X_note;
    if (clean_axes >= 2)
        kappa_X_note = std::isfinite(rep.kappa_X_struct)
            ? std::format(" [BASIS-DEPENDENT: {} interchangeable axes leave the spectrum "
                          "degenerate, so this value is the eigensolver's basis choice; "
                          "canonical tensor-basis kappa(X)={:.3e}]",
                          clean_axes, rep.kappa_X_struct)
            : std::format(" [BASIS-DEPENDENT: {} interchangeable axes leave the spectrum "
                          "degenerate; no canonical value (operator does not factor)]",
                          clean_axes);

    // The direct, conversion-free test: does the spectral prediction get the integer s_max
    // right? Raw s_max moves by 1 on the CholQR2 threshold, so agreement is judged at
    // kSMaxStepTolerance (a chosen tolerance, not a measured noise floor); only 2+ matters.
    // Both sides use the same vector v_perm, or supply and demand come from different ones.
    const int ceil_s = std::min(a.s_ceiling, static_cast<int>(op.n) - 1);
    int s_pred_int = 0;
    for (int s = 1; s <= ceil_s; ++s) {
        if (!(predicted_kappa(eig, s) < cholqr_kappa_limit())) break;
        s_pred_int = s;
    }
    const int s_meas_same = measured_s_max(op.A, v_perm, ceil_s);   // same vector
    const int s_int_diff  = std::abs(s_pred_int - s_meas_same);

    // One vector cannot tell "robust" from "broken but unlucky", so report the distribution
    // of |predicted - measured| over the physical ensemble: a worst case of 0-1 is genuine
    // robustness, any 2 is the measure regime.
    int diff_worst = 0, diff_n = 0;
    std::string diff_hist;
    for (const Eigen::VectorXd& ev : physical_ensemble(op, v_base)) {
        const Eigen::VectorXd evp = permute(op.perm, ev);
        const Coeffs cf = coeffs_for(eig, a.spec, ev, evp);
        int sp_e = 0;
        for (int s = 1; s <= ceil_s; ++s) {
            if (!(predicted_kappa_c(eig, cf, s) < cholqr_kappa_limit())) break;
            sp_e = s;
        }
        const int sm_e = measured_s_max(op.A, evp, ceil_s);
        const int df   = std::abs(sp_e - sm_e);
        diff_worst = std::max(diff_worst, df);
        ++diff_n;
        diff_hist += std::format("{}", df);
    }
    // Invariant: any predicted-vs-measured s_max comparison must be taken on the same
    // vector. Both sides above use v_perm.
    rep.s_pred_int  = s_pred_int;
    rep.s_meas_same = s_meas_same;
    rep.s_cons      = s_cons;
    rep.diff_worst  = diff_worst;

    record("C6 non-normal: does the prediction survive? (finding)",
           "whether the spectrum-alone prediction survives non-normality is "
           "MECHANISM-DEPENDENT: constant-coefficient advection degrades logarithmically "
           "and stays under one s-step; variable-coefficient advection can exceed one "
           "s-step and move s_max. The DECISIVE test is the integer: does the spectral "
           "prediction get s_max right?",
           diff_worst <= kSMaxStepTolerance,   // ensemble worst case, at the chosen tolerance
           std::format("[{}] henrici={:.2e}, kappa(X)={:.2e}{}{} | s_max INTEGERS (same "
                       "vector): spectral-predicted={} vs measured={} (differ by {}: {}) "
                       "| ENSEMBLE |diff| over {} vectors: [{}] worst={} -> {} "
                       "| gap={:.3f} dec = {:.2f} s-steps ({}) | conservative certificate "
                       "s_max={} vs ensemble-measured {}",
                       mechanism, henrici, kX,
                       kX_reliable ? "" : " [UNRELIABLE: near-defective X]",
                       kappa_X_note,
                       s_pred_int, s_meas_same, s_int_diff,
                       s_int_diff <= kSMaxStepTolerance
                           ? std::format("within the chosen +/-{} s-step tolerance",
                                         kSMaxStepTolerance)
                           : "DECISION-RELEVANT: moves s_max",
                       diff_n, diff_hist, diff_worst,
                       diff_worst <= kSMaxStepTolerance
                           ? "ROBUST across vectors: prediction USABLE"
                           : "MEASURE regime: some vectors move s_max by 2+",
                       gap_dec, gap_steps,
                       survives ? "under one step" : "over one step",
                       s_cons, s_max_worst),
           /*gating=*/false);
}

}  // namespace

int main(int argc, char* argv[])
{
    try {
        const Args a = parse(std::span<const char* const>(argv, static_cast<std::size_t>(argc)));
        const Machine& mc = lookup_machine(a.machine);

        // The control is a numerics gate, not a timing run: C3/C4/C5 form dense N x (s+1)
        // bases repeatedly, so a large scaffold costs gigabytes to answer an N-independent
        // question (large-N is the sweep harness's job). The ceiling applies to whichever
        // operator is built: real BS (N = real_bs^3) or the synthetic scaffold (N = n1^dim).
        constexpr int64_t kControlMaxN = 32768;
        const int64_t control_N = a.real_bs > 0
            ? static_cast<int64_t>(a.real_bs) * a.real_bs * a.real_bs
            : synthetic_dimension(a.spec);
        if (control_N > kControlMaxN)
            throw std::invalid_argument(std::format(
                "operator N={} exceeds the control limit {} ({}). The control "
                "validates arithmetic, which is N-independent; use a smaller {}. "
                "Large-N timing belongs to the scaling harness.",
                control_N, kControlMaxN,
                a.real_bs > 0 ? std::format("real-bs n={}", a.real_bs)
                              : std::format("n1={}, dim={}", a.spec.n1, a.spec.dim),
                a.real_bs > 0 ? "real-bs n" : "n1/dim"));

        std::println("CA-KSM-EI regime study: numerics control (Laplacian scaffold)");
        std::println("  scaffold: n1={} dim={} -> N={}  scale={:g} shift={:g} advection={:g}",
                     a.spec.n1, a.spec.dim, synthetic_dimension(a.spec),
                     a.spec.spec_scale, a.spec.spec_shift, a.spec.advection);
        std::println("  machine:  {}  scale={:g} shift={:g} advection={:g} correlation={:g}",
                     mc.name, a.spec.spec_scale, a.spec.spec_shift, a.spec.advection,
                     a.spec.correlation);
        std::println("");

        // The control runs on the banded operator (perm = identity); C2 builds the scattered
        // twin. With --real-bs n it runs the actual Black-Scholes operator instead, with its
        // payoff u0 as the starting vector, so the transfer is measured here.
        SyntheticSpec banded_spec = a.spec;
        banded_spec.scatter_block = 1;

        Eigen::VectorXd v0;
        SyntheticOperator op;
        if (a.real_bs > 0) {
            op = real_bs_operator(a.real_bs, v0);
            std::println("  REAL Black-Scholes basket operator: n={} N={} nnz={}",
                         a.real_bs, op.n, op.nnz);
            std::println("  (log-price coordinates: sigma, rho, r are CONSTANT, so despite "
                         "the mixed-derivative\n   cross terms this is a CONSTANT-coefficient "
                         "operator; starting vector = the payoff u0)");
        } else {
            op = build_synthetic(banded_spec);
            v0 = sine_vector(op.n);
        }
        std::println("  operator: normal={} analytic_spectrum={} henrici={:.3e}{}",
                     op.is_normal, op.has_analytic_spectrum, op.henrici,
                     op.has_analytic_spectrum && !op.is_normal
                         ? std::format(" kappa(X)={:.2e}", op.eigvec_condition)
                         : std::string{});
        std::println("");

        const Eigen::VectorXd v0_perm = permute(op.perm, v0);

        // Decompose once (analytic / dense-symmetric / dense-general): the spectral
        // prediction survives loss of the closed form.
        const EigenData eig = decompose(op, a.spec, v0, v0_perm);
        if (!eig.ok)
            std::println("  WARNING: eigensolver did not converge; C4/C6 predictions "
                         "unavailable for this operator");
        std::println("  spectrum source: {}  kappa(X)={:.3e}", eig.source, eig.kappa_X);
        std::println("");

        int m_measured = 0;
        int s_max_pred = 0, s_max_worst = 0, blocks_worst = 1, s_llt = 0;
        std::vector<KappaRow> kappa_curve;

        check_spectrum(a.spec, op);
        // C2 and C3b are synthetic-instrument checks (they build a scattered/unshifted twin
        // from the spec); they do not apply to the real operator, which has no such knobs.
        if (a.real_bs == 0) {
            check_permutation_invariance(a, op);
            if (a.spec.spec_shift != 0.0) check_shift_invariance(a, op);
        }
        check_krylov_dimension(a, op, v0_perm, m_measured);
        check_basis_conditioning(a, op, eig, v0_perm, kappa_curve);
        check_cholqr_certificate(a, op, v0, m_measured, s_max_pred, s_max_worst,
                                 blocks_worst, s_llt);
        NonNormalReport nn;
        check_nonnormality(a, op, eig, v0, v0_perm, s_max_worst, nn);

        // predicted-boundary ledger
        std::println("Predicted-boundary ledger");
        std::println("{}", std::string(78, '='));
        // Only gating checks decide whether the instrument may be used. Findings are
        // reported with their verdict and cannot fail the gate.
        bool all_passed = true;
        for (const Check& c : g_checks) {
            if (c.gating) all_passed = all_passed && c.passed;
            const char* tag = c.gating ? (c.passed ? "PASS" : "FAIL")
                                       : (c.passed ? "yes " : "no  ");
            std::println("[{}] {}", tag, c.name);
            std::println("     predicts: {}", c.prediction);
            std::println("     observed: {}", c.detail);
            std::println("");
        }

        // where this scaffold sits on the map (predictors only)
        // m is the measured Krylov dimension (an output of spectrum, h, tol).
        // The block width is certified: the largest certified block has s_max_worst+1 columns,
        // so blocks = ceil(m/(s_max+1)).
        const int s_block = s_max_worst + 1;
        const int blocks  = (m_measured + s_block - 1) / s_block;

        const RegimePoint pt = place(mc, a.P, op.nnz, op.n, m_measured,
                                     modelled_x_reuse(mc, a.P, banded_spec.scatter_block, op.n));
        constexpr double MiB = 1024.0 * 1024.0;
        std::println("Regime coordinates of this control point (predictors, not outcomes)");
        std::println("{}", std::string(78, '-'));
        std::println("  working set   {:.2f} MiB     aggregate L3  {:.2f} MiB",
                     pt.working_set / MiB, pt.llc / MiB);
        // R_v is reported but never tested here: with N <= 32768 the working set is a few MB,
        // far below aggregate L3, so every control point is R_v << 1. The control validates
        // the numerics and the horizontal mechanism only; the vertical crossover is the
        // sweep's to measure.
        std::println("  R_v = {:.4f}   (theta_v = {:g})   {}  [not tested here: R_v<<1 by design]",
                     pt.rv, kThetaV, pt.rv >= kThetaV ? "above" : "below");
        // R_h's magnitude, and hence any above/below theta_h verdict, is meaningless
        // unless the reduction was calibrated on this machine. Only its shape survives.
        if (!mc.reduction_calibrated)
            std::println("  R_h = {:.4e} (theta_h = {:g})   [verdict SUPPRESSED: reduction "
                         "uncalibrated on this preset -> magnitude meaningless]",
                         pt.rh, kThetaH);
        else
            std::println("  R_h = {:.4e} (theta_h = {:g})   {}",
                         pt.rh, kThetaH, pt.rh >= kThetaH ? "above" : "below");
        // The stable s-step arm needs CholeskyQR2, so it is charged two Gram
        // all-reduces per block, not one. The mechanism survives the correction.
        std::println("  reductions/cycle (m={} measured): MGS {} vs s-step(block<={}, "
                     "{} block(s), CholQR2) {}  -- {:.1f}x",
                     m_measured, mgs_reductions(m_measured), s_block, blocks,
                     ca_reductions(m_measured, s_block, true),
                     static_cast<double>(mgs_reductions(m_measured))
                         / static_cast<double>(ca_reductions(m_measured, s_block, true)));
        std::println("  arithmetic intensity: SpMV {:.4f}  MGS {:.4f} FLOP/byte   "
                     "W {:.3e} FLOP", pt.ai, pt.ai_mgs, pt.W);
        // R_h's denominator is priced per kernel: each sits on the roof its own working set
        // selects, and which roof is the largest single term in the cycle.
        std::println("  cycle {:.3e} s   R_v(SpMV) {:.3f} [{}]   R_v(MGS) {:.3f} [{}]",
                     pt.cycle_s, pt.rv_spmv, pt.spmv_resident ? "L3" : "DRAM",
                     pt.rv_mgs, pt.mgs_resident ? "L3" : "DRAM");
        if (!mc.reduction_calibrated)
            std::println("  NOTE: the reduction is UNCALIBRATED on preset '{}' -- the cost "
                         "parameters are another host's. R_h's magnitude, and hence "
                         "theta_h, is not trustworthy here.", mc.key);
        std::println("");

        // CSV: one row per (s) of the conditioning curve, plus the header facts
        if (!a.csv_path.empty()) {
            const bool exists = std::filesystem::exists(a.csv_path)
                             && std::filesystem::file_size(a.csv_path) > 0;
            std::ofstream f(a.csv_path, std::ios::app);
            if (!f) throw std::runtime_error("Cannot open CSV: " + a.csv_path);
            if (!exists)
                f << "machine,P,n1,dim,N,nnz,scale,shift,advection,correlation,var_advection,real_bs,"
                     "kappa_X,kappa_X_struct,henrici,henrici_rel,gap_dec,s_pred_int,s_meas_same,s_cons,"
                     "diff_worst,is_normal,analytic_spectrum,h,tol,spread,m_measured,"
                     "s_max_phys_worst,s_max_pred_spectrum,blocks_worst,s_block,"
                     "s_llt_survives,R_v,R_h,ai,ai_mgs,cycle_s,rv_spmv,rv_mgs,"
                     "spmv_resident,mgs_resident,W,red_mgs,red_ca_cholqr2,s,"
                     "kappa_measured,kappa_predicted,kappa_trusted,all_passed\n";
            // s_max_phys_worst is the measured worst case over the physical ensemble;
            // blocks_worst = worst ceil(m(v)/(s_max(v)+1)) over it. s_max_pred_spectrum is
            // the a-priori spectrum-alone value (analytic operators only).
            const double spread = op.has_analytic_spectrum
                ? op.lambda_max - op.lambda_min : std::numeric_limits<double>::quiet_NaN();
            for (const auto& [s, k_meas, k_pred, trusted] : kappa_curve) {
                f << mc.key << ',' << a.P << ',' << a.spec.n1 << ',' << a.spec.dim << ','
                  << op.n << ',' << op.nnz << ',' << a.spec.spec_scale << ','
                  << a.spec.spec_shift << ',' << a.spec.advection << ','
                  << a.spec.correlation << ',' << a.spec.var_advection << ',' << a.real_bs << ','
                  << nn.kappa_X << ',' << nn.kappa_X_struct << ',' << op.henrici << ','
                  << nn.henrici_rel << ','
                  << nn.gap_dec << ',' << nn.s_pred_int << ',' << nn.s_meas_same << ','
                  << nn.s_cons << ',' << nn.diff_worst << ',' << (op.is_normal ? 1 : 0) << ','
                  << (op.has_analytic_spectrum ? 1 : 0) << ',' << a.h << ',' << a.tol << ','
                  << spread << ',' << m_measured << ',' << s_max_worst << ','
                  << s_max_pred << ',' << blocks_worst << ',' << s_block << ','
                  << s_llt << ',' << pt.rv << ',' << pt.rh << ',' << pt.ai << ','
                  << pt.ai_mgs << ',' << pt.cycle_s << ','
                  << pt.rv_spmv << ',' << pt.rv_mgs << ','
                  << (pt.spmv_resident ? 1 : 0) << ',' << (pt.mgs_resident ? 1 : 0) << ','
                  << pt.W << ',' << mgs_reductions(m_measured) << ','
                  << ca_reductions(m_measured, s_block, true) << ',' << s << ','
                  << k_meas << ',' << k_pred << ',' << (trusted ? 1 : 0) << ','
                  << (all_passed ? 1 : 0) << '\n';
            }
            std::println("  [Appended {} rows to {}]", kappa_curve.size(), a.csv_path);
        }

        if (!all_passed) {
            std::println(std::cerr,
                "CONTROL FAILED: do not proceed to the scattered-pattern "
                "operator until the control passes.");
            return EXIT_FAILURE;
        }
        std::println("Control passed. The scattered-pattern operator may be used.");
        return EXIT_SUCCESS;

    } catch (const std::exception& e) {
        std::println(std::cerr, "Error: {}", e.what());
        return EXIT_FAILURE;
    }
}
