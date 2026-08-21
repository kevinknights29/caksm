"""
Figures for the CA-KSM-EI regime study (puffin, AMD Threadripper 3960X).

This script consumes:
    data/regime/regime_control.csv    (scripts/regime/regime_control.sh)

Outputs:

  regime_reductions.png     Global reductions per Arnoldi cycle vs m: MGS's 1 + m(m+3)/2
                            against the stable s-step arm's 1 + 2*ceil(m/s_max). The real
                            problem's measured m = 8-11 window is shaded.

  regime_nonnormality.png   Prediction error in decades vs log10 kappa(X), fitted per
                            mechanism against the rigorous y=x bound.

  regime_trajectory.png     The timed (R_v, R_h) regime map: four quadrants split by
                            theta_v=1 and theta_h=1, with the real Black-Scholes
                            operator's grid-refinement trajectory. Dense-control points
                            anchor the original low-grid continuation; placement-only
                            measurements at every production grid expose its bias and
                            replace it over the measured range. The operator refines
                            Lower-Right -> Lower-Left -> Upper-Left, crossing theta_v
                            between n=61 and n=74, and never enters Upper-Right.

The reduction and non-normality figures are arithmetic, not timing: no wall-clock and no
dependence on the machine. regime_trajectory.png is the exception, since it is the map: it
places points using the
calibrated predictor coordinates (R_v from cache geometry, R_h from the calibrated
reduction cost). Those are still predictors computed before a run, not measured runtime
fractions, so the firewall holds; but the figure does license a claim about where CA pays,
which the other figures deliberately do not.

The poster-branch replacement for regime_trajectory.png is specified in
docs/poster_regime_trajectory_spec.md.
"""
# /// script
# dependencies = [
#   "matplotlib",
#   "numpy",
# ]
# ///

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import figstyle as fs
import cpu_figstyle as cpu
from figstyle import mark_better

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data" / "regime"
CONTROL_CSV = DATA / "regime_control.csv"
PLACEMENT_CSV = DATA / "regime_placement.csv"

# Palette, shared with scaling_plot.py where the roles line up.
# Drawn from the shared CPU vocabulary so a hue means the same thing here as it
# does in the scaling and sweep figures.
C_PRED = cpu.C_PREDICTED          # a-priori prediction
C_MEAS = cpu.C_MEASURED           # measurement
C_LIMIT = cpu.C_CEILING           # the certificate
C_GUIDE = cpu.C_GUIDE
C_MGS = cpu.KERNEL_COLOR["SpMV"]  # the modified Gram-Schmidt reduction count
C_CA = cpu.PINE                   # the s-step arm
C_REAL = cpu.OPERATOR_COLOR["black-scholes"]
C_BAND = cpu.SEQUENTIAL[0]

def load_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


# A "run" is one invocation of regime-control: many s-rows sharing one scaffold and one
# spectrum. real_bs is part of the key because the real Black-Scholes operator carries zero
# in every synthetic knob (advection/correlation/var_advection); without it all six real-BS
# grids would collapse into a single run and scalar()/gap_at() would read mixed fields from
# whichever grid loaded first. Appended last so existing positional indices stay valid.
RUN_KEY = ("n1", "dim", "scale", "shift", "h", "tol", "advection", "correlation",
           "var_advection", "real_bs")


def group_runs(rows) -> dict[tuple, list[dict]]:
    runs: dict[tuple, list[dict]] = {}
    for r in rows:
        runs.setdefault(tuple(r[k] for k in RUN_KEY), []).append(r)
    for v in runs.values():
        v.sort(key=lambda r: int(r["s"]))
    return runs


def scalar(run, key, cast=float):
    """A run-level field, constant across the run's s-rows."""
    return cast(run[0][key])


def kappa_X_of(run):
    """kappa(X) for the law's x-axis, preferring the canonical tensor-basis value.

    The dense kappa_X names a property of the operator only while the spectrum is simple.
    The constant-coefficient arm (correlation = 0, var_advection = 0, dim >= 2) is a pure
    Kronecker sum, so its axes are interchangeable, the spectrum repeats, and the dense value
    is inflated by up to 173x, non-monotone in gamma (1731 at 0.05 falling to 1372 at 0.1),
    and moves 1.4-7.9x under exact similarities. The variable-coefficient arm ramps axis 0
    alone, leaving one clean axis, so its spectrum is simple and its dense value is sound.
    Fitting the two arms against x-axes of different validity is what this avoids.

    Returns (value, is_canonical). Falls back to the dense value for CSVs written before
    kappa_X_struct existed, and for operators that do not factor (real BS, var_advection).
    """
    try:
        ks = float(run[0].get("kappa_X_struct", "nan"))
    except (TypeError, ValueError):
        ks = float("nan")
    if np.isfinite(ks) and ks > 0.0:
        return ks, True
    return scalar(run, "kappa_X"), False

def normal_scaffold(k):
    """A normal, separable synthetic run: advection, correlation and var_advection all 0,
    and not the real BS operator. The real operator zeroes every synthetic knob yet carries
    mixed-derivative cross terms (henrici > 0), so it would otherwise pass this filter and
    contaminate the confound/reductions/conditioning figures as a spurious normal scaffold."""
    return (float(k[6]) == 0.0 and float(k[7]) == 0.0 and float(k[8]) == 0.0
            and float(k[9]) == 0.0)


def fixed_spectrum(runs):
    """Normal runs on the baseline 2D scaffold, untouched spectrum: only m varies."""
    out = [r for k, r in runs.items()
           if k[0] == "24" and k[1] == "2"
           and float(k[2]) == 1.0 and float(k[3]) == 0.0 and normal_scaffold(k)]
    return sorted(out, key=lambda run: scalar(run, "m_measured", int))


# The reduction ledger.
def plot_reductions(runs) -> None:
    group = fixed_spectrum(runs)
    m    = np.array([scalar(r, "m_measured", int) for r in group])
    mgs  = np.array([scalar(r, "red_mgs", int) for r in group])
    ca   = np.array([scalar(r, "red_ca_cholqr2", int) for r in group])
    # Same quantity as the confound plot: s_max is the certified power; a block holds
    # s_max+1 vectors. Label BOTH so the two figures read consistently.
    smax_p = scalar(group[0], "s_max_phys_worst", int)   # certified s_max (== confound)
    block  = smax_p + 1                                   # block width used in ca_reductions

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)

    # The analytic MGS law, to show the markers sit on it: one norm for V.col(0),
    # then (j+1) dots and one norm for each of the m steps.
    mm = np.linspace(1, m.max() + 1, 200)
    ax.plot(mm, 1 + mm * (mm + 3) / 2, "-", color=C_MGS, lw=1.4, alpha=0.5, zorder=1)

    ax.plot(m, mgs, "o", color=C_MGS, ms=7, zorder=3,
            label=r"MGS baseline: $1 + m(m{+}3)/2$")
    ax.plot(m, ca, "s-", color=C_CA, lw=1.8, ms=6, zorder=3,
            label=f"s-step + CholQR2: $1 + 2\\lceil m/(s_{{max}}{{+}}1)\\rceil$, "
                  f"$s_{{max}}={smax_p}$ (block {block})")

    # The real problem's measured Krylov window.
    ax.axvspan(8, 11, color=C_BAND, zorder=0,
               label="real operator's measured $m$ = 8-11")

    ax.set_xlabel("measured Krylov dimension $m$")
    ax.set_ylabel("global reductions per Arnoldi cycle")
    fs.style_axes(ax)
    fs.legend(ax, loc="upper left")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    # Against the left axis: fewer reductions per cycle is the win. (The twin axis
    # opposite carries the ratio, where the sense is inverted.) Mid-height on the left
    # edge: the corner below it is where the MGS law enters the axes.
    mark_better(ax, "down", at=(0.045, 0.45))

    # Speedup on a twin axis: the size of the horizontal mechanism's lever.
    ax2 = ax.twinx()
    ax2.plot(m, mgs / ca, "^--", color=C_GUIDE, lw=1.0, ms=5, alpha=0.8)
    ax2.set_ylabel("reduction-count cut (x)", color=C_GUIDE)
    ax2.tick_params(axis="y", colors=C_GUIDE)
    ax2.set_ylim(0, (mgs / ca).max() * 1.35)
    for x, y in zip(m, mgs / ca):
        ax2.annotate(f"{y:.1f}x", xy=(x, y), xytext=(0, 6),
                     textcoords="offset points", fontsize=7.5,
                     color=C_GUIDE, ha="center")

    out = HERE / "regime_reductions.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")


# Normality is load-bearing; the prediction error is bounded by kappa(X).
def plot_nonnormality(runs) -> None:
    # 2D baseline runs, untouched spectrum, across every knob combination.
    group = [r for k, r in runs.items()
             if k[0] == "24" and k[1] == "2" and float(k[2]) == 1.0 and float(k[3]) == 0.0]
    if not any(scalar(r, "henrici") > 1e-9 or scalar(r, "correlation") != 0
               for r in group):
        print("(skip) regime_nonnormality.png - no advection/correlation runs in the CSV")
        return

    def classify(r):
        # The real BS operator is the transfer target, not a member of a synthetic law
        # family. Its synthetic knob columns are all zero even though it carries
        # mixed-derivative cross terms, so it would otherwise be mislabeled
        # "constant-coeff advection" and folded into that fit. Its own class and mechanism
        # "real" is absent from `fam` below, so it is placed on the figure but excluded
        # from every law fit.
        if scalar(r, "real_bs") != 0.0:
            return ("real Black-Scholes (transfer)", "real", C_REAL, "*")
        henr = scalar(r, "henrici")
        # Hue is the mechanism family, since that is what the fits are taken
        # over; marker separates the two constant-coefficient constructions,
        # which previously shared both channels and so could not be told apart.
        if henr <= 1e-9:
            return ("normal (prediction exact)", "normal", C_PRED, "o")
        if scalar(r, "var_advection") != 0.0:
            return ("variable-coeff advection", "var", C_MEAS, "o")
        if scalar(r, "correlation") != 0.0:
            return ("$\\rho+\\gamma$ (const-coeff)", "const", C_CA, "D")
        return ("constant-coeff advection", "const", C_CA, "s")

    # gap in decades at a fixed s, from the recorded kappa columns.
    def gap_at(run, s_target=6):
        for r in run:
            if int(r["s"]) == s_target and r["kappa_trusted"] == "1":
                kp, km = float(r["kappa_predicted"]), float(r["kappa_measured"])
                if kp > 0 and km > 0:
                    return abs(np.log10(km) - np.log10(kp))
        return None

    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)

    # Prediction error (decades) vs log10(kappa(X)). The rigorous two-sided bound is the
    # y=x line (gap <= log10 kappa(X)); the data lies FAR under it and fits a shallow line
    # gap ~ slope * log10(kappa(X)) -- logarithmic, not linear, so the spectral prediction
    # survives deep into the non-normal regime.
    fam = {"const": ([], []), "var": ([], [])}   # per-MECHANISM points for separate fits
    real_gaps, all_gaps = [], []
    seen = set()
    n_canon, n_dense = 0, 0
    for run in sorted(group, key=lambda r: kappa_X_of(r)[0]):
        g = gap_at(run)
        if g is None:
            continue
        name, mech, col, mark = classify(run)
        kx, canonical = kappa_X_of(run)
        n_canon += int(canonical)
        n_dense += int(not canonical)
        lx = np.log10(max(kx, 1.0))
        # A star for the real operator, so it reads as a placed test point rather than one
        # more sample in a synthetic family. Larger and drawn on top.
        is_real = (mech == "real")
        all_gaps.append(g)
        if is_real:
            real_gaps.append(g)
        ax.scatter([lx], [g], color=col, s=170 if is_real else 48,
                   marker="*" if is_real else mark,
                   edgecolor="#222" if is_real else "none", linewidth=0.6,
                   zorder=4 if is_real else 3,
                   label=name if name not in seen else None)
        seen.add(name)
        if mech in fam and lx > 0:   # "real" is not a key, so it never enters a fit
            fam[mech][0].append(lx); fam[mech][1].append(g)

    lxmax = max([8.0] + fam["const"][0] + fam["var"][0])
    xline = np.linspace(0, lxmax, 50)
    ax.plot(xline, xline, "--", color=C_LIMIT, lw=1.4)
    ax.fill_between(xline, xline, lxmax + 1, color="#f2ece0", alpha=0.6, zorder=0)
    # Fit each mechanism separately: the two slopes differ by ~5x, so the "law" is
    # scoped to the constant-coefficient family, not mechanism-independent.
    slopes = {}
    for mech, col, lab in (("const", C_CA, "const-coeff"), ("var", C_MEAS, "variable-coeff")):
        xf, yf = fam[mech]
        if len(xf) >= 2:
            sl = float(np.linalg.lstsq(np.array(xf)[:, None], np.array(yf), rcond=None)[0][0])
            slopes[mech] = sl
            ax.plot(xline, sl * xline, "-", color=col, lw=2.0)
            # Named on the line. Two more legend rows for curves that are far
            # apart at the right-hand edge would only crowd the key.
            fs.label_series(ax, xline[-1], sl * xline[-1],
                            f"{lab} fit, slope {sl:.3f}", col,
                            dy=6.0, dx=-6.0, ha="right", size=8.0)
    ax.axhline(1.2, color=fs.C_INK, lw=1.0, ls=":", zorder=1)
    ax.set_xlim(-0.3, lxmax + 0.5)
    ax.set_ylim(-0.1, min(lxmax + 0.5, 3.2))
    ax.set_xlabel(r"$\log_{10}\kappa(X)$   (canonical tensor basis where the spectrum "
                  r"is degenerate)")
    ax.set_ylabel(r"prediction error  $|\log_{10}\kappa_{meas}-\log_{10}\kappa_{pred}|$")
    fs.style_axes(ax)
    # The rigorous bound climbs through the upper left, so the key sits on an
    # opaque ground rather than being crossed by it.
    fs.legend(ax, loc="upper left", opaque=True)
    fs.label_value(ax, ax.get_xlim()[1], 1.2,
                   r"one $s$-step ($\approx$1.2 decades)", dy=5.0, dx=-4.0,
                   ha="right", size=7.8, color=fs.C_INK)
    fs.label_value(ax, 2.4, 2.4, r"rigorous bound: gap $\leq\log_{10}\kappa(X)$",
                   dy=6.0, dx=6.0, ha="left", size=7.8, color=C_LIMIT)
    mark_better(ax, "down", loc="lower right")

    out = HERE / "regime_nonnormality.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")

    # The numbers the caption quotes. Printed, not drawn.
    ratio = (slopes.get("var", 0) / slopes["const"]) if slopes.get("const") else 0
    print(f"  const-coeff slope    : {slopes.get('const', 0):.3f}")
    print(f"  variable-coeff slope : {slopes.get('var', 0):.3f}  (~{ratio:.0f}x steeper)")
    print(f"  x-axis: {n_canon} point(s) canonical, {n_dense} dense")
    if real_gaps:
        print(f"  real Black-Scholes gap: {min(real_gaps):.2f} to "
              f"{max(real_gaps):.2f} decades, against a one-step budget of 1.2")
    above = [g for g in all_gaps if g > 1.2]
    print(f"  {len(above)} of {len(all_gaps)} sampled points exceed one s-step")
    if n_dense and not n_canon:
        print("  (warn) no kappa_X_struct column anywhere: this CSV predates it, so the "
              "constant-\n         coefficient arm is fitted against a degenerate, "
              "basis-dependent x-axis.\n         Re-run the control sweeps. The const slope "
              "shifts by ~10% either way\n         depending on which points are in the fit; "
              "the ~4x separation is robust.")


# Operator properties, for reference when writing the caption. Normality is what the
# spectral prediction needs; definiteness and separability are conveniences.
#
#   operator             normal?  def?   separ?  spectral prediction
#   Laplacian            yes      PSD    yes     exact (closed form)
#   shift mu             yes      indef  yes     exact (kappa(X)=1)
#   correlation rho      yes      PSD    no      exact (dense eig)
#   const-gamma adv      no       -      yes     survives (<1 step)
#   rho+gamma            no       -      no      survives (<1 step)
#   variable-gamma adv   no       -      no      measure (>1 step)

# The timed regime map and the real operator's trajectory.
#
# The (R_h, R_v) plane, four quadrants split by the predicted boundaries theta_v = 1 and
# theta_h = 1. Dense real-BS controls anchor the original low-grid continuation. Direct
# placement-only measurements at the production grids are drawn separately: they preserve
# the qualitative region but show that the low-grid m(n) fit understates both coordinates.

# machine.hpp amd-3960x, mirrored so the plot can place points the binary cannot reach.
_SLICE, _CPS, _PEAK, _DRAM, _L3S = 16 * 1024**2, 3, 68.12, 95.0, 58.19
_T_INTRA, _T_CROSS, _LSTAR = 120.8e-9, 652.8e-9, 2.20


def _slices(P):  return -(-P // _CPS)
def _agg(P):     return _slices(P) * _SLICE
def _bw(P, ws):  return _L3S * _slices(P) if ws <= _agg(P) else _DRAM
def _att(P, ai, b):  return min(_PEAK * P, ai * b) * 1e9


def _red_cost(P):
    if P <= 1:
        return 0.0
    lv = [d for d in (1, 2, 4, 8, 16, 32) if d < P]
    intra = sum(1 for d in lv if d < _CPS)
    cross = sum(1 for d in lv if d >= _CPS)
    return _T_INTRA * intra + _T_CROSS * min(cross, _LSTAR)


def place_rv_rh(P, nnz, N, m, x_reuse=1.0):
    """R_v, R_h for a banded operator -- the validated replica of regime.hpp's place()."""
    arn_ws = nnz * 12 + (N + 1) * 4 + (m + 1) * N * 8 + N * 8
    rv = arn_ws / _agg(P)
    spmv_ws = nnz * 12 + (N + 1) * 4 + N * 16
    mgs_ws = (m + 1) * N * 8 + N * 8
    spmv_ai = 2 * nnz / (nnz * 12 + (N + 1) * 4 + N * 8
                         + x_reuse * N * 8 + (1 - x_reuse) * nnz * 8)
    mgs_f = 3 * N + 2 * N * m * (m + 1) + 3 * N * m
    mgs_b = 8 * N * (m * (m + 1) // 2) + 24 * N * m + 24 * N
    mgs_ai = mgs_f / mgs_b
    cyc = (2 * nnz * m / _att(P, spmv_ai, _bw(P, spmv_ws))
           + mgs_f / _att(P, mgs_ai, _bw(P, mgs_ws)))
    rmgs = 1 + m * (m + 3) // 2
    rh = _red_cost(P) / (cyc / rmgs) if cyc > 0 else 0.0
    return rv, rh


def plot_trajectory(runs) -> None:
    P = 24
    real = sorted((r for k, r in runs.items() if k[9] != "0"),
                  key=lambda run: scalar(run, "N"))
    if not real:
        print("(skip) regime_trajectory.png - no real-BS runs (run PHASE 5)")
        return

    # Measured anchor, straight from the CSV's R_v/R_h columns.
    m_rh = np.array([scalar(run, "R_h") for run in real])
    m_rv = np.array([scalar(run, "R_v") for run in real])
    m_n = np.array([scalar(run, "real_bs", int) for run in real])

    placement = load_csv(PLACEMENT_CSV) if PLACEMENT_CSV.exists() else []
    placement.sort(key=lambda r: int(r["n"]))
    q_n = np.array([int(r["n"]) for r in placement])
    q_rv = np.array([float(r["R_v_measured"]) for r in placement])
    q_rh = np.array([float(r["R_h_measured"]) for r in placement])

    # Predictor validation: the replica must reproduce the measured placement before it is
    # trusted to extrapolate. Loudly flag any drift rather than draw a wrong continuation.
    worst = 0.0
    for run in real:
        rv, rh = place_rv_rh(P, scalar(run, "nnz", int), scalar(run, "N", int),
                             scalar(run, "m_measured", int))
        worst = max(worst, abs(rv / scalar(run, "R_v") - 1), abs(rh / scalar(run, "R_h") - 1))
    if worst > 0.02:
        print(f"  WARNING: trajectory predictor drifts {worst:.0%} from measured R_v/R_h; "
              "the continuation is NOT trustworthy -- check machine.hpp constants.")

    # m(n) fit from the measured grids, floored at the largest measured m, for the
    # production continuation. nnz -> the interior stencil 2d^2+1 = 19 at d=3.
    pm = np.polyfit(np.log(m_n), np.log([scalar(r, "m_measured", int) for r in real]), 1)
    m_of = lambda n: max(int(round(math.exp(np.polyval(pm, math.log(n))))),
                         int(max(scalar(r, "m_measured", int) for r in real)))
    prod_n = np.array([25, 30, 40, 50, 61, 74, 90, 120])
    p_rv, p_rh = [], []
    for n in prod_n:
        N = int(n) ** 3
        rv, rh = place_rv_rh(P, 19 * N, N, m_of(int(n)))
        p_rv.append(rv); p_rh.append(rh)
    p_rv, p_rh = np.array(p_rv), np.array(p_rh)

    # Original theta_v crossing (R_v = 1), by bisection on the low-grid fit.
    lo, hi = 40.0, 150.0
    for _ in range(50):
        mid = (lo + hi) / 2
        N = int(mid) ** 3
        rv, _ = place_rv_rh(P, 19 * N, N, m_of(int(mid)))
        lo, hi = (mid, hi) if rv < 1 else (lo, mid)
    n_cross = mid

    old_n_cross = mid

    # The updated trajectory uses direct measurements. Interpolate only to place the
    # visual crossing marker; the prose reports the observed bracket, not this estimate as
    # another measured grid.
    measured_cross = None
    for i in range(len(q_n) - 1):
        if q_rv[i] < 1.0 <= q_rv[i + 1]:
            t = -math.log(q_rv[i]) / math.log(q_rv[i + 1] / q_rv[i])
            measured_cross = (
                math.exp(math.log(q_n[i]) + t * math.log(q_n[i + 1] / q_n[i])),
                math.exp(math.log(q_rh[i]) + t * math.log(q_rh[i + 1] / q_rh[i])),
                int(q_n[i]), int(q_n[i + 1]),
            )
            break

    fig, ax = plt.subplots(figsize=(8.2, 6.6), constrained_layout=True)
    xlo, xhi, ylo, yhi = 1.5e-3, 20.0, 1.2e-3, 10.0

    # Upper-Right: the contested corner, unreachable on this machine. Caption sits low in
    # the shaded box so it clears the legend in the top-right corner.
    ax.fill_between([1.0, xhi], 1.0, yhi, color=C_BAND, zorder=0)
    ax.text(4.5, 1.35, "UPPER-RIGHT\nboth mechanisms\nunreachable on puffin",
            ha="center", va="center", fontsize=8.5, color="#7a6a3a", style="italic")
    # The other three quadrant labels.
    ax.text(0.06, 3.2, "UPPER-LEFT\nvertical only\n(matrix-powers)",
            ha="center", va="center", fontsize=8.5, color=C_GUIDE)
    ax.text(0.06, 3.6e-3, "LOWER-LEFT\nneither", ha="center", va="center",
            fontsize=8.5, color=C_GUIDE)
    ax.text(5.5, 3.6e-3, "LOWER-RIGHT\nhorizontal only\n(s-step)",
            ha="center", va="center", fontsize=8.5, color=C_GUIDE)

    ax.axhline(1.0, color=C_LIMIT, lw=1.3, ls="--", zorder=1)
    ax.axvline(1.0, color=C_LIMIT, lw=1.3, ls="--", zorder=1)
    ax.text(xlo * 1.3, 1.15, r"$\theta_v = 1$", color=C_LIMIT, fontsize=9)
    ax.text(1.15, ylo * 1.3, r"$\theta_h = 1$", color=C_LIMIT, fontsize=9)

    # Preserve the original low-grid extrapolation so its bias remains visible.
    ax.plot(p_rh, p_rv, "--", color=C_REAL, lw=1.6, zorder=3)
    ax.scatter(p_rh, p_rv, s=40, facecolor="white", edgecolor=C_REAL,
               linewidth=1.4, zorder=4, label="original low-grid extrapolation")

    # Dense-control measurements.
    ax.plot(m_rh, m_rv, "-", color=C_REAL, lw=1.0, alpha=0.6, zorder=3)
    ax.scatter(m_rh, m_rv, s=55, marker="o", color=C_REAL, edgecolor="#222",
               linewidth=0.5, zorder=5, label="measured, dense control")
    ax.annotate("n=10", (m_rh[0], m_rv[0]), textcoords="offset points",
                xytext=(7, -3), fontsize=7.5, color="#7a6a3a")
    ax.annotate("n=20", (m_rh[-1], m_rv[-1]), textcoords="offset points",
                xytext=(7, -3), fontsize=7.5, color="#7a6a3a")

    # Placement-only measurements replace the extrapolation over the production range.
    if len(q_n):
        ax.plot(q_rh, q_rv, "-", color=C_MEAS, lw=1.8, zorder=4)
        ax.scatter(q_rh, q_rv, s=75, marker="s", color=C_MEAS, edgecolor="#222",
                   linewidth=0.5, zorder=6, label="measured, placement-only")
        for n, x, y in zip(q_n, q_rh, q_rv):
            if n in (25, 61, 120):
                offset = (-7, 7) if n == 120 else (6, -9)
                ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                            xytext=offset, fontsize=7.5, color=C_MEAS,
                            ha="right" if n == 120 else "left")

    if measured_cross is not None:
        nc, rhc, nlo, nhi = measured_cross
        ax.scatter([rhc], [1.0], s=70, marker="D", color=C_MEAS, zorder=7,
                   label=fr"measured $\theta_v$ bracket: ${nlo}<n<{nhi}$")
    else:
        N = int(old_n_cross) ** 3
        _, rhc = place_rv_rh(P, 19 * N, N, m_of(int(old_n_cross)))
        ax.scatter([rhc], [1.0], s=70, marker="D", color=C_MEAS, zorder=6,
                   label=fr"predicted $\theta_v$ crossing ($n\approx${old_n_cross:.0f}$)")

    # Direction of travel: n=10 -> n=120 already reads up-and-left; a small caption on the
    # low-R_h end names it without a line that would double the trajectory.
    # The direction of travel, set clear of the predicted trajectory it
    # describes rather than beside the n=120 label at the same end of it.
    ax.text(xlo * 1.6, ylo * 2.2, "finer grid $\\longrightarrow$", fontsize=8,
            color=C_REAL, ha="left", va="bottom", alpha=0.9)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_xlabel(r"$R_h$  (reduction cost / compute between reductions)")
    ax.set_ylabel(r"$R_v$  (working set / aggregate L3)")
    fs.style_axes(ax)
    fs.legend(ax, loc="upper right")

    out = HERE / "regime_trajectory.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")


def main() -> None:
    if not CONTROL_CSV.exists():
        print(f"(skip) {CONTROL_CSV} not found - run scripts/regime/regime_control.sh")
        return

    rows = load_csv(CONTROL_CSV)
    if not rows:
        print(f"(skip) {CONTROL_CSV} is empty")
        return

    failed = {r["all_passed"] for r in rows} - {"1"}
    if failed:
        print("WARNING: the CSV contains rows from a run whose control gate FAILED. "
              "Re-run scripts/regime/regime_control.sh.")

    runs = group_runs(rows)
    print(f"{len(rows)} rows, {len(runs)} runs")

    plot_reductions(runs)
    plot_nonnormality(runs)
    plot_trajectory(runs)


if __name__ == "__main__":
    main()
