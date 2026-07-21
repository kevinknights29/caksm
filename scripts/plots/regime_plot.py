"""
Figures for the CA-KSM-EI regime study (puffin, AMD Threadripper 3960X).

This script consumes:
    data/regime/regime_control.csv    (scripts/regime/regime_control.sh)

Outputs:

  regime_conditioning.png   kappa([v, Av, ..., A^s v]) vs s. The a-priori prediction
                            (row-scaled Vandermonde from the analytic spectrum) as a
                            line, the measured condition number as markers. Three panels:
                            scaffold dimension, the spectrum scale knob, and the spectrum
                            shift knob. The CholeskyQR certificate u^(-1/2) and each
                            panel's certified s_max are marked.

  regime_confound.png       The confound control. Left: s_max against the measured Krylov
                            dimension m, at a fixed spectrum. Right: s_max against basis
                            conditioning, at a fixed m.

  regime_reductions.png     Global reductions per Arnoldi cycle vs m: MGS's 1 + m(m+3)/2
                            against the stable s-step arm's 1 + 2*ceil(m/s_max). The real
                            problem's measured m = 8-11 window is shaded.

  regime_nonnormality.png   Prediction error in decades vs log10 kappa(X), fitted per
                            mechanism against the rigorous y=x bound.

  regime_trajectory.png     The timed (R_v, R_h) regime map: four quadrants split by
                            theta_v=1 and theta_h=1, with the real Black-Scholes
                            operator's grid-refinement trajectory. Measured phase-5 points
                            anchor it; a predicted continuation (the same a-priori
                            placement the binary uses, validated against the measured
                            R_v/R_h first) carries it to production grids. The operator
                            refines Lower-Right -> Lower-Left -> Upper-Left, crossing
                            theta_v near n=73, and never enters Upper-Right. The measured
                            points read R_h > 1 only because the eigensolvable grids
                            (n <= 20) sit below any resolution a desk would price at, an
                            irrelevant regime rather than a grid where s-step pays.

Figures 1-4 are arithmetic, not timing: no wall-clock, no dependence on the machine.
regime_trajectory.png is the exception, since it is the map: it places points using the
calibrated predictor coordinates (R_v from cache geometry, R_h from the calibrated
reduction cost). Those are still predictors computed before a run, not measured runtime
fractions, so the firewall holds; but the figure does license a claim about where CA pays,
which the other four deliberately do not.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from figstyle import mark_better

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data" / "regime"
CONTROL_CSV = DATA / "regime_control.csv"

# u^(-1/2): the CholeskyQR certificate. Below it, CholQR2 attains O(u) orthogonality.
KAPPA_LIMIT = 1.0 / math.sqrt(np.finfo(float).eps / 2.0)
# Above this, a *measured* kappa is itself roundoff-limited and says nothing.
TRUST_HORIZON = 1e12

# Palette, shared with scaling_plot.py where the roles line up.
C_PRED  = "#534AB7"   # a-priori prediction
C_MEAS  = "#D85A30"   # measurement
C_LIMIT = "#993C1D"   # the certificate
C_GUIDE = "#5F5E5A"
C_MGS   = "#B0483A"
C_CA    = "#0F6E56"
C_REAL  = "#C79A00"   # the real Black-Scholes operator (the transfer target)
C_BAND  = "#E8E4D9"

SERIES = ["#534AB7", "#D85A30", "#0F6E56", "#B0483A", "#7A7A73", "#993C1D"]


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


def kappa_at(run, s_target):
    """Predicted kappa at a given s, or None if the curve stopped short."""
    for r in run:
        if int(r["s"]) == s_target:
            return float(r["kappa_predicted"])
    return None

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


def fixed_m(runs, m=5):
    """Normal runs on the baseline 2D scaffold whose measured m is m: spectrum varies."""
    out = [r for k, r in runs.items()
           if k[0] == "24" and k[1] == "2" and normal_scaffold(k)
           and scalar(r, "m_measured", int) == m]
    return sorted(out, key=lambda run: kappa_at(run, m) or 0.0)


# Figure 1 - basis conditioning: prediction vs measurement
def plot_conditioning(runs) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True,
                             sharey=True)

    panels = [
        ("scaffold dimension",
         [r for k, r in runs.items()
          if float(k[2]) == 1.0 and float(k[3]) == 0.0 and float(k[4]) == 0.01
          and normal_scaffold(k)],
         lambda run: f"dim={scalar(run,'dim',int)}, N={scalar(run,'N',int)}"),

        ("spectrum scale $\\sigma$",
         [r for k, r in runs.items()
          if k[0] == "24" and k[1] == "2" and float(k[3]) == 0.0 and normal_scaffold(k)
          and float(k[2]) in (1.0, 2.0, 4.0, 8.0)
          and abs(float(k[4]) * float(k[2]) - 0.01) < 1e-9],
         lambda run: f"$\\sigma$={scalar(run,'scale'):g}, spread={scalar(run,'spread'):.1f}"),

        ("spectrum shift $\\mu$",
         [r for k, r in runs.items()
          if k[0] == "24" and k[1] == "2" and float(k[2]) == 1.0 and normal_scaffold(k)
          and float(k[4]) == 0.01 and float(k[5]) == 1e-8],
         lambda run: f"$\\mu$={scalar(run,'shift'):g}"),
    ]

    for ax, (title, group, label) in zip(axes, panels):
        group = sorted(group, key=lambda run: (scalar(run, "dim", int),
                                               scalar(run, "scale"),
                                               scalar(run, "shift")))
        for run, col in zip(group, SERIES):
            s    = np.array([int(r["s"]) for r in run])
            pred = np.array([float(r["kappa_predicted"]) for r in run])
            meas = np.array([float(r["kappa_measured"]) for r in run])
            trust = np.array([r["kappa_trusted"] == "1" for r in run])

            # Prediction: a continuous line, computed from the spectrum alone.
            ax.plot(s, pred, "-", color=col, lw=1.6, zorder=2)
            # Measurement: markers. Hollow where the measured kappa is itself
            # roundoff-limited and therefore says nothing.
            ax.plot(s[trust], meas[trust], "o", color=col, ms=5, zorder=3,
                    label=label(run))
            if (~trust).any():
                ax.plot(s[~trust], meas[~trust], "o", mfc="none", mec=col, ms=5,
                        zorder=3)

            smax = scalar(run, "s_max_phys_worst", int)
            ax.axvline(smax, color=col, lw=0.8, ls=":", alpha=0.7, zorder=1)

        ax.axhline(KAPPA_LIMIT, color=C_LIMIT, lw=1.4, ls="--", zorder=4)
        ax.axhspan(TRUST_HORIZON, 1e20, color="#f2f2f2", zorder=0)
        ax.set_yscale("log")
        ax.set_ylim(1e0, 1e18)
        ax.set_xlabel("matrix-powers block size $s$")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))  # s is a block size

        # The figure's own key: what the non-series marks mean. Proxies, because the
        # verticals are drawn once per run in that run's color.
        keys = [
            Line2D([], [], color=C_LIMIT, lw=1.4, ls="--",
                   label="CholQR certificate $u^{-1/2}$"),
            Line2D([], [], color=C_GUIDE, lw=0.8, ls=":",
                   label="certified $s_{max}$"),
            Patch(facecolor="#f2f2f2", label="measured $\\kappa$ roundoff-limited"),
        ]
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles + keys, fontsize=7, loc="lower right")
        mark_better(ax, "down", loc="upper left")

    axes[0].set_ylabel(r"$\kappa_2([v, Av, \ldots, A^s v])$")

    out = HERE / "regime_conditioning.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")


# Figure 2 - the confound control
def plot_confound(runs) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)

    # Left: m varies, spectrum FIXED. If safety tracked m, this would slope.
    ax = axes[0]
    group = fixed_spectrum(runs)
    m    = np.array([scalar(r, "m_measured", int) for r in group])
    smax = np.array([scalar(r, "s_max_pred_spectrum", int) for r in group])

    ax.plot(m, smax, "o-", color=C_CA, lw=1.8, ms=6, label="certified $s_{max}$")
    ax.axhspan(smax.min() - 0.25, smax.max() + 0.25, color=C_BAND, zorder=0)
    ax.plot(m, m, ":", color=C_GUIDE, lw=1.2, label="$s = m$ (one block)")

    ax.set_xlabel("measured Krylov dimension $m$   (moved by $h$ and tol)")
    ax.set_ylabel("certified $s_{max}$")
    ax.set_title("spectrum pinned, $m$ varies", fontsize=10)
    ax.set_ylim(0, max(m.max(), smax.max()) + 2)
    ax.grid(alpha=0.25, lw=0.5)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    # A block produces up to s_max+1 vectors, so a cycle needs >1 block only when
    # m > s_max+1 (not m > s_max).
    over = m > smax + 1
    if over.any():
        ax.axvline(m[over].min() - 0.5, color=C_LIMIT, lw=1.0, ls="--",
                   label="$m > s_{max}+1$ (cycle needs $>$1 block)")
    ax.legend(fontsize=8, loc="upper left")
    mark_better(ax, "up", loc="lower right")

    # Right: spectrum varies, m FIXED. This is where safety actually lives.
    ax = axes[1]
    group = fixed_m(runs, m=5)
    kap  = np.array([kappa_at(r, 5) for r in group], dtype=float)
    smax = np.array([scalar(r, "s_max_pred_spectrum", int) for r in group])
    tags = [f"$\\sigma$={scalar(r,'scale'):g}, $\\mu$={scalar(r,'shift'):g}"
            for r in group]

    ax.plot(kap, smax, "o-", color=C_MEAS, lw=1.8, ms=6)
    for x, y, t in zip(kap, smax, tags):
        ax.annotate(t, xy=(x, y), xytext=(0, 8), textcoords="offset points",
                    fontsize=7.5, ha="center", color=C_GUIDE)

    ax.set_xscale("log")
    ax.set_xlabel(r"$\kappa_2$ of the $m=5$ basis   (moved by the spectrum knobs)")
    ax.set_ylabel("certified $s_{max}$")
    ax.set_title("$m$ pinned at 5, spectrum varies", fontsize=10)
    ax.set_ylim(0, smax.max() + 3)
    # Pad both ends: the per-point labels sit above their markers and the outermost
    # ones would otherwise be clipped by the axes.
    ax.set_xlim(kap.min() / 4.0, kap.max() * 4.0)
    ax.grid(alpha=0.25, lw=0.5)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    mark_better(ax, "up", loc="lower left")

    out = HERE / "regime_confound.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")


# Figure 3 - the reduction ledger
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
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=9, loc="upper left")
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
    print(f"wrote {out}")


# Figure 4 - normality is load-bearing; the prediction error is bounded by kappa(X)
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
            return ("real Black-Scholes (transfer)", "real", C_REAL)
        henr = scalar(r, "henrici")
        if henr <= 1e-9:
            return ("normal (prediction exact)", "normal", C_PRED)
        if scalar(r, "var_advection") != 0.0:
            return ("variable-coeff advection", "var", C_MEAS)
        if scalar(r, "correlation") != 0.0:
            return ("$\\rho+\\gamma$ (const-coeff)", "const", C_CA)
        return ("constant-coeff advection", "const", C_CA)

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
    seen = set()
    for run in sorted(group, key=lambda r: scalar(r, "kappa_X")):
        g = gap_at(run)
        if g is None:
            continue
        name, mech, col = classify(run)
        lx = np.log10(max(scalar(run, "kappa_X"), 1.0))
        # A star for the real operator, so it reads as a placed test point rather than one
        # more sample in a synthetic family. Larger and drawn on top.
        is_real = (mech == "real")
        ax.scatter([lx], [g], color=col, s=170 if is_real else 48,
                   marker="*" if is_real else "o",
                   edgecolor="#222" if is_real else "none", linewidth=0.6,
                   zorder=4 if is_real else 3,
                   label=name if name not in seen else None)
        seen.add(name)
        if mech in fam and lx > 0:   # "real" is not a key, so it never enters a fit
            fam[mech][0].append(lx); fam[mech][1].append(g)

    lxmax = max([8.0] + fam["const"][0] + fam["var"][0])
    xline = np.linspace(0, lxmax, 50)
    ax.plot(xline, xline, "--", color=C_LIMIT, lw=1.4,
            label=r"rigorous bound  gap $\leq \log_{10}\kappa(X)$")
    ax.fill_between(xline, xline, lxmax + 1, color="#f2ece0", alpha=0.6, zorder=0)
    # Fit each mechanism separately: the two slopes differ by ~5x, so the "law" is
    # scoped to the constant-coefficient family, not mechanism-independent.
    slopes = {}
    for mech, col, lab in (("const", C_CA, "const-coeff"), ("var", C_MEAS, "variable-coeff")):
        xf, yf = fam[mech]
        if len(xf) >= 2:
            sl = float(np.linalg.lstsq(np.array(xf)[:, None], np.array(yf), rcond=None)[0][0])
            slopes[mech] = sl
            ax.plot(xline, sl * xline, "-", color=col, lw=2.0,
                    label=f"{lab} fit: slope {sl:.3f}")
    ax.axhline(1.2, color="#333", lw=1.0, ls=":", zorder=1,
               label="one $s$-step ($\\approx$1.2 decades)")
    ax.set_xlim(-0.3, lxmax + 0.5)
    ax.set_ylim(-0.1, min(lxmax + 0.5, 3.2))
    ax.set_xlabel(r"$\log_{10}\kappa(X)$   (unit-2-norm eigenvector conditioning)")
    ax.set_ylabel(r"prediction error  $|\log_{10}\kappa_{meas}-\log_{10}\kappa_{pred}|$")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7.5, loc="upper left")
    mark_better(ax, "down", loc="lower right")

    out = HERE / "regime_nonnormality.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")

    # The numbers the caption quotes. Printed, not drawn.
    ratio = (slopes.get("var", 0) / slopes["const"]) if slopes.get("const") else 0
    print(f"  const-coeff slope    : {slopes.get('const', 0):.3f}")
    print(f"  variable-coeff slope : {slopes.get('var', 0):.3f}  (~{ratio:.0f}x steeper)")


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

# Figure 5: the timed regime map and the real operator's trajectory.
#
# The (R_h, R_v) plane, four quadrants split by the predicted boundaries theta_v = 1 and
# theta_h = 1. The measured real-BS points anchor it; a predicted continuation carries the
# operator to production grids using the same predictor the binary places points with,
# validated to match the measured R_v/R_h to 1% before it is trusted to extrapolate.

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

    # theta_v crossing (R_v = 1), by bisection on n.
    lo, hi = 40.0, 150.0
    for _ in range(50):
        mid = (lo + hi) / 2
        N = int(mid) ** 3
        rv, _ = place_rv_rh(P, 19 * N, N, m_of(int(mid)))
        lo, hi = (mid, hi) if rv < 1 else (lo, mid)
    n_cross = mid

    fig, ax = plt.subplots(figsize=(8.2, 6.6), constrained_layout=True)
    xlo, xhi, ylo, yhi = 1.5e-3, 20.0, 1.2e-3, 6.0

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

    # Predicted continuation (a-priori placement, no eigensolve).
    ax.plot(p_rh, p_rv, "--", color=C_REAL, lw=1.6, zorder=3)
    ax.scatter(p_rh, p_rv, s=40, facecolor="white", edgecolor=C_REAL,
               linewidth=1.4, zorder=4, label="real BS, predicted (grid refinement)")
    for n, x, y in zip(prod_n, p_rh, p_rv):
        if n in (25, 61, 120):
            ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                        xytext=(6, -9), fontsize=7.5, color=C_REAL)

    # Measured anchor.
    ax.plot(m_rh, m_rv, "-", color=C_REAL, lw=1.0, alpha=0.6, zorder=3)
    ax.scatter(m_rh, m_rv, s=150, marker="*", color=C_REAL, edgecolor="#222",
               linewidth=0.6, zorder=5, label="real BS, measured (PHASE 5)")
    ax.annotate("n=10", (m_rh[0], m_rv[0]), textcoords="offset points",
                xytext=(7, -3), fontsize=7.5, color="#7a6a3a")
    ax.annotate("n=20", (m_rh[-1], m_rv[-1]), textcoords="offset points",
                xytext=(7, -3), fontsize=7.5, color="#7a6a3a")

    # theta_v crossing marker.
    N = int(n_cross) ** 3
    rvc, rhc = place_rv_rh(P, 19 * N, N, m_of(int(n_cross)))
    ax.scatter([rhc], [1.0], s=70, marker="D", color=C_MEAS, zorder=6,
               label=fr"$\theta_v$ crossing (n$\approx${n_cross:.0f}, N$\approx${N/1e3:.0f}k)")

    # Direction of travel: n=10 -> n=120 already reads up-and-left; a small caption on the
    # low-R_h end names it without a line that would double the trajectory.
    ax.text(3.5e-3, 5.0, "finer grid\n$\\longrightarrow$", fontsize=8, color=C_REAL,
            ha="left", va="top", alpha=0.9)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_xlabel(r"$R_h$  (reduction cost / compute between reductions)")
    ax.set_ylabel(r"$R_v$  (working set / aggregate L3)")
    ax.set_title("The timed regime map: where the real Black-Scholes operator travels\n"
                 "(AMD Threadripper 3960X / puffin, P=24)", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5, which="both")
    ax.legend(fontsize=7.8, loc="upper right", framealpha=0.95)

    out = HERE / "regime_trajectory.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")


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

    plot_conditioning(runs)
    plot_confound(runs)
    plot_reductions(runs)
    plot_nonnormality(runs)
    plot_trajectory(runs)


if __name__ == "__main__":
    main()
