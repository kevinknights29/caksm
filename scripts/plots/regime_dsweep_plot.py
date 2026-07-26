"""
The asset-dimension law: how does kappa(X) grow with the number of assets?

Consumes:  data/regime/regime_dsweep.csv   (scripts/regime/regime_dsweep.sh)
Produces:  scripts/plots/regime_dsweep.png

The law is linear in d, with a slope known in closed form. build_synthetic couples the cross
term to axes 0 and 1 only, so the operator factors as A_d = B01 (+) T (+) ... (+) T and a
Kronecker product multiplies singular values:

    kappa(X) = kappa(X01) * kappa(X1)^(d-2)   =>   log kappa(X) = d log kappa(X1) + const

The cross term contributes a d-independent offset: one coupled pair at every d, never
C(d,2). An earlier revision fitted a quadratic a*C(d,2)+b*d against a linear Cd and reported
that the two diverge beyond the data; that comparison is withdrawn. The C(d,2) premise does
not describe this operator, and the curvature that motivated it was an artifact: from d=4
the d-2 cross-term-free axes are interchangeable, an exact symmetry of A, so the spectrum is
degenerate and the dense eigensolver reports its arbitrary basis choice within each repeated
eigenspace.

Left panel plots kappa_X_struct, the canonical tensor-basis value (regime_control's
structured_kappa_X), which is basis-independent at every d and needs one n1^2 x n1^2
eigensolve rather than one N x N. The dense kappa_X is shown hollow for contrast: it tracks
the structured value exactly while the spectrum is simple (d=2,3) and departs upward once
degeneracy sets in.

Right panel: does the spectral prediction still get s_max right as d grows? |pred - meas| is
judged at kSMaxStepTolerance, a chosen tolerance of one s-step rather than a measured
platform noise floor; see its note in regime_control_support.hpp.
"""
# /// script
# dependencies = ["matplotlib", "numpy"]
# ///

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from figstyle import mark_better

HERE = Path(__file__).resolve().parent
CSV = HERE.parent.parent / "data" / "regime" / "regime_dsweep.csv"

C_FIT   = "#D85A30"
C_PT    = "#534AB7"
C_RHO   = "#0F6E56"
C_GUIDE = "#5F5E5A"
C_WARN  = "#993C1D"

# kappa(X) regime bands. The earlier BENIGN_MAX = 1e3 was a synthetic-only estimate drawn
# too tight: the real BS operator at n1=20 reaches kappa(X) = 2304 (see
# regime_control.cpp real_bs_operator) and the spectral prediction of s_max is still
# integer-exact there (s_pred = s_meas = 5). So BENIGN_MAX is the measured floor, not a
# breakdown point; the true ceiling is above it and must be measured rather than read off
# the extrapolated non-normality fit.
BENIGN_MAX   = 2.3e3   # prediction integer-exact, measured to here on the real operator
CONSERV_MAX  = 1e5     # conservative certificate starts costing real block width


def load():
    if not CSV.exists():
        print(f"(skip) {CSV} not found - run scripts/regime/regime_dsweep.sh on the server")
        return None
    with CSV.open() as f:
        rows = list(csv.DictReader(f))
    # One row per s-step of the conditioning curve; collapse to one row per run.
    runs = {}
    for r in rows:
        key = (r["n1"], r["dim"], r["advection"], r["correlation"])
        runs.setdefault(key, r)
    return list(runs.values())


def main():
    runs = load()
    if not runs:
        return

    # The d-sweep: correlation held fixed, dim varying.
    dsweep = {}
    rhosweep = {}
    for r in runs:
        d   = int(r["dim"])
        rho = float(r["correlation"])
        kx  = float(r["kappa_X"])
        if not np.isfinite(kx) or kx <= 0:
            continue
        # Canonical tensor-basis kappa(X)
        try:
            kxs = float(r.get("kappa_X_struct", "nan"))
        except (TypeError, ValueError):
            kxs = float("nan")
        if not np.isfinite(kxs) or kxs <= 0:
            kxs = float("nan")
        rec = (d, rho, kx, int(r["s_pred_int"]), int(r["s_meas_same"]), int(r["N"]), kxs)
        # d-sweep family keyed by (n1, advection, rho) so the n1=4 asset sweep and the
        # n1=8 correlation stress never mix; rho-sweep keyed by (n1, advection, dim).
        dsweep.setdefault((r["n1"], r["advection"], rho), []).append(rec)
        rhosweep.setdefault((r["n1"], r["advection"], d), []).append(rec)

    # pick the family with the widest d coverage
    rho_star = max(dsweep, key=lambda k: len({x[0] for x in dsweep[k]}))
    dpts = sorted({x[0]: x for x in dsweep[rho_star]}.values())
    if len(dpts) < 2:
        print("(skip) need >=2 asset dimensions in the CSV")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.9), constrained_layout=True)

    # Left: log10 kappa(X) vs asset dimension d. The tensor factorization makes the law
    # linear with slope log10 kappa(X1), so the fit is a check on a known answer rather than
    # a model selection. The dense kappa_X is overlaid hollow: it agrees while the spectrum
    # is simple and departs upward once the interchangeable axes make it degenerate.
    ax = axes[0]
    d = np.array([p[0] for p in dpts], float)
    y_dense = np.log10(np.array([p[2] for p in dpts], float))
    k_struct = np.array([p[6] for p in dpts], float)
    have_struct = np.isfinite(k_struct)

    # Degeneracy onset: the cross term occupies axes 0 and 1, leaving d-2 interchangeable
    # axes. Two or more of them is an exact symmetry of A, hence a repeated spectrum.
    clean_axes = d - 2
    degenerate = clean_axes >= 2

    if have_struct.any():
        ds, ys = d[have_struct], np.log10(k_struct[have_struct])
        slope, icept = np.polyfit(ds, ys, 1) if len(ds) >= 2 else (np.nan, np.nan)
        dx = np.linspace(1.5, 10, 200)
        ax.plot(dx, slope * dx + icept, "-", color=C_FIT, lw=2.2,
                label=rf"$\log_{{10}}\kappa(X)={slope:.3f}\,d{icept:+.3f}$  (tensor law)")
        ax.scatter(ds, ys, s=70, color=C_PT, zorder=4,
                   label=r"canonical tensor basis $\kappa(X_{01})\kappa(X_1)^{d-2}$")
        ymax = slope * 10 + icept
    else:
        print("(warn) no kappa_X_struct column: CSV predates structured_kappa_X.")
        print("       Re-run the sweep; the dense kappa_X is basis-dependent from d=4.")
        slope = np.nan
        ymax = y_dense.max()

    # The dense eigensolver value, hollow. Split so the degenerate points are visibly a
    # different kind of measurement rather than more of the same curve.
    # Ringed, not overlaid: at d=2,3 the dense value equals the canonical one to 3e-7, and a
    # ring around the filled point is what shows that agreement rather than hiding it.
    if (~degenerate).any():
        ax.scatter(d[~degenerate], y_dense[~degenerate], s=260, facecolors="none",
                   edgecolors=C_GUIDE, lw=1.3, zorder=3,
                   label=r"dense $\kappa(X)$, simple spectrum (agrees)")
    if degenerate.any():
        ax.scatter(d[degenerate], y_dense[degenerate], s=90, facecolors="none",
                   edgecolors=C_WARN, lw=1.6, marker="^", zorder=3,
                   label=r"dense $\kappa(X)$, DEGENERATE (basis-dependent)")
        ax.axvline(3.5, color=C_WARN, ls=":", lw=1.2, zorder=1)
        ax.annotate("degeneracy onset\n($d-2\\geq2$ interchangeable axes)",
                    xy=(3.5, 0.06), xycoords=("data", "axes fraction"),
                    xytext=(4, 0.06), textcoords=("data", "axes fraction"),
                    fontsize=7, color=C_WARN, va="bottom")
        ymax = max(ymax, y_dense[degenerate].max())

    ax.scatter([3], [np.log10(40.0)], s=110, marker="*", color=C_WARN, zorder=5,
               label="REAL BS, 3 assets")

    ax.set_xlim(1.5, 10)
    ax.set_ylim(0, ymax + 0.5)
    ax.set_xlabel("asset dimension $d$   (the axis the thesis actually grows)")
    ax.set_ylabel(r"$\log_{10}\kappa(X)$")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7.5, loc="upper left")
    mark_better(ax, "down", loc="lower right")

    # ---- Right: the decision-relevant integer, vs d and vs rho ----
    ax = axes[1]
    diff_d = [abs(p[3] - p[4]) for p in dpts]
    ax.plot(d, diff_d, "o-", color=C_PT, lw=1.8, ms=7, label="vs asset dimension $d$")

    # correlation stress at the widest-covered fixed d
    d_star = max(rhosweep, key=lambda k: len({x[1] for x in rhosweep[k]}))
    rpts = sorted({x[1]: x for x in rhosweep[d_star]}.values(), key=lambda p: p[1])
    if len(rpts) >= 2:
        ax2 = ax.twiny()
        rr = np.array([p[1] for p in rpts])
        dr = [abs(p[3] - p[4]) for p in rpts]
        ax2.plot(rr, dr, "s--", color=C_RHO, lw=1.6, ms=6,
                 label=f"vs correlation $\\rho$ (d={d_star[2]})")
        ax2.set_xlabel(r"correlation $\rho$   (stressed basket: $\rho \to 1$)",
                       color=C_RHO)
        ax2.tick_params(axis="x", colors=C_RHO)
        ax2.legend(fontsize=8, loc="upper center")

    ax.axhspan(0, 1, color="#E4EFE9", zorder=0,
               label="chosen tolerance ($\\leq$1 s-step)")
    ax.set_ylim(0, max(3, max(diff_d) + 1))
    ax.set_xlabel("asset dimension $d$", color=C_PT)
    ax.tick_params(axis="x", colors=C_PT)
    ax.set_ylabel("$|s_{max}$ predicted $-$ measured$|$  (same vector)")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, loc="upper left")
    mark_better(ax, "down", loc="lower right")

    out = HERE / "regime_dsweep.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")

    if np.isfinite(slope):
        print(f"tensor law  : log10 kappa(X) = {slope:.4f} d {icept:+.4f}   "
              f"(slope = log10 kappa(X_1), analytic)")
        print(f"{'d':>3}{'structured':>14}{'dense':>14}{'ratio':>10}  spectrum")
        for p in dpts:
            dd, kx, kxs = p[0], p[2], p[6]
            simple = dd - 2 < 2
            ratio = f"{kx / kxs:.2f}x" if np.isfinite(kxs) and kxs > 0 else "--"
            sk = f"{kxs:.4g}" if np.isfinite(kxs) else "--"
            print(f"{dd:>3}{sk:>14}{kx:>14.4g}{ratio:>10}  "
                  f"{'simple' if simple else 'DEGENERATE (dense meaningless)'}")
        print("\nThe cross term couples ONE axis pair at every d, so it contributes a\n"
              "d-independent offset: the law is linear, not C(d,2)-quadratic. The dense\n"
              "kappa(X) tracks it until degeneracy at d=4, then reports basis choice.")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
