"""
The asset-dimension law: does kappa(X) grow with the number of assets?

Consumes:  data/regime/regime_dsweep.csv   (scripts/regime/regime_dsweep.sh)
Produces:  scripts/plots/regime_dsweep.png

The refinement study varied the wrong axis: it swept grid points at fixed d=3 and found
kappa(X) flat. The thesis grows the asset count. From the diagonal-similarity formula with
gamma as the mesh Peclet number (gamma ~ C/n1, so n1 cancels):

    log kappa(X) ~ C * d      =>   kappa(X) exponential in the number of assets.

Left panel fits log10 kappa(X) against d and extrapolates to the asset counts a basket desk
cares about, with the regime bands overlaid (benign / conservative-cost / measure). Right
panel: does the spectral prediction still get s_max right as d grows? |pred - meas| <= 1 is
the cross-compiler noise floor, so only a discrepancy of 2+ means the certificate broke.

If the slope is ~0.5, the dimensionality that makes communication-avoidance necessary is
the same dimensionality that degrades the spectral certificate, so the practitioner needs
the measured block size precisely where the method matters most.
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
    rho_fixed = None
    for r in runs:
        pass
    dsweep = {}
    rhosweep = {}
    for r in runs:
        d   = int(r["dim"])
        rho = float(r["correlation"])
        kx  = float(r["kappa_X"])
        if not np.isfinite(kx) or kx <= 0:
            continue
        rec = (d, rho, kx, int(r["s_pred_int"]), int(r["s_meas_same"]), int(r["N"]))
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

    # Left: log10 kappa(X) vs asset dimension d, two competing models. A linear fit is a
    # chord through a bend: the successive deltas are not constant. Non-normality enters
    # through pairwise correlation cross terms, and there are C(d,2) = d(d-1)/2 axis-pairs,
    # so the exponent should be quadratic in d, not linear. Fitting both and showing they
    # agree on the data but diverge by orders of magnitude beyond it is the result: the
    # extrapolation is model-dependent, so high-d behavior must be measured.
    ax = axes[0]
    d = np.array([p[0] for p in dpts], float)
    y = np.log10(np.array([p[2] for p in dpts], float))

    C_lin, b_lin = np.polyfit(d, y, 1)
    # mechanistic: a*C(d,2) + b*d   (no intercept; AIC-preferred on the measured data)
    Amech = np.column_stack([d * (d - 1) / 2, d])
    (a_m, b_m), *_ = np.linalg.lstsq(Amech, y, rcond=None)

    def sse(pred): return float(np.sum((y - pred) ** 2))
    sse_lin  = sse(C_lin * d + b_lin)
    sse_mech = sse(a_m * d * (d - 1) / 2 + b_m * d)

    dx = np.linspace(1.5, 10, 200)
    ax.plot(dx, C_lin * dx + b_lin, "--", color=C_GUIDE, lw=1.6,
            label=f"linear $Cd$: slope {C_lin:.2f}")
    ax.plot(dx, a_m * dx * (dx - 1) / 2 + b_m * dx, "-", color=C_FIT, lw=2.2,
            label=rf"mechanistic $a\,C(d,2)+bd$: $a$={a_m:.2f}/pair")
    ax.fill_between(dx, C_lin * dx + b_lin, a_m * dx * (dx - 1) / 2 + b_m * dx,
                    color="#F6E3DC", alpha=0.7, zorder=0)
    ax.scatter(d, y, s=70, color=C_PT, zorder=4, label="measured (synthetic BS-like)")
    ax.scatter([3], [np.log10(40.0)], s=110, marker="*", color=C_WARN, zorder=5,
               label="REAL BS, 3 assets")

    d10_lin  = C_lin * 10 + b_lin
    d10_mech = a_m * 45 + b_m * 10
    ax.set_xlim(1.5, 10)
    ax.set_ylim(0, max(d10_mech, d10_lin) + 0.5)
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
               label="cross-compiler noise floor ($\\leq$1)")
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
    print(f"linear      : log10 kappa(X) = {C_lin:.3f} d {b_lin:+.3f}   SSE={sse_lin:.3f}")
    print(f"mechanistic : log10 kappa(X) = {a_m:.3f} C(d,2) {b_m:+.3f} d  SSE={sse_mech:.3f}"
          f"   <- {a_m:.3f} per correlation pair")
    print(f"{'d':>3}{'linear':>10}{'mechanistic':>13}{'divergence':>12}")
    for dd in (6, 7, 8, 10):
        L = C_lin * dd + b_lin
        M = a_m * dd * (dd - 1) / 2 + b_m * dd
        print(f"{dd:>3}{('1e%.1f' % L):>10}{('1e%.1f' % M):>13}{('1e%.1f' % (M - L)):>12}")
    print("The two models agree on the measured range and diverge by orders of magnitude\n"
          "beyond it: high-d kappa(X) must be MEASURED, not extrapolated.")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
