"""
The s(n) figure: the two roofs on the block width, and which one binds.

Consumes data/regime/regime_swidth.csv (scripts/regime/regime_swidth.sh).

s is pushed up by both mechanisms (wider block = fewer reductions and more powers per
operator-stream) and pulled down by two independent ceilings:

  numerical  kappa([v, ..., A^s v]) leaving the CholeskyQR certificate u^(-1/2)
  capacity   the s*w halo shrinking the panel until none fits and the tiling collapses

Left panel: kappa(B_s) against the certificate, with the numerical roof marked.
Right panel: measured reuse (tiled/baseline) against s, with the capacity roof marked.
Reuse climbs while a panel fits and collapses to ~1 when the halo pushes the tiling over.
The binding roof is whichever arrives first.
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

from figstyle import mark_better

HERE = Path(__file__).resolve().parent
CSV = HERE.parent.parent / "data" / "regime" / "regime_swidth.csv"

C_KAPPA = "#534AB7"
C_REUSE = "#0F6E56"
C_CERT  = "#993C1D"
C_GHOST = "#D85A30"


def main():
    if not CSV.exists():
        print(f"(skip) {CSV} not found - run scripts/regime/regime_swidth.sh on puffin")
        return
    with CSV.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("(skip) empty CSV")
        return

    # One curve per (tile_level, n1). Plot the widest grid at each level.
    groups = {}
    for r in rows:
        groups.setdefault((r["tile_level"], int(r["n1"])), []).append(r)
    for v in groups.values():
        v.sort(key=lambda r: int(r["s"]))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), constrained_layout=True)

    for (level, n1), g in sorted(groups.items()):
        s = np.array([int(r["s"]) for r in g])
        kappa = np.array([float(r["kappa_Bs"]) for r in g])
        reuse = np.array([float(r["reuse"]) for r in g])
        s_cert = int(g[0]["s_certified"])
        s_ghost = int(g[0]["s_ghost_max"])
        lab = f"{level.upper()}, n1={n1}"

        axes[0].semilogy(s, kappa, "-o", ms=3.5, lw=1.3, label=lab)
        axes[1].plot(s, reuse, "-o", ms=3.5, lw=1.3, label=f"{lab}  (roofs: cert {s_cert}, ghost {s_ghost})")

    # The certificate: kappa <= u^(-1/2). Derived exactly as regime.hpp does
    # (kUnitRoundoff = eps/2), so the line cannot drift from the C++ constant.
    cert_limit = 1.0 / np.sqrt(np.finfo(float).eps / 2.0)
    axes[0].axhline(cert_limit, color=C_CERT, ls="--", lw=1.4,
                    label=r"CholeskyQR certificate $u^{-1/2}$")
    axes[0].set_xlabel("block width $s$ (highest power per block)")
    axes[0].set_ylabel(r"$\kappa([v, Av, \ldots, A^s v])$")
    axes[0].set_title("numerical roof", fontsize=10)
    axes[0].grid(alpha=0.25, lw=0.5, which="both")
    axes[0].legend(fontsize=7.5, loc="lower right")
    # Lower conditioning keeps more block widths under the certificate.
    mark_better(axes[0], "down", loc="upper left")

    axes[1].axhline(1.0, color="#999", ls=":", lw=1.0, label="break-even (reuse = 1)")
    axes[1].set_xlabel("block width $s$ (highest power per block)")
    axes[1].set_ylabel("reuse: baseline / tiled wall-clock")
    axes[1].set_title("capacity roof", fontsize=10)
    axes[1].grid(alpha=0.25, lw=0.5)
    axes[1].legend(fontsize=7.5, loc="upper left")
    mark_better(axes[1], "up", loc="lower right")

    out = HERE / "regime_swidth.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")

    # The verdict, printed for the caption.
    for (level, n1), g in sorted(groups.items()):
        sc, sg = int(g[0]["s_certified"]), int(g[0]["s_ghost_max"])
        binds = ("certificate (numerical)" if sc < sg
                 else "ghost (capacity)" if sg < sc else "both together")
        print(f"  {level.upper()} n1={n1}: s_certified={sc}  s_ghost_max={sg}"
              f"  -> binding roof: {binds}, operative s = {min(sc, sg)}")


if __name__ == "__main__":
    main()
