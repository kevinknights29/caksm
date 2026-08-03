"""How many block columns the evolving solver will actually certify.

The companion to ca_basis_conditioning.py, drawn from the same artifact. The
finding is the gap: the a-priori Gershgorin bound admits every requested column
while the solver certifies two to three fewer, and monomial survives one column
further than Newton or Chebyshev.

Not split by option, since every basis certifies the same count under both. Each
point is the minimum over the two options.

Source: data/ca-integrator-certificate, produced by
scripts/regime/ca_certificate_sweep.sh.

  uv run scripts/plots/ca_certified_width.py
"""
# /// script
# dependencies = ["matplotlib", "numpy"]
# ///
from __future__ import annotations

import matplotlib.pyplot as plt

from figstyle import mark_better
import ca_figstyle as ca
import ca_figlib as lib

FIGURE = lib.Figure("certified_width")

BASES = ("monomial", "newton", "chebyshev")
# Newton and Chebyshev certify exactly the same count. Chebyshev is drawn last
# and dashed, so Newton stays visible underneath it.
BASIS_LINESTYLE = {"monomial": "-", "newton": "-", "chebyshev": (0, (4, 2))}


def draw() -> str | None:
    predicted, measured, measured_by_key, reason = lib.load_certificate()
    if reason is not None:
        return FIGURE.blocked(reason)

    grids = sorted({int(r["n"]) for r in predicted})
    options = ("rainbow", "basket")
    widths = sorted({int(r["block_width"]) for r in measured})

    fig, axes = plt.subplots(
        1, len(grids), figsize=(5.6 * len(grids), 4.8),
        constrained_layout=True, squeeze=False, sharey=True)
    rows = []
    optimism: list[int] = []

    for column, n in enumerate(grids):
        ax = axes[0][column]
        ceiling = max(
            int(r["predicted_block_width_max"])
            for r in predicted if int(r["n"]) == n)
        promised = [min(width, ceiling) for width in widths]

        certified_by_basis = {}
        for basis in BASES:
            certified = []
            for width in widths:
                values = {
                    int(measured_by_key[
                        (option, basis, n, width)
                    ]["min_certified_block_width"])
                    for option in options
                    if int(measured_by_key[
                        (option, basis, n, width)]["unconverged"]) == 0
                }
                certified.append(min(values) if values else None)
            certified_by_basis[basis] = certified

        # The gap is the finding. Shading it against the weakest basis at each
        # width states the worst case the a-priori bound is wrong by.
        weakest = [
            min(certified_by_basis[basis][index]
                for basis in BASES
                if certified_by_basis[basis][index] is not None)
            for index in range(len(widths))
        ]
        ax.fill_between(
            widths, promised, weakest, color=ca.DIVERGING_LOW, alpha=0.12,
            lw=0, zorder=1)
        optimism.extend(p - w for p, w in zip(promised, weakest))

        ax.plot(
            widths, promised, color=ca.C_MUTED, ls="--", lw=1.4, marker="o",
            ms=4.5, markerfacecolor="none", zorder=3)
        for basis in BASES:
            ax.plot(
                widths, certified_by_basis[basis], marker="o", ms=4.5, lw=1.5,
                ls=BASIS_LINESTYLE[basis], color=ca.BASIS_COLOR[basis],
                zorder=4)

        # An unconverged arm is a diagnostic, never a certificate result.
        stopped = [
            (int(r["block_width"]), int(r["min_certified_block_width"]))
            for r in measured
            if int(r["n"]) == n and int(r["unconverged"]) != 0
        ]
        if stopped:
            ax.plot(
                [w for w, _ in stopped], [c for _, c in stopped],
                marker="x", ls="none", ms=6.5, mew=1.4, color=ca.C_STOPPED,
                zorder=5)

        for basis in BASES:
            for width, certified in zip(widths, certified_by_basis[basis]):
                rows.append({
                    "n": n, "basis": basis, "block_width": width,
                    "a_priori_certified_block_width": min(width, ceiling),
                    "a_priori_block_width_max": ceiling,
                    "evolving_certified_block_width": certified,
                    "a_priori_optimism_columns":
                        None if certified is None
                        else min(width, ceiling) - certified,
                })

        ax.set_xticks(widths)
        ax.set_yticks(widths)
        ax.set_xlabel("requested block columns")
        lib.panel_title(ax, f"n = {n}")
        lib.style_axes(ax)
        if column == 0:
            ax.set_ylabel("certified block columns\n(minimum over integration)")
            mark_better(ax, "up", loc="lower right")
            handles = [
                plt.Line2D(
                    [], [], marker="o", ms=5.0, lw=1.4,
                    ls=BASIS_LINESTYLE[basis], color=ca.BASIS_COLOR[basis],
                    label=basis)
                for basis in BASES
            ]
            handles.append(
                plt.Line2D(
                    [], [], marker="o", ms=5.0, lw=1.4, ls="--",
                    markerfacecolor="none", color=ca.C_MUTED,
                    label="a priori (Gershgorin) bound"))
            lib.legend(ax, loc="upper left", handles=handles)

    lib.title(fig, "Monomial stays closest to the a-priori Gershgorin bound")
    print(f"  a-priori bound over-promises by up to {max(optimism)} columns")

    FIGURE.write(fig, rows)
    return None


if __name__ == "__main__":
    raise SystemExit(lib.run(FIGURE, draw))
