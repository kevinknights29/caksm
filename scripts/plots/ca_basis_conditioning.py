"""Basis conditioning against block width for monomial, Newton and Chebyshev.

Why monomial stays the production basis. The plotted condition number is the
largest the evolving solver reached, not a start-vector probe, against the
CholeskyQR certificate line kappa <= u^(-1/2).

Split by option as well as by grid, since the same basis at the same width can
differ by half a decade between Basket and Rainbow. What the solver will
actually certify is a separate figure, ca_certified_width.py.

Source: data/ca-integrator-certificate, produced by
scripts/regime/ca_certificate_sweep.sh.

  uv run scripts/plots/ca_basis_conditioning.py
"""
# /// script
# dependencies = ["matplotlib", "numpy"]
# ///
from __future__ import annotations

import matplotlib.pyplot as plt

from figstyle import mark_better
import ca_figstyle as ca
import ca_figlib as lib

FIGURE = lib.Figure("basis_conditioning")

BASES = ("monomial", "newton", "chebyshev")
# Chebyshev is drawn last and dashed, so Newton stays visible underneath it
# wherever the two coincide.
BASIS_LINESTYLE = {"monomial": "-", "newton": "-", "chebyshev": (0, (4, 2))}


def draw() -> str | None:
    predicted, measured, measured_by_key, reason = lib.load_certificate()
    if reason is not None:
        return FIGURE.blocked(reason)

    grids = sorted({int(r["n"]) for r in predicted})
    options = ("rainbow", "basket")
    limit = float(predicted[0]["limit"])
    widths = sorted({int(r["block_width"]) for r in predicted})

    fig, axes = plt.subplots(
        1, len(options) * len(grids),
        figsize=(3.6 * len(options) * len(grids), 4.6),
        constrained_layout=True, squeeze=False, sharey=True)
    rows = []

    for column, (option, n) in enumerate(
            [(o, g) for o in options for g in grids]):
        ax = axes[0][column]
        for basis in BASES:
            selected = sorted(
                (r for r in predicted
                 if int(r["n"]) == n and r["basis"] == basis
                 and r["option"] == option),
                key=lambda r: int(r["block_width"]))
            block_widths = [int(r["block_width"]) for r in selected]
            kappa = [
                float(measured_by_key[
                    (option, basis, n, int(r["block_width"]))
                ]["max_block_kappa"])
                for r in selected
            ]
            ax.semilogy(
                block_widths, kappa, marker="o", ms=4.2, lw=1.4,
                ls=BASIS_LINESTYLE[basis], color=ca.BASIS_COLOR[basis],
                zorder=4)
            for record, value in zip(selected, kappa):
                measured_record = measured_by_key[
                    (option, basis, n, int(record["block_width"]))]
                rows.append({
                    "n": n, "option": option, "basis": basis,
                    "block_width": int(record["block_width"]),
                    "recurrence_degree": int(record["recurrence_degree"]),
                    "evolving_max_block_kappa": value,
                    "block_basis_kappa": float(record["block_basis_kappa"]),
                    "certificate_limit": limit,
                    "fraction_of_certificate_limit": value / limit,
                    "evolving_min_certified_block_width":
                        int(measured_record["min_certified_block_width"]),
                    "unconverged": int(measured_record["unconverged"]),
                })

        ax.axhline(limit, color=ca.DIVERGING_LOW, ls="--", lw=1.4, zorder=3)
        ax.set_xticks(widths)
        ax.set_xlabel("requested block columns")
        lib.panel_title(ax, f"{option.capitalize()}, n={n}")
        lib.style_axes(ax)
        if column == 0:
            ax.set_ylabel(r"maximum evolving block-basis $\kappa$")
            mark_better(ax, "down", loc="lower right")
            lib.label_value(
                ax, ax.get_xlim()[0], limit,
                r"CholeskyQR certificate $u^{-1/2}$", dy=6.0, dx=4.0,
                ha="left", size=7.8, color=ca.DIVERGING_LOW)
            # Center right, not upper left: the certificate rule sits near the
            # top of a log axis that spans eight decades, and its label owns
            # that corner. The curves have reached the rule by the fourth
            # column, so the band below them on the right is empty.
            lib.legend(
                ax, loc="center right",
                handles=[
                    plt.Line2D(
                        [], [], marker="o", ms=5.0, lw=1.4,
                        ls=BASIS_LINESTYLE[basis],
                        color=ca.BASIS_COLOR[basis], label=basis)
                    for basis in BASES
                ])

    lib.title(fig, "Monomial stays the best-conditioned basis at every width")

    # No basis exceeds the limit: each flattens against it as the solver falls
    # back, so "crosses the certificate" is not a fact this artifact contains.
    # What it does contain is how close each basis comes and how much better
    # conditioned monomial is on the way there.
    def closest(basis: str) -> float:
        return max(
            r["fraction_of_certificate_limit"] for r in rows
            if r["basis"] == basis)

    advantage = max(
        max(r["evolving_max_block_kappa"] for r in rows
            if r["basis"] != "monomial" and r["n"] == n
            and r["block_width"] == width)
        / max(r["evolving_max_block_kappa"] for r in rows
              if r["basis"] == "monomial" and r["n"] == n
              and r["block_width"] == width)
        for n in grids for width in widths
    )
    print("  closest approach to the certificate: " + ", ".join(
        f"{basis} {closest(basis):.0%}" for basis in BASES)
        + f"; monomial is up to {advantage:.0f}x better conditioned")

    FIGURE.write(fig, rows)
    return None


if __name__ == "__main__":
    raise SystemExit(lib.run(FIGURE, draw))
