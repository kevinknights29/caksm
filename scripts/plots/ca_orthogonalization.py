"""Orthogonality loss and cost for CholQR2, TSQR and two-pass block Gram-Schmidt.

Why the stable arms are rejected on cost and not on correctness. Every group is
one identical block, the same n, m, s, basis and option across all three
methods, since a method measured on an easier block would look both faster and
better conditioned.

Cost and correctness are separate panels because they answer on different
scales: cost as a bar chart, correctness as a strip against the solver's
tolerance, which all three methods clear by five or more orders of magnitude.

Source: data/ca-integrator-certificate-m39/orthogonalization.csv, produced by
scripts/regime/ca_certificate_sweep.sh.

  uv run scripts/plots/ca_orthogonalization.py
"""
# /// script
# dependencies = ["matplotlib", "numpy"]
# ///
from __future__ import annotations

import csv
import math

import matplotlib.pyplot as plt
import numpy as np

from figstyle import mark_better
import ca_figstyle as ca
import ca_figlib as lib

FIGURE = lib.Figure("orthogonalization")

METHODS = ("cholqr2", "tsqr", "bgs2")
METHOD_LABEL = {
    "cholqr2": "CholQR2",
    "tsqr": "TSQR",
    "bgs2": "BGS2",
}
# What the loss has to be small against. The integrator's Krylov stopping
# tolerance is the scale on which an orthogonality loss either matters or does
# not, so it is the reference the correctness panel is drawn against rather than
# an axis fitted to the losses themselves.
SOLVER_TOLERANCE = 1e-8
BAR = 0.26


def draw() -> str | None:
    path = lib.CERTIFICATE / "orthogonalization.csv"
    if not path.exists():
        return FIGURE.blocked(
            "no artifact in data/ carries orthogonality loss and basis time "
            "together for CholQR2, TSQR and BGS2. The final report states the "
            "rejection without recording the two numbers on identical blocks; "
            "run scripts/regime/ca_certificate_sweep.sh to produce them")

    with path.open() as handle:
        all_records = list(csv.DictReader(handle))
    timing_fields = {
        "option", "basis", "orth", "n", "m", "s",
        "orthogonality_loss", "basis_min_ms", "basis_ms", "basis_max_ms",
        "contended", "repeats",
    }
    if not all_records or not timing_fields <= set(all_records[0]):
        return FIGURE.blocked(
            "the orthogonalization artifact predates repeat distributions")

    idle = [r for r in all_records if r["contended"] == "0"]
    grouped: dict[tuple, dict[str, dict]] = {}
    for record in idle:
        key = (
            record["option"], record["basis"], int(record["n"]),
            int(record["m"]), int(record["s"]),
        )
        grouped.setdefault(key, {})[record["orth"]] = record
    required_methods = {"cholqr2", "tsqr", "bgs2"}
    expected_groups = {
        (option, "monomial", n, 8, 4)
        for option in ("basket", "rainbow")
        for n in (31, 61)
    }
    incomplete = [
        key for key in expected_groups
        if key not in grouped
        or set(grouped[key]) != required_methods
        or any(
            not grouped[key][method]["basis_ms"]
            or not grouped[key][method]["orthogonality_loss"]
            or int(grouped[key][method]["repeats"]) < 7
            for method in required_methods
        )
    ]
    extra_groups = set(grouped) - expected_groups
    if not grouped or incomplete or extra_groups:
        return FIGURE.blocked(
            f"{len(incomplete) if grouped else len(expected_groups)} required "
            "identical-block group(s) lack all three idle seven-repeat method "
            f"measurements; {len(extra_groups)} unexpected group(s) are present")
    groups_in_order = sorted(expected_groups)
    rows = []

    fig, axes = plt.subplots(
        1, 2, figsize=(12.4, 5.2), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.25, 1.0]})
    cost_ax, loss_ax = axes

    positions = np.arange(len(groups_in_order), dtype=float)
    cheapest: dict[tuple, float] = {}
    for key in groups_in_order:
        cheapest[key] = min(
            float(grouped[key][method]["basis_ms"]) for method in METHODS)

    for index, method in enumerate(METHODS):
        offset = (index - 1) * BAR
        for position, key in zip(positions, groups_in_order):
            record = grouped[key][method]
            median = float(record["basis_ms"])
            low = float(record["basis_min_ms"])
            high = float(record["basis_max_ms"])
            x = position + offset
            cost_ax.bar(
                [x], [median], BAR, color=ca.ORTH_COLOR[method],
                edgecolor="white", lw=0.6, zorder=3)
            cost_ax.errorbar(
                [x], [median],
                yerr=[[max(median - low, 0.0)], [max(high - median, 0.0)]],
                color=ca.C_INK, lw=0, elinewidth=0.9, capsize=2.5, zorder=5)
            # Against the cheapest method on the same block, which is the
            # comparison the rejection actually rests on.
            lib.label_value(
                cost_ax, x, median, f"{median / cheapest[key]:.2f}×",
                dy=6.0, size=7.6,
                color=ca.C_INK if median > cheapest[key] else ca.C_MUTED)

            loss = float(record["orthogonality_loss"])
            loss_ax.plot(
                [x], [loss], marker="o", ms=6.5,
                color=ca.ORTH_COLOR[method], markeredgecolor="white",
                markeredgewidth=0.8, lw=0, zorder=4)

            rows.append({
                "orth": method, "option": record["option"],
                "basis": record["basis"], "n": int(record["n"]),
                "m": int(record["m"]), "s": int(record["s"]),
                "basis_ms": median,
                "basis_min_ms": low,
                "basis_max_ms": high,
                "cost_over_cheapest_on_block": median / cheapest[key],
                "repeats": int(record["repeats"]),
                "orthogonality_loss": loss,
                "loss_over_solver_tolerance": loss / SOLVER_TOLERANCE,
            })

    tick_labels = [
        f"{option.capitalize()}\nn={n}, m={m}, s={s}"
        for option, _, n, m, s in groups_in_order
    ]
    for axis in (cost_ax, loss_ax):
        axis.set_xticks(positions)
        axis.set_xticklabels(tick_labels, fontsize=8.2)
        axis.set_xlim(positions[0] - 0.62, positions[-1] + 0.62)
        lib.style_axes(axis, grid_axis="y")

    cost_ax.set_ylabel("basis and H assembly (ms)")
    cost_ax.set_ylim(0.0, max(r["basis_ms"] for r in rows) * 1.18)
    lib.panel_title(
        cost_ax, "Cost, as a multiple of the cheapest method on each block")
    mark_better(cost_ax, "down", loc="upper left")
    lib.legend(
        cost_ax, loc="upper center", ncol=3,
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor=ca.ORTH_COLOR[method],
                          label=METHOD_LABEL[method])
            for method in METHODS
        ])

    # The correctness panel is drawn against the tolerance, not against the
    # losses. Fitting the axis to the losses turns a twenty-fold spread of
    # numbers that are all irrelevant into an apparent ranking.
    loss_ax.set_yscale("log")
    losses = [r["orthogonality_loss"] for r in rows]
    loss_ax.set_ylim(min(losses) / 4.0, SOLVER_TOLERANCE * 4.0)
    loss_ax.axhline(
        SOLVER_TOLERANCE, color=ca.DIVERGING_LOW, ls="--", lw=1.4, zorder=3)
    lib.label_value(
        loss_ax, loss_ax.get_xlim()[0], SOLVER_TOLERANCE,
        f"solver Krylov tolerance, {SOLVER_TOLERANCE:.0e}", dy=-12.0, dx=4.0,
        ha="left", size=7.8, color=ca.DIVERGING_LOW)
    loss_ax.set_ylabel(r"orthogonality loss $\|V^{T}V - I\|_F$")
    lib.panel_title(
        loss_ax, "Orthogonality loss against the solver's tolerance")
    mark_better(loss_ax, "down", loc="lower right")

    worst_ratio = max(r["cost_over_cheapest_on_block"] for r in rows)
    decades = math.log10(SOLVER_TOLERANCE / max(losses))
    print(f"  alternatives cost up to {worst_ratio:.2f}x CholQR2; every loss "
          f"is at least {math.floor(decades):.0f} decades under the tolerance")

    FIGURE.write(fig, rows)
    return None


if __name__ == "__main__":
    raise SystemExit(lib.run(FIGURE, draw))
