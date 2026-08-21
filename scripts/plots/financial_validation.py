"""Financial validation: the convergence figure and the compact thesis table.

Source: data/financial-validation, produced by the financial-reference and
financial-validation executables, and data/krylov-indicator-calibration for the
block-stability rows.

  ./build/financial-reference
  ./build/financial-validation
  ./build/krylov-indicator-calibration
  uv run scripts/plots/financial_validation.py

Draws the spatial refinement trend for both payoffs and both methods against the
independent references, and writes the compact table the thesis quotes.

The two methods land on the same spatial error at every grid, to within a third
of a percent, so one drawn over the other simply hides it. ADI-HV-S is the line
and KSM-EI is drawn as open markers on top of it: the agreement is then the
thing the panel shows rather than something the reader has to take on trust. The reference uncertainty is
drawn as a floor rather than left implicit: a PDE error plotted below it would
be reporting noise, and the figure should make that visible instead of leaving
it to the caption.

The temporal panel that used to sit beside this one is gone. It showed the
theta-damping of a non-smooth payoff, which is standard and well documented, and
it added a second subject to a figure that makes one claim.
"""
# /// script
# dependencies = ["matplotlib"]
# ///
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

import ca_figstyle as ca
import figstyle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "financial-validation"
CALIBRATION = ROOT / "data" / "krylov-indicator-calibration"
DPI = 300

NAME = "financial_validation"
PNG = HERE / f"{NAME}.png"
SIDECAR = HERE / f"{NAME}.blocked.txt"

PAYOFFS = ("basket", "rainbow")
# The methods coincide to plotting precision, so they cannot share a drawing
# style: the second would simply erase the first. ADI-HV-S carries the line and
# KSM-EI rides on it as open markers. Hue stays with the payoff, which is what
# separates the two trends on the axis.
METHOD_STYLE = {
    "ADI-HV-S": dict(ls="-", lw=1.5, ms=5.0),
    "KSM-EI": dict(ls="none", ms=8.0, markerfacecolor="none",
                   markeredgewidth=1.5),
}
PAYOFF_MARKER = {"basket": "o", "rainbow": "s"}


def blocked(reason: str) -> str:
    """Record the figure as blocked, naming the artifact it waits on."""
    PNG.unlink(missing_ok=True)
    with SIDECAR.open("w") as handle:
        handle.write(f"figure={NAME}\n")
        handle.write("status=blocked\n")
        handle.write(f"reason={reason}\n")
    print(f"  BLOCKED {NAME}: {reason}")
    return reason


def read(path: Path) -> list[dict]:
    """Read one CSV artifact, or return an empty list if it is absent."""
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def decade_ticks(ax) -> None:
    """One labeled tick per decade on the log value axis.

    Left to itself over five decades, matplotlib labels minor ticks too and the
    axis ends up with a repeated exponent.
    """
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=16))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10)),
                                          numticks=160))
    ax.yaxis.set_minor_formatter(NullFormatter())


def guide(ax, x, order: int, anchor: float, label: str, va: str = "top") -> None:
    """Draw a slope guide through one anchor point, offset clear of the data."""
    scaled = [anchor * (value / x[0]) ** order for value in x]
    ax.plot(x, scaled, color=ca.C_MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.annotate(
        label, (x[-1], scaled[-1]), textcoords="offset points",
        xytext=(-4, -14), fontsize=8, color=ca.C_MUTED, ha="right", va=va)


def draw_spatial(ax, space: list[dict], reference: dict) -> list[dict]:
    """Absolute price error against grid spacing, one series per payoff."""
    rows = []
    smallest = math.inf
    for payoff in PAYOFFS:
        for method in METHOD_STYLE:
            points = sorted(
                (float(r["dx_1"]), abs(float(r["error"])), int(r["n"]))
                for r in space
                if r["payoff"] == payoff and r["method"] == method)
            if len(points) < 2:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            smallest = min(smallest, min(ys))
            ax.plot(
                xs, ys, color=ca.OPTION_COLOR[payoff],
                marker=PAYOFF_MARKER[payoff],
                label=f"{payoff}, {method}",
                zorder=3 if method == "ADI-HV-S" else 5,
                **METHOD_STYLE[method])
            rows.extend(
                {"panel": "spatial", "payoff": payoff, "method": method,
                 "n": p[2], "dx": p[0], "error": p[1]}
                for p in points)
            if method == "ADI-HV-S":
                guide(ax, xs, 2, ys[0] * 0.55, "second order")

    # Only the sampled reference carries an uncertainty worth drawing. The
    # analytic one is below 1e-13, so a line for it would stretch the axis over
    # ten empty decades and hide the trend the panel exists to show; it goes in
    # the subtitle instead. The exact value is platform-dependent at that scale,
    # which is another reason not to draw it.
    floor = reference["basket"]["uncertainty"]
    ax.axhline(floor, color=ca.OPTION_COLOR["basket"], lw=0.9, ls=(0, (1, 2)),
               zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=min(floor, smallest) / 4.0)
    ax.annotate(
        "basket QMC reference uncertainty", (1.0, floor),
        xycoords=("axes fraction", "data"), textcoords="offset points",
        xytext=(-4, 4), fontsize=7.5, ha="right",
        color=ca.OPTION_COLOR["basket"])
    widest = 0.0
    for payoff in PAYOFFS:
        by_method = {}
        for record in space:
            if record["payoff"] == payoff:
                by_method.setdefault(record["method"], {})[
                    int(record["n"])] = abs(float(record["error"]))
        if len(by_method) == 2:
            first, second = by_method.values()
            widest = max(
                [widest] + [abs(first[n] / second[n] - 1.0)
                            for n in set(first) & set(second)])
    if widest:
        print(f"  the two methods agree to within {widest:.2%} "
              "at every sampled grid")

    ax.set_xlabel("central grid spacing in log price")
    ax.set_ylabel("absolute price error")
    figstyle.style_axes(ax)
    decade_ticks(ax)
    figstyle.legend(ax, loc="upper left", ncol=1)
    return rows


def write_table(data_dir: Path, reference: dict, greeks: list[dict],
                timing: list[dict], budget: list[dict],
                stability: list[dict]) -> None:
    """Write the compact table the thesis quotes, in one machine-readable file."""
    out = data_dir / "thesis_summary.csv"
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["section", "payoff", "quantity", "method", "value", "uncertainty",
             "note"])

        for payoff in PAYOFFS:
            entry = reference[payoff]
            writer.writerow(
                ["reference", payoff, "price", entry["source"], entry["value"],
                 entry["uncertainty"], "independent financial reference"])

        # Every bump is kept, not just the reported one: the bump dependence is
        # the evidence that the interpolated Gamma is resolution-limited, and a
        # table that dropped it would leave that discrepancy unexplained.
        for row in greeks:
            if row["method"] != "ADI-HV-S":
                continue
            writer.writerow(
                ["greek", row["payoff"],
                 f"{row['estimator']}_{row['asset']}", row["method"],
                 row["pde_value"], row["difference"],
                 f"bump={row['relative_bump']} cells={row['bump_over_spacing']} "
                 f"node={row['node_value']} reference={row['reference_value']}"])

        for row in timing:
            writer.writerow(
                ["equal-accuracy", row["payoff"], "median_seconds", row["method"],
                 row["median_seconds"], row["max_seconds"],
                 f"n={row['n']} steps={row['steps']} error={row['error']} "
                 f"target={row['target']} threads={row['threads']}"])

        for row in budget:
            writer.writerow(
                ["error-budget", row["payoff"], row["component"], "",
                 row["magnitude"], "", row["evidence"]])

        for row in stability:
            writer.writerow(
                ["block-stability", row["option"],
                 f"tau={row['tau']} s={row['s']}", "CA-Arnoldi",
                 row["max_block_kappa"], row["max_block_residual"],
                 f"blocks={row['blocks']} failures={row['cholesky_failures']} "
                 f"ortho={row['orthogonality']} m={row['selected_m']} "
                 f"referee={row['referee_agreement']}"])
    print(f"  wrote {out.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Draw the financial-validation figure and thesis table.")
    parser.add_argument(
        "--data-dir", type=Path, default=DATA,
        help=f"validation artifact directory (default: {DATA.relative_to(ROOT)})")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()

    figstyle.validate_palette()
    print(f"{NAME}:")

    summary = read(data_dir / "reference_summary.csv")
    space = read(data_dir / "pde_space.csv")
    if not summary:
        return 1 if blocked("run ./build/financial-reference first") else 1
    if not space:
        return 1 if blocked("run ./build/financial-validation first") else 1

    reference = {}
    for payoff in PAYOFFS:
        row = next(
            (r for r in summary
             if r["payoff"] == payoff and r["quantity"] == "price"), None)
        if row is None:
            return 1 if blocked(
                f"the reference summary carries no {payoff} price") else 1
        reference[payoff] = {
            "value": float(row["value"]),
            "uncertainty": float(row["half_width"]) + float(row["bump_error"]),
            "source": row["source"],
        }

    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    rows = draw_spatial(ax, space, reference)
    if not rows:
        return 1 if blocked("the refinement tables carry too few points") else 1

    SIDECAR.unlink(missing_ok=True)
    fig.savefig(PNG, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {PNG.relative_to(ROOT)}")

    write_table(
        data_dir,
        reference,
        read(data_dir / "pde_greeks.csv"),
        read(data_dir / "equal_accuracy.csv"),
        read(data_dir / "error_budget.csv"),
        read(CALIBRATION / "block_stability.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
