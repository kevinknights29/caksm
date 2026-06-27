"""
Recreates Figure 1 from Niesen and Wright:
  Log-log plots of ODE error vs CPU time for five numerical methods,
  on 31-point (left column) and 61-point (right column) grids,
  for rainbow (top row) and basket (bottom row) options.

Data is read from:
  data/n31/caksm_sweep_n31.csv   (always expected)
  data/n61/caksm_sweep_n61.csv   (optional, panels grayed out if missing)

Per-method sweep selection:
  KSM-EI: tolerance sweep (temporal_steps=100, tol_ei varies)
  Others: step-size sweep (tol_ei=1e-8, temporal_steps varies)
"""
# /// script
# dependencies = [
#   "matplotlib",
#   "numpy",
# ]
# ///

import csv
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# Paths (relative to the repo root, or absolute from anywhere)
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "data"
OUTPUT    = REPO_ROOT / "scripts" / "benchmark_plots.png"

N31_CSV = DATA_DIR / "n31" / "caksm_sweep_n31.csv"
N61_CSV = DATA_DIR / "n61" / "caksm_sweep_n61.csv"

# Method metadata (paper's Figure 1 color scheme)
METHOD_META = {
    "KSM-EI":  {"label": "Krylov (KSM-EI)",           "color": "black",  "lw": 1.5},
    "ADI-HV":  {"label": "Hundsdorfer–Verwer (HV)",   "color": "blue",   "lw": 1.5},
    "ADI-DR":  {"label": "Douglas (DR)",               "color": "green",  "lw": 1.5},
    "CN":      {"label": "Crank–Nicolson (CN)",        "color": "red",    "lw": 1.5},
    "ME":      {"label": "Al-Mohy–Higham (ME)",        "color": "cyan",   "lw": 1.5},
}

OPTION_TYPES = ["rainbow", "basket"]
GRID_SIZES   = [31, 61]

EXPECTED_COLUMNS = {"option_type", "method", "ode_err", "time_ms"}

# Data loading
def load_sweep(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and not EXPECTED_COLUMNS.issubset(rows[0].keys()):
        missing = EXPECTED_COLUMNS - rows[0].keys()
        print(
            f"ERROR: {path} is missing expected columns: {sorted(missing)}\n"
            f"  Found columns: {sorted(rows[0].keys())}\n"
            f"  The sweep CSV may have been assembled without a header due to\n"
            f"  empty or malformed export files. Check the individual exports in\n"
            f"  {path.parent}/ and re-run the aggregation step.",
            file=sys.stderr,
        )
        sys.exit(1)
    return rows


def select_curves(rows: list[dict], n_grid: int) -> dict[str, dict[str, list]]:
    """
    Return {method: {"times": [...], "errors": [...]}} sorted by time.

    Selection rules:
      KSM-EI : temporal_steps == 100, all tol_ei values
      Others : tol_ei == 1e-8 (string match), all temporal_steps values
    """
    curves: dict[str, dict[str, list]] = {}

    for method in METHOD_META:
        if method == "KSM-EI":
            subset = [
                r for r in rows
                if r["method"] == method
                and int(r["temporal_steps"]) == 100
            ]
        else:
            subset = [
                r for r in rows
                if r["method"] == method
                and abs(float(r["tol_ei"]) - 1e-8) < 1e-12
            ]

        if not subset:
            continue

        # sort ascending by time
        subset.sort(key=lambda r: float(r["time_ms"]))

        times  = [float(r["time_ms"]) / 1000.0 for r in subset]  # ms → s
        errors = [float(r["ode_err"])            for r in subset]

        curves[method] = {"times": times, "errors": errors}

    return curves


# Plotting
def plot_panel(
    ax: plt.Axes,
    rows: list[dict],
    option_type: str,
    n_grid: int,
) -> None:
    """Draw one panel of the 2x2 figure."""
    opt_rows = [r for r in rows if r["option_type"] == option_type]
    curves = select_curves(opt_rows, n_grid)

    for method, meta in METHOD_META.items():
        if method not in curves:
            continue

        t = np.array(curves[method]["times"])
        e = np.array(curves[method]["errors"])

        # Drop non-positive values (log-log requires > 0)
        mask = (t > 0) & (e > 0)
        t, e = t[mask], e[mask]
        if len(t) == 0:
            continue

        if len(t) == 1:
            ax.plot(t, e, "o", color=meta["color"], label=meta["label"],
                    ms=6, lw=meta["lw"])
        else:
            ax.plot(t, e, "o-", color=meta["color"], label=meta["label"],
                    ms=4, lw=meta["lw"])

    ax.axhline(1e-4, color="gray", linestyle=":", linewidth=1.0, alpha=0.8,
               label=r"$10^{-4}$ target")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.set_xlabel("CPU time (s)", fontsize=9)
    ax.set_ylabel(r"$\|u_{\mathrm{cube}} - u_{\mathrm{ref\_cube}}\|$", fontsize=9)
    ax.legend(fontsize=8, framealpha=0.9)


def main() -> None:
    n31_rows = load_sweep(N31_CSV)
    n61_rows = load_sweep(N61_CSV)

    for n, rows, csv_path, bash_script in [
        (31, n31_rows, N31_CSV, "scripts/sweep_n31.sh"),
        (61, n61_rows, N61_CSV, "scripts/sweep_n61.sh"),
    ]:
        if not rows:
            print(
                f"ERROR: n={n} sweep data not found at {csv_path}\n"
                f"  Run the sweep first: {bash_script}",
                file=sys.stderr,
            )
            sys.exit(1)

    fig, axes = plt.subplots(
        nrows=2, ncols=2,
        figsize=(10, 8),
        constrained_layout=True,
    )

    col_titles = [f"$n={n}$" for n in GRID_SIZES]
    row_titles = ["Rainbow option", "Basket option"]
    rows_data  = {31: n31_rows, 61: n61_rows}

    for row_idx, option_type in enumerate(OPTION_TYPES):
        for col_idx, n_grid in enumerate(GRID_SIZES):
            ax = axes[row_idx][col_idx]

            plot_panel(
                ax=ax,
                rows=rows_data[n_grid],
                option_type=option_type,
                n_grid=n_grid,
            )

            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(
                    row_titles[row_idx] + "\n"
                    + r"$\|u_{\mathrm{cube}} - u_{\mathrm{ref\_cube}}\|$",
                    fontsize=9,
                )

    fig.suptitle("ODE Error vs CPU time", fontsize=12)

    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
