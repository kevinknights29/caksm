"""Referee cost: scaling substeps and SpMV count against grid.

The augmented Basket forcing columns dominate the operator 1-norm and the
substep count follows it, so the two methods differ by orders of magnitude and
both vertical axes are logarithmic. The Rainbow compatibility series is the
control: it sits on the scaled series at every grid, so the contract is
expensive on the augmented operator specifically rather than in general.

Source: docs/ca_referee_baseline.md, the frozen manifest, parsed rather than
transcribed so the figure cannot drift from it.

  uv run scripts/plots/ca_referee_cost.py
"""
# /// script
# dependencies = ["matplotlib", "numpy"]
# ///
from __future__ import annotations

import re

import matplotlib.pyplot as plt

from figstyle import mark_better
import ca_figstyle as ca
import ca_figlib as lib

FIGURE = lib.Figure("referee_cost")

# Three tables in the manifest carry the cost: the historical fixed-contract
# runs, the Stage 1 scaled checkpoint, and the promoted weak-scaling reference
# set. Each has its own column layout, hence three patterns.
HISTORICAL_ROW = re.compile(
    r"^\|\s*(Basket|Rainbow)\s*\|\s*(\d+)\s*\|\s*[\d,]+\s*\|\s*[\d,]+\s*\|"
    r"\s*`[^`]+`\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|")
STAGE1_ROW = re.compile(
    r"^\|\s*(\d+)\s*\|\s*`2\^-\d+`\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|"
    r"\s*([\d,]+)\s*\|")
REFERENCE_ROW = re.compile(
    r"^\|\s*(Basket|Rainbow)\s*\|\s*(\d+)\s*\|\s*[\d,]+\s*\|\s*[\d,]+\s*\|"
    r"\s*`?[^|`]+`?\s*\|\s*([\d,]+)\s*/\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|")


def _count(text: str) -> int:
    return int(text.replace(",", "").strip())


def draw() -> str | None:
    if not lib.REFEREE_DOC.exists():
        return FIGURE.blocked("docs/ca_referee_baseline.md is absent")

    rows: list[dict] = []
    for line in lib.REFEREE_DOC.read_text().splitlines():
        line = line.strip()

        found = REFERENCE_ROW.match(line)
        if found is not None:
            option = found.group(1).lower()
            method = ("scaled-augmentation" if option == "basket"
                      else "fixed-compatibility")
            rows.append({
                "option": option, "n": int(found.group(2)), "method": method,
                "contract_scalings": _count(found.group(3)),
                "selected_scalings": _count(found.group(4)),
                "spmv_count": _count(found.group(5)),
                "table": "promoted weak-scaling reference",
            })
            continue

        found = HISTORICAL_ROW.match(line)
        if found is not None:
            option = found.group(1).lower()
            rows.append({
                "option": option, "n": int(found.group(2)),
                "method": "fixed-compatibility",
                "contract_scalings": _count(found.group(3)),
                "selected_scalings": _count(found.group(3)),
                "spmv_count": _count(found.group(4)),
                "table": "preserved historical evidence",
            })
            continue

        found = STAGE1_ROW.match(line)
        if found is not None:
            rows.append({
                "option": "basket", "n": int(found.group(1)),
                "method": "scaled-augmentation",
                "contract_scalings": _count(found.group(2)),
                "selected_scalings": _count(found.group(3)),
                "spmv_count": _count(found.group(4)),
                "table": "stage 1 checkpoint",
            })

    if not rows:
        return FIGURE.blocked(
            "no cost table in docs/ca_referee_baseline.md parsed")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), constrained_layout=True)

    def series(option: str, method: str, key: str) -> list[tuple[int, int]]:
        # The Basket compatibility contract is what the fixed method requires at
        # each grid, which the scaled rows also record, so it is keyed on the
        # column rather than on which method happened to run there.
        collected: dict[int, int] = {}
        for record in rows:
            if record["option"] != option:
                continue
            if key != "contract_scalings" and record["method"] != method:
                continue
            if record[key] > 0:
                collected[record["n"]] = record[key]
        return sorted(collected.items())

    series_color = {
        ("basket", "fixed-compatibility"): ca.CATEGORICAL[2],
        ("basket", "scaled-augmentation"): ca.CATEGORICAL[1],
        ("rainbow", "fixed-compatibility"): ca.CATEGORICAL[5],
    }
    lines = [
        ("basket", "fixed-compatibility", "contract_scalings", None,
         "Basket, compatibility contract"),
        ("basket", "scaled-augmentation", "selected_scalings", "spmv_count",
         "Basket, scaled augmentation"),
        ("rainbow", "fixed-compatibility", "selected_scalings", "spmv_count",
         "Rainbow, compatibility contract"),
    ]
    for option, method, key, spmv_key, label in lines:
        scalings = series(option, method, key)
        if not scalings:
            continue
        axes[0].semilogy(
            [n for n, _ in scalings], [value for _, value in scalings],
            marker="o" if option == "basket" else "s", ms=5, lw=1.3,
            ls="-" if method == "scaled-augmentation" else "--",
            color=series_color[(option, method)], label=label)
        if spmv_key is None:
            continue
        counts = series(option, method, spmv_key)
        axes[1].semilogy(
            [n for n, _ in counts], [value for _, value in counts],
            marker="o" if option == "basket" else "s", ms=5, lw=1.3,
            ls="-" if method == "scaled-augmentation" else "--",
            color=series_color[(option, method)], label=label)
    # The fixed contract's own SpMV counts, which stop where it stopped running:
    # the Basket compatibility series has no count beyond n=31 because it was
    # never run there.
    fixed_basket = sorted(
        ((r["n"], r["spmv_count"]) for r in rows
         if r["option"] == "basket" and r["method"] == "fixed-compatibility"))
    if fixed_basket:
        axes[1].semilogy(
            [n for n, _ in fixed_basket], [v for _, v in fixed_basket],
            marker="o", ms=5, lw=1.3, ls="--",
            color=series_color[("basket", "fixed-compatibility")],
            label="Basket, compatibility contract")

    axes[0].set_xlabel("grid n")
    axes[0].set_ylabel("scaling substeps")
    lib.panel_title(axes[0], "Scaling substeps the method requires")
    axes[1].set_xlabel("grid n")
    axes[1].set_ylabel("sparse matrix-vector products per action")
    lib.panel_title(axes[1], "SpMVs one referee action costs")
    for axis, corner in ((axes[0], "upper left"), (axes[1], "lower right")):
        lib.style_axes(axis)
        lib.legend(axis, loc=corner)
    mark_better(axes[0], "down", loc="lower right")
    mark_better(axes[1], "down", loc="upper left")

    # The gap is the claim, so it is measured off the parsed tables rather than
    # described, and at one shared grid, because a ratio taken across two
    # different grids would compare the methods and the problem size at once.
    def counts(option: str, method: str) -> dict[int, int]:
        return {
            r["n"]: r["spmv_count"] for r in rows
            if r["option"] == option and r["method"] == method
            and r["spmv_count"] > 0
        }

    fixed = counts("basket", "fixed-compatibility")
    scaled = counts("basket", "scaled-augmentation")
    shared = sorted(set(fixed) & set(scaled))
    lib.title(
        fig,
        "Scaled augmentation reaches grids the compatibility contract cannot")
    if shared:
        n = shared[-1]
        print(f"  at n={n} on Basket the fixed contract needs {fixed[n]:,} "
              f"SpMVs per action against {scaled[n]:,} scaled, "
              f"{fixed[n] / scaled[n]:,.0f}x as many")

    FIGURE.write(fig, rows)
    return None


if __name__ == "__main__":
    raise SystemExit(lib.run(FIGURE, draw))
