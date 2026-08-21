"""Largest odd grid each GPU topology admits, and the common grid it forces.

Bars are the allocation model's answer per topology, option and width, predicted
before anything was allocated. The common grid is the minimum over all of them,
so one arm binds the choice and the figure names it.

The ceiling is a memory budget: every V100 reports the same usable capacity
after the reserve, so a topology's limit is set by how much of the domain each
device has to hold. Hue carries the option and the width is a texture, so
neither reads as rank.

Source: data/ca-integrator-largest-common-m39.

  uv run scripts/plots/ca_memory_model.py
"""
# /// script
# dependencies = ["matplotlib", "numpy"]
# ///
from __future__ import annotations

import csv
import re

import matplotlib.pyplot as plt
import numpy as np

import ca_figstyle as ca
import ca_figlib as lib

FIGURE = lib.Figure("memory_model")

TOPOLOGY_ORDER = [
    "one_gpu", "two_gpu_one_node", "two_gpu_two_nodes", "four_gpu_two_nodes"]
TOPOLOGY_GPUS = {
    "one_gpu": 1, "two_gpu_one_node": 2,
    "two_gpu_two_nodes": 2, "four_gpu_two_nodes": 4}
# Spelled the way every other figure in this set spells a topology, so a reader
# moving between them is not re-learning the vocabulary.
TOPOLOGY_LABEL = {
    "one_gpu": "1 GPU",
    "two_gpu_one_node": "2 GPUs\n1 node",
    "two_gpu_two_nodes": "2 GPUs\n2 nodes",
    "four_gpu_two_nodes": "4 GPUs\n2 nodes",
}


def usable_mib() -> float | None:
    """Read the per-device usable capacity the reports agree on.

    Args:
        None.

    Returns:
        The usable MiB per device, or None if the reports disagree or carry no
        capacity line. Disagreement is not averaged away: a mixed capacity means
        the bars are not comparable and the figure should say nothing.
    """
    seen = set()
    for report in sorted(lib.LARGEST.glob("*_s[0-9].txt")):
        for found in re.finditer(
                r"usable=([\d.]+) MiB", report.read_text(errors="replace")):
            seen.add(round(float(found.group(1)), 3))
    return seen.pop() if len(seen) == 1 else None


def draw() -> str | None:
    reports = lib.LARGEST / "memory_reports.csv"
    selected = lib.LARGEST / "selected_n.txt"
    if not reports.exists():
        return FIGURE.blocked(
            "data/ca-integrator-largest-common-m39/memory_reports.csv is absent")

    with reports.open() as handle:
        records = list(csv.DictReader(handle))
    order = [t for t in TOPOLOGY_ORDER if any(r["topology"] == t for r in records)]
    positions = np.arange(len(order), dtype=float)

    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    rows = []
    bar = 0.2
    for index, (option, width) in enumerate(
            [(o, w) for o in ("basket", "rainbow") for w in (1, 4)]):
        values = []
        for topology in order:
            match = [r for r in records
                     if r["topology"] == topology and r["option"] == option
                     and int(r["s"]) == width]
            values.append(int(match[0]["recommended_n"]) if match else 0)
            if match:
                rows.append({
                    "topology": topology,
                    "gpus": TOPOLOGY_GPUS[topology],
                    "option": option, "s": width,
                    "recommended_largest_odd_n": int(match[0]["recommended_n"]),
                })
        ax.bar(
            positions + (index - 1.5) * bar, values, bar,
            color=ca.OPTION_COLOR[option],
            hatch="///" if width == 1 else None,
            edgecolor="white", lw=0.6, zorder=3,
            label=f"{option}, s={width}")

    capacity = usable_mib()
    chosen = None
    if selected.exists():
        chosen = int(selected.read_text().split()[0])
        # Drawn heavier than a grid line and over the bars, since it is the
        # figure's answer rather than a reference the bars sit against.
        ax.axhline(chosen, color=ca.DIVERGING_LOW, ls="--", lw=2.0, zorder=5)
        for row in rows:
            row["selected_common_n"] = chosen

    # The binding arms are what taking a minimum means, so they are named once
    # over the group that carries them rather than once per bar.
    smallest = min(r["recommended_largest_odd_n"] for r in rows)
    binding = [r for r in rows if r["recommended_largest_odd_n"] == smallest]
    if binding and chosen is not None:
        binding_topology = binding[0]["topology"]
        lib.label_value(
            ax, positions[order.index(binding_topology)],
            max(r["recommended_largest_odd_n"] for r in rows
                if r["topology"] == binding_topology),
            f"binds at n={chosen}", dy=13.0, size=8.5,
            color=ca.DIVERGING_LOW)

    ax.set_xticks(positions)
    ax.set_xticklabels([TOPOLOGY_LABEL[t] for t in order], fontsize=8.5)
    ax.set_ylabel("largest odd grid n the allocation model admits")
    ax.set_ylim(0.0, max(r["recommended_largest_odd_n"] for r in rows) * 1.22)
    lib.style_axes(ax, grid_axis="y")
    lib.legend(ax, loc="upper left", ncol=2)
    # No "better" arrow: the legend holds one upper corner and the memory-budget
    # note the other, and a larger admissible grid needs no arrow to say so.

    # The memory budget behind every bar goes to the run log rather than onto
    # the image, where it was a second sentence competing with the title for a
    # figure that only makes one claim.
    print(f"  binds at n={chosen} on "
          + ", ".join(sorted({
              f"{TOPOLOGY_LABEL[r['topology']]} s={r['s']}".replace("\n", " ")
              for r in binding}))
          + (f"; {capacity / 1024.0:.2f} GiB usable per device"
             if capacity is not None else ""))

    FIGURE.write(fig, rows)
    return None


if __name__ == "__main__":
    raise SystemExit(lib.run(FIGURE, draw))
