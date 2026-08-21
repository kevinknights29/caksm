"""Collective and halo latency against payload, at two and four participants.

The participant-count term a node-tier constant cannot express: one number per
rung cannot say that the same all-reduce costs more at four participants than at
two. These are the constants every predicted floor in this set is built from.

The third panel carries the finding. Doubling the participants costs the
all-reduce about half again its latency and costs the halo almost nothing, which
is the case for s-step: it removes all-reduces and pays for them in deeper
halos. The shaded band marks the payloads the solver's own collectives occupy,
so the 64 MiB point reads as the bandwidth anchor it is.

Source: data/ca-participant-calibration-quiet.

  uv run scripts/plots/ca_participant_latency.py
"""
# /// script
# dependencies = ["matplotlib", "numpy"]
# ///
from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np

from figstyle import mark_better
import ca_figstyle as ca
import ca_figlib as lib

FIGURE = lib.Figure("participant_latency")

OPERATIONS = (
    ("allreduce", "NCCL all-reduce", "all-reduce"),
    ("halo", "Deep-halo send and receive", "halo exchange"),
)
OPERATION_COLOR = {
    "allreduce": ca.CATEGORICAL[2],
    "halo": ca.CATEGORICAL[4],
}
# The all-reduce sweep includes a 64 MiB bandwidth anchor that no integrator
# collective approaches. Naming the working band keeps the anchor from being
# read as an operating point.
SOLVER_COLLECTIVE_MAX_BYTES = 1024.0


def draw() -> str | None:
    files = {
        2: lib.PARTICIPANTS / "node_2participants.csv",
        4: lib.PARTICIPANTS / "node_4participants.csv",
    }
    if not all(path.exists() for path in files.values()):
        return FIGURE.blocked(
            "data/ca-participant-calibration-quiet is incomplete")

    # One row per rank names the same global distribution; keep one per label.
    measured: dict[int, dict[tuple[str, str], dict]] = {}
    for participants, path in files.items():
        with path.open() as handle:
            unique: dict[tuple[str, str], dict] = {}
            for record in csv.DictReader(handle):
                if record["contended"] != "0":
                    continue
                unique.setdefault(
                    (record["operation"], record["label"]), record)
        measured[participants] = unique

    fig, axes = plt.subplots(
        1, 3, figsize=(15.0, 5.0), constrained_layout=True)
    rows = []

    for axis, (operation, title, _) in zip(axes, OPERATIONS):
        for participants in (2, 4):
            selected = sorted(
                (record for (kind, _), record in
                 measured[participants].items() if kind == operation),
                key=lambda r: float(r["payload_bytes"]))
            if not selected:
                continue
            payload = np.array(
                [float(r["payload_bytes"]) for r in selected])
            median = np.array(
                [float(r["global_median_s"]) * 1e6 for r in selected])
            low = np.array(
                [float(r["global_min_s"]) * 1e6 for r in selected])
            high = np.array(
                [float(r["global_max_s"]) * 1e6 for r in selected])
            axis.errorbar(
                payload, median, yerr=[median - low, high - median],
                color=ca.TOPOLOGY_COLOR[participants],
                marker="o" if participants == 2 else "s",
                ms=4.5, lw=1.3, capsize=2.5, elinewidth=0.9, zorder=4)
            for record, value in zip(selected, median):
                rows.append({
                    "participants": participants,
                    "operation": operation,
                    "label": record["label"],
                    "payload_bytes": float(record["payload_bytes"]),
                    "median_us": value,
                    "min_us": float(record["global_min_s"]) * 1e6,
                    "max_us": float(record["global_max_s"]) * 1e6,
                    "repeats": int(record["repeats"]),
                })

        axis.set_xscale("log")
        axis.set_yscale("log")

        # After the scales and limits are settled, never before: a span anchored
        # at zero on a logarithmic axis has no left edge, and the axis grows
        # without bound trying to give it one.
        if operation == "allreduce":
            left, _ = axis.get_xlim()
            axis.axvspan(
                left, SOLVER_COLLECTIVE_MAX_BYTES, color=ca.SEQUENTIAL[0],
                alpha=0.55, lw=0, zorder=0)
            lib.label_value(
                axis, SOLVER_COLLECTIVE_MAX_BYTES, axis.get_ylim()[1],
                "every collective\nthe solver issues", dy=-8.0, dx=-6.0,
                ha="right", size=7.6)
            axis.set_xlim(left, axis.get_xlim()[1])

        axis.set_xlabel("payload (bytes)")
        axis.set_ylabel("latency (microseconds)")
        lib.panel_title(axis, title)
        lib.style_axes(axis)
        # The two curves converge at the right-hand end of both panels, so a
        # label on the curve would sit on the other one. A key in the empty
        # lower right costs the reader one glance and never collides.
        lib.legend(
            axis, loc="lower right",
            handles=[
                plt.Line2D(
                    [], [], marker="o" if participants == 2 else "s",
                    ms=5.5, lw=1.3, color=ca.TOPOLOGY_COLOR[participants],
                    label=f"{participants} participants")
                for participants in (2, 4)
            ])

    # The third panel is the finding: the same step from two participants to
    # four is expensive for one operation and nearly free for the other.
    ratio_ax = axes[2]
    penalties: dict[str, list[float]] = {}
    for operation, _, short in OPERATIONS:
        shared = sorted(
            (key for key in measured[2] if key in measured[4]
             and key[0] == operation),
            key=lambda key: float(measured[2][key]["payload_bytes"]))
        if not shared:
            continue
        payload = np.array(
            [float(measured[2][key]["payload_bytes"]) for key in shared])
        ratio = np.array([
            float(measured[4][key]["global_median_s"])
            / float(measured[2][key]["global_median_s"])
            for key in shared
        ])
        penalties[operation] = list(ratio)
        ratio_ax.plot(
            payload, ratio, color=OPERATION_COLOR[operation],
            marker="o" if operation == "allreduce" else "s",
            ms=5.0, lw=1.4, zorder=4)
        # Both names sit below their curve. The all-reduce ratio rises to its
        # peak just right of the midpoint, so a label above it lands on the
        # climb; the band beneath is empty down to the halo series.
        lib.label_series(
            ratio_ax, payload[len(payload) // 2], ratio[len(ratio) // 2],
            short, OPERATION_COLOR[operation], dy=-16.0, size=9.0)

    ratio_ax.axhline(1.0, color=ca.C_INK, lw=1.1, zorder=3)
    ratio_ax.set_xscale("log")
    # Read the limit only after the scale is set. On the linear axis this panel
    # starts out with, the left limit is negative, and a label placed there and
    # then rescaled drags the saved bounding box across a hundred thousand
    # pixels of empty canvas.
    lib.label_value(
        ratio_ax, ratio_ax.get_xlim()[0], 1.0,
        "no participant penalty", dy=5.0, dx=4.0, ha="left", size=7.8,
        color=ca.C_INK)
    ratio_ax.set_xlabel("payload (bytes)")
    ratio_ax.set_ylabel("latency at four participants ÷ at two")
    lib.panel_title(ratio_ax, "Cost of doubling the participants")
    lib.style_axes(ratio_ax)
    mark_better(ratio_ax, "down", loc="upper right")

    def band(operation: str) -> str:
        values = penalties.get(operation, [])
        return (f"{min(values):.2f}-{max(values):.2f}x" if values else "n/a")

    print(f"  doubling participants costs the all-reduce {band('allreduce')} "
          f"and the halo {band('halo')}")

    FIGURE.write(fig, rows)
    return None


if __name__ == "__main__":
    raise SystemExit(lib.run(FIGURE, draw))
