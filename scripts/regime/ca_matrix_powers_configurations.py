#!/usr/bin/env python3
"""Generate the matrix-powers tuning configuration table and its build variants.

One table drives both compilation and execution.  Candidate shapes are produced
from the declared axes, filtered against the target device, and written out with
the model quantities that justified keeping or dropping each one.  A rejected
shape stays in the table with its reason, so the searched range is recoverable
from the record and not only the part of it that survived.

The filters are a device model, not a measurement.  Shared memory and threads per
block are known before the build; registers per thread are not, so the predicted
residency below is an upper bound from the shared-memory and thread limits alone
and the measured occupancy comes back from the runs.

Outputs:
  data/ca-matrix-powers-tuning/configurations.csv   every candidate and its verdict
  cmake/matrix_powers_variants.cmake                the accepted build variants
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

# Target device limits, from the same GV100 the machine model in
# include/gpu_machine.hpp describes.  Held here as plain numbers because this
# runs before any build and cannot query a device.
SM_COUNT = 80
SHARED_OPTIN_BYTES = 96 * 1024
MAX_THREADS_PER_SM = 2048
MAX_BLOCKS_PER_SM = 32
BYTES_PER_DOUBLE = 8

# The production default, which every sweep is measured against, and the target
# already built on it. That binary carries both families' defaults, so neither
# baseline is emitted as a variant.
BASELINE_TILE = (8, 4, 4)
BASELINE_THREADS = 128
BASELINE_BINARY = "ca-matrix-powers"

# Section 5.3 of the specification: the interior extents shapes are drawn from.
TILE_X_AXIS = (8, 10, 12, 16, 18, 20, 24, 26, 32)
TILE_Y_AXIS = (4, 8, 12, 16)
TILE_Z_AXIS = (4, 8, 12, 16)
THREAD_AXIS = (64, 128, 256, 512)

# Recurrence widths the sweep screens at.  The tile is a compile-time extent
# while the width is a run-time argument, so one binary is asked for several.
REQUIRED_WIDTHS = (1, 2, 3, 4)

# Widths the operator gate covers, which is the whole range the kernel
# instantiates.  Reported, never used to filter: a shape that cannot hold the
# widest width is still worth screening at the widths it can hold, and knowing
# where it stops is what a depth-based dispatch would be built on.
GATE_WIDTHS = (1, 2, 3, 4, 5, 6)

# The production grids the wave filter is applied at.
PRODUCTION_GRIDS = (61, 97)

# Points per thread the first screen admits, and the wave floor a candidate must
# clear unless it is explicitly there to test low occupancy.
MAX_POINTS_PER_THREAD = 16
MIN_WAVES = 2.0

# The hand-auditable set the specification names, kept as an explicit list so a
# change in the generated shapes cannot silently drop one of them.
SEED_SHAPES = (
    (8, 4, 4), (8, 8, 8), (16, 8, 4), (16, 8, 8),
    (12, 8, 8), (20, 8, 4), (24, 8, 4), (24, 4, 4), (32, 4, 4),
    (16, 12, 4), (16, 4, 12), (8, 16, 4), (8, 4, 16),
)

# Plane-streamed candidates, from section 9.3.  The segment length costs no
# shared memory and is a run-time argument, so it is swept by the runner rather
# than by the build and does not multiply the binaries.
STREAM_TILE_X_AXIS = (8, 16, 24, 32)
STREAM_TILE_Y_AXIS = (4, 8, 12, 16)
STREAM_THREAD_AXIS = (64, 128, 256)
STREAM_HEIGHTS = (4, 8, 16, 32, 0)  # 0 streams the whole z extent
# The header's default streamed tile, which is the shape the family's own
# baseline rows are recorded under.
STREAM_BASELINE_TILE = (16, 8)

# The streamed footprint grows as (S+1) queue (tx+2S)(ty+2S), which rises faster
# in the recurrence width than the full-volume tile's two buffers do for a thin z
# extent. The consequence is a lower ceiling on width: the default 16x8 tile
# stops at s=5 monomial and s=4 Chebyshev, so it cannot cover the operator gate's
# full range on its own. This narrower tile can, and is pinned for that reason
# rather than because the model rates it promising.
STREAM_GATE_TILE = (8, 8)
STREAM_GATE_THREADS = 64


def staged_points(tile, steps):
    """Points one block stages: the interior plus its ghost shell."""
    x, y, z = tile
    return (x + 2 * steps) * (y + 2 * steps) * (z + 2 * steps)


def interior_points(tile):
    x, y, z = tile
    return x * y * z


def full_volume_shared_bytes(tile, steps, shared_pad_x, buffers):
    """Dynamic shared memory one full-volume block requests at this width."""
    x, y, z = tile
    return (buffers * (x + 2 * steps + shared_pad_x)
            * (y + 2 * steps) * (z + 2 * steps) * BYTES_PER_DOUBLE)


def stream_shared_bytes(tile_x, tile_y, steps, shared_pad_x, chebyshev):
    """Dynamic shared memory one streamed block requests: every level's queue."""
    queue = 5 if chebyshev else 4
    plane = (tile_x + 2 * steps + shared_pad_x) * (tile_y + 2 * steps)
    return (steps + 1) * queue * plane * BYTES_PER_DOUBLE


def blocks_per_sm(shared_bytes, threads):
    """Residency the shared-memory and thread limits allow, registers aside.

    An upper bound: a register-hungry kernel will do worse.  Used to prune, and
    reported beside the measured occupancy rather than in place of it.
    """
    if shared_bytes <= 0 or shared_bytes > SHARED_OPTIN_BYTES:
        return 0
    by_shared = SHARED_OPTIN_BYTES // shared_bytes
    by_threads = MAX_THREADS_PER_SM // threads
    return max(0, min(by_shared, by_threads, MAX_BLOCKS_PER_SM))


def grid_blocks(tile, n):
    x, y, z = tile
    return (math.ceil(n / x) * math.ceil(n / y) * math.ceil(n / z))


def waves(tile, n, resident):
    if resident <= 0:
        return 0.0
    return grid_blocks(tile, n) / (resident * SM_COUNT)


def variant_name(family, tile, threads, stream_height, extras):
    """A name that reads as the configuration it was built from.

    The three geometry-only names at the production thread count keep the form
    the earlier tile sweep used, so results recorded under them stay comparable.
    """
    if family == "plane-streamed":
        parts = [f"stream-{tile[0]}x{tile[1]}"]
        if stream_height:
            parts.append(f"h{stream_height}")
    else:
        parts = [f"{tile[0]}x{tile[1]}x{tile[2]}"]
    if threads != BASELINE_THREADS:
        parts.append(f"t{threads}")
    parts.extend(extras)
    return "-".join(parts)


def widest_feasible(footprint, widths=REQUIRED_WIDTHS):
    """The largest width in the range whose footprint fits, or zero if none does.

    A width is a run-time argument while a tile is a compile-time extent, so a
    shape that fits at three of the four screened widths is still worth building:
    the harness reports the fourth as not admissible and the runner records that
    row rather than losing the shape entirely.
    """
    feasible = [s for s in widths if footprint(s) <= SHARED_OPTIN_BYTES]
    return max(feasible) if feasible else 0


def evaluate_full_volume(tile, threads, shared_pad_x=0, low_occupancy=False):
    """Apply the pre-build filters to one full-volume candidate.

    Returns (accepted, reason, model) where reason names the first filter that
    rejected it.  The order matters: an infeasible shape is reported as
    infeasible rather than as whatever the next filter would also have said.
    """
    model = {}
    monomial = lambda s: full_volume_shared_bytes(tile, s, shared_pad_x, 2)
    # Chebyshev holds a third buffer, and gpu_pde_chebyshev_basis caps a chunk at
    # the preferred width, so its footprint is set by the chunk and not by the
    # requested width. Recorded, never used to reject: the monomial and Newton
    # paths are what the geometry screen times.
    chebyshev = lambda s: full_volume_shared_bytes(
        tile, min(s, 3), shared_pad_x, 3)

    model["max_feasible_width"] = widest_feasible(monomial)
    model["chebyshev_max_feasible_width"] = widest_feasible(chebyshev)
    model["gate_max_width"] = widest_feasible(monomial, GATE_WIDTHS)
    model["chebyshev_gate_max_width"] = widest_feasible(chebyshev, GATE_WIDTHS)
    width = model["max_feasible_width"] or REQUIRED_WIDTHS[0]
    model["shared_bytes_max"] = monomial(width)
    model["shared_bytes_s4"] = monomial(4)
    model["interior_points"] = interior_points(tile)
    model["staged_points_s4"] = staged_points(tile, 4)
    model["redundant_fraction_s4"] = (
        (staged_points(tile, 4) - interior_points(tile))
        / interior_points(tile))
    model["points_per_thread"] = interior_points(tile) / threads
    # Residency at the widest width the shape can run, which is where it is
    # tightest, so the wave filter is applied to the least favourable case.
    resident = blocks_per_sm(model["shared_bytes_max"], threads)
    model["blocks_per_sm"] = resident
    model["warps_per_sm"] = resident * threads // 32
    model["blocks_n61"] = grid_blocks(tile, 61)
    model["blocks_n97"] = grid_blocks(tile, 97)
    model["waves_n61"] = waves(tile, 61, resident)
    model["waves_n97"] = waves(tile, 97, resident)
    model["shared_headroom_bytes"] = (
        SHARED_OPTIN_BYTES - model["shared_bytes_max"])

    if model["max_feasible_width"] == 0:
        return False, "two shared volumes fit at no screened width", model
    if interior_points(tile) < threads:
        return False, "interior volume below the thread count", model
    if model["points_per_thread"] > MAX_POINTS_PER_THREAD:
        return False, "points per thread above the first-screen limit", model
    if resident <= 0:
        return False, "no block resident at the modeled limits", model
    if not low_occupancy:
        worst_waves = min(model[f"waves_n{n}"] for n in PRODUCTION_GRIDS)
        if worst_waves < MIN_WAVES:
            return False, "fewer than two waves at a production grid", model
    return True, "", model


def evaluate_stream(tile_x, tile_y, threads, shared_pad_x=0):
    """The same filters for a plane-streamed candidate.

    The segment length is deliberately absent: it changes neither the shared
    footprint nor the thread mapping, so it cannot make a build feasible or
    infeasible and is swept at run time instead.  That is the whole point of the
    family, and it is why one streamed binary covers five segment lengths where
    a full-volume z extent would have needed five.
    """
    model = {}
    monomial = lambda s: stream_shared_bytes(
        tile_x, tile_y, s, shared_pad_x, False)
    chebyshev = lambda s: stream_shared_bytes(
        tile_x, tile_y, s, shared_pad_x, True)

    model["max_feasible_width"] = widest_feasible(monomial)
    model["chebyshev_max_feasible_width"] = widest_feasible(chebyshev)
    model["gate_max_width"] = widest_feasible(monomial, GATE_WIDTHS)
    model["chebyshev_gate_max_width"] = widest_feasible(chebyshev, GATE_WIDTHS)
    width = model["max_feasible_width"] or REQUIRED_WIDTHS[0]
    plane_interior = tile_x * tile_y
    model["shared_bytes_max"] = monomial(width)
    model["shared_bytes_s4"] = monomial(4)
    model["interior_points"] = plane_interior
    model["staged_points_s4"] = (tile_x + 8) * (tile_y + 8)
    model["redundant_fraction_s4"] = (
        (model["staged_points_s4"] - plane_interior) / plane_interior)
    model["points_per_thread"] = plane_interior / threads
    resident = blocks_per_sm(model["shared_bytes_max"], threads)
    model["blocks_per_sm"] = resident
    model["warps_per_sm"] = resident * threads // 32
    # Blocks depend on the segment length, so the wave count is reported at the
    # shortest segment the runner will ask for, which is its most parallel end.
    # It is not a filter here: a long segment deliberately trades waves for
    # redundancy, and rejecting on waves would remove the arm under test.
    tile_at_h4 = (tile_x, tile_y, 4)
    model["blocks_n61"] = grid_blocks(tile_at_h4, 61)
    model["blocks_n97"] = grid_blocks(tile_at_h4, 97)
    model["waves_n61"] = waves(tile_at_h4, 61, resident)
    model["waves_n97"] = waves(tile_at_h4, 97, resident)
    model["shared_headroom_bytes"] = (
        SHARED_OPTIN_BYTES - model["shared_bytes_max"])

    if model["max_feasible_width"] == 0:
        return False, "the plane queue fits at no screened width", model
    if plane_interior < threads:
        return False, "plane interior below the thread count", model
    if model["points_per_thread"] > MAX_POINTS_PER_THREAD:
        return False, "points per thread above the first-screen limit", model
    if resident <= 0:
        return False, "no block resident at the modeled limits", model
    return True, "", model


# Origins that are never dropped by a build limit.  These are the shapes the
# specification names and the tile the production default uses, so earlier
# results are recorded under them and the sweep has to be able to compare
# against those rows.
PINNED_ORIGINS = (
    "production baseline",
    "named in the specification",
    "streamed gate width coverage",
)


def candidate_rows(stage):
    """Every candidate the requested stage considers, before filtering.

    Stage one holds the production thread count and screens geometry, which is
    what separates shape from work per thread.  Stage two crosses the named
    geometries with the thread axis.  Keeping the thread axis out of stage one
    is the point of the split: with threads fixed, a difference between two rows
    is a shape difference and nothing else.
    """
    rows = []
    seen = set()

    def add_full(tile, threads, note, low_occupancy=False):
        key = ("full-volume", tile, threads)
        if key in seen:
            return
        seen.add(key)
        rows.append({
            "family": "full-volume", "tile": tile, "threads": threads,
            "note": note, "low_occupancy": low_occupancy,
        })

    def add_stream(tile_x, tile_y, threads, note):
        key = ("plane-streamed", (tile_x, tile_y), threads)
        if key in seen:
            return
        seen.add(key)
        rows.append({
            "family": "plane-streamed", "tile": (tile_x, tile_y, 0),
            "threads": threads, "note": note, "low_occupancy": False,
        })

    add_full(BASELINE_TILE, BASELINE_THREADS, "production baseline")
    add_stream(*STREAM_BASELINE_TILE, BASELINE_THREADS, "production baseline")
    add_stream(*STREAM_GATE_TILE, STREAM_GATE_THREADS,
               "streamed gate width coverage")
    for tile in SEED_SHAPES:
        add_full(tile, BASELINE_THREADS, "named in the specification")
    for x in TILE_X_AXIS:
        for y in TILE_Y_AXIS:
            for z in TILE_Z_AXIS:
                add_full((x, y, z), BASELINE_THREADS, "generated shape")
    for x in STREAM_TILE_X_AXIS:
        for y in STREAM_TILE_Y_AXIS:
            add_stream(x, y, BASELINE_THREADS, "streamed candidate")
    if stage >= 2:
        for tile in SEED_SHAPES:
            for threads in THREAD_AXIS:
                add_full(tile, threads, "thread-count screen")
        for x in STREAM_TILE_X_AXIS:
            for y in STREAM_TILE_Y_AXIS:
                for threads in STREAM_THREAD_AXIS:
                    add_stream(x, y, threads, "streamed thread-count screen")
    return rows


MODEL_FIELDS = (
    "interior_points", "staged_points_s4", "redundant_fraction_s4",
    "points_per_thread", "shared_bytes_s4", "shared_bytes_max",
    "shared_headroom_bytes", "max_feasible_width",
    "chebyshev_max_feasible_width", "gate_max_width",
    "chebyshev_gate_max_width", "blocks_per_sm", "warps_per_sm",
    "blocks_n61", "blocks_n97", "waves_n61", "waves_n97",
)

FIELDS = [
    "configuration", "kernel_family", "tile_x", "tile_y", "tile_z",
    "threads_per_block", "shared_pad_x", "physical_pitch_x", "register_cap",
    "cache_policy", "extra_device_vectorization", "fast_math",
    "elide_basis_barrier", "use_restrict", "accepted", "rejection_reason",
    "origin", *MODEL_FIELDS, "build_variant",
]


def build_table(stage, limits):
    """The whole table: every candidate, its verdict, and its model quantities.

    The build limit applies only to generated shapes.  A shape the specification
    names, and the production baseline, are always built: they are the ones
    earlier results were recorded under, and dropping one would break the
    comparison the sweep exists to make.
    """
    rows = []
    for candidate in candidate_rows(stage):
        family = candidate["family"]
        tile = candidate["tile"]
        threads = candidate["threads"]
        if family == "full-volume":
            accepted, reason, model = evaluate_full_volume(
                tile, threads, low_occupancy=candidate["low_occupancy"])
        else:
            accepted, reason, model = evaluate_stream(tile[0], tile[1], threads)
        name = variant_name(family, tile, threads, 0, ())
        rows.append({
            "configuration": name,
            "kernel_family": family,
            "tile_x": tile[0], "tile_y": tile[1],
            "tile_z": tile[2] if family == "full-volume" else 0,
            "threads_per_block": threads,
            "shared_pad_x": 0,
            "physical_pitch_x": 1,
            "register_cap": 0,
            "cache_policy": "default",
            "extra_device_vectorization": 0,
            "fast_math": 0,
            "elide_basis_barrier": 0,
            "use_restrict": 0,
            "accepted": 1 if accepted else 0,
            "rejection_reason": reason,
            "origin": candidate["note"],
            # ca-matrix-powers is built on every default, so it is the baseline
            # of both families at once and neither needs a second binary.
            "build_variant": (
                BASELINE_BINARY if candidate["note"] == "production baseline"
                else (name if accepted else "")),
            **{key: model[key] for key in MODEL_FIELDS},
        })

    accepted = []
    for family, limit in limits.items():
        passing = [row for row in rows
                   if row["accepted"] and row["kernel_family"] == family]
        named = [row for row in passing if row["origin"] in PINNED_ORIGINS]
        generated = [row for row in passing
                     if row["origin"] not in PINNED_ORIGINS]
        # Generated shapes are ordered by modeled redundancy, so a truncated
        # round still covers the ones the model rates most promising, and the
        # cut is recorded in the table rather than inferred from its length.
        generated.sort(
            key=lambda row: (row["redundant_fraction_s4"],
                             row["configuration"]))
        room = max(0, limit - len(named)) if limit else len(generated)
        for row in generated[room:]:
            row["accepted"] = 0
            row["build_variant"] = ""
            row["rejection_reason"] = (
                f"beyond the {limit}-variant {family} build limit "
                f"for this round")
        accepted.extend(named)
        accepted.extend(generated[:room])
    return rows, accepted


CMAKE_HEADER = """# Matrix-powers tuning variants.
#
# Generated by scripts/regime/ca_matrix_powers_configurations.py.  Edit that
# script and regenerate rather than editing this file: the same table produces
# data/ca-matrix-powers-tuning/configurations.csv, which records why every
# rejected shape was excluded, and the two have to agree.
#
# Every entry below passed the pre-build filters in section 5.3 of the
# specification: the shared-memory volumes fit at at least one screened width,
# the interior volume is at least the thread count, work per thread is within
# the first-screen limit, and the grid supplies at least two waves at the
# production grids. A width a shape cannot hold is reported by the harness as
# not admissible and recorded as such, so it costs a row rather than the shape.
#
# The two baselines are absent on purpose: ca-matrix-powers is built on every
# default and is already the full-volume and plane-streamed baseline at once.
"""


def write_cmake(path, accepted):
    lines = [CMAKE_HEADER]
    for row in accepted:
        if row["origin"] == "production baseline":
            continue
        arguments = [
            f'NAME {row["configuration"]}',
            f'FAMILY {row["kernel_family"]}',
            f'TILE_X {row["tile_x"]}',
            f'TILE_Y {row["tile_y"]}',
        ]
        if row["kernel_family"] == "full-volume":
            arguments.append(f'TILE_Z {row["tile_z"]}')
        arguments.append(f'THREADS {row["threads_per_block"]}')
        lines.append(
            "caksm_add_matrix_powers_variant(\n        "
            + "\n        ".join(arguments)
            + ")\n")
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2],
        help="repository root")
    parser.add_argument(
        "--stage", type=int, default=1, choices=(1, 2),
        help="1 holds threads at the production count and screens geometry; "
             "2 also crosses the named shapes with the thread axis")
    # The defaults are what the checked-in table and variant list were generated
    # with, so a plain re-run reproduces them. The specification caps the first
    # streamed round at 24; eight is well inside that and keeps the first build
    # to something that finishes in one sitting.
    parser.add_argument(
        "--limit-full-volume", type=int, default=20,
        help="most full-volume build variants to emit; 0 emits all of them")
    parser.add_argument(
        "--limit-streamed", type=int, default=8,
        help="most plane-streamed build variants to emit; 0 emits all of them "
             "(the specification's first-round ceiling is 24)")
    args = parser.parse_args()

    rows, accepted = build_table(
        args.stage,
        {"full-volume": args.limit_full_volume,
         "plane-streamed": args.limit_streamed})

    data_dir = args.root / "data" / "ca-matrix-powers-tuning"
    data_dir.mkdir(parents=True, exist_ok=True)
    table = data_dir / "configurations.csv"
    with table.open("w", newline="") as handle:
        # csv defaults to CRLF, which leaves a carriage return glued to the last
        # field of every line. The shell runners read this table with awk, where
        # that turns a header name into a key that never matches: the lookup
        # yields an empty field index, and an empty index is the whole record in
        # some awk implementations and an error in others. Neither is a parse.
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    cmake_dir = args.root / "cmake"
    cmake_dir.mkdir(parents=True, exist_ok=True)
    write_cmake(cmake_dir / "matrix_powers_variants.cmake", accepted)

    rejected = len(rows) - len(accepted)
    print(f"matrix-powers configurations, stage {args.stage}")
    print(f"  candidates : {len(rows)}")
    print(f"  accepted   : {len(accepted)}")
    print(f"  rejected   : {rejected} (reasons in the table)")
    print(f"  table      : {table}")
    print(f"  variants   : {cmake_dir / 'matrix_powers_variants.cmake'}")
    reasons = {}
    for row in rows:
        if not row["accepted"]:
            reasons[row["rejection_reason"]] = (
                reasons.get(row["rejection_reason"], 0) + 1)
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"    {count:4d}  {reason}")


if __name__ == "__main__":
    main()
