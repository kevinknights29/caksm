#!/usr/bin/env bash
# What actually limits the matrix-powers kernel, measured rather than modeled.
#
# The kernel's reported rate had only ever been an ideal-traffic figure. For
# every point this records the interior points per block, the shared tile
# volume, the ghost redundancy factor, modeled compulsory traffic, and the
# achieved rate against the DRAM and L2 roofs, then names a verdict.
#
# Two passes. The ncu pass runs --profile-only, which executes exactly one
# complete basis evaluation and no correctness or timing launches; its byte
# count feeds the ordinary correctness-gated timing pass. A capture that is
# missing, disabled, failed, contended or unparsable stays an explicit
# diagnostic row and cannot carry a verdict.
#
# Tile geometry is a compile-time extent, so this runs one binary per geometry.
# A geometry whose shared tile exceeds the device opt-in limit exits 2 and is
# recorded as not admissible rather than as a failure.
#
# The plan is deliberately sparse: the production 8x4x4 tile covers both
# options, all four grids and all four widths, which is 32 points, and each of
# the three alternative tiles covers (n=61,s=3) and (n=97,s=4) for both options,
# which is 12 more. A trimmed environment request produces fewer rows, and the
# figure blocks on the missing points rather than drawing a subset.
#
# Run under the batch scheduler on one exclusive Synge node. Every verdict
# divides a measured byte count by a measured time, so this is the recommended
# route:
#
#   sbatch --nodes=1 --ntasks=10 --partition=compute --time=01:00:00 \
#          --nodelist=synge-n01 --exclusive \
#          --export=ALL,NCU=/usr/local/cuda-12.8/nsight-compute-2025.1.1/ncu \
#          scripts/regime/ca_mpk_vertical.sh
#
# Interactively, for a smoke test only. The runs below share the calling shell's
# CPUs, which inflates the denominator and can move a point across a roof:
#
#   salloc -N 1 -n 10 -p compute -t 01:00:00 --nodelist=synge-n01
#   NCU=/usr/local/cuda-12.8/nsight-compute-2025.1.1/ncu ./scripts/regime/ca_mpk_vertical.sh

set -uo pipefail

ROOT="${ROOT:-$PWD}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
OUT_DIR="${OUT_DIR:-$ROOT/data/ca-integrator-mpk-vertical}"
DEVICE="${DEVICE:-0}"
REPEATS="${REPEATS:-7}"
GRIDS="${GRIDS:-31 61 77 97}"
WIDTHS="${WIDTHS:-1 2 3 4}"
OPTIONS="${OPTIONS:-basket rainbow}"
TILES="${TILES:-8x4x4 16x8x8 8x8x8 16x8x4}"
NCU="${NCU:-ncu}"
NCU_METRIC="${NCU_METRIC:-dram__bytes.sum}"
CANONICAL_TILES="8x4x4 16x8x8 8x8x8 16x8x4"
CANONICAL_GRIDS="31 61 77 97"
CANONICAL_WIDTHS="1 2 3 4"
CANONICAL_OPTIONS="basket rainbow"
CANONICAL_ALTERNATIVE_TILES="16x8x8 8x8x8 16x8x4"
CANONICAL_ALTERNATIVE_POINTS="61:3 97:4"

mkdir -p "$OUT_DIR"
ROWS="$OUT_DIR/mpk_vertical.csv"

fail() {
    echo "Error: $*" >&2
    echo "Partial vertical-sweep evidence retained in: $OUT_DIR" >&2
    exit 1
}

word_set() {
    tr ' ' '\n' <<< "$1" | sed '/^$/d' | LC_ALL=C sort -u \
        | paste -sd' ' -
}

printf '%s\n' \
    "tile,option,basis,n,s,verdict,redundancy,effective_read_redundancy,seconds,seconds_min,seconds_median,seconds_max,repeats,compulsory_gbs,tiled_gbs,dram_fraction,l2_fraction,launch_fraction,occupancy,measured_over_modeled,nearest_roof,nearest_roof_fraction,dram_roof_gbs,l2_roof_gbs,dram_source,dram_bytes,compulsory_bytes,tiled_bytes,input_bytes,basis_write_bytes,face_bytes,boundary_tail_bytes,tail_kernel_bytes,tile_doubles,blocks,total_blocks,chunks,chunk_max,pde_launches,tail_launches,shared_buffers,contended,measurement_status,record_class,ncu_log,log" \
    > "$ROWS"

binary_for_tile() {
    if [[ "$1" == "8x4x4" ]]; then
        echo "$BUILD_DIR/ca-matrix-powers"
    else
        echo "$BUILD_DIR/ca-matrix-powers-$1"
    fi
}

# One ncu capture of the same point, reduced to a single DRAM byte count.  The
# profile-only executable path contains exactly one basis evaluation.  Emit a
# status, byte count, and report path separated by "|" so every failure mode is
# retained in the CSV rather than silently becoming a modeled measurement.
capture_dram_bytes() {
    local binary="$1" option="$2" n="$3" width="$4" stem="$5"
    local report="${stem}_ncu.txt"
    if [[ "$NCU" == "none" ]]; then
        printf 'disabled||%s\n' "$report"
        return
    fi
    if ! command -v "$NCU" >/dev/null 2>&1; then
        printf 'missing||%s\n' "$report"
        return
    fi
    if ! "$NCU" --target-processes all --kernel-name-base function \
            --metrics "$NCU_METRIC" --csv --print-units base \
            "$binary" --device "$DEVICE" --n "$n" --s "$width" \
            --option "$option" --basis monomial --profile-only \
            > "$report" 2>&1; then
        printf 'failed||%s\n' "$report"
        return
    fi
    local marker
    marker="$(grep 'MPK_PROFILE basis_evaluations=1' "$report" | tail -n 1)"
    if [[ -z "$marker" ]]; then
        printf 'profile-marker-missing||%s\n' "$report"
        return
    fi
    local pde_launches tail_launches expected_rows
    pde_launches="$(sed -n 's/.* pde_launches=\([0-9][0-9]*\).*/\1/p' \
        <<< "$marker")"
    tail_launches="$(sed -n 's/.* tail_launches=\([0-9][0-9]*\).*/\1/p' \
        <<< "$marker")"
    if [[ -z "$pde_launches" || -z "$tail_launches" ]]; then
        printf 'profile-marker-invalid||%s\n' "$report"
        return
    fi
    expected_rows=$((pde_launches + tail_launches))

    local parser="${PYTHON:-python3}"
    if ! command -v "$parser" >/dev/null 2>&1; then
        printf 'parser-missing||%s\n' "$report"
        return
    fi
    local parsed
    if ! parsed="$("$parser" - "$report" "$NCU_METRIC" <<'PY'
import csv
import math
import sys

# ncu emits the metric unit in its own column. --print-units base should keep
# that at "byte", but a scaled unit would otherwise be read as a raw count and
# silently under-report the traffic, so the unit is honored rather than assumed.
SCALE = {
    "": 1.0, "byte": 1.0, "bytes": 1.0,
    "Kbyte": 1.0e3, "Mbyte": 1.0e6, "Gbyte": 1.0e9, "Tbyte": 1.0e12,
    "KB": 1.0e3, "MB": 1.0e6, "GB": 1.0e9,
}

path, metric = sys.argv[1:3]
total = 0.0
count = 0
with open(path, newline="", errors="replace") as handle:
    for row in csv.reader(handle):
        if not row or metric not in row:
            continue
        value = row[-1].replace(",", "").strip()
        unit = row[-2].strip() if len(row) >= 2 else ""
        try:
            number = float(value)
        except ValueError:
            continue
        if unit not in SCALE:
            raise SystemExit(3)
        if math.isfinite(number) and number >= 0.0:
            total += number * SCALE[unit]
            count += 1
if count == 0:
    raise SystemExit(2)
print(f"{count}|{total:.0f}")
PY
    )"; then
        # Exit 3 is an unrecognised unit, which must not be guessed at.
        if [[ $? -eq 3 ]]; then
            printf 'unknown-metric-unit||%s\n' "$report"
        else
            printf 'parse-failed||%s\n' "$report"
        fi
        return
    fi
    local metric_rows
    IFS='|' read -r metric_rows parsed <<< "$parsed"
    if [[ "$metric_rows" -ne "$expected_rows" || -z "$parsed" \
          || "$parsed" == "0" ]]; then
        printf 'kernel-count-mismatch||%s\n' "$report"
        return
    fi

    if grep -q 'MPK_PROFILE .*contended=1' "$report"; then
        printf 'measured-contended|%s|%s\n' "$parsed" "$report"
    else
        printf 'measured|%s|%s\n' "$parsed" "$report"
    fi
}

record_point() {
    local tile="$1" option="$2" n="$3" width="$4"
    local binary
    binary="$(binary_for_tile "$tile")"
    if [[ ! -x "$binary" ]]; then
        fail "required tile binary is missing or not executable: $binary"
    fi

    local stem="$OUT_DIR/${tile}_${option}_n${n}_s${width}"
    local log="${stem}.txt"
    local log_record
    log_record="$(basename "$log")"

    local capture measurement_status dram_bytes ncu_log
    capture="$(capture_dram_bytes \
        "$binary" "$option" "$n" "$width" "$stem")"
    IFS='|' read -r measurement_status dram_bytes ncu_log <<< "$capture"
    local ncu_record
    ncu_record="$(basename "$ncu_log")"
    if [[ "$measurement_status" != "measured" ]]; then
        echo "  ncu status: $measurement_status ($ncu_log)"
    fi

    local extra=()
    if [[ "$measurement_status" == measured* && -n "$dram_bytes" ]]; then
        extra+=(--ncu-dram-bytes "$dram_bytes")
    fi

    "$binary" --device "$DEVICE" --n "$n" --s "$width" \
        --option "$option" --basis monomial --repeats "$REPEATS" \
        "${extra[@]}" > "$log" 2>&1
    local status=$?
    if [[ $status -eq 2 ]]; then
        local column
        {
            printf '%s,%s,monomial,%s,%s,not-admissible' \
                "$tile" "$option" "$n" "$width"
            for ((column = 7; column <= 39; ++column)); do
                printf ','
            done
            printf ',%s,diagnostic,%s,%s\n' \
                "$measurement_status" "$ncu_record" "$log_record"
        } >> "$ROWS"
        echo "  tile=$tile n=$n s=$width option=$option: not admissible"
        return
    fi
    if [[ $status -ne 0 ]]; then
        echo "FAILED: $binary exited $status; see $log"
        cat "$log"
        fail "matrix-powers timing run exited $status: $log"
    fi

    local row
    # gawk reserves `log` as a builtin, so the column name and the awk variable
    # carrying it cannot be spelled the same way.
    if ! row="$(awk -v tile="$tile" -v log_record="$log_record" \
        -v measurement_status="$measurement_status" -v ncu_log="$ncu_record" '
        /^MPK_VERTICAL / {
            ++records
            for (q = 2; q <= NF; ++q) {
                split($q, kv, "=")
                field[kv[1]] = kv[2]
            }
            # awk allows a newline after && but not after an open paren, so
            # this is a statement rather than a parenthesised expression.
            record_class = "diagnostic"
            if (measurement_status == "measured" &&
                field["dram_source"] == "ncu" &&
                field["contended"] == "0")
                record_class = "measurement"
            printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s," \
                   "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s," \
                   "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s," \
                   "%s,%s,%s,%s,%s,%s,%s\n",
                tile, field["option"], field["basis"], field["n"], field["s"],
                field["verdict"], field["redundancy"],
                field["effective_read_redundancy"], field["seconds"],
                field["seconds_min"], field["seconds_median"],
                field["seconds_max"], field["repeats"],
                field["compulsory_gbs"], field["tiled_gbs"],
                field["dram_fraction"], field["l2_fraction"],
                field["launch_fraction"], field["occupancy"],
                field["measured_over_modeled"],
                field["nearest_roof"], field["nearest_roof_fraction"],
                field["dram_roof_gbs"], field["l2_roof_gbs"],
                field["dram_source"], field["dram_bytes"],
                field["compulsory_bytes"], field["tiled_bytes"],
                field["input_bytes"], field["basis_write_bytes"],
                field["face_bytes"], field["boundary_tail_bytes"],
                field["tail_kernel_bytes"], field["tile_doubles"],
                field["blocks"], field["total_blocks"], field["chunks"],
                field["chunk_max"], field["pde_launches"],
                field["tail_launches"], field["shared_buffers"],
                field["contended"], measurement_status, record_class,
                ncu_log, log_record } END {
            if (records != 1) exit 3 } ' "$log")"; then
        fail "expected exactly one MPK_VERTICAL record: $log"
    fi
    printf '%s\n' "$row" >> "$ROWS"

    sed -n 's/^  vertical verdict: /  /p' "$log" \
        | sed "s|^|  tile=$tile n=$n s=$width option=$option verdict:|"
}

echo "==============================================================================="
echo " CA matrix-powers vertical coordinate and tile sweep"
echo "==============================================================================="
echo "  output   : $OUT_DIR"
echo "  device   : $DEVICE"
echo "  repeats  : $REPEATS"
echo "  tiles    : $TILES"
echo "  grids    : $GRIDS"
echo "  widths   : $WIDTHS"
if [[ "$NCU" == "none" ]]; then
    echo "  ncu      : disabled; every row is diagnostic and verdict=undetermined"
else
    echo "  ncu      : $(command -v "$NCU" 2>/dev/null \
        || echo '<not found: every row will be diagnostic>')"
fi
echo

# What each verdict in the rows below will mean. The binary tests these in the
# order printed and takes the first that trips, so a launch-bound point is not
# also reported as DRAM-bound; every threshold is reproducible from the CSV
# columns named beside it.
echo "  verdicts, in the order the binary tests them:"
echo "    launch-bound   launch_fraction >= 0.25: launch overhead alone accounts"
echo "                   for a quarter of the measured time"
echo "    DRAM-bound     dram_fraction >= 0.75: measured DRAM traffic reaches"
echo "                   three quarters of this device's achieved DRAM roof"
echo "    L2-bound       l2_fraction >= 0.75: modeled tiled traffic reaches three"
echo "                   quarters of the achieved L2 roof while DRAM does not"
echo "    latency-bound  no roof is within reach; the access pattern itself is"
echo "                   the cost, and which geometry minimizes it is settled"
echo "                   across rows rather than at any single point"
echo "  and the two non-verdicts:"
echo "    undetermined   no idle-device ncu byte count backs this point, so the"
echo "                   rates stay modeled and no cause is named"
echo "    not-admissible the tile's shared buffer exceeds the device opt-in"
echo "                   limit; the geometry cannot run here and did not fail"
echo

canonical_request=1
if [[ "$(word_set "$TILES")" != "$(word_set "$CANONICAL_TILES")" \
      || "$(word_set "$GRIDS")" != "$(word_set "$CANONICAL_GRIDS")" \
      || "$(word_set "$WIDTHS")" != "$(word_set "$CANONICAL_WIDTHS")" \
      || "$(word_set "$OPTIONS")" != "$(word_set "$CANONICAL_OPTIONS")" ]]; then
    canonical_request=0
fi

expected_rows=0
if [[ $canonical_request -eq 1 ]]; then
    # Canonical production coverage is dense, while the three alternative
    # geometries are measured only at the two predeclared representative points.
    # This is the 44-point plan from the run protocol, encoded here rather than
    # assembled by appending several independently partial CSVs.
    for option in $CANONICAL_OPTIONS; do
        for n in $CANONICAL_GRIDS; do
            for width in $CANONICAL_WIDTHS; do
                expected_rows=$((expected_rows + 1))
                record_point "8x4x4" "$option" "$n" "$width"
            done
        done
    done
    for tile in $CANONICAL_ALTERNATIVE_TILES; do
        for option in $CANONICAL_OPTIONS; do
            for point in $CANONICAL_ALTERNATIVE_POINTS; do
                n="${point%%:*}"
                width="${point##*:}"
                expected_rows=$((expected_rows + 1))
                record_point "$tile" "$option" "$n" "$width"
            done
        done
    done
else
    # Environment-trimmed invocations are smoke/diagnostic cross-products. They
    # remain in their hidden stage and never replace the canonical 44-point CSV.
    for tile in $TILES; do
        for option in $OPTIONS; do
            for n in $GRIDS; do
                for width in $WIDTHS; do
                    expected_rows=$((expected_rows + 1))
                    record_point "$tile" "$option" "$n" "$width"
                done
            done
        done
    done
fi

actual_rows="$(
    awk -F, 'NR > 1 { ++n } END { print n + 0 }' "$ROWS")"
measurement_rows="$(
    awk -F, 'NR > 1 && $44 == "measurement" { ++n } END { print n + 0 }' \
        "$ROWS")"
diagnostic_rows="$(
    awk -F, 'NR > 1 && $44 == "diagnostic" { ++n } END { print n + 0 }' \
        "$ROWS")"
supported_rows="$(
    awk -F, '
        NR > 1 && ($44 == "measurement" || $6 == "not-admissible") { ++n }
        END { print n + 0 }
    ' "$ROWS")"
production_measurement_rows="$(
    awk -F, '
        NR > 1 && $1 == "8x4x4" && $44 == "measurement" { ++n }
        END { print n + 0 }
    ' "$ROWS")"
production_decided_rows="$(
    awk -F, '
        NR > 1 && $1 == "8x4x4" && $44 == "measurement" &&
            $6 != "undetermined" { ++n }
        END { print n + 0 }
    ' "$ROWS")"

canonical_complete=0
if [[ $canonical_request -eq 1 && "$NCU" != "none" \
      && "$REPEATS" =~ ^[0-9]+$ && "$REPEATS" -ge 7 \
      && "$expected_rows" -eq 44 && "$actual_rows" -eq "$expected_rows" \
      && "$supported_rows" -eq "$expected_rows" \
      && "$measurement_rows" -eq "$expected_rows" \
      && "$diagnostic_rows" -eq 0 \
      && "$production_measurement_rows" -eq 32 \
      && "$production_decided_rows" -eq 32 ]]; then
    canonical_complete=1
fi

# The sweep's domain and what it managed to record, beside the CSV. The figure
# re-derives completeness from the CSV itself, so this is for a reader, not a
# gate.
{
    echo "device=$DEVICE"
    echo "repeats=$REPEATS"
    echo "tiles=$TILES"
    echo "grids=$GRIDS"
    echo "widths=$WIDTHS"
    echo "options=$OPTIONS"
    if [[ "$NCU" == "none" ]]; then
        echo "ncu=disabled"
    else
        echo "ncu=$(command -v "$NCU" 2>/dev/null || echo unavailable)"
    fi
    echo "ncu_metric=$NCU_METRIC"
    echo "ncu_profile_mode=one basis evaluation via --profile-only"
    echo "recordable_rule=measurement_status=measured,dram_source=ncu,contended=0"
    echo "canonical_plan=production-full-plus-alternative-representatives"
    echo "canonical_request=$canonical_request"
    echo "canonical_complete=$canonical_complete"
    echo "sweep_expected_rows=$expected_rows"
    echo "sweep_actual_rows=$actual_rows"
    echo "measurement_rows=$measurement_rows"
    echo "diagnostic_rows=$diagnostic_rows"
} > "$OUT_DIR/settings.txt"

echo "==============================================================================="
echo "Vertical sweep attempted."
echo "  expected   : $expected_rows"
echo "  recorded   : $actual_rows"
echo "  measured   : $measurement_rows"
echo "  diagnostic : $diagnostic_rows"
echo "  rows       : $ROWS"
echo "  settings   : $OUT_DIR/settings.txt"
if [[ $canonical_complete -eq 1 ]]; then
    echo "  status     : complete sweep recorded"
else
    echo "  status     : incomplete; the figure will block on the missing points"
fi
echo "Draw figure with:"
echo "  uv run scripts/plots/ca_matrix_powers_roofline.py"
echo "==============================================================================="

if [[ $canonical_request -eq 1 && $canonical_complete -ne 1 ]]; then
    exit 1
fi
