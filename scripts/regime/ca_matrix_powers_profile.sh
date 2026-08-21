#!/usr/bin/env bash
# Capture matrix-powers load balance, occupancy, and warp stalls with Nsight
# Compute. Profile replay time is attribution data, not a timing measurement.

set -uo pipefail

# Slurm executes a private copy of a batch script from /var/spool, so $0 is not
# a repository path inside an sbatch job. SLURM_SUBMIT_DIR is the authoritative
# location there; an interactive invocation falls back to its current directory.
ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
OUT_DIR="${OUT_DIR:-$ROOT/data/ca-matrix-powers-profile}"
DEVICE="${DEVICE:-0}"
NCU="${NCU:-ncu}"
TILES="${TILES:-8x4x4 16x8x8 8x8x8 16x8x4}"
OPTIONS="${OPTIONS:-basket rainbow}"
POINTS="${POINTS:-61:3 97:4}"
EXPECTED_TILES="8x4x4 16x8x8 8x8x8 16x8x4"
EXPECTED_OPTIONS="basket rainbow"
EXPECTED_POINTS="61:3 97:4"
METRICS="${METRICS:-sm__cycles_active.avg,sm__cycles_active.min,sm__cycles_active.max,sm__cycles_elapsed.max,sm__throughput.avg.pct_of_peak_sustained_elapsed,launch__waves_per_multiprocessor,sm__warps_active.avg.pct_of_peak_sustained_active}"

if [[ ! -f "$ROOT/CMakeLists.txt" || ! -d "$ROOT/scripts/regime" ]]; then
    echo "Error: ROOT is not a caksm checkout: $ROOT" >&2
    echo "Submit from the repository root or export ROOT=/absolute/path/to/caksm." >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
MANIFEST="$OUT_DIR/captures.csv"
PROVENANCE="$OUT_DIR/provenance.txt"

fail() {
    echo "Error: $*" >&2
    echo "Partial profile data retained in: $OUT_DIR" >&2
    exit 1
}

word_set() {
    tr ' ' '\n' <<< "$1" | sed '/^$/d' | LC_ALL=C sort -u | paste -sd' ' -
}

binary_for_tile() {
    if [[ "$1" == "8x4x4" ]]; then
        echo "$BUILD_DIR/ca-matrix-powers"
    else
        echo "$BUILD_DIR/ca-matrix-powers-$1"
    fi
}

{
    echo "host=$(uname -n)"
    echo "date=$(date -Is)"
    echo "slurm_job_id=${SLURM_JOB_ID:-none}"
    echo "slurm_nodes=${SLURM_JOB_NODELIST:-none}"
    echo "device=$DEVICE"
    echo "ncu=$(command -v "$NCU" 2>/dev/null || echo unavailable)"
    echo "ncu_metrics=$METRICS"
    echo "ncu_sections=WarpStateStats,SchedulerStats"
    echo "kernel_filter=regex:.*mpk_tiled_kernel.*"
    echo "tiles=$TILES"
    echo "options=$OPTIONS"
    echo "points=$POINTS"
    echo "build_command=not-run; binaries supplied by a separate build"
    echo "profiled_time_is_attribution_only=true"
} > "$PROVENANCE"

DEVICE_PROBE="$BUILD_DIR/gpu-device-probe"
if [[ -x "$DEVICE_PROBE" ]]; then
    if ! "$DEVICE_PROBE" --csv "$OUT_DIR/gpu_capabilities.csv" \
            > "$OUT_DIR/gpu_capabilities.txt" 2>&1; then
        fail "gpu-device-probe failed"
    fi
else
    printf '%s\n' \
        "gpu-device-probe not found; capability values were not recorded by this run." \
        > "$OUT_DIR/gpu_capabilities.txt"
fi

command -v "$NCU" >/dev/null 2>&1 || fail "Nsight Compute not found: $NCU"

printf '%s\n' "tile,option,n,s,status,contended,ncu_csv,ncu_report" > "$MANIFEST"

attempted=0
recorded=0
for tile in $TILES; do
    binary="$(binary_for_tile "$tile")"
    [[ -x "$binary" ]] || fail "missing matrix-powers binary: $binary"
    for option in $OPTIONS; do
        for point in $POINTS; do
            n="${point%%:*}"
            width="${point##*:}"
            stem="$OUT_DIR/${tile}_${option}_n${n}_s${width}"
            csv_log="${stem}_ncu.csv"
            report="${stem}.ncu-rep"
            cmd=("$NCU" --target-processes all --kernel-name-base function
                 --kernel-name 'regex:.*mpk_tiled_kernel.*'
                 --metrics "$METRICS" --section WarpStateStats
                 --section SchedulerStats --page raw --csv --print-units base
                 --export "$report" --force-overwrite
                 "$binary" --device "$DEVICE" --n "$n" --s "$width"
                 --option "$option" --basis monomial --profile-only)
            {
                printf 'capture_command[%s,%s,%s,%s]=' "$tile" "$option" "$n" "$width"
                printf '%q ' "${cmd[@]}"
                echo
            } >> "$PROVENANCE"

            attempted=$((attempted + 1))
            status="recorded"
            if ! "${cmd[@]}" > "$csv_log" 2>&1; then
                status="ncu-failed"
            elif ! grep -q 'MPK_PROFILE basis_evaluations=1' "$csv_log"; then
                status="profile-marker-missing"
            fi

            contended=0
            if grep -q 'MPK_PROFILE .*contended=1' "$csv_log"; then
                contended=1
                status="contended"
            fi
            if [[ "$status" == "recorded" ]]; then
                recorded=$((recorded + 1))
            fi
            printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
                "$tile" "$option" "$n" "$width" "$status" "$contended" \
                "$(basename "$csv_log")" "$(basename "$report")" >> "$MANIFEST"
            echo "  tile=$tile option=$option n=$n s=$width status=$status"
        done
    done
done

full_sweep=0
if [[ "$(word_set "$TILES")" == "$(word_set "$EXPECTED_TILES")" \
      && "$(word_set "$OPTIONS")" == "$(word_set "$EXPECTED_OPTIONS")" \
      && "$(word_set "$POINTS")" == "$(word_set "$EXPECTED_POINTS")" ]]; then
    full_sweep=1
fi
{
    echo "full_sweep=$full_sweep"
    echo "capture_count=$attempted"
    echo "recorded_count=$recorded"
} >> "$PROVENANCE"

if [[ "$full_sweep" -eq 1 && "$attempted" -ne 16 ]]; then
    fail "full matrix-powers sweep must contain 16 captures, found $attempted"
fi
if [[ "$recorded" -ne "$attempted" ]]; then
    fail "only $recorded of $attempted captures are recordable"
fi

echo "Matrix-powers profile capture complete"
echo "  manifest   : $MANIFEST"
echo "  provenance : $PROVENANCE"
