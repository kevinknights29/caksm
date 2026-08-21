#!/usr/bin/env bash
# Capture shared-memory conflicts and full kernel diagnostics on one idle GPU.
# The binaries are built separately so profiler output never mixes with a build.
#
# Promotion profiles for the accepted dispatch, without repeating the closed
# bank-conflict experiment:
#
#   RUN_BANK=0 FULL_FAMILIES="full-volume plane-streamed" \
#   FULL_POINTS="61:3 97:4" \
#   OUT_DIR="$PWD/data/ca-matrix-powers-promotion-profile" \
#   sbatch --nodes=1 --ntasks=10 --partition=compute --time=02:00:00 \
#          --nodelist=synge-n02 --exclusive \
#          scripts/regime/ca_matrix_powers_diagnostics.sh

set -euo pipefail

ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
OUT_DIR="${OUT_DIR:-$ROOT/data/ca-matrix-powers-diagnostics}"
DEVICE="${DEVICE:-0}"
NCU="${NCU:-ncu}"
BANK_N="${BANK_N:-97}"
BANK_WIDTHS="${BANK_WIDTHS:-1 2 3 4}"
OPTIONS="${OPTIONS:-basket rainbow}"
# Full profiles are optional because replay is expensive. Supply production
# points explicitly only when bank-conflict results need more attribution.
FULL_POINTS="${FULL_POINTS:-}"
FULL_FAMILIES="${FULL_FAMILIES:-full-volume}"
# The bank-conflict experiment is already closed. Set this to zero when only
# the promoted plane-streamed candidate and its baseline need full profiles.
RUN_BANK="${RUN_BANK:-1}"
BANK_METRICS="${BANK_METRICS:-l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__t_requests_pipe_lsu_mem_shared_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_shared_op_st.sum}"

BASE="$BUILD_DIR/ca-matrix-powers"
PADDED="$BUILD_DIR/ca-matrix-powers-padded"

fail()
{
    echo "Error: $*" >&2
    echo "Partial diagnostics retained in: $OUT_DIR" >&2
    exit 1
}

if [[ ! -f "$ROOT/CMakeLists.txt" ]]; then
    fail "ROOT is not a caksm checkout: $ROOT"
fi
[[ -x "$BASE" ]] || fail "missing executable: $BASE"
if [[ "$RUN_BANK" == 1 ]]; then
    [[ -x "$PADDED" ]] || fail "missing executable: $PADDED"
elif [[ "$RUN_BANK" != 0 ]]; then
    fail "RUN_BANK must be 0 or 1"
fi
for family in $FULL_FAMILIES; do
    case "$family" in
        full-volume|plane-streamed) ;;
        *) fail "FULL_FAMILIES accepts full-volume and plane-streamed" ;;
    esac
done
command -v "$NCU" >/dev/null 2>&1 || fail "Nsight Compute not found: $NCU"

mkdir -p "$OUT_DIR"
PROVENANCE="$OUT_DIR/provenance.txt"
BANK_MANIFEST="$OUT_DIR/bank_conflicts.csv"
FULL_MANIFEST="$OUT_DIR/full_profiles.csv"

{
    echo "host=$(uname -n)"
    echo "date=$(date -Is)"
    echo "slurm_job_id=${SLURM_JOB_ID:-none}"
    echo "slurm_nodes=${SLURM_JOB_NODELIST:-none}"
    echo "device=$DEVICE"
    echo "ncu=$(command -v "$NCU")"
    echo "bank_metrics=$BANK_METRICS"
    echo "bank_n=$BANK_N"
    echo "bank_widths=$BANK_WIDTHS"
    echo "options=$OPTIONS"
    echo "run_bank=$RUN_BANK"
    echo "full_points=$FULL_POINTS"
    echo "full_families=$FULL_FAMILIES"
    echo "kernel_filters=regex:.*mpk_tiled_kernel.* regex:.*mpk_stream_kernel.*"
    echo "build_command=not-run; binaries supplied by a separate build"
    echo "profiled_time_is_attribution_only=true"
} > "$PROVENANCE"

if [[ "$RUN_BANK" == 1 ]]; then
    printf '%s\n' \
        "arm,option,n,s,status,contended,ncu_csv,ncu_report" > "$BANK_MANIFEST"
    for arm in baseline padded; do
        binary="$BASE"
        if [[ "$arm" == "padded" ]]; then
            binary="$PADDED"
        fi
        for option in $OPTIONS; do
            for width in $BANK_WIDTHS; do
                stem="$OUT_DIR/bank_${arm}_${option}_n${BANK_N}_s${width}"
                csv_log="${stem}.csv"
                report="${stem}.ncu-rep"
                cmd=(
                    "$NCU" --target-processes all --kernel-name-base function
                    --kernel-name 'regex:.*mpk_tiled_kernel.*'
                    --metrics "$BANK_METRICS" --page raw --csv
                    --print-units base --export "$report" --force-overwrite
                    "$binary" --device "$DEVICE" --n "$BANK_N" --s "$width"
                    --option "$option" --basis monomial --profile-only
                )
                {
                    printf 'bank_command[%s,%s,%s]=' "$arm" "$option" "$width"
                    printf '%q ' "${cmd[@]}"
                    echo
                } >> "$PROVENANCE"

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
                printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
                    "$arm" "$option" "$BANK_N" "$width" "$status" \
                    "$contended" "$(basename "$csv_log")" \
                    "$(basename "$report")" >> "$BANK_MANIFEST"
                [[ "$status" == "recorded" ]] \
                    || fail "bank-conflict capture failed: $arm $option s=$width"
                echo "  bank arm=$arm option=$option n=$BANK_N s=$width status=$status"
            done
        done
    done
fi

printf '%s\n' \
    "family,stream_height,option,n,s,status,contended,ncu_csv,ncu_report" \
    > "$FULL_MANIFEST"

stream_height_for()
{
    local n="$1" best=4 best_distance=-1 height blocks distance
    for height in 4 8 16 32; do
        blocks=$(((n + height - 1) / height))
        distance=$((blocks > 8 ? blocks - 8 : 8 - blocks))
        if [[ $best_distance -lt 0 || $distance -lt $best_distance ]]; then
            best="$height"
            best_distance="$distance"
        fi
    done
    echo "$best"
}

for family in $FULL_FAMILIES; do
    for option in $OPTIONS; do
        for point in $FULL_POINTS; do
            n="${point%%:*}"
            width="${point##*:}"
            height=0
            kernel_filter='regex:.*mpk_tiled_kernel.*'
            if [[ "$family" == "plane-streamed" ]]; then
                height="$(stream_height_for "$n")"
                kernel_filter='regex:.*mpk_stream_kernel.*'
            fi
            stem="$OUT_DIR/full_${family}_${option}_n${n}_s${width}"
            csv_log="${stem}.csv"
            report="${stem}.ncu-rep"
            cmd=(
                "$NCU" --target-processes all --kernel-name-base function
                --kernel-name "$kernel_filter"
                --set full --page raw --csv --print-units base
                --export "$report" --force-overwrite
                "$BASE" --device "$DEVICE" --n "$n" --s "$width"
                --option "$option" --basis monomial --profile-only
                --kernel-family "$family"
            )
            if [[ "$family" == "plane-streamed" ]]; then
                cmd+=(--stream-height "$height")
            fi
            {
                printf 'full_command[%s,%s,%s,%s]=' \
                    "$family" "$option" "$n" "$width"
                printf '%q ' "${cmd[@]}"
                echo
            } >> "$PROVENANCE"

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
            printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
                "$family" "$height" "$option" "$n" "$width" "$status" \
                "$contended" "$(basename "$csv_log")" \
                "$(basename "$report")" >> "$FULL_MANIFEST"
            [[ "$status" == "recorded" ]] \
                || fail "full capture failed: $family $option n=$n s=$width"
            echo "  full family=$family h=$height option=$option n=$n s=$width status=$status"
        done
    done
done

echo "Matrix-powers diagnostics complete"
if [[ "$RUN_BANK" == 1 ]]; then
    echo "  bank manifest : $BANK_MANIFEST"
fi
echo "  full manifest : $FULL_MANIFEST"
echo "  provenance    : $PROVENANCE"
