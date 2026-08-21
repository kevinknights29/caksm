#!/usr/bin/env bash
# Screen matrix-powers geometry independently of threads per block, then cross
# the shortlisted geometries with the thread axis. Canonical cases select and
# held-out grids and bases confirm; the two sets remain separate. Runs are
# interleaved, using 3, 7, or 15 repeats for screen, shortlist, or final stages.
#
# Configurations come from data/ca-matrix-powers-tuning/configurations.csv,
# which also records why every rejected shape was excluded. Regenerate it and
# the build variants together with:
#
#   python3 scripts/regime/ca_matrix_powers_configurations.py --stage 1
#
# Run under the batch scheduler on one exclusive Synge node:
#
#   sbatch --nodes=1 --ntasks=10 --partition=compute --time=04:00:00 \
#          --nodelist=synge-n01 --exclusive \
#          scripts/regime/ca_matrix_powers_tuning.sh
#
# After the kernel and canonical solver gates close, run only the held-out
# whole-solver confirmation without repeating the kernel sweep:
#
#   KERNEL_SWEEP=0 SOLVER_CASE_SETS=confirmation OUTPUT_LABEL=solver-confirmation \
#   SOLVER_FAMILIES="full-volume auto" STAGE=final \
#   sbatch --nodes=1 --ntasks=10 --partition=compute --time=02:00:00 \
#          --nodelist=synge-n02 --exclusive \
#          scripts/regime/ca_matrix_powers_tuning.sh

set -uo pipefail

# Where the checkout is. Under sbatch this script runs as a private copy from
# /var/spool, so its own path is not a repository path and SLURM_SUBMIT_DIR is
# the authoritative one. Under an interactive salloc the opposite holds:
# SLURM_SUBMIT_DIR points at wherever salloc was invoked, which is usually not
# the checkout the caller has since changed into. Neither variable is right on
# its own, so the candidates are tried in order and the first that actually is a
# checkout wins. An explicit ROOT is honored strictly rather than falling back,
# because an escape hatch that silently goes somewhere else is worse than none.
caksm_checkout()
{
    [[ -n "$1" && -f "$1/CMakeLists.txt" && -d "$1/scripts/regime" ]]
}

CAKSM_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${ROOT:-}" ]]; then
    if ! caksm_checkout "$ROOT"; then
        echo "Error: ROOT=$ROOT is not a caksm checkout" >&2
        exit 1
    fi
else
    for candidate in "$PWD" "${SLURM_SUBMIT_DIR:-}" "$CAKSM_SCRIPT_DIR/../.."; do
        if caksm_checkout "$candidate"; then
            ROOT="$(cd -- "$candidate" && pwd)"
            break
        fi
    done
    if [[ -z "${ROOT:-}" ]]; then
        echo "Error: no caksm checkout found" >&2
        echo "  tried PWD=$PWD" >&2
        echo "  tried SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}" >&2
        echo "  tried $CAKSM_SCRIPT_DIR/../.." >&2
        echo "  run from the repository root, or export ROOT=/absolute/path/to/caksm" >&2
        exit 1
    fi
fi
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
OUT_DIR="${OUT_DIR:-$ROOT/data/ca-matrix-powers-tuning}"
TABLE="${TABLE:-$OUT_DIR/configurations.csv}"
DEVICE="${DEVICE:-0}"
# screen, shortlist, or final. Sets the repeat count the protocol asks for.
STAGE="${STAGE:-screen}"
# File label may differ from the repeat-count stage. This lets a solver-only
# confirmation run retain its own CSV and provenance instead of truncating the
# final selection evidence.
OUTPUT_LABEL="${OUTPUT_LABEL:-$STAGE}"
# Set to zero after the kernel shortlist is closed and only the end-to-end
# solver confirmation remains.
KERNEL_SWEEP="${KERNEL_SWEEP:-1}"
# Segment lengths the streamed family is swept over. Zero streams the whole z
# extent, which is its lowest-redundancy and lowest-parallelism end.
STREAM_HEIGHTS="${STREAM_HEIGHTS:-4 8 16 32 0}"
# Restrict the sweep to named configurations. Empty means every accepted row.
CONFIGURATIONS="${CONFIGURATIONS:-}"
# Kernel families the solver arm compares, on one binary. The solver selects a
# family at run time now, so an end-to-end comparison no longer needs a binary
# per configuration: the same executable runs both arms back to back, which also
# removes the build as a variable between them. Empty skips the solver arm,
# because section 12.2 is entered only after the kernel gate is cleared.
SOLVER_FAMILIES="${SOLVER_FAMILIES:-}"
SOLVER_BINARY="${SOLVER_BINARY:-$BUILD_DIR/ca-integrator}"
# Which predeclared case sets the solver arm runs. Kernel selection continues to
# use both sets according to STAGE; this switch exists for the independent
# whole-solver confirmation required by the production promotion rule.
SOLVER_CASE_SETS="${SOLVER_CASE_SETS:-selection}"
# Time steps and Krylov dimension. Both matter for the gate: a large time step in
# front of a bounded Krylov dimension does not converge, and a timing on a solve
# that did not converge is not the quantity section 12.2 asks about, however
# reproducible it is. Keep 100 steps fixed across grids in the m_max=39 rerun;
# changing the step count after a convergence result would define another arm.
SOLVER_STEPS="${SOLVER_STEPS:-100}"
# Krylov dimension. The default matches CAKSM_GPU_CA_MAX_M in the production
# build. A lower experiment value needs no rebuild; a higher value is rejected
# by the solver because the four projected-exponential arrays are statically
# sized at compile time.
SOLVER_M="${SOLVER_M:-39}"

case "$STAGE" in
    screen)    REPEATS="${REPEATS:-3}"  ;;
    shortlist) REPEATS="${REPEATS:-7}"  ;;
    final)     REPEATS="${REPEATS:-15}" ;;
    *)
        echo "Error: STAGE must be screen, shortlist, or final" >&2
        exit 1
        ;;
esac

# The four canonical selection cases, and the held-out confirmation cases.
# Encoded as option:n:s:basis so one loop covers both sets and the label is
# carried into the row rather than reconstructed from the values.
SELECTION_CASES="${SELECTION_CASES:-\
basket:61:3:monomial rainbow:61:3:monomial \
basket:97:4:monomial rainbow:97:4:monomial}"
CONFIRMATION_CASES="${CONFIRMATION_CASES:-\
basket:31:1:monomial rainbow:31:1:monomial \
basket:77:3:monomial rainbow:77:3:monomial \
basket:31:3:chebyshev basket:61:3:chebyshev \
basket:31:3:newton basket:61:3:newton}"

case "$KERNEL_SWEEP" in
    0|1) ;;
    *)
        echo "Error: KERNEL_SWEEP must be 0 or 1" >&2
        exit 1
        ;;
esac
for solver_set in $SOLVER_CASE_SETS; do
    case "$solver_set" in
        selection|confirmation) ;;
        *)
            echo "Error: SOLVER_CASE_SETS accepts selection and confirmation" >&2
            exit 1
            ;;
    esac
done

fail()
{
    echo "Error: $*" >&2
    echo "Partial tuning evidence retained in: $OUT_DIR" >&2
    exit 1
}

[[ -f "$TABLE" ]] \
    || fail "configuration table missing: $TABLE (run ca_matrix_powers_configurations.py)"

mkdir -p "$OUT_DIR"
# One file per stage. A screen and a shortlist answer different questions at
# different repeat counts, and a single fixed name meant the second run silently
# truncated the first one's evidence. The figure reads all of them.
ROWS="$OUT_DIR/kernel_timings_${OUTPUT_LABEL}.csv"
SOLVER_ROWS="$OUT_DIR/solver_timings_${OUTPUT_LABEL}.csv"
PROVENANCE="$OUT_DIR/provenance_${OUTPUT_LABEL}.txt"

{
    echo "host=$(uname -n)"
    echo "date=$(date -Is)"
    echo "slurm_job_id=${SLURM_JOB_ID:-none}"
    echo "slurm_nodes=${SLURM_JOB_NODELIST:-none}"
    echo "device=$DEVICE"
    echo "stage=$STAGE"
    echo "output_label=$OUTPUT_LABEL"
    echo "repeats=$REPEATS"
    echo "kernel_sweep=$KERNEL_SWEEP"
    echo "table=$TABLE"
    echo "stream_heights=$STREAM_HEIGHTS"
    echo "selection_cases=$SELECTION_CASES"
    echo "confirmation_cases=$CONFIRMATION_CASES"
    echo "solver_families=$SOLVER_FAMILIES"
    echo "solver_case_sets=$SOLVER_CASE_SETS"
    echo "solver_binary=$SOLVER_BINARY"
    echo "build_command=not-run; binaries supplied by a separate build"
    echo "interleaved=baseline before every candidate case"
} > "$PROVENANCE"

# The row schema. measurement_set is what keeps selection and confirmation
# apart, and status separates a numerical failure from a geometry the device
# will not admit at that width.
printf '%s\n' \
    "measurement_set,stage,configuration,family,option,basis,n,s,stream_height,status,tile_x,tile_y,tile_z,threads_per_block,shared_pad_x,pitch_alignment,register_cap,elide_basis_barrier,restrict,cuda_runtime,gpu_name,build_type,sm,registers_per_thread,local_frame_bytes,static_shared_bytes,dynamic_shared_bytes,shared_optin_bytes,median_ms,min_ms,max_ms,repetitions,useful_points,staged_points,redundant_fraction,points_per_thread,blocks,active_blocks_per_sm,occupancy,waves,multiprocessors,pde_launches,tail_launches,chunks,barriers_per_block,worst_rel,worst_abs,padding_residual,correctness_status,contended,log,reported_configuration" \
    > "$ROWS"

# Accepted configurations and the binary each one was built as. Read from the
# table rather than from the build directory, so a configuration that failed to
# build is a missing binary and an explicit error, not a silently shorter sweep.
mapfile -t ACCEPTED < <(
    awk -F, '
    # A carriage return on the last field would make that header name unmatchable
    # and its column index empty, which awk then reads as the whole record.
    { sub(/\r$/, "") }
    NR == 1 {
        for (q = 1; q <= NF; ++q) column[$q] = q
        for (name in required)
            if (!(name in column))
                missing = missing " " name
        # Reported from the header rather than at the end: a missing name gives
        # an empty column index, and the first data row would reference it and
        # abort with awk complaining about the field rather than the header.
        if (missing != "") { print "MISSING:" missing; exit 3 }
        next
    }
    $column["accepted"] == 1 {
        printf "%s|%s|%s\n", $column["configuration"], $column["kernel_family"],
               $column["build_variant"]
    }
    BEGIN {
        required["configuration"]; required["kernel_family"]
        required["build_variant"]; required["accepted"]
    }
    ' "$TABLE")

# A header the runner cannot read is a stale or malformed table, not an empty
# sweep, and the two need different fixes.
for entry in "${ACCEPTED[@]}"; do
    if [[ "$entry" == MISSING:* ]]; then
        fail "$TABLE has no column(s):${entry#MISSING:}
  Regenerate it with scripts/regime/ca_matrix_powers_configurations.py"
    fi
done

[[ ${#ACCEPTED[@]} -gt 0 ]] || fail "no accepted configurations in $TABLE"

binary_for()
{
    local variant="$1"
    if [[ "$variant" == "ca-matrix-powers" ]]; then
        echo "$BUILD_DIR/ca-matrix-powers"
    else
        echo "$BUILD_DIR/ca-matrix-powers-$variant"
    fi
}

# One point: run it, parse its MPK_TUNING record, and append a row. A geometry
# the device refuses at this width exits 2 and is recorded as not admissible,
# which is a property of the shape rather than a failure of the run.
record_point()
{
    local set_label="$1" configuration="$2" family="$3" binary="$4"
    local option="$5" n="$6" width="$7" basis="$8" height="$9"

    local stem="$OUT_DIR/${set_label}_${configuration}_${option}_${basis}_n${n}_s${width}"
    if [[ "$family" == "plane-streamed" ]]; then
        stem="${stem}_h${height}"
    fi
    local log="${stem}.txt"
    local log_record
    log_record="$(basename "$log")"

    local command=(
        "$binary" --device "$DEVICE" --n "$n" --s "$width"
        --option "$option" --basis "$basis" --repeats "$REPEATS"
        --kernel-family "$family"
    )
    if [[ "$family" == "plane-streamed" ]]; then
        command+=(--stream-height "$height")
    fi

    "${command[@]}" > "$log" 2>&1
    local status=$?

    if [[ $status -eq 2 ]]; then
        # Column count matches the header; only the identifying fields and the
        # status are known for a shape the device will not admit.
        {
            printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,not-admissible' \
                "$set_label" "$STAGE" "$configuration" "$family" \
                "$option" "$basis" "$n" "$width" "$height"
            local column
            for ((column = 11; column <= 50; ++column)); do
                printf ','
            done
            # The trailing empty field is reported_configuration: a geometry the
            # device refused never ran, so it reported nothing to record.
            printf ',%s,\n' "$log_record"
        } >> "$ROWS"
        echo "  $set_label $configuration $option $basis n=$n s=$width h=$height: not admissible"
        return
    fi
    if [[ $status -ne 0 ]]; then
        echo "FAILED: $binary exited $status; see $log" >&2
        sed -n '1,20p' "$log" >&2
        fail "matrix-powers run exited $status: $log"
    fi

    local row
    # The configuration column is the name the sweep asked for, not the one the
    # binary reports. One binary can serve several logical configurations: the
    # default executable is simultaneously the full-volume baseline, the tile it
    # was built on, and the streamed default. Recording its self-report here
    # would collapse those onto one label and lose whichever of them was not the
    # binary's own name. The self-report is kept as its own column instead, so a
    # stale or mismatched binary is still visible in the record.
    if ! row="$(awk -v set_label="$set_label" -v stage="$STAGE" \
        -v requested="$configuration" \
        -v height="$height" -v log_record="$log_record" '
        /^MPK_TUNING / {
            ++records
            for (q = 2; q <= NF; ++q) {
                split($q, kv, "=")
                field[kv[1]] = kv[2]
            }
            printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,measured,", \
                set_label, stage, requested, field["family"], \
                field["option"], field["basis"], field["n"], field["s"], height
            printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,", \
                field["tile_x"], field["tile_y"], field["tile_z"], \
                field["threads_per_block"], field["shared_pad_x"], \
                field["pitch_alignment"], field["register_cap"], \
                field["elide_basis_barrier"], field["restrict"]
            printf "%s,%s,%s,%s,", \
                field["cuda_runtime"], field["gpu_name"], \
                field["build_type"], field["sm"]
            printf "%s,%s,%s,%s,%s,", \
                field["registers_per_thread"], field["local_frame_bytes"], \
                field["static_shared_bytes"], field["dynamic_shared_bytes"], \
                field["shared_optin_bytes"]
            printf "%s,%s,%s,%s,", \
                field["median_ms"], field["min_ms"], field["max_ms"], \
                field["repetitions"]
            printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,", \
                field["useful_points"], field["staged_points"], \
                field["redundant_fraction"], field["points_per_thread"], \
                field["blocks"], field["active_blocks_per_sm"], \
                field["occupancy"], field["waves"], field["multiprocessors"]
            printf "%s,%s,%s,%s,", \
                field["pde_launches"], field["tail_launches"], \
                field["chunks"], field["barriers_per_block"]
            printf "%s,%s,%s,%s,%s,%s,%s\n", \
                field["worst_rel"], field["worst_abs"], \
                field["padding_residual"], field["correctness_status"], \
                field["contended"], log_record, field["configuration"]
        } END {
            if (records != 1) exit 3 }' "$log")"; then
        fail "expected exactly one MPK_TUNING record: $log"
    fi
    printf '%s\n' "$row" >> "$ROWS"

    local median
    median="$(awk -F, '{ print $29 }' <<< "$row")"
    echo "  $set_label $configuration $option $basis n=$n s=$width h=$height: ${median} ms"
}

# Every case a configuration runs, in the requested set. The streamed family
# additionally sweeps its segment length, which costs no shared memory and so
# needs no second binary.
run_case_set()
{
    local set_label="$1" cases="$2" configuration="$3" family="$4" binary="$5"
    local case option n width basis height
    for case in $cases; do
        IFS=: read -r option n width basis <<< "$case"
        if [[ "$family" == "plane-streamed" ]]; then
            for height in $STREAM_HEIGHTS; do
                record_point "$set_label" "$configuration" "$family" \
                    "$binary" "$option" "$n" "$width" "$basis" "$height"
            done
        else
            record_point "$set_label" "$configuration" "$family" \
                "$binary" "$option" "$n" "$width" "$basis" 0
        fi
    done
}

echo "==============================================================================="
echo " CA matrix-powers tile, thread, and segment tuning"
echo "==============================================================================="
echo "  output        : $OUT_DIR"
echo "  device        : $DEVICE"
echo "  stage         : $STAGE ($REPEATS repeats)"
echo "  kernel sweep  : $KERNEL_SWEEP"
echo "  configurations: ${#ACCEPTED[@]} accepted in $TABLE"
echo

BASELINE_BINARY="$BUILD_DIR/ca-matrix-powers"
if [[ "$KERNEL_SWEEP" == 1 ]]; then
    [[ -x "$BASELINE_BINARY" ]] || fail "missing baseline binary: $BASELINE_BINARY"
fi

attempted=0
if [[ "$KERNEL_SWEEP" == 1 ]]; then
    for entry in "${ACCEPTED[@]}"; do
        IFS='|' read -r configuration family variant <<< "$entry"
        if [[ -n "$CONFIGURATIONS" ]]; then
            case " $CONFIGURATIONS " in
                *" $configuration "*) ;;
                *) continue ;;
            esac
        fi
        binary="$(binary_for "$variant")"
        [[ -x "$binary" ]] \
            || fail "configuration $configuration is accepted but its binary is missing: $binary"

        # Interleave: the baseline is re-measured immediately before each candidate,
        # so a clock or temperature drift over the sweep moves both together instead
        # of accumulating against whichever configuration ran last.
        run_case_set "selection" "$SELECTION_CASES" "baseline" "full-volume" \
            "$BASELINE_BINARY"
        run_case_set "selection" "$SELECTION_CASES" "$configuration" "$family" \
            "$binary"
        if [[ "$STAGE" != "screen" ]]; then
            # The baseline runs on the confirmation cases too. A speedup is a ratio
            # against the same case, so without this the confirmation rows have no
            # denominator and the held-out comparison the stage exists to make
            # cannot be formed at all.
            run_case_set "confirmation" "$CONFIRMATION_CASES" "baseline" \
                "full-volume" "$BASELINE_BINARY"
            run_case_set "confirmation" "$CONFIRMATION_CASES" "$configuration" \
                "$family" "$binary"
        fi
        attempted=$((attempted + 1))
    done
fi

# The solver arm: the end-to-end measurement section 12.2 gates promotion on.
# The kernel win is 1.66x, but matrix-powers is only part of a cycle, so what
# matters here is what survives Amdahl. Both families run on one binary, back to
# back within a case, so the build is not a variable between the arms and drift
# over the run lands on both.
if [[ -n "$SOLVER_FAMILIES" ]]; then
    [[ -x "$SOLVER_BINARY" ]] || fail "no solver binary: $SOLVER_BINARY"
    # resolved_family is what the run actually dispatched to, which differs from
    # the requested one under auto. The ratio must be formed against the
    # resolved arm, not the requested one.
    printf '%s\n' \
        "measurement_set,requested_family,resolved_family,stream_height,option,basis,n,s,status,validation,exit_status,solve_median_ms,solve_min_ms,solve_max_ms,cycle_ms_per_step,repeats,tile,threads_per_block,log,steps,krylov_m" \
        > "$SOLVER_ROWS"
    for set_label in $SOLVER_CASE_SETS; do
        if [[ "$set_label" == selection ]]; then
            solver_cases="$SELECTION_CASES"
        else
            solver_cases="$CONFIRMATION_CASES"
        fi
        for case in $solver_cases; do
            IFS=: read -r option n width basis <<< "$case"
            for family in $SOLVER_FAMILIES; do
                log="$OUT_DIR/solver_${set_label}_${family}_${option}_${basis}_n${n}_s${width}.txt"
                steps="$SOLVER_STEPS"
                "$SOLVER_BINARY" --device "$DEVICE" --n "$n" --s "$width" \
                    --m "$SOLVER_M" \
                    --option "$option" --basis "$basis" --steps "$steps" \
                    --repeats "$REPEATS" --kernel-family "$family" > "$log" 2>&1
                status=$?
                # A non-zero exit is not necessarily a lost measurement. The solver
                # prints its validation verdict and its timing distribution, then
                # returns failure, so a solve that did not converge still carries a
                # complete and comparable time. Only a run with no distribution line
                # lost its data, which is also how a contended run presents itself
                # because the solver withholds timing there. The verdict gets its own
                # column, so a time can never be read as accepted when the solve was
                # not.
                if ! grep -q 'distribution: \[' "$log"; then
                    printf '%s,%s,,,%s,%s,%s,%s,no-timing,,%s,,,,,,,,%s,%s,%s\n' \
                        "$set_label" "$family" "$option" "$basis" "$n" "$width" \
                        "$status" "$(basename "$log")" "$steps" "$SOLVER_M" \
                        >> "$SOLVER_ROWS"
                    echo "  solver $set_label $family $option $basis n=$n: exited $status, no timing"
                    continue
                fi
                # The integrator mode reports its timing as prose rather than as a
                # marker line, and the accepted acceptance script already parses that
                # same line. Reading it here too keeps one definition of what a solver
                # timing is, instead of adding a second and letting them drift.
                row="$(awk -v set_label="$set_label" \
                    -v requested="$family" -v option="$option" \
                    -v basis="$basis" -v n="$n" -v width="$width" \
                    -v exit_status="$status" -v steps="$steps" \
                    -v krylov_m="$SOLVER_M" \
                    -v log_record="$(basename "$log")" '
                /^MPK_LAUNCH / {
                    for (q = 2; q <= NF; ++q) {
                        split($q, kv, "="); launch[kv[1]] = kv[2]
                    }
                }
                /^  validation: / {
                    split($0, verdict_fields, " ")
                    verdict = verdict_fields[2]
                }
                /solve median: / {
                    split($0, part, "|")
                    split(part[1], a, ":"); median = a[2] + 0
                    split(part[2], b, ":"); cycle = b[2] + 0
                    if (match(part[3], /\[[^]]*\]/)) {
                        span = substr(part[3], RSTART + 1, RLENGTH - 2)
                        split(span, ends, ",")
                        low = ends[1] + 0
                        high = ends[2] + 0
                    }
                    if (match(part[3], /[0-9]+ runs/))
                        runs = substr(part[3], RSTART, RLENGTH - 5) + 0
                }
                END {
                    # Assigned before the call rather than inlined: a ternary in
                    # an argument list is not portable across awk versions.
                    if (verdict == "") verdict = "unknown"
                    printf "%s,%s,%s,%s,%s,%s,%s,%s,measured,%s,%s,%.6f,%.6f,%.6f,%.6f,%d,%sx%sx%s,%s,%s,%s,%s\n",
                        set_label, requested, launch["family"], launch["stream_height"],
                        option, basis, n, width,
                        verdict, exit_status,
                        median, low, high, cycle, runs,
                        launch["tile_x"], launch["tile_y"], launch["tile_z"],
                        launch["threads_per_block"], log_record, steps, krylov_m
                    }' "$log")"
                printf '%s\n' "$row" >> "$SOLVER_ROWS"
                echo "  solver $set_label $family $option $basis n=$n:" \
                     "$(awk -F, '{ print $12 }' <<< "$row") ms" \
                     "(validation $(awk -F, '{ print $10 }' <<< "$row"))"
            done
        done
    done
fi

rows="$(awk -F, 'NR > 1 { ++n } END { print n + 0 }' "$ROWS")"
measured="$(awk -F, 'NR > 1 && $10 == "measured" { ++n } END { print n + 0 }' "$ROWS")"
inadmissible="$(awk -F, 'NR > 1 && $10 == "not-admissible" { ++n } END { print n + 0 }' "$ROWS")"
contended="$(awk -F, 'NR > 1 && $50 == 1 { ++n } END { print n + 0 }' "$ROWS")"

{
    echo "configurations_run=$attempted"
    echo "rows=$rows"
    echo "measured_rows=$measured"
    echo "not_admissible_rows=$inadmissible"
    echo "contended_rows=$contended"
} >> "$PROVENANCE"

echo "==============================================================================="
echo "Tuning sweep complete."
echo "  configurations : $attempted"
echo "  rows           : $rows ($measured measured, $inadmissible not admissible)"
echo "  kernel CSV     : $ROWS"
if [[ -f "$SOLVER_ROWS" ]]; then
    echo "  solver CSV     : $SOLVER_ROWS"
    # A solve that did not validate still produced a comparable time, and the
    # rows keep it. Saying so here as well is what stops it becoming a headline
    # figure by the time anyone reads the CSV.
    solver_unvalidated="$(awk -F, '
        NR > 1 && $10 != "PASS" { ++n } END { print n + 0 }' "$SOLVER_ROWS")"
    if [[ "$solver_unvalidated" -gt 0 ]]; then
        echo
        echo "  WARNING: $solver_unvalidated solver row(s) did not pass validation."
        echo "  Their timings are recorded and are comparable between arms, but"
        echo "  they are not a whole-solver result: section 12.2 gates on a run"
        echo "  that converged. Check the validation column before quoting them."
    fi
fi
echo "  provenance     : $PROVENANCE"
echo "==============================================================================="

if [[ "$contended" -gt 0 ]]; then
    echo "Warning: $contended rows were measured on a contended device and are" >&2
    echo "not comparable with the rest. Rerun those points on an idle node." >&2
    exit 1
fi
