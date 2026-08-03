#!/usr/bin/env bash
# Do the one-GPU and the one-slab solvers agree, once every reason they might
# not has been removed?
#
# Four questions, all answerable without a second device, which is why this is a
# one-GPU/one-slab test and not a performance sweep. Both executables see the
# same device, problem, basis, final time and seven repeats.
#
#   Do the two paths dispatch the same operator kernel? They used to chunk the
#   recurrence differently, so their states are compared to 1e-13.
#
#   Do they integrate to the same final time? The distributed solver once
#   hardcoded the operator scale as 1/steps while the one-GPU solver used
#   expiry/steps, which agree only at expiry 1. EXPIRY is therefore non-unit
#   here, and a unit value is rejected.
#
#   Does deferring the certificate read change anything? The immediate and
#   deferred arms must make identical certificate decisions and save bitwise
#   identical states.
#
#   Are the two comparable at all? Their timing intervals must overlap on an
#   idle device.
#
# Cross-rank agreement is vacuous with one rank and is not exercised here.
# ca_exact_depth_weak.sh covers it, where two global GPU ranks exist.
#
# Run under the batch scheduler on one exclusive Synge node. Nothing here is
# recorded as a timing, but the fourth question compares two distributions for
# overlap, and that comparison only means anything on an idle device: a
# contended GPU widens one distribution and not the other, and the gate then
# reports cycle_distributions=DISJOINT for a pair that never disagreed. A false
# failure costs a re-run of the whole gate, so this is the recommended route:
#
#   sbatch --nodes=1 --ntasks=10 --partition=compute --time=01:00:00 \
#          --nodelist=synge-n01 --exclusive \
#          scripts/regime/ca_integrator_acceptance.sh
#
# Interactively, for a smoke test only. The runs below share the calling shell's
# CPUs and whatever else holds the device, which is the usual cause of a false DISJOINT:
#
#   salloc -N 1 -n 10 -p compute -t 01:00:00 --nodelist=synge-n01
#   ./scripts/regime/ca_integrator_acceptance.sh

set -euo pipefail

ROOT="${ROOT:-$PWD}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
SINGLE="${SINGLE:-$BUILD_DIR/ca-integrator}"
DISTRIBUTED="${DISTRIBUTED:-$BUILD_DIR/ca-integrator-2gpu}"
OUT_DIR="${OUT_DIR:-$ROOT/data/ca-integrator-acceptance}"
DEVICE="${DEVICE:-0}"
N="${N:-31}"
M="${M:-24}"
STEPS="${STEPS:-100}"
TOL="${TOL:-1e-8}"
STATE_TOL="${STATE_TOL:-1e-13}"
EXPIRY="${EXPIRY:-0.5}"
REPEATS="${REPEATS:-7}"
OPTIONS="${OPTIONS:-basket rainbow}"
WIDTHS="${WIDTHS:-1 4}"

for binary in "$SINGLE" "$DISTRIBUTED"; do
    if [[ ! -x "$binary" ]]; then
        echo "Error: missing executable: $binary"
        exit 1
    fi
done
if [[ "$EXPIRY" == "1" || "$EXPIRY" == "1.0" ]]; then
    echo "Error: EXPIRY must be non-unit; at expiry 1 the two operator"
    echo "       scales agree and the check proves nothing."
    exit 1
fi
if [[ "$REPEATS" -lt 7 ]]; then
    echo "Error: correction acceptance requires at least seven repeats."
    exit 1
fi

mkdir -p "$OUT_DIR"
mkdir -p "$OUT_DIR/states"
FAILURES=0

run_logged() {
    local log="$1"
    shift
    set +e
    "$@" 2>&1 | tee "$log"
    local exit_code="${PIPESTATUS[0]}"
    set -e
    if [[ $exit_code -ne 0 ]]; then
        echo "ACCEPTANCE_RESULT status=FAIL exit_code=$exit_code" | tee -a "$log"
        FAILURES=$((FAILURES + 1))
        return 1
    fi
    echo "ACCEPTANCE_RESULT status=PASS exit_code=0" | tee -a "$log"
    return 0
}

timing_range() {
    sed -n \
        's/.*distribution: \[\([^,]*\), \([^]]*\)\] ms.*/\1 \2/p' \
        "$1" | tail -n 1
}

intervals_overlap() {
    local first_log="$1"
    local second_log="$2"
    local first_range second_range
    first_range="$(timing_range "$first_log")"
    second_range="$(timing_range "$second_log")"
    if [[ -z "$first_range" || -z "$second_range" ]]; then
        return 1
    fi
    awk -v a="$first_range" -v b="$second_range" '
        BEGIN {
            split(a, ar, " ")
            split(b, br, " ")
            exit !((ar[1] + 0.0) <= (br[2] + 0.0) &&
                   (br[1] + 0.0) <= (ar[2] + 0.0))
        }
    '
}

decisions_match() {
    local immediate_log="$1"
    local deferred_log="$2"
    diff -u \
        <(grep -E 'Krylov m:|certified width:|CA blocks:|validation:' \
            "$immediate_log") \
        <(grep -E 'Krylov m:|certified width:|CA blocks:|validation:' \
            "$deferred_log") \
        >/dev/null
}

for option in $OPTIONS; do
    for width in $WIDTHS; do
        stem="${option}_s${width}_expiry${EXPIRY}"
        single_state="$OUT_DIR/states/${stem}_single.bin"
        immediate_state="$OUT_DIR/states/${stem}_slab_immediate.bin"
        deferred_state="$OUT_DIR/states/${stem}_slab_deferred.bin"
        single_log="$OUT_DIR/${stem}_single.txt"
        immediate_log="$OUT_DIR/${stem}_slab_immediate.txt"
        deferred_log="$OUT_DIR/${stem}_slab_deferred.txt"

        common=(
            --n "$N" --m "$M" --s "$width" --steps "$STEPS"
            --tol "$TOL" --repeats "$REPEATS" --option "$option"
            --basis monomial --orth cholqr2 --arm exact-depth
            --expiry "$EXPIRY"
        )

        echo "### one GPU option=$option s=$width expiry=$EXPIRY"
        if ! run_logged "$single_log" \
            "$SINGLE" --device "$DEVICE" "${common[@]}" \
            --save-state "$single_state"; then
            continue
        fi

        echo "### one slab immediate option=$option s=$width expiry=$EXPIRY"
        if ! run_logged "$immediate_log" \
            "$DISTRIBUTED" --devices "$DEVICE" "${common[@]}" \
            --certificate immediate \
            --single-gpu-state "$single_state" \
            --single-state-tol "$STATE_TOL" \
            --save-state "$immediate_state"; then
            continue
        fi

        if ! intervals_overlap "$single_log" "$immediate_log"; then
            echo "ACCEPTANCE_CHECK cycle_distributions=DISJOINT" \
                | tee -a "$immediate_log"
            FAILURES=$((FAILURES + 1))
        else
            echo "ACCEPTANCE_CHECK cycle_distributions=OVERLAP" \
                | tee -a "$immediate_log"
        fi

        echo "### one slab deferred option=$option s=$width expiry=$EXPIRY"
        if ! run_logged "$deferred_log" \
            "$DISTRIBUTED" --devices "$DEVICE" "${common[@]}" \
            --certificate deferred \
            --single-gpu-state "$immediate_state" \
            --single-state-tol "$STATE_TOL" \
            --save-state "$deferred_state"; then
            continue
        fi

        if ! decisions_match "$immediate_log" "$deferred_log"; then
            echo "ACCEPTANCE_CHECK certificate_decisions=MISMATCH" \
                | tee -a "$deferred_log"
            FAILURES=$((FAILURES + 1))
        elif ! cmp -s "$immediate_state" "$deferred_state"; then
            echo "ACCEPTANCE_CHECK certificate_states=MISMATCH" \
                | tee -a "$deferred_log"
            FAILURES=$((FAILURES + 1))
        else
            echo "ACCEPTANCE_CHECK certificate_decisions=MATCH certificate_states=MATCH" \
                | tee -a "$deferred_log"
        fi
    done
done

# The problem the gate ran at, beside its verdict.
{
    echo "device=$DEVICE"
    echo "n=$N"
    echo "m=$M"
    echo "steps=$STEPS"
    echo "tol=$TOL"
    echo "state_tol=$STATE_TOL"
    echo "expiry=$EXPIRY"
    echo "repeats=$REPEATS"
    echo "options=$OPTIONS"
    echo "widths=$WIDTHS"
    echo "failures=$FAILURES"
} > "$OUT_DIR/settings.txt"

# Transcripts carrying a given marker, or nothing. Used to explain only the
# failures a reader actually has rather than handing them the whole legend.
carrying() {
    grep -l "$1" "$OUT_DIR"/*.txt 2>/dev/null || true
}

# failures=0 is the whole claim, so a non-zero count needs to say which of the
# four questions in the header went unanswered and what that implies. Each of
# these is a disagreement the gate exists to catch, not a fault in the harness,
# and the evidence for every one of them is kept.
explain_failures() {
    local logs
    echo
    echo "What failures=$FAILURES means"
    echo
    echo "Each failure is one check disagreeing, not a crash: the run that"
    echo "produced it is on disk, and the marker naming the disagreement is the"
    echo "last line that check appended to its transcript."
    echo

    logs="$(carrying 'ACCEPTANCE_RESULT status=FAIL')"
    if [[ -n "$logs" ]]; then
        echo "  a run exited non-zero"
        echo "    Either the solver stopped on its own, or the one-slab state"
        echo "    left the one-GPU state by more than $STATE_TOL, which is the"
        echo "    two paths dispatching different operator kernels or"
        echo "    integrating to different final times. The transcript's own"
        echo "    error line says which. The remaining checks for that"
        echo "    configuration were skipped, so their markers are absent"
        echo "    rather than passing."
        sed 's/^/      /' <<< "$logs"
        echo
    fi

    logs="$(carrying 'ACCEPTANCE_CHECK cycle_distributions=DISJOINT')"
    if [[ -n "$logs" ]]; then
        echo "  cycle_distributions=DISJOINT"
        echo "    The one-GPU and one-slab timing ranges do not overlap, so the"
        echo "    two are not comparable on this device. This is a"
        echo "    comparability result, not a numerical one: the states still"
        echo "    agreed to $STATE_TOL. A contended or thermally throttled"
        echo "    device is the usual cause, and re-running on an idle one is"
        echo "    the first thing to try."
        sed 's/^/      /' <<< "$logs"
        echo
    fi

    logs="$(carrying 'ACCEPTANCE_CHECK certificate_decisions=MISMATCH')"
    if [[ -n "$logs" ]]; then
        echo "  certificate_decisions=MISMATCH"
        echo "    Deferring the certificate read changed an adaptive decision:"
        echo "    the Krylov dimension, certified width, CA block count or"
        echo "    validation verdict differs between the immediate and deferred"
        echo "    arms. The two arms are meant to decide identically and only"
        echo "    differ in when the verdict is read, so this is a defect in"
        echo "    the deferred path. Diff the two transcripts on those four"
        echo "    lines to see which decision moved."
        sed 's/^/      /' <<< "$logs"
        echo
    fi

    logs="$(carrying 'ACCEPTANCE_CHECK certificate_states=MISMATCH')"
    if [[ -n "$logs" ]]; then
        echo "  certificate_states=MISMATCH"
        echo "    The two arms decided identically but did not save bitwise"
        echo "    identical states. Same decisions and different bits means a"
        echo "    reordering somewhere on the deferred path, which makes it"
        echo "    non-reproducible even where it is accurate."
        sed 's/^/      /' <<< "$logs"
        echo
    fi

    echo "  evidence retained in: $OUT_DIR"
}

echo "==============================================================================="
echo "Correction acceptance complete: failures=$FAILURES"
if [[ $FAILURES -ne 0 ]]; then
    explain_failures
    echo "==============================================================================="
    exit 1
fi

# The ideal outcome, stated rather than implied by a zero, so a reader knows
# what the gate just certified and not merely that nothing tripped.
echo "  every configuration agreed on all four counts:"
echo "    one-GPU and one-slab final states within $STATE_TOL"
echo "    both integrated to expiry $EXPIRY, which is non-unit by construction"
echo "    immediate and deferred certificates decided alike and saved"
echo "      bitwise identical states"
echo "    their cycle-time distributions overlap, so the two are comparable"
echo "  transcripts : $OUT_DIR"
echo "  states      : $OUT_DIR/states"
echo "  settings    : $OUT_DIR/settings.txt"
echo "==============================================================================="
