#!/usr/bin/env bash
# Validate actual allocation and adaptive convergence at the preselected
# largest-common grid. These single-repeat runs are diagnostics, not timings.
#
# The expensive referee and distributed correctness phases are allowed only
# when all four one-GPU option/width controls pass this unchanged workload.
#
# Run after ca_largest_common.sh, which writes the grid this reads. One node is
# enough: every run here is a single-GPU control.
#
#   salloc -N 1 -n 10 -p compute -t 01:00:00 --nodelist=synge-n01
#   ./scripts/regime/ca_largest_common_gate.sh
#
# Inside an interactive `srun --pty` shell the parent step already holds the
# allocation and the steps below are refused with "Requested nodes are busy".
# Sharing is fine here, because nothing in this script is a recorded timing:
#
#   SLURM_OVERLAP=1 ./scripts/regime/ca_largest_common_gate.sh
#
# A STOP is an outcome, not a malfunction. The gate exists to refuse the later
# phases, and the closing message names which kind of failure stopped it.

set -euo pipefail

ROOT="${ROOT:-$PWD}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
SINGLE="${SINGLE:-$BUILD_DIR/ca-integrator}"
STATE_DIR="${STATE_DIR:-$ROOT/data/ca-integrator-largest-common}"
SELECTED_FILE="${SELECTED_FILE:-$STATE_DIR/selected_n.txt}"
GATE_DIR="${GATE_DIR:-$STATE_DIR/convergence-gate}"
M="${M:-24}"
STEPS="${STEPS:-100}"
TOL="${TOL:-1e-8}"
TIME_CAP_SECONDS="${TIME_CAP_SECONDS:-1800}"

if [[ ! -x "$SINGLE" ]]; then
    echo "Error: missing executable: $SINGLE"
    exit 1
fi
if [[ ! -s "$SELECTED_FILE" ]]; then
    echo "Error: missing selected grid: $SELECTED_FILE"
    exit 1
fi
if ! command -v srun >/dev/null 2>&1; then
    echo "Error: srun is required."
    exit 1
fi

N="$(tr -d '[:space:]' < "$SELECTED_FILE")"
if [[ ! "$N" =~ ^[0-9]+$ ]] || (( N < 3 || N % 2 == 0 )); then
    echo "Error: selected_n must be an odd integer >= 3."
    exit 1
fi

mkdir -p "$GATE_DIR"
RESULTS="$GATE_DIR/results.csv"
printf '%s\n' \
    "option,s,n,steps,m,tol,exit_status,validation,unconverged,report" \
    > "$RESULTS"

failures=0
# A numerical failure and a scheduler expiration both stop the gate, but they are
# not the same evidence: one is a result about the method, the other is a result
# about the queue. Counting them together makes the artifact and the report
# disagree on their face, so they are counted apart.
numerical_failures=0
scheduler_expirations=0
numerical_arms=""
scheduler_arms=""
for option in basket rainbow; do
    # The memory-limiting production width is first so an immediate numerical
    # stop still validates the selected grid's actual device allocation.
    for width in 4 1; do
        report="$GATE_DIR/${option}_s${width}.txt"
        echo "### largest-common gate option=$option s=$width n=$N"

        command=(
            srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --mpi=none
            "$SINGLE"
            --n "$N" --steps "$STEPS" --repeats 1
            --tol "$TOL" --m "$M" --s "$width"
            --basis monomial --orth cholqr2 --option "$option"
        )

        set +e
        if command -v timeout >/dev/null 2>&1; then
            timeout --signal=TERM --kill-after=30 \
                "$TIME_CAP_SECONDS" "${command[@]}" 2>&1 | tee "$report"
            status="${PIPESTATUS[0]}"
        else
            "${command[@]}" 2>&1 | tee "$report"
            status="${PIPESTATUS[0]}"
        fi
        set -e

        validation="$(
            sed -n 's/^  validation: //p' "$report" | tail -n 1
        )"
        unconverged="$(
            sed -n \
                's/^  Krylov m: .* | unconverged=\([0-9][0-9]*\)$/\1/p' \
                "$report" | tail -n 1
        )"
        validation="${validation:-NOT_REPORTED}"
        unconverged="${unconverged:-NOT_REPORTED}"

        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "$option" "$width" "$N" "$STEPS" "$M" "$TOL" \
            "$status" "$validation" "$unconverged" "$report" \
            >> "$RESULTS"

        if [[ "$status" -ne 0 || "$validation" != "PASS" \
              || "$unconverged" != "0" ]]; then
            failures=$((failures + 1))
            # timeout returns 124, and a canceled step leaves 137 or 143.
            if [[ "$status" == "124" || "$status" == "137" \
                  || "$status" == "143" ]]; then
                scheduler_expirations=$((scheduler_expirations + 1))
                scheduler_arms="${scheduler_arms:+$scheduler_arms;}$option/s$width"
            else
                numerical_failures=$((numerical_failures + 1))
                numerical_arms="${numerical_arms:+$numerical_arms;}$option/s$width"
            fi
            # This is a gate, not a parameter sweep. Once one unchanged
            # production control fails, referee generation and every later
            # arm are forbidden; stop promptly and preserve allocation time.
            break 2
        fi
    done
done

{
    echo "selected_n=$N"
    echo "steps=$STEPS"
    echo "m=$M"
    echo "tol=$TOL"
    echo "time_cap_seconds=$TIME_CAP_SECONDS"
    echo "failures=$failures"
    echo "numerical_failures=$numerical_failures"
    echo "scheduler_expirations=$scheduler_expirations"
    echo "numerical_failure_arms=${numerical_arms:-none}"
    echo "scheduler_expiration_arms=${scheduler_arms:-none}"
    if (( failures == 0 )); then
        echo "decision=PROCEED_TO_REFEREE_AND_DISTRIBUTED_CORRECTNESS"
    else
        echo "decision=STOP_AT_PREDECLARED_CONVERGENCE_OR_ALLOCATION_GATE"
    fi
} > "$GATE_DIR/decision.txt"

echo
if (( failures == 0 )); then
    echo "Largest-common allocation/convergence gate: PASS"
    echo "Proceed to referee generation and distributed correctness."
    exit 0
fi

echo "Largest-common allocation/convergence gate: STOP ($failures failed arm(s))"
if (( numerical_failures > 0 )); then
    echo "  numerical: $numerical_arms"
    echo "    The grid allocated, then did not converge at the unchanged m,"
    echo "    steps and tolerance. That is a result about the method. Changing"
    echo "    any of those to make it converge is a new arm, not this one."
fi
if (( scheduler_expirations > 0 )); then
    echo "  scheduler: $scheduler_arms"
    echo "    Hit the ${TIME_CAP_SECONDS}s cap rather than failing numerically."
    echo "    Raise TIME_CAP_SECONDS and re-run; nothing was learned about the"
    echo "    method here."
fi
echo "  decision: $GATE_DIR/decision.txt"
echo "No referee or recordable timing should be started."
exit 1
