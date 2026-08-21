#!/usr/bin/env bash
# One-arm postmortem for the largest-common Basket failure. This changes only
# the representation of the three forcing coordinates by the exact power-of-two
# similarity already used by the independent Basket referee. It is deliberately
# separate from the predeclared gate and must not overwrite that gate's record.

set -euo pipefail

ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
SINGLE="${SINGLE:-$BUILD_DIR/ca-integrator}"
OUT_DIR="${OUT_DIR:-$ROOT/data/ca-integrator-largest-common-m39/scaled-augmentation-postmortem}"
N="${N:-337}"
M="${M:-39}"
WIDTH="${WIDTH:-4}"
STEPS="${STEPS:-100}"
TOL="${TOL:-1e-8}"
TIME_CAP_SECONDS="${TIME_CAP_SECONDS:-6900}"

if [[ ! -f "$ROOT/CMakeLists.txt" ]]; then
    echo "Error: ROOT is not the caksm checkout: $ROOT" >&2
    exit 1
fi
if [[ ! -x "$SINGLE" ]]; then
    echo "Error: missing executable: $SINGLE" >&2
    exit 1
fi
if ! command -v srun >/dev/null 2>&1; then
    echo "Error: srun is required." >&2
    exit 1
fi
if [[ ! "$N" =~ ^[0-9]+$ ]] || (( N < 3 || N % 2 == 0 )); then
    echo "Error: N must be an odd integer >= 3." >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
REPORT="$OUT_DIR/basket_s${WIDTH}_n${N}.txt"
DECISION="$OUT_DIR/decision.txt"
PROVENANCE="$OUT_DIR/provenance.txt"

{
    echo "host=$(uname -n)"
    echo "date=$(date -Is)"
    echo "slurm_job_id=${SLURM_JOB_ID:-none}"
    echo "binary=$SINGLE"
    echo "arm=scaled-augmentation"
    echo "option=basket"
    echo "n=$N"
    echo "steps=$STEPS"
    echo "m=$M"
    echo "s=$WIDTH"
    echo "tol=$TOL"
    echo "basis=monomial"
    echo "orth=cholqr2"
    echo "kernel_family=auto"
} > "$PROVENANCE"

echo "### scaled-augmentation postmortem option=basket s=$WIDTH n=$N"
command=(
    srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --mpi=none
    "$SINGLE"
    --n "$N" --steps "$STEPS" --repeats 1
    --tol "$TOL" --m "$M" --s "$WIDTH"
    --basis monomial --orth cholqr2 --option basket
    --arm scaled-augmentation --kernel-family auto
)

set +e
if command -v timeout >/dev/null 2>&1; then
    timeout --signal=TERM --kill-after=30 "$TIME_CAP_SECONDS" \
        "${command[@]}" 2>&1 | tee "$REPORT"
    status="${PIPESTATUS[0]}"
else
    "${command[@]}" 2>&1 | tee "$REPORT"
    status="${PIPESTATUS[0]}"
fi
set -e

validation="$(
    sed -n 's/^  validation: \([^ ]*\).*/\1/p' "$REPORT" | tail -n 1
)"
tail_error="$(
    sed -n 's/^  boundary-state error: //p' "$REPORT" | tail -n 1
)"
augmentation="$(
    sed -n 's/^  augmentation: //p' "$REPORT" | tail -n 1
)"

if [[ "$status" -eq 0 && "$validation" == "PASS" ]]; then
    outcome="SCALED_AUGMENTATION_PASSES_POSTMORTEM"
elif [[ "$status" == "124" || "$status" == "137" \
       || "$status" == "143" ]]; then
    outcome="INCOMPLETE_SCHEDULER_OR_TIME_CAP"
elif [[ "$validation" == "FAIL" ]]; then
    outcome="SCALED_AUGMENTATION_COMPLETED_BUT_FAILED_NUMERICALLY"
else
    outcome="INCOMPLETE_EXECUTION_FAILURE"
fi

{
    echo "outcome=$outcome"
    echo "exit_status=$status"
    echo "validation=${validation:-NOT_REPORTED}"
    echo "boundary_state_error=${tail_error:-NOT_REPORTED}"
    echo "augmentation=${augmentation:-NOT_REPORTED}"
    echo "report=$REPORT"
    echo "provenance=$PROVENANCE"
    echo "interpretation=postmortem arm; does not replace the predeclared largest-common gate"
} > "$DECISION"

echo
echo "Scaled-augmentation postmortem: $outcome"
echo "  report     : $REPORT"
echo "  decision   : $DECISION"
echo "  provenance : $PROVENANCE"

if [[ "$outcome" == INCOMPLETE_* ]]; then
    exit 1
fi
exit 0
