#!/usr/bin/env bash
# Measure the exact-depth arm beside the as-measured arm at fixed local volume.
#
# A block of width s consumes columns 0 to s-1, so it needs s-1 recurrence steps
# and an s-1 deep halo. The as-measured arm ran s of each and threw the last
# column away; the exact-depth arm runs what the block consumes, which at s=1 is
# a copy with no exchange at all. Both arms run here in one job on the same
# devices, because the comparison between them is the finding.
#
# Nothing is overwritten: the as-measured transcripts under
# data/ca-integrator-weak are untouched, and everything written here lands under
# data/ca-integrator-exact-depth.
#
# Run under the batch scheduler on two exclusive Synge nodes. The gap between
# the two arms is the finding, so this is the recommended route:
#
#   sbatch --nodes=2 --ntasks=20 --partition=compute --time=01:00:00 \
#          --nodelist=synge-n01,synge-n02 --exclusive \
#          scripts/regime/ca_exact_depth_weak.sh
#
# Interactively, for a smoke test only. SLURM_OVERLAP shares the calling shell's
# CPUs with every step below, so its timings can vary:
#
#   salloc -N 2 -n 20 -p compute -t 01:00:00 --nodelist=synge-n01,synge-n02
#   SLURM_OVERLAP=1 ./scripts/regime/ca_exact_depth_weak.sh
#
# The integrator withholds its timing distribution if any participating device is
# contended, so a contended run produces correctness evidence and no timings.

set -euo pipefail

ROOT="${ROOT:-$PWD}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
DISTRIBUTED="${DISTRIBUTED:-$BUILD_DIR/ca-integrator-2gpu}"
SINGLE="${SINGLE:-$BUILD_DIR/ca-integrator}"
OUT_DIR="${OUT_DIR:-$ROOT/data/ca-integrator-exact-depth}"
VALIDATION_DIR="${VALIDATION_DIR:-$ROOT/data/ca-integrator-validation}"
REFEREE_ROOT="${REFEREE_ROOT:-$ROOT/data}"
M="${M:-39}"
STEPS="${STEPS:-100}"
TOL="${TOL:-1e-8}"
REPEATS="${REPEATS:-7}"
SRUN_MPI="${SRUN_MPI:-pmix}"
STATE_RTOL="${STATE_RTOL:-1e-10}"

for binary in "$DISTRIBUTED" "$SINGLE"; do
    if [[ ! -x "$binary" ]]; then
        echo "Error: missing executable: $binary"
        exit 1
    fi
done
if ! command -v srun >/dev/null 2>&1; then
    echo "Error: srun is required."
    exit 1
fi
# Slurm names the allocation's node count differently depending on how this
# shell was obtained: sbatch and srun set SLURM_JOB_NUM_NODES, salloc sets
# SLURM_NNODES, and inside an interactive `srun --pty` step both report that
# step's size rather than the allocation's. The nodelist always describes the
# allocation, so that is counted first and fall back to the variables.
allocated_nodes()
{
    if [[ -n "${SLURM_JOB_NODELIST:-}" ]] \
       && command -v scontrol >/dev/null 2>&1; then
        scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null \
            | wc -l | tr -d '[:space:]'
    else
        echo "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-0}}"
    fi
}

report_allocation()
{
    echo "  SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<unset>}" >&2
    echo "  SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-<unset>}" \
         "SLURM_NNODES=${SLURM_NNODES:-<unset>}" >&2
    if [[ -n "${SLURM_STEP_ID:-}" ]]; then
        echo "  SLURM_STEP_ID=${SLURM_STEP_ID}: this shell is itself an srun" >&2
        echo "  step, which cannot launch the wider steps this sweep needs." >&2
        echo "  Leave it and run from the salloc shell owning the allocation." >&2
    fi
}

if [[ "$(allocated_nodes)" -lt 2 ]]; then
    echo "Error: run inside an allocation containing two nodes." >&2
    report_allocation
    exit 1
fi

mkdir -p "$OUT_DIR"
export PMIX_MCA_gds="${PMIX_MCA_gds:-hash}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# The weak-scaling ladder: approximately 227,000 rows per GPU at each rung.
# n=77 on two GPUs and n=97 on four are the rungs the as-measured arm reported.
RUNGS=("77:2" "97:4")
STOPPED_POINTS=0
ACCEPTANCE_FAILURES=0
CONTENDED_POINTS=0
INCOMPLETE_POINTS=0

for rung in "${RUNGS[@]}"; do
    n="${rung%%:*}"
    for option in basket rainbow; do
        single_state="$VALIDATION_DIR/single_n${n}_${option}.bin"
        referee="$REFEREE_ROOT/n${n}/referee_n${n}_${option}.bin"
        if [[ ! -f "$single_state" || ! -f "$referee" ]]; then
            echo "Error: the correction sweep requires both validation artifacts:"
            echo "  $single_state"
            echo "  $referee"
            exit 1
        fi
    done
done

# The sweep's own certificate decisions are only trustworthy if a cross-rank
# disagreement would actually stop the solver, so a disagreement is injected
# first and the named abort is required before any rung is admitted.
run_agreement_self_test() {
    local log="$OUT_DIR/agreement_self_test.txt"

    echo "### cross-rank agreement self-test (expected certificate disagreement)"
    set +e
    srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --mpi=none \
        "$DISTRIBUTED" --devices 0,1 \
        --n 31 --m 8 --s 4 --steps 1 --tol "$TOL" --repeats 1 \
        --option rainbow --basis monomial --orth cholqr2 \
        --arm exact-depth --agreement-self-test \
        2>&1 | tee "$log"
    local exit_code="${PIPESTATUS[0]}"
    set -e

    if [[ $exit_code -eq 0 ]] \
       || ! grep -q \
           'ranks disagreed at the block certificate checkpoint on certificate acceptance' \
           "$log"; then
        echo "AGREEMENT_SELF_TEST status=failed exit_code=$exit_code" \
            | tee -a "$log"
        echo "Error: the agreement self-test did not stop at the named" >&2
        echo "       certificate field, so a real cross-rank disagreement" >&2
        echo "       would not be caught either." >&2
        return 1
    fi
    echo "AGREEMENT_SELF_TEST status=passed expected_exit_code=$exit_code" \
        | tee -a "$log"
    echo
}

record_run_status() {
    local log="$1"
    local exit_code="$2"
    local label="$3"
    local status="passed"

    if grep -q 'CONTENDED DEVICE' "$log" \
       || grep -q 'recordable=no' "$log"; then
        status="contended"
        CONTENDED_POINTS=$((CONTENDED_POINTS + 1))
    elif [[ $exit_code -ne 0 ]]; then
        if grep -Eq \
                'validation: FAIL.*(boundary|single|referee)=FAIL' "$log"; then
            status="acceptance-fail"
            ACCEPTANCE_FAILURES=$((ACCEPTANCE_FAILURES + 1))
        elif grep -Eq 'unconverged=[1-9][0-9]*' "$log"; then
            status="stopped"
            STOPPED_POINTS=$((STOPPED_POINTS + 1))
        elif grep -q 'validation: FAIL' "$log"; then
            status="acceptance-fail"
            ACCEPTANCE_FAILURES=$((ACCEPTANCE_FAILURES + 1))
        else
            {
                echo "RUN_STATUS status=execution-fail exit_code=$exit_code label=$label"
                echo "Unexpected execution failure; aborting the sweep."
            } | tee -a "$log"
            return "$exit_code"
        fi
    elif ! grep -Eq "distribution: .*\\(${REPEATS} runs\\)" "$log"; then
        status="incomplete"
        INCOMPLETE_POINTS=$((INCOMPLETE_POINTS + 1))
    fi

    echo "RUN_STATUS status=$status exit_code=$exit_code label=$label" \
        | tee -a "$log"
    return 0
}

run_point() {
    local arm="$1"
    local n="$2"
    local gpus="$3"
    local option="$4"
    local width="$5"
    local log="$OUT_DIR/${arm}_n${n}_${option}_${gpus}gpu_s${width}.txt"

    local single_state="$VALIDATION_DIR/single_n${n}_${option}.bin"
    local gate=(
        --single-gpu-state "$single_state"
        --single-state-rtol "$STATE_RTOL"
        --referee-dir "$REFEREE_ROOT/n${n}"
    )

    echo "### arm=$arm n=$n gpus=$gpus option=$option s=$width"
    set +e
    srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --mpi="$SRUN_MPI" \
        "$DISTRIBUTED" --require-mpi \
        --devices "$( [[ $gpus -eq 4 ]] && echo 0,1 || echo 0 )" \
        --n "$n" --m "$M" --s "$width" --steps "$STEPS" --tol "$TOL" \
        --repeats "$REPEATS" --option "$option" \
        --basis monomial --orth cholqr2 --arm "$arm" \
        "${gate[@]}" 2>&1 | tee "$log"
    local exit_code="${PIPESTATUS[0]}"
    set -e
    record_run_status \
        "$log" "$exit_code" \
        "arm=$arm,n=$n,gpus=$gpus,option=$option,s=$width"
    echo
}

# The one-GPU reference each rung's efficiency is measured against. Recorded per
# arm, because the arm changes the reference as well as the distributed point.
run_reference() {
    local arm="$1"
    local n="$2"
    local option="$3"
    local width="$4"
    local log="$OUT_DIR/${arm}_reference_n${n}_${option}_s${width}.txt"

    echo "### arm=$arm reference n=$n option=$option s=$width on one V100"
    set +e
    srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --mpi=none \
        "$SINGLE" \
        --n "$n" --m "$M" --s "$width" --steps "$STEPS" --tol "$TOL" \
        --repeats "$REPEATS" --option "$option" \
        --basis monomial --orth cholqr2 --arm "$arm" \
        --referee-dir "$REFEREE_ROOT/n${n}" \
        2>&1 | tee "$log"
    local exit_code="${PIPESTATUS[0]}"
    set -e
    record_run_status \
        "$log" "$exit_code" \
        "arm=$arm,reference=one-gpu,n=$n,option=$option,s=$width"
    echo
}

run_agreement_self_test

for arm in as-measured exact-depth; do
    for rung in "${RUNGS[@]}"; do
        n="${rung%%:*}"
        gpus="${rung##*:}"
        for option in basket rainbow; do
            for width in 1 4; do
                run_point "$arm" "$n" "$gpus" "$option" "$width"
                run_reference "$arm" "$n" "$option" "$width"
            done
        done
    done
done

# What the sweep was asked for and what it ran into, kept beside the transcripts
# so the classified point counts do not have to be recounted by hand.
{
    echo "arms=as-measured exact-depth"
    echo "rungs=${RUNGS[*]}"
    echo "m=$M"
    echo "steps=$STEPS"
    echo "tol=$TOL"
    echo "single_state_rtol=$STATE_RTOL"
    echo "repeats=$REPEATS"
    echo "stopped_points=$STOPPED_POINTS"
    echo "acceptance_failures=$ACCEPTANCE_FAILURES"
    echo "contended_points=$CONTENDED_POINTS"
    echo "incomplete_points=$INCOMPLETE_POINTS"
    echo "agreement_self_test=passed"
} > "$OUT_DIR/settings.txt"

echo "==============================================================================="
echo "Both arms attempted at every rung."
echo "  transcripts : $OUT_DIR"
echo "  settings    : $OUT_DIR/settings.txt"
echo "  stopped     : $STOPPED_POINTS predeclared numerical point(s)"
echo "  acceptance  : $ACCEPTANCE_FAILURES failed correction bound(s)"
echo "  contended   : $CONTENDED_POINTS timing-withheld diagnostic point(s)"
echo "  incomplete  : $INCOMPLETE_POINTS point(s) without the requested distribution"
echo "Draw the figures this sweep feeds:"
echo "  uv run scripts/plots/ca_correction_impact.py   (the two arms, side by side)"
echo "  uv run scripts/plots/ca_cycle_decomposition.py (where the cycle goes)"
echo "  uv run scripts/plots/ca_predicted_measured.py  (predicted against measured)"
echo "==============================================================================="

if [[ $ACCEPTANCE_FAILURES -ne 0 || $CONTENDED_POINTS -ne 0 \
      || $INCOMPLETE_POINTS -ne 0 ]]; then
    exit 1
fi
