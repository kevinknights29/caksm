#!/usr/bin/env bash
# Re-run strong scaling at fixed global problem size, and keep the transcripts.
#
# The headline strong-scaling numbers were read off terminal output that was
# never downloaded, so no artifact backs them. This produces the same ladder as
# a recorded artifact under both arms, which either reproduces that result or
# shows it cannot be.
#
# The rungs are one V100, two on one node, two on two nodes, and four on two
# nodes, all at fixed n. The two-on-one-node rung earns its place by separating
# transport from participant count: same participants as two nodes, different
# interconnect.
#
# Run under the batch scheduler on two exclusive Synge nodes. The cycle times
# are the result here, so this is the recommended route:
#
#   sbatch --nodes=2 --ntasks=20 --partition=compute --time=01:00:00 \
#          --nodelist=synge-n01,synge-n02 --exclusive \
#          scripts/regime/ca_strong_scaling.sh
#
# Interactively, for a smoke test only. SLURM_OVERLAP shares the calling shell's
# CPUs with every step below, so its timings can vary:
#
#   salloc -N 2 -n 20 -p compute -t 01:00:00 --nodelist=synge-n01,synge-n02
#   SLURM_OVERLAP=1 ./scripts/regime/ca_strong_scaling.sh

set -euo pipefail

ROOT="${ROOT:-$PWD}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
SINGLE="${SINGLE:-$BUILD_DIR/ca-integrator}"
DISTRIBUTED="${DISTRIBUTED:-$BUILD_DIR/ca-integrator-2gpu}"
OUT_DIR="${OUT_DIR:-$ROOT/data/ca-integrator-strong}"
REFEREE_ROOT="${REFEREE_ROOT:-$ROOT/data}"
N="${N:-61}"
M="${M:-39}"
STEPS="${STEPS:-100}"
TOL="${TOL:-1e-8}"
REPEATS="${REPEATS:-7}"
ARMS="${ARMS:-as-measured exact-depth}"
SRUN_MPI="${SRUN_MPI:-pmix}"
STATE_RTOL="${STATE_RTOL:-1e-10}"

for binary in "$SINGLE" "$DISTRIBUTED"; do
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

# The single-GPU state each distributed arm is validated against. Generated once
# per (arm, option, width) before the distributed rungs, and never overwritten.
STATE_DIR="$OUT_DIR/states"
mkdir -p "$STATE_DIR"

gate_args() {
    local option="$1"
    local args=()
    if [[ -f "$REFEREE_ROOT/n${N}/referee_n${N}_${option}.bin" ]]; then
        args+=(--referee-dir "$REFEREE_ROOT/n${N}")
    fi
    if [[ ${#args[@]} -ne 0 ]]; then
        printf '%s\n' "${args[@]}"
    fi
}

run_rung() {
    local arm="$1" topology="$2" option="$3" width="$4"
    shift 4
    local log="$OUT_DIR/${arm}_${topology}_${option}_s${width}.txt"

    echo "### arm=$arm topology=$topology option=$option s=$width n=$N"
    set +e
    "$@" 2>&1 | tee "$log"
    local exit_code="${PIPESTATUS[0]}"
    set -e
    if [[ $exit_code -ne 0 ]]; then
        local status="execution-fail"
        if grep -Eq \
                'validation: FAIL.*(boundary|single|referee)=FAIL' "$log"; then
            status="acceptance-fail"
        elif grep -Eq 'unconverged=[1-9][0-9]*' "$log"; then
            status="stopped"
        elif grep -q 'validation: FAIL' "$log"; then
            status="acceptance-fail"
        fi
        echo "RUN_STATUS status=$status exit_code=$exit_code" | tee -a "$log"
        echo "Error: strong-scaling rung failed; evidence: $log" >&2
        return "$exit_code"
    fi
    if grep -q 'CONTENDED DEVICE' "$log" \
       || grep -q 'recordable=no' "$log"; then
        echo "RUN_STATUS status=contended exit_code=0" | tee -a "$log"
        echo "Error: contended strong-scaling rung; evidence: $log" >&2
        return 1
    fi
    if ! grep -Eq "distribution: .*\\(${REPEATS} runs\\)" "$log"; then
        echo "RUN_STATUS status=incomplete exit_code=0" | tee -a "$log"
        echo "Error: missing ${REPEATS}-repeat timing distribution; evidence: $log" >&2
        return 1
    fi
    echo "RUN_STATUS status=passed exit_code=0" | tee -a "$log"
    echo
}

for arm in $ARMS; do
    for option in basket rainbow; do
        for width in 1 4; do
            mapfile -t gate < <(gate_args "$option")
            state="$STATE_DIR/${arm}_single_n${N}_${option}_s${width}.bin"

            common=(
                --n "$N" --m "$M" --s "$width" --steps "$STEPS"
                --tol "$TOL" --repeats "$REPEATS" --option "$option"
                --basis monomial --orth cholqr2 --arm "$arm"
            )

            run_rung "$arm" one_gpu "$option" "$width" \
                srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --mpi=none \
                "$SINGLE" "${common[@]}" --save-state "$state" \
                "${gate[@]+"${gate[@]}"}"

            run_rung "$arm" two_gpu_one_node "$option" "$width" \
                srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --mpi=none \
                "$DISTRIBUTED" --devices 0,1 "${common[@]}" \
                --single-gpu-state "$state" \
                --single-state-rtol "$STATE_RTOL" \
                "${gate[@]+"${gate[@]}"}"

            run_rung "$arm" two_gpu_two_nodes "$option" "$width" \
                srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
                --mpi="$SRUN_MPI" \
                "$DISTRIBUTED" --require-mpi --devices 0 "${common[@]}" \
                --single-gpu-state "$state" \
                --single-state-rtol "$STATE_RTOL" \
                "${gate[@]+"${gate[@]}"}"

            run_rung "$arm" four_gpu_two_nodes "$option" "$width" \
                srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
                --mpi="$SRUN_MPI" \
                "$DISTRIBUTED" --require-mpi --devices 0,1 "${common[@]}" \
                --single-gpu-state "$state" \
                --single-state-rtol "$STATE_RTOL" \
                "${gate[@]+"${gate[@]}"}"
        done
    done
done

# The problem the ladder was run at, kept beside the transcripts so two sweeps
# can be told apart without reading a transcript header.
{
    echo "arms=$ARMS"
    echo "n=$N"
    echo "m=$M"
    echo "steps=$STEPS"
    echo "tol=$TOL"
    echo "single_state_rtol=$STATE_RTOL"
    echo "repeats=$REPEATS"
} > "$OUT_DIR/settings.txt"

echo "==============================================================================="
echo "Strong-scaling ladder recorded."
echo "  transcripts : $OUT_DIR"
echo "  states      : $STATE_DIR"
echo "  settings    : $OUT_DIR/settings.txt"
echo "Draw the figures this ladder feeds:"
echo "  uv run --script scripts/plots/ca_strong_scaling.py   P1: the ladder itself"
echo "  uv run --script scripts/plots/ca_weak_scaling.py     P2: its one-GPU efficiency denominator"
echo "==============================================================================="
