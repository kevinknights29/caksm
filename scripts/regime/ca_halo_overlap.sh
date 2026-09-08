#!/usr/bin/env bash
# Compare serialized and overlapped halo exchange at the production points.
#
# Run the complete experiment inside an exclusive two-node Synge allocation:
#
#   sbatch --nodes=2 --ntasks=20 --partition=compute --time=01:00:00 \
#          --nodelist=synge-n01,synge-n02 --exclusive \
#          scripts/regime/ca_halo_overlap.sh
#
# A one-node backfill can run only the peer-copy points:
#
#   POINTS="61:3:1:2:basket 61:3:1:2:rainbow" \
#   sbatch --nodes=1 --ntasks=10 --partition=compute --time=01:00:00 \
#          --nodelist=synge-n02 --exclusive \
#          --export=ALL,POINTS scripts/regime/ca_halo_overlap.sh

set -euo pipefail

ROOT="${ROOT:-$PWD}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
SINGLE="${SINGLE:-$BUILD_DIR/ca-integrator}"
DISTRIBUTED="${DISTRIBUTED:-$BUILD_DIR/ca-integrator-multigpu}"
OUT_DIR="${OUT_DIR:-$ROOT/data/ca-halo-overlap}"
M="${M:-39}"
STEPS="${STEPS:-100}"
TOL="${TOL:-1e-8}"
STATE_RTOL="${STATE_RTOL:-1e-10}"
PAIR_RTOL="${PAIR_RTOL:-1e-12}"
REPEATS="${REPEATS:-7}"
SRUN_MPI="${SRUN_MPI:-pmix}"
POINTS="${POINTS:-61:3:1:2:basket 61:3:1:2:rainbow 97:4:2:4:rainbow}"

for binary in "$SINGLE" "$DISTRIBUTED"; do
    if [[ ! -x "$binary" ]]; then
        echo "Error: missing executable: $binary" >&2
        exit 1
    fi
done
if [[ "$REPEATS" -lt 7 ]]; then
    echo "Error: overlap measurements require at least seven repeats." >&2
    exit 1
fi

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

required_nodes=1
for point in $POINTS; do
    IFS=: read -r n width nodes gpus option <<< "$point"
    if [[ ! "$n" =~ ^[0-9]+$ || ! "$width" =~ ^[0-9]+$ \
          || ! "$nodes" =~ ^[0-9]+$ || ! "$gpus" =~ ^[0-9]+$ \
          || ("$option" != "basket" && "$option" != "rainbow") ]]; then
        echo "Error: invalid point: $point" >&2
        exit 1
    fi
    if [[ "$gpus" -ne $((2 * nodes)) ]]; then
        echo "Error: point $point must request two GPUs per node." >&2
        exit 1
    fi
    if [[ "$nodes" -gt "$required_nodes" ]]; then
        required_nodes="$nodes"
    fi
done

if [[ "$(allocated_nodes)" -lt "$required_nodes" ]]; then
    echo "Error: POINTS requires an allocation with $required_nodes node(s)." >&2
    exit 1
fi

mkdir -p "$OUT_DIR/states"
export PMIX_MCA_gds="${PMIX_MCA_gds:-hash}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

run_reference()
{
    local n="$1" width="$2" option="$3" state="$4" log="$5"
    echo "### reference n=$n s=$width option=$option"
    srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --mpi=none \
        "$SINGLE" --device 0 \
        --n "$n" --m "$M" --s "$width" --steps "$STEPS" --tol "$TOL" \
        --repeats 1 --option "$option" --basis monomial --orth cholqr2 \
        --arm exact-depth --save-state "$state" 2>&1 | tee "$log"
}

run_distributed()
{
    local n="$1" width="$2" nodes="$3" gpus="$4" option="$5"
    local overlap="$6" comparison="$7" state="$8" log="$9"
    local devices="0,1"
    local comparison_rtol="$STATE_RTOL"
    if [[ "$overlap" == "on" ]]; then
        comparison_rtol="$PAIR_RTOL"
    fi
    local common=(
        "$DISTRIBUTED" --devices "$devices"
        --n "$n" --m "$M" --s "$width" --steps "$STEPS" --tol "$TOL"
        --repeats "$REPEATS" --option "$option"
        --basis monomial --orth cholqr2 --arm exact-depth
        --certificate deferred --agreement-check step
        --halo-overlap "$overlap"
        --single-gpu-state "$comparison"
        --single-state-rtol "$comparison_rtol"
        --save-state "$state"
    )

    echo "### n=$n s=$width gpus=$gpus option=$option halo-overlap=$overlap"
    if [[ "$nodes" -eq 1 ]]; then
        srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --mpi=none \
            "${common[@]}" 2>&1 | tee "$log"
    else
        srun --nodes="$nodes" --ntasks="$nodes" --ntasks-per-node=1 \
            --mpi="$SRUN_MPI" --export=ALL,PMIX_MCA_gds=hash \
            "${common[@]}" --require-mpi 2>&1 | tee "$log"
    fi
}

for point in $POINTS; do
    IFS=: read -r n width nodes gpus option <<< "$point"
    stem="n${n}_s${width}_${option}_${gpus}gpu"
    reference_state="$OUT_DIR/states/${stem}_reference.bin"
    off_state="$OUT_DIR/states/${stem}_overlap_off.bin"
    on_state="$OUT_DIR/states/${stem}_overlap_on.bin"

    run_reference \
        "$n" "$width" "$option" "$reference_state" \
        "$OUT_DIR/${stem}_reference.txt"
    run_distributed \
        "$n" "$width" "$nodes" "$gpus" "$option" off \
        "$reference_state" "$off_state" \
        "$OUT_DIR/${stem}_overlap_off.txt"
    run_distributed \
        "$n" "$width" "$nodes" "$gpus" "$option" on \
        "$off_state" "$on_state" \
        "$OUT_DIR/${stem}_overlap_on.txt"

    if cmp -s "$off_state" "$on_state"; then
        state_match="bitwise"
    else
        state_match="within_relative_tolerance"
    fi
    echo "OVERLAP_PAIR point=$point state_match=$state_match" \
        | tee -a "$OUT_DIR/${stem}_overlap_on.txt"
done

{
    echo "points=$POINTS"
    echo "m=$M"
    echo "steps=$STEPS"
    echo "tol=$TOL"
    echo "state_rtol=$STATE_RTOL"
    echo "pair_rtol=$PAIR_RTOL"
    echo "repeats=$REPEATS"
} > "$OUT_DIR/settings.txt"

echo "==============================================================================="
echo "Serialized and overlapped halo arms completed."
echo "  transcripts : $OUT_DIR"
echo "  states      : $OUT_DIR/states"
echo "  settings    : $OUT_DIR/settings.txt"
echo "==============================================================================="
