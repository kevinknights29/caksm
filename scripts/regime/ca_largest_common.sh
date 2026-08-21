#!/usr/bin/env bash
# Predict the largest odd grid common to every production CA-integrator arm.
#
# Nothing is allocated and nothing is timed. Each executable reports its own
# allocation model against cudaMemGetInfo, and this takes the minimum
# recommendation across every topology, option and width, so the binding arm
# sets the grid before any correctness or performance run is attempted.
#
# Run inside an exclusive two-node Synge allocation:
#
#   salloc -N 2 -n 20 -p compute -t 01:00:00 --nodelist=synge-n01,synge-n02
#   SLURM_OVERLAP=1 ./scripts/regime/ca_largest_common.sh
#
# SLURM_OVERLAP shares an allocation the calling shell already holds, which the
# steps below otherwise refuse with "Requested nodes are busy". Sharing is fine
# here, since every run reports an allocation model and nothing is timed, so there is
# no reason to prefer sbatch.

set -euo pipefail

ROOT="${ROOT:-$PWD}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
SINGLE="${SINGLE:-$BUILD_DIR/ca-integrator}"
DISTRIBUTED="${DISTRIBUTED:-$BUILD_DIR/ca-integrator-2gpu}"
STATE_DIR="${STATE_DIR:-$ROOT/data/ca-integrator-largest-common}"
M="${M:-39}"
RESERVE="${RESERVE:-0.10}"
CANDIDATE_N="${CANDIDATE_N:-97}"
SRUN_MPI="${SRUN_MPI:-pmix}"

if [[ ! -x "$SINGLE" ]]; then
    echo "Error: missing executable: $SINGLE"
    exit 1
fi
if [[ ! -x "$DISTRIBUTED" ]]; then
    echo "Error: missing executable: $DISTRIBUTED"
    exit 1
fi
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

mkdir -p "$STATE_DIR"
REPORTS="$STATE_DIR/memory_reports.csv"
printf '%s\n' \
    "topology,option,s,recommended_n,report" > "$REPORTS"

export PMIX_MCA_gds="${PMIX_MCA_gds:-hash}"
export NCCL_DEBUG=WARN

run_report() {
    local topology="$1"
    local option="$2"
    local width="$3"
    shift 3
    local report="$STATE_DIR/${topology}_${option}_s${width}.txt"

    # The header goes to the terminal, the command's own output to the report.
    # Note the absence of a line continuation: joining these two would make the
    # header echo swallow the command as arguments and never run it.
    echo "### topology=$topology option=$option s=$width"
    "$@" 2>&1 | tee "$report"

    local recommended
    recommended="$(
        sed -n \
            's/.*MEMORY_REPORT .*recommended_n=\([0-9][0-9]*\).*/\1/p' \
            "$report" | tail -n 1
    )"
    if [[ -z "$recommended" ]]; then
        echo "Error: no MEMORY_REPORT record in $report"
        exit 1
    fi
    printf '%s,%s,%s,%s,%s\n' \
        "$topology" "$option" "$width" "$recommended" "$report" \
        >> "$REPORTS"
}

for option in basket rainbow; do
    for width in 1 4; do
        common=(
            --n "$CANDIDATE_N" --m "$M" --s "$width"
            --option "$option" --basis monomial --orth cholqr2
            --memory-report --memory-reserve "$RESERVE"
        )

        run_report one_gpu "$option" "$width" \
            srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --mpi=none \
            "$SINGLE" "${common[@]}"

        run_report two_gpu_one_node "$option" "$width" \
            srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --mpi=none \
            "$DISTRIBUTED" --devices 0,1 "${common[@]}"

        run_report two_gpu_two_nodes "$option" "$width" \
            srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
            --mpi="$SRUN_MPI" \
            "$DISTRIBUTED" --require-mpi --devices 0 "${common[@]}"

        run_report four_gpu_two_nodes "$option" "$width" \
            srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
            --mpi="$SRUN_MPI" \
            "$DISTRIBUTED" --require-mpi --devices 0,1 "${common[@]}"
    done
done

SELECTED_N="$(
    awk -F, 'NR > 1 {print $4}' "$REPORTS" \
        | sort -n | head -n 1
)"
LIMITING_ARMS="$(
    awk -F, -v selected="$SELECTED_N" \
        'NR > 1 && $4 == selected {
            if (arms != "") arms = arms ";"
            arms = arms $1 "/" $2 "/s" $3
        }
        END {print arms}' \
        "$REPORTS"
)"

printf '%s\n' "$SELECTED_N" > "$STATE_DIR/selected_n.txt"

# The run settings that change the answer, kept beside it so a reader can see
# which reserve and which m produced this grid.
{
    echo "m=$M"
    echo "reserve=$RESERVE"
    echo "probe_candidate_n=$CANDIDATE_N"
    echo "selected_n=$SELECTED_N"
    echo "limiting_arms=$LIMITING_ARMS"
} > "$STATE_DIR/settings.txt"

echo
echo "Largest-common-grid memory phase complete"
echo "  selected odd n : $SELECTED_N"
echo "  limiting arms  : $LIMITING_ARMS"
echo "  reports        : $REPORTS"
echo "  settings       : $STATE_DIR/settings.txt"
echo
echo "No solver timing or correctness allocation was performed."
echo "Draw figure P11 with:"
echo "  uv run --script scripts/plots/ca_memory_model.py"
