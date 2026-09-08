#!/usr/bin/env bash
# Attribute the distributed cycle to its phases with Nsight Systems.
#
# The solver carries an NVTX range around every phase and a profile window set
# with --profile-start and --profile-steps, so a trace answers what the printed
# account can only bound: which phase the cycle is spent in, how much of a host
# round-trip is the collective against the pipeline drain, and whether a stream
# is gap-dominated.
#
# Nothing here is a recordable timing. Nsight costs about 1.7 ms/step at the
# n=97 four-GPU point, so a cycle time read out of a trace is larger than the
# real one. These are attribution evidence, like the ncu captures in the
# vertical sweep; every log carries that header and the directory is excluded
# from the figure inputs.
#
# Two traps this wraps, each of which otherwise costs an allocation to
# rediscover. OpenMPI's PMIx shmem2 component segfaults inside PMIx_Init on this
# host before any application code runs, so every harness exports
# PMIX_MCA_gds=hash and a hand-typed srun does not. And --steps sets
# h = expiry/steps, so shortening a run changes the problem: a short run at n=97
# is ten times stiffer, pins m at m_max and diverges. The window is bounded with
# --profile-start instead, and --steps stays at its production value.
#
# Run inside an exclusive two-node Synge allocation:
#
#   salloc -N 2 -n 20 -p compute -t 01:00:00 --nodelist=synge-n01,synge-n02
#   SLURM_OVERLAP=1 ./scripts/regime/ca_nsight_profile.sh
#
# A one-node allocation can capture the two-GPU peer-copy path:
#
#   SINGLE_NODE=1 N=61 WIDTHS=3 ARMS=exact-depth CERTIFICATE=deferred \
#   OVERLAPS="off on" \
#   ./scripts/regime/ca_nsight_profile.sh

set -uo pipefail

ROOT="${ROOT:-$PWD}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
DISTRIBUTED="${DISTRIBUTED:-$BUILD_DIR/ca-integrator-multigpu}"
OUT_DIR="${OUT_DIR:-$ROOT/data/ca-integrator-nsight}"
NSYS="${NSYS:-/usr/local/cuda-12.8/nsight-systems-2024.6.2/bin/nsys}"
N="${N:-97}"
M="${M:-39}"
STEPS="${STEPS:-100}"
TOL="${TOL:-1e-8}"
OPTION="${OPTION:-rainbow}"
WIDTHS="${WIDTHS:-1 4}"
ARMS="${ARMS:-as-measured exact-depth}"
CERTIFICATE="${CERTIFICATE:-deferred}"
AGREEMENT="${AGREEMENT:-step}"
OVERLAPS="${OVERLAPS:-off}"
PROFILE_START="${PROFILE_START:-50}"
PROFILE_STEPS="${PROFILE_STEPS:-3}"
SRUN_MPI="${SRUN_MPI:-pmix}"
# One node with two local GPUs needs no MPI at all, which removes the PMIx path
# from the experiment entirely. Useful when the launcher is the thing failing.
SINGLE_NODE="${SINGLE_NODE:-0}"

if [[ ! -x "$DISTRIBUTED" ]]; then
    echo "Error: missing executable: $DISTRIBUTED" >&2
    exit 1
fi
if [[ ! -x "$NSYS" ]]; then
    echo "Error: nsys not found at $NSYS; set NSYS=<path>." >&2
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

if [[ "$SINGLE_NODE" -eq 0 && "$(allocated_nodes)" -lt 2 ]]; then
    echo "Error: run inside a two-node allocation, or set SINGLE_NODE=1." >&2
    report_allocation
    exit 1
fi

mkdir -p "$OUT_DIR"
export PMIX_MCA_gds="${PMIX_MCA_gds:-hash}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

not_a_timing_header() {
    cat <<'HEADER'
# ATTRIBUTION EVIDENCE ONLY, NOT A RECORDABLE TIMING.
# Nsight Systems instruments every CUDA API call and NVTX range, which inflates
# the cycle time. Read phase shares and call counts from this trace; read cycle
# times from an unprofiled run.
HEADER
}

capture() {
    local arm="$1" width="$2" overlap="$3"
    local stem="$OUT_DIR/${arm}_n${N}_${OPTION}_s${width}_overlap_${overlap}"
    local log="${stem}.txt"

    local solver=(
        "$DISTRIBUTED" --devices 0,1
        --n "$N" --m "$M" --s "$width" --steps "$STEPS" --tol "$TOL"
        --repeats 1 --option "$OPTION"
        --basis monomial --orth cholqr2 --arm "$arm"
        --certificate "$CERTIFICATE"
        --agreement-check "$AGREEMENT"
        --halo-overlap "$overlap"
        --profile-start "$PROFILE_START" --profile-steps "$PROFILE_STEPS"
    )
    local profiler=(
        "$NSYS" profile --trace=cuda,nvtx --sample=none
        -o "${stem}_%q{SLURM_PROCID}" --force-overwrite true
    )

    printf '### nsight arm=%s n=%s option=%s s=%s overlap=%s ' \
        "$arm" "$N" "$OPTION" "$width" "$overlap"
    printf 'certificate=%s agreement=%s\n' "$CERTIFICATE" "$AGREEMENT"
    {
        not_a_timing_header
        echo "arm=$arm"
        echo "n=$N option=$OPTION s=$width steps=$STEPS m=$M tol=$TOL"
        echo "agreement=$AGREEMENT"
        echo "certificate=$CERTIFICATE"
        echo "halo_overlap=$overlap"
        echo "profile_window=steps ${PROFILE_START}..$((PROFILE_START + PROFILE_STEPS - 1))"
        echo
    } > "$log"

    if [[ "$SINGLE_NODE" -eq 1 ]]; then
        srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --mpi=none \
            "${profiler[@]}" "${solver[@]}" 2>&1 | tee -a "$log"
    else
        srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --mpi="$SRUN_MPI" \
            --export=ALL,PMIX_MCA_gds=hash \
            "${profiler[@]}" "${solver[@]}" --require-mpi 2>&1 | tee -a "$log"
    fi
    echo
}

summarize() {
    local report="$1"
    local stem="${report%.nsys-rep}"
    echo "### stats $(basename "$report")"
    {
        not_a_timing_header
        "$NSYS" stats \
            --report nvtx_sum --report cuda_api_sum \
            --report cuda_gpu_kern_sum \
            --force-export=true "$report" 2>&1
    } > "${stem}_stats.txt"
    # The three lines that decide where the cycle goes.
    grep -E "cross-rank|root residual|deep-halo|NCCL halo|cudaMemcpyAsync|cudaStreamSynchronize" \
        "${stem}_stats.txt" | head -12 | sed 's/^/    /'
    echo
}

for arm in $ARMS; do
    for width in $WIDTHS; do
        for overlap in $OVERLAPS; do
            capture "$arm" "$width" "$overlap"
        done
    done
done

while IFS= read -r report; do
    summarize "$report"
done < <(find "$OUT_DIR" -maxdepth 1 -name '*.nsys-rep' -print | sort)

# The window and the arms the trace covers, with the not-a-timing header so the
# file cannot be mistaken for a measurement on its own.
{
    not_a_timing_header
    echo "role=attribution_only"
    echo "arms=$ARMS"
    echo "widths=$WIDTHS"
    echo "n=$N"
    echo "m=$M"
    echo "option=$OPTION"
    echo "steps=$STEPS"
    echo "tol=$TOL"
    echo "agreement=$AGREEMENT"
    echo "certificate=$CERTIFICATE"
    echo "overlaps=$OVERLAPS"
    echo "profile_start=$PROFILE_START"
    echo "profile_steps=$PROFILE_STEPS"
    echo "single_node=$SINGLE_NODE"
    echo "nsys_version=$("$NSYS" --version 2>/dev/null | head -n 1)"
} > "$OUT_DIR/settings.txt"

echo "==============================================================================="
echo "Nsight attribution captured."
echo "  traces     : $OUT_DIR/*.nsys-rep"
echo "  summaries  : $OUT_DIR/*_stats.txt"
echo "  settings   : $OUT_DIR/settings.txt"
echo "These are attribution evidence, not recordable timings."
echo "==============================================================================="
