#!/bin/bash
#SBATCH --job-name=caksm-gpu-calibration
#SBATCH --output=caksm-gpu-calibration-%j.out
#
# Calibrate every GPU arrangement exposed by a Slurm allocation.
# Build the project and load its runtime environment before submission. The job uses
# BUILD_DIR, which defaults to the repository's build directory.
set -euo pipefail

fail() {
    echo "Error: $*" >&2
    exit 1
}

require_idle_csv() {
    local csv="$1"
    [[ -s "$csv" ]] || fail "expected calibration output is missing: $csv"
    awk -F, '
        NR == 1 { for (i = 1; i <= NF; ++i) if ($i == "contended") column = i }
        NR > 1 && column && $column != 0 { bad = 1 }
        END { exit !column || bad }
    ' "$csv" || fail "calibration output reports contention: $csv"
}

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "submit this script with sbatch"

WORK_DIR="${ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[[ -f "$WORK_DIR/CMakeLists.txt" ]] || fail "repository root not found at $WORK_DIR"
git -C "$WORK_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || fail "$WORK_DIR is not a Git checkout"

MACHINE="${MACHINE:?export MACHINE when submitting the job}"
EXPECTED_GIT_REV="${EXPECTED_GIT_REV:-}"
REQUIRE_CLEAN_GIT="${REQUIRE_CLEAN_GIT:-0}"
EXPECTED_NODES="${EXPECTED_NODES:-${SLURM_JOB_NUM_NODES:-0}}"
EXPECTED_GPUS_PER_NODE="${EXPECTED_GPUS_PER_NODE:?export the expected GPU count per node}"
SINGLE_RUNS="${SINGLE_RUNS:-3}"
RUN_SINGLE="${RUN_SINGLE:-auto}"
RUN_TESTS="${RUN_TESTS:-1}"
PROBE_SRUN_EXTRA="${PROBE_SRUN_EXTRA:-}"
P2P_SRUN_EXTRA="${P2P_SRUN_EXTRA:-}"
for flag in "$REQUIRE_CLEAN_GIT" "$RUN_TESTS"; do
    [[ "$flag" == "0" || "$flag" == "1" ]] \
        || fail "REQUIRE_CLEAN_GIT and RUN_TESTS must be 0 or 1"
done

ACTUAL_GIT_REV="$(git -C "$WORK_DIR" rev-parse HEAD)"
if [[ -n "$EXPECTED_GIT_REV" ]]; then
    EXPECTED_GIT_REV="$(git -C "$WORK_DIR" rev-parse "$EXPECTED_GIT_REV^{commit}")"
    [[ "$ACTUAL_GIT_REV" == "$EXPECTED_GIT_REV" ]] \
        || fail "checkout is $ACTUAL_GIT_REV, expected $EXPECTED_GIT_REV"
fi
if [[ "$REQUIRE_CLEAN_GIT" -eq 1 ]] \
        && { ! git -C "$WORK_DIR" diff --quiet \
             || ! git -C "$WORK_DIR" diff --cached --quiet; }; then
    fail "tracked files changed after the submitted revision was selected"
fi

NODE_COUNT="${SLURM_JOB_NUM_NODES:-0}"
[[ "$NODE_COUNT" =~ ^[0-9]+$ && "$NODE_COUNT" -gt 0 ]] \
    || fail "SLURM_JOB_NUM_NODES is invalid"
[[ "$EXPECTED_NODES" =~ ^[0-9]+$ && "$EXPECTED_NODES" -gt 0 ]] \
    || fail "EXPECTED_NODES must be positive"
[[ "$NODE_COUNT" -eq "$EXPECTED_NODES" ]] \
    || fail "allocation has $NODE_COUNT nodes, expected $EXPECTED_NODES"
[[ "$EXPECTED_GPUS_PER_NODE" =~ ^[0-9]+$ && "$EXPECTED_GPUS_PER_NODE" -gt 1 ]] \
    || fail "EXPECTED_GPUS_PER_NODE must be at least 2"
[[ "$SINGLE_RUNS" =~ ^[0-9]+$ && "$SINGLE_RUNS" -gt 0 ]] \
    || fail "SINGLE_RUNS must be positive"
[[ "$RUN_SINGLE" == "auto" || "$RUN_SINGLE" == "0" || "$RUN_SINGLE" == "1" ]] \
    || fail "RUN_SINGLE must be auto, 0, or 1"

STAMP="$(date +%Y-%m-%dT%H%M%S)"
DEFAULT_RUN_ROOT="$WORK_DIR/data/calibration/$MACHINE"
DEFAULT_RUN_ROOT="$DEFAULT_RUN_ROOT/$STAMP-job$SLURM_JOB_ID-${NODE_COUNT}nodes"
RUN_ROOT="${RUN_ROOT:-$DEFAULT_RUN_ROOT}"
mkdir -p "$(dirname "$RUN_ROOT")"
mkdir "$RUN_ROOT" || fail "result directory already exists: $RUN_ROOT"
BUILD_DIR="${BUILD_DIR:-$WORK_DIR/build}"
ARCHIVE="$RUN_ROOT.tar.gz"

finish() {
    local status=$?
    set +e
    printf 'exit_status=%d\nfinished=%s\n' "$status" "$(date -Is)" > "$RUN_ROOT/job_status.txt"
    if command -v sha256sum >/dev/null 2>&1; then
        (cd "$RUN_ROOT" \
            && find . -type f ! -name SHA256SUMS ! -name job.log -print0 \
            | sort -z | xargs -0 -r sha256sum > SHA256SUMS)
    fi
    if ! tar -C "$(dirname "$RUN_ROOT")" -czf "$ARCHIVE" "$(basename "$RUN_ROOT")"; then
        status=1
    fi
    echo
    echo "Result directory: $RUN_ROOT"
    echo "Result archive  : $ARCHIVE"
    trap - EXIT
    exit "$status"
}
trap finish EXIT

exec > >(tee "$RUN_ROOT/job.log") 2>&1

echo "==============================================================================="
echo " CAKSM GPU calibration job"
echo "==============================================================================="
echo "started             : $(date -Is)"
echo "machine             : $MACHINE"
echo "job                 : $SLURM_JOB_ID"
echo "nodes               : $NODE_COUNT"
echo "GPUs per node       : $EXPECTED_GPUS_PER_NODE"
echo "Git revision        : $ACTUAL_GIT_REV"
echo "repository          : $WORK_DIR"
echo "build directory     : $BUILD_DIR"
echo "result directory    : $RUN_ROOT"
echo

for command_name in nvidia-smi srun scontrol git; do
    command -v "$command_name" >/dev/null 2>&1 \
        || fail "$command_name is not available in the submitted environment"
done

CALIBRATION_BINARIES=(
    calibrate-gpu-reduction
    gpu-stream
    gpu-fma-loop
    gpu-device-probe
    calibrate-gpu-p2p
)
for binary in "${CALIBRATION_BINARIES[@]}"; do
    [[ -x "$BUILD_DIR/$binary" ]] \
        || fail "$binary is missing from BUILD_DIR=$BUILD_DIR; build before submitting"
done

if command -v ldd >/dev/null 2>&1; then
    RUNTIME_LIBRARIES="$RUN_ROOT/runtime_libraries.txt"
    for binary in "${CALIBRATION_BINARIES[@]}"; do
        echo "### $binary" >> "$RUNTIME_LIBRARIES"
        ldd "$BUILD_DIR/$binary" >> "$RUNTIME_LIBRARIES" 2>&1 || true
    done
    if grep -q 'not found' "$RUNTIME_LIBRARIES"; then
        fail "a calibration binary has a missing runtime library; see $RUNTIME_LIBRARIES"
    fi
fi

export EXPECTED_GPUS_PER_NODE
HOST_FILE="$RUN_ROOT/hosts.txt"
scontrol show hostnames "${SLURM_JOB_NODELIST:-}" > "$HOST_FILE"
[[ "$(wc -l < "$HOST_FILE")" -eq "$NODE_COUNT" ]] \
    || fail "Slurm did not resolve the expected number of hosts"
GPU_COUNT_FILE="$RUN_ROOT/gpu_counts.txt"
# PROBE_SRUN_EXTRA is intentionally split into launcher arguments.
# shellcheck disable=SC2086
srun --nodes="$NODE_COUNT" --ntasks="$NODE_COUNT" --ntasks-per-node=1 \
    ${PROBE_SRUN_EXTRA} --export=ALL \
    bash -c '
        count=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)
        capabilities=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u)
        capability_count=$(wc -l <<< "$capabilities")
        printf "%s %s %s\n" "$(hostname)" "$count" "$capabilities"
        test "$count" -eq "$EXPECTED_GPUS_PER_NODE"
        test "$capability_count" -eq 1
    ' > "$GPU_COUNT_FILE"
cat "$GPU_COUNT_FILE"
[[ "$(awk 'NF == 3 && $2 ~ /^[0-9]+$/ {count++} END {print count + 0}' \
        "$GPU_COUNT_FILE")" -eq "$NODE_COUNT" ]] \
    || fail "not every allocated node reported its GPU count"

COMPUTE_CAPABILITY="$(awk 'NF == 3 {print $3}' "$GPU_COUNT_FILE" | sort -u)"
[[ "$(wc -l <<< "$COMPUTE_CAPABILITY")" -eq 1 ]] \
    || fail "the allocation contains more than one CUDA compute capability"

{
    echo "started=$(date -Is)"
    echo "machine=$MACHINE"
    echo "git_revision=$ACTUAL_GIT_REV"
    echo "slurm_job=$SLURM_JOB_ID"
    echo "slurm_nodelist=${SLURM_JOB_NODELIST:-unknown}"
    echo "nodes=$NODE_COUNT"
    echo "gpus_per_node=$EXPECTED_GPUS_PER_NODE"
    echo "compute_capability=$COMPUTE_CAPABILITY"
    echo "build_dir=$BUILD_DIR"
    echo "probe_srun_extra=${PROBE_SRUN_EXTRA:-none}"
    echo "p2p_srun_extra=${P2P_SRUN_EXTRA:-none}"
    echo "nccl_home=${NCCL_HOME:-${NCCL_ROOT:-unknown}}"
} > "$RUN_ROOT/provenance.txt"

if command -v module >/dev/null 2>&1; then
    module list 2> "$RUN_ROOT/modules.txt" || true
fi
{
    for command_name in cmake nvcc mpicc mpirun; do
        if command -v "$command_name" >/dev/null 2>&1; then
            echo "$command_name=$(command -v "$command_name")"
        else
            echo "$command_name=not found"
        fi
    done
} > "$RUN_ROOT/toolchain.txt"
nvidia-smi > "$RUN_ROOT/nvidia_smi.txt"

if [[ "$RUN_TESTS" -eq 1 ]]; then
    command -v ctest >/dev/null 2>&1 \
        || fail "ctest is unavailable; load the build environment or set RUN_TESTS=0"
    echo "### Existing build tests"
    ctest --test-dir "$BUILD_DIR" -R '^regime\.' --output-on-failure
    echo
fi

echo "### Probe every allocated node"
PROBE_DIR="$RUN_ROOT/probes"
mkdir "$PROBE_DIR"
# PROBE_SRUN_EXTRA is intentionally split into launcher arguments.
# shellcheck disable=SC2086
srun --nodes="$NODE_COUNT" --ntasks="$NODE_COUNT" --ntasks-per-node=1 \
    ${PROBE_SRUN_EXTRA} \
    --export=ALL,ROOT="$WORK_DIR",DATA_DIR="$PROBE_DIR",BUILD_DIR="$BUILD_DIR" \
    bash "$WORK_DIR/scripts/regime/gpu_probe.sh"
echo

if [[ "$RUN_SINGLE" == "auto" ]]; then
    [[ "$NODE_COUNT" -eq 1 ]] && RUN_SINGLE=1 || RUN_SINGLE=0
fi
if [[ "$RUN_SINGLE" -eq 1 ]]; then
    echo "### Single GPU calibration"
    for ((run = 1; run <= SINGLE_RUNS; ++run)); do
        SINGLE_DIR="$RUN_ROOT/single_run_$run"
        mkdir "$SINGLE_DIR"
        echo "  run $run of $SINGLE_RUNS"
        CUDA_VISIBLE_DEVICES=0 MACHINE="$MACHINE" BUILD_DIR="$BUILD_DIR" \
            DATA_DIR="$SINGLE_DIR" \
            bash "$WORK_DIR/scripts/regime/calibrate_gpu.sh" \
            > "$SINGLE_DIR/calibrate_gpu.log" 2>&1
        require_idle_csv "$SINGLE_DIR/gpu_fma_loop.csv"
        require_idle_csv "$SINGLE_DIR/calibrate_gpu_reduction.csv"
        require_idle_csv "$SINGLE_DIR/gpu_stream.csv"
    done
    echo
fi

echo "### Participant sweep"
P2P_DIR="$RUN_ROOT/p2p"
MACHINE="$MACHINE" P2P="$BUILD_DIR/calibrate-gpu-p2p" \
    PROBE="$BUILD_DIR/gpu-device-probe" RUN_DIR="$P2P_DIR" \
    SRUN_EXTRA="$P2P_SRUN_EXTRA" REQUIRE_COMPLETE_SWEEP=1 REQUIRE_PROBE=1 \
    REQUIRE_NCCL_TRANSPORT="${REQUIRE_NCCL_TRANSPORT:-}" REQUIRE_IDLE=1 \
    bash "$WORK_DIR/scripts/regime/calibrate_gpu_p2p.sh"
bash "$WORK_DIR/scripts/regime/topology_entry.sh" "$P2P_DIR" \
    > "$RUN_ROOT/topology_entries.txt"

echo
echo "Calibration completed successfully."
