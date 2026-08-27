#!/usr/bin/env bash
# Collect the canonical CSV behind the poster's GPU regime-trajectory figure.
#
# Places the real Basket operator once per participant/node topology, over a common global
# grid sequence and again at a held-fixed local problem size, and records beside the result
# every input it drew on and the checksum of each.
#
# Two stages, deliberately separable:
#
#   measure     (MEASURE=1, runs on synge under Slurm)  Launches the four arrangements
#               at each grid, one solver run each, and records the converged Krylov
#               dimension and the validation verdict that run reported. This is the
#               stage that turns a placed point into a measured one, and the only
#               thing that lets a row say validation=PASS.
#   placement   (default, runs anywhere)  Host-only. regime-gpu-trajectory reads the accepted
#               placement table for the operator and its measured Krylov dimension, the
#               calibrated preset for the device capacities, and include/gpu_topology.hpp for
#               each arrangement's measured collective latency, then applies the predeclared
#               byte model. No CUDA, no timer, no scheduler.
#   verify      (VERIFY_TOPOLOGY=1, runs on synge under Slurm)  Checks that the allocation can
#               actually supply every arrangement the table claims, and captures the fabric
#               topology of each node it spans. A declared two-node arm that the scheduler
#               packed onto one node reduces at the wrong rung, so a mismatch fails the run
#               rather than annotating it.
#
# The separation is the point. The collective latencies were measured once, on an idle
# allocation, by calibrate_gpu.sh and calibrate_ca_participants.sh; re-measuring them here
# would put a fresh, possibly contended number under a figure whose whole claim is that its
# coordinates were fixed before any of this was drawn. This script transcribes and checks.
#
# Placement alone, which needs no cluster and no GPU:
#
#   ./scripts/regime/regime_gpu_trajectory.sh
#
# The measurement stage, under the batch scheduler on two exclusive synge nodes.
# --exclusive is not optional here: a co-tenant shares the L2 that the vertical
# coordinate is measured against, the contention gate then discards the run, and
# the arm it belonged to loses its whole line. This is the recommended route:
#
#   sbatch --nodes=2 --ntasks=20 --partition=compute --time=04:00:00 \
#          --nodelist=synge-n01,synge-n02 --exclusive \
#          --export=ALL,MEASURE=1,VERIFY_TOPOLOGY=1 \
#          scripts/regime/regime_gpu_trajectory.sh
#
# Panel C needs only three grids, so it is much the cheaper thing to run first.
# A grid list holds spaces, so it is exported by name rather than written into the
# comma-separated --export value:
#
#   MEASURE_N1_LIST="40 50 61" \
#   sbatch --nodes=2 --ntasks=20 --partition=compute --time=01:00:00 \
#          --nodelist=synge-n01,synge-n02 --exclusive \
#          --export=ALL,MEASURE=1,VERIFY_TOPOLOGY=1,MEASURE_N1_LIST \
#          scripts/regime/regime_gpu_trajectory.sh
#
# Interactively, for a smoke test only. The runs then share the calling shell's
# CPUs and whatever else holds the device, which is the usual cause of a spurious
# contention verdict:
#
#   salloc -N 2 -n 20 -p compute -t 01:00:00 --nodelist=synge-n01,synge-n02
#   MEASURE=1 ./scripts/regime/regime_gpu_trajectory.sh
#
# Assumes the project has been built. The measurement stage additionally needs the
# CUDA targets, which build only where a toolkit is present:
#
#   cmake --build build --target regime-gpu-trajectory --parallel   # anywhere
#   cmake --build build --target ca-integrator ca-integrator-2gpu --parallel  # synge
set -uo pipefail

WORK_DIR="${ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"
if [[ ! -f "$WORK_DIR/CMakeLists.txt" || ! -d "$WORK_DIR/scripts/regime" ]]; then
    echo "Error: repository root not found at $WORK_DIR" >&2
    echo "Run from the repository root or export ROOT=/absolute/path/to/caksm." >&2
    exit 1
fi

BUILD_DIR="${BUILD_DIR:-$WORK_DIR/build}"
TRAJ="${TRAJ:-$BUILD_DIR/regime-gpu-trajectory}"
DATA_DIR="${DATA_DIR:-$WORK_DIR/data/regime}"
OUT_DIR="${OUT_DIR:-$DATA_DIR/regime_gpu_trajectory}"
CSV="${CSV:-$DATA_DIR/regime_gpu_trajectory.csv}"
LADDER_CSV="${LADDER_CSV:-$DATA_DIR/regime_reduction_ladder.csv}"
# The grid whose measured Krylov dimension prices the ladder. tau* is free of N but
# not of m, so which grid it is has to be stated rather than assumed.
LADDER_N1="${LADDER_N1:-61}"
REPORT="$OUT_DIR/report.txt"
PROVENANCE="$OUT_DIR/provenance.txt"

MACHINE="${MACHINE:-v100-pcie-16gb}"
POLICY="${POLICY:-both}"
PLACEMENT_CSV="${PLACEMENT_CSV:-$DATA_DIR/regime_placement.csv}"
# The arrangements to place, in order, e.g. TOPOLOGIES="1gpu-1node 8gpu-1node". Empty places
# every record the manifest carries. Name them for a publication run, so it repeats the same
# arrangements rather than whatever the manifest has grown since.
TOPOLOGIES="${TOPOLOGIES:-}"
# The common global grid sequence, predeclared. Every value is a grid the placement table
# measured, so no Krylov dimension on the figure is interpolated.
N1_LIST="${N1_LIST:-25 30 40 50 61 74 90 120}"
# The single-GPU grid the fixed-local arm sizes its heaviest slab against.
LOCAL_REF="${LOCAL_REF:-40}"
S="${S:-8}"
X_REUSE="${X_REUSE:-1.0}"
VERIFY_TOPOLOGY="${VERIFY_TOPOLOGY:-0}"
MEASURE="${MEASURE:-0}"
# Under sbatch the job's stdout goes to slurm-<jobid>.out in the submission
# directory unless --output said otherwise; the report below is written regardless.
MEASURED_CSV="${MEASURED_CSV:-$OUT_DIR/measured_m.csv}"
# The integrator that supplies the measured Krylov dimension per arrangement.
SINGLE="${SINGLE:-$BUILD_DIR/ca-integrator}"
DISTRIBUTED="${DISTRIBUTED:-$BUILD_DIR/ca-integrator-2gpu}"
# Grids to measure. Smaller than N1_LIST by default because a measurement run is
# 4 arms per grid; grids left out keep their placed coordinates and say so.
MEASURE_N1_LIST="${MEASURE_N1_LIST:-$N1_LIST}"
# m is deterministic given the operator and the tolerance, so one run settles it.
# The timed constants R_h needs were calibrated separately, over seven repeats.
MEASURE_REPEATS="${MEASURE_REPEATS:-1}"
# The integrator takes an expiry and a step count, and its per-step h is
# expiry/steps (src/ca_integrator_2gpu.cu). The placement table measured its Krylov
# dimension at h=0.01, so the two agree only when expiry/steps is 0.01. Getting this
# wrong does not fail: it silently measures a different operator. A ten-times step
# asked for m=30 where the placement contract needs 12, and the two numbers then
# describe different problems while looking like a machine disagreement.
MEASURE_EXPIRY="${MEASURE_EXPIRY:-1.0}"
MEASURE_STEPS="${MEASURE_STEPS:-100}"
MEASURE_H="${MEASURE_H:-0.01}"
MEASURE_TOL="${MEASURE_TOL:-1e-8}"
# The Krylov ceiling handed to the integrator, not a fixed width: the solver
# converges adaptively below it and reports the dimension each step needed. It
# cannot exceed kGpuCaMaxM, which include/gpu_ca_config.hpp fixes at 39 because
# the projected exponential's static shared footprint grows as m^2 and 39 is what
# fits 48 KiB. A larger value is not clamped, it is rejected, and under MPI the
# rejection surfaces as a signal rather than a message.
MEASURE_M="${MEASURE_M:-39}"
MEASURE_ARM="${MEASURE_ARM:-as-measured}"
MEASURE_OPTION="${MEASURE_OPTION:-basket}"
MEASURE_S="${MEASURE_S:-1}"
# Mirrors kGpuCaMaxM. Checked here so a bad ceiling costs a message rather than a
# scheduled job that fails once per arm per grid.
GPU_CA_MAX_M="${GPU_CA_MAX_M:-39}"
SRUN_MPI="${SRUN_MPI:-pmix}"

# Whether to run the host-only placement stage after any measurement.
#   auto  run it when its inputs are all present; otherwise say what is missing and,
#         if a measurement was taken, leave that measurement in place rather than
#         discarding a cluster job over a table the measurement never needed.
#   1     require the inputs and fail without them.
#   0     skip it: take the measurements here and render elsewhere.
PLACEMENT="${PLACEMENT:-auto}"

# What the placement stage transcribes. Checksummed into the provenance so a later
# recalibration that moved a constant cannot pass unnoticed under an unchanged figure.
#
# The measurement stage needs none of this. It needs two CUDA binaries and a
# scheduler, which is why the two stages are checked separately: a synge job that
# only takes measurements must not be turned away for want of a puffin table.
PLACEMENT_SOURCES=(
    "$PLACEMENT_CSV"
    "$DATA_DIR/calibrate_gpu_reduction.csv"
    "$WORK_DIR/include/gpu_topology.hpp"
    "$WORK_DIR/include/gpu_machine.hpp"
    "$WORK_DIR/src/regime_gpu_trajectory.cpp"
)

# The multi-node arms bootstrap NCCL through PMIx, whose default gds component
# crashes on this cluster; hash is the standard remedy and every other multi-node
# script here sets it. Without it the two-node arms die in MPI_Init, which surfaces
# as a signal rather than a message.
export PMIX_MCA_gds="${PMIX_MCA_gds:-hash}"

fail() { echo "Error: $*" >&2; exit 1; }

# macOS ships shasum, Linux sha256sum; this script runs on both, since the placement stage
# is host-only and is expected to run off the cluster as often as on it.
checksum() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1"
    else shasum -a 256 "$1"; fi
}

# BSD date has no -Is. Same reason as above.
stamp() { date -Is 2>/dev/null || date -u "+%Y-%m-%dT%H:%M:%S+00:00"; }

# What the placement stage is missing, if anything. Collected rather than fatal on
# sight, so the report can name every gap at once instead of one per resubmission.
PLACEMENT_MISSING=()
[[ -x "$TRAJ" ]] || PLACEMENT_MISSING+=("$TRAJ (build: cmake --build build --target regime-gpu-trajectory)")
for src in "${PLACEMENT_SOURCES[@]}"; do
    [[ -f "$src" ]] || PLACEMENT_MISSING+=("$src")
done

RUN_PLACEMENT=1
if [[ ${#PLACEMENT_MISSING[@]} -gt 0 ]]; then
    case "$PLACEMENT" in
        0) RUN_PLACEMENT=0 ;;
        auto)
            if [[ "$MEASURE" == "1" ]]; then
                RUN_PLACEMENT=0
            else
                printf 'Error: the placement stage is missing:\n' >&2
                printf '  %s\n' "${PLACEMENT_MISSING[@]}" >&2
                printf '%s\n' \
                    "" \
                    "regime_placement.csv is written by regime_placement.sh on puffin," \
                    "and data/ is untracked, so a checkout on another host will not have" \
                    "it. Either copy it in from a host that does:" \
                    "" \
                    "  scp <host>:<repo>/data/regime/regime_placement.csv \\" \
                    "      \"$DATA_DIR/regime_placement.csv\"" \
                    "" \
                    "or take the measurements here and render where the table already is:" \
                    "" \
                    "  PLACEMENT=0 MEASURE=1 $0" >&2
                exit 1
            fi
            ;;
        *)
            printf 'Error: PLACEMENT=1 was asked for but these are missing:\n' >&2
            printf '  %s\n' "${PLACEMENT_MISSING[@]}" >&2
            exit 1
            ;;
    esac
elif [[ "$PLACEMENT" == "0" ]]; then
    RUN_PLACEMENT=0
fi

mkdir -p "$OUT_DIR"

echo "==============================================================================="
echo " GPU regime trajectory   machine=$MACHINE  policy=$POLICY"
echo " host=$(uname -n)  date=$(stamp)"
echo "==============================================================================="
echo

# Stage 2 first when asked: a topology that the allocation cannot supply should stop the run
# before a CSV exists to be mistaken for a checked one.
if [[ "$VERIFY_TOPOLOGY" == "1" ]]; then
    echo "### Topology verification"
    [[ -n "${SLURM_JOB_ID:-}" ]] || fail "VERIFY_TOPOLOGY=1 needs a Slurm allocation"

    nodes="${SLURM_JOB_NUM_NODES:-0}"
    gpus_per_node="${SLURM_GPUS_PER_NODE:-}"
    if [[ -z "$gpus_per_node" ]] && command -v nvidia-smi >/dev/null 2>&1; then
        gpus_per_node="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"
    fi
    echo "  job          : ${SLURM_JOB_ID}"
    echo "  nodelist     : ${SLURM_JOB_NODELIST:-unknown}"
    echo "  nodes        : $nodes"
    echo "  GPUs per node: ${gpus_per_node:-unknown}"

    # The most demanding arm in the table: 2 nodes, 2 GPUs each.
    [[ "$nodes" -ge 2 ]] \
        || fail "the 2-node arms need -N 2; this allocation has $nodes node(s)"
    [[ "${gpus_per_node:-0}" -ge 2 ]] \
        || fail "the 2-GPU-per-node arms need 2 devices per node; found ${gpus_per_node:-0}"

    if command -v srun >/dev/null 2>&1; then
        srun --nodes="$nodes" --ntasks-per-node=1 bash -c \
            'echo "  --- $(uname -n) ---"; nvidia-smi topo -m 2>/dev/null | head -8' \
            > "$OUT_DIR/topology.txt" 2>&1
        sed 's/^/  /' "$OUT_DIR/topology.txt"
        echo "  [wrote $OUT_DIR/topology.txt]"
    fi
    echo "  every declared arrangement is supplied by this allocation."
    echo
fi

# The measurement stage. Each arrangement is launched exactly as
# ca_strong_scaling.sh launches the same four rungs, so the srun shapes here and
# the ones the accepted scaling study used are the same shapes.
run_arm() {
    local key="$1" n1="$2"; shift 2
    local log="$OUT_DIR/measure_${key}_n${n1}.txt"
    "$@" > "$log" 2>&1
    local status=$?

    # Contention first: a co-tenant shares the L2 the vertical coordinate is
    # measured against, so a contended run is discarded rather than annotated.
    local contended=0
    if grep -q 'CONTENDED DEVICE' "$log" || grep -q 'recordable=no' "$log"; then
        contended=1
    fi

    local verdict m unconverged
    if [[ $status -ne 0 ]]; then
        verdict="FAIL:exit_${status}"
        # An argument the binary rejects surfaces as exit 1 alone, and under MPI as
        # a signal, so the reason is only ever in the log. Carry its first line up.
        local why
        why="$(grep -m1 -iE '^(error|terminate|what\(\)|invalid)' "$log" \
               | cut -c1-72)"
        [[ -n "$why" ]] && printf '    %s\n' "$why"
    else
        verdict="$(grep -m1 'validation:' "$log" \
                   | sed -E 's/.*validation:[[:space:]]*([A-Za-z]+).*/\1/')"
        [[ -n "$verdict" ]] || verdict="FAIL:no-validation-line"
        # A step that never converged had its dimension truncated by the ceiling,
        # so max= is the ceiling rather than what the operator asked for.
        unconverged="$(grep -m1 'Krylov m:' "$log" \
                       | sed -E 's/.*unconverged=([0-9]+).*/\1/')"
        if [[ -n "$unconverged" && "$unconverged" != "0" ]]; then
            verdict="FAIL:unconverged_${unconverged}"
            printf '    ceiling m=%s bound on %s step(s); this grid needs a larger m\n' \
                "$MEASURE_M" "$unconverged"
        fi
    fi
    # The converged dimension is the largest any step needed: that is the width
    # the basis has to be sized for, and it is what the placement table records.
    m="$(grep -m1 'Krylov m:' "$log" \
         | sed -E 's/.*max=([0-9]+).*/\1/')"
    [[ -n "$m" ]] || m=0

    printf '%s,%s,%s,%s,%s,"%s"\n' \
        "$key" "$n1" "$m" "$verdict" "$contended" "$log" >> "$MEASURED_CSV"
    printf '  %-16s n=%-4s m=%-3s %-24s contended=%s\n' \
        "$key" "$n1" "$m" "$verdict" "$contended"
}

if [[ "$MEASURE" == "1" ]]; then
    echo "### Measurement: one solver run per arrangement and grid"
    [[ "$MEASURE_M" -le "$GPU_CA_MAX_M" ]] || fail \
"MEASURE_M=$MEASURE_M exceeds the GPU build's compile-time ceiling of $GPU_CA_MAX_M.
  include/gpu_ca_config.hpp caps m there because the projected exponential's static
  shared footprint grows as m^2. Lower MEASURE_M, or rebuild with a larger
  CAKSM_GPU_CA_MAX_M if the shared-memory budget allows it."
    step_h="$(awk -v e="$MEASURE_EXPIRY" -v s="$MEASURE_STEPS" \
                  'BEGIN { printf "%.10g", e / s }')"
    awk -v a="$step_h" -v b="$MEASURE_H" \
        'BEGIN { exit (a - b < 1e-12 && b - a < 1e-12) ? 0 : 1 }' || fail \
"the integrator's step size is expiry/steps = $step_h, but the placement table
  measured its Krylov dimension at h=$MEASURE_H. Those are different operators, and
  comparing their Krylov dimensions would compare different problems. Set
  MEASURE_STEPS=\$(expiry/h), or MEASURE_H to the placement table's h."

    [[ -n "${SLURM_JOB_ID:-}" ]] || fail "MEASURE=1 needs a Slurm allocation"
    command -v srun >/dev/null 2>&1 || fail "MEASURE=1 needs srun"
    for binary in "$SINGLE" "$DISTRIBUTED"; do
        [[ -x "$binary" ]] || fail "missing $binary
  Build the CUDA targets on synge: cmake --build build --parallel"
    done
    echo "  grids   : $MEASURE_N1_LIST"
    echo "  contract: option=$MEASURE_OPTION arm=$MEASURE_ARM"
    echo "            m_ceiling=$MEASURE_M (GPU build cap $GPU_CA_MAX_M) tol=$MEASURE_TOL"
    echo "            expiry=$MEASURE_EXPIRY steps=$MEASURE_STEPS -> h=$step_h"
    echo "            (the placement table's h; a mismatch measures another operator)"
    echo "            s=$MEASURE_S repeats=$MEASURE_REPEATS"
    echo

    printf '%s\n' "topology,n_global,m_measured,validation,contended,log" \
        > "$MEASURED_CSV"

    for n1 in $MEASURE_N1_LIST; do
        common=(
            --n "$n1" --m "$MEASURE_M" --s "$MEASURE_S"
            --steps "$MEASURE_STEPS" --expiry "$MEASURE_EXPIRY"
            --tol "$MEASURE_TOL"
            --repeats "$MEASURE_REPEATS" --option "$MEASURE_OPTION"
            --basis monomial --orth cholqr2 --arm "$MEASURE_ARM"
        )
        run_arm 1gpu-1node "$n1" \
            srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --mpi=none \
            "$SINGLE" "${common[@]}"
        run_arm 2gpu-1node "$n1" \
            srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --mpi=none \
            "$DISTRIBUTED" --devices 0,1 "${common[@]}"
        run_arm 2gpu-2node "$n1" \
            srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --mpi="$SRUN_MPI" \
            "$DISTRIBUTED" --require-mpi --devices 0 "${common[@]}"
        run_arm 4gpu-2node "$n1" \
            srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --mpi="$SRUN_MPI" \
            "$DISTRIBUTED" --require-mpi --devices 0,1 "${common[@]}"
    done
    echo
    echo "  [wrote $MEASURED_CSV]"
    echo
fi

# A measured table from any earlier run is used whether or not this invocation
# took the measurements, so the figure does not silently regress to placed points.
MEASURED_ARG=""
[[ -f "$MEASURED_CSV" ]] && MEASURED_ARG=1

if [[ "$RUN_PLACEMENT" == "0" ]]; then
    echo "### Placement: skipped"
    if [[ ${#PLACEMENT_MISSING[@]} -gt 0 ]]; then
        echo "  missing:"
        printf '    %s\n' "${PLACEMENT_MISSING[@]}"
    fi
    echo
    echo "  The measurements above stand on their own and are written to"
    echo "    $MEASURED_CSV"
    echo "  Render them where the placement table lives:"
    echo "    scp synge01:$MEASURED_CSV <repo>/data/regime/regime_gpu_trajectory/"
    echo "    ./scripts/regime/regime_gpu_trajectory.sh"
    echo "    uv run scripts/plots/regime_trajectory.py"
    echo "==============================================================================="
    exit 0
fi

echo "### Placement"
"$TRAJ" --machine "$MACHINE" \
        --policy "$POLICY" \
        --placement-csv "$PLACEMENT_CSV" \
        ${TOPOLOGIES:+--topologies "$TOPOLOGIES"} \
        --n1-list "$N1_LIST" \
        --local-ref "$LOCAL_REF" \
        ${MEASURED_ARG:+--measured-csv "$MEASURED_CSV"} \
        --ladder-csv "$LADDER_CSV" \
        --ladder-n1 "$LADDER_N1" \
        --s "$S" \
        --x-reuse "$X_REUSE" \
        --csv "$CSV" 2>&1 | tee "$REPORT"
STATUS="${PIPESTATUS[0]}"

{
    echo "host=$(uname -n)"
    echo "date=$(stamp)"
    echo "git_revision=$(git -C "$WORK_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_dirty=$(git -C "$WORK_DIR" diff --quiet 2>/dev/null && echo 0 || echo 1)"
    echo "slurm_job_id=${SLURM_JOB_ID:-none}"
    echo "slurm_nodes=${SLURM_JOB_NODELIST:-none}"
    echo "topology_verified=$VERIFY_TOPOLOGY"
    echo "machine=$MACHINE"
    echo "policy=$POLICY"
    echo "n1_list=$N1_LIST"
    echo "local_ref=$LOCAL_REF"
    echo "s=$S"
    echo "x_reuse=$X_REUSE"
    echo "exit_status=$STATUS"
    echo "measurement_stance=placement-only; no solver and no timer is in this loop."
    echo "  m_measured comes from the accepted placement table, the collective latencies from"
    echo "  the accepted idle-device calibrations, and the coordinates from the predeclared"
    echo "  byte model. Nothing here is fitted to an observed runtime."
    echo "measured_table=${MEASURED_ARG:+$MEASURED_CSV}"
    echo "measure_stage_run=$MEASURE"
    echo "source_checksums_sha256:"
    for src in "${PLACEMENT_SOURCES[@]}"; do
        [[ -f "$src" ]] && checksum "$src" | sed 's/^/  /'
    done
    echo "binary_checksums_sha256:"
    checksum "$TRAJ" | sed 's/^/  /'
} > "$PROVENANCE"

echo
echo "==============================================================================="
if [[ "$STATUS" -ne 0 ]]; then
    echo "FAILED ($STATUS). No line is drawn from a partial table."
    exit 1
fi
echo "Done."
echo "  $CSV            <- one row per (policy, topology, n)"
echo "  $LADDER_CSV   <- every rung against tau*"
echo "  $REPORT   <- the console report"
echo "  $PROVENANCE  <- inputs, checksums and the measurement stance"
echo
echo "Render it:"
echo "  uv run scripts/plots/regime_trajectory.py"
echo "==============================================================================="
