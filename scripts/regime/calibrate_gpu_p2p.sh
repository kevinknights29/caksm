#!/bin/bash
#SBATCH --job-name=caksm-p2p
#SBATCH --output=caksm-p2p-%j.out
#
# Calibrate the two interconnect rungs, at every participant count the allocation can supply.
#
# These are the reason a multi-GPU host is in the study at all. A single device's reduction is
# on-die and near-free, so one GPU can test theta_v but cannot test the horizontal mechanism or
# the Upper-Right corner: both live in the rungs measured here. Holding N and the device fixed
# and changing only which rung the reduction crosses is the swept horizontal axis.
#
# The collective cost is keyed to the PARTICIPANT COUNT, not to the rung. Eight H200s on one
# node cost 2.011x what four of the same node cost, at the same DEVICE_P2P rung. So this script
# sweeps counts rather than taking one measurement per rung, and every count lands in its own
# file. An unmeasured count is left absent; nothing here interpolates one.
#
# Nothing about this script is specific to a cluster. The sweep is derived from what the
# allocation exposes: the device rung walks the powers of two up to the visible GPU count, and
# the fabric rung walks ranks per node the same way over however many nodes were allocated. On
# synge (2 GPUs, 2 nodes) that is exactly the arrangements synge has always measured; on an
# eight-GPU node it is 2, 4 and 8; on anything larger it adapts without an edit.
#
# Usage:
#   # inside an allocation, both rungs, sweep derived from the allocation
#   ./scripts/regime/calibrate_gpu_p2p.sh
#
#   # as a batch job; the geometry comes from the sbatch flags, not from this file
#   sbatch --nodes=2 --gpus-per-node=8 --exclusive --time=00:30:00 \
#          --export=ALL,MACHINE=h200 scripts/regime/calibrate_gpu_p2p.sh
#
#   # explicit arrangements, for a publication run that must repeat the same ones
#   MACHINE=h200 LOCAL_GPUS="2 4 8" RANKS_PER_NODE="1 8" ./scripts/regime/calibrate_gpu_p2p.sh
#
# Every run writes into its own directory, which is how the overwrite problem is fixed:
#
#   data/calibration/<machine>/<stamp>-job<jobid>/
#
# The calibrators write fixed filenames, so two runs of the same rung used to collide: the
# eight-participant H200 run overwrote the four-participant one and left only a transcript.
# A run directory named by the moment and the job cannot collide with an earlier one, so no
# accepted measurement is reachable by a later write. The file stems inside carry the exact
# arrangement, so two counts within one run cannot collide either.
#
# The data tree is untracked by design (.gitignore). What enters git is the reviewed entry in
# kGpuTopologies, which cites these paths in its source column; scripts/regime/topology_entry.sh
# prints that entry from a run directory so no measured number is ever retyped.
set -uo pipefail

WORK_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
P2P="${P2P:-$WORK_DIR/build/calibrate-gpu-p2p}"
PROBE="${PROBE:-$WORK_DIR/build/gpu-device-probe}"

MACHINE="${MACHINE:-v100-pcie-16gb}"
ITERS="${ITERS:-2000}"
REPEATS="${REPEATS:-7}"
BW_BYTES="${BW_BYTES:-67108864}"   # 64 MiB for the bandwidth arm

STAMP="${STAMP:-$(date +%Y-%m-%dT%H%M%S)}"
JOBID="${SLURM_JOB_ID:-local$$}"
RUN_DIR="${RUN_DIR:-$WORK_DIR/data/calibration/$MACHINE/$STAMP-job$JOBID}"

SRUN_EXTRA="${SRUN_EXTRA:-}"
SRUN_MPI="${SRUN_MPI:-pmix}"

if [[ ! -x "$P2P" ]]; then
    echo "Error: calibrate-gpu-p2p not found at $P2P"
    echo
    echo "It needs NCCL at configure time (MPI is optional, and buys only the NODE rung)."
    echo "If cmake printed 'calibrate-gpu-p2p skipped: NCCL not found', point it at one:"
    echo "  export NCCL_HOME=/path/holding/include/nccl.h/and/lib/libnccl.so"
    echo "  cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel"
    echo
    echo "Do not work around this with a hand-rolled ring reduction. R_h's non-circularity"
    echo "rests on the calibrated constant being the slope of code an implementation would"
    echo "actually call, and a bespoke primitive measures something nobody runs."
    exit 1
fi

# What this launch can actually see. CUDA_VISIBLE_DEVICES wins where it is already set, because
# the caller has then already chosen a subset and the sweep must stay inside it; otherwise the
# allocation's whole device list is the base.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    BASE_DEVICES="$CUDA_VISIBLE_DEVICES"
elif command -v nvidia-smi >/dev/null 2>&1; then
    BASE_DEVICES="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null \
                    | paste -sd, -)"
else
    BASE_DEVICES=""
fi
VISIBLE=0
[[ -n "$BASE_DEVICES" ]] && VISIBLE="$(awk -F, '{print NF}' <<< "$BASE_DEVICES")"
NODES="${NODES:-${SLURM_JOB_NUM_NODES:-1}}"

# The powers of two up to a bound. The participant counts a sweep should walk: doubling is what
# separates one arrangement from the next on the horizontal axis, and an arithmetic sweep would
# spend most of its runs where the collective barely moves.
power_ladder() {   # <max> -> "1 2 4 ... <= max"
    local max="$1" k=1 out=()
    while (( k <= max )); do out+=("$k"); k=$((k * 2)); done
    echo "${out[*]}"
}

# The first n of the visible devices, as a CUDA_VISIBLE_DEVICES mask.
first_devices() {  # <n> -> "0,1,..."
    awk -F, -v n="$1" '{for (i = 1; i <= n; i++) printf "%s%s", $i, (i < n ? "," : "")}' \
        <<< "$BASE_DEVICES"
}

# Participant counts for the intra-node rung. Starts at two: the rung is a crossing, and one
# device has nothing to cross.
if [[ -z "${LOCAL_GPUS:-}" ]]; then
    LOCAL_GPUS="$(power_ladder "$VISIBLE" | sed 's/^1 //; s/^1$//')"
fi
# Ranks per node for the fabric rung, one rank per GPU throughout, so participants = ranks.
RANKS_PER_NODE="${RANKS_PER_NODE:-$(power_ladder "$VISIBLE")}"

# A launcher that exits 0 without writing a table has measured nothing. Trusting the exit
# status would report a row that does not exist, which is the failure this layout exists to
# prevent, so the file is what gets checked.
record_output() {   # <csv> -> 0 when it is a real table
    local csv="$1"
    if [[ -s "$csv" ]]; then
        OUTPUTS+=("$csv")
        return 0
    fi
    echo "     FAILED: exited 0 but wrote no table to $(basename "$csv")"
    return 1
}

mkdir -p "$RUN_DIR" || { echo "Error: cannot create $RUN_DIR" >&2; exit 1; }

echo "==============================================================================="
echo " GPU interconnect calibration   machine=$MACHINE"
echo "==============================================================================="
echo
echo "Provenance:"
echo "  host   : $(uname -n)"
echo "  job    : ${SLURM_JOB_ID:-<not under the scheduler>}"
echo "  nodes  : ${SLURM_JOB_NODELIST:-<none>}  (allocated: $NODES)"
echo "  devices: $VISIBLE visible [$BASE_DEVICES]"
echo "  run dir: ${RUN_DIR#"$WORK_DIR"/}"
echo "  sweep  : device rung [${LOCAL_GPUS:-none}]  fabric rung [$RANKS_PER_NODE] x $NODES node(s)"
echo

# Provenance travels with the numbers, in the same directory. Which devices a subset used is
# not a detail: on a node whose halves have different NUMA and NIC affinity, two four-GPU
# subsets are different arrangements, and a measurement that cannot name its devices cannot be
# placed. That is exactly why one archived H200 run is still blocked.
{
    echo "machine=$MACHINE"
    echo "host=$(uname -n)"
    echo "kernel=$(uname -sr)"
    echo "slurm_job=${SLURM_JOB_ID:-none}"
    echo "slurm_nodelist=${SLURM_JOB_NODELIST:-none}"
    echo "nodes=$NODES"
    echo "visible_devices=$BASE_DEVICES"
    echo "local_gpus_sweep=${LOCAL_GPUS:-none}"
    echo "ranks_per_node_sweep=$RANKS_PER_NODE"
    echo "iters=$ITERS"
    echo "repeats=$REPEATS"
    echo "bw_bytes=$BW_BYTES"
    echo "git_revision=$(git -C "$WORK_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_dirty=$(git -C "$WORK_DIR" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
} > "$RUN_DIR/provenance.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
    # Raw, escape bytes and all. The manifest parser strips ANSI before reading the table; an
    # edited copy is no longer evidence.
    nvidia-smi topo -m                                        > "$RUN_DIR/topology.txt"  2>&1
    nvidia-smi --query-gpu=index,uuid,name --format=csv        > "$RUN_DIR/devices.csv"   2>&1
fi

DEVICE_STATUS=0
NODE_STATUS=0
OUTPUTS=()

# The DEVICE_P2P rung: every participant on one host, driven from a single process by
# ncclCommInitAll. No srun, no MPI. This is the rung Phase 0 predicts carries the production
# operator into Upper-Right, so it is deliberately the one with the fewest dependencies.
#
# The subset is selected with CUDA_VISIBLE_DEVICES rather than --devices because the binary's
# single-process path takes its participant count from cudaGetDeviceCount and currently ignores
# the parsed --devices list. Making --devices effective is Phase 2 of the port spec; until then
# masking is the honest way to pick a subset, and the mask is recorded beside every row.
echo "### 1/2  DEVICE_P2P: participants on one host, one process, no MPI"
echo
if [[ "$VISIBLE" -lt 2 ]]; then
    echo "  SKIPPED: $VISIBLE visible GPU(s). The rung is a crossing; one device has"
    echo "  nothing to cross. Check CUDA_VISIBLE_DEVICES and the allocation's GPU request."
    DEVICE_STATUS=2
else
    for COUNT in $LOCAL_GPUS; do
        MASK="$(first_devices "$COUNT")"
        CSV="$RUN_DIR/device_${COUNT}participants_1node.csv"
        LOG="$RUN_DIR/device_${COUNT}participants_1node.log"
        echo "  -- $COUNT participant(s), devices [$MASK] --"
        if CUDA_VISIBLE_DEVICES="$MASK" "$P2P" \
                --machine "$MACHINE" --tier device-p2p \
                --iters "$ITERS" --repeats "$REPEATS" --bw-bytes "$BW_BYTES" \
                --csv "$CSV" > "$LOG" 2>&1 && record_output "$CSV"; then
            grep -E 'all-reduce (latency|bus)' "$LOG" | sed 's/^/     /'
            echo "     ok"
        else
            echo "     FAILED; see ${LOG#"$WORK_DIR"/}"
            DEVICE_STATUS=1
        fi
        echo
    done
fi

# The NODE rung: the crossing is the fabric whenever more than one host takes part.
#
# Which transport NCCL chose is part of the measurement, not a debugging detail. NCCL has no
# native Omni-Path transport, so on an OPA fabric it falls back to its socket-based net
# transport. That still gives a real, recordable node-tier latency, but a very different one
# from IB verbs, and the write-up must say which it was. NCCL_DEBUG=INFO names the transport,
# captured to its own file rather than stdout because at INFO it is extremely verbose.
#
# No --gpus-per-node here. The binary picks its device as local_rank % visible devices, so R
# ranks per node take R distinct devices whether or not GRES was requested, and on a site where
# GPUs are handed out with the node rather than through explicit GRES (synge is one) a
# --gpus-per-node request can be rejected outright. Override via SRUN_EXTRA if your site needs
# an explicit GPU request.
#
# --mpi=pmix is required, not cosmetic. Open MPI 5.x bootstraps through PMIx and srun supplies
# it only when asked; without it MPI_Init succeeds but every rank is a singleton with
# MPI_COMM_WORLD size 1. Each then drives its own node's GPUs and produces a DEVICE_P2P number
# that would be recorded against the NODE rung, a wrong constant that looks right. The binary
# refuses that outright, but the fix belongs here.
echo "### 2/2  NODE: the cluster fabric"
echo
if [[ "$NODES" -lt 2 ]]; then
    echo "  SKIPPED: $NODES node(s). There is no fabric to cross from one host; allocate"
    echo "  more nodes to reach this rung."
    NODE_STATUS=2
elif ! command -v srun >/dev/null 2>&1; then
    echo "  SKIPPED: no srun. The fabric rung needs a launcher."
    NODE_STATUS=2
else
    echo "  PMI available at this site (the NODE rung needs one Open MPI 5.x can use):"
    srun --mpi=list 2>&1 | sed 's/^/    /'
    echo "  using --mpi=$SRUN_MPI  (override with SRUN_MPI=...)"
    echo
fi

# One geometry, two launch attempts, because a Spack Open MPI against a site Slurm has one
# common failure.
#
#   srun --mpi=pmix   Slurm's PMIx server (pmix_v5 here) talks to the PMIx client bundled
#                     inside Open MPI (libpmix.so.2). When those disagree, PMIx_Init can
#                     segfault in the shared-memory GDS component, seen on synge as
#                     pmix_gds_shmem2_fetch -> pmix_hwloc_setup_topology -> PMIx_Init.
#                     PMIX_MCA_gds=hash disables that component and is the standard remedy.
#   mpirun            Open MPI's own launcher (prte) bootstraps with its own PMIx end to end,
#                     so the mismatch cannot arise. Slower to start, but it does not care what
#                     Slurm's PMIx version is. The fallback rather than the default, because
#                     srun inherits binding and GPU visibility more predictably.
run_node_geometry() {   # <nodes> <ranks-per-node>
    local nodes="$1" rpn="$2"
    local participants=$((nodes * rpn))
    local csv="$RUN_DIR/node_${participants}participants_${nodes}nodes.csv"
    local log="$RUN_DIR/node_${participants}participants_${nodes}nodes.log"
    local nccl="$RUN_DIR/node_${participants}participants_${nodes}nodes_nccl.log"

    echo "  -- $participants participant(s): $rpn rank(s) x $nodes node(s) --"

    # Two ranks on one device is not a participant count. local_rank % ndev wraps, so the extra
    # ranks reduce over a GPU another rank already holds, and the row would name a crossing
    # that never happened.
    if [[ "$VISIBLE" -gt 0 && "$rpn" -gt "$VISIBLE" ]]; then
        echo "     SKIPPED: $rpn ranks per node but only $VISIBLE visible GPU(s)."
        echo "     Ranks would share devices, which is not a participant count."
        return 1
    fi

    if srun --nodes="$nodes" --ntasks="$participants" --ntasks-per-node="$rpn" \
            --mpi="$SRUN_MPI" ${SRUN_EXTRA} \
            --export=ALL,NCCL_DEBUG=INFO,NCCL_DEBUG_SUBSYS=INIT,NET,PMIX_MCA_gds=hash \
            "$P2P" --machine "$MACHINE" --tier node \
                   --iters "$ITERS" --repeats "$REPEATS" --bw-bytes "$BW_BYTES" \
                   --csv "$csv" > "$log" 2> "$nccl"; then
        echo "     ok (srun)"
    elif command -v mpirun >/dev/null 2>&1 && \
         PMIX_MCA_gds=hash NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET \
         mpirun -np "$participants" --map-by "ppr:$rpn:node" \
             "$P2P" --machine "$MACHINE" --tier node \
                    --iters "$ITERS" --repeats "$REPEATS" --bw-bytes "$BW_BYTES" \
                    --csv "$csv" > "$log" 2> "$nccl"; then
        echo "     ok (mpirun; srun's PMIx path failed, so record which launcher was used)"
    else
        echo "     FAILED under both launchers; see ${nccl#"$WORK_DIR"/}"
        return 1
    fi

    record_output "$csv" || return 1

    grep -E 'all-reduce (latency|bus)' "$log" | sed 's/^/     /'
    echo "     transport:"
    grep -Eo 'NET/[A-Za-z]+|Using network [A-Za-z]+' "$nccl" 2>/dev/null \
        | sort -u | sed 's/^/       /' || echo "       (not reported; see $nccl)"
    if grep -q 'NET/Socket' "$nccl" 2>/dev/null; then
        echo "     NOTE: socket transport, not verbs. Expected on Omni-Path, which NCCL does"
        echo "     not support natively. The latency is real and recordable, but it is the"
        echo "     latency of this fabric via sockets: say so rather than implying IB."
    fi
    return 0
}

if [[ $NODE_STATUS -eq 0 ]]; then
    NODE_LANDED=0
    for RPN in $RANKS_PER_NODE; do
        run_node_geometry "$NODES" "$RPN" && NODE_LANDED=$((NODE_LANDED + 1))
        echo
    done
    if [[ $NODE_LANDED -eq 0 ]]; then
        echo "  No fabric geometry landed. Diagnose from the symptom in the NCCL logs:"
        echo
        echo "    Segfault inside PMIx_Init (pmix_gds_shmem2_fetch, pmix_hwloc_setup_topology)"
        echo "      Slurm's PMIx server and Open MPI's bundled PMIx client disagree. Already"
        echo "      mitigated above with PMIX_MCA_gds=hash; if it still crashes, try also"
        echo "      PMIX_MCA_psec=native, or an Open MPI built against the SITE's PMIx."
        echo
        echo "    'MPI_COMM_WORLD has 1 rank' from the binary"
        echo "      the ranks never formed a communicator: a PMI bootstrap failure, not a bad"
        echo "      geometry. Try SRUN_MPI=pmi2, or let the mpirun fallback handle it."
        echo
        echo "    'NCCL only, no MPI' at cmake time"
        echo "      no fabric path is compiled in. Load an MPI module and reconfigure."
        NODE_STATUS=1
        echo
    fi
fi

# The geometry and overlap fields the calibrators do not report, and which a preset must not
# guess. Cheap, so it rides along with every run rather than needing its own job.
if [[ -x "$PROBE" ]]; then
    "$PROBE" > "$RUN_DIR/gpu_device_probe.txt" 2>&1 \
        && echo "### gpu-device-probe recorded" \
        || echo "### gpu-device-probe FAILED; see gpu_device_probe.txt"
    echo
fi

# Checksums travel with the run, so a later reader can prove the directory matches what was
# measured without trusting that nobody edited it in place. It also survives the trip when the
# tree is mailed as an attachment, which is how these reach the repository owner.
( cd "$RUN_DIR" && { shasum -a 256 -- * 2>/dev/null || sha256sum -- * ; } > SHA256SUMS.tmp \
  && mv SHA256SUMS.tmp SHA256SUMS ) 2>/dev/null

echo "==============================================================================="
if [[ ${#OUTPUTS[@]} -eq 0 ]]; then
    echo "Nothing was measured. Neither rung produced a row; see the logs in"
    echo "  ${RUN_DIR#"$WORK_DIR"/}"
    exit 1
fi
echo "Measured, in ${RUN_DIR#"$WORK_DIR"/}"
for out in "${OUTPUTS[@]}"; do
    echo "  $(basename "$out")"
done
echo
echo "Recording. The printed latencies are cumulative (a cross-link all-reduce performs the"
echo "on-device combines first). Two destinations, and they are not the same thing:"
echo
echo "  1. gpu_machine.hpp stores INCREMENTS over the rung below, and may hold an off-device"
echo "     rung only where the cluster offers exactly one participant count:"
echo "       t_reduce_s[DEVICE_P2P] = t_allreduce(device) - reduction_cost_s(gm, GRID)"
echo "       t_reduce_s[NODE]       = t_allreduce(node)   - reduction_cost_s(gm, DEVICE_P2P)"
echo
echo "  2. kGpuTopologies in include/gpu_topology.hpp stores the TOTAL for each exact"
echo "     arrangement, one entry per participant/node/local-GPU key. That is where a sweep"
echo "     belongs: a preset indexed by rung alone cannot hold two participant counts."
echo
echo "The entries for this run, ready to paste, with every measured field filled in and only"
echo "the judgments left as TODO:"
echo
"$WORK_DIR/scripts/regime/topology_entry.sh" "$RUN_DIR" | sed 's/^/  /'
if [[ $DEVICE_STATUS -eq 1 || $NODE_STATUS -eq 1 ]]; then
    echo
    echo "Some arrangements failed. Leave those out of kGpuTopologies rather than recording"
    echo "them uncalibrated: an arrangement that never completed has no latency at all."
fi
echo "==============================================================================="
exit 0
