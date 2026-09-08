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
# node cost 1.890x what four of the same node cost, at the same DEVICE_P2P rung. So this script
# sweeps counts rather than taking one measurement per rung, and every count lands in its own
# file. An unmeasured count is left absent; nothing here interpolates one.
#
# Nothing about this script is specific to a cluster. The sweep is derived from what the
# allocation exposes: the device rung walks the powers of two up to the visible GPU count, and
# the fabric rung walks ranks per node the same way over however many nodes were allocated. On
# synge (2 GPUs, 2 nodes) that is exactly the arrangements synge has always measured; on an
# eight-GPU node it is 2 through 8; on anything larger it adapts without an edit.
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
REQUIRE_COMPLETE_SWEEP="${REQUIRE_COMPLETE_SWEEP:-0}"
REQUIRE_PROBE="${REQUIRE_PROBE:-0}"
REQUIRE_NCCL_TRANSPORT="${REQUIRE_NCCL_TRANSPORT:-}"
REQUIRE_IDLE="${REQUIRE_IDLE:-0}"
# Run the multi-rank launch on a single node, where it measures the device rung rather than
# the fabric. It changes only the launch structure, which is what separates that structure
# from the node split when two arrangements differ in both.
MPI_ON_ONE_NODE="${MPI_ON_ONE_NODE:-0}"

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
[[ "$NODES" =~ ^[0-9]+$ && "$NODES" -gt 0 ]] \
    || { echo "Error: NODES must be positive." >&2; exit 1; }
for flag in "$REQUIRE_COMPLETE_SWEEP" "$REQUIRE_PROBE" "$REQUIRE_IDLE"; do
    [[ "$flag" == "0" || "$flag" == "1" ]] \
        || { echo "Error: requirement flags must be 0 or 1." >&2; exit 1; }
done

# Every count from `lo` to `max`. The sweep is dense rather than doubling because these arms are
# nearly free -- a whole eight-GPU job, single-device calibration and device probe included, ran
# in 34 seconds -- while a sparse sweep quietly assumes the shape of the curve it exists to
# measure. Three points (2, 4, 8) fit an affine law and a p^0.95 power law equally well, to
# within 3%, so they cannot establish that the collective is linear in participant count; seven
# can. A dense sweep is also the only one that can see a discontinuity at a count that does not
# factor as a power of two, which is where NCCL might switch algorithm or channel layout.
#
# The expensive sweep is the other one: scripts/regime/regime_gpu_trajectory.sh runs a full
# solver per arm and keeps doubling on purpose.
participant_counts() {   # <lo> <max> -> "lo lo+1 ... max"
    local lo="$1" max="$2" k out=()
    for (( k = lo; k <= max; ++k )); do out+=("$k"); done
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
    LOCAL_GPUS="$(participant_counts 2 "$VISIBLE")"
fi
# Ranks per node for the fabric rung, one rank per GPU throughout, so participants = ranks.
RANKS_PER_NODE="${RANKS_PER_NODE:-$(participant_counts 1 "$VISIBLE")}"

# A launcher that exits 0 without writing a table has measured nothing. Trusting the exit
# status would report a row that does not exist, which is the failure this layout exists to
# prevent, so the file is what gets checked.
record_output() {   # <csv> -> 0 when it is a real table
    local csv="$1"
    if [[ -s "$csv" ]]; then
        OUTPUTS+=("$csv")
        if [[ "$REQUIRE_IDLE" -eq 1 ]] && awk -F, '
                NR == 1 { for (i = 1; i <= NF; ++i) if ($i == "contended") column = i }
                NR > 1 && column && $column != 0 { bad = 1 }
                END { exit !column || bad }
            ' "$csv"; then
            :
        elif [[ "$REQUIRE_IDLE" -eq 1 ]]; then
            echo "     FAILED: $(basename "$csv") reports a contended device."
            return 1
        fi
        return 0
    fi
    echo "     FAILED: exited 0 but wrote no table to $(basename "$csv")"
    return 1
}

selected_nccl_transport() {   # <log>
    local log="$1" selected
    selected="$(sed -n -E \
        's/.*Using network ([A-Za-z0-9_.-]+).*/\1/p' "$log" 2>/dev/null | tail -1)"
    if [[ -z "$selected" ]]; then
        selected="$(sed -n -E \
            's@.*NET/([A-Za-z0-9_.-]+).*:[[:space:]]+Using .*@\1@p' \
            "$log" 2>/dev/null | tail -1)"
    fi
    echo "$selected"
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
    echo "required_nccl_transport=${REQUIRE_NCCL_TRANSPORT:-none}"
    echo "git_revision=$(git -C "$WORK_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
    if git -C "$WORK_DIR" diff --quiet 2>/dev/null \
            && git -C "$WORK_DIR" diff --cached --quiet 2>/dev/null; then
        echo "git_tracked_dirty=0"
    else
        echo "git_tracked_dirty=1"
    fi
} > "$RUN_DIR/provenance.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
    # Raw, escape bytes and all. The manifest parser strips ANSI before reading the table; an
    # edited copy is no longer evidence.
    nvidia-smi topo -m                                        > "$RUN_DIR/topology.txt"  2>&1
    # The matrix gives the link COUNT; this gives each link's rate and whether it is up, which
    # is what attributes an intra-node rung to NVLink rather than inferring it from bandwidth.
    # On a host without NVLink the tool's own message is the evidence, so it is kept.
    nvidia-smi nvlink -s                                      > "$RUN_DIR/nvlink.txt"    2>&1
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
        if [[ ! "$COUNT" =~ ^[0-9]+$ || "$COUNT" -lt 2 || "$COUNT" -gt "$VISIBLE" ]]; then
            echo "  FAILED: invalid device participant count '$COUNT' for $VISIBLE GPUs."
            DEVICE_STATUS=1
            continue
        fi
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
# Which transport NCCL chose is part of the measurement, not a debugging detail. NCCL_DEBUG
# names it, and the output goes to a separate file because INFO is verbose.
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
if [[ "$NODES" -lt 2 ]]; then
    echo "### 2/2  DEVICE_P2P again, through the multi-rank launch structure"
else
    echo "### 2/2  NODE: the cluster fabric"
fi
echo
if [[ "$NODES" -lt 2 && "$MPI_ON_ONE_NODE" != "1" ]]; then
    echo "  SKIPPED: $NODES node(s). There is no fabric to cross from one host; allocate more"
    echo "  nodes to reach that rung, or set MPI_ON_ONE_NODE=1 to run the multi-rank launch"
    echo "  here, which measures the device rung a second way and is the control for the"
    echo "  launch-structure confound."
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
    if [[ ! "$nodes" =~ ^[0-9]+$ || ! "$rpn" =~ ^[0-9]+$ \
            || "$nodes" -lt 1 || "$rpn" -lt 1 ]]; then
        echo "     FAILED: invalid node geometry '$nodes nodes x $rpn ranks'."
        return 1
    fi
    local participants=$((nodes * rpn))
    # One node is not the fabric. The binary decides the rung from the rank hostnames and
    # refuses a --tier that disagrees, which is the check that stops a single-node latency
    # being recorded against the NODE rung. The stem differs too: same devices, same split,
    # different launch structure from the single-process file of that count, and telling the
    # two apart is the entire purpose of running this.
    local tier="node" stem="node_${participants}participants_${nodes}nodes"
    if [[ "$nodes" -lt 2 ]]; then
        tier="device-p2p"
        stem="device_${participants}participants_1node_mpi"
    fi
    local csv="$RUN_DIR/${stem}.csv"
    local log="$RUN_DIR/${stem}.log"
    local nccl="$RUN_DIR/${stem}_nccl.log"
    local nccl_base="${nccl%.log}"
    local launcher=""
    local stderr=""
    local debug_file=""

    echo "  -- $participants participant(s): $rpn rank(s) x $nodes node(s), 1 GPU per rank"\
"  [$tier] --"

    # Two ranks on one device is not a participant count. local_rank % ndev wraps, so the extra
    # ranks reduce over a GPU another rank already holds, and the row would name a crossing
    # that never happened.
    if [[ "$VISIBLE" -gt 0 && "$rpn" -gt "$VISIBLE" ]]; then
        echo "     SKIPPED: $rpn ranks per node but only $VISIBLE visible GPU(s)."
        echo "     Ranks would share devices, which is not a participant count."
        return 1
    fi

    stderr="${nccl_base}.srun.stderr"
    debug_file="${nccl_base}.srun.rank.%h.%p.log"
    if NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET NCCL_DEBUG_FILE="$debug_file" \
       PMIX_MCA_gds=hash \
       srun --nodes="$nodes" --ntasks="$participants" --ntasks-per-node="$rpn" \
            --mpi="$SRUN_MPI" ${SRUN_EXTRA} \
            --export=ALL \
            "$P2P" --machine "$MACHINE" --tier "$tier" \
                   --iters "$ITERS" --repeats "$REPEATS" --bw-bytes "$BW_BYTES" \
                   --csv "$csv" > "$log" 2> "$stderr"; then
        launcher="srun"
        echo "     ok (srun)"
    elif command -v mpirun >/dev/null 2>&1; then
        stderr="${nccl_base}.mpirun.stderr"
        debug_file="${nccl_base}.mpirun.rank.%h.%p.log"
        if PMIX_MCA_gds=hash NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET \
           NCCL_DEBUG_FILE="$debug_file" \
           mpirun -np "$participants" --map-by "ppr:$rpn:node" \
               "$P2P" --machine "$MACHINE" --tier "$tier" \
                      --iters "$ITERS" --repeats "$REPEATS" --bw-bytes "$BW_BYTES" \
                      --csv "$csv" > "$log" 2> "$stderr"; then
            launcher="mpirun"
            echo "     ok (mpirun; srun's PMIx path failed, so record which launcher was used)"
        fi
    fi
    if [[ -z "$launcher" ]]; then
        echo "     FAILED under both launchers; see ${nccl#"$WORK_DIR"/}"
        return 1
    fi

    record_output "$csv" || return 1

    : > "$nccl"
    [[ ! -s "$stderr" ]] || cat "$stderr" >> "$nccl"
    local shard
    for shard in "${nccl_base}.${launcher}.rank."*.log; do
        [[ -f "$shard" ]] || continue
        cat "$shard" >> "$nccl"
    done
    # Keep a fallback for NCCL versions that ignore NCCL_DEBUG_FILE.
    grep 'NCCL INFO' "$log" >> "$nccl" 2>/dev/null || true

    grep -E 'all-reduce (latency|bus)' "$log" | sed 's/^/     /'
    local transport
    transport="$(selected_nccl_transport "$nccl")"
    echo "     transport:"
    if [[ -n "$transport" ]]; then
        echo "       $transport"
    else
        echo "       (not reported; see $nccl)"
    fi
    if [[ -n "$REQUIRE_NCCL_TRANSPORT" ]]; then
        local required="${REQUIRE_NCCL_TRANSPORT#NET/}"
        if [[ -z "$transport" ]]; then
            echo "     FAILED: NCCL did not report the selected transport."
            return 1
        fi
        if [[ "$required" != "any" && "$transport" != "$required" ]]; then
            echo "     FAILED: required NCCL transport '$required', got '$transport'."
            return 1
        fi
    fi
    if [[ "$transport" == "Socket" ]]; then
        echo "     NOTE: socket transport, not verbs. The latency is recordable, but it must"
        echo "     be identified as a socket result rather than an InfiniBand result."
    fi
    return 0
}

if [[ $NODE_STATUS -eq 0 ]]; then
    NODE_LANDED=0
    NODE_FAILED=0
    for RPN in $RANKS_PER_NODE; do
        if run_node_geometry "$NODES" "$RPN"; then
            NODE_LANDED=$((NODE_LANDED + 1))
        else
            NODE_FAILED=$((NODE_FAILED + 1))
        fi
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
    elif [[ $NODE_FAILED -gt 0 ]]; then
        NODE_STATUS=1
        echo "  $NODE_FAILED requested fabric geometry or geometries failed."
        echo
    fi
fi

# The geometry and overlap fields the calibrators do not report, and which a preset must not
# guess. Cheap, so it rides along with every run rather than needing its own job.
PROBE_STATUS=2
if [[ -x "$PROBE" ]]; then
    if "$PROBE" > "$RUN_DIR/gpu_device_probe.txt" 2>&1; then
        PROBE_STATUS=0
        echo "### gpu-device-probe recorded"
    else
        PROBE_STATUS=1
        echo "### gpu-device-probe FAILED; see gpu_device_probe.txt"
    fi
    echo
elif [[ "$REQUIRE_PROBE" -eq 1 ]]; then
    PROBE_STATUS=1
    echo "### gpu-device-probe missing at $PROBE"
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
if [[ "$REQUIRE_COMPLETE_SWEEP" -eq 1 ]] \
        && [[ $DEVICE_STATUS -eq 1 || $NODE_STATUS -eq 1 ]]; then
    exit 1
fi
if [[ "$REQUIRE_PROBE" -eq 1 && $PROBE_STATUS -ne 0 ]]; then
    exit 1
fi
exit 0
