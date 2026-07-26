#!/bin/bash
# Calibrate the top two rungs: device-to-device (SYS) and node-to-node.
#
# These are the reason synge is in the study. A single V100's reduction is on-die and
# near-free, so one device can test theta_v but cannot test the horizontal mechanism or the
# Upper-Right corner at all: both live in the two rungs measured here. Holding N and the device
# fixed and changing only which rung the reduction crosses is the swept horizontal axis, and
# these two numbers are its right-hand half.
#
# On synge the intra-node link is SYS, PCIe plus a cross-socket UPI hop with no NVLink, so the
# two-GPU one-node configuration is already a strong, confound-free horizontal probe before any
# inter-node fabric enters. Confirm that with scripts/regime/gpu_probe.sh first: if topo -m
# reports NV# instead of SYS, this rung is nearly free and the ladder loses a rung.
#
# The binary detects which rung it is actually on from the rank hostnames and refuses to run
# if that disagrees with --tier, so a mis-specified --ntasks-per-node cannot silently record a
# node-tier latency against the intra-node slot.
#
# Usage:
#   ./scripts/regime/calibrate_gpu_p2p.sh              # both rungs, inside an allocation
#   sbatch --nodes=2 --gpus-per-node=2 --exclusive scripts/regime/calibrate_gpu_p2p.sh
set -uo pipefail

WORK_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
P2P="$WORK_DIR/build/calibrate-gpu-p2p"
DATA_DIR="$WORK_DIR/data/regime"

MACHINE="${MACHINE:-v100-pcie-16gb}"
ITERS="${ITERS:-2000}"
REPEATS="${REPEATS:-7}"
BW_BYTES="${BW_BYTES:-67108864}"   # 64 MiB for the bandwidth arm

if [[ ! -x "$P2P" ]]; then
    echo "Error: calibrate-gpu-p2p not found at $P2P"
    echo
    echo "It needs NCCL at configure time (MPI is optional, and buys only the NODE rung)."
    echo "If cmake printed 'calibrate-gpu-p2p skipped: NCCL not found', point it at one:"
    echo "  export NCCL_HOME=/path/holding/include/nccl.h/and/lib/libnccl.so"
    echo "  cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel"
    echo
    echo "No root needed: the nvidia-nccl-cu12 wheel carries both the header and the library,"
    echo "and the CMake search covers its nvidia/nccl/{include,lib} layout."
    echo
    echo "Do not work around this with a hand-rolled ring reduction. R_h's non-circularity"
    echo "rests on the calibrated constant being the slope of code an implementation would"
    echo "actually call, and a bespoke primitive measures something nobody runs."
    exit 1
fi

mkdir -p "$DATA_DIR"

echo "==============================================================================="
echo " GPU interconnect calibration   machine=$MACHINE"
echo "==============================================================================="
echo
echo "Provenance:"
echo "  job    : ${SLURM_JOB_ID:-<not under the scheduler>}"
echo "  nodes  : ${SLURM_JOB_NODELIST:-<none>}"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "  topology (the link the DEVICE_P2P rung crosses):"
    nvidia-smi topo -m 2>/dev/null | head -6 | sed 's/^/    /'
fi
echo

DEVICE_STATUS=0
NODE_STATUS=0

# The DEVICE_P2P rung. Both V100s are on one node, so ncclCommInitAll drives them from a
# single process: no srun, no MPI. This is the rung Phase 0 predicts carries the production
# operator into Upper-Right, so it is deliberately the one with the fewest dependencies.
echo "### 1/2  DEVICE_P2P: both local GPUs, one process, no MPI (the SYS link)"
echo
if "$P2P" --machine "$MACHINE" --tier device-p2p \
          --iters "$ITERS" --repeats "$REPEATS" --bw-bytes "$BW_BYTES" \
          --csv "$DATA_DIR/calibrate_gpu_p2p_device.csv"; then
    echo "  ok"
else
    echo "  FAILED. Needs two visible GPUs: check CUDA_VISIBLE_DEVICES and that the"
    echo "  allocation actually holds both devices."
    DEVICE_STATUS=1
fi
echo

# The NODE rung: one rank per node, so the crossing is the fabric. Needs an MPI build.
#
# Which transport NCCL chose is part of the measurement, not a debugging detail. NCCL has no
# native Omni-Path transport, so on an OPA fabric it falls back to its socket-based net
# transport. That still gives a real, recordable node-tier latency, but a very different one
# from IB verbs, and the write-up must say which it was. NCCL_DEBUG=INFO names the transport,
# captured to its own file rather than stdout because at INFO it is extremely verbose.
#
# No --gpus-per-node here. The binary picks its device by local rank, so one rank per node
# selects device 0 correctly whether or not the other GPU is visible, and on a site where GPUs
# are handed out with the node rather than through explicit GRES (synge is one) a
# --gpus-per-node request can be rejected outright. Override via SRUN_EXTRA if your site needs
# an explicit GPU request.
#
# --mpi=pmix is required, not cosmetic. Open MPI 5.x bootstraps through PMIx and srun supplies
# it only when asked; without it MPI_Init succeeds but every rank is a singleton with
# MPI_COMM_WORLD size 1. Each then drives its own node's GPUs and produces a DEVICE_P2P number
# that would be recorded against the NODE rung, a wrong constant that looks right. The binary
# refuses that outright, but the fix belongs here.
NCCL_LOG="$DATA_DIR/calibrate_gpu_p2p_node_nccl.log"
SRUN_EXTRA="${SRUN_EXTRA:-}"
SRUN_MPI="${SRUN_MPI:-pmix}"
echo "### 2/2  NODE: 1 rank per node, 2 nodes (the cluster fabric)"
echo
if ! command -v srun >/dev/null 2>&1; then
    echo "  SKIPPED: no srun. The fabric rung needs a launcher."
    NODE_STATUS=2
else
    echo "  PMI available at this site (the NODE rung needs one Open MPI 5.x can use):"
    srun --mpi=list 2>&1 | sed 's/^/    /'
    echo "  using --mpi=$SRUN_MPI  (override with SRUN_MPI=...)"
    echo
fi
# Two launch attempts, because a Spack Open MPI against a site Slurm has one common failure.
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
#
# Both produce the same measurement: one rank per node, so the crossing is the fabric.
run_node_rung() {   # <launcher-label> <command...>
    local label="$1"; shift
    echo "  attempt: $label"
    "$@" 2> >(tee "$NCCL_LOG" >&2)
}

if [[ $NODE_STATUS -eq 2 ]]; then
    : # already skipped above
elif run_node_rung "srun --mpi=$SRUN_MPI (PMIX_MCA_gds=hash)" \
        srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --mpi="$SRUN_MPI" ${SRUN_EXTRA} \
             --export=ALL,NCCL_DEBUG=INFO,NCCL_DEBUG_SUBSYS=INIT,NET,PMIX_MCA_gds=hash \
             "$P2P" --machine "$MACHINE" --tier node \
                    --iters "$ITERS" --repeats "$REPEATS" --bw-bytes "$BW_BYTES" \
                    --csv "$DATA_DIR/calibrate_gpu_p2p_node.csv"; then
    echo "  ok"
    echo
elif command -v mpirun >/dev/null 2>&1 && \
     PMIX_MCA_gds=hash NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET \
     run_node_rung "mpirun (Open MPI's own PMIx, bypassing Slurm's)" \
        mpirun -np 2 --map-by ppr:1:node \
             "$P2P" --machine "$MACHINE" --tier node \
                    --iters "$ITERS" --repeats "$REPEATS" --bw-bytes "$BW_BYTES" \
                    --csv "$DATA_DIR/calibrate_gpu_p2p_node.csv"; then
    echo "  ok (via mpirun; srun's PMIx path failed, so record which launcher was used)"
    echo
    echo "  Transport NCCL selected for this rung (record it alongside the latency):"
    grep -Eo 'NET/[A-Za-z]+|Using network [A-Za-z]+' "$NCCL_LOG" 2>/dev/null \
        | sort -u | sed 's/^/    /' || echo "    (not reported; see $NCCL_LOG)"
    if grep -q 'NET/Socket' "$NCCL_LOG" 2>/dev/null; then
        echo "    NOTE: socket transport, not verbs. Expected on Omni-Path, which NCCL does"
        echo "    not support natively. The latency is real and recordable, but it is the"
        echo "    latency of this fabric via sockets: say so rather than implying IB."
    fi
else
    echo "  FAILED: both srun and mpirun. Diagnose from the symptom in $NCCL_LOG:"
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
    echo "    'Only allocated 1 nodes asked for 2'"
    echo "      the enclosing allocation had -N 1. Carry the same partition and time limit:"
    echo "        salloc -N 2 -n 10 -p compute -t 14:00:00 --nodelist=synge-n01,synge-n02"
    echo
    echo "    'NCCL only, no MPI' at cmake time"
    echo "      no fabric path is compiled in. Load an MPI module and reconfigure."
    NODE_STATUS=1
fi
echo

echo "==============================================================================="
if [[ $DEVICE_STATUS -ne 0 ]]; then
    echo "The DEVICE_P2P rung did not land. That is the blocking one: it is where Phase 0"
    echo "predicts the corner opens at production resolution. reduction_calibrated must stay"
    echo "false and no horizontal verdict may be published."
    exit 1
fi
if [[ $NODE_STATUS -ne 0 ]]; then
    echo "The DEVICE_P2P rung landed; the NODE rung did not."
    echo
    echo "This is a usable state, but a partial one. Record DEVICE_P2P and set its"
    echo "tier_calibrated, and leave reduction_calibrated false: the ladder's top rung is"
    echo "still missing, so highest_reachable_tier() would name a tier with no measurement"
    echo "behind it. Phase B can proceed on the SYS rung alone, which Phase 0 predicts is the"
    echo "rung production resolution needs, but the swept horizontal axis is then two rungs"
    echo "rather than three and the write-up must say so."
    exit 0
fi
echo "Done."
echo "  data/regime/calibrate_gpu_p2p_device.csv  <- the SYS rung"
echo "  data/regime/calibrate_gpu_p2p_node.csv    <- the fabric rung"
echo
echo "Recording. The printed latencies are cumulative (a cross-link all-reduce performs the"
echo "on-device combines first); gpu_machine.hpp stores increments, so subtract the rung below"
echo "before recording or the lower rungs are counted twice:"
echo "  t_reduce_s[DEVICE_P2P] = t_allreduce(device) - reduction_cost_s(gm, GRID)"
echo "  t_reduce_s[NODE]       = t_allreduce(node)   - reduction_cost_s(gm, DEVICE_P2P)"
echo
echo "Then set tier_calibrated for both, interconnect_bw_gbs, and, once every reachable rung"
echo "is measured, reduction_calibrated = true."
echo "==============================================================================="
