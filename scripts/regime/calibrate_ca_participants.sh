#!/bin/bash
# Measure the participant-count term used by the distributed CA integrator.
#
# The production solver uses two MPI processes, one on each Synge node.  Each
# process drives either one local GPU (two NCCL participants total) or both
# local GPUs (four participants total).  This script preserves that topology
# while sweeping the exact norm, projection, Gram, and halo payloads.
#
# Run under the batch scheduler on two exclusive nodes. These latencies become
# model constants that later predictions are checked against, so this is the
# recommended route:
#
#   sbatch --nodes=2 --ntasks=20 --partition=compute --time=01:00:00 \
#          --nodelist=synge-n01,synge-n02 --exclusive \
#          scripts/regime/calibrate_ca_participants.sh
#
# Interactively, for a smoke test only. SLURM_OVERLAP shares the calling shell's
# CPUs with every step below, so its timings can vary:
#
#   salloc -N 2 -n 20 -p compute -t 01:00:00 --nodelist=synge-n01,synge-n02
#   SLURM_OVERLAP=1 ./scripts/regime/calibrate_ca_participants.sh
#
# The binary rejects singleton MPI worlds, same-node placement, asymmetric
# device counts, unavailable devices, and any contended participant.

set -uo pipefail

WORK_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
CALIBRATOR="$WORK_DIR/build/calibrate-gpu-p2p"
OUT_DIR="${OUT_DIR:-$WORK_DIR/data/ca-participant-calibration}"

MACHINE="${MACHINE:-v100-pcie-16gb}"
ITERS="${ITERS:-2000}"
HALO_ITERS="${HALO_ITERS:-200}"
REPEATS="${REPEATS:-7}"
BW_BYTES="${BW_BYTES:-67108864}"
SRUN_MPI="${SRUN_MPI:-pmix}"
SRUN_EXTRA="${SRUN_EXTRA:-}"

if [[ ! -x "$CALIBRATOR" ]]; then
    echo "Error: calibrate-gpu-p2p not found at $CALIBRATOR"
    echo "Build the calibrator after configuring with MPI and NCCL."
    exit 1
fi

if ! command -v srun >/dev/null 2>&1 \
   && ! command -v mpirun >/dev/null 2>&1; then
    echo "Error: neither srun nor mpirun is available."
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "==============================================================================="
echo " CA participant-count calibration"
echo "==============================================================================="
echo "  output       : $OUT_DIR"
echo "  machine      : $MACHINE"
echo "  repeats      : $REPEATS"
echo "  collectives  : $ITERS iterations/repeat"
echo "  halos        : $HALO_ITERS iterations/repeat"
echo "  bandwidth    : $BW_BYTES bytes"
echo "  Slurm job    : ${SLURM_JOB_ID:-<not under Slurm>}"
echo "  Slurm nodes  : ${SLURM_JOB_NODELIST:-<not under Slurm>}"
echo

# The iteration and repeat counts these constants were measured at, kept beside
# the CSVs because a reader comparing two calibrations needs to know whether
# they were taken at the same depth.
{
    echo "machine=$MACHINE"
    echo "iters=$ITERS"
    echo "halo_iters=$HALO_ITERS"
    echo "repeats=$REPEATS"
    echo "bw_bytes=$BW_BYTES"
    echo "timing_nccl_debug=WARN"
    echo "diagnostic_nccl_debug=INFO"
    echo "diagnostic_nccl_debug_subsys=INIT,NET,GRAPH,TUNING,P2P"
} > "$OUT_DIR/settings.txt"

export PMIX_MCA_gds=hash

run_with_srun() {
    local devices="$1"
    local csv="$2"
    local iterations="$3"
    local halo_iterations="$4"
    local repeats="$5"
    local debug="$6"
    local debug_subsys="$7"
    export NCCL_DEBUG="$debug"
    export NCCL_DEBUG_SUBSYS="$debug_subsys"
    srun \
        --nodes=2 --ntasks=2 --ntasks-per-node=1 --mpi="$SRUN_MPI" \
        ${SRUN_EXTRA} --export=ALL \
        "$CALIBRATOR" \
            --machine "$MACHINE" --tier node --production \
            --devices "$devices" \
            --iters "$iterations" --halo-iters "$halo_iterations" \
            --repeats "$repeats" --bw-bytes "$BW_BYTES" \
            --csv "$csv"
}

run_with_mpirun() {
    local devices="$1"
    local csv="$2"
    local iterations="$3"
    local halo_iterations="$4"
    local repeats="$5"
    local debug="$6"
    local debug_subsys="$7"
    export NCCL_DEBUG="$debug"
    export NCCL_DEBUG_SUBSYS="$debug_subsys"
    mpirun -np 2 --map-by ppr:1:node \
        "$CALIBRATOR" \
            --machine "$MACHINE" --tier node --production \
            --devices "$devices" \
            --iters "$iterations" --halo-iters "$halo_iterations" \
            --repeats "$repeats" --bw-bytes "$BW_BYTES" \
            --csv "$csv"
}

run_arm() {
    local participants="$1"
    local devices="$2"
    local stem="$OUT_DIR/node_${participants}participants"
    local csv="${stem}.csv"
    local stdout_log="${stem}.txt"
    local nccl_log="${stem}_nccl.log"
    local diagnostic_csv="${stem}_diagnostic.csv"
    local diagnostic_log="${stem}_diagnostic.txt"

    echo "### ${participants} NCCL participants: devices=$devices"
    echo

    # Timing must be quiet.  NCCL's TUNING subsystem prints one line per
    # collective, and putting it on the 2,000-iteration path measures logging
    # and terminal backpressure rather than the solver's submission cost.
    local completed=0
    if command -v srun >/dev/null 2>&1; then
        if run_with_srun \
             "$devices" "$csv" "$ITERS" "$HALO_ITERS" "$REPEATS" \
             WARN INIT 2>&1 | tee "$stdout_log"; then
            echo "  completed with srun --mpi=$SRUN_MPI"
            completed=1
        fi
    fi
    if [[ $completed -eq 0 ]] && command -v mpirun >/dev/null 2>&1; then
        if run_with_mpirun \
             "$devices" "$csv" "$ITERS" "$HALO_ITERS" "$REPEATS" \
             WARN INIT 2>&1 | tee "$stdout_log"; then
            echo "  completed with mpirun after the srun path failed"
            completed=1
        fi
    fi
    if [[ $completed -eq 0 ]]; then
        echo "FAILED: ${participants}-participant calibration did not complete."
        echo "Inspect $stdout_log and $nccl_log."
        return 1
    fi

    # Collect algorithm, protocol, channel, transport, and GDR evidence in a
    # separate one-repeat diagnostic.  These timings are deliberately not
    # recordable and never overwrite the quiet production CSV.
    local diagnosed=0
    if command -v srun >/dev/null 2>&1; then
        if run_with_srun \
             "$devices" "$diagnostic_csv" 1 1 1 \
             INFO INIT,NET,GRAPH,TUNING,P2P \
             2>&1 | tee "$diagnostic_log"; then
            diagnosed=1
        fi
    fi
    if [[ $diagnosed -eq 0 ]] && command -v mpirun >/dev/null 2>&1; then
        if run_with_mpirun \
             "$devices" "$diagnostic_csv" 1 1 1 \
             INFO INIT,NET,GRAPH,TUNING,P2P \
             2>&1 | tee "$diagnostic_log"; then
            diagnosed=1
        fi
    fi
    if [[ $diagnosed -eq 0 ]]; then
        echo "FAILED: ${participants}-participant NCCL diagnostic did not complete."
        echo "The quiet timing CSV exists, but its required transport evidence is incomplete."
        return 1
    fi

    grep 'NCCL INFO' "$diagnostic_log" > "$nccl_log" || true
    if [[ ! -s "$nccl_log" ]]; then
        echo "FAILED: no NCCL INFO evidence was recovered from $diagnostic_log"
        return 1
    fi

    echo "  transport/tuning evidence:"
    grep -Ei \
        'NET/|GDR|Channel|Pattern|Protocol|algorithm|TUNING|P2P' \
        "$nccl_log" 2>/dev/null | sort -u | sed 's/^/    /' | head -80 \
        || true
    echo
}

if ! run_arm 2 "0"; then
    exit 1
fi

if ! run_arm 4 "0,1"; then
    exit 1
fi

echo "==============================================================================="
echo "Completed both production topologies."
echo "  $OUT_DIR/node_2participants.csv"
echo "  $OUT_DIR/node_4participants.csv"
echo "  $OUT_DIR/node_2participants_nccl.log"
echo "  $OUT_DIR/node_4participants_nccl.log"
echo "  diagnostic CSV/TXT files are topology evidence only, not timings"
echo "  $OUT_DIR/settings.txt"
echo "Draw figure P5 with:"
echo "  uv run --script scripts/plots/ca_participant_latency.py"
echo "==============================================================================="
