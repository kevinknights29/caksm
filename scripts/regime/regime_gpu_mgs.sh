#!/bin/bash
# Timed MGS across a basis-size sweep: the reduction-bound half of one Arnoldi cycle.
#
# Reports the achieved bandwidth of the basis re-reads and the compute between reductions, and
# the ratio of a reduction cost to that compute, which should track the predicted R_h. Each
# machine uses the cost of one reduction at its most expensive reachable tier: the V100's SYS
# inter-GPU rung (11.46 us), the 3090's on-device grid rung (11.59 us).
#
# Assumes the project has been built with a CUDA toolkit: cmake --build build --parallel
set -uo pipefail

WORK_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
MGS="$WORK_DIR/build/regime-gpu-mgs"
DATA_DIR="$WORK_DIR/data/regime"

MACHINE="${MACHINE:-v100-pcie-16gb}"
DEVICE="${DEVICE:-0}"
M="${M:-12}"
REPEATS="${REPEATS:-7}"
N_LIST="${N_LIST:-1000 8000 61000 227000 705000 1728000 4096000 8000000 13824000}"

case "$MACHINE" in
    v100-pcie-16gb) DRAM_GBS="${DRAM_GBS:-818.3}"; T_REDUCE_US="${T_REDUCE_US:-11.46}" ;;
    rtx-3090)       DRAM_GBS="${DRAM_GBS:-821.1}"; T_REDUCE_US="${T_REDUCE_US:-11.59}" ;;
    *)              DRAM_GBS="${DRAM_GBS:-0}";     T_REDUCE_US="${T_REDUCE_US:-0}" ;;
esac

if [[ ! -x "$MGS" ]]; then
    echo "Error: regime-gpu-mgs not found at $MGS"
    echo "Build it with a CUDA toolkit: cmake --build build --parallel"
    exit 1
fi

mkdir -p "$DATA_DIR"

"$MGS" --machine "$MACHINE" --device "$DEVICE" --m "$M" --repeats "$REPEATS" \
       --n-list "$N_LIST" --dram-gbs "$DRAM_GBS" --t-reduce-us "$T_REDUCE_US" \
       --csv "$DATA_DIR/regime_gpu_mgs_${MACHINE}.csv"
