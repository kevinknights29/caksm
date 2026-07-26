#!/bin/bash
# Timed s-step orthogonalization against MGS: the horizontal crossover and the negative arm.
#
# Sweeps the block width s at a few operator sizes. Reports the Gram block's FP64 rate against
# each card's measured peak, and the reduction cost at which s-step overtakes MGS. Pass the
# measured FP64 peak and a reduction cost from the calibrated ladder: the V100's SYS rung
# (11.46 us), the 3090's on-device grid rung (11.59 us).
#
# Assumes the project has been built with a CUDA toolkit: cmake --build build --parallel
set -uo pipefail

WORK_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SSTEP="$WORK_DIR/build/regime-gpu-sstep"
DATA_DIR="$WORK_DIR/data/regime"

MACHINE="${MACHINE:-v100-pcie-16gb}"
DEVICE="${DEVICE:-0}"
M="${M:-12}"
REPEATS="${REPEATS:-7}"
N_LIST="${N_LIST:-61000 227000 705000}"
S_LIST="${S_LIST:-1 2 4 6 8}"
S_MAX="${S_MAX:-9}"              # certified block-width cap, s beyond it loses orthogonality
GRAM_N="${GRAM_N:-1000000}"      # rows for the split-K Gram rate, spills L2 for a clean roofline read

case "$MACHINE" in
    v100-pcie-16gb) FP64="${FP64:-6.375}"; DRAM_GBS="${DRAM_GBS:-818.3}"; T_REDUCE_US="${T_REDUCE_US:-11.46}" ;;
    rtx-3090)       FP64="${FP64:-0.570}"; DRAM_GBS="${DRAM_GBS:-821.1}"; T_REDUCE_US="${T_REDUCE_US:-11.59}" ;;
    *)              FP64="${FP64:-0}";     DRAM_GBS="${DRAM_GBS:-0}";     T_REDUCE_US="${T_REDUCE_US:-0}" ;;
esac

if [[ ! -x "$SSTEP" ]]; then
    echo "Error: regime-gpu-sstep not found at $SSTEP"
    echo "Build it with a CUDA toolkit: cmake --build build --parallel"
    exit 1
fi

mkdir -p "$DATA_DIR"

"$SSTEP" --machine "$MACHINE" --device "$DEVICE" --m "$M" --repeats "$REPEATS" \
         --n-list "$N_LIST" --s-list "$S_LIST" --s-max "$S_MAX" --gram-n "$GRAM_N" \
         --fp64-tflops "$FP64" --dram-gbs "$DRAM_GBS" --t-reduce-us "$T_REDUCE_US" \
         --csv "$DATA_DIR/regime_gpu_sstep_${MACHINE}.csv"
