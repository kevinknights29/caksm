#!/bin/bash
# Timed baseline SpMV across a grid sweep: is the kernel DRAM-bound where R_v > 1?
#
# The achieved rate is reported against the DRAM roof measured by gpu-stream, so pass the same
# value recorded in the preset. Below R_v = 1 the rate should sit near the L2 roof and above it
# near the DRAM roof. Only the second regime gives the vertical mechanism traffic to convert.
#
# Assumes the project has been built with a CUDA toolkit: cmake --build build --parallel
set -uo pipefail

WORK_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SPMV="$WORK_DIR/build/regime-gpu-spmv"
DATA_DIR="$WORK_DIR/data/regime"

MACHINE="${MACHINE:-v100-pcie-16gb}"
DEVICE="${DEVICE:-0}"
M="${M:-12}"
DIM="${DIM:-3}"
REPEATS="${REPEATS:-7}"
N1_LIST="${N1_LIST:-31 45 61 74 89 120 160 200 240 280 320}"

# The DRAM roof to compare against, from gpu-stream: 818 GB/s on the V100, 821 on the 3090.
case "$MACHINE" in
    v100-pcie-16gb) DRAM_GBS="${DRAM_GBS:-818.3}" ;;
    rtx-3090)       DRAM_GBS="${DRAM_GBS:-821.1}" ;;
    *)              DRAM_GBS="${DRAM_GBS:-0}" ;;
esac

if [[ ! -x "$SPMV" ]]; then
    echo "Error: regime-gpu-spmv not found at $SPMV"
    echo "Build it with a CUDA toolkit: cmake --build build --parallel"
    exit 1
fi

mkdir -p "$DATA_DIR"

"$SPMV" --machine "$MACHINE" --device "$DEVICE" --m "$M" --dim "$DIM" \
        --repeats "$REPEATS" --n1-list "$N1_LIST" --dram-gbs "$DRAM_GBS" \
        --csv "$DATA_DIR/regime_gpu_spmv_${MACHINE}.csv"
