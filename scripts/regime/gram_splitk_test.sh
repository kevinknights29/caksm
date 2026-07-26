#!/bin/bash
# Validate and time the split-K Gram kernel against cuBLAS.
#
# Correctness compares split-K G = B^T B to a cuBLAS gemm.
# Rate compares split-K to cuBLAS dsyrk on a tall block, against the card's measured FP64
# peak and DRAM roof. The negative-arm reading is whether split-K reaches the ~s/4 roofline
# (3090 compute-bound near its FP64 peak, V100 memory-bound near bandwidth) where cuBLAS
# dsyrk stays latency-bound.
#
# Assumes the project has been built with a CUDA toolkit: cmake --build build --parallel
set -uo pipefail

WORK_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$WORK_DIR/build/gram-splitk-test"
DATA_DIR="$WORK_DIR/data/regime"

MACHINE="${MACHINE:-v100-pcie-16gb}"
DEVICE="${DEVICE:-0}"
N_RATE="${N_RATE:-1000000}"
N_CHECK="${N_CHECK:-4096}"
S_LIST="${S_LIST:-1 2 4 6 8}"
REPEATS="${REPEATS:-7}"
BLOCKS_PER_SM="${BLOCKS_PER_SM:-4}"

case "$MACHINE" in
    v100-pcie-16gb) FP64="${FP64:-6.375}"; DRAM_GBS="${DRAM_GBS:-818.3}" ;;
    rtx-3090)       FP64="${FP64:-0.570}"; DRAM_GBS="${DRAM_GBS:-821.1}" ;;
    *)              FP64="${FP64:-0}";     DRAM_GBS="${DRAM_GBS:-0}" ;;
esac

if [[ ! -x "$BIN" ]]; then
    echo "Error: gram-splitk-test not found at $BIN"
    echo "Build it with a CUDA toolkit: cmake --build build --parallel"
    exit 1
fi

mkdir -p "$DATA_DIR"

"$BIN" --machine "$MACHINE" --device "$DEVICE" --n-rate "$N_RATE" --n-check "$N_CHECK" \
       --s-list "$S_LIST" --repeats "$REPEATS" --blocks-per-sm "$BLOCKS_PER_SM" \
       --fp64-tflops "$FP64" --dram-gbs "$DRAM_GBS" \
       --csv "$DATA_DIR/gram_splitk_${MACHINE}.csv"
