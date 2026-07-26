#!/bin/bash
# Two-GPU measured horizontal crossover: MGS against s-step with real inter-GPU all-reduces.
#
# Splits the n basis rows across both local GPUs, so every MGS dot/norm and every s-step Gram is
# a real NCCL all-reduce at the DEVICE_P2P rung. Sweeps n to see where s-step stops beating MGS is
# (the measured theta_h), checked against the predicted R_h at the calibrated rung (V100 SYS rung,
# 11.46 us). Only synge has two local GPUs, so this is a V100 instrument.
#
# Assumes the project has been built with a CUDA toolkit and NCCL:
#   cmake --build build --parallel --target regime-gpu-sstep-2gpu
set -uo pipefail

WORK_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SSTEP2="$WORK_DIR/build/regime-gpu-sstep-2gpu"
DATA_DIR="$WORK_DIR/data/regime"

MACHINE="${MACHINE:-v100-pcie-16gb}"
M="${M:-12}"
REPEATS="${REPEATS:-7}"
# Spans the launch-bound floor (flat mgs_local, ~8k-227k) down to where s-step reaches parity
# (below ~8k) and up into the bandwidth-bound rise (>=705k), to bracket the crossover.
N_LIST="${N_LIST:-500 1000 2000 4000 8000 16000 32000 61000 125000 227000 705000 1728000}"
S_LIST="${S_LIST:-1 2 4 6 8}"
S_MAX="${S_MAX:-9}"

case "$MACHINE" in
    v100-pcie-16gb) T_REDUCE_US="${T_REDUCE_US:-11.46}" ;;
    *)              T_REDUCE_US="${T_REDUCE_US:-0}" ;;
esac

if [[ ! -x "$SSTEP2" ]]; then
    echo "Error: regime-gpu-sstep-2gpu not found at $SSTEP2"
    echo "Build it with a CUDA toolkit and NCCL: cmake --build build --parallel"
    echo "(the target only configures when NCCL is found; see CMakeLists.txt)"
    exit 1
fi

mkdir -p "$DATA_DIR"

"$SSTEP2" --machine "$MACHINE" --m "$M" --repeats "$REPEATS" \
          --n-list "$N_LIST" --s-list "$S_LIST" --s-max "$S_MAX" \
          --t-reduce-us "$T_REDUCE_US" \
          --csv "$DATA_DIR/regime_gpu_sstep_2gpu_${MACHINE}.csv"
