#!/bin/bash
# Measured horizontal crossover over N local GPUs: MGS against s-step with real inter-GPU
# all-reduces.
#
# Splits the n basis rows across the participating GPUs, so every MGS dot/norm and every s-step
# Gram is a real NCCL all-reduce at the DEVICE_P2P rung. Sweeps n to find where s-step stops
# beating MGS (the measured theta_h), checked against the predicted R_h at the calibrated rung.
#
# DEVICES selects the participants, so the participant count is sweepable on one node. Set
# T_REDUCE_US to the calibrated cost at THAT count, not at another: the collective grows more
# expensive with participants, and gpu_topology.hpp keys it per count for that reason.
#
#   DEVICES=0,1     T_REDUCE_US=10.51 ./scripts/regime/regime_gpu_sstep_multigpu.sh
#   DEVICES=0,1,2,3 T_REDUCE_US=20.63 ./scripts/regime/regime_gpu_sstep_multigpu.sh
#
# Assumes the project has been built with a CUDA toolkit and NCCL:
#   cmake --build build --parallel --target regime-gpu-sstep-multigpu
set -uo pipefail

WORK_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SSTEP="$WORK_DIR/build/regime-gpu-sstep-multigpu"
DATA_DIR="$WORK_DIR/data/regime"

MACHINE="${MACHINE:-v100-pcie-16gb}"
M="${M:-12}"
REPEATS="${REPEATS:-7}"
# Spans the launch-bound floor (flat mgs_local, ~8k-227k) down to where s-step reaches parity
# (below ~8k) and up into the bandwidth-bound rise (>=705k), to bracket the crossover.
N_LIST="${N_LIST:-500 1000 2000 4000 8000 16000 32000 61000 125000 227000 705000 1728000}"
S_LIST="${S_LIST:-1 2 4 6 8}"
S_MAX="${S_MAX:-9}"
# Which visible GPUs take part. Empty uses every one the allocation exposes.
DEVICES="${DEVICES:-}"

case "$MACHINE" in
    v100-pcie-16gb) T_REDUCE_US="${T_REDUCE_US:-11.46}" ;;
    *)              T_REDUCE_US="${T_REDUCE_US:-0}" ;;
esac

if [[ ! -x "$SSTEP" ]]; then
    echo "Error: regime-gpu-sstep-multigpu not found at $SSTEP"
    echo "Build it with a CUDA toolkit and NCCL: cmake --build build --parallel"
    echo "(the target only configures when NCCL is found; see CMakeLists.txt)"
    exit 1
fi

mkdir -p "$DATA_DIR"

# The participant count goes in the file name, so two counts on one machine do not overwrite
# each other and a row can always be traced back to the arrangement that produced it.
STEM="${DEVICES:+_dev$(echo "$DEVICES" | tr ',' '-')}"

"$SSTEP" --machine "$MACHINE" --m "$M" --repeats "$REPEATS" \
         ${DEVICES:+--devices "$DEVICES"} \
         --n-list "$N_LIST" --s-list "$S_LIST" --s-max "$S_MAX" \
         --t-reduce-us "$T_REDUCE_US" \
         --csv "$DATA_DIR/regime_gpu_sstep_multigpu_${MACHINE}${STEM}.csv"
