#!/bin/bash
# Regenerate the GPU map's a-priori placements and the Phase 0 tables.
#
# Pure prediction: regime-gpu-place links neither a solver nor a timer and never calls CUDA,
# so this runs on a login node or a laptop. The predictions are published before the hardware
# is touched, which is what makes the eventual measurement a test rather than a fit.
#
# Two modes:
#
#   uncalibrated  The preset's tiers are zero, so R_h would be zero everywhere. The ASSUME_*
#                 priors below fill them in, and every figure they touch is stamped ASSUMED in
#                 both the console output and the CSV. These are the Phase 0 tables: order-of-
#                 magnitude predictions to be falsified, never inputs to a verdict.
#   calibrated    Once scripts/regime/calibrate_gpu.sh and calibrate_gpu_p2p.sh have landed and
#                 the preset is edited, the priors are ignored for any rung that carries a
#                 measured constant, and the ASSUMED markers disappear on their own.
#
# The priors are stated in the open, in one place, so a guess cannot later be mistaken for a
# measurement. Tuning constants until a crossover landed at 1 would prove nothing.
#
# Assumes the project has been built: cmake --build build --parallel
# No CUDA toolkit needed; regime-gpu-place is host-only by design.
set -uo pipefail

WORK_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PLACE="$WORK_DIR/build/regime-gpu-place"
DATA_DIR="$WORK_DIR/data/regime"

MACHINE="${MACHINE:-v100-pcie-16gb}"
M="${M:-12}"            # production Krylov dimension at n=61
DIM="${DIM:-3}"
S="${S:-8}"             # certified block width; the Gram gate reads it, the coordinates do not
N1_LIST="${N1_LIST:-31 45 61 74 89}"

# Stated priors, used only where the preset is uncalibrated. Cumulative microseconds.
# Sources: kernel-launch and grid-combine costs are the usual few-microsecond figures for a
# CUDA launch; the SYS rung is a PCIe + cross-socket hop; the node rung assumes a mid-range
# fabric. All three are the things calibrate_gpu*.sh exists to replace.
ASSUME_GRID_US="${ASSUME_GRID_US:-5}"
ASSUME_P2P_US="${ASSUME_P2P_US:-10}"
ASSUME_NODE_US="${ASSUME_NODE_US:-25}"
ASSUME_BW_GBS="${ASSUME_BW_GBS:-800}"    # ~89% of the V100's 900 GB/s theoretical

if [[ ! -x "$PLACE" ]]; then
    echo "Error: regime-gpu-place not found at $PLACE"
    echo "Build it: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel"
    exit 1
fi

mkdir -p "$DATA_DIR"

echo "==============================================================================="
echo " GPU regime placement   machine=$MACHINE  m=$M  dim=$DIM  s=$S"
echo "==============================================================================="
echo

"$PLACE" --machine "$MACHINE" --m "$M" --dim "$DIM" --s "$S" \
         --n1-list "$N1_LIST" \
         --assume-grid-us "$ASSUME_GRID_US" \
         --assume-p2p-us "$ASSUME_P2P_US" \
         --assume-node-us "$ASSUME_NODE_US" \
         --assume-bw-gbs "$ASSUME_BW_GBS" \
         --csv "$DATA_DIR/regime_gpu_${MACHINE}.csv"
STATUS=$?
echo

# The negative arm. The 3090 holds L2 and bandwidth nearly fixed and drops FP64 by ~12x, so
# the pair varies only the compute roof. Phase 0 showed the discriminating kernel is not SpMV,
# since both cards are on-map for the baseline, but the tall-skinny Gram matrix that
# communication-avoiding Arnoldi introduces and MGS does not have.
if [[ "$MACHINE" == "v100-pcie-16gb" ]]; then
    echo "==============================================================================="
    echo " Negative arm: the same operator on puffin's RTX 3090"
    echo "==============================================================================="
    echo
    "$PLACE" --machine rtx-3090 --m "$M" --dim "$DIM" --s "$S" \
             --n1-list "$N1_LIST" \
             --assume-grid-us "$ASSUME_GRID_US" \
             --assume-bw-gbs "$ASSUME_BW_GBS" \
             --csv "$DATA_DIR/regime_gpu_rtx-3090.csv"
    echo
fi

echo "==============================================================================="
if [[ $STATUS -ne 0 ]]; then
    echo "FAILED ($STATUS)."
    exit 1
fi
echo "Done."
echo "  data/regime/regime_gpu_${MACHINE}.csv   <- the placed points, with gate verdicts"
echo
echo "Reading it:"
echo "  - Any row with assumed=1 is a prediction from the priors above, not a verdict."
echo "  - A row with on_map=0 is off-map, not lower-left: its corner label is meaningless,"
echo "    because neither coordinate charts a compute-bound kernel."
echo "  - treatment_on_map=0 with on_map=1 is the sharpened negative arm: the map charts the"
echo "    baseline there, but the kernel s-step adds is compute-bound, so the advertised win"
echo "    is eaten by a roof neither coordinate can see."
echo
echo "The derivation behind the invariant and window tables is docs/regime_gpu_phase0.md."
echo "==============================================================================="
