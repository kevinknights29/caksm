#!/bin/bash
# Calibrate one GPU: the on-device reduction ladder and the achieved memory roofs.
#
# A prerequisite, not an experiment, and the counterpart of calibrate_alpha.sh. Until it lands,
# R_h's numerator has a shape in the tier but no magnitude, theta_h has no place on any figure,
# and the roofline gate has no denominator, so every placed point is provisional.
#
# Three binaries, all offline, none linked against the solver:
#   calibrate-gpu-reduction   warp / block / grid rungs, and kernel-launch latency as its own
#                             term. Launch is separated deliberately: if it swamps every rung
#                             above the block tier, R_h's ladder collapses to a constant and
#                             the horizontal mechanism becomes uninteresting. That is a result,
#                             and it is invisible if launch is folded into the tier.
#   gpu-stream                achieved HBM and L2 roofs, plus the occupancy sweep the CPU
#                             study has no counterpart for.
#   gpu-fma-loop              measured FP64 and FP32 peak. The roofline gate divides by the
#                             FP64 figure and the study's negative arm is entirely a claim
#                             about it, so it is measured rather than transcribed, the same
#                             rule that makes the CPU's peak_gflops_core a measurement.
#
# All three run on one device. Nothing here needs NCCL, MPI, a launcher or a second GPU, so
# this script is the whole of what a single-GPU host can contribute, which on puffin's RTX 3090
# is the negative arm, complete.
#
# These are properties of one device and do not transfer. Run on the host you intend to place,
# and record against that host's preset only.
#
# The interconnect rungs are a separate script: scripts/regime/calibrate_gpu_p2p.sh.
#
# Assumes the project has already been built with a CUDA toolkit visible:
#   cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel
set -uo pipefail

WORK_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$WORK_DIR/build}"
RED="$BUILD_DIR/calibrate-gpu-reduction"
STREAM="$BUILD_DIR/gpu-stream"
FMA="$BUILD_DIR/gpu-fma-loop"
# Overridable so independent invocations can be kept apart. The replicate policy needs
# several runs of this script on one host, and a fixed path would have each overwrite the
# last: see the h200 preset comment in include/gpu_machine.hpp.
DATA_DIR="${DATA_DIR:-$WORK_DIR/data/regime}"

MACHINE="${MACHINE:-v100-pcie-16gb}"
DEVICE="${DEVICE:-0}"
ITERS="${ITERS:-20000}"
REPEATS="${REPEATS:-7}"
BLOCK_THREADS="${BLOCK_THREADS:-256}"
MAX_MIB="${MAX_MIB:-2048}"

# Wait for the device to go idle before measuring. On a shared workstation the recordable run
# is gated on someone else's job finishing, and the alternative is polling nvidia-smi by hand
# and hoping to catch the gap. The binaries refuse to emit recordable constants from a
# contended device anyway (include/gpu_contention.cuh), so waiting here is what turns a
# smoke test into a calibration.
#
#   WAIT_FOR_IDLE=7200 ./scripts/regime/calibrate_gpu.sh    # wait up to 2 h, then run
#
# 0 (the default) runs immediately, which is what you want for a smoke test.
WAIT_FOR_IDLE="${WAIT_FOR_IDLE:-0}"      # seconds; 0 = do not wait
IDLE_MIB="${IDLE_MIB:-256}"              # device counts as idle below this many MiB in use
POLL_S="${POLL_S:-60}"

for bin in "$RED" "$STREAM" "$FMA"; do
    if [[ ! -x "$bin" ]]; then
        echo "Error: $(basename "$bin") not found at $bin"
        echo "Build with a CUDA toolkit, pinning the host compiler via the environment:"
        echo "  CUDAHOSTCXX=/usr/bin/g++ cmake -B build -DCMAKE_BUILD_TYPE=Release \\"
        echo "        -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \\"
        echo "        -DCMAKE_CUDA_ARCHITECTURES=<86 for RTX 3090, 70 for V100>"
        echo "  cmake --build build --parallel"
        echo
        echo "Use CUDAHOSTCXX, not -DCMAKE_CUDA_HOST_COMPILER: CMake's compiler-identification"
        echo "step runs before that cache variable exists and calls bare 'gcc' from PATH, so"
        echo "nvcc rejects a too-new default compiler no matter what the -D flag says."
        exit 1
    fi
done

mkdir -p "$DATA_DIR"

echo "==============================================================================="
echo " GPU calibration   machine=$MACHINE  device=$DEVICE  host=$(uname -n)"
echo "==============================================================================="
echo
echo "Provenance (record this alongside the numbers):"
echo "  host   : $(uname -n)  $(uname -srm)"
echo "  job    : ${SLURM_JOB_ID:-<not under the scheduler>}"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,compute_cap,memory.total,clocks.max.sm,persistence_mode,compute_mode \
               --format=csv,noheader | sed 's/^/  gpu    : /'
    # Clocks move latencies, which is the whole point of the measurement, so they are
    # recorded. clocks.applications.* is deprecated on recent drivers and returns an error
    # string; clocks.sm/clocks.mem are the live equivalents.
    nvidia-smi --query-gpu=index,clocks.sm,clocks.mem --format=csv,noheader 2>/dev/null \
        | sed 's/^/  clocks : /'
fi
# The nvcc on PATH is not necessarily the one that built these binaries: puffin carries seven
# toolkits and the build is pinned with -DCMAKE_CUDA_COMPILER. Each binary prints its own
# compiled-in __CUDACC_VER_*, which is the figure to trust and to record.
if command -v nvcc >/dev/null 2>&1; then
    echo "  nvcc   : $(nvcc --version | tail -1)   <- on PATH; see each binary's own"
    echo "           'built with nvcc X.Y' line for the toolkit that actually compiled it"
fi
echo

if [[ "$WAIT_FOR_IDLE" -gt 0 ]] && command -v nvidia-smi >/dev/null 2>&1; then
    echo "Waiting up to ${WAIT_FOR_IDLE}s for device $DEVICE to fall below ${IDLE_MIB} MiB in use"
    echo "(polling every ${POLL_S}s; Ctrl-C to give up and run contended as a smoke test)"
    deadline=$(( SECONDS + WAIT_FOR_IDLE ))
    while (( SECONDS < deadline )); do
        used=$(nvidia-smi --id="$DEVICE" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 999999)
        if [[ "$used" =~ ^[0-9]+$ ]] && (( used < IDLE_MIB )); then
            echo "  device idle (${used} MiB in use) after $(( SECONDS ))s: measuring now"
            break
        fi
        echo "  $(date +%H:%M:%S)  ${used} MiB in use, still contended; $(( deadline - SECONDS ))s left"
        sleep "$POLL_S"
    done
    if (( SECONDS >= deadline )); then
        echo "  Timed out, still contended. Running anyway: the binaries suppress their"
        echo "  recording blocks, so this is a smoke test and nothing can be transcribed."
    fi
    echo
fi

echo "### 1/3  Compute roof: measured FP64 and FP32 peak"
echo
"$FMA" --machine "$MACHINE" --device "$DEVICE" \
       --iters "${FMA_ITERS:-4000}" --repeats "$REPEATS" \
       --csv "$DATA_DIR/gpu_fma_loop.csv"
FMA_STATUS=$?
echo

echo "### 2/3  Reduction ladder (warp, block, grid) and kernel-launch latency"
echo
"$RED" --machine "$MACHINE" --device "$DEVICE" \
       --iters "$ITERS" --repeats "$REPEATS" --block-threads "$BLOCK_THREADS" \
       --csv "$DATA_DIR/calibrate_gpu_reduction.csv"
RED_STATUS=$?
echo

echo "### 3/3  Achieved memory roofs and the occupancy sweep"
echo
"$STREAM" --machine "$MACHINE" --device "$DEVICE" \
          --repeats "$REPEATS" --max-mib "$MAX_MIB" \
          --csv "$DATA_DIR/gpu_stream.csv" \
          --occupancy-csv "$DATA_DIR/gpu_stream_occupancy.csv"
STREAM_STATUS=$?
echo

echo "==============================================================================="
if [[ $FMA_STATUS -ne 0 || $RED_STATUS -ne 0 || $STREAM_STATUS -ne 0 ]]; then
    echo "FAILED (fma=$FMA_STATUS reduction=$RED_STATUS stream=$STREAM_STATUS)."
    echo "Do not record partial results: a half-calibrated preset is worse than an"
    echo "uncalibrated one, because the flags would be set on constants never measured."
    exit 1
fi
echo "Done."
echo "  $DATA_DIR/gpu_fma_loop.csv             <- the compute roof, and the FP64:FP32 ratio"
echo "  $DATA_DIR/calibrate_gpu_reduction.csv  <- the on-device ladder"
echo "  $DATA_DIR/gpu_stream.csv               <- the size sweep, and the two memory roofs"
echo "  $DATA_DIR/gpu_stream_occupancy.csv     <- is the kernel occupancy-bound?"
echo
echo "Next:"
echo "  1. Transcribe the printed constants into the '$MACHINE' preset in"
echo "     include/gpu_machine.hpp. No binary writes a preset: a calibration is always a"
echo "     deliberate, reviewable edit, exactly as on the CPU side."
echo "  2. Set tier_calibrated for WARP, BLOCK and GRID, and roofline_gated = true."
echo "     Leave reduction_calibrated false unless every reachable rung is measured. On a"
echo "     single-GPU host such as puffin, warp/block/grid are the only reachable rungs, so"
echo "     it may legitimately go true there while staying false on synge until"
echo "     calibrate_gpu_p2p.sh has landed the interconnect."
echo "  3. Re-run scripts/regime/regime_gpu_place.sh; the Phase 0 tables regenerate from"
echo "     the measured constants and the ASSUMED markers disappear."
echo "==============================================================================="
