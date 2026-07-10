#!/bin/bash
# Thread-count sweep for STREAM Triad bandwidth.
# Hardware: AMD Ryzen Threadripper 3960X (24 cores, 48 threads, AVX2, quad-channel DDR4).
# Assumes the project has already been built: cmake --build build --parallel
# Reports the saturation knee, the thread count where Triad bandwidth plateaus.
set -euo pipefail

WORK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STREAM="$WORK_DIR/build/stream"

if [[ ! -x "$STREAM" ]]; then
    echo "Error: stream not found at $STREAM"
    echo "Build it first: cmake --build build --parallel"
    exit 1
fi

echo "========================================"
echo " STREAM Triad thread-count sweep"
echo " Binary : $STREAM"
echo "========================================"
echo ""
printf "%-12s  %s\n" "Threads" "Triad (MB/s)"
echo "--------------------------------"

# Sweep covers: single-threaded baseline, doubling up to physical cores (24),
# then full SMT (48). Bandwidth typically saturates well before 24 threads on a
# quad-channel system. The knee is the point of interest.
for t in 1 2 4 8 12 16 24 48; do
    result=$(OMP_NUM_THREADS=$t OMP_PROC_BIND=close OMP_PLACES=cores "$STREAM" 2>&1 \
        | grep "^Triad" \
        | awk '{print $2}')
    printf "%-12s  %s\n" "$t" "${result:-N/A}"
done

echo ""
echo "Plateau of Triad column = aggregate memory bandwidth roof."
echo "Knee = thread count beyond which bandwidth no longer grows."
