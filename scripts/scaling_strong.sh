#!/bin/bash
# Strong-scaling sweep for the KSM-EI OpenMP harness.
#
# Holds the problem fixed and adds cores, so each kernel's ceiling, and where
# (at what core count) it is hit, becomes visible. Runs BOTH grid sizes and
# BOTH init arms so the deliverables are produced from one sweep:
#
#   n = 31  SpMV working set fits inside one 16 MiB L3 slice (cache-resident)
#   n = 61  SpMV working set (~53 MiB) spills to DRAM. MGS still inside one slice
#   arm A   naive master-thread init (matrix parked on thread 0's slice)
#   arm B   parallel first-touch (matrix distributed across engaged slices)
#
# Core-count sweep is CCX-boundary aligned: each step engages exactly one more
# L3 slice under `close` packing (3 physical cores share one CCX / L3 slice).
# No CCX boundary below 12 is skipped, the SpMV knee is expected low and must
# be sampled densely.  All points <= 24 physical cores. SMT lanes are excluded
# (a second bandwidth-bound thread on a core adds contention, not throughput).
#
# Execution controls held fixed for every run (spec section 8):
#   OMP_PLACES=cores       per-physical-core places from the machine topology
#   OMP_PROC_BIND=close    fill one CCX before engaging the next (discrete risers)
#   OMP_NUM_THREADS=P      the swept core count
#
# Assumes the project has already been built: cmake --build build --parallel
set -euo pipefail

WORK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCALING="$WORK_DIR/build/scaling"
DATA_DIR="$WORK_DIR/data/scaling"
CSV="$DATA_DIR/scaling_strong.csv"

# Overridable knobs
STEPS="${STEPS:-100}"      # KSM-EI time steps per solve
M="${M:-8}"                # FIXED Krylov dimension
REPEATS="${REPEATS:-7}"    # timed repeats after 1 warm-up
OPTION="${OPTION:-basket}"

# CCX-boundary aligned core counts (P/3 = CCX slices engaged under `close`)
CORES=(1 3 6 9 12 15 18 21 24)
GRIDS=(31 61)
ARMS=(A B)

if [[ ! -x "$SCALING" ]]; then
    echo "Error: scaling binary not found at $SCALING"
    echo "Build it first: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel"
    exit 1
fi

mkdir -p "$DATA_DIR"
# Fresh CSV each run so the header is written exactly once by the binary.
rm -f "$CSV"

echo "========================================"
echo " KSM-EI strong-scaling sweep"
echo " Binary  : $SCALING"
echo " Output  : $CSV"
echo " steps=$STEPS  m=$M  repeats=$REPEATS  option=$OPTION"
echo "========================================"

for n in "${GRIDS[@]}"; do
    for arm in "${ARMS[@]}"; do
        echo ""
        echo "=== n=$n  arm=$arm ==="
        for P in "${CORES[@]}"; do
            printf "  P=%-2s (CCX=%s) ... " "$P" "$(( (P + 2) / 3 ))"
            OMP_NUM_THREADS="$P" OMP_PLACES=cores OMP_PROC_BIND=close \
                "$SCALING" --arm "$arm" --n "$n" --steps "$STEPS" --m "$M" \
                           --repeats "$REPEATS" --option "$OPTION" --csv "$CSV" \
                > /dev/null
            echo "done"
        done
    done
done

echo ""
row_count=$(( $(wc -l < "$CSV") - 1 ))
echo "Strong-scaling sweep complete: $row_count rows -> $CSV"
echo "Plot with: uv run scripts/scaling_plot.py"
