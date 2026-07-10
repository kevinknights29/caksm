#!/bin/bash
# Locality experiment: what actually causes the n=61 strong-scaling transition?
#
# The naive-init (arm A) vs first-touch (arm B) contrast varies NOTHING on this
# machine: the 3960X is single-NUMA, and Zen 2 L3 is a per-CCX victim cache, so
# L3 residency is set by which core ACCESSES a row (the compute partition), not
# by which thread first-touched the page. The real A/B is therefore over the
# COMPUTE SCHEDULE, not the init:
#
#   block   contiguous band per core, fixed across SpMVs (locality-preserving
#           control).  Each CCX re-reads the same ~3*ws/P MiB every SpMV and
#           becomes L3-resident once that fits 16 MiB (P >= 12).  Expect the
#           superlinear transition at P=12.
#
#   cyclic  cache-line blocks dealt round-robin, fixed across SpMVs.  Rows are
#           scattered across the whole matrix, but each CCX's VOLUME is still
#           3*ws/P MiB. Volume, not contiguity, sets L3 residency. Expect
#           this to STILL transition (~ block): scatter alone changes nothing.
#           This is the control that shows the mechanism is capacity, not layout.
#
#   rotate  contiguous bands, but the band a core computes ROTATES by one every
#           SpMV. Over the reuse window each CCX sweeps ALL bands, so its 16 MiB
#           slice can never retain a stable cache-fitting subset: the per-CCX
#           footprint becomes the WHOLE matrix and every SpMV pays DRAM. If the
#           per-CCX-capacity mechanism is correct, the transition MUST degrate.
#
# Run at n=61 (the size whose SpMV working set, 52.8 MiB, spills DRAM->L3 as
# slices are engaged), arm B, across the CCX-aligned core counts.
#
# Assumes the project has already been built: cmake --build build --parallel
set -euo pipefail

WORK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCALING="$WORK_DIR/build/scaling"
DATA_DIR="$WORK_DIR/data/scaling"
CSV="$DATA_DIR/scaling_locality.csv"

N="${N:-61}"
STEPS="${STEPS:-100}"
M="${M:-8}"
REPEATS="${REPEATS:-7}"
OPTION="${OPTION:-basket}"
ARM="${ARM:-B}"

CORES=(1 3 6 9 12 15 18 21 24)
SCHEDS=(block cyclic rotate)

if [[ ! -x "$SCALING" ]]; then
    echo "Error: scaling binary not found at $SCALING"
    echo "Build it first: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel"
    exit 1
fi

mkdir -p "$DATA_DIR"
rm -f "$CSV"

echo "========================================"
echo " KSM-EI locality experiment (compute-schedule A/B)"
echo " Binary  : $SCALING"
echo " Output  : $CSV"
echo " n=$N  arm=$ARM  steps=$STEPS  m=$M  repeats=$REPEATS  option=$OPTION"
echo "========================================"

for sched in "${SCHEDS[@]}"; do
    echo ""
    echo "=== sched=$sched ==="
    for P in "${CORES[@]}"; do
        printf "  P=%-2s (CCX=%s) ... " "$P" "$(( (P + 2) / 3 ))"
        OMP_NUM_THREADS="$P" OMP_PLACES=cores OMP_PROC_BIND=close \
            "$SCALING" --arm "$ARM" --sched "$sched" --n "$N" --steps "$STEPS" \
                       --m "$M" --repeats "$REPEATS" --option "$OPTION" --csv "$CSV" \
            > /dev/null
        echo "done"
    done
done

echo ""
row_count=$(( $(wc -l < "$CSV") - 1 ))
echo "Locality experiment complete: $row_count rows -> $CSV"
echo "Plot with: uv run scripts/scaling_plot.py"
