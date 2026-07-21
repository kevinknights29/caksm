#!/bin/bash
# Weak-scaling sweep for the KSM-EI OpenMP harness.
#
# Work-per-core is held (approximately) constant by growing the problem with the core
# count. Since N ~ n^3, the schedule is n(P) = n1 * P^(1/3) rounded to the nearest odd
# integer (a hard grid constraint), so N/P cannot be held exactly constant. The achieved
# ratio is recorded per point (aug_N and P columns in the CSV) so the reader sees the seam.
#
#   P    n (odd)   role
#   1    31        anchor
#   5    53
#   9    65
#   13   73
#   17   79
#   21   85
#   24   89        endpoint (full physical cores)
#
# Bandwidth-bound weak scaling gives time ~ P*W/B (linear in P), so the honest
# presentation is weak-scaling efficiency T(1)/T(P), a decay from 1.0, produced by
# scripts/plots/scaling_plot.py. SpMV crosses into DRAM around n ~ 39 and GS around n ~ 57
# mid-sweep; those tier crossings are annotated on the plot.
#
# Default arm is B (parallel first-touch): a bandwidth-bound kernel only scales
# if its data is placed on the slices that compute on it.  Set ARM=A to also run
# the naive placement.
#
# Assumes the project has already been built: cmake --build build --parallel
set -euo pipefail

WORK_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SCALING="$WORK_DIR/build/scaling"
DATA_DIR="$WORK_DIR/data/scaling"
CSV="$DATA_DIR/scaling_weak.csv"

STEPS="${STEPS:-100}"
M="${M:-8}"
REPEATS="${REPEATS:-7}"
OPTION="${OPTION:-basket}"
ARM="${ARM:-B}"

# P and the odd n that holds N/P ~ constant.
P_LIST=(1  5  9  13 17 21 24)
N_LIST=(31 53 65 73 79 85 89)

if [[ ! -x "$SCALING" ]]; then
    echo "Error: scaling binary not found at $SCALING"
    echo "Build it first: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel"
    exit 1
fi

mkdir -p "$DATA_DIR"
rm -f "$CSV"

echo "========================================"
echo " KSM-EI weak-scaling sweep"
echo " Binary  : $SCALING"
echo " Output  : $CSV"
echo " arm=$ARM  steps=$STEPS  m=$M  repeats=$REPEATS  option=$OPTION"
echo "========================================"
echo ""
printf "  %-4s %-6s %-6s\n" "P" "n" "CCX"

for i in "${!P_LIST[@]}"; do
    P="${P_LIST[$i]}"
    n="${N_LIST[$i]}"
    printf "  %-4s %-6s %-6s ... " "$P" "$n" "$(( (P + 2) / 3 ))"
    OMP_NUM_THREADS="$P" OMP_PLACES=cores OMP_PROC_BIND=close \
        "$SCALING" --arm "$ARM" --n "$n" --steps "$STEPS" --m "$M" \
                   --repeats "$REPEATS" --option "$OPTION" --csv "$CSV" \
        > /dev/null
    echo "done"
done

echo ""
row_count=$(( $(wc -l < "$CSV") - 1 ))
echo "Weak-scaling sweep complete: $row_count rows -> $CSV"
echo "Plot with: uv run scripts/plots/scaling_plot.py"
