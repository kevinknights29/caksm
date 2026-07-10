#!/bin/bash
# m(n) Krylov-growth sweep
#
# The scaling study FREEZES the Krylov dimension at m=8 so the kernel mix does
# not drift.  That quarantines, but does not hide, the fact that the true
# per-unknown Krylov cost m(n) GROWS under grid refinement: as h ~ 1/n shrinks,
# the spectrum of the discretized generator widens, so exp(A~)v needs more
# Arnoldi steps to hold tolerance. This sweep measures that growth with the
# genuine convergence-driven KSM-EI (via the profiler's avg_krylov), and reports
# it as an independent numerical-analysis result that itself motivates CA: more
# Arnoldi steps under refinement means more global reductions.
#
# Runs on the SAME grid sizes the weak sweep visits, so the two curves overlay.
#
# Assumes the project has already been built: cmake --build build --parallel
set -euo pipefail

WORK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROFILER="$WORK_DIR/build/profiler"
DATA_DIR="$WORK_DIR/data/scaling"
CSV="$DATA_DIR/mn_growth.csv"

HW="${HW:-amd-3960x}"
TOL="${TOL:-1e-8}"
OPTION="${OPTION:-basket}"

# The odd grid sizes from the weak sweep table
N_LIST=(31 53 65 73 79 85 89)

if [[ ! -x "$PROFILER" ]]; then
    echo "Error: profiler not found at $PROFILER"
    echo "Build it first: cmake --build build --parallel"
    exit 1
fi

mkdir -p "$DATA_DIR"
rm -f "$CSV"

echo "========================================"
echo " KSM-EI m(n) Krylov-growth sweep"
echo " Binary : $PROFILER   hw=$HW  tol=$TOL  option=$OPTION"
echo " Output : $CSV"
echo "========================================"

for n in "${N_LIST[@]}"; do
    printf "  n=%-4s ... " "$n"
    # Single-threaded: m(n) is a numerical property, independent of core count.
    OMP_NUM_THREADS=1 "$PROFILER" --hw "$HW" --n "$n" --tol "$TOL" \
        --option "$OPTION" --csv "$CSV" > /dev/null
    m=$(tail -n 1 "$CSV" | cut -d, -f6)
    echo "avg m = $m"
done

echo ""
echo "m(n) sweep complete -> $CSV"
echo "Plot with: uv run scripts/scaling_plot.py"
