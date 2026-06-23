#!/bin/bash
# Parameter sweep for n=61.
# Assumes the project has already been built: cmake --build build --parallel
set -euo pipefail

GRID_N=61
WORK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PRICER="$WORK_DIR/build/pricer"
DATA_DIR="$WORK_DIR/data/n${GRID_N}"

if [[ ! -x "$PRICER" ]]; then
    echo "Error: pricer not found at $PRICER"
    echo "Build it first: cmake --build build --parallel"
    exit 1
fi

mkdir -p "$DATA_DIR"
echo "Pricer          : $PRICER"
echo "Data directory  : $DATA_DIR"

# Run pricer then immediately move the new export file into DATA_DIR.
run_and_collect() {
    "$PRICER" "$@" --export
    mv "$WORK_DIR"/caksm_export_n${GRID_N}_*.csv "$DATA_DIR/"
}

echo ""
echo "========================================"
echo " Parameter sweep  n=$GRID_N"
echo "========================================"

# Pre-compute ME referees (basket + rainbow) once and cache to DATA_DIR.
# All benchmark runs below will load them instead of recomputing.
echo ""
echo "=== Pre-computing ME referees ==="
"$PRICER" --n "$GRID_N" --save-referee --referee-dir "$DATA_DIR"

# Time-stepping sweep: 2^n steps for n = 4, ..., 10  (16, 32, 64, 128, 256, 512, 1024)
echo ""
echo "=== Time-stepping sweep (steps = 2^n, n = 4, ..., 10) ==="
for exp in $(seq 4 10); do
    steps=$((1 << exp))
    echo "  steps=$steps"
    run_and_collect --benchmark --n "$GRID_N" --steps "$steps" --referee-dir "$DATA_DIR"
done

# KSM-EI tolerance sweep: 5 log-spaced points from 1e-1 to 1e-7
echo ""
echo "=== KSM-EI tolerance sweep (5 log-spaced points: 1e-1 to 1e-7) ==="
for tol in "1e-1" "3.162e-3" "1e-4" "3.162e-6" "1e-7"; do
    echo "  tol=$tol"
    run_and_collect --benchmark --n "$GRID_N" --tol "$tol" --referee-dir "$DATA_DIR"
done

# Compile all individual CSVs into one
echo ""
echo "Compiling results..."
COMBINED="$DATA_DIR/caksm_sweep_n${GRID_N}.csv"
first_csv=$(ls "$DATA_DIR"/caksm_export_n${GRID_N}_*.csv 2>/dev/null | sort | head -1)
if [[ -z "$first_csv" ]]; then
    echo "No CSV files found to compile!"
    exit 1
fi

head -1 "$first_csv" > "$COMBINED"
for f in $(ls "$DATA_DIR"/caksm_export_n${GRID_N}_*.csv | sort); do
    tail -n +2 "$f" >> "$COMBINED"
done

row_count=$(tail -n +2 "$COMBINED" | wc -l)
echo "Results compiled to $COMBINED  ($row_count rows)"
