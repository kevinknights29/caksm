#!/bin/bash
#SBATCH -J caksm_n31
#SBATCH -o ./%x.out
#SBATCH -e ./%x.err
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --get-user-env
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00

# Sweep parameters
GRID_N=31

# Modules
# Load gcc first so it lands on PATH, then cmake (which also needs gcc-15 libs).
module load gcc/15.2.0-gcc-8.5.0-r7c4jsu
module load cmake/3.31.9-gcc-15.2.0-ylutpfi
echo "Loaded GCC and CMake"
echo "  CC  = $(which gcc)  ($(gcc --version | head -1))"
echo "  CXX = $(which g++)  ($(g++ --version | head -1))"

# Directories
WORK_DIR=$(pwd)
DATA_DIR="$WORK_DIR/data/n${GRID_N}"
mkdir -p "$DATA_DIR"
echo "Working directory : $WORK_DIR"
echo "Data directory    : $DATA_DIR"

# Build
# Pass the compiler explicitly so CMake does not fall back to Seagull's
# /usr/bin/c++ (GCC 8.5.0), which does not support C++23.
# Wipe any stale CMakeCache that may have locked in the wrong compiler.
rm -f "$WORK_DIR/build/CMakeCache.txt"

echo "Configuring..."
if ! cmake -B "$WORK_DIR/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER="$(which gcc)" \
        -DCMAKE_CXX_COMPILER="$(which g++)" \
        -DFETCHCONTENT_UPDATES_DISCONNECTED=ON \
        -S "$WORK_DIR"; then
    echo "CMake configuration failed!"
    exit 1
fi

echo "Building..."
if ! cmake --build "$WORK_DIR/build" --parallel; then
    echo "Build failed!"
    exit 1
fi
echo "Build successful!"

PRICER="$WORK_DIR/build/pricer"

echo ""
echo "========================================"
echo " Parameter sweep  n=$GRID_N"
echo "========================================"

# Helper: run pricer then immediately move the new export file into DATA_DIR.
# The filename embeds the grid size (caksm_export_n31_TIMESTAMP.csv) so it
# cannot collide with files from a concurrent n61 job in the same tree.
run_and_collect() {
    "$PRICER" "$@" --export
    mv "$WORK_DIR"/caksm_export_n${GRID_N}_*.csv "$DATA_DIR/"
}

# Time-stepping sweep
# CN, ADI-DR, ADI-HV, KSM-EI: 2^n steps for n = 4, 5, ..., 10
# (16, 32, 64, 128, 256, 512, 1024)
# ME is also run at each point. However, it ignores --steps and gives the same result
# every time, providing a built-in consistency check across the sweep.
# Pre-compute ME referees (basket + rainbow) once and cache to DATA_DIR.
# All benchmark runs below will load them instead of recomputing.
echo ""
echo "=== Pre-computing ME referees ==="
"$PRICER" --n "$GRID_N" --save-referee --referee-dir "$DATA_DIR"

echo ""
echo "=== Time-stepping sweep (steps = 2^n, n = 4, ..., 10) ==="
for exp in $(seq 4 10); do
    steps=$((1 << exp))
    echo "  steps=$steps"
    run_and_collect --benchmark --n "$GRID_N" --steps "$steps" --referee-dir "$DATA_DIR"
done

# KSM-EI tolerance sweep
# Five points equally spaced in log10 between 1e-1 and 1e-7:
#  10^{-1, -2.5, -4, -5.5, -7}
# When --tol is given, KSM-EI uses the default step count (100).
# All other methods also run with 100 steps for a consistent comparison baseline.
echo ""
echo "=== KSM-EI tolerance sweep (5 log-spaced points: 1e-1 to 1e-7) ==="
for tol in "1e-1" "3.162e-3" "1e-4" "3.162e-6" "1e-7"; do
    echo "  tol=$tol"
    run_and_collect --benchmark --n "$GRID_N" --tol "$tol" --referee-dir "$DATA_DIR"
done

# Compile individual CSVs into one file
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
echo "Sweep complete!"
