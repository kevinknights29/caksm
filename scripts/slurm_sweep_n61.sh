#!/bin/bash
#SBATCH -J caksm_n61
#SBATCH -o ./%x.out
#SBATCH -e ./%x.err
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --get-user-env
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

# Sweep parameters
GRID_N=61

# Modules
module load gcc/15.2.0-gcc-8.5.0-r7c4jsu
module load cmake/3.31.9-gcc-15.2.0-ylutpfi
echo "Loaded GCC and CMake"
echo "  CC  = $(which gcc)  ($(gcc --version | head -1))"
echo "  CXX = $(which g++)  ($(g++ --version | head -1))"

# Build
WORK_DIR=$(pwd)
echo "Working directory: $WORK_DIR"

# FETCHCONTENT_UPDATES_DISCONNECTED prevents CMake from re-fetching Eigen /
# Catch2 on compute nodes that may have no outbound internet access.
# Run "cmake -B build ..." once on the login node first to populate the cache.
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

# Time-stepping sweep
# CN, ADI-DR, ADI-HV, KSM-EI: 2^n steps for n = 4, 5, ..., 10
# (16, 32, 64, 128, 256, 512, 1024)
# ME is also run at each point. However, it ignores --steps and gives the same result
# every time, providing a built-in consistency check across the sweep.
echo ""
echo "=== Time-stepper sweep (steps = 2^n, n = 4, ..., 10) ==="
for exp in $(seq 4 10); do
    steps=$((1 << exp))
    echo "  steps=$steps"
    "$PRICER" --benchmark --n "$GRID_N" --steps "$steps" --export
done

# KSM-EI tolerance sweep
# Five points equally spaced in log10 between 1e-1 and 1e-7:
#  10^{-1, -2.5, -4, -5.5, -7}
# When --tol is given, KSM-EI uses the default step count (100)
# All other methods also run with 100 steps for a consistent comparison baseline.
echo ""
echo "=== KSM-EI tolerance sweep (5 log-spaced points: 1e-1 to 1e-7) ==="
for tol in "1e-1" "3.162e-3" "1e-4" "3.162e-6" "1e-7"; do
    echo "  tol=$tol"
    "$PRICER" --benchmark --n "$GRID_N" --tol "$tol" --export
done

# Compile individual CSVs into one file
echo ""
echo "Compiling results..."
COMBINED="$WORK_DIR/caksm_sweep_n${GRID_N}.csv"

first_csv=$(ls "$WORK_DIR"/caksm_export_*.csv 2>/dev/null | sort | head -1)
if [[ -z "$first_csv" ]]; then
    echo "No CSV files found to compile!"
    exit 1
fi

head -1 "$first_csv" > "$COMBINED"
for f in $(ls "$WORK_DIR"/caksm_export_*.csv | sort); do
    tail -n +2 "$f" >> "$COMBINED"
done

row_count=$(tail -n +2 "$COMBINED" | wc -l)
echo "Results compiled to $COMBINED  ($row_count rows)"
echo "Sweep complete!"
