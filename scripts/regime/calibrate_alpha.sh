#!/bin/bash
# Calibrate alpha: the per-tree-level latency of one global reduction.
#
# A prerequisite, not an experiment. Until it lands, R_h's numerator alpha*h*log2(P)
# has a shape in P but no magnitude, so theta_h has no place on any figure. The
# measurement is a bare scalar all-reduce over an empty team, timed once offline,
# so it is known before any run it places.
#
# alpha is a property of one machine's cores, interconnect and OpenMP runtime.
# It does not transfer. Run it on the host you intend to place, and record the result against
# that host's preset only.
#
# Three arms, only the first produces a number to record:
#   tree            binary tree, point-to-point flag sync. The primitive src/scaling.cpp
#                   reduces through.
#   linear/shared   barrier + O(P) redundant scan. The artifact the first calibration
#                   measured. Kept so the size of the correction is on the record.
#   linear/padded   the same scan with one cache line per thread. Sizes false sharing.
#
# Assumes the project has already been built: cmake --build build --parallel
set -euo pipefail

WORK_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
CALIB="$WORK_DIR/build/calibrate-alpha"
DATA_DIR="$WORK_DIR/data/regime"

MACHINE="${MACHINE:-amd-3960x}"
REDUCES="${REDUCES:-20000}"   # reductions per timed repeat
REPEATS="${REPEATS:-7}"       # timed repeats after 1 warm-up

if [[ ! -x "$CALIB" ]]; then
    echo "Error: calibrate-alpha not found at $CALIB"
    echo "Build it first: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel"
    exit 1
fi

mkdir -p "$DATA_DIR"

echo "==============================================================================="
echo " alpha calibration   machine=$MACHINE  reduces=$REDUCES  repeats=$REPEATS"
echo "==============================================================================="
echo
echo "Toolchain / host provenance (record this alongside the number):"
echo "  host   : $(uname -n)  $(uname -srm)"
if command -v lscpu >/dev/null 2>&1; then
    lscpu | grep -E 'Model name|Socket|NUMA node\(s\)|Thread\(s\) per core' | sed 's/^/  cpu    : /'
fi
# Absolute latencies are machine-state dependent. The governor does not change the fit's
# shape, but it moves alpha's magnitude, which is the whole point here, so it is recorded.
if command -v cpupower >/dev/null 2>&1; then
    echo "  governor: $(cpupower frequency-info -p 2>/dev/null | tail -1 | tr -s ' ')"
elif [[ -r /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
    echo "  governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
else
    echo "  governor: (unreadable; treat the alpha magnitude as machine-state dependent)"
fi
echo

run_arm() {   # <tag> <reduce-kind> <layout>
    echo "### $1"
    OMP_PLACES=cores OMP_PROC_BIND=close \
        "$CALIB" --machine "$MACHINE" --reduces "$REDUCES" --repeats "$REPEATS" \
                 --reduce "$2" --layout "$3" --csv "$DATA_DIR/calibrate_alpha_$1.csv"
    echo
}

run_arm tree          tree   shared
run_arm linear_shared linear shared
run_arm linear_padded linear padded

echo "==============================================================================="
echo "Done."
echo "  data/regime/calibrate_alpha_tree.csv           <- the calibration"
echo "  data/regime/calibrate_alpha_linear_shared.csv  <- the artifact"
echo "  data/regime/calibrate_alpha_linear_padded.csv  <- false-sharing diagnostic"
echo "==============================================================================="
