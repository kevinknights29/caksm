#!/bin/bash
# The vertical crossover sweep: does cache-blocked matrix-powers pay above R_v=1?
#
# Result up front: measured tiled/baseline speedup stayed near 1x (0.8-1.4x) on every
# grid, never nearing the modeled traffic-ratio ceiling (4-8x). The kernel never reached
# a bandwidth-bound regime, so this is a ceiling, not a refuted boundary.
# The no-halo diagnostic splits by tile level:
#
#   banded L3   thin halo, large panels. Removing the halo barely moves GFLOP/s: neither
#               arm is bandwidth-bound, so tiling has no traffic to convert. Most
#               consistent with gather latency (the indirect column access), not capacity.
#   banded L2   fat halo, tiny panels. Removing the halo recovers throughput toward
#               baseline: the panel is cache-resident and the mechanism works, it just
#               costs more in redundant halo arithmetic than it saves at this panel size.
#   scattered   the control: halo ~ N, tiling refuses, so the tiled arm is the baseline
#               and speedup is ~1 everywhere, as designed.
#
# Single-threaded and independent per point, small enough to run serially.
# The binary's AI gate flags any point where the tiled kernel missed its
# modeled reuse.
#
# Assumes the project has already been built: cmake --build build --parallel
set -uo pipefail

WORK_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SWEEP="$WORK_DIR/build/regime-sweep"
DATA_DIR="$WORK_DIR/data/regime"
CSV="$DATA_DIR/regime_sweep.csv"
# The no-halo diagnostic's structured output, DELIBERATELY separate from $CSV: its basis
# is known wrong, so it must never sit in the file a plot might treat as real points.
DEBUG_CSV="$DATA_DIR/regime_sweep_nohalo.csv"

MACHINE="${MACHINE:-amd-3960x}"
M="${M:-8}"
REPEATS="${REPEATS:-7}"
DIM="${DIM:-2}"
# n1 grid: N = n1^2 spans ~1e4 (cache-resident) to ~3e5 (well spilled past a 16 MiB slice).
# 16 MiB / ~144 B per point ~ 116k points ~ n1 ~ 340, so bracket that.
N1_LIST=(${N1_LIST:-64 128 200 280 340 400 480 560})

if [[ ! -x "$SWEEP" ]]; then
    echo "Error: regime-sweep not found at $SWEEP"
    echo "Build it: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel"
    exit 1
fi

mkdir -p "$DATA_DIR"
rm -f "$CSV" "$DEBUG_CSV"

# Every point's full output lands in $LOG; the lines that matter (AI gate, DEBUG no-halo,
# and the reasoning after them) are also echoed live, so a kernel bug is distinguishable
# from a genuine boundary without a second pass through the file.
LOG="$DATA_DIR/regime_sweep.out"
rm -f "$LOG"

run() {   # <pattern> <tile-level> <n1>
    local out status
    if out=$("$SWEEP" --machine "$MACHINE" --m "$M" --dim "$DIM" --repeats "$REPEATS" \
                       --pattern "$1" --tile-level "$2" --n1 "$3" --csv "$CSV" \
                       --debug-csv "$DEBUG_CSV" 2>&1); then
        status=ok
    else
        status=FAIL
    fi
    { echo "### pattern=$1 tile-level=$2 n1=$3"; echo "$out"; echo; } >> "$LOG"

    echo "$status"   # completes the caller's un-terminated 'n1=... N=... ... ' line

    local important
    important=$(grep -E 'AI gate:|DEBUG no-halo|^  ->|^     ' <<<"$out" || true)
    [[ -n "$important" ]] && printf '%s\n' "$important" | sed 's/^/    /'
}

echo "==============================================================================="
echo " Vertical crossover sweep   machine=$MACHINE  m=$M  dim=$DIM  repeats=$REPEATS"
echo " Full per-point output -> $LOG"
echo "==============================================================================="
for arm in "banded l3" "banded l2" "scattered l3"; do
    set -- $arm
    echo ""
    echo "=== pattern=$1  tile-level=$2 ==="
    for n1 in "${N1_LIST[@]}"; do
        printf '  n1=%-4s N=%-8s ... ' "$n1" "$(( n1 ** DIM ))"
        run "$1" "$2" "$n1"
    done
done

echo ""
echo "Rows -> $CSV"
echo "No-halo diagnostic (structured, spilled points only) -> $DEBUG_CSV"
echo "Full output, including every AI-gate verdict and DEBUG no-halo diagnostic -> $LOG"
echo "Plot with: uv run scripts/plots/regime_sweep_plot.py"
