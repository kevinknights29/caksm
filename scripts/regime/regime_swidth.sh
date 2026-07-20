#!/bin/bash
# The s(n) experiment: sweep the block width s and let the two roofs pick it.
#
# Everywhere else s is pinned at the certified width. That hides the question: s is pushed
# up by both mechanisms (a wider block means fewer reductions and more powers per
# operator-stream) and pulled down by two independent ceilings:
#
#   numerical (the certificate)  kappa([v, ..., A^s v]) grows geometrically in s; past
#                                u^(-1/2) the monomial basis leaves the CholeskyQR
#                                certificate (the block builds but won't orthogonalize).
#   capacity  (the ghost)        the halo is s*w rows per side, so a wider block shrinks
#                                the panel until none fits and the tiling collapses.
#
# Which binds first is a joint property of operator and machine. If the certificate binds,
# a better-conditioned basis (Newton, Chebyshev) converts the halo's slack into real
# reductions, if the ghost binds, only a thinner bandwidth or a coarser tiling level moves
# it. Pinning s assumes the first answer without measuring it.
#
# Run it spilled: the reuse half is meaningless while the operator fits in cache, so the
# grids below have R_v(1 core) > 1 (for the 2D scaffold, ~132 bytes/point spills a 16 MiB
# slice past N ~ 127k, n1 ~ 356). Check the binary's R_v > 1 before reading reuse.
#
# Both tiling levels are swept: L2 has less capacity, so its ghost roof arrives sooner. If
# the certificate binds at L3 but the ghost at L2, the binding roof depends on the tiling
# level too.
#
# Assumes the project has already been built: cmake --build build --parallel
set -uo pipefail

WORK_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SWEEP="$WORK_DIR/build/regime-sweep"
DATA_DIR="$WORK_DIR/data/regime"
CSV="$DATA_DIR/regime_swidth.csv"

MACHINE="${MACHINE:-amd-3960x}"
M="${M:-8}"
DIM="${DIM:-2}"
REPEATS="${REPEATS:-5}"
S_CEIL="${S_CEIL:-24}"
# Spilled grids: N = n1^2 well past the ~127k where a 16 MiB slice overflows.
N1_LIST=(${N1_LIST:-420 480 560})

if [[ ! -x "$SWEEP" ]]; then
    echo "Error: regime-sweep not found at $SWEEP"
    echo "Build it: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel"
    exit 1
fi

mkdir -p "$DATA_DIR"
rm -f "$CSV"

echo "==============================================================================="
echo " s-width sweep   machine=$MACHINE  m=$M  dim=$DIM  s=1..$S_CEIL  repeats=$REPEATS"
echo "==============================================================================="
for level in l3 l2; do
    for n1 in "${N1_LIST[@]}"; do
        echo ""
        echo "=== tile-level=$level  n1=$n1  (N=$(( n1 ** DIM ))) ==="
        "$SWEEP" --machine "$MACHINE" --m "$M" --dim "$DIM" --repeats "$REPEATS" \
                 --pattern banded --tile-level "$level" --n1 "$n1" \
                 --swidth "$S_CEIL" --csv "$CSV" \
            || echo "!!! point FAILED: level=$level n1=$n1"
    done
done

echo ""
echo "Rows -> $CSV"
echo "Plot with: uv run scripts/plots/regime_swidth_plot.py"
