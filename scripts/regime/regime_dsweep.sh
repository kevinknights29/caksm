#!/bin/bash
# The asset-dimension law: how does kappa(X) grow with the number of assets?
#
# The earlier refinement study swept the wrong axis (grid N at fixed d=3) and found
# kappa(X) flat. But the thesis grows the number of assets, not the grid. From the
# diagonal-similarity formula, log kappa(X) ~ d*n1*g with mesh Peclet g ~ C/n1, so n1
# cancels and only d is left: the flatness in n1 confirms the mesh-Peclet reading and says
# nothing about d.
#
# Measured (n1=4, gamma=0.3, rho=0.3), kappa(X) grows super-linearly in d:
#     d      N    kappa(X)   log10
#     2     16      6.7      0.824
#     3     64     17.0      1.229
#     4    256    109.5      2.039
#     5   1024   5.2e3       3.716
#     6   4096   3.7e4       4.564
#
# Non-normality enters through pairwise cross terms (C(d,2) axis-pairs), so the fit is
# quadratic in d: log10 kappa(X) ~ a*C(d,2) + b*d. The two models agree on the measured
# range but diverge by ~10^3.8 at d=10, so high-d kappa(X) must be measured, never
# extrapolated.
#
# The operator is the synthetic BS-like one (advection = mesh Peclet, correlation = the
# mixed-derivative cross term); the real operator is hard-coded to 3 assets and cannot
# supply this axis. Each invocation is independent and single-threaded, so points run one
# job per core, each writing its own part-CSV; --merge stitches them.
#
#     ./scripts/regime/regime_dsweep.sh --list          # show the plan and job assignment
#     ./scripts/regime/regime_dsweep_launch.sh          # one tmux session per job (recommended)
#
#   or by hand, one tmux session each:
#     JOB=cheap ./scripts/regime/regime_dsweep.sh
#     JOB=d5    ./scripts/regime/regime_dsweep.sh
#     JOB=d6    ./scripts/regime/regime_dsweep.sh
#     JOB=rho   ./scripts/regime/regime_dsweep.sh
#     ./scripts/regime/regime_dsweep.sh --merge         # once they have all finished
#
#   or serial (everything in one process):
#     nohup stdbuf -oL ./scripts/regime/regime_dsweep.sh > logs/regime_dsweep.out 2>&1 &
#
# The kappa(X) SVD uses Eigen's BDCSVD, not JacobiSVD (see include/regime.hpp): Jacobi is
# unblocked and cost ~190x more at N=512, dominating the run. If a point feels slow, check
# that first.
#
# Assumes the project has already been built: cmake --build build --parallel
set -uo pipefail          # NOT -e: one oversized point must not kill the sweep

WORK_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
CONTROL="$WORK_DIR/build/regime-control"
DATA_DIR="$WORK_DIR/data/regime"
PARTS_DIR="$DATA_DIR/parts"
CSV="$DATA_DIR/regime_dsweep.csv"

# Mesh-Peclet advection and correlation, held fixed so that d is the only axis moving.
GAMMA="${GAMMA:-0.30}"
RHO="${RHO:-0.30}"
N1="${N1:-4}"             # small and fixed: only the asset dimension varies
D_LIST=(${D_LIST:-2 3 4 5 6})
RHO_LIST=(${RHO_LIST:-0.0 0.3 0.6 0.9 0.99})
RHO_DIM="${RHO_DIM:-3}"   # asset dimension for the correlation stress sweep
RHO_N1="${RHO_N1:-8}"
JOB="${JOB:-all}"

# The point list. Each entry is "<job-tag>|<regime-control args>". Cost is wildly
# imbalanced (a d=6 point dwarfs every cheap point put together), so the split is by cost:
# each expensive point gets its own job and the cheap ones share one, keeping the
# wall-clock set by the single slowest point.
build_points() {
    points=()
    for d in "${D_LIST[@]}"; do
        local N=$(( N1 ** d ))
        local tag
        if (( N >= 1024 )); then tag="d$d"; else tag="cheap"; fi   # heavy => own job
        points+=("$tag|--n1 $N1 --dim $d --advection $GAMMA --correlation $RHO")
    done
    for r in "${RHO_LIST[@]}"; do
        points+=("rho|--n1 $RHO_N1 --dim $RHO_DIM --advection $GAMMA --correlation $r")
    done
}

jobs_of() {   # unique job tags, in first-seen order
    local seen=" " p tag
    for p in "${points[@]}"; do
        tag="${p%%|*}"
        [[ "$seen" == *" $tag "* ]] || { echo "$tag"; seen+="$tag "; }
    done
}

build_points

# --list : show the plan
if [[ "${1:-}" == "--list" ]]; then
    echo "Point plan (n1=$N1, gamma=$GAMMA, rho=$RHO):"
    printf '%-8s %-10s %s\n' JOB N ARGS
    for p in "${points[@]}"; do
        args="${p#*|}"; tag="${p%%|*}"
        d=$(sed -n 's/.*--dim \([0-9]*\).*/\1/p' <<<"$args")
        n1=$(sed -n 's/.*--n1 \([0-9]*\).*/\1/p' <<<"$args")
        printf '%-8s %-10s %s\n' "$tag" "$(( n1 ** d ))" "$args"
    done
    echo
    echo "Jobs:  $(jobs_of | tr '\n' ' ')"
    echo "Run one tmux session per job, then:  ./scripts/regime/regime_dsweep.sh --merge"
    exit 0
fi

# --merge : stitch the part-CSVs into the final CSV (header once)
if [[ "${1:-}" == "--merge" ]]; then
    shopt -s nullglob
    parts=("$PARTS_DIR"/regime_dsweep_*.csv)
    if (( ${#parts[@]} == 0 )); then
        echo "No parts in $PARTS_DIR; nothing to merge."
        exit 1
    fi
    head -1 "${parts[0]}" > "$CSV"
    for f in "${parts[@]}"; do
        tail -n +2 "$f" >> "$CSV"
        printf '  merged %-52s %5d rows\n' "$(basename "$f")" "$(( $(wc -l < "$f") - 1 ))"
    done
    echo "Merged $(( ${#parts[@]} )) parts -> $CSV  ($(( $(wc -l < "$CSV") - 1 )) rows)"
    echo "Fit the law with:  uv run scripts/plots/regime_dsweep_plot.py"
    exit 0
fi

if [[ ! -x "$CONTROL" ]]; then
    echo "Error: regime-control not found at $CONTROL"
    echo "Build first: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel"
    exit 1
fi

mkdir -p "$PARTS_DIR"
OUT="$PARTS_DIR/regime_dsweep_${JOB}.csv"
rm -f "$OUT"        # only THIS job's part; never touches other jobs' results

run() {
    echo "--- regime-control $*"
    if ! "$CONTROL" --csv "$OUT" "$@"; then
        echo "!!! point FAILED or was killed (N too large for a dense eigensolve?): $*"
    fi
    echo
}

echo "==============================================================================="
echo "JOB '$JOB'   ->   $OUT"
echo "  n1=$N1 (fixed)  gamma=$GAMMA  rho=$RHO"
echo "  each point is single-threaded; run other jobs concurrently in their own tmux"
echo "==============================================================================="
ran=0
for p in "${points[@]}"; do
    tag="${p%%|*}"; args="${p#*|}"
    [[ "$JOB" == "all" || "$JOB" == "$tag" ]] || continue
    d=$(sed -n 's/.*--dim \([0-9]*\).*/\1/p' <<<"$args")
    n1=$(sed -n 's/.*--n1 \([0-9]*\).*/\1/p' <<<"$args")
    echo "### [$tag] N = $n1^$d = $(( n1 ** d ))"
    # shellcheck disable=SC2086
    run $args
    ran=$(( ran + 1 ))
done

if (( ran == 0 )); then
    echo "No points matched JOB='$JOB'.  Valid jobs: $(jobs_of | tr '\n' ' ')"
    exit 1
fi

echo "==============================================================================="
echo "JOB '$JOB' done: $ran point(s) -> $OUT"
echo "When every job has finished:  ./scripts/regime/regime_dsweep.sh --merge"
echo "==============================================================================="
