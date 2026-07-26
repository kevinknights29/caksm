#!/bin/bash
# Numerics gate, not a timing run: nothing is threaded or timed, and no result depends on
# the host. It establishes that the synthetic instrument's a-priori predictions about
# arithmetic hold before any communication claim is made:
#
#   C1  the Kronecker-sum spectrum is analytic and correct
#   C2  the scatter knob is a similarity transform (pure-axis motion exists)
#   C3  measured Krylov dimension m obeys the Hochbruck-Lubich spectral bound
#   C4  kappa of the matrix-powers basis equals the Vandermonde from the spectrum
#   C5  the CholeskyQR certificate kappa <= u^(-1/2) is sound
#
# Five phases, grouped into jobs by cost (build_points below has the point list):
#   Phase 1  gate       1D/2D/3D Kronecker scaffolds. All three must exit 0; a gate
#                       failure is fatal (the instrument is wrong).
#   Phase 2  confound   separate spectrum spread from m: arm A pins m and varies spread,
#                       arm B pins the spectrum and varies m. Expect s_max set by the
#                       spectrum, not m. Plus a shift control.
#   Phase 3  m-range    push m across the real problem's measured 8-11 window.
#   Phase 4  non-normal measured (Henrici departure), by mechanism: advection (removable),
#                       correlation (non-separable), rho+gamma (genuinely non-normal),
#                       var-advection (a second, family-scope-testing mechanism). C1/C3/C4
#                       become findings where their inputs vanish; they never abort.
#   Phase 5  real-BS    the actual discretized BS basket operator (--real-bs) with its
#                       real payoff, swept across grids. Transfer by measurement: it lands
#                       the real operator on the same (R_v, R_h) plane.
#
# The scaffold is small by design (claims are N-independent) and the dense eigensolve caps
# grids at n <= 32 (N = n^3 <= 32768). Each invocation is independent and single-threaded,
# so jobs run concurrently, each writing its own part-CSV; --merge stitches them.
#
#   ./scripts/regime/regime_control.sh --list      # show the plan and job assignment
#   ./scripts/regime/regime_control_launch.sh      # gate first, then one tmux session per job
#
#   or by hand, one tmux session each:
#     JOB=gate      ./scripts/regime/regime_control.sh   # must exit 0 before trusting the rest
#     JOB=confound  ./scripts/regime/regime_control.sh
#     JOB=nonnormal ./scripts/regime/regime_control.sh
#     JOB=realbs    ./scripts/regime/regime_control.sh
#     JOB=realbs20  ./scripts/regime/regime_control.sh
#     ./scripts/regime/regime_control.sh --merge          # once they have all finished
#
#   or serial:
#     ./scripts/regime/regime_control.sh                  # JOB defaults to "all"
#     ./scripts/regime/regime_control.sh --merge
#
# Assumes the project has already been built: cmake --build build --parallel
set -uo pipefail          # NOT -e: a non-gate point failing must not kill the sweep

WORK_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
CONTROL="$WORK_DIR/build/regime-control"
DATA_DIR="$WORK_DIR/data/regime"
PARTS_DIR="$DATA_DIR/parts"
CSV="$DATA_DIR/regime_control.csv"

MACHINE="${MACHINE:-amd-3960x}"
P="${P:-24}"
S_CEILING="${S_CEILING:-40}"
REAL_BS_GRIDS=(${REAL_BS_GRIDS:-10 12 14 16 18 20})
# Real-BS grids at or above this N get their own job (the eigensolve dominates); smaller
# ones share the "realbs" job. n^3 >= 4096  <=>  n >= 16.
REALBS_HEAVY_N="${REALBS_HEAVY_N:-4096}"
JOB="${JOB:-all}"

# The point list. Each entry is "<job-tag>|<regime-control args>". A gate point failing
# is fatal (the instrument is wrong); every other tag logs a failure and carries on.
build_points() {
    points=(
        # Phase 1  gate: 1D/2D/3D scaffolds. d = 3,5,7 nonzeros/row; all
        # dense-eigensolver-checkable (C1 verifies the analytic spectrum below N=2048).
        "gate|--n1 64 --dim 1"                              # N =   64
        "gate|--n1 24 --dim 2"                              # N =  576
        "gate|--n1 12 --dim 3"                              # N = 1728

        # Phase 2  confound. Arm A: spread varies, m pinned (h scaled by 1/sigma).
        "confound|--n1 24 --dim 2 --scale 1 --h 0.01"
        "confound|--n1 24 --dim 2 --scale 2 --h 0.005"
        "confound|--n1 24 --dim 2 --scale 4 --h 0.0025"
        "confound|--n1 24 --dim 2 --scale 8 --h 0.00125"
        # Arm B: spectrum pinned, m varies (h and tol only).
        "confound|--n1 24 --dim 2 --h 0.02"
        "confound|--n1 24 --dim 2 --h 0.05"
        "confound|--n1 24 --dim 2 --h 0.20"
        "confound|--n1 24 --dim 2 --h 0.40 --tol 1e-10"
        # Shift control: moves conditioning, cannot move m.
        "confound|--n1 24 --dim 2 --shift 2.0"
        "confound|--n1 24 --dim 2 --shift 3.5"

        # Phase 3  m across the real problem's 8-11 window.
        "confound|--n1 24 --dim 2 --h 0.15 --tol 1e-08"
        "confound|--n1 24 --dim 2 --h 0.20 --tol 1e-08"
        "confound|--n1 24 --dim 2 --h 0.30 --tol 1e-10"

        # Phase 4  non-normality. Advection sweep (removable, analytic kappa(X)).
        "nonnormal|--n1 24 --dim 2 --advection 0.05"
        "nonnormal|--n1 24 --dim 2 --advection 0.10"
        "nonnormal|--n1 24 --dim 2 --advection 0.15"
        "nonnormal|--n1 24 --dim 2 --advection 0.20"
        "nonnormal|--n1 24 --dim 2 --advection 0.25"
        "nonnormal|--n1 24 --dim 2 --advection 0.30"
        "nonnormal|--n1 24 --dim 2 --advection 0.40"
        # Correlation (normal, non-separable) and rho+gamma (genuinely non-normal).
        "nonnormal|--n1 24 --dim 2 --correlation 0.15"
        "nonnormal|--n1 24 --dim 2 --correlation 0.30"
        "nonnormal|--n1 24 --dim 2 --correlation 0.20 --advection 0.10"
        "nonnormal|--n1 24 --dim 2 --correlation 0.30 --advection 0.20"
        "nonnormal|--n1 24 --dim 2 --correlation 0.30 --advection 0.30"
        # Variable-coefficient advection: second, family-scope-testing mechanism.
        "nonnormal|--n1 24 --dim 2 --var-advection 0.20"
        "nonnormal|--n1 24 --dim 2 --var-advection 0.40"
        "nonnormal|--n1 24 --dim 2 --var-advection 0.60"
        "nonnormal|--n1 24 --dim 2 --var-advection 0.80"
        "nonnormal|--n1 24 --dim 2 --var-advection 0.95"
    )
    # Phase 5  real-BS trajectory. Heavy grids (N >= REALBS_HEAVY_N) get their own job.
    #
    # h must be the real solver's per-step propagation time, not an arbitrary probe value.
    # The KSM-EI solver takes t_final/temporal_steps = 1.0/100 = 0.01 per step, so m and
    # blocks_worst reported here are the real solver's, at its real operating point. An
    # oversized h (e.g. 0.20) inflates rho = h*spread/4, hence m, hence the s-step block
    # count, misrepresenting exactly the transfer numbers this phase exists to measure.
    for n in "${REAL_BS_GRIDS[@]}"; do
        local N=$(( n * n * n ))
        local tag="realbs"
        if (( N >= REALBS_HEAVY_N )); then tag="realbs$n"; fi
        points+=("$tag|--real-bs $n --h ${REAL_BS_H:-0.01} --tol 1e-08")
    done
}

jobs_of() {   # unique job tags, in first-seen order (gate first, by construction)
    local seen=" " p tag
    for p in "${points[@]}"; do
        tag="${p%%|*}"
        [[ "$seen" == *" $tag "* ]] || { echo "$tag"; seen+="$tag "; }
    done
}

# A short, human label for a point (for --list and progress lines).
point_label() {
    local args="$1"
    local rb; rb=$(sed -n 's/.*--real-bs \([0-9]*\).*/\1/p' <<<"$args")
    if [[ -n "$rb" ]]; then
        echo "real-BS n=$rb  (N=$(( rb*rb*rb )))"
    else
        local d n1; d=$(sed -n 's/.*--dim \([0-9]*\).*/\1/p' <<<"$args")
        n1=$(sed -n 's/.*--n1 \([0-9]*\).*/\1/p' <<<"$args")
        echo "N=$(( n1 ** d ))  ${args}"
    fi
}

build_points

# --list : show the plan and the job assignment
if [[ "${1:-}" == "--list" ]]; then
    echo "Point plan (machine=$MACHINE, P=$P):"
    printf '%-10s %s\n' JOB POINT
    for p in "${points[@]}"; do
        printf '%-10s %s\n' "${p%%|*}" "$(point_label "${p#*|}")"
    done
    echo
    echo "Jobs:  $(jobs_of | tr '\n' ' ')"
    echo "The gate job must pass before the rest are trusted."
    echo "Run one tmux session per job, then:  ./scripts/regime/regime_control.sh --merge"
    exit 0
fi

# --merge : stitch the part-CSVs into the final CSV (header once, gate part first)
if [[ "${1:-}" == "--merge" ]]; then
    shopt -s nullglob
    # Gate part first so the final CSV opens with the gating scaffolds.
    parts=()
    [[ -f "$PARTS_DIR/regime_control_gate.csv" ]] && parts+=("$PARTS_DIR/regime_control_gate.csv")
    for f in "$PARTS_DIR"/regime_control_*.csv; do
        [[ "$f" == "$PARTS_DIR/regime_control_gate.csv" ]] && continue
        parts+=("$f")
    done
    if (( ${#parts[@]} == 0 )); then
        echo "No parts in $PARTS_DIR; nothing to merge."
        exit 1
    fi
    head -1 "${parts[0]}" > "$CSV"
    for f in "${parts[@]}"; do
        tail -n +2 "$f" >> "$CSV"
        printf '  merged %-48s %4d rows\n' "$(basename "$f")" "$(( $(wc -l < "$f") - 1 ))"
    done
    echo "Merged ${#parts[@]} parts -> $CSV  ($(( $(wc -l < "$CSV") - 1 )) rows)"
    echo "Plot with:  uv run scripts/plots/regime_plot.py"
    exit 0
fi

if [[ ! -x "$CONTROL" ]]; then
    echo "Error: regime-control not found at $CONTROL"
    echo "Build first: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel"
    exit 1
fi

mkdir -p "$PARTS_DIR"
OUT="$PARTS_DIR/regime_control_${JOB}.csv"
rm -f "$OUT"        # only THIS job's part; never touches other jobs' results

# run <tag> <args...> : one point. A gate failure is fatal; any other is logged.
run() {
    local tag="$1"; shift
    echo "--- regime-control $*"
    if "$CONTROL" --machine "$MACHINE" --P "$P" --s-ceiling "$S_CEILING" --csv "$OUT" "$@"; then
        echo
        return 0
    fi
    echo
    if [[ "$tag" == "gate" ]]; then
        echo "!!! GATE POINT FAILED: $* "
        echo "!!! The instrument is wrong; every downstream number is meaningless. Stopping."
        exit 1
    fi
    echo "!!! point FAILED or was killed (N too large for a dense eigensolve?): $*"
}

echo "==============================================================================="
echo "JOB '$JOB'   ->   $OUT"
echo "  machine=$MACHINE  P=$P  s-ceiling=$S_CEILING"
echo "  each point is single-threaded; run other jobs concurrently in their own tmux"
echo "==============================================================================="
ran=0
for p in "${points[@]}"; do
    tag="${p%%|*}"; args="${p#*|}"
    [[ "$JOB" == "all" || "$JOB" == "$tag" ]] || continue
    echo "### [$tag] $(point_label "$args")"
    # shellcheck disable=SC2086
    run "$tag" $args
    ran=$(( ran + 1 ))
done

if (( ran == 0 )); then
    echo "No points matched JOB='$JOB'.  Valid jobs: $(jobs_of | tr '\n' ' ')"
    exit 1
fi

echo "==============================================================================="
echo "JOB '$JOB' done: $ran point(s) -> $OUT"
echo "When every job has finished:  ./scripts/regime/regime_control.sh --merge"
echo
echo "Columns of interest:"
echo "  s_max_phys_worst / blocks_worst         measured per-vector certificate"
echo "  henrici, is_normal, analytic_spectrum   the non-normality diagnostics"
echo "  m_measured                              Krylov dimension the solver converges at"
echo "  red_mgs / red_ca_cholqr2                reductions/cycle, MGS vs stable s-step"
echo "  real_bs                                 1 = the actual BS operator (Phase 5 trajectory)"
echo "  R_v / R_h                               map coordinates, rv_spmv/rv_mgs and"
echo "                                          spmv/mgs_resident give the per-kernel residency"
echo "                                          R_h's denominator is priced from"
echo "==============================================================================="
