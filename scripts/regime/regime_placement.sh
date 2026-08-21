#!/usr/bin/env bash
# Validate the modeled real-Black--Scholes trajectory at every production-grid
# point drawn by regime_plot.py.  The dense eigensolve controls stop at n=20;
# these placement-only actions retain the same operator, payoff, h, tolerance,
# participant count, and machine constants while skipping infeasible dense
# diagnostics.
#
# Run on puffin.  The points are serial because each large sparse action uses a
# substantial fraction of host memory; no timing is interpreted.
#
#   tmux new-session -d -s regime-placement \
#     'cd "$PWD" && ./scripts/regime/regime_placement.sh > data/regime/regime_placement.log 2>&1'
#
# Follow with:
#   tmux attach -t regime-placement
#   column -s, -t data/regime/regime_placement.csv

set -euo pipefail

ROOT="${ROOT:-$PWD}"
CONTROL="${CONTROL:-$ROOT/build/regime-control}"
DATA_DIR="${DATA_DIR:-$ROOT/data/regime}"
REPORT_DIR="${REPORT_DIR:-$DATA_DIR/placement}"
CSV="${CSV:-$DATA_DIR/regime_placement.csv}"

MACHINE="${MACHINE:-amd-3960x}"
P="${P:-24}"
H="${H:-0.01}"
TOL="${TOL:-1e-8}"
M_CEILING="${M_CEILING:-64}"
PARSE_ONLY="${PARSE_ONLY:-0}"
# n:predicted-m pairs from the current log--log continuation fitted to the
# measured n=10,...,20 controls.  Override POINTS to test another declared fit.
POINTS=(${POINTS:-25:8 30:9 40:10 50:11 61:12 74:13 90:14 120:15})

if [[ ! -x "$CONTROL" ]]; then
    echo "Error: regime-control not found at $CONTROL"
    echo "Build first: cmake --build build --parallel"
    exit 1
fi
if [[ "$PARSE_ONLY" != "1" ]] \
   && ! "$CONTROL" --help 2>&1 | grep -q -- '--placement-only'; then
    echo "Error: $CONTROL predates the placement-only interface."
    echo "Sync include/regime_control_support.hpp and src/regime_control.cpp, then rebuild:"
    echo "  cmake --build build --target regime-control --parallel"
    exit 1
fi

mkdir -p "$REPORT_DIR"
printf '%s\n' \
    "n,N,nnz,h,tol,P,m_ceiling,m_predicted,m_measured,R_v_predicted,R_v_measured,R_v_ratio,R_h_predicted,R_h_measured,R_h_ratio,exit_status,report" \
    > "$CSV"

failures=0
for point in "${POINTS[@]}"; do
    n="${point%%:*}"
    predicted_m="${point#*:}"
    if [[ ! "$n" =~ ^[0-9]+$ || ! "$predicted_m" =~ ^[0-9]+$ ]]; then
        echo "Error: invalid POINTS entry '$point'; expected n:predicted-m"
        exit 1
    fi

    report="$REPORT_DIR/n${n}.txt"
    echo "### placement n=$n predicted_m=$predicted_m"
    if [[ "$PARSE_ONLY" == "1" ]]; then
        if [[ ! -s "$report" ]]; then
            echo "Error: missing placement report for parse-only mode: $report"
            failures=$((failures + 1))
            continue
        fi
        status=0
    else
        set +e
        "$CONTROL" \
            --machine "$MACHINE" --P "$P" \
            --real-bs "$n" --placement-only --predicted-m "$predicted_m" \
            --h "$H" --tol "$TOL" --m-ceiling "$M_CEILING" \
            > "$report" 2>&1
        status=$?
        set -e
    fi

    operator="$({
        sed -nE \
            's/.*REAL Black-Scholes basket operator:.*N[[:space:]]*=[[:space:]]*([0-9]+).*nnz[[:space:]]*=[[:space:]]*([0-9]+).*/\1 \2/p' \
            "$report"
    } | tail -n 1)"
    measured="$({
        sed -nE \
            's/.*measured[^:]*:[[:space:]]*m[[:space:]]*=[[:space:]]*([0-9]+).*R_v[[:space:]]*=[[:space:]]*([^[:space:]]+).*R_h[[:space:]]*=[[:space:]]*([^[:space:]]+).*/\1 \2 \3/p' \
            "$report"
    } | tail -n 1)"
    predicted="$({
        sed -nE \
            's/.*predicted[^:]*:[[:space:]]*m[[:space:]]*=[[:space:]]*[0-9]+.*R_v[[:space:]]*=[[:space:]]*([^[:space:]]+).*R_h[[:space:]]*=[[:space:]]*([^[:space:]]+).*/\1 \2/p' \
            "$report"
    } | tail -n 1)"
    ratios="$({
        sed -nE \
            's/.*ratio measured\/predicted:[[:space:]]*R_v[[:space:]]*=[[:space:]]*([^[:space:]]+).*R_h[[:space:]]*=[[:space:]]*([^[:space:]]+).*/\1 \2/p' \
            "$report"
    } | tail -n 1)"

    if [[ -z "$operator" || -z "$measured" || -z "$predicted" \
          || -z "$ratios" ]]; then
        echo "Error: incomplete placement record in $report"
        echo "  matching lines were:"
        grep -E 'operator:|measured|predicted|ratio' "$report" \
            | tail -n 12 | sed 's/^/    /' || true
        failures=$((failures + 1))
        continue
    fi

    read -r N nnz <<< "$operator"
    read -r measured_m measured_rv measured_rh <<< "$measured"
    read -r predicted_rv predicted_rh <<< "$predicted"
    read -r ratio_rv ratio_rh <<< "$ratios"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$n" "$N" "$nnz" "$H" "$TOL" "$P" "$M_CEILING" \
        "$predicted_m" "$measured_m" \
        "$predicted_rv" "$measured_rv" "$ratio_rv" \
        "$predicted_rh" "$measured_rh" "$ratio_rh" \
        "$status" "$report" >> "$CSV"

    printf '  measured m=%s | R_v ratio=%s | R_h ratio=%s | status=%s\n' \
        "$measured_m" "$ratio_rv" "$ratio_rh" "$status"
    if (( status != 0 )); then
        failures=$((failures + 1))
    fi
done

echo
echo "Placement sweep complete"
echo "  results : $CSV"
echo "  reports : $REPORT_DIR"
if (( failures > 0 )); then
    echo "  failures: $failures"
    exit 1
fi
