#!/usr/bin/env bash
# Where the CholQR certificate breaks on a real basis, and what the stable arms
# cost. Two questions, one sweep, because both read the same instrument.
#
#   How many block columns does the certificate actually accept, and does the
#   a-priori prediction match? The prediction comes from ca-matrix-powers,
#   which walks the analytic Vandermonde condition number over the scaled
#   Gershgorin enclosure. The measurement comes from the solver, which reports
#   the narrowest width its certificate accepted over a full integration, so
#   this is an evolving basis rather than a single start vector.
#
#   What do the stable alternatives cost? Orthogonality loss against basis time
#   for CholQR2, TSQR and BGS2 on identical blocks. TSQR and BGS2 are rejected
#   on cost while passing their stability gates, and no artifact carried those
#   two numbers together until this produced them.
#
# Run under the batch scheduler on one exclusive Synge node. The orthogonalization
# arms are compared on cost, so the basis and phi milliseconds are a result, and
# this is the recommended route:
#
#   sbatch --nodes=1 --ntasks=10 --partition=compute --time=01:00:00 \
#          --nodelist=synge-n01 --exclusive \
#          scripts/regime/ca_certificate_sweep.sh
#
# Interactively, for a smoke test only. The runs below share the calling shell's
# CPUs, so the milliseconds they produce are not recordable:
#
#   salloc -N 1 -n 10 -p compute -t 01:00:00 --nodelist=synge-n01
#   ./scripts/regime/ca_certificate_sweep.sh

set -euo pipefail

ROOT="${ROOT:-$PWD}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
SINGLE="${SINGLE:-$BUILD_DIR/ca-integrator}"
MATRIX_POWERS="${MATRIX_POWERS:-$BUILD_DIR/ca-matrix-powers}"
OUT_DIR="${OUT_DIR:-$ROOT/data/ca-integrator-certificate}"
DEVICE="${DEVICE:-0}"
GRIDS="${GRIDS:-31 61}"
OPTIONS="${OPTIONS:-basket rainbow}"
BASES="${BASES:-monomial newton chebyshev}"
ORTH_BASES="${ORTH_BASES:-monomial}"
ORTHS="${ORTHS:-cholqr2 tsqr bgs2}"
WIDTHS="${WIDTHS:-1 2 3 4 5 6}"
M="${M:-39}"
GATE_M="${GATE_M:-8}"
STEPS="${STEPS:-100}"
TOL="${TOL:-1e-8}"
REPEATS="${REPEATS:-7}"
SCALE="${SCALE:-0.01}"

for binary in "$SINGLE" "$MATRIX_POWERS"; do
    if [[ ! -x "$binary" ]]; then
        echo "Error: missing executable: $binary"
        exit 1
    fi
done

mkdir -p "$OUT_DIR"
PREDICTED="$OUT_DIR/predicted_width.csv"
MEASURED="$OUT_DIR/measured_width.csv"
ORTHOGONALIZATION="$OUT_DIR/orthogonalization.csv"

fail()
{
    echo "Error: $*" >&2
    echo "Partial logs retained for diagnosis in: $OUT_DIR" >&2
    exit 1
}

require_recordable()
{
    local log="$1"
    if grep -q 'CONTENDED DEVICE' "$log" \
       || grep -Eq '(^|[[:space:]])contended=1($|[[:space:]])' "$log"; then
        fail "contended result is not a production artifact: $log"
    fi
}

printf '%s\n' \
    "option,basis,n,recurrence_degree,block_width,predicted_block_width_max,basis_block_width_max,predicted_monomial_block_width_max,predicted_newton_block_width_max,predicted_chebyshev_block_width_max,block_basis_kappa,block_monomial_kappa,block_newton_kappa,block_chebyshev_kappa,limit,contended,log" \
    > "$PREDICTED"
printf '%s\n' \
    "option,basis,n,block_width,min_certified_block_width,fallback_blocks,first_fallback_step,max_block_kappa,avg_m,unconverged,contended,log" \
    > "$MEASURED"
printf '%s\n' \
    "option,basis,orth,n,m,s,orthogonality_loss,basis_ms,phi_ms,total_ms,basis_min_ms,basis_max_ms,phi_min_ms,phi_max_ms,total_min_ms,total_max_ms,block_kappa,contended,repeats,log" \
    > "$ORTHOGONALIZATION"

# The a-priori side: one row per (option, basis, n, block columns).
for option in $OPTIONS; do
    for basis in $BASES; do
        for n in $GRIDS; do
            for width in $WIDTHS; do
                name="predicted_${option}_${basis}_n${n}_s${width}.txt"
                log="$OUT_DIR/$name"
                final_log="$OUT_DIR/$name"
                echo "### predicted option=$option basis=$basis n=$n block_columns=$width"
                if "$MATRIX_POWERS" --device "$DEVICE" --n "$n" --s "$width" \
                        --certificate-block-width "$width" \
                        --option "$option" --basis "$basis" --scale "$SCALE" \
                        --repeats 1 > "$log" 2>&1; then
                    status=0
                else
                    status=$?
                fi
                if [[ $status -eq 2 ]]; then
                    echo "  not admissible at this width"
                    continue
                fi
                if [[ $status -ne 0 ]]; then
                    fail "prediction failed (exit $status): $log"
                fi
                require_recordable "$log"
                row="$(awk -v log_path="$final_log" '
                    /^MPK_CERTIFICATE / {
                        found = 1
                        for (q = 2; q <= NF; ++q) {
                            split($q, kv, "=")
                            field[kv[1]] = kv[2]
                        }
                        split("option basis n s block_width " \
                            "predicted_block_width_max " \
                            "basis_block_width_max " \
                            "predicted_monomial_block_width_max " \
                            "predicted_newton_block_width_max " \
                            "predicted_chebyshev_block_width_max " \
                            "block_basis_kappa block_monomial_kappa " \
                            "block_newton_kappa block_chebyshev_kappa " \
                            "limit contended",
                            required)
                        for (r in required)
                            if (field[required[r]] == "")
                                exit 4
                        if (field["s"] != field["block_width"])
                            exit 4
                        prediction_key = "predicted_" field["basis"] \
                            "_block_width_max"
                        kappa_key = "block_" field["basis"] "_kappa"
                        if (field["predicted_block_width_max"] != field[prediction_key] ||
                            field["block_basis_kappa"] != field[kappa_key] )
                            exit 4
                        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s," \
                               "%s,%s,%s,%s,%s,%s,%s\n",
                            field["option"], field["basis"], field["n"],
                            field["s"], field["block_width"],
                            field["predicted_block_width_max"],
                            field["basis_block_width_max"],
                            field["predicted_monomial_block_width_max"],
                            field["predicted_newton_block_width_max"],
                            field["predicted_chebyshev_block_width_max"],
                            field["block_basis_kappa"],
                            field["block_monomial_kappa"],
                            field["block_newton_kappa"],
                            field["block_chebyshev_kappa"],
                            field["limit"], field["contended"], log_path
                    }
                    END { if (!found) exit 3 }
                ' "$log")" || fail "missing MPK_CERTIFICATE record: $log"
                printf '%s\n' "$row" >> "$PREDICTED"
            done
        done
    done
done

# The measured side: what the certificate accepted over a full integration.
for option in $OPTIONS; do
    for basis in $BASES; do
        for n in $GRIDS; do
            for width in $WIDTHS; do
                name="measured_${option}_${basis}_n${n}_s${width}.txt"
                log="$OUT_DIR/$name"
                final_log="$OUT_DIR/$name"
                echo "### measured option=$option basis=$basis n=$n block_columns=$width"
                if "$SINGLE" --device "$DEVICE" --n "$n" --m "$M" --s "$width" \
                        --steps "$STEPS" --tol "$TOL" --repeats "$REPEATS" \
                        --option "$option" --basis "$basis" --orth cholqr2 \
                        --arm exact-depth \
                        > "$log" 2>&1; then
                    status=0
                else
                    status=$?
                fi
                if [[ $status -ne 0 ]]; then
                    # A width that reaches the predeclared Krylov cap is a
                    # stopped numerical diagnostic, not a missing artifact.
                    # Keep it so the figure can mark it stopped, but do not
                    # forgive launch/runtime failures or partial transcripts.
                    if ! grep -Eq 'unconverged=[1-9][0-9]*' "$log" \
                       || ! grep -q 'solve median:' "$log" \
                       || ! grep -Eq \
                           'validation: FAIL.*convergence=FAIL.*boundary=PASS.*referee=PASS' \
                           "$log"; then
                        fail "evolving-basis solve failed (exit $status): $log"
                    fi
                fi
                require_recordable "$log"
                row="$(awk -F'[ =|]+' -v option="$option" -v basis="$basis" \
                    -v n="$n" -v width="$width" -v log_path="$final_log" '
                    /certified width: requested=/ {
                        certificate = 1
                        for (q = 1; q <= NF; ++q) {
                            if ($q == "requested") requested = $(q + 1)
                            if ($q == "min") min_s = $(q + 1)
                            if ($q == "blocks" && $(q - 1) == "fallback")
                                fallback = $(q + 1)
                        }
                    }
                    /CA blocks:/ {
                        for (q = 1; q <= NF; ++q)
                            if ($q == "step" && $(q - 1) == "fallback")
                                first = $(q + 1)
                    }
                    /max block kappa:/ { kappa = $NF }
                    /Krylov m:/ {
                        for (q = 1; q <= NF; ++q)
                            if ($q ~ /^avg$/) avg = $(q + 1)
                        unconverged = $NF
                    }
                    END {
                        if (!certificate || requested != width ||
                            min_s == "" || fallback == "" || first == "" ||
                            kappa == "" || avg == "" || unconverged == "")
                            exit 3
                        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0,%s\n",
                            option, basis, n, width, min_s, fallback, first,
                            kappa, avg, unconverged, log_path
                    }
                ' "$log")" || fail "incomplete evolving-basis record: $log"
                printf '%s\n' "$row" >> "$MEASURED"
            done
        done
    done
done

# The conditioning comparison is only meaningful over the full Cartesian
# product, not over whatever a trimmed environment request happened to run.
# Refuse on any missing, duplicate, or out-of-domain key.
require_certificate_grid()
{
    local csv="$1"
    local width_column="$2"
    if ! awk -F, -v width_column="$width_column" '
        BEGIN {
            split("basket rainbow", options, /[[:space:]]+/)
            split("monomial newton chebyshev", bases, /[[:space:]]+/)
            split("31 61", grids, /[[:space:]]+/)
            for (o in options)
                for (b in bases)
                    for (n in grids)
                        for (width = 1; width <= 6; ++width)
                            expected[options[o] SUBSEP bases[b] SUBSEP grids[n] SUBSEP width ] = 1
        }
        NR == 1 { next }
        {
            key = $1 SUBSEP $2 SUBSEP $3 SUBSEP $(width_column)
            if (!(key in expected) || seen[key]++)
                bad = 1
        }
        END {
            for (key in expected)
                if (!(key in seen))
                    bad = 1
            if (NR - 1 != 72)
                bad = 1
            exit bad ? 1 : 0
        }
    ' "$csv"; then
        fail "certificate CSV is not the exact 72-key canonical grid: $csv"
    fi
}

require_certificate_grid "$PREDICTED" 5
require_certificate_grid "$MEASURED" 4

# The orthogonalization arms, on identical blocks: same n, m, s, basis, option.
for option in $OPTIONS; do
    for basis in $ORTH_BASES; do
        for orth in $ORTHS; do
            for n in $GRIDS; do
                name="orth_${option}_${basis}_${orth}_n${n}.txt"
                log="$OUT_DIR/$name"
                final_log="$OUT_DIR/$name"
                echo "### orth option=$option basis=$basis orth=$orth n=$n"
                if ! "$SINGLE" --device "$DEVICE" --n "$n" --m "$GATE_M" --s 4 \
                        --scale "$SCALE" --option "$option" --basis "$basis" \
                        --orth "$orth" --arm exact-depth \
                        --repeats "$REPEATS" > "$log" 2>&1; then
                    fail "orthogonalization arm failed: $log"
                fi
                require_recordable "$log"
                row="$(awk -v option="$option" -v basis="$basis" \
                    -v orth="$orth" -v n="$n" -v m="$GATE_M" \
                    -v expected_repeats="$REPEATS" -v log_path="$final_log" '
                    /^  block kappa:/ {
                        line = $0
                        sub(/^  block kappa:[[:space:]]*/, "", line)
                        gsub(/[[:space:]]+/, ";", line)
                        kappa = line
                    }
                    /orthogonality loss/ {
                        loss = $NF
                        numerical = 1
                    }
                    /^ARNOLDI_TIMING / {
                        timing = 1
                        for (q = 2; q <= NF; ++q) {
                            split($q, kv, "=")
                            field[kv[1]] = kv[2]
                        }
                    }
                    END {
                        if (!numerical || !timing ||
                            field["contended"] != 0 ||
                            field["repeats"] != expected_repeats ||
                            field["basis_median_ms"] == "")
                            exit 3
                        printf "%s,%s,%s,%s,%s,4,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0,%s,%s\n",
                            option, basis, orth, n, m, loss,
                            field["basis_median_ms"],
                            field["phi_median_ms"],
                            field["total_median_ms"],
                            field["basis_min_ms"], field["basis_max_ms"],
                            field["phi_min_ms"], field["phi_max_ms"],
                            field["total_min_ms"], field["total_max_ms"],
                            kappa, field["repeats"], log_path
                    }
                ' "$log")" || fail "incomplete orthogonalization record: $log"
                printf '%s\n' "$row" >> "$ORTHOGONALIZATION"
            done
        done
    done
done

# The three orthogonalization methods are only comparable on identical blocks,
# so a trimmed ORTHS/ORTH_BASES/GRIDS invocation is rejected rather than
# published as a plausible-looking subset.
if ! awk -F, '
    BEGIN {
        split("basket rainbow", options, /[[:space:]]+/)
        split("cholqr2 tsqr bgs2", methods, /[[:space:]]+/)
        split("31 61", grids, /[[:space:]]+/)
        for (o in options)
            for (method in methods)
                for (n in grids)
                    expected[options[o] SUBSEP "monomial" SUBSEP methods[method] SUBSEP grids[n] SUBSEP 8 SUBSEP 4 ] = 1
    }
    NR == 1 { next }
    {
        key = $1 SUBSEP $2 SUBSEP $3 SUBSEP $4 SUBSEP $5 SUBSEP $6
        if (!(key in expected) || seen[key]++)
            bad = 1
    }
    END {
        for (key in expected)
            if (!(key in seen))
                bad = 1
        if (NR - 1 != 12)
            bad = 1
        exit bad ? 1 : 0
    }
' "$ORTHOGONALIZATION"; then
    fail "orthogonalization CSV is not the exact 12-key canonical grid"
fi

# The sweep's own domain, beside its CSVs. The key counts are what the figures
# check the CSVs against, and the width unit is here because the two width
# conventions in this workstream differ by one column.
{
    echo "device=$DEVICE"
    echo "grids=$GRIDS"
    echo "options=$OPTIONS"
    echo "bases=$BASES"
    echo "orth_bases=$ORTH_BASES"
    echo "orths=$ORTHS"
    echo "widths=$WIDTHS"
    echo "m=$M"
    echo "gate_m=$GATE_M"
    echo "steps=$STEPS"
    echo "tol=$TOL"
    echo "repeats=$REPEATS"
    echo "scale=$SCALE"
    echo "integrator_arm=exact-depth"
    echo "certificate_width_units=block_columns"
    echo "prediction_recurrence_degree_equals_block_width=1"
    echo "certificate_expected_keys=72"
    echo "orthogonalization_expected_keys=12"
    echo "prediction=enclosure-sampled Vandermonde, not a spectral decomposition"
} > "$OUT_DIR/settings.txt"

echo "==============================================================================="
echo "Certificate sweep complete."
echo "  predicted        : $PREDICTED"
echo "  measured         : $MEASURED"
echo "  orthogonalization: $ORTHOGONALIZATION"
echo "  settings         : $OUT_DIR/settings.txt"
echo "Draw the figure this sweep feeds:"
echo "  uv run scripts/plots/ca_orthogonalization.py  (loss against basis time)"
echo "==============================================================================="
