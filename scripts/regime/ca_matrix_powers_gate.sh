#!/usr/bin/env bash
# The operator-column gate: every generated basis column against the assembled
# CPU operator, at n=31, for both options, all three recurrences, s=1 to 6.
#
# This is the gate every geometry, thread count, layout and flag change has to
# clear before it is timed, so it is a script rather than a shell loop: a wall
# of per-run banners cannot be read as a verdict, and the thing that matters is
# which points passed, which the device could not admit, and which failed.
#
# Those three outcomes are different and are kept apart. A width whose shared
# tile exceeds the device opt-in limit is a property of the geometry, reported
# by the harness as exit 2, and is not a failure. Only a numerical mismatch is.
# A configuration is gated on the widths it can actually run, and the widths it
# cannot are recorded so the dispatch range is visible rather than inferred.
#
#   ./scripts/regime/ca_matrix_powers_gate.sh
#   CONFIGURATIONS="baseline stream-8x8-t64" ./scripts/regime/ca_matrix_powers_gate.sh

set -uo pipefail

# Where the checkout is. Under sbatch this script runs as a private copy from
# /var/spool, so its own path is not a repository path and SLURM_SUBMIT_DIR is
# the authoritative one. Under an interactive salloc the opposite holds:
# SLURM_SUBMIT_DIR points at wherever salloc was invoked, which is usually not
# the checkout the caller has since changed into. Neither variable is right on
# its own, so the candidates are tried in order and the first that actually is a
# checkout wins. An explicit ROOT is honored strictly rather than falling back,
# because an escape hatch that silently goes somewhere else is worse than none.
caksm_checkout()
{
    [[ -n "$1" && -f "$1/CMakeLists.txt" && -d "$1/scripts/regime" ]]
}

CAKSM_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${ROOT:-}" ]]; then
    if ! caksm_checkout "$ROOT"; then
        echo "Error: ROOT=$ROOT is not a caksm checkout" >&2
        exit 1
    fi
else
    for candidate in "$PWD" "${SLURM_SUBMIT_DIR:-}" "$CAKSM_SCRIPT_DIR/../.."; do
        if caksm_checkout "$candidate"; then
            ROOT="$(cd -- "$candidate" && pwd)"
            break
        fi
    done
    if [[ -z "${ROOT:-}" ]]; then
        echo "Error: no caksm checkout found" >&2
        echo "  tried PWD=$PWD" >&2
        echo "  tried SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}" >&2
        echo "  tried $CAKSM_SCRIPT_DIR/../.." >&2
        echo "  run from the repository root, or export ROOT=/absolute/path/to/caksm" >&2
        exit 1
    fi
fi
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
OUT_DIR="${OUT_DIR:-$ROOT/data/ca-matrix-powers-gate}"
DEVICE="${DEVICE:-0}"
GRID="${GRID:-31}"
WIDTHS="${WIDTHS:-1 2 3 4 5 6}"
OPTIONS="${OPTIONS:-basket rainbow}"
BASES="${BASES:-monomial newton chebyshev}"
FAMILIES="${FAMILIES:-full-volume plane-streamed}"
STREAM_HEIGHT="${STREAM_HEIGHT:-8}"
# Which binaries to gate. "baseline" is ca-matrix-powers, which carries the
# default of both families; any other name is ca-matrix-powers-<name>.
CONFIGURATIONS="${CONFIGURATIONS:-baseline}"

fail()
{
    echo "Error: $*" >&2
    echo "Partial gate evidence retained in: $OUT_DIR" >&2
    exit 1
}


mkdir -p "$OUT_DIR"
ROWS="$OUT_DIR/gate_columns.csv"
SUMMARY="$OUT_DIR/gate_summary.csv"
PROVENANCE="$OUT_DIR/provenance.txt"

{
    echo "host=$(uname -n)"
    echo "date=$(date -Is)"
    echo "device=$DEVICE"
    echo "grid=$GRID"
    echo "widths=$WIDTHS"
    echo "options=$OPTIONS"
    echo "bases=$BASES"
    echo "families=$FAMILIES"
    echo "configurations=$CONFIGURATIONS"
    echo "stream_height=$STREAM_HEIGHT"
    echo "tolerance=relative 5e-11 or absolute 5e-13, as the harness applies it"
} > "$PROVENANCE"

printf '%s\n' \
    "configuration,family,option,basis,n,s,column,abs_error,rel_error" > "$ROWS"
printf '%s\n' \
    "configuration,family,option,basis,n,s,outcome,columns,worst_abs,worst_rel,log" \
    > "$SUMMARY"

passed=0
inadmissible=0
failed=0

echo "==============================================================================="
echo " CA matrix-powers operator-column gate"
echo "==============================================================================="
echo "  grid   : n=$GRID | widths: $WIDTHS"
echo "  arms   : $CONFIGURATIONS | families: $FAMILIES"
echo

for configuration in $CONFIGURATIONS; do
    binary="$BUILD_DIR/ca-matrix-powers"
    if [[ "$configuration" != "baseline" ]]; then
        binary="$BUILD_DIR/ca-matrix-powers-$configuration"
    fi
    if [[ ! -x "$binary" ]]; then
        fail "missing binary for $configuration: $binary
  The swept variants are generated, not committed. Produce them and configure
  again:
    python3 scripts/regime/ca_matrix_powers_configurations.py
    cmake -B build ... && cmake --build build --parallel"
    fi

    # A binary carries the configuration it was compiled as, so it can be asked
    # rather than assumed. This catches the case that matters most here: a build
    # directory reconfigured to a different variant list leaves the previous
    # sweep's executables on disk, and they will run perfectly well while being
    # compiled from source that has since changed.
    reported="$("$binary" --device "$DEVICE" --n 15 --s 1 --option basket \
        --basis monomial --repeats 1 2>/dev/null \
        | sed -n 's/^MPK_TUNING configuration=\([^ ]*\).*/\1/p' | tail -n 1)"
    if [[ -z "$reported" ]]; then
        fail "$binary produced no MPK_TUNING record; it predates this harness
  and must be rebuilt before its results can be recorded"
    fi
    if [[ "$reported" != "$configuration" ]]; then
        fail "$binary reports configuration '$reported', not '$configuration'
  This binary is stale. Rebuild before gating:
    cmake --build build --parallel"
    fi

    for family in $FAMILIES; do
        for option in $OPTIONS; do
            for basis in $BASES; do
                line="  ${configuration} ${family} ${option} ${basis}:"
                for width in $WIDTHS; do
                    log="$OUT_DIR/${configuration}_${family}_${option}_${basis}_n${GRID}_s${width}.txt"
                    command=(
                        "$binary" --device "$DEVICE" --n "$GRID" --s "$width"
                        --option "$option" --basis "$basis"
                        --kernel-family "$family" --repeats 1
                    )
                    if [[ "$family" == "plane-streamed" ]]; then
                        command+=(--stream-height "$STREAM_HEIGHT")
                    fi
                    "${command[@]}" > "$log" 2>&1
                    status=$?

                    if [[ $status -eq 2 ]]; then
                        outcome="not-admissible"
                        inadmissible=$((inadmissible + 1))
                    elif [[ $status -eq 0 ]]; then
                        outcome="pass"
                        passed=$((passed + 1))
                    else
                        outcome="FAIL"
                        failed=$((failed + 1))
                    fi

                    # Per-column errors, which is the form the gate is stated in.
                    awk -v configuration="$configuration" '
                        /^MPK_COLUMN / {
                            for (q = 2; q <= NF; ++q) {
                                split($q, kv, "="); field[kv[1]] = kv[2]
                            }
                            printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
                                configuration, field["family"], field["option"],
                                field["basis"], field["n"], field["s"],
                                field["column"], field["abs"], field["rel"]
                        }' "$log" >> "$ROWS"

                    read -r columns worst_abs worst_rel < <(awk '
                        /^MPK_COLUMN / {
                            for (q = 2; q <= NF; ++q) {
                                split($q, kv, "="); field[kv[1]] = kv[2]
                            }
                            ++n
                            if (field["abs"] + 0 > a) a = field["abs"] + 0
                            if (field["rel"] + 0 > r) r = field["rel"] + 0
                        }
                        END { printf "%d %.9e %.9e\n", n + 0, a + 0, r + 0 }' "$log")

                    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
                        "$configuration" "$family" "$option" "$basis" \
                        "$GRID" "$width" "$outcome" "$columns" \
                        "$worst_abs" "$worst_rel" "$(basename "$log")" \
                        >> "$SUMMARY"

                    case "$outcome" in
                        pass)           line+=" s${width}=ok" ;;
                        not-admissible) line+=" s${width}=-"  ;;
                        FAIL)           line+=" s${width}=FAIL" ;;
                    esac
                done
                echo "$line"
            done
        done
    done
done

# A recurrence that could not run at any width is not a pass. It means the arm
# has no admissible range at all, which the per-point rows would otherwise show
# only as a row of dashes.
empty="$(awk -F, '
    NR > 1 {
        key = $1 "/" $2 "/" $3 "/" $4
        total[key]++
        if ($7 == "pass") ran[key]++
    }
    END {
        for (key in total) if (!(key in ran)) print key
    }' "$SUMMARY")"

{
    echo "passed=$passed"
    echo "not_admissible=$inadmissible"
    echo "failed=$failed"
    echo "arms_with_no_admissible_width=$(wc -w <<< "$empty" | tr -d ' ')"
} >> "$PROVENANCE"

echo
echo "==============================================================================="
echo "  ok   = column-by-column agreement with the assembled operator"
echo "  -    = shared tile above the device opt-in limit at that width"
echo "  FAIL = numerical mismatch"
echo
echo "  passed         : $passed"
echo "  not admissible : $inadmissible"
echo "  failed         : $failed"
echo "  per-column CSV : $ROWS"
echo "  summary CSV    : $SUMMARY"
if [[ -n "$empty" ]]; then
    echo
    echo "  no admissible width at all:"
    printf '    %s\n' $empty
fi
echo "==============================================================================="

[[ $failed -eq 0 ]] || exit 1
