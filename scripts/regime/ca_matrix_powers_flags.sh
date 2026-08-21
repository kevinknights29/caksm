#!/usr/bin/env bash
# Screen matrix-powers compilation flags by device code before timing any of them.
#
# The rule this enforces: a flag that produces the same SASS and the same
# resource report as the baseline is a no-op, and timing it would only measure
# run-to-run noise and then attribute the noise to the flag. So every arm is
# disassembled first, its matrix-powers kernels are normalized and hashed, and
# only the arms whose device code actually differs are run.
#
# The disassembly is normalized before hashing because the raw listing carries
# instruction addresses and the mangled names of the specific cubin, neither of
# which is a code difference. What survives is the opcode and operand stream.
#
# Arms are built by caksm_add_matrix_powers_variant in CMakeLists.txt, on the
# baseline geometry, so a difference here is attributable to the flag and not
# to a shape. This screen consumes binaries supplied by a separate build and
# fails explicitly if any required arm is absent.
#
# Two arms the specification asks about are answered here rather than built.
# The -O flags are host optimization by nvcc's own definition, so the question
# is settled by --dryrun and the cubin hash rather than by a timing run; and
# device LTO has no cross-file device code to combine, because each executable
# is one translation unit over header-only code.

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
OUT_DIR="${OUT_DIR:-$ROOT/data/ca-matrix-powers-flags}"
DEVICE="${DEVICE:-0}"
CUOBJDUMP="${CUOBJDUMP:-cuobjdump}"
REPEATS="${REPEATS:-7}"
BASELINE="${BASELINE:-ca-matrix-powers}"
ARMS="${ARMS:-vectorized cache-ca cache-cg fastmath regcap96 regcap80 regcap72 regcap64}"
# The points a differing arm is timed at: the four canonical selection cases.
CASES="${CASES:-basket:61:3 rainbow:61:3 basket:97:4 rainbow:97:4}"
KERNEL_FILTER="${KERNEL_FILTER:-mpk_}"

fail()
{
    echo "Error: $*" >&2
    echo "Partial flag evidence retained in: $OUT_DIR" >&2
    exit 1
}

command -v "$CUOBJDUMP" >/dev/null 2>&1 \
    || fail "cuobjdump not found: $CUOBJDUMP"

mkdir -p "$OUT_DIR"
ROWS="$OUT_DIR/flag_screen.csv"
TIMINGS="$OUT_DIR/flag_timings.csv"
PROVENANCE="$OUT_DIR/provenance.txt"

{
    echo "host=$(uname -n)"
    echo "date=$(date -Is)"
    echo "slurm_job_id=${SLURM_JOB_ID:-none}"
    echo "device=$DEVICE"
    echo "cuobjdump=$(command -v "$CUOBJDUMP")"
    echo "baseline=$BASELINE"
    echo "arms=$ARMS"
    echo "cases=$CASES"
    echo "repeats=$REPEATS"
    echo "kernel_filter=$KERNEL_FILTER"
    echo "normalization=strip addresses, hex literals and section headers"
    echo "device_lto=not built; one translation unit per executable"
    echo "host_optimization=answered by nvcc --dryrun and the cubin hash"
} > "$PROVENANCE"

# The disassembly of one binary's matrix-powers kernels, normalized to the parts
# a flag can legitimately change. Addresses, the /*hhhh*/ offset comments, and
# the encoded instruction words are all position-dependent, so they are removed:
# two identical kernels laid out at different addresses must hash the same.
normalized_sass()
{
    local binary="$1"
    "$CUOBJDUMP" -sass "$binary" 2>/dev/null \
        | awk -v filter="$KERNEL_FILTER" '
            /Function : / { keep = ($0 ~ filter) }
            keep && (/Function : / || /^[[:space:]]+\/\*[0-9a-fA-F]+\*\//) {
                print
            }' \
        | sed -E 's@/\*[0-9a-f]+\*/@@g; s@/\* 0x[0-9a-f]+ \*/@@g' \
        | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//' \
        | sed -E 's/[[:space:]]+/ /g' \
        | grep -v '^$'
}

hash_of()
{
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum | awk '{ print $1 }'
    else
        shasum -a 256 | awk '{ print $1 }'
    fi
}

# Registers, spills and code size, from the same disassembly rather than from a
# build log that may be from a different configure. ptxas reports spill bytes at
# compile time; what a stripped binary still carries is the register count and
# the stack frame, so those are what this records.
resource_report()
{
    local binary="$1"
    "$CUOBJDUMP" -res-usage "$binary" 2>/dev/null \
        | grep -A6 -E "$KERNEL_FILTER" \
        | sed -E 's/^[[:space:]]+//' \
        | grep -E "REG:|STACK:|SHARED:|LOCAL:|Function" \
        | tr '\n' ';'
}

printf '%s\n' \
    "arm,binary,device_code,sass_hash,sass_lines,registers,resource_report,decision" \
    > "$ROWS"

baseline_binary="$BUILD_DIR/$BASELINE"
[[ -x "$baseline_binary" ]] || fail "missing baseline binary: $baseline_binary"

baseline_sass="$OUT_DIR/${BASELINE}.sass"
normalized_sass "$baseline_binary" > "$baseline_sass"
[[ -s "$baseline_sass" ]] \
    || fail "no matrix-powers SASS found in $baseline_binary; check KERNEL_FILTER"
baseline_instructions="$(grep -vc '^Function : ' "$baseline_sass")"
[[ "$baseline_instructions" -gt 0 ]] \
    || fail "the normalized SASS contains kernel names but no instructions"
baseline_hash="$(hash_of < "$baseline_sass")"
baseline_lines="$(wc -l < "$baseline_sass" | tr -d ' ')"
baseline_resources="$(resource_report "$baseline_binary")"
baseline_registers="$(grep -oE 'REG:[0-9]+' <<< "$baseline_resources" \
    | head -n 1 | cut -d: -f2)"

printf '%s,%s,%s,%s,%s,%s,"%s",%s\n' \
    "baseline" "$BASELINE" "reference" "$baseline_hash" "$baseline_lines" \
    "${baseline_registers:-unknown}" "$baseline_resources" "reference" \
    >> "$ROWS"

echo "==============================================================================="
echo " CA matrix-powers compilation-flag screen"
echo "==============================================================================="
echo "  baseline : $BASELINE (${baseline_instructions} instructions, ${baseline_registers:-unknown} registers)"
echo "  arms     : $ARMS"
echo

DIFFERING=()
for arm in $ARMS; do
    binary="$BUILD_DIR/ca-matrix-powers-$arm"
    if [[ ! -x "$binary" ]]; then
        printf '%s,%s,%s,,,,"",%s\n' \
            "$arm" "ca-matrix-powers-$arm" "not-built" "not screened" >> "$ROWS"
        echo "  $arm: binary not built; not screened"
        continue
    fi
    arm_sass="$OUT_DIR/${arm}.sass"
    normalized_sass "$binary" > "$arm_sass"
    arm_hash="$(hash_of < "$arm_sass")"
    arm_lines="$(wc -l < "$arm_sass" | tr -d ' ')"
    arm_resources="$(resource_report "$binary")"
    arm_registers="$(grep -oE 'REG:[0-9]+' <<< "$arm_resources" \
        | head -n 1 | cut -d: -f2)"

    if [[ "$arm_hash" == "$baseline_hash" \
          && "$arm_resources" == "$baseline_resources" ]]; then
        device_code="identical"
        decision="no-op; not timed"
    else
        device_code="different"
        decision="timed"
        DIFFERING+=("$arm")
        diff -u "$baseline_sass" "$arm_sass" > "$OUT_DIR/${arm}.sass.diff" \
            2>/dev/null || true
    fi
    printf '%s,%s,%s,%s,%s,%s,"%s",%s\n' \
        "$arm" "ca-matrix-powers-$arm" "$device_code" "$arm_hash" \
        "$arm_lines" "${arm_registers:-unknown}" "$arm_resources" \
        "$decision" >> "$ROWS"
    echo "  $arm: $device_code (${arm_lines} lines, ${arm_registers:-unknown} registers) -> $decision"
done

echo
if [[ ${#DIFFERING[@]} -eq 0 ]]; then
    echo "No arm changed device code. Nothing to time, which is the result."
    echo "  screen : $ROWS"
    exit 0
fi

printf '%s\n' \
    "arm,option,n,s,status,median_ms,min_ms,max_ms,registers,worst_rel,log" \
    > "$TIMINGS"

time_point()
{
    local arm="$1" binary="$2" option="$3" n="$4" width="$5"
    local log="$OUT_DIR/time_${arm}_${option}_n${n}_s${width}.txt"
    "$binary" --device "$DEVICE" --n "$n" --s "$width" --option "$option" \
        --basis monomial --repeats "$REPEATS" > "$log" 2>&1
    local status=$?
    if [[ $status -ne 0 ]]; then
        printf '%s,%s,%s,%s,failed,,,,,,%s\n' \
            "$arm" "$option" "$n" "$width" "$(basename "$log")" >> "$TIMINGS"
        echo "  $arm $option n=$n s=$width: exited $status"
        return
    fi
    local row
    row="$(awk -v arm="$arm" -v log_record="$(basename "$log")" '
        /^MPK_TUNING / {
            for (q = 2; q <= NF; ++q) { split($q, kv, "="); field[kv[1]] = kv[2] }
            printf "%s,%s,%s,%s,measured,%s,%s,%s,%s,%s,%s\n",
                arm, field["option"], field["n"], field["s"],
                field["median_ms"], field["min_ms"], field["max_ms"],
                field["registers_per_thread"], field["worst_rel"], log_record
        }' "$log")"
    printf '%s\n' "$row" >> "$TIMINGS"
    echo "  $arm $option n=$n s=$width: $(awk -F, '{ print $6 }' <<< "$row") ms"
}

echo "Timing the ${#DIFFERING[@]} arm(s) whose device code differs."
for case in $CASES; do
    IFS=: read -r option n width <<< "$case"
    # The baseline is re-timed inside the same loop, so drift over the run lands
    # on the reference as well as on the arms.
    time_point "baseline" "$baseline_binary" "$option" "$n" "$width"
    for arm in "${DIFFERING[@]}"; do
        time_point "$arm" "$BUILD_DIR/ca-matrix-powers-$arm" \
            "$option" "$n" "$width"
    done
done

{
    echo "differing_arms=${DIFFERING[*]}"
    echo "timed_arms=${#DIFFERING[@]}"
} >> "$PROVENANCE"

echo
echo "==============================================================================="
echo "Flag screen complete."
echo "  screen     : $ROWS"
echo "  timings    : $TIMINGS"
echo "  provenance : $PROVENANCE"
echo "A fast-math arm that reaches this point still needs the numerical gate:"
echo "its prices, Greeks and basis columns must be compared before promotion."
echo "==============================================================================="
