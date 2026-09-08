#!/usr/bin/env bash
# Measure the exact-depth arm beside the as-measured arm at fixed local volume.
#
# A block of width s consumes columns 0 to s-1, so it needs s-1 recurrence steps
# and an s-1 deep halo. The as-measured arm ran s of each and threw the last
# column away; the exact-depth arm runs what the block consumes, which at s=1 is
# a copy with no exchange at all. Both arms run here in one job on the same
# devices, because the comparison between them is the finding.
#
# Nothing is overwritten: the as-measured transcripts under
# data/ca-integrator-weak are untouched, and everything written here lands under
# data/ca-integrator-exact-depth.
#
# Weak scaling, so the grid grows with the participant count to hold the local
# volume fixed at about LOCAL_ROWS rows per GPU. The rungs are derived from what
# the allocation holds rather than written out by hand: n = (LOCAL_ROWS * P)^(1/3)
# reproduces n=61 at one participant, 77 at two, 97 at four and 122 at eight.
# EXACT_RUNGS overrides the derivation with an explicit "<n>:<nodes>x<local>"
# list, which is what a publication run should pass.
#
# A rung is admitted only if its validation artifacts exist, since the sweep
# accepts every point against a single-GPU state and a referee. Derived rungs
# beyond the measured grids have neither, so they are named and skipped rather
# than failing the whole sweep.
#
# Run under the batch scheduler on an exclusive allocation. The gap between the
# two arms is the finding, so this is the recommended route:
#
#   sbatch --nodes=2 --ntasks=2 --ntasks-per-node=1 --exclusive \
#          --time=02:00:00 scripts/regime/ca_exact_depth_weak.sh
#
# On a single node of eight, with the machine naming its own output directory:
#
#   sbatch --nodes=1 --gpus=8 --ntasks-per-node=1 --exclusive \
#          --time=02:00:00 --export=ALL,MACHINE=h200 \
#          scripts/regime/ca_exact_depth_weak.sh
#
# Interactively, for a smoke test only. SLURM_OVERLAP shares the calling shell's
# CPUs with every step below, so its timings can vary:
#
#   salloc -N 2 -n 2 -t 02:00:00
#   SLURM_OVERLAP=1 ./scripts/regime/ca_exact_depth_weak.sh
#
# The integrator withholds its timing distribution if any participating device is
# contended, so a contended run produces correctness evidence and no timings.

set -euo pipefail

ROOT="${ROOT:-$PWD}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
DISTRIBUTED="${DISTRIBUTED:-$BUILD_DIR/ca-integrator-multigpu}"
SINGLE="${SINGLE:-$BUILD_DIR/ca-integrator}"
VALIDATION_DIR="${VALIDATION_DIR:-$ROOT/data/ca-integrator-validation}"
REFEREE_ROOT="${REFEREE_ROOT:-$ROOT/data}"
M="${M:-39}"
# The machine names the output directory, so two installations never merge into
# one sweep, and m goes in it because it is a build constant: sweeps taken at
# different ceilings are not the same experiment.
MACHINE="${MACHINE:-v100-pcie-16gb}"
OUT_DIR="${OUT_DIR:-$ROOT/data/ca-integrator-exact-depth-$MACHINE-m$M}"
FORCE="${FORCE:-0}"
# Rows per GPU the weak ladder holds fixed. 227,000 is what the as-measured arm
# reported, and reproduces its n=77 at two participants and n=97 at four.
LOCAL_ROWS="${LOCAL_ROWS:-227000}"
EXACT_NODES="${EXACT_NODES:-}"
EXACT_LOCAL_GPUS="${EXACT_LOCAL_GPUS:-}"
EXACT_RUNGS="${EXACT_RUNGS:-}"
SKIPPED_POINTS=0
STEPS="${STEPS:-100}"
TOL="${TOL:-1e-8}"
REPEATS="${REPEATS:-7}"
SRUN_MPI="${SRUN_MPI:-pmix}"
STATE_RTOL="${STATE_RTOL:-1e-10}"

for binary in "$DISTRIBUTED" "$SINGLE"; do
    if [[ ! -x "$binary" ]]; then
        echo "Error: missing executable: $binary"
        exit 1
    fi
done
if ! command -v srun >/dev/null 2>&1; then
    echo "Error: srun is required."
    exit 1
fi
# Slurm names the allocation's node count differently depending on how this
# shell was obtained: sbatch and srun set SLURM_JOB_NUM_NODES, salloc sets
# SLURM_NNODES, and inside an interactive `srun --pty` step both report that
# step's size rather than the allocation's. The nodelist always describes the
# allocation, so that is counted first and fall back to the variables.
allocated_nodes()
{
    if [[ -n "${SLURM_JOB_NODELIST:-}" ]] \
       && command -v scontrol >/dev/null 2>&1; then
        scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null \
            | wc -l | tr -d '[:space:]'
    else
        echo "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-0}}"
    fi
}

report_allocation()
{
    echo "  SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<unset>}" >&2
    echo "  SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-<unset>}" \
         "SLURM_NNODES=${SLURM_NNODES:-<unset>}" >&2
    if [[ -n "${SLURM_STEP_ID:-}" ]]; then
        echo "  SLURM_STEP_ID=${SLURM_STEP_ID}: this shell is itself an srun" >&2
        echo "  step, which cannot launch the wider steps this sweep needs." >&2
        echo "  Leave it and run from the salloc shell owning the allocation." >&2
    fi
}

if [[ -z "$EXACT_NODES" ]]; then
    EXACT_NODES="$(allocated_nodes)"
fi
if [[ "$EXACT_NODES" -lt 1 ]]; then
    echo "Error: no Slurm allocation found; every rung here is launched with srun." >&2
    report_allocation
    exit 1
fi
if [[ -z "$EXACT_LOCAL_GPUS" ]]; then
    if [[ -n "${SLURM_GPUS_PER_NODE:-}" ]]; then
        EXACT_LOCAL_GPUS="${SLURM_GPUS_PER_NODE##*:}"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        EXACT_LOCAL_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    else
        EXACT_LOCAL_GPUS=1
    fi
fi
[[ "$EXACT_NODES" =~ ^[0-9]+$ && "$EXACT_NODES" -ge 1 ]] \
    || { echo "Error: EXACT_NODES must be a positive integer, got '$EXACT_NODES'" >&2; exit 1; }
[[ "$EXACT_LOCAL_GPUS" =~ ^[0-9]+$ && "$EXACT_LOCAL_GPUS" -ge 1 ]] \
    || { echo "Error: EXACT_LOCAL_GPUS must be a positive integer, got '$EXACT_LOCAL_GPUS'" >&2; exit 1; }
if [[ $(( EXACT_NODES * EXACT_LOCAL_GPUS )) -lt 2 ]]; then
    echo "Error: this sweep compares two arms on a distributed decomposition," >&2
    echo "  which needs at least two participants; the allocation holds one." >&2
    exit 1
fi

if [[ -d "$OUT_DIR" ]] && compgen -G "$OUT_DIR/*.txt" >/dev/null \
   && [[ "$FORCE" != "1" ]]; then
    echo "Error: $OUT_DIR already holds transcripts." >&2
    echo "  A sweep overwritten in place cannot be told from the one it replaced." >&2
    echo "  Write elsewhere with OUT_DIR=..., or pass FORCE=1 to overwrite." >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
export PMIX_MCA_gds="${PMIX_MCA_gds:-hash}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# The powers of two up to a bound: the participant counts that separate one
# rung from the next.
exact_ladder() {   # <max> -> "1 2 4 ... <= max"
    local max="$1" k=1 out=()
    while (( k <= max )); do out+=("$k"); k=$((k * 2)); done
    echo "${out[*]}"
}

# The grid that holds LOCAL_ROWS rows on each of P participants, rounded to the
# nearest whole grid. Weak scaling in three dimensions, so n grows as P^(1/3).
weak_grid() {   # <participants> -> n
    awk -v rows="$LOCAL_ROWS" -v p="$1" \
        'BEGIN { printf "%d", int(exp(log(rows * p) / 3.0) + 0.5) }'
}

# Every participant count the allocation can present, as "<n>:<nodes>x<local>".
# Nodes are used before local devices, so a two-node allocation reaches two
# participants as 2x1 rather than 1x2, which is what the recorded sweep did.
if [[ -z "$EXACT_RUNGS" ]]; then
    derived=()
    for participants in $(exact_ladder $(( EXACT_NODES * EXACT_LOCAL_GPUS ))); do
        [[ "$participants" -lt 2 ]] && continue
        rung_nodes="$EXACT_NODES"
        [[ "$participants" -lt "$rung_nodes" ]] && rung_nodes="$participants"
        rung_local=$(( participants / rung_nodes ))
        [[ $(( rung_nodes * rung_local )) -eq "$participants" ]] || continue
        [[ "$rung_local" -le "$EXACT_LOCAL_GPUS" ]] || continue
        derived+=("$(weak_grid "$participants"):${rung_nodes}x${rung_local}")
    done
    EXACT_RUNGS="${derived[*]}"
fi

RUNGS=($EXACT_RUNGS)
STOPPED_POINTS=0
ACCEPTANCE_FAILURES=0
CONTENDED_POINTS=0
INCOMPLETE_POINTS=0

# A rung is admitted only with both acceptance artifacts. A derived rung beyond
# the measured grids has neither, which is the expected case at the wide end, so
# it is named and dropped rather than failing every rung that does have them.
ADMITTED=()
for rung in "${RUNGS[@]}"; do
    n="${rung%%:*}"
    missing=""
    for option in basket rainbow; do
        single_state="$VALIDATION_DIR/single_n${n}_${option}.bin"
        referee="$REFEREE_ROOT/n${n}/referee_n${n}_${option}.bin"
        [[ -f "$single_state" ]] || missing="$missing $single_state"
        [[ -f "$referee" ]]      || missing="$missing $referee"
    done
    if [[ -n "$missing" ]]; then
        {
            echo "SKIPPED rung $rung: no acceptance artifacts for n=$n"
            for path in $missing; do echo "  missing $path"; done
        } | tee -a "$OUT_DIR/skipped.txt"
        SKIPPED_POINTS=$((SKIPPED_POINTS + 1))
        continue
    fi
    ADMITTED+=("$rung")
done
if [[ ${#ADMITTED[@]} -eq 0 ]]; then
    echo "Error: no rung has both acceptance artifacts; see $OUT_DIR/skipped.txt" >&2
    echo "  Generate them with scripts/regime/ca_integrator_acceptance.sh" >&2
    exit 1
fi
RUNGS=("${ADMITTED[@]}")

echo "==============================================================================="
echo " Exact-depth correction sweep   machine=$MACHINE  m=$M"
echo " host=$(uname -n)  job=${SLURM_JOB_ID:-none}"
echo "==============================================================================="
echo "  alloc : $EXACT_NODES node(s) x $EXACT_LOCAL_GPUS GPU(s)"
echo "  rungs : ${RUNGS[*]}   (<n>:<nodes>x<local GPUs>)"
echo "  output: $OUT_DIR"
echo

# The sweep's own certificate decisions are only trustworthy if a cross-rank
# disagreement would actually stop the solver, so a disagreement is injected
# first and the named abort is required before any rung is admitted.
run_agreement_self_test() {
    local log="$OUT_DIR/agreement_self_test.txt"

    echo "### cross-rank agreement self-test (expected certificate disagreement)"
    set +e
    srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --mpi=none \
        "$DISTRIBUTED" --devices 0,1 \
        --n 31 --m 8 --s 4 --steps 1 --tol "$TOL" --repeats 1 \
        --option rainbow --basis monomial --orth cholqr2 \
        --arm exact-depth --agreement-self-test \
        2>&1 | tee "$log"
    local exit_code="${PIPESTATUS[0]}"
    set -e

    if [[ $exit_code -eq 0 ]] \
       || ! grep -q \
           'ranks disagreed at the block certificate checkpoint on certificate acceptance' \
           "$log"; then
        echo "AGREEMENT_SELF_TEST status=failed exit_code=$exit_code" \
            | tee -a "$log"
        echo "Error: the agreement self-test did not stop at the named" >&2
        echo "       certificate field, so a real cross-rank disagreement" >&2
        echo "       would not be caught either." >&2
        return 1
    fi
    echo "AGREEMENT_SELF_TEST status=passed expected_exit_code=$exit_code" \
        | tee -a "$log"
    echo
}

record_run_status() {
    local log="$1"
    local exit_code="$2"
    local label="$3"
    local status="passed"

    if grep -q 'CONTENDED DEVICE' "$log" \
       || grep -q 'recordable=no' "$log"; then
        status="contended"
        CONTENDED_POINTS=$((CONTENDED_POINTS + 1))
    elif [[ $exit_code -ne 0 ]]; then
        if grep -Eq \
                'validation: FAIL.*(boundary|single|referee)=FAIL' "$log"; then
            status="acceptance-fail"
            ACCEPTANCE_FAILURES=$((ACCEPTANCE_FAILURES + 1))
        elif grep -Eq 'unconverged=[1-9][0-9]*' "$log"; then
            status="stopped"
            STOPPED_POINTS=$((STOPPED_POINTS + 1))
        elif grep -q 'validation: FAIL' "$log"; then
            status="acceptance-fail"
            ACCEPTANCE_FAILURES=$((ACCEPTANCE_FAILURES + 1))
        else
            {
                echo "RUN_STATUS status=execution-fail exit_code=$exit_code label=$label"
                echo "Unexpected execution failure; aborting the sweep."
            } | tee -a "$log"
            return "$exit_code"
        fi
    elif ! grep -Eq "distribution: .*\\(${REPEATS} runs\\)" "$log"; then
        status="incomplete"
        INCOMPLETE_POINTS=$((INCOMPLETE_POINTS + 1))
    fi

    echo "RUN_STATUS status=$status exit_code=$exit_code label=$label" \
        | tee -a "$log"
    return 0
}

# The thinnest slab a decomposition produces, against the halo depth an arm
# needs. z_begin = rank * n / P, so the thinnest slab holds floor(n / P) planes,
# and the halo exchange sends the neighbour's last `depth` owned planes from
# input + (z_count - depth) * n^2. A slab thinner than the halo makes that
# offset negative and reads before the start of its own buffer, which is a
# memory error rather than a numerical one, so the point is refused.
thinnest_slab() {   # <n> <participants>
    echo $(( $1 / $2 ))
}

# A block of width s consumes columns 0..s-1, so it needs an s-1 deep halo. The
# as-measured arm exchanges s and discards the extra column.
halo_depth() {      # <arm> <width>
    case "$1" in
        exact-depth) echo $(( $2 - 1 )) ;;
        *)           echo "$2" ;;
    esac
}

run_point() {
    local arm="$1"
    local n="$2"
    local nodes="$3"
    local local_gpus="$4"
    local option="$5"
    local width="$6"
    local gpus=$(( nodes * local_gpus ))
    local log="$OUT_DIR/${arm}_n${n}_${option}_${gpus}gpu_${nodes}node_s${width}.txt"

    local depth thinnest
    depth="$(halo_depth "$arm" "$width")"
    thinnest="$(thinnest_slab "$n" "$gpus")"
    if [[ "$depth" -gt "$thinnest" ]]; then
        printf 'SKIPPED arm=%s n=%s %sgpu_%snode s=%s: halo depth %s exceeds the thinnest slab (%s planes)\n' \
            "$arm" "$n" "$gpus" "$nodes" "$width" "$depth" "$thinnest" \
            | tee -a "$OUT_DIR/skipped.txt"
        SKIPPED_POINTS=$((SKIPPED_POINTS + 1))
        echo
        return 0
    fi

    local single_state="$VALIDATION_DIR/single_n${n}_${option}.bin"
    local gate=(
        --single-gpu-state "$single_state"
        --single-state-rtol "$STATE_RTOL"
        --referee-dir "$REFEREE_ROOT/n${n}"
    )

    # The devices each rank drives, built in the shell rather than with
    # `seq -s,`: BSD seq appends the separator after the last element and GNU
    # seq does not, and the integrator rejects the empty token that leaves.
    local -a device_list=()
    local d
    for (( d = 0; d < local_gpus; ++d )); do device_list+=("$d"); done
    local devices
    devices="$(IFS=,; echo "${device_list[*]}")"

    # One array rather than an optional one expanded: "${pin[@]}" on an empty
    # array is an unbound-variable error under `set -u` in bash 3.2.
    local -a launcher=(srun)
    local -a mpi_flag=()
    if [[ "$nodes" -eq 1 ]]; then
        launcher+=(--nodes=1 --ntasks=1 --ntasks-per-node=1 --mpi=none)
    else
        launcher+=(--nodes="$nodes" --ntasks="$nodes" --ntasks-per-node=1
                   --mpi="$SRUN_MPI")
        mpi_flag=(--require-mpi)
    fi

    echo "### arm=$arm n=$n gpus=$gpus nodes=$nodes option=$option s=$width"
    set +e
    "${launcher[@]}" \
        "$DISTRIBUTED" "${mpi_flag[@]+"${mpi_flag[@]}"}" \
        --devices "$devices" \
        --n "$n" --m "$M" --s "$width" --steps "$STEPS" --tol "$TOL" \
        --repeats "$REPEATS" --option "$option" \
        --basis monomial --orth cholqr2 --arm "$arm" \
        "${gate[@]}" 2>&1 | tee "$log"
    local exit_code="${PIPESTATUS[0]}"
    set -e
    record_run_status \
        "$log" "$exit_code" \
        "arm=$arm,n=$n,gpus=$gpus,nodes=$nodes,option=$option,s=$width"
    echo
}

# The one-GPU reference each rung's efficiency is measured against. Recorded per
# arm, because the arm changes the reference as well as the distributed point.
run_reference() {
    local arm="$1"
    local n="$2"
    local option="$3"
    local width="$4"
    local log="$OUT_DIR/${arm}_reference_n${n}_${option}_s${width}.txt"

    echo "### arm=$arm reference n=$n option=$option s=$width on one V100"
    set +e
    srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --mpi=none \
        "$SINGLE" \
        --n "$n" --m "$M" --s "$width" --steps "$STEPS" --tol "$TOL" \
        --repeats "$REPEATS" --option "$option" \
        --basis monomial --orth cholqr2 --arm "$arm" \
        --referee-dir "$REFEREE_ROOT/n${n}" \
        2>&1 | tee "$log"
    local exit_code="${PIPESTATUS[0]}"
    set -e
    record_run_status \
        "$log" "$exit_code" \
        "arm=$arm,reference=one-gpu,n=$n,option=$option,s=$width"
    echo
}

run_agreement_self_test

for arm in as-measured exact-depth; do
    for rung in "${RUNGS[@]}"; do
        n="${rung%%:*}"
        shape="${rung##*:}"
        rung_nodes="${shape%x*}"
        rung_local="${shape#*x}"
        for option in basket rainbow; do
            for width in 1 4; do
                run_point "$arm" "$n" "$rung_nodes" "$rung_local" \
                    "$option" "$width"
                run_reference "$arm" "$n" "$option" "$width"
            done
        done
    done
done

# What the sweep was asked for and what it ran into, kept beside the transcripts
# so the classified point counts do not have to be recounted by hand.
{
    echo "machine=$MACHINE"
    echo "arms=as-measured exact-depth"
    echo "rungs=${RUNGS[*]}"
    echo "nodes=$EXACT_NODES"
    echo "local_gpus=$EXACT_LOCAL_GPUS"
    echo "local_rows=$LOCAL_ROWS"
    echo "skipped_points=$SKIPPED_POINTS"
    echo "m=$M"
    echo "steps=$STEPS"
    echo "tol=$TOL"
    echo "single_state_rtol=$STATE_RTOL"
    echo "repeats=$REPEATS"
    echo "stopped_points=$STOPPED_POINTS"
    echo "acceptance_failures=$ACCEPTANCE_FAILURES"
    echo "contended_points=$CONTENDED_POINTS"
    echo "incomplete_points=$INCOMPLETE_POINTS"
    echo "agreement_self_test=passed"
} > "$OUT_DIR/settings.txt"

# Provenance and checksums beside the transcripts, in the same shape the
# calibration job writes: a cycle time whose build cannot be identified cannot
# be compared against another sweep's.
{
    echo "machine=$MACHINE"
    echo "host=$(uname -n)"
    echo "kernel=$(uname -sr)"
    echo "date=$(date -Is 2>/dev/null || date -u '+%Y-%m-%dT%H:%M:%S+00:00')"
    echo "slurm_job=${SLURM_JOB_ID:-none}"
    echo "slurm_nodelist=${SLURM_JOB_NODELIST:-none}"
    echo "nodes=$EXACT_NODES"
    echo "local_gpus=$EXACT_LOCAL_GPUS"
    echo "rungs=${RUNGS[*]}"
    echo "single_binary=$SINGLE"
    echo "distributed_binary=$DISTRIBUTED"
    echo "git_revision=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    if git -C "$ROOT" diff --quiet 2>/dev/null \
            && git -C "$ROOT" diff --cached --quiet 2>/dev/null; then
        echo "git_tracked_dirty=0"
    else
        echo "git_tracked_dirty=1"
    fi
} > "$OUT_DIR/provenance.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi topo -m                                  > "$OUT_DIR/topology.txt" 2>&1
    nvidia-smi nvlink -s                                > "$OUT_DIR/nvlink.txt"   2>&1
    nvidia-smi --query-gpu=index,uuid,name --format=csv > "$OUT_DIR/devices.csv"  2>&1
fi

# find rather than a glob: the glob would hand the tool a directory, which makes
# it exit non-zero and leaves the manifest as a .tmp nobody notices.
if command -v sha256sum >/dev/null 2>&1; then
    CHECKSUM_TOOL="sha256sum"
else
    CHECKSUM_TOOL="shasum -a 256"
fi
( cd "$OUT_DIR" \
  && find . -type f ! -name 'SHA256SUMS*' | LC_ALL=C sort \
     | xargs $CHECKSUM_TOOL > SHA256SUMS.tmp \
  && mv SHA256SUMS.tmp SHA256SUMS ) || {
    echo "Warning: could not write $OUT_DIR/SHA256SUMS" >&2
    rm -f "$OUT_DIR/SHA256SUMS.tmp"
}

echo "==============================================================================="
echo "Both arms attempted at every rung."
echo "  transcripts : $OUT_DIR"
echo "  settings    : $OUT_DIR/settings.txt"
echo "  stopped     : $STOPPED_POINTS predeclared numerical point(s)"
echo "  acceptance  : $ACCEPTANCE_FAILURES failed correction bound(s)"
echo "  contended   : $CONTENDED_POINTS timing-withheld diagnostic point(s)"
echo "  incomplete  : $INCOMPLETE_POINTS point(s) without the requested distribution"
if [[ "$SKIPPED_POINTS" -ne 0 ]]; then
    echo "  skipped     : $SKIPPED_POINTS point(s) or rung(s); see $OUT_DIR/skipped.txt"
fi
echo "Draw the figures this sweep feeds:"
echo "  uv run scripts/plots/ca_correction_impact.py   (the two arms, side by side)"
echo "  uv run scripts/plots/ca_cycle_decomposition.py (where the cycle goes)"
echo "  uv run scripts/plots/ca_predicted_measured.py  (predicted against measured)"
echo "==============================================================================="

if [[ $ACCEPTANCE_FAILURES -ne 0 || $CONTENDED_POINTS -ne 0 \
      || $INCOMPLETE_POINTS -ne 0 ]]; then
    exit 1
fi
