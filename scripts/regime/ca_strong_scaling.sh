#!/usr/bin/env bash
# Strong scaling at fixed global problem size, with the transcripts kept.
#
# One solver run per (arm, option, block width, arrangement) at fixed n. An
# arrangement is <nodes>x<local GPUs>, and the sweep is derived from what the
# allocation holds rather than written out by hand: the powers of two of both
# give 1x1, 1x2, 2x1 and 2x2 on a two-GPU pair of nodes, and 1, 2, 4 and 8
# participants on an eight-GPU node, with no edit. STRONG_ARMS overrides the
# derivation with an explicit list, which is what a publication run should pass
# so it repeats the same arrangements every time.
#
# The one-node arrangement at the same participant count as a two-node one earns
# its place by separating transport from participant count: same participants,
# different interconnect, and one rank driving N local devices rather than N
# ranks driving one each.
#
# Every rung is validated against the single-GPU state, so 1x1 must be in the
# sweep: it is both the ladder's denominator and the reference the distributed
# arms are checked against.
#
# Run under the batch scheduler on an exclusive allocation. --exclusive is not
# optional: a co-tenant shares the L2 these cycle times are measured against,
# the contention gate then discards the run, and the arm loses its whole line.
#
#   sbatch --nodes=2 --ntasks=2 --ntasks-per-node=1 --exclusive \
#          --time=04:00:00 scripts/regime/ca_strong_scaling.sh
#
# On a single node of eight, which sweeps 1, 2, 4 and 8 participants:
#
#   sbatch --nodes=1 --gpus=8 --ntasks-per-node=1 --exclusive \
#          --time=04:00:00 --export=ALL,MACHINE=h200 \
#          scripts/regime/ca_strong_scaling.sh
#
# Interactively, for a smoke test only. SLURM_OVERLAP shares the calling shell's
# CPUs with every step below, so its timings can vary:
#
#   salloc -N 2 -n 2 -t 01:00:00
#   SLURM_OVERLAP=1 ./scripts/regime/ca_strong_scaling.sh

set -euo pipefail

ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
SINGLE="${SINGLE:-$BUILD_DIR/ca-integrator}"
DISTRIBUTED="${DISTRIBUTED:-$BUILD_DIR/ca-integrator-multigpu}"
REFEREE_ROOT="${REFEREE_ROOT:-$ROOT/data}"

# The machine names the output directory, so two installations never merge into
# one ladder. Nominally identical nodes differ materially in collective latency,
# and a mixed directory would put that difference inside the participant axis.
MACHINE="${MACHINE:-v100-pcie-16gb}"
N="${N:-61}"
M="${M:-39}"
STEPS="${STEPS:-100}"
TOL="${TOL:-1e-8}"
REPEATS="${REPEATS:-7}"
ARMS="${ARMS:-as-measured exact-depth}"
OPTIONS="${OPTIONS:-basket rainbow}"
WIDTHS="${WIDTHS:-1 4}"
SRUN_MPI="${SRUN_MPI:-pmix}"
STATE_RTOL="${STATE_RTOL:-1e-10}"

# m is a BUILD constant (CAKSM_GPU_CA_MAX_M), so one binary is one ceiling and
# two ceilings are two sweeps. It goes in the directory name for that reason:
# ladders taken at different m are not the same experiment and must not merge.
OUT_DIR="${OUT_DIR:-$ROOT/data/ca-integrator-strong-$MACHINE-m$M}"
# Refuse to write into a directory that already holds transcripts. Uniqueness is
# enforced here rather than by an archive step afterwards, because by then the
# overwritten run is already gone.
FORCE="${FORCE:-0}"

# The sweep, derived from the allocation unless named.
STRONG_NODES="${STRONG_NODES:-}"
STRONG_LOCAL_GPUS="${STRONG_LOCAL_GPUS:-}"
STRONG_ARMS="${STRONG_ARMS:-}"
SKIPPED_POINTS=0

for binary in "$SINGLE" "$DISTRIBUTED"; do
    if [[ ! -x "$binary" ]]; then
        echo "Error: missing executable: $binary" >&2
        exit 1
    fi
done
if ! command -v srun >/dev/null 2>&1; then
    echo "Error: srun is required." >&2
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

if [[ -z "$STRONG_NODES" ]]; then
    STRONG_NODES="$(allocated_nodes)"
fi
if [[ "$STRONG_NODES" -lt 1 ]]; then
    echo "Error: no Slurm allocation found; this sweep launches every rung with srun." >&2
    report_allocation
    exit 1
fi

# What the allocation actually holds, unless the caller named it.
if [[ -z "$STRONG_LOCAL_GPUS" ]]; then
    if [[ -n "${SLURM_GPUS_PER_NODE:-}" ]]; then
        STRONG_LOCAL_GPUS="${SLURM_GPUS_PER_NODE##*:}"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        STRONG_LOCAL_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    else
        STRONG_LOCAL_GPUS=1
    fi
fi
[[ "$STRONG_NODES" =~ ^[0-9]+$ && "$STRONG_NODES" -ge 1 ]] \
    || { echo "Error: STRONG_NODES must be a positive integer, got '$STRONG_NODES'" >&2; exit 1; }
[[ "$STRONG_LOCAL_GPUS" =~ ^[0-9]+$ && "$STRONG_LOCAL_GPUS" -ge 1 ]] \
    || { echo "Error: STRONG_LOCAL_GPUS must be a positive integer, got '$STRONG_LOCAL_GPUS'" >&2; exit 1; }

# The powers of two up to a bound: the counts that separate one arrangement from
# the next. An increment of one would repeat arrangements that differ by a
# fraction of a slab and cost a full solver run each.
strong_ladder() {   # <max> -> "1 2 4 ... <= max"
    local max="$1" k=1 out=()
    while (( k <= max )); do out+=("$k"); k=$((k * 2)); done
    echo "${out[*]}"
}

if [[ -z "$STRONG_ARMS" ]]; then
    strong_arms=()
    for arm_nodes in $(strong_ladder "$STRONG_NODES"); do
        for arm_local in $(strong_ladder "$STRONG_LOCAL_GPUS"); do
            strong_arms+=("${arm_nodes}x${arm_local}")
        done
    done
    STRONG_ARMS="${strong_arms[*]}"
fi

has_single=0
for arm in $STRONG_ARMS; do
    [[ "$arm" =~ ^[0-9]+x[0-9]+$ ]] \
        || { echo "Error: STRONG_ARMS entries must read <nodes>x<local>, got '$arm'" >&2; exit 1; }
    [[ "${arm%x*}" -le "$STRONG_NODES" ]] \
        || { echo "Error: arm $arm asks for more nodes than the allocation holds" >&2; exit 1; }
    [[ "${arm#*x}" -le "$STRONG_LOCAL_GPUS" ]] \
        || { echo "Error: arm $arm asks for more local GPUs than the allocation holds" >&2; exit 1; }
    [[ "$arm" == "1x1" ]] && has_single=1
done
if [[ "$has_single" -ne 1 ]]; then
    echo "Error: STRONG_ARMS must contain 1x1." >&2
    echo "  It is the ladder's denominator and the state every distributed rung" >&2
    echo "  is validated against, so a sweep without it can neither be scaled" >&2
    echo "  nor accepted." >&2
    exit 1
fi

if [[ -d "$OUT_DIR" ]] && compgen -G "$OUT_DIR/*.txt" >/dev/null \
   && [[ "$FORCE" != "1" ]]; then
    echo "Error: $OUT_DIR already holds transcripts." >&2
    echo "  A ladder overwritten in place cannot be told from the one it replaced." >&2
    echo "  Write elsewhere with OUT_DIR=..., or pass FORCE=1 to overwrite." >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
export PMIX_MCA_gds="${PMIX_MCA_gds:-hash}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# The single-GPU state each distributed arm is validated against. Generated once
# per (arm, option, width) before the distributed rungs, and never overwritten.
STATE_DIR="$OUT_DIR/states"
mkdir -p "$STATE_DIR"

# The acceptance gate's flags for one option, empty when no referee exists for
# this grid. Filled into `gate` by the sweep below rather than returned, because
# `mapfile` is bash 4 and this has to parse under the 3.2 that ships on macOS,
# where the launch geometry is dry-run before a job is submitted.
set_gate_args() {   # <option> -> fills the `gate` array
    local option="$1"
    gate=()
    if [[ -f "$REFEREE_ROOT/n${N}/referee_n${N}_${option}.bin" ]]; then
        gate=(--referee-dir "$REFEREE_ROOT/n${N}")
    fi
}

run_rung() {
    local arm="$1" topology="$2" option="$3" width="$4"
    shift 4
    local log="$OUT_DIR/${arm}_${topology}_${option}_s${width}.txt"

    echo "### arm=$arm topology=$topology option=$option s=$width n=$N"
    set +e
    "$@" 2>&1 | tee "$log"
    local exit_code="${PIPESTATUS[0]}"
    set -e
    if [[ $exit_code -ne 0 ]]; then
        local status="execution-fail"
        if grep -Eq \
                'validation: FAIL.*(boundary|single|referee)=FAIL' "$log"; then
            status="acceptance-fail"
        elif grep -Eq 'unconverged=[1-9][0-9]*' "$log"; then
            status="stopped"
        elif grep -q 'validation: FAIL' "$log"; then
            status="acceptance-fail"
        fi
        echo "RUN_STATUS status=$status exit_code=$exit_code" | tee -a "$log"
        echo "Error: strong-scaling rung failed; evidence: $log" >&2
        return "$exit_code"
    fi
    if grep -q 'CONTENDED DEVICE' "$log" \
       || grep -q 'recordable=no' "$log"; then
        echo "RUN_STATUS status=contended exit_code=0" | tee -a "$log"
        echo "Error: contended strong-scaling rung; evidence: $log" >&2
        return 1
    fi
    if ! grep -Eq "distribution: .*\\(${REPEATS} runs\\)" "$log"; then
        echo "RUN_STATUS status=incomplete exit_code=0" | tee -a "$log"
        echo "Error: missing ${REPEATS}-repeat timing distribution; evidence: $log" >&2
        return 1
    fi
    echo "RUN_STATUS status=passed exit_code=0" | tee -a "$log"
    echo
}

# The thinnest slab a decomposition produces, and the halo depth an arm needs.
#
# The integrator splits z as z_begin = rank * n / P, so the thinnest slab holds
# floor(n / P) planes, and exchange_halos_nccl sends the neighbour's last
# `depth` owned planes from input + (z_count - depth) * n^2. A slab thinner than
# the halo makes that offset negative and the send reads before the start of its
# own buffer. That is a memory error rather than a numerical one, so the point is
# refused here instead of being run and discarded afterwards.
#
# It is reachable: at n=61 on sixteen participants the thinnest slab is three
# planes, and the as-measured arm at s=4 asks for four.
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

# One arrangement, launched. The geometry is the only thing that varies: one
# rank per node, each rank driving the first `local` devices, so participants =
# nodes x local. One node with one device takes the non-distributed binary,
# which is the one case that is not a smaller instance of the same launch.
launch_rung() {   # <nodes> <local> <arm> <option> <width> <state>
    local nodes="$1" local_gpus="$2" arm="$3" option="$4" width="$5" state="$6"
    local participants=$(( nodes * local_gpus ))
    local topology="${participants}gpu_${nodes}node"

    local depth thinnest
    depth="$(halo_depth "$arm" "$width")"
    thinnest="$(thinnest_slab "$N" "$participants")"
    if [[ "$depth" -gt "$thinnest" ]]; then
        printf 'SKIPPED %s arm=%s s=%s: halo depth %s exceeds the thinnest slab (%s planes at n=%s over %s participants)\n' \
            "$topology" "$arm" "$width" "$depth" "$thinnest" "$N" "$participants" \
            | tee -a "$OUT_DIR/skipped.txt"
        SKIPPED_POINTS=$((SKIPPED_POINTS + 1))
        return 0
    fi

    # Built in the shell rather than with `seq -s,`: BSD seq appends the
    # separator after the last element and GNU seq does not, so that would send
    # "--devices 0,1," from a Mac and "--devices 0,1" from the cluster, and the
    # integrator rejects the empty token.
    local -a device_list=()
    local d
    for (( d = 0; d < local_gpus; ++d )); do device_list+=("$d"); done
    local devices
    devices="$(IFS=,; echo "${device_list[*]}")"

    # Built as one array rather than expanding an optional one: "${pin[@]}" on
    # an empty array is an unbound-variable error under `set -u` in bash 3.2.
    local -a launcher=(srun)
    if [[ "$nodes" -eq 1 ]]; then
        launcher+=(--nodes=1 --ntasks=1 --ntasks-per-node=1 --mpi=none)
    else
        launcher+=(--nodes="$nodes" --ntasks="$nodes" --ntasks-per-node=1
                   --mpi="$SRUN_MPI")
    fi

    if [[ "$nodes" -eq 1 && "$local_gpus" -eq 1 ]]; then
        run_rung "$arm" "$topology" "$option" "$width" \
            "${launcher[@]}" "$SINGLE" "${common[@]}" --save-state "$state" \
            "${gate[@]+"${gate[@]}"}"
    elif [[ "$nodes" -eq 1 ]]; then
        run_rung "$arm" "$topology" "$option" "$width" \
            "${launcher[@]}" "$DISTRIBUTED" --devices "$devices" \
            "${common[@]}" --single-gpu-state "$state" \
            --single-state-rtol "$STATE_RTOL" "${gate[@]+"${gate[@]}"}"
    else
        run_rung "$arm" "$topology" "$option" "$width" \
            "${launcher[@]}" "$DISTRIBUTED" --require-mpi --devices "$devices" \
            "${common[@]}" --single-gpu-state "$state" \
            --single-state-rtol "$STATE_RTOL" "${gate[@]+"${gate[@]}"}"
    fi
}

echo "==============================================================================="
echo " Strong-scaling ladder   machine=$MACHINE  n=$N  m=$M"
echo " host=$(uname -n)  job=${SLURM_JOB_ID:-none}"
echo "==============================================================================="
echo "  alloc   : $STRONG_NODES node(s) x $STRONG_LOCAL_GPUS GPU(s)"
echo "  arms    : $STRONG_ARMS   (<nodes>x<local GPUs>)"
echo "  contract: arms=$ARMS options=$OPTIONS widths=$WIDTHS"
echo "            steps=$STEPS tol=$TOL repeats=$REPEATS"
echo "  output  : $OUT_DIR"
echo

for arm in $ARMS; do
    for option in $OPTIONS; do
        for width in $WIDTHS; do
            set_gate_args "$option"
            state="$STATE_DIR/${arm}_single_n${N}_${option}_s${width}.bin"

            common=(
                --n "$N" --m "$M" --s "$width" --steps "$STEPS"
                --tol "$TOL" --repeats "$REPEATS" --option "$option"
                --basis monomial --orth cholqr2 --arm "$arm"
            )

            # 1x1 first, so the state the distributed rungs validate against
            # exists before any of them runs, whatever order STRONG_ARMS lists.
            launch_rung 1 1 "$arm" "$option" "$width" "$state"
            for rung in $STRONG_ARMS; do
                [[ "$rung" == "1x1" ]] && continue
                launch_rung "${rung%x*}" "${rung#*x}" \
                    "$arm" "$option" "$width" "$state"
            done
        done
    done
done

# The problem the ladder was run at, kept beside the transcripts so two sweeps
# can be told apart without reading a transcript header.
{
    echo "machine=$MACHINE"
    echo "arms=$ARMS"
    echo "options=$OPTIONS"
    echo "widths=$WIDTHS"
    echo "arrangements=$STRONG_ARMS"
    echo "n=$N"
    echo "m=$M"
    echo "steps=$STEPS"
    echo "tol=$TOL"
    echo "single_state_rtol=$STATE_RTOL"
    echo "repeats=$REPEATS"
    echo "skipped_points=$SKIPPED_POINTS"
} > "$OUT_DIR/settings.txt"

# Provenance beside the transcripts, in the same shape the calibration job
# writes: which host, which allocation, which revision. A cycle time whose build
# cannot be identified cannot be compared against another sweep's.
{
    echo "machine=$MACHINE"
    echo "host=$(uname -n)"
    echo "kernel=$(uname -sr)"
    echo "date=$(date -Is 2>/dev/null || date -u '+%Y-%m-%dT%H:%M:%S+00:00')"
    echo "slurm_job=${SLURM_JOB_ID:-none}"
    echo "slurm_nodelist=${SLURM_JOB_NODELIST:-none}"
    echo "nodes=$STRONG_NODES"
    echo "local_gpus=$STRONG_LOCAL_GPUS"
    echo "arrangements=$STRONG_ARMS"
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

# Checksums last, so the manifest covers everything above it. Lets a transcript
# be quoted later without trusting that nobody edited it in place.
#
# find rather than a glob, for two reasons: the glob would hand the tool the
# states/ directory, which makes it exit non-zero and leaves the manifest as a
# .tmp nobody notices; and the reference states belong under the checksums as
# much as the transcripts do, since they are what every distributed rung was
# accepted against. macOS ships shasum, Linux sha256sum.
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
echo "Strong-scaling ladder recorded."
echo "  transcripts : $OUT_DIR"
echo "  states      : $STATE_DIR"
echo "  settings    : $OUT_DIR/settings.txt"
echo "  provenance  : $OUT_DIR/provenance.txt"
echo "  checksums   : $OUT_DIR/SHA256SUMS"
if [[ "$SKIPPED_POINTS" -ne 0 ]]; then
    echo "  skipped     : $SKIPPED_POINTS point(s) whose halo exceeded the thinnest slab"
    echo "                see $OUT_DIR/skipped.txt; raise n or drop the arrangement"
fi
echo "Draw the figures this ladder feeds:"
echo "  uv run --script scripts/plots/ca_strong_scaling.py   P1: the ladder itself"
echo "  uv run --script scripts/plots/ca_weak_scaling.py     P2: its one-GPU efficiency denominator"
echo "==============================================================================="
