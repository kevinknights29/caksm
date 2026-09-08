#!/bin/bash
# Print the kGpuTopologies entry for every collective measured in an archived run.
#
# The point is that nobody retypes a measured number. Of the fields in a GpuTopology entry,
# almost all are already recorded by the run: the participant count, the node count, the rung,
# the median and its quartiles, the bus bandwidth, the repeat count and the idle-device verdict
# come from the calibrator's own CSV, and the host, device list and link label come from the
# provenance files beside it. Retyping those is a chance to mistype a constant in a model whose
# whole premise is that constants are not mistyped.
#
# What it cannot decide is left as a marked judgment, because these are the fields a person is
# supposed to think about:
#
#   label           how the figure names the arrangement
#   invocations     how many independent runs the published figure pools, and therefore
#                   whether the quartiles are one run's IQR or the spread across runs
#   calibrated      whether this row may place a point at all
#   link_signature  proposed from the observed matrix, but a subset of a node whose halves
#                   differ must name its devices, and a run that cannot is "unknown"
#
# Usage:
#   scripts/regime/topology_entry.sh <run directory>
#
# calibrate_gpu_p2p.sh writes run directories as
# data/calibration/<machine>/<timestamp>-job<jobid>-<N>nodes/.
# They are not committed, so pass whichever one the job produced.
set -uo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 RUN_DIR" >&2
    echo "  RUN_DIR is a run directory written by calibrate_gpu_p2p.sh, e.g." >&2
    echo "  data/calibration/<machine>/<timestamp>-job<jobid>-<N>nodes" >&2
    exit 2
fi

RUN_DIR="${1%/}"
WORK_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

if [[ ! -d "$RUN_DIR" ]]; then
    echo "error: no such run directory: $RUN_DIR" >&2
    exit 1
fi

# calibrate_gpu_p2p.sh puts the collective CSVs in a p2p/ subdirectory, so accept either the
# run directory or that subdirectory. Descending only when the top level holds no CSV of its
# own keeps a flat archive working.
if [[ -d "$RUN_DIR/p2p" ]] \
   && ! compgen -G "$RUN_DIR/*participants_*node*.csv" >/dev/null \
   && ! compgen -G "$RUN_DIR/calibrate_gpu_p2p_*.csv" >/dev/null; then
    RUN_DIR="$RUN_DIR/p2p"
fi

# Repo-relative, since that is what the source column records.
REL_DIR="$RUN_DIR"
case "$RUN_DIR" in /*) REL_DIR="${RUN_DIR#"$WORK_DIR"/}" ;; esac

read_provenance() {   # <key> -> value, or empty
    [[ -f "$RUN_DIR/provenance.txt" ]] || return 0
    awk -F= -v k="$1" '$1 == k { print $2; exit }' "$RUN_DIR/provenance.txt"
}

# A run without provenance.txt still has the facts in its logs, so they are recovered there
# rather than reported as unknown.
HOST="${HOST_OVERRIDE:-$(read_provenance host)}"
[[ -z "$HOST" ]] && HOST="$(sed -e 's/\x1b\[[0-9;]*[A-Za-z]//g' "$RUN_DIR"/*.log 2>/dev/null \
                            | awk '/^ *host  *:/ { print $3; exit }')"
[[ -z "$HOST" ]] && HOST="TODO-host"

VISIBLE_DEVICES="$(read_provenance visible_devices)"
if [[ -z "$VISIBLE_DEVICES" ]]; then
    VISIBLE_DEVICES="$(awk -F= '/CUDA_VISIBLE_DEVICES=/ { print $2; exit }' \
                       "$RUN_DIR"/*visible_devices*.txt 2>/dev/null)"
fi

# The uniform GPU-to-GPU label, from the observed matrix. nvidia-smi writes ANSI escapes into
# the header, so they are stripped before the table is read; the raw file is left untouched
# because an edited copy is no longer evidence. A matrix with more than one label is reported
# as MIXED rather than reduced to whichever cell came first.
link_label() {
    local topo=""
    for c in "$RUN_DIR/topology.txt" "$RUN_DIR"/*topology*.txt; do
        [[ -f "$c" ]] && { topo="$c"; break; }
    done
    [[ -z "$topo" ]] && { echo "unknown"; return; }
    # Only the GPU-to-GPU block. The same rows continue into NIC affinity and CPU columns,
    # and reading those would report a node whose every GPU pair is NV18 as MIXED, because its
    # NICs are legitimately PIX, PXB, NODE and SYS. The block is square, so the GPU row count
    # gives its width.
    local labels
    labels="$(sed -e 's/\x1b\[[0-9;]*[A-Za-z]//g' "$topo" \
        | awk '/^GPU[0-9]+/ { rows[++n] = $0 }
               END { for (r = 1; r <= n; r++) {
                         split(rows[r], f, /\t/)
                         for (i = 2; i <= n + 1; i++) {
                             v = f[i]; gsub(/^ +| +$/, "", v)
                             if (v != "" && v != "X") print v } } }' \
        | sort -u | paste -sd, -)"
    [[ -z "$labels" ]] && { echo "unknown"; return; }
    [[ "$labels" == *,* ]] && { echo "MIXED"; return; }
    echo "$labels"
}

LINK="$(link_label)"

# The transport NCCL selected, where a diagnostic was captured. Read rather than assumed: the
# calibrator's code path says which primitive was called, not which network was chosen.
transport_for() {   # <participants> <nodes>
    local nccl="" candidate
    for candidate in "$RUN_DIR"/*"${1}participants_${2}node"*_nccl.log \
                     "$RUN_DIR/node_${1}participants_${2}nodes.log"; do
        [[ -f "$candidate" ]] || continue
        if grep -q 'NCCL INFO' "$candidate" 2>/dev/null; then
            nccl="$candidate"
            break
        fi
    done
    if [[ -n "$nccl" ]]; then
        local net
        net="$(sed -n -E \
            's/.*Using network ([A-Za-z0-9_.-]+).*/\1/p' "$nccl" 2>/dev/null \
            | tail -1)"
        if [[ -z "$net" ]]; then
            net="$(sed -n -E \
                's@.*NET/([A-Za-z0-9_.-]+).*:[[:space:]]+Using .*@\1@p' \
                "$nccl" 2>/dev/null | tail -1)"
        fi
        case "$net" in
            IB)     echo "nccl-net-ib";     return ;;
            Socket) echo "nccl-net-socket"; return ;;
            ?*)     echo "nccl-net-$net";   return ;;
        esac
    fi
    [[ "$2" -gt 1 ]] && { echo "nccl-net-UNKNOWN"; return; }
    echo "nccl-p2p"
}

MACHINE_SEEN=""
COUNT=0

emit_entry() {   # <csv>
    local csv="$1"
    # The calibrator's own columns. Only the header names are trusted, not their order.
    local row
    row="$(awk -F, 'NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; next }
                    NR == 2 { printf "%s|%s|%s|%s|%s|%s|%s|%s|%s|%s",
                        $col["machine"], $col["participants"], $col["hosts"], $col["tier"],
                        $col["repeats"], $col["t_allreduce_s"], $col["t_q1"], $col["t_q3"],
                        $col["bw_gbs"], $col["contended"] }' "$csv")"
    [[ -z "$row" ]] && { echo "  // SKIPPED (no data row): $(basename "$csv")"; return; }

    IFS='|' read -r machine participants hosts tier repeats med q1 q3 bw contended <<< "$row"
    local local_gpus=$((participants / hosts))

    # GPUs one process drives: the launch structure. No CSV column records it, so it comes
    # from the file stem. The single-process path gives one process every local device; the
    # MPI path gives each rank one.
    local gpus_per_rank="$local_gpus"
    case "$(basename "$csv")" in
        node_*|*_mpi.csv) gpus_per_rank=1 ;;
    esac
    local key="${participants}gpu-${hosts}node"
    local plural_g="s" plural_n="s"
    [[ "$participants" -eq 1 ]] && plural_g=""
    [[ "$hosts" -eq 1 ]] && plural_n=""

    # The devices used on each host. A contiguous run from zero collapses to "0-7";
    # anything else is listed, and an unknown set stays unknown.
    local devices="unknown"
    if [[ -n "$VISIBLE_DEVICES" ]]; then
        local first last n
        n="$(awk -F, '{print NF}' <<< "$VISIBLE_DEVICES")"
        if [[ "$n" -ge "$local_gpus" ]]; then
            first="$(cut -d, -f1 <<< "$VISIBLE_DEVICES")"
            last="$(cut -d, -f"$local_gpus" <<< "$VISIBLE_DEVICES")"
            if [[ "$local_gpus" -eq 1 ]]; then devices="$first"
            else devices="$first-$last"; fi
        fi
    fi

    local signature="$LINK:$devices"
    local calibrated="true" note=""
    if [[ "$contended" != "0" ]]; then
        calibrated="false"
        note="    // BLOCKED: the idle-device gate failed. A contended collective costs what"$'\n'"    // its slowest participant costs, so this is not a measurement of the link."
    elif [[ "$devices" == "unknown" || "$LINK" == "unknown" || "$LINK" == "MIXED" ]]; then
        calibrated="false"
        signature="unknown"
        note="    // BLOCKED: the devices or the link matrix were not recorded, so this"$'\n'"    // arrangement cannot be told apart from a differently connected subset."
    fi

    local tier_enum
    case "$tier" in
        grid)       tier_enum="ReductionTier::GRID" ;;
        device-p2p) tier_enum="ReductionTier::DEVICE_P2P" ;;
        node)       tier_enum="ReductionTier::NODE" ;;
        *)          tier_enum="ReductionTier::/* UNKNOWN: $tier */" ;;
    esac

    [[ -z "$MACHINE_SEEN" ]] && MACHINE_SEEN="$machine"

    echo "    {"
    [[ -n "$note" ]] && echo "$note"
    local cluster_todo=""
    [[ "$HOST" == "TODO-host" ]] && cluster_todo="  // TODO cluster: the host was not recorded"
    echo "        \"$machine\", \"$machine-$HOST\",$cluster_todo"
    echo "        \"$key\", \"$participants GPU$plural_g, $hosts node$plural_n\","
    echo "        $participants, $hosts, $local_gpus, $gpus_per_rank, \"$devices\", $tier_enum,"
    echo "        ${med}, ${q1}, ${q3},"
    echo "        ${bw}, ${repeats}, 1,  // TODO invocations: raise it if this pools several runs,"
    echo "                               // and widen the quartiles to their spread if you do"
    echo "        $([[ "$contended" == "0" ]] && echo false || echo true), $calibrated,  // TODO confirm calibrated"
    echo "        \"$signature\", \"$(transport_for "$participants" "$hosts")\","
    echo "        \"calibrate-gpu-p2p\","
    echo "        \"$REL_DIR/$(basename "$csv")\","
    echo "    },"
    COUNT=$((COUNT + 1))
}

echo "// Generated by scripts/regime/topology_entry.sh from"
echo "//   $REL_DIR"
echo "// Paste into kGpuTopologies in include/gpu_topology.hpp and resolve every TODO."
echo ""

# Per-arrangement stems first, then the older fixed names, so both layouts yield an entry.
shopt -s nullglob
for csv in "$RUN_DIR"/device_*participants_*node*.csv "$RUN_DIR"/node_*participants_*node*.csv \
           "$RUN_DIR"/calibrate_gpu_p2p_device.csv "$RUN_DIR"/calibrate_gpu_p2p_node.csv; do
    # nullglob drops an unmatched pattern but not an unmatched literal, and the two legacy
    # names have no wildcard in them.
    [[ -f "$csv" ]] && emit_entry "$csv"
done
shopt -u nullglob

echo ""
if [[ $COUNT -eq 0 ]]; then
    echo "// No collective CSV found in $REL_DIR."
    echo "// Expected device_<N>participants_<M>node.csv, node_<N>participants_<M>nodes.csv,"
    echo "// or the legacy calibrate_gpu_p2p_{device,node}.csv."
    exit 1
fi
echo "// $COUNT arrangement(s). The single-participant grid rung is NOT one of them: with one"
echo "// participant no collective leaves the device, so its entry is the two-kernel grid total"
echo "// from calibrate_gpu_reduction.csv in this run, not an NCCL measurement."
