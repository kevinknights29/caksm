#!/bin/bash
# Launch the numerics-gate sweep as detached tmux sessions, one per job.
#
# This sweep has a hard gate: the gate job (phase 1) must pass before any downstream
# number means anything, so it runs synchronously here and the rest are fanned out to
# tmux only if it exits 0. That keeps "if the instrument is wrong, stop" under parallel
# launch.
#
# Each regime-control point is independent and single-threaded (the binary is not linked
# against OpenMP), so the rest is embarrassingly parallel: one job per core. Jobs write to
# separate part-CSVs and never touch each other's results.
#
#   ./scripts/regime/regime_control_launch.sh          # gate, then launch the rest detached
#   ./scripts/regime/regime_control_launch.sh --status # who is still running
#   tmux attach -t rctl-realbs20                # watch one
#   ./scripts/regime/regime_control.sh --merge         # once --status says all done
#
# Logs land in logs/rctl_<job>.out (line-buffered, so tail -f works).
set -uo pipefail

WORK_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SWEEP="$WORK_DIR/scripts/regime/regime_control.sh"
PREFIX="rctl"

command -v tmux >/dev/null || { echo "tmux not found (try: screen, or run serially)"; exit 1; }
mkdir -p "$WORK_DIR/logs"

# Ask the sweep itself for the job list, so the two can never drift apart.
mapfile -t JOBS < <("$SWEEP" --list | sed -n 's/^Jobs:  *//p' | tr ' ' '\n' | grep -v '^$')
if (( ${#JOBS[@]} == 0 )); then
    echo "Could not determine the job list from $SWEEP --list"; exit 1
fi

if [[ "${1:-}" == "--status" ]]; then
    echo "Session            state      rows  log"
    all_done=1
    for j in "${JOBS[@]}"; do
        s="$PREFIX-$j"
        part="$WORK_DIR/data/regime/parts/regime_control_${j}.csv"
        rows=0; [[ -f "$part" ]] && rows=$(( $(wc -l < "$part") - 1 ))
        if tmux has-session -t "$s" 2>/dev/null; then st="RUNNING"; all_done=0; else st="done"; fi
        printf '%-18s %-10s %5d  logs/rctl_%s.out\n' "$s" "$st" "$rows" "$j"
    done
    (( all_done )) && echo -e "\nAll jobs finished. Merge with:\n  ./scripts/regime/regime_control.sh --merge"
    exit 0
fi

if [[ ! -x "$WORK_DIR/build/regime-control" ]]; then
    echo "Error: build/regime-control missing. Build first:"
    echo "  cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel"
    exit 1
fi

# The gate, synchronously. If it fails, regime_control.sh exits non-zero and we stop:
# no point launching the data-collection jobs against a broken instrument.
if printf '%s\n' "${JOBS[@]}" | grep -qx gate; then
    echo "Running the gate synchronously (PHASE 1) before fanning out..."
    log="$WORK_DIR/logs/rctl_gate.out"
    if ! JOB=gate stdbuf -oL -eL "$SWEEP" 2>&1 | tee "$log"; then
        echo
        echo "GATE FAILED; not launching the rest. See $log."
        exit 1
    fi
    echo "Gate passed."
fi

echo "Launching detached tmux sessions for the remaining jobs (one core each)..."
for j in "${JOBS[@]}"; do
    [[ "$j" == gate ]] && continue          # already run, synchronously, above
    s="$PREFIX-$j"
    if tmux has-session -t "$s" 2>/dev/null; then
        echo "  $s already exists; skipping (kill it with: tmux kill-session -t $s)"
        continue
    fi
    log="$WORK_DIR/logs/rctl_${j}.out"
    # stdbuf -oL: regime-control's stdout is block-buffered when redirected, so without
    # this the log looks frozen for minutes and a live job reads as a hung one.
    tmux new-session -d -s "$s" \
        "cd '$WORK_DIR' && JOB='$j' stdbuf -oL -eL ./scripts/regime/regime_control.sh 2>&1 | tee '$log'"
    echo "  $s  -> $log"
done

first_bg="$(printf '%s\n' "${JOBS[@]}" | grep -vx gate | head -1)"
cat <<EOF
Watch:    tmux attach -t ${PREFIX}-${first_bg}      (detach with Ctrl-b then d)
          tail -f logs/rctl_${first_bg}.out
Status:   ./scripts/regime/regime_control_launch.sh --status
Merge:    ./scripts/regime/regime_control.sh --merge       (after all are done)
EOF
