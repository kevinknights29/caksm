#!/bin/bash
# Launch the asset-dimension sweep as one detached tmux session per job.
#
# Each regime-control point is independent and single-threaded (the binary is not linked
# against OpenMP), so the sweep is embarrassingly parallel: one job per core. Jobs write
# to separate part-CSVs and never touch each other's results.
#
#   ./scripts/regime/regime_dsweep_launch.sh          # launch all jobs, detached
#   ./scripts/regime/regime_dsweep_launch.sh --status # who is still running
#   tmux attach -t dsweep-d6                   # watch one
#   ./scripts/regime/regime_dsweep.sh --merge         # once --status says all done
#
# Logs land in logs/dsweep_<job>.out (line-buffered, so tail -f works).
set -uo pipefail

WORK_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SWEEP="$WORK_DIR/scripts/regime/regime_dsweep.sh"
PREFIX="dsweep"

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
        part="$WORK_DIR/data/regime/parts/regime_dsweep_${j}.csv"
        rows=0; [[ -f "$part" ]] && rows=$(( $(wc -l < "$part") - 1 ))
        if tmux has-session -t "$s" 2>/dev/null; then st="RUNNING"; all_done=0; else st="done"; fi
        printf '%-18s %-10s %5d  logs/dsweep_%s.out\n' "$s" "$st" "$rows" "$j"
    done
    (( all_done )) && echo -e "\nAll jobs finished. Merge with:\n  ./scripts/regime/regime_dsweep.sh --merge"
    exit 0
fi

if [[ ! -x "$WORK_DIR/build/regime-control" ]]; then
    echo "Error: build/regime-control missing. Build first:"
    echo "  cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel"
    exit 1
fi

echo "Launching ${#JOBS[@]} detached tmux sessions (one core each)..."
for j in "${JOBS[@]}"; do
    s="$PREFIX-$j"
    if tmux has-session -t "$s" 2>/dev/null; then
        echo "  $s already exists; skipping (kill it with: tmux kill-session -t $s)"
        continue
    fi
    log="$WORK_DIR/logs/dsweep_${j}.out"
    # stdbuf -oL: regime-control's stdout is block-buffered when redirected, so without
    # this the log looks frozen for minutes at a time and a live job reads as a hung one.
    tmux new-session -d -s "$s" \
        "cd '$WORK_DIR' && JOB='$j' stdbuf -oL -eL ./scripts/regime/regime_dsweep.sh 2>&1 | tee '$log'"
    echo "  $s  -> $log"
done

cat <<EOF

Watch:    tmux attach -t ${PREFIX}-${JOBS[0]}      (detach with Ctrl-b then d)
          tail -f logs/dsweep_${JOBS[0]}.out
Status:   ./scripts/regime/regime_dsweep_launch.sh --status
Merge:    ./scripts/regime/regime_dsweep.sh --merge       (after all are done)
EOF
