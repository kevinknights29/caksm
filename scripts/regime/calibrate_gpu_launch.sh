#!/bin/bash
# Launch the GPU calibration as a detached tmux session.
#
# calibrate_gpu.sh is a foreground job: on an SSH disconnect the session leader takes SIGHUP,
# the script and its `sleep` child die, and writes to the dead pty fail. That is survivable for
# a five-minute smoke test but not for the run that matters, which is gated on WAIT_FOR_IDLE
# and may wait hours for someone else's job to clear.
#
# Same shape as regime_control_launch.sh and regime_dsweep_launch.sh: one detached session, a
# line-buffered log under logs/, and a --status that answers the question worth asking, which
# is not "is it running" but "did it produce numbers I may record".
#
#   ./scripts/regime/calibrate_gpu_launch.sh                 # detach and wait for an idle GPU
#   ./scripts/regime/calibrate_gpu_launch.sh --status        # running? and was it recordable?
#   tmux attach -t calgpu-rtx-3090                    # watch (detach: Ctrl-b then d)
#   tail -f logs/calgpu_rtx-3090.out
#
# Environment (passed through to calibrate_gpu.sh):
#   MACHINE        preset key; also names the session and log. Default v100-pcie-16gb.
#   WAIT_FOR_IDLE  seconds to wait for the device to fall idle. Defaults to 4 h here, unlike
#                  calibrate_gpu.sh's 0: a detached run has nothing to lose by waiting, and
#                  waiting is the reason to detach.
#   DEVICE, IDLE_MIB, POLL_S, REPEATS, ITERS, MAX_MIB   as calibrate_gpu.sh documents them.
set -uo pipefail

WORK_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
CALIB="$WORK_DIR/scripts/regime/calibrate_gpu.sh"

MACHINE="${MACHINE:-v100-pcie-16gb}"
WAIT_FOR_IDLE="${WAIT_FOR_IDLE:-14400}"    # 4 h; see above for why this differs
DEVICE="${DEVICE:-0}"

SESSION="calgpu-$MACHINE"
LOG="$WORK_DIR/logs/calgpu_${MACHINE}.out"

command -v tmux >/dev/null || { echo "tmux not found (try: screen, or nohup ... &)"; exit 1; }
mkdir -p "$WORK_DIR/logs"

if [[ "${1:-}" == "--status" ]]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        state="RUNNING"
    else
        state="done"
    fi
    printf '%-22s %-10s %s\n' "$SESSION" "$state" "logs/calgpu_${MACHINE}.out"

    if [[ ! -f "$LOG" ]]; then
        echo "No log yet."
        exit 0
    fi

    # Still waiting? The wait loop is the long pole, so surface it rather than making
    # someone read the log to find out nothing has started.
    if tail -5 "$LOG" 2>/dev/null | grep -q 'MiB in use, still contended'; then
        echo
        echo "Still waiting for an idle device:"
        tail -1 "$LOG" | sed 's/^/  /'
    fi

    # The question that matters. A finished run is worthless if it ran contended, and the
    # difference is one line in the log rather than anything in the numbers themselves.
    # `grep -c` prints the count and exits 1 when it is zero, so `|| echo 0` would append a
    # second zero and the arithmetic tests below would fail on "0\n0". Take the count and
    # discard the status instead.
    suppressed=$(grep -c 'Recording block suppressed' "$LOG" 2>/dev/null) || true
    recordable=$(grep -c 'To record, set these' "$LOG" 2>/dev/null) || true
    echo
    if [[ "$state" == "done" && "$recordable" -gt 0 && "$suppressed" -eq 0 ]]; then
        echo "RECORDABLE: all three instruments ran on an idle device."
        echo "Transcribe into the '$MACHINE' preset in include/gpu_machine.hpp:"
        grep -A4 'To record, set these' "$LOG" | sed 's/^/  /'
    elif [[ "$suppressed" -gt 0 ]]; then
        echo "NOT RECORDABLE: $suppressed of the instruments hit a contended device."
        echo "The run is a valid smoke test and nothing more. Re-launch when the card clears:"
        echo "  tmux kill-session -t $SESSION 2>/dev/null; MACHINE=$MACHINE $0"
    else
        echo "In progress, or no instrument has finished a section yet."
    fi
    exit 0
fi

if [[ ! -x "$CALIB" ]]; then
    echo "Error: $CALIB not found or not executable"
    exit 1
fi
if [[ ! -x "$WORK_DIR/build/gpu-fma-loop" ]]; then
    echo "Error: build/gpu-fma-loop missing; the GPU instruments are not built."
    echo "See scripts/regime/calibrate_gpu.sh for the CUDAHOSTCXX build line."
    exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session $SESSION already exists."
    echo "  watch:  tmux attach -t $SESSION"
    echo "  status: $0 --status"
    echo "  kill:   tmux kill-session -t $SESSION"
    exit 1
fi

# stdbuf -oL: the calibration binaries block-buffer stdout when it is a pipe, so without this
# a live run's log looks frozen for minutes at a time and reads as a hang. The same reason
# regime_control_launch.sh does it.
tmux new-session -d -s "$SESSION" \
    "cd '$WORK_DIR' && MACHINE='$MACHINE' DEVICE='$DEVICE' WAIT_FOR_IDLE='$WAIT_FOR_IDLE' \
     ${IDLE_MIB:+IDLE_MIB='$IDLE_MIB'} ${POLL_S:+POLL_S='$POLL_S'} \
     ${REPEATS:+REPEATS='$REPEATS'} ${ITERS:+ITERS='$ITERS'} ${MAX_MIB:+MAX_MIB='$MAX_MIB'} \
     stdbuf -oL -eL ./scripts/regime/calibrate_gpu.sh 2>&1 | tee '$LOG'"

cat <<EOF
Launched $SESSION (machine=$MACHINE device=$DEVICE), waiting up to ${WAIT_FOR_IDLE}s for an
idle GPU before it measures. Safe to disconnect now.

Watch:    tmux attach -t $SESSION        (detach with Ctrl-b then d)
          tail -f logs/calgpu_${MACHINE}.out
Status:   $0 --status
Kill:     tmux kill-session -t $SESSION

--status reports whether the run was RECORDABLE, which is the thing to check before
transcribing anything: a contended run still completes and still writes CSVs, but its
constants must not reach include/gpu_machine.hpp.
EOF
