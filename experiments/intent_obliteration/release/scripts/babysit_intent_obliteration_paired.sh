#!/bin/bash
# Watch the paired-obliteration pipeline on the remote instance.
# Periodically pulls progress, syncs final outputs, and (on success or hard
# failure) destroys the instance.
#
# Reads instance info from experiments/intent_obliteration_paired/instance.txt
# (written by scripts/launch_intent_obliteration_paired.sh).
#
# Stdout is line-by-line so this script can be run under Monitor.
#
# Exits:
#   0  pipeline completed cleanly, instance destroyed
#   1  pipeline failed, instance destroyed (per "destroy when done" mandate)
#   2  could not reach the instance for too long; left alone for manual triage
set -u

LOCAL_TURNSTILE="/Users/julianquick/portfolio_copy/turnstile"
INSTANCE_FILE="$LOCAL_TURNSTILE/experiments/intent_obliteration_paired/instance.txt"
LOCAL_OUT="$LOCAL_TURNSTILE/experiments/intent_obliteration_paired"

if [ ! -f "$INSTANCE_FILE" ]; then
    echo "FATAL no instance file at $INSTANCE_FILE"
    exit 2
fi
read -r INSTANCE_ID SSH_HOST SSH_PORT < "$INSTANCE_FILE"
SSH="ssh -p $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=20 root@$SSH_HOST"
RSYNC_E="ssh -p $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=20"
REMOTE_OUT="/workspace/turnstile/experiments/intent_obliteration_paired"

echo "BABYSIT instance=$INSTANCE_ID host=$SSH_HOST port=$SSH_PORT"

POLL_INTERVAL=180   # 3 min
N_SSH_FAILS=0
MAX_SSH_FAILS=10    # ~30 min of consecutive SSH failures => give up

sync_progress() {
    rsync -az -e "$RSYNC_E" \
        --include="*.log" --include="*.json" --include="*.jsonl" \
        --include=".step*_done" --include=".mission*" --include="*/" \
        --exclude="*" \
        "root@$SSH_HOST:$REMOTE_OUT/" "$LOCAL_OUT/" 2>/dev/null || true
    rsync -az -e "$RSYNC_E" \
        "root@$SSH_HOST:/workspace/run.log" "$LOCAL_OUT/run.log" 2>/dev/null || true
}

sync_full_results() {
    echo "SYNCING full outputs (including .pt) ..."
    rsync -az -e "$RSYNC_E" \
        "root@$SSH_HOST:$REMOTE_OUT/" "$LOCAL_OUT/" 2>/dev/null || true
    rsync -az -e "$RSYNC_E" \
        "root@$SSH_HOST:/workspace/run.log" "$LOCAL_OUT/run.log" 2>/dev/null || true
}

destroy_instance() {
    echo "DESTROYING instance $INSTANCE_ID"
    echo "n" | vastai destroy instance "$INSTANCE_ID" 2>&1 | grep -v "Update\|selected" || true
}

while true; do
    # Try to reach the instance
    HEALTH=$($SSH "ls $REMOTE_OUT/.mission_complete $REMOTE_OUT/.mission_failed 2>/dev/null; echo done; pgrep -f run_remote.sh | head -1" 2>/dev/null)
    if [ -z "$HEALTH" ]; then
        N_SSH_FAILS=$((N_SSH_FAILS + 1))
        echo "PROGRESS ssh-fail $N_SSH_FAILS/$MAX_SSH_FAILS"
        if [ "$N_SSH_FAILS" -ge "$MAX_SSH_FAILS" ]; then
            echo "FATAL too many SSH failures; leaving instance for manual triage"
            exit 2
        fi
        sleep "$POLL_INTERVAL"
        continue
    fi
    N_SSH_FAILS=0

    sync_progress

    if echo "$HEALTH" | grep -q ".mission_complete"; then
        echo "PROGRESS mission_complete detected"
        sync_full_results
        destroy_instance
        echo "EXIT success"
        exit 0
    fi
    if echo "$HEALTH" | grep -q ".mission_failed"; then
        echo "PROGRESS mission_failed detected"
        sync_full_results
        destroy_instance
        echo "EXIT failure"
        exit 1
    fi

    # Process alive check (HEALTH includes a pgrep line; 'done' marker is in there)
    PID=$(echo "$HEALTH" | tail -1)
    if [ -z "$PID" ] || [ "$PID" = "done" ]; then
        # No process and no marker => unexpected exit. Sync, dump tail, destroy.
        echo "PROGRESS process gone but no marker; treating as failure"
        TAIL=$($SSH "tail -40 /workspace/run.log" 2>/dev/null || echo "(no log)")
        echo "$TAIL" | sed 's/^/  log| /'
        sync_full_results
        destroy_instance
        echo "EXIT crashed"
        exit 1
    fi

    # Show progress: the latest "STEP" or "[progress]" line
    LATEST=$($SSH "tail -3 /workspace/run.log; for f in $REMOTE_OUT/.step{1,2,3,4}_done; do [ -f \$f ] && echo done:\$(basename \$f); done" 2>/dev/null)
    echo "$LATEST" | sed 's/^/  /'
    sleep "$POLL_INTERVAL"
done
