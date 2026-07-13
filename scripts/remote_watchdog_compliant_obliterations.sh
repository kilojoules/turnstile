#!/bin/bash
# Watchdog for run_compliant_obliterations_remote.sh.
# Usage: ./remote_watchdog_compliant_obliterations.sh <instance_id> <train_pid>
# Polls every 30s for up to MAX_RUNTIME seconds, then force-destroys.
set -u

INSTANCE_ID="$1"
TRAIN_PID="$2"
MARKER_DIR="experiments/compliant_obliterations_v1"
MAX_RUNTIME="${MAX_RUNTIME:-25200}"  # 7 hr (4 phases × ~1.5 hr each)
GRACE=600
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15"

start=$(date +%s)
echo "[watchdog] instance=$INSTANCE_ID pid=$TRAIN_PID started $(date -Is)"

destroy() {
    echo "[watchdog] destroying instance $INSTANCE_ID"
    echo "y" | vastai destroy instance "$INSTANCE_ID" --raw 2>&1 || true
    # curl fallback
    curl -s -X DELETE \
        "https://console.vast.ai/api/v0/instances/$INSTANCE_ID/" \
        -H "Authorization: Bearer $(cat ~/.fastai_key)" || true
    echo "[watchdog] destroy sent $(date -Is)"
}

while true; do
    now=$(date +%s)
    elapsed=$(( now - start ))

    if [ $elapsed -ge $MAX_RUNTIME ]; then
        echo "[watchdog] MAX_RUNTIME reached, destroying"
        destroy; exit 1
    fi

    # Check done/failed markers
    DONE=$(ssh $SSH_OPTS "$SSH_HOST" \
        "ls /workspace/turnstile/$MARKER_DIR/.training_done 2>/dev/null && echo DONE; \
         ls /workspace/turnstile/$MARKER_DIR/.training_failed 2>/dev/null && echo FAILED" 2>/dev/null)

    if echo "$DONE" | grep -q "DONE"; then
        echo "[watchdog] DONE at $(date -Is)"
        destroy; exit 0
    fi
    if echo "$DONE" | grep -q "FAILED"; then
        echo "[watchdog] FAILED at $(date -Is)"
        destroy; exit 1
    fi

    # Check if train PID is still alive
    ALIVE=$(ssh $SSH_OPTS "$SSH_HOST" \
        "kill -0 $TRAIN_PID 2>/dev/null && echo ALIVE" 2>/dev/null)
    if ! echo "$ALIVE" | grep -q "ALIVE"; then
        echo "[watchdog] PID $TRAIN_PID dead, checking markers..."
        sleep "$GRACE"
        DONE2=$(ssh $SSH_OPTS "$SSH_HOST" \
            "ls /workspace/turnstile/$MARKER_DIR/.training_done 2>/dev/null && echo DONE; \
             ls /workspace/turnstile/$MARKER_DIR/.training_failed 2>/dev/null && echo FAILED" 2>/dev/null)
        if echo "$DONE2" | grep -q "DONE"; then
            echo "[watchdog] DONE (after grace) at $(date -Is)"
            destroy; exit 0
        fi
        echo "[watchdog] PID dead, no DONE marker, destroying"
        destroy; exit 1
    fi

    sleep 30
done
