#!/bin/bash
# Babysit the remote experiment: check health, sync data, teardown when done.
# Usage: bash scripts/babysit.sh
set -u

SSH_HOST="root@ssh9.vast.ai"
SSH_PORT=11234
SSH="ssh -p $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15 $SSH_HOST"
SCP="scp -P $SSH_PORT -o StrictHostKeyChecking=no"
RSYNC="rsync -az -e 'ssh -p $SSH_PORT -o StrictHostKeyChecking=no'"
REMOTE_DIR="/workspace/turnstile"
LOCAL_DIR="/Users/julianquick/portfolio_copy/turnstile/experiments/stealth_jbb_v1"
INSTANCE_ID=33811234
EXPERIMENT="stealth_jbb_v1"
LOG_FILE="loop_stealth.log"
INTERVAL=300  # 5 minutes

# Follow-up: control arm (same adapter, no stealth) for A/B comparison
FOLLOWUP_SCRIPT="/tmp/run_control.sh"
FOLLOWUP_LOG="loop_control.log"
FOLLOWUP_EXPERIMENT="control_jbb_v1"
FOLLOWUP_LOCAL_DIR="/Users/julianquick/portfolio_copy/turnstile/experiments/control_jbb_v1"

sync_data() {
    echo "  [sync] Pulling data..."
    rsync -az -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
        "$SSH_HOST:$REMOTE_DIR/experiments/$EXPERIMENT/" \
        "$LOCAL_DIR/" 2>/dev/null
    rsync -az -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
        "$SSH_HOST:$REMOTE_DIR/$LOG_FILE" \
        "$LOCAL_DIR/loop.log" 2>/dev/null
    local count=$(cat "$LOCAL_DIR/metrics.jsonl" 2>/dev/null | wc -l | tr -d ' ')
    echo "  [sync] Done. Rounds synced: $count"
}

check_health() {
    # Check if any experiment process is alive (loop, training, or wrapper script)
    local procs=$($SSH "ps aux | grep -E '[t]urnstile|/tmp/run_|train_lora' | grep -v grep | wc -l" 2>/dev/null)
    if [ "$procs" = "0" ] 2>/dev/null || [ -z "$procs" ]; then
        echo "  [WARN] No experiment processes found!"
        # Check if it finished or crashed
        local last_line=$($SSH "tail -3 $REMOTE_DIR/$LOG_FILE" 2>/dev/null)
        if echo "$last_line" | grep -q "LOOP COMPLETE\|^DONE:"; then
            echo "  [OK] Experiment finished successfully."
            return 2  # signal: done
        else
            echo "  [ERROR] Loop crashed. Last log lines:"
            $SSH "tail -10 $REMOTE_DIR/$LOG_FILE" 2>/dev/null
            return 1  # signal: error
        fi
    fi

    # Check for OOM or errors
    local errors=$($SSH "grep -c -i 'OutOfMemoryError\|CUDA out of memory\|RuntimeError\|Traceback' $REMOTE_DIR/$LOG_FILE 2>/dev/null" 2>/dev/null)
    if [ "$errors" != "0" ] 2>/dev/null && [ -n "$errors" ]; then
        echo "  [WARN] Found $errors error lines in log"
        $SSH "grep -i 'OutOfMemoryError\|CUDA out of memory' $REMOTE_DIR/$LOG_FILE 2>/dev/null" 2>/dev/null
    fi

    # Show latest progress
    local latest=$($SSH "grep -E 'Conv |PHASE|Metrics|ROUND' $REMOTE_DIR/$LOG_FILE 2>/dev/null | tail -3" 2>/dev/null)
    echo "  [status] $latest"
    return 0
}

destroy_instance() {
    echo "  [teardown] Destroying instance $INSTANCE_ID..."
    echo "n" | vastai destroy instance $INSTANCE_ID 2>&1 | grep -v "Update"
    echo "  [teardown] Instance destroyed."
}

echo "=== BABYSITTER STARTED ==="
echo "  Instance: $INSTANCE_ID"
echo "  Check interval: ${INTERVAL}s"
echo "  Local sync dir: $LOCAL_DIR"
echo ""

while true; do
    echo "[$(date '+%H:%M:%S')] Checking..."

    check_health
    status=$?

    sync_data

    if [ $status -eq 2 ]; then
        echo ""
        echo "=== EXPERIMENT COMPLETE ==="
        echo "  Final sync..."
        sync_data
        echo "  Metrics:"
        cat "$LOCAL_DIR/metrics.jsonl" 2>/dev/null | python3 -c "
import json, sys
for line in sys.stdin:
    m = json.loads(line)
    breach = f'  breach={m[\"mean_turn_of_breach\"]:.1f}' if m.get('mean_turn_of_breach') else ''
    print(f'  Round {m[\"round\"]:2d}: ASR={m[\"asr\"]:.1%}  wins={m[\"wins\"]}/{m[\"candidates\"]}{breach}')
" 2>/dev/null || echo "  (no metrics)"

        # Launch follow-up if script exists
        if [ -n "$FOLLOWUP_SCRIPT" ]; then
            echo ""
            echo "=== LAUNCHING FOLLOW-UP: $FOLLOWUP_SCRIPT ==="
            $SSH "nohup bash $FOLLOWUP_SCRIPT > /workspace/turnstile/$FOLLOWUP_LOG 2>&1 &"
            echo "  Follow-up launched. Switching to monitor $FOLLOWUP_EXPERIMENT..."

            # Switch babysitter to follow-up
            LOCAL_DIR="$FOLLOWUP_LOCAL_DIR"
            EXPERIMENT="$FOLLOWUP_EXPERIMENT"
            LOG_FILE="$FOLLOWUP_LOG"
            mkdir -p "$LOCAL_DIR"
            FOLLOWUP_SCRIPT=""  # don't chain again
            echo "  Sleeping 60s for follow-up to start..."
            sleep 60
            continue
        fi

        destroy_instance
        echo "=== BABYSITTER DONE ==="
        exit 0
    fi

    if [ $status -eq 1 ]; then
        echo "  [!] Loop crashed. Data synced. NOT destroying instance (investigate)."
        echo "  SSH: ssh -p $SSH_PORT $SSH_HOST"
        exit 1
    fi

    echo "  Sleeping ${INTERVAL}s..."
    echo ""
    sleep $INTERVAL
done
