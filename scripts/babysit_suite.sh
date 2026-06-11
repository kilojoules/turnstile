#!/bin/bash
# Babysit the full A/B experiment suite.
# Monitors the master suite script, syncs all experiment dirs periodically,
# and destroys the instance when the suite completes.
set -u

SSH_HOST="root@ssh9.vast.ai"
SSH_PORT=11234
SSH="ssh -p $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15 $SSH_HOST"
REMOTE_DIR="/workspace/turnstile"
LOCAL_BASE="/Users/julianquick/portfolio_copy/turnstile/experiments"
INSTANCE_ID=33811234
INTERVAL=300  # 5 minutes
LOG_FILE="loop_suite.log"

EXPERIMENTS="stealth_s42 control_s42 stealth_s123 control_s123 stealth_s456 control_s456"

sync_data() {
    echo "  [sync] Pulling data..."
    for exp in $EXPERIMENTS; do
        if $SSH "test -d $REMOTE_DIR/experiments/$exp" 2>/dev/null; then
            mkdir -p "$LOCAL_BASE/$exp"
            rsync -az -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
                "$SSH_HOST:$REMOTE_DIR/experiments/$exp/metrics.jsonl" \
                "$LOCAL_BASE/$exp/metrics.jsonl" 2>/dev/null
            rsync -az -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
                "$SSH_HOST:$REMOTE_DIR/experiments/$exp/rounds/" \
                "$LOCAL_BASE/$exp/rounds/" 2>/dev/null
        fi
    done
    rsync -az -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
        "$SSH_HOST:$REMOTE_DIR/$LOG_FILE" \
        "$LOCAL_BASE/suite_log.txt" 2>/dev/null

    # Show progress
    for exp in $EXPERIMENTS; do
        local mf="$LOCAL_BASE/$exp/metrics.jsonl"
        if [ -f "$mf" ]; then
            local rounds=$(wc -l < "$mf" | tr -d ' ')
            echo "    $exp: $rounds rounds"
        fi
    done
    echo "  [sync] Done."
}

check_health() {
    local procs=$($SSH "ps aux | grep -E 'run_experiment_suite|stealth_loop' | grep -v grep | wc -l" 2>/dev/null)
    if [ "$procs" = "0" ] 2>/dev/null || [ -z "$procs" ]; then
        echo "  [WARN] No suite processes found!"
        local last_line=$($SSH "tail -3 $REMOTE_DIR/$LOG_FILE" 2>/dev/null)
        if echo "$last_line" | grep -q "SUITE COMPLETE"; then
            echo "  [OK] Suite finished successfully."
            return 2
        else
            echo "  [ERROR] Suite crashed. Last log lines:"
            $SSH "tail -10 $REMOTE_DIR/$LOG_FILE" 2>/dev/null
            return 1
        fi
    fi

    # Show latest status
    local latest=$($SSH "grep -E 'RUN:|Metrics|ROUND [0-9]' $REMOTE_DIR/$LOG_FILE 2>/dev/null | tail -3" 2>/dev/null)
    echo "  [status] $latest"
    return 0
}

destroy_instance() {
    echo "  [teardown] Destroying instance $INSTANCE_ID..."
    echo "n" | vastai destroy instance $INSTANCE_ID 2>&1 | grep -v "Update"
    echo "  [teardown] Instance destroyed."
}

echo "=== SUITE BABYSITTER STARTED ==="
echo "  Instance: $INSTANCE_ID"
echo "  Experiments: $EXPERIMENTS"
echo "  Check interval: ${INTERVAL}s"
echo ""

while true; do
    echo "[$(date '+%H:%M:%S')] Checking..."

    check_health
    status=$?

    sync_data

    if [ $status -eq 2 ]; then
        echo ""
        echo "=== SUITE COMPLETE ==="
        echo "  Final sync..."
        sync_data
        destroy_instance
        echo "=== BABYSITTER DONE ==="
        exit 0
    fi

    if [ $status -eq 1 ]; then
        echo "  [!] Suite crashed. Data synced. NOT destroying instance."
        echo "  SSH: ssh -p $SSH_PORT $SSH_HOST"
        exit 1
    fi

    echo "  Sleeping ${INTERVAL}s..."
    echo ""
    sleep $INTERVAL
done
