#!/bin/bash
# Babysit the layer-31 clamping experiment.
# Checks health, syncs results, destroys instance when done.
#
# Usage: bash scripts/babysit_clamp.sh
set -u

INSTANCE_FILE="/Users/julianquick/portfolio_copy/turnstile/experiments/clamp_instance.txt"
LOCAL_BASE="/Users/julianquick/portfolio_copy/turnstile/experiments"
LOCAL_DIR="$LOCAL_BASE/clamp_v1"
LOCAL_LOG="$LOCAL_BASE/babysit_clamp.log"
REMOTE_DIR="/workspace/turnstile"
EXPERIMENT="clamp_v1"
LOG_FILE="experiments/clamp.log"
INTERVAL=180  # 3 minutes (experiment runs faster per condition than self-play)
MAX_SSH_FAILURES=20  # ~1 hour of failures before giving up
EXPECTED_CONDITIONS=9  # number of conditions in CONDITIONS list

# Read instance info
if [ ! -f "$INSTANCE_FILE" ]; then
    echo "[ERROR] No instance file at $INSTANCE_FILE"
    echo "  Run scripts/launch_clamp.sh first"
    exit 1
fi

read INST HOST PORT < "$INSTANCE_FILE"
echo "Instance: $INST, Host: $HOST, Port: $PORT"

ssh_failures=0

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOCAL_LOG"
}

do_ssh() {
    ssh -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15 root@$HOST "$@" 2>/dev/null
}

sync_data() {
    mkdir -p "$LOCAL_DIR"
    mkdir -p "$LOCAL_BASE/clamp_v1_singleturn"
    # Sync multi-turn metrics
    rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15" \
        "root@$HOST:$REMOTE_DIR/experiments/$EXPERIMENT/" \
        "$LOCAL_DIR/" 2>/dev/null
    # Sync multi-turn log
    rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15" \
        "root@$HOST:$REMOTE_DIR/$LOG_FILE" \
        "$LOCAL_DIR/clamp.log" 2>/dev/null
    # Sync single-turn results
    rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15" \
        "root@$HOST:$REMOTE_DIR/experiments/clamp_v1_singleturn/" \
        "$LOCAL_BASE/clamp_v1_singleturn/" 2>/dev/null
}

show_progress() {
    if [ -f "$LOCAL_DIR/metrics.jsonl" ]; then
        python3 -c "
import json
with open('$LOCAL_DIR/metrics.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]
if not lines:
    print('  (no results yet)')
else:
    print(f'  Conditions completed: {len(lines)}/$EXPECTED_CONDITIONS')
    for r in lines:
        layer = str(r['layer']) if r['layer'] is not None else '-'
        print(f'    {r[\"condition\"]:>20s}  L={layer:>2s}  a={r[\"alpha\"]:>+3.0f}  '
              f'ASR={r[\"asr\"]:.1%}  ({r[\"wins\"]}/{r[\"total\"]})')
" 2>/dev/null || echo "  (parse error)"
    fi
}

check_done() {
    # Check both multi-turn and single-turn experiments
    local mt_done=0
    local st_done=0
    if [ -f "$LOCAL_DIR/metrics.jsonl" ]; then
        local n=$(wc -l < "$LOCAL_DIR/metrics.jsonl" | tr -d ' ')
        [ "$n" -ge "$EXPECTED_CONDITIONS" ] && mt_done=1
    fi
    if [ -f "$LOCAL_BASE/clamp_v1_singleturn/metrics.jsonl" ]; then
        local sn=$(wc -l < "$LOCAL_BASE/clamp_v1_singleturn/metrics.jsonl" | tr -d ' ')
        [ "$sn" -ge 8 ] && st_done=1
    fi
    # Done when both are complete
    [ "$mt_done" -eq 1 ] && [ "$st_done" -eq 1 ] && return 0
    return 1  # not done
}

check_health() {
    # Check if any clamp process is alive (multi-turn or single-turn or watcher)
    local procs=$(do_ssh "ps aux | grep -E 'clamp_experiment|clamp_singleturn|run_singleturn' | grep -v grep | wc -l")
    if [ "${procs:-0}" = "0" ]; then
        # Process gone — check if it finished or crashed
        local last=$(do_ssh "tail -3 $REMOTE_DIR/$LOG_FILE")
        if echo "$last" | grep -q "EXPERIMENT COMPLETE"; then
            return 2  # done
        else
            log "  [WARN] Process not running. Last log:"
            do_ssh "tail -10 $REMOTE_DIR/$LOG_FILE" | while read line; do log "    $line"; done
            return 1  # crashed
        fi
    fi

    # Show latest progress from remote log
    local latest=$(do_ssh "grep -E 'CONDITION:|ASR=|Extracting|goals' $REMOTE_DIR/$LOG_FILE | tail -3")
    log "  [status] $latest"
    return 0
}

destroy_instance() {
    log "  [teardown] Destroying instance $INST..."
    echo "n" | vastai destroy instance $INST 2>&1 | grep -v "Update"
    log "  [teardown] Instance destroyed."
}

# === Main loop ===
log "=== CLAMP BABYSITTER STARTED ==="
log "  Instance: $INST ($HOST:$PORT)"
log "  Check interval: ${INTERVAL}s"
log "  Expected conditions: $EXPECTED_CONDITIONS"
log ""

while true; do
    log "Checking..."

    # SSH connectivity test
    if ! do_ssh "echo ok" >/dev/null; then
        ssh_failures=$((ssh_failures + 1))
        log "  [SSH] Failed ($ssh_failures/$MAX_SSH_FAILURES)"
        if [ $ssh_failures -ge $MAX_SSH_FAILURES ]; then
            log "  [FATAL] SSH failed for too long. Instance may be dead."
            sync_data 2>/dev/null
            exit 1
        fi
        sleep $INTERVAL
        continue
    fi
    ssh_failures=0

    # Sync data
    sync_data

    # Check completion
    check_health
    status=$?

    # Show local progress
    show_progress

    if [ $status -eq 2 ] || check_done; then
        log ""
        log "=== EXPERIMENT COMPLETE ==="
        sync_data
        sync_data  # double sync for safety

        log ""
        log "=== FINAL RESULTS ==="
        show_progress

        # Check if results.json exists (written at very end)
        if [ -f "$LOCAL_DIR/results.json" ]; then
            log "  results.json synced successfully"
        else
            log "  [WARN] results.json not found, pulling again..."
            sleep 5
            sync_data
        fi

        destroy_instance
        log "=== BABYSITTER DONE ==="
        exit 0
    fi

    if [ $status -eq 1 ]; then
        log "  [!] Experiment crashed. Data synced. NOT destroying (investigate)."
        log "  SSH: ssh -p $PORT root@$HOST"
        log "  Log: ssh -p $PORT root@$HOST 'tail -50 $REMOTE_DIR/$LOG_FILE'"
        exit 1
    fi

    log "  Sleeping ${INTERVAL}s..."
    log ""
    sleep $INTERVAL
done
