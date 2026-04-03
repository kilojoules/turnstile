#!/bin/bash
# Resilient babysitter for the A/B experiment suite.
# - Checks every 5 minutes
# - Syncs data
# - If the suite script crashes, restarts it from where it left off
# - If the instance dies, logs it
# - Destroys instance when suite completes
# - Logs everything to a local file for review
set -u

SSH_HOST="root@ssh9.vast.ai"
SSH_PORT=11234
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15 -o ServerAliveInterval=30"
SSH="ssh -p $SSH_PORT $SSH_OPTS $SSH_HOST"
REMOTE_DIR="/workspace/turnstile"
LOCAL_BASE="/Users/julianquick/portfolio_copy/turnstile/experiments"
LOCAL_LOG="/Users/julianquick/portfolio_copy/turnstile/experiments/babysitter.log"
INSTANCE_ID=33811234
INTERVAL=300

EXPERIMENTS="stealth_s42 control_s42 stealth_s123 control_s123 stealth_s456 control_s456 stealth_hard_s42 control_hard_s42 stealth_hard_s123 control_hard_s123 stealth_hard_s456 control_hard_s456"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOCAL_LOG"
}

sync_data() {
    for exp in $EXPERIMENTS; do
        if $SSH "test -d $REMOTE_DIR/experiments/$exp" 2>/dev/null; then
            mkdir -p "$LOCAL_BASE/$exp"
            rsync -az -e "ssh -p $SSH_PORT $SSH_OPTS" \
                "$SSH_HOST:$REMOTE_DIR/experiments/$exp/" \
                "$LOCAL_BASE/$exp/" 2>/dev/null
        fi
    done
    rsync -az -e "ssh -p $SSH_PORT $SSH_OPTS" \
        "$SSH_HOST:$REMOTE_DIR/loop_suite.log" \
        "$LOCAL_BASE/suite_log.txt" 2>/dev/null

    # Count completed rounds per experiment
    local summary=""
    for exp in $EXPERIMENTS; do
        local mf="$LOCAL_BASE/$exp/metrics.jsonl"
        if [ -f "$mf" ]; then
            local rounds=$(wc -l < "$mf" | tr -d ' ')
            summary="$summary $exp:${rounds}r"
        fi
    done
    log "  [sync] $summary"
}

check_ssh() {
    $SSH "echo ok" 2>/dev/null
    return $?
}

check_suite() {
    # Returns: 0=running, 1=crashed, 2=complete, 3=ssh failed
    if ! check_ssh; then
        return 3
    fi

    local procs=$($SSH "ps aux | grep -E 'run_experiment_suite|stealth_loop' | grep -v grep | wc -l" 2>/dev/null)
    if [ "$procs" = "0" ] 2>/dev/null || [ -z "$procs" ]; then
        local last=$($SSH "tail -3 $REMOTE_DIR/loop_suite.log" 2>/dev/null)
        if echo "$last" | grep -q "SUITE COMPLETE"; then
            return 2
        else
            return 1
        fi
    fi
    return 0
}

get_status() {
    $SSH "grep -E 'RUN:|Metrics|ROUND [0-9]' $REMOTE_DIR/loop_suite.log 2>/dev/null | tail -3" 2>/dev/null
}

restart_suite() {
    log "  [restart] Attempting to restart suite..."
    # Find which experiment to resume from
    local last_complete=$($SSH "grep 'Finished:' $REMOTE_DIR/loop_suite.log 2>/dev/null | tail -1" 2>/dev/null)
    log "  [restart] Last completed: $last_complete"

    # Just restart the whole suite — it skips experiments that already have 15 rounds
    $SSH "cd $REMOTE_DIR && nohup bash scripts/run_experiment_suite.sh >> loop_suite.log 2>&1 &" 2>/dev/null
    log "  [restart] Suite relaunched (appending to log)"
}

destroy_instance() {
    log "  [teardown] Destroying instance $INSTANCE_ID..."
    echo "n" | vastai destroy instance $INSTANCE_ID 2>&1 | grep -v "Update"
    log "  [teardown] Instance destroyed."
}

# --- Main loop ---
log "=== RESILIENT BABYSITTER STARTED ==="
log "  Instance: $INSTANCE_ID"
log "  Experiments: $EXPERIMENTS"

consecutive_ssh_failures=0
consecutive_crashes=0

while true; do
    check_suite
    status=$?

    case $status in
        0)  # Running
            consecutive_ssh_failures=0
            consecutive_crashes=0
            local_status=$(get_status)
            log "  [ok] Suite running. $local_status"
            sync_data
            ;;
        1)  # Crashed
            consecutive_crashes=$((consecutive_crashes + 1))
            log "  [WARN] Suite crashed (attempt $consecutive_crashes)"
            sync_data

            if [ $consecutive_crashes -le 3 ]; then
                restart_suite
                sleep 60  # Give it time to start
            else
                log "  [FATAL] 3 consecutive crashes. Stopping. SSH: ssh -p $SSH_PORT $SSH_HOST"
                sync_data
                exit 1
            fi
            ;;
        2)  # Complete
            log "  [DONE] Suite complete!"
            sync_data
            sync_data  # Double sync for safety

            log "  Final metrics:"
            for exp in $EXPERIMENTS; do
                local mf="$LOCAL_BASE/$exp/metrics.jsonl"
                if [ -f "$mf" ]; then
                    local rounds=$(wc -l < "$mf" | tr -d ' ')
                    local mean_asr=$(python3 -c "
import json
asrs = [json.loads(l)['asr'] for l in open('$mf')]
print(f'{sum(asrs)/len(asrs):.1%}') if asrs else print('?')
" 2>/dev/null)
                    log "    $exp: ${rounds} rounds, mean_ASR=$mean_asr"
                fi
            done

            destroy_instance
            log "=== BABYSITTER DONE ==="
            exit 0
            ;;
        3)  # SSH failed
            consecutive_ssh_failures=$((consecutive_ssh_failures + 1))
            log "  [SSH] Connection failed (attempt $consecutive_ssh_failures)"

            if [ $consecutive_ssh_failures -ge 6 ]; then
                log "  [FATAL] 6 consecutive SSH failures. Instance may be dead."
                log "  Check: vastai show instances"
                exit 1
            fi
            ;;
    esac

    sleep $INTERVAL
done
