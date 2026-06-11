#!/bin/bash
# Watchdog: prevents idle Vast.ai instances from billing forever.
#
# Polls a remote instance periodically. Destroys it if:
#   - Target process stops AND no progress for GRACE_MIN
#   - Log file unchanged for STALL_MIN (stuck process)
#   - Max runtime MAX_HOURS exceeded
#   - SSH fails MAX_SSH_FAILS in a row
#
# Always syncs results before destroying.
#
# Usage:
#   bash scripts/watchdog.sh <instance_file> <remote_work_dir> <proc_pattern> [sync_paths...]
#
# Example:
#   bash scripts/watchdog.sh \
#       experiments/pooled_sweep_instance.txt \
#       /workspace/turnstile \
#       'extract_pooled|outcome_probe|sae_ablation' \
#       experiments/pooled_hs \
#       experiments/outcome_probe_v1
set -u

INSTANCE_FILE="${1:?need instance file (INST HOST PORT)}"
REMOTE_DIR="${2:?need remote work dir}"
PROC_PATTERN="${3:?need process grep pattern}"
shift 3
SYNC_PATHS=("$@")

# Tunables
INTERVAL=${WATCHDOG_INTERVAL:-180}      # poll every 3 min
GRACE_MIN=${WATCHDOG_GRACE_MIN:-10}     # min idle after process exit before destroy
STALL_MIN=${WATCHDOG_STALL_MIN:-20}     # min without log change = stuck
MAX_HOURS=${WATCHDOG_MAX_HOURS:-12}     # hard cap total runtime
MAX_SSH_FAILS=${WATCHDOG_MAX_SSH_FAILS:-20}

[ ! -f "$INSTANCE_FILE" ] && echo "[ERROR] instance file missing: $INSTANCE_FILE" && exit 1
read INST HOST PORT < "$INSTANCE_FILE"

LOCAL_BASE="/Users/julianquick/portfolio_copy/turnstile"
WATCHDOG_LOG="$LOCAL_BASE/experiments/watchdog_${INST}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$WATCHDOG_LOG"
}

do_ssh() {
    ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=15 "root@$HOST" "$@" 2>/dev/null
}

sync_all() {
    log "  [sync] syncing ${#SYNC_PATHS[@]} paths..."
    for p in "${SYNC_PATHS[@]}"; do
        mkdir -p "$LOCAL_BASE/$p"
        rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=30" \
            "root@$HOST:$REMOTE_DIR/$p/" "$LOCAL_BASE/$p/" 2>/dev/null || \
            log "  [sync] WARN: failed to sync $p"
    done
}

destroy_and_exit() {
    local reason="$1"
    log "=== DESTROYING INSTANCE: $reason ==="
    sync_all
    sync_all  # twice for safety
    log "  [teardown] destroying $INST..."
    echo "n" | vastai destroy instance "$INST" 2>&1 | grep -v "Update" | tee -a "$WATCHDOG_LOG"
    log "=== WATCHDOG EXIT ==="
    exit "${2:-0}"
}

# State
ssh_fails=0
idle_since=""       # epoch when process first seen stopped
last_log_mtime=""   # remote log mtime tracker
last_log_size=""
log_stall_since=""
start_time=$(date +%s)

log "=== WATCHDOG START ==="
log "  instance: $INST ($HOST:$PORT)"
log "  remote_dir: $REMOTE_DIR"
log "  proc_pattern: $PROC_PATTERN"
log "  sync_paths: ${SYNC_PATHS[*]}"
log "  interval: ${INTERVAL}s  grace: ${GRACE_MIN}m  stall: ${STALL_MIN}m  max: ${MAX_HOURS}h"

while true; do
    now=$(date +%s)
    elapsed_hours=$(( (now - start_time) / 3600 ))

    # Hard runtime cap
    if [ "$elapsed_hours" -ge "$MAX_HOURS" ]; then
        destroy_and_exit "max runtime ${MAX_HOURS}h exceeded"
    fi

    # SSH liveness
    if ! do_ssh "echo ok" >/dev/null; then
        ssh_fails=$((ssh_fails + 1))
        log "  [ssh] FAIL ($ssh_fails/$MAX_SSH_FAILS)"
        if [ "$ssh_fails" -ge "$MAX_SSH_FAILS" ]; then
            log "  [ssh] dead — attempting final sync"
            sync_all || true
            destroy_and_exit "ssh unreachable" 1
        fi
        sleep "$INTERVAL"
        continue
    fi
    ssh_fails=0

    # Process check
    procs=$(do_ssh "ps aux | grep -E '$PROC_PATTERN' | grep -v grep | wc -l" | tr -d ' ')
    procs=${procs:-0}

    # Log activity check (find most-recent-modified .log in experiments/)
    log_info=$(do_ssh "ls -t $REMOTE_DIR/experiments/*.log 2>/dev/null | head -1 | xargs -r stat -c '%Y %s'")
    cur_mtime=$(echo "$log_info" | awk '{print $1}')
    cur_size=$(echo "$log_info" | awk '{print $2}')

    if [ -n "$cur_mtime" ] && [ "$cur_mtime" = "$last_log_mtime" ] && [ "$cur_size" = "$last_log_size" ]; then
        # Log frozen
        if [ -z "$log_stall_since" ]; then
            log_stall_since=$now
        fi
        stall_min=$(( (now - log_stall_since) / 60 ))
        if [ "$stall_min" -ge "$STALL_MIN" ] && [ "$procs" -gt 0 ]; then
            log "  [stall] log frozen ${stall_min}m with $procs procs — killing"
            destroy_and_exit "log stall ${stall_min}m"
        fi
    else
        log_stall_since=""
        last_log_mtime=$cur_mtime
        last_log_size=$cur_size
    fi

    if [ "$procs" = "0" ]; then
        # Process stopped
        if [ -z "$idle_since" ]; then
            idle_since=$now
            log "  [idle] no procs matching '$PROC_PATTERN' — grace period starts"
        fi
        idle_min=$(( (now - idle_since) / 60 ))
        log "  [idle] ${idle_min}m since process exit (grace=${GRACE_MIN}m)"

        if [ "$idle_min" -ge "$GRACE_MIN" ]; then
            destroy_and_exit "process stopped, idle ${idle_min}m (elapsed ${elapsed_hours}h)"
        fi
    else
        # Running
        idle_since=""
        log "  [ok] $procs procs running, ${elapsed_hours}h elapsed"
    fi

    sync_all
    sleep "$INTERVAL"
done
