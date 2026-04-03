#!/bin/bash
# Babysit 3 parallel instances running different seeds.
# Syncs data from all, destroys each when its seed completes.
set -u

LOCAL_BASE="/Users/julianquick/portfolio_copy/turnstile/experiments"
LOCAL_LOG="$LOCAL_BASE/babysitter_parallel.log"
INTERVAL=300
INSTANCES_FILE="$LOCAL_BASE/parallel_instances.txt"

EXPERIMENTS_PER_SEED="stealth control stealth_hard control_hard"

GPU_IDLE_THRESHOLD=20  # minutes of <1% GPU before killing
declare -A IDLE_COUNT   # per-instance idle check counter

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOCAL_LOG"
}

check_gpu_idle() {
    local INST=$1 HOST=$2 PORT=$3
    local SSH="ssh -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15 root@$HOST"
    local gpu_util=$($SSH "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits" 2>/dev/null | tr -d ' ')

    if [ -z "$gpu_util" ]; then return; fi

    if [ "$gpu_util" -lt 1 ] 2>/dev/null; then
        IDLE_COUNT[$INST]=$(( ${IDLE_COUNT[$INST]:-0} + 1 ))
        local idle_min=$(( ${IDLE_COUNT[$INST]} * INTERVAL / 60 ))
        log "  [s$4] GPU idle ($gpu_util%), ${idle_min}min / ${GPU_IDLE_THRESHOLD}min"
        if [ $idle_min -ge $GPU_IDLE_THRESHOLD ]; then
            log "  [s$4] GPU IDLE FOR ${GPU_IDLE_THRESHOLD}+ MINUTES — destroying instance $INST"
            sync_instance $INST $4 $HOST $PORT
            destroy_instance $INST
            sed -i '' "/^$INST /d" "$INSTANCES_FILE"
        fi
    else
        IDLE_COUNT[$INST]=0
    fi
}

sync_instance() {
    local INST=$1 SEED=$2 HOST=$3 PORT=$4
    local SSH="ssh -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15 root@$HOST"

    for cond in $EXPERIMENTS_PER_SEED; do
        local exp="${cond}_s${SEED}"
        if $SSH "test -d /workspace/turnstile/experiments/$exp" 2>/dev/null; then
            mkdir -p "$LOCAL_BASE/$exp"
            rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
                "root@$HOST:/workspace/turnstile/experiments/$exp/" \
                "$LOCAL_BASE/$exp/" 2>/dev/null
        fi
    done
    rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
        "root@$HOST:/workspace/turnstile/loop_s${SEED}.log" \
        "$LOCAL_BASE/loop_s${SEED}.log" 2>/dev/null
}

check_instance() {
    local INST=$1 SEED=$2 HOST=$3 PORT=$4
    local SSH="ssh -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15 root@$HOST"

    if ! $SSH "echo ok" 2>/dev/null >/dev/null; then
        log "  [s$SEED] SSH failed (inst $INST)"
        return 3
    fi

    local procs=$($SSH "ps aux | grep -E 'run_seed|stealth_loop' | grep -v grep | wc -l" 2>/dev/null)
    if [ "$procs" = "0" ] 2>/dev/null || [ -z "$procs" ]; then
        local last=$($SSH "tail -2 /workspace/turnstile/loop_s${SEED}.log" 2>/dev/null)
        if echo "$last" | grep -q "SEED $SEED COMPLETE"; then
            log "  [s$SEED] COMPLETE"
            return 2
        else
            log "  [s$SEED] CRASHED. Last: $last"
            return 1
        fi
    fi

    local status=$($SSH "grep -E 'RUN:|Metrics' /workspace/turnstile/loop_s${SEED}.log 2>/dev/null | tail -2" 2>/dev/null)
    log "  [s$SEED] Running. $status"
    return 0
}

destroy_instance() {
    local INST=$1
    log "  Destroying instance $INST..."
    echo "n" | vastai destroy instance $INST 2>&1 | grep -v "Update"
}

# --- Main ---
log "=== PARALLEL BABYSITTER ==="

while true; do
    log "--- Check ---"

    all_done=true
    while IFS=' ' read -r INST SEED HOST PORT; do
        [ -z "$INST" ] && continue

        check_instance $INST $SEED $HOST $PORT
        status=$?

        sync_instance $INST $SEED $HOST $PORT

        case $status in
            0)  # still running — check for idle GPU
                all_done=false
                check_gpu_idle $INST $HOST $PORT $SEED
                ;;
            1) all_done=false ;;  # crashed, keep monitoring
            2)  # complete — destroy
                sync_instance $INST $SEED $HOST $PORT  # final sync
                destroy_instance $INST
                sed -i '' "/^$INST /d" "$INSTANCES_FILE"
                ;;
            3) all_done=false ;;  # SSH failed, keep trying
        esac
    done < "$INSTANCES_FILE"

    # Check if all done
    if [ ! -s "$INSTANCES_FILE" ]; then
        log "=== ALL INSTANCES COMPLETE ==="

        # Print summary
        log "Final metrics:"
        for cond in $EXPERIMENTS_PER_SEED; do
            for seed in 42 123 456; do
                local mf="$LOCAL_BASE/${cond}_s${seed}/metrics.jsonl"
                if [ -f "$mf" ]; then
                    local rounds=$(wc -l < "$mf" | tr -d ' ')
                    log "  ${cond}_s${seed}: ${rounds} rounds"
                fi
            done
        done
        exit 0
    fi

    sleep $INTERVAL
done
