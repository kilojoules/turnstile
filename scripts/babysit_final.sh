#!/bin/bash
# Final babysitter: monitors control_hard_s456 on L40 instance.
# Syncs data, destroys instance when done. Self-restarts on failure.
set -u

INST=34136065
HOST=ssh2.vast.ai
PORT=16064
EXP=control_hard_s456
LOCAL_BASE="/Users/julianquick/portfolio_copy/turnstile/experiments"
LOCAL_LOG="$LOCAL_BASE/babysitter_final.log"
INTERVAL=300
MAX_SSH_FAILURES=12  # 1 hour of failures before giving up

ssh_failures=0

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOCAL_LOG"
}

sync_data() {
    mkdir -p "$LOCAL_BASE/$EXP"
    rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15" \
        "root@$HOST:/workspace/turnstile/experiments/$EXP/metrics.jsonl" \
        "$LOCAL_BASE/$EXP/metrics.jsonl" 2>/dev/null
    rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15" \
        "root@$HOST:/workspace/turnstile/experiments/$EXP/rounds/" \
        "$LOCAL_BASE/$EXP/rounds/" 2>/dev/null
    # Also sync stealth_hard_s456 rounds if not already done
    mkdir -p "$LOCAL_BASE/stealth_hard_s456/rounds"
    rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15" \
        "root@$HOST:/workspace/turnstile/experiments/stealth_hard_s456/" \
        "$LOCAL_BASE/stealth_hard_s456/" 2>/dev/null
}

log "=== FINAL BABYSITTER: $EXP on $HOST:$PORT ==="

while true; do
    # Try SSH
    if ! ssh -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15 root@$HOST "echo ok" 2>/dev/null >/dev/null; then
        ssh_failures=$((ssh_failures + 1))
        log "  [SSH] Failed ($ssh_failures/$MAX_SSH_FAILURES)"
        if [ $ssh_failures -ge $MAX_SSH_FAILURES ]; then
            log "  [FATAL] SSH failed for 1 hour. Instance may be dead."
            sync_data
            exit 1
        fi
        sleep $INTERVAL
        continue
    fi
    ssh_failures=0

    # Check rounds
    R=$(ssh -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15 root@$HOST \
        "wc -l < /workspace/turnstile/experiments/$EXP/metrics.jsonl 2>/dev/null" 2>&1 | grep -oE '^[0-9]+')

    if [ -n "$R" ] && [ "$R" -ge 15 ]; then
        log "  [DONE] $EXP complete ($R rounds)"
        sync_data
        sync_data  # double sync

        # Show final results
        log "  Final hardened A/B:"
        python3 -c "
import json, os, numpy as np
base = '$LOCAL_BASE'
def load(name):
    f = os.path.join(base, name, 'metrics.jsonl')
    if not os.path.exists(f): return []
    return [json.loads(l) for l in open(f) if l.strip()]
for seed in [42, 123, 456]:
    s = load(f'stealth_hard_s{seed}')
    c = load(f'control_hard_s{seed}')
    if s and c:
        min_r = min(len(s), len(c))
        sm = np.mean([m['asr'] for m in s[:min_r]])
        cm = np.mean([m['asr'] for m in c[:min_r]])
        print(f'  s{seed} ({min_r}r): stealth={sm:.1%} control={cm:.1%} diff={sm-cm:+.1%}')
" 2>/dev/null

        # Destroy instance
        log "  Destroying instance $INST..."
        echo "n" | vastai destroy instance $INST 2>&1 | grep -v "Update"
        log "=== ALL EXPERIMENTS COMPLETE. INSTANCE DESTROYED. ==="
        exit 0
    fi

    # Check if process is alive
    PROCS=$(ssh -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15 root@$HOST \
        "ps aux | grep -E 'stealth_loop|run_seed' | grep -v grep | wc -l" 2>&1 | grep -oE '^[0-9]+')

    if [ "${PROCS:-0}" = "0" ]; then
        log "  [WARN] No process running but only $R rounds. Crashed?"
        log "  Last log:"
        ssh -p $PORT -o StrictHostKeyChecking=no root@$HOST \
            "tail -5 /workspace/turnstile/loop_hard_s456.log" 2>&1 | grep -v "Welcome\|Have fun" | while read line; do log "    $line"; done
        sync_data
        # Don't exit — might just be between experiments
    else
        log "  [ok] $EXP: ${R:-0}/15 rounds"
    fi

    sync_data
    sleep $INTERVAL
done
