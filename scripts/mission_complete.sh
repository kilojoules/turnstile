#!/bin/bash
# Last-resort mission completion script.
# Runs independently of the babysitter. Checks every 10 minutes.
# When control_hard_s456 reaches 15 rounds:
#   1. Final sync of ALL experiment data
#   2. Destroy the instance
#   3. Write completion marker
# If the instance is already dead, just exits.

INST=34136065
HOST=ssh2.vast.ai
PORT=16064
LOCAL_BASE="/Users/julianquick/portfolio_copy/turnstile/experiments"
MARKER="$LOCAL_BASE/.mission_complete"
LOG="$LOCAL_BASE/mission_complete.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG"; }

log "=== MISSION COMPLETE WATCHDOG STARTED ==="

while true; do
    # Already done?
    if [ -f "$MARKER" ]; then
        log "Marker exists. Already complete. Exiting."
        exit 0
    fi

    # Can we reach the instance?
    if ! ssh -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15 root@$HOST "echo ok" 2>/dev/null >/dev/null; then
        log "SSH failed. Instance may already be destroyed."
        # Check if vastai says it's gone
        STATUS=$(echo "n" | vastai show instance $INST --raw 2>&1 | grep -v "Update\|selected" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('actual_status','?'))" 2>/dev/null)
        if [ "$STATUS" != "running" ]; then
            log "Instance not running (status=$STATUS). Mission complete or instance lost."
            touch "$MARKER"
            exit 0
        fi
        log "Instance says running but SSH failed. Retrying in 10min."
        sleep 600
        continue
    fi

    # Check round count
    R=$(ssh -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15 root@$HOST \
        "wc -l < /workspace/turnstile/experiments/control_hard_s456/metrics.jsonl 2>/dev/null" 2>&1 | grep -oE '^[0-9]+')

    log "control_hard_s456: ${R:-?}/15 rounds"

    if [ -n "$R" ] && [ "$R" -ge 15 ]; then
        log "COMPLETE! Syncing all data..."

        # Sync everything from this instance
        for exp in stealth_hard_s456 control_hard_s456; do
            mkdir -p "$LOCAL_BASE/$exp/rounds"
            rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
                "root@$HOST:/workspace/turnstile/experiments/$exp/" \
                "$LOCAL_BASE/$exp/" 2>/dev/null
            log "  Synced $exp"
        done

        # Double sync metrics
        for exp in stealth_hard_s456 control_hard_s456; do
            rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
                "root@$HOST:/workspace/turnstile/experiments/$exp/metrics.jsonl" \
                "$LOCAL_BASE/$exp/metrics.jsonl" 2>/dev/null
        done
        log "  Double-synced metrics"

        # Verify local data
        for exp in stealth_hard_s456 control_hard_s456; do
            LOCAL_R=$(wc -l < "$LOCAL_BASE/$exp/metrics.jsonl" 2>/dev/null | tr -d ' ')
            log "  Local $exp: ${LOCAL_R:-0} rounds"
        done

        # Destroy instance
        log "Destroying instance $INST..."
        echo "n" | vastai destroy instance $INST 2>&1 | grep -v "Update" >> "$LOG"
        log "Instance destroyed."

        # Write completion marker with summary
        python3 -c "
import json, os, numpy as np
base = '$LOCAL_BASE'
def load(name):
    f = os.path.join(base, name, 'metrics.jsonl')
    if not os.path.exists(f): return []
    return [json.loads(l) for l in open(f) if l.strip()]

with open('$MARKER', 'w') as out:
    out.write('MISSION COMPLETE\n')
    for seed in [42, 123, 456]:
        s = load(f'stealth_hard_s{seed}')
        c = load(f'control_hard_s{seed}')
        if s and c:
            min_r = min(len(s), len(c))
            sm = np.mean([m['asr'] for m in s[:min_r]])
            cm = np.mean([m['asr'] for m in c[:min_r]])
            out.write(f's{seed}: stealth={sm:.1%} control={cm:.1%} diff={sm-cm:+.1%} ({min_r}r)\n')
" 2>/dev/null

        log "=== MISSION COMPLETE ==="
        cat "$MARKER" >> "$LOG"
        exit 0
    fi

    sleep 600
done
