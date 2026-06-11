#!/bin/bash
# Monitor themed experiments running on Vast.ai.
# Periodically checks status, syncs metrics, and destroys instances when done.
#
# Usage: bash scripts/babysit_themed.sh
set -e

LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
INSTANCE_FILE="$LOCAL/experiments/themed_instances.txt"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"
CHECK_INTERVAL=300  # 5 minutes

if [ ! -f "$INSTANCE_FILE" ]; then
    echo "No instance file found at $INSTANCE_FILE"
    echo "Run launch_themed.sh first."
    exit 1
fi

echo "=============================================="
echo "  BABYSITTING THEMED EXPERIMENTS"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=============================================="

while true; do
    ALL_DONE=true
    echo ""
    echo "--- Check at $(date -u '+%H:%M:%S UTC') ---"

    while read INST THEME HOST PORT; do
        [ -z "$INST" ] && continue

        # Check if instance still exists
        STATUS=$(echo "n" | vastai show instance $INST --raw 2>&1 | \
            grep -v "Update\|selected" | \
            python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('actual_status','destroyed'))" 2>/dev/null || echo "destroyed")

        if [ "$STATUS" = "destroyed" ]; then
            echo "  $THEME: already destroyed"
            continue
        fi

        # Check if experiment is still running
        PROCS=$(ssh -p $PORT $SSH_OPTS root@$HOST \
            "ps aux | grep 'run_themed_experiments' | grep -v grep | wc -l" 2>/dev/null || echo "0")

        if [ "$PROCS" = "0" ]; then
            # Check if it finished or crashed
            LAST=$(ssh -p $PORT $SSH_OPTS root@$HOST \
                "tail -5 /workspace/turnstile/${THEME}.log 2>/dev/null" 2>/dev/null || echo "")

            if echo "$LAST" | grep -q "COMPLETE"; then
                echo "  $THEME: COMPLETE -- syncing results and destroying"

                # Sync everything back
                mkdir -p "$LOCAL/experiments/${THEME}_dpo"
                rsync -az -e "ssh -p $PORT $SSH_OPTS" \
                    "root@$HOST:/workspace/turnstile/experiments/${THEME}_dpo/" \
                    "$LOCAL/experiments/${THEME}_dpo/" 2>/dev/null

                # Sync log
                scp -P $PORT $SSH_OPTS \
                    "root@$HOST:/workspace/turnstile/${THEME}.log" \
                    "$LOCAL/experiments/${THEME}_dpo/${THEME}.log" 2>/dev/null

                # Destroy instance
                echo "n" | vastai destroy instance $INST 2>/dev/null
                echo "  $THEME: instance $INST destroyed"
            else
                echo "  $THEME: CRASHED? Last log lines:"
                echo "$LAST" | head -3 | sed 's/^/    /'
                ALL_DONE=false
            fi
        else
            ALL_DONE=false

            # Sync metrics for progress check
            mkdir -p "$LOCAL/experiments/${THEME}_dpo"
            rsync -az -e "ssh -p $PORT $SSH_OPTS" \
                "root@$HOST:/workspace/turnstile/experiments/${THEME}_dpo/metrics.jsonl" \
                "$LOCAL/experiments/${THEME}_dpo/metrics.jsonl" 2>/dev/null

            # Report progress
            if [ -f "$LOCAL/experiments/${THEME}_dpo/metrics.jsonl" ]; then
                ROUNDS=$(wc -l < "$LOCAL/experiments/${THEME}_dpo/metrics.jsonl" | tr -d ' ')
                LAST_ASR=$(tail -1 "$LOCAL/experiments/${THEME}_dpo/metrics.jsonl" 2>/dev/null | \
                    python3 -c "import json,sys; m=json.load(sys.stdin); print(f'ASR={m[\"asr\"]:.1%}')" 2>/dev/null || echo "")
                echo "  $THEME: running (round $ROUNDS/20 $LAST_ASR)"
            else
                # Still bootstrapping
                BOOT_PROGRESS=$(ssh -p $PORT $SSH_OPTS root@$HOST \
                    "grep -c 'Seed ' /workspace/turnstile/${THEME}.log 2>/dev/null" 2>/dev/null || echo "0")
                echo "  $THEME: bootstrapping ($BOOT_PROGRESS/2000 seeds)"
            fi
        fi
    done < "$INSTANCE_FILE"

    if $ALL_DONE; then
        echo ""
        echo "=============================================="
        echo "  ALL EXPERIMENTS COMPLETE OR DESTROYED"
        echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
        echo "=============================================="

        # Final comparison
        echo ""
        echo "=== RESULTS ==="
        python3 -c "
import json
themes = ['urgency_dpo', 'incrementalism_dpo', 'reward_dpo', 'authority_dpo']
for name in themes:
    try:
        asrs = []
        with open(f'$LOCAL/experiments/{name}/metrics.jsonl') as f:
            for line in f:
                asrs.append(json.loads(line)['asr'])
        mean = sum(asrs)/len(asrs) if asrs else 0
        final = asrs[-1] if asrs else 0
        print(f'{name:24s}: mean_ASR={mean:.1%}  final_ASR={final:.1%}  ({len(asrs)} rounds)')
    except FileNotFoundError:
        print(f'{name:24s}: (not synced yet)')
" 2>/dev/null
        break
    fi

    sleep $CHECK_INTERVAL
done
