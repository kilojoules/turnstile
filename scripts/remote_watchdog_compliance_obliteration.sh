#!/bin/bash
# Self-destruct watchdog. Polls training markers + a hard runtime cap; on
# any terminal state calls `vastai destroy instance <self>` after a brief
# grace period so the local rsync can grab the adapter + log.
#
# Args: $1 = instance id, $2 = train pid
set -u

INSTANCE_ID="$1"
TRAIN_PID="$2"
OUT="${3:-/workspace/turnstile/experiments/compliance_obliteration_v1}"
LOG="$OUT/watchdog.log"
mkdir -p "$OUT"
MAX_RUNTIME="${MAX_RUNTIME:-21600}"  # default 6h; override via env
GRACE=600           # 10 min between terminal-state detection and destroy
                    # (long enough to run capability eval on the hot instance)
POLL=30
START=$(date +%s)

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

setup_api() {
    if ! command -v vastai >/dev/null; then
        pip install -q vastai 2>&1 | tail -1
    fi
    vastai set api-key "$(cat /root/.fastai_key)" >/dev/null 2>&1
}

destroy_self() {
    local reason="$1"
    log "TERMINAL: $reason  (sleeping $GRACE then destroying $INSTANCE_ID)"
    sleep $GRACE
    setup_api
    log "calling vastai destroy instance $INSTANCE_ID (yes-confirm)"
    OUTPUT=$(echo "y" | vastai destroy instance "$INSTANCE_ID" --raw 2>&1)
    log "destroy output: $OUTPUT"
    # Best-effort secondary: HTTPS DELETE via curl
    sleep 5
    KEY=$(cat /root/.fastai_key)
    log "curl DELETE fallback"
    curl -sS -X DELETE -H "Authorization: Bearer $KEY" \
        "https://console.vast.ai/api/v0/instances/$INSTANCE_ID/" 2>&1 | tee -a "$LOG"
    log "destroy attempts complete; exiting watchdog"
    exit 0
}

log "watchdog up. instance=$INSTANCE_ID train_pid=$TRAIN_PID max=${MAX_RUNTIME}s"

while true; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - START))

    if [ -f "$OUT/.training_done" ]; then
        destroy_self "training_done marker"
    fi
    if [ -f "$OUT/.training_failed" ]; then
        destroy_self "training_failed marker"
    fi

    if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        # PID is dead. Re-check markers; if still none after 60s grace, assume crash.
        sleep 60
        if [ -f "$OUT/.training_done" ]; then
            destroy_self "training_done after pid death"
        fi
        if [ -f "$OUT/.training_failed" ]; then
            destroy_self "training_failed after pid death"
        fi
        destroy_self "train pid died with no marker"
    fi

    if [ $ELAPSED -gt $MAX_RUNTIME ]; then
        destroy_self "max_runtime ${MAX_RUNTIME}s exceeded"
    fi

    sleep $POLL
done
