#!/bin/bash
# Run all 4 conditions for a single seed on one GPU.
# Usage: bash scripts/run_seed.sh <SEED>
set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH=/workspace/turnstile:$PYTHONPATH
cd /workspace/turnstile

SEED=${1:?Usage: run_seed.sh <SEED>}
TOGETHER_KEY=$(cat ~/.together)
STARTING_ADAPTER="experiments/selfplay_jbb_v1/adapters"
PROBE_PATH="results/probe_jbb/probe.pkl"
WINS_FILE="data/jbb_verified_wins.jsonl"
ROUNDS=15
CANDIDATES=100

echo "=============================================="
echo "  SEED $SEED: All 4 conditions"
echo "  Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=============================================="

run_one() {
    local NAME=$1
    local MODE=$2
    local HARDEN=${3:-""}

    local METRICS="experiments/$NAME/metrics.jsonl"
    if [ -f "$METRICS" ]; then
        local DONE=$(wc -l < "$METRICS" | tr -d ' ')
        if [ "$DONE" -ge 15 ]; then
            echo "  SKIP: $NAME ($DONE rounds). "
            return
        fi
    fi

    echo ""
    echo "=== $NAME (mode=$MODE, harden=$HARDEN) ==="
    rm -rf experiments/$NAME
    mkdir -p experiments/$NAME/{rounds,data,adapters,hidden_states}
    cp -r $STARTING_ADAPTER/* experiments/$NAME/adapters/
    cp $WINS_FILE experiments/$NAME/data/train.jsonl

    python3 -u -m turnstile.stealth_loop \
        --name $NAME \
        --adversary-model meta-llama/Llama-3.2-3B-Instruct \
        --rounds $ROUNDS --candidates $CANDIDATES --num-turns 5 \
        --together-key "$TOGETHER_KEY" \
        --probe-path $PROBE_PATH \
        --stealth-mode $MODE --stealth-alpha 3.0 \
        --retrain-probe-every 5 \
        --mode buffered --lora-iters 50 \
        --seed $SEED $HARDEN

    echo "  Done: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
}

# Frozen victim
run_one "stealth_s${SEED}" "weighted"
run_one "control_s${SEED}" "none"

# Hardened victim
run_one "stealth_hard_s${SEED}" "weighted" "--harden-victim"
run_one "control_hard_s${SEED}" "none" "--harden-victim"

echo ""
echo "=== SEED $SEED COMPLETE ==="
echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
