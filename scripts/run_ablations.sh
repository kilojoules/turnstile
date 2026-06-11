#!/bin/bash
# Run all ablation conditions for a single seed on one GPU.
#
# Conditions:
#   1. Alpha ablation:       alpha in {1, 2, 3, 5} with weighted mode (original)
#   2. Importance-weighted:  iw_weighted mode (no bootstrap/dedup)
#   3. Probe-aware DPO:      probe_dpo mode (preference learning from probe)
#   4. Controls:             none mode (frozen + hardened) for comparison
#
# Each condition runs against both frozen and hardened victims.
#
# Usage: bash scripts/run_ablations.sh <SEED>
set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH=/workspace/turnstile:$PYTHONPATH
cd /workspace/turnstile

SEED=${1:?Usage: run_ablations.sh <SEED>}
TOGETHER_KEY=$(cat ~/.together)
STARTING_ADAPTER="experiments/selfplay_jbb_v1/adapters"
PROBE_PATH="results/probe_jbb/probe.pkl"
WINS_FILE="data/jbb_verified_wins.jsonl"
ROUNDS=15
CANDIDATES=100

echo "=============================================="
echo "  SEED $SEED: ABLATION SUITE"
echo "  Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=============================================="

run_one() {
    local NAME=$1
    local MODE=$2
    local ALPHA=$3
    local HARDEN=${4:-""}

    local METRICS="experiments/$NAME/metrics.jsonl"
    if [ -f "$METRICS" ]; then
        local DONE=$(wc -l < "$METRICS" | tr -d ' ')
        if [ "$DONE" -ge "$ROUNDS" ]; then
            echo "  SKIP: $NAME ($DONE rounds)."
            return
        fi
    fi

    echo ""
    echo "=== $NAME (mode=$MODE, alpha=$ALPHA, harden=$HARDEN) ==="
    rm -rf experiments/$NAME
    mkdir -p experiments/$NAME/{rounds,data,adapters,hidden_states}
    cp -r $STARTING_ADAPTER/* experiments/$NAME/adapters/
    cp $WINS_FILE experiments/$NAME/data/train.jsonl
    if [ -n "$HARDEN" ]; then
        mkdir -p experiments/$NAME/{victim_adapters,victim_data}
    fi

    python3 -u -m turnstile.stealth_loop \
        --name $NAME \
        --adversary-model meta-llama/Llama-3.2-3B-Instruct \
        --rounds $ROUNDS --candidates $CANDIDATES --num-turns 5 \
        --together-key "$TOGETHER_KEY" \
        --probe-path $PROBE_PATH \
        --stealth-mode $MODE --stealth-alpha $ALPHA \
        --retrain-probe-every 5 \
        --mode buffered --lora-iters 50 \
        --seed $SEED $HARDEN

    echo "  Done: $NAME at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
}

# =============================================
# 1. ALPHA ABLATION (weighted mode, frozen)
# =============================================
echo ""
echo ">>> ALPHA ABLATION (frozen victim) <<<"
for ALPHA in 1 2 5; do
    # alpha=3 already exists as "stealth_s${SEED}" from run_seed.sh
    run_one "alpha${ALPHA}_s${SEED}" "weighted" "$ALPHA"
done

# =============================================
# 2. IMPORTANCE-WEIGHTED SFT (frozen + hardened)
# =============================================
echo ""
echo ">>> IMPORTANCE-WEIGHTED SFT <<<"
run_one "iw_s${SEED}"      "iw_weighted" 3.0
run_one "iw_hard_s${SEED}" "iw_weighted" 3.0 "--harden-victim"

# =============================================
# 3. PROBE-AWARE DPO (frozen + hardened)
# =============================================
echo ""
echo ">>> PROBE-AWARE DPO <<<"
run_one "pdpo_s${SEED}"      "probe_dpo" 3.0
run_one "pdpo_hard_s${SEED}" "probe_dpo" 3.0 "--harden-victim"

# =============================================
# 4. CONTROL (if not already run by run_seed.sh)
# =============================================
echo ""
echo ">>> CONTROLS (if needed) <<<"
run_one "control_s${SEED}"      "none" 3.0
run_one "control_hard_s${SEED}" "none" 3.0 "--harden-victim"

echo ""
echo "=== SEED $SEED ABLATION SUITE COMPLETE ==="
echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
