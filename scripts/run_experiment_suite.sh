#!/bin/bash
# A/B experiment suite: stealth vs control, 3 seeds each.
# Runs 6 experiments sequentially on a single GPU.
set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH=/workspace/turnstile:$PYTHONPATH
cd /workspace/turnstile

TOGETHER_KEY=$(cat ~/.together)
STARTING_ADAPTER="experiments/selfplay_jbb_v1/adapters"
PROBE_PATH="results/probe_jbb/probe.pkl"
WINS_FILE="data/jbb_verified_wins.jsonl"

SEEDS="42 123 456"
ROUNDS=15
CANDIDATES=100

echo "=============================================="
echo "  A/B EXPERIMENT SUITE"
echo "  Stealth (weighted) vs Control (none)"
echo "  Seeds: $SEEDS"
echo "  Rounds: $ROUNDS, Candidates: $CANDIDATES"
echo "  Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=============================================="

# --- Phase 0: Ensure probe exists ---
if [ ! -f "$PROBE_PATH" ]; then
    echo ""
    echo "=== PHASE 0: COLLECTING HIDDEN STATES + TRAINING PROBE ==="
    python3 -u scripts/collect_hidden_states.py \
        --rejudge-files results/rejudge_jbb.jsonl results/rejudge_jbb_remaining.jsonl \
        --output-dir results/hidden_states_jbb

    python3 -u << 'PYEOF'
import torch, numpy as np, pickle, os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

data = torch.load("results/hidden_states_jbb/hidden_states_jbb.pt",
                   map_location="cpu", weights_only=False)
X, y = [], []
for hs, label in zip(data["hidden_states"], data["labels"]):
    for t in range(hs.shape[0]):
        X.append(hs[t].numpy())
        y.append(1 if label else 0)
X, y = np.array(X), np.array(y)
print(f"Probe dataset: {X.shape[0]} examples, {y.sum()} positive ({y.mean():.1%})")

clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
print(f"Probe AUC: {scores.mean():.3f} +/- {scores.std():.3f}")

clf.fit(X, y)
os.makedirs("results/probe_jbb", exist_ok=True)
with open("results/probe_jbb/probe.pkl", "wb") as f:
    pickle.dump(clf, f)
print("Probe saved.")
PYEOF
    echo "  Phase 0 done: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
else
    echo "Probe already exists at $PROBE_PATH"
fi

run_experiment() {
    local NAME=$1
    local MODE=$2   # "weighted" or "none"
    local SEED=$3
    local HARDEN=${4:-""}  # "--harden-victim" or ""

    echo ""
    echo "======================================================"
    echo "  RUN: $NAME (stealth-mode=$MODE, seed=$SEED, harden=$HARDEN)"
    echo "  Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "======================================================"

    # Check if this experiment already has enough rounds
    local METRICS="experiments/$NAME/metrics.jsonl"
    if [ -f "$METRICS" ]; then
        local DONE_ROUNDS=$(wc -l < "$METRICS" | tr -d ' ')
        # Accept anything >= 15 as complete (handles mid-run ROUNDS changes)
        if [ "$DONE_ROUNDS" -ge 15 ]; then
            echo "  SKIP: $NAME already has $DONE_ROUNDS rounds. Skipping."
            return
        fi
        echo "  NOTE: $NAME has $DONE_ROUNDS rounds (incomplete). Starting fresh."
    fi

    # Fresh experiment directory with same starting adapter
    rm -rf experiments/$NAME
    mkdir -p experiments/$NAME/{rounds,data,adapters,hidden_states}
    cp -r $STARTING_ADAPTER/* experiments/$NAME/adapters/
    cp $WINS_FILE experiments/$NAME/data/train.jsonl

    python3 -u -m turnstile.stealth_loop \
        --name $NAME \
        --adversary-model meta-llama/Llama-3.2-3B-Instruct \
        --rounds $ROUNDS \
        --candidates $CANDIDATES \
        --num-turns 5 \
        --together-key "$TOGETHER_KEY" \
        --probe-path $PROBE_PATH \
        --stealth-mode $MODE \
        --stealth-alpha 3.0 \
        --retrain-probe-every 5 \
        --mode buffered \
        --lora-iters 50 \
        --seed $SEED \
        $HARDEN

    echo "  Finished: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

    # Print summary
    echo "  --- Metrics ($NAME) ---"
    python3 -c "
import json
with open('experiments/$NAME/metrics.jsonl') as f:
    for line in f:
        m = json.loads(line)
        print(f'  Round {m[\"round\"]:2d}: ASR={m[\"asr\"]:.1%}  stealth={m.get(\"stealth_asr\",0):.1%}')
" 2>/dev/null || echo "  (no metrics)"
    echo ""
}

# --- Phase 1: Frozen victim (baseline) ---
echo ""
echo "========== PHASE 1: FROZEN VICTIM =========="
for SEED in $SEEDS; do
    run_experiment "stealth_s${SEED}" "weighted" $SEED
    run_experiment "control_s${SEED}" "none" $SEED
done

# --- Phase 2: Hardened victim (arms race) ---
echo ""
echo "========== PHASE 2: HARDENED VICTIM =========="
for SEED in $SEEDS; do
    run_experiment "stealth_hard_s${SEED}" "weighted" $SEED "--harden-victim"
    run_experiment "control_hard_s${SEED}" "none" $SEED "--harden-victim"
done

echo "=============================================="
echo "  SUITE COMPLETE"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=============================================="

# Final summary
echo ""
echo "=== FINAL SUMMARY ==="
for COND in stealth control stealth_hard control_hard; do
    echo "--- $COND ---"
    for SEED in $SEEDS; do
        NAME="${COND}_s${SEED}"
        python3 -c "
import json
asrs = []
with open('experiments/$NAME/metrics.jsonl') as f:
    for line in f:
        asrs.append(json.loads(line)['asr'])
mean_asr = sum(asrs)/len(asrs) if asrs else 0
print(f'  {\"$NAME\":25s}: mean_ASR={mean_asr:.1%} ({len(asrs)} rounds)')
" 2>/dev/null || echo "  $NAME: (missing)"
    done
done
