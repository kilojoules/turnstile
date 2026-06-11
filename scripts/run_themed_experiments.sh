#!/bin/bash
# Run themed adversary experiments: urgency, threat, reward, authority.
# Each: bootstrap (2000 seeds) with theme -> 20-round DPO loop.
#
# Usage:
#   bash scripts/run_themed_experiments.sh              # all 4 sequentially
#   bash scripts/run_themed_experiments.sh urgency      # single theme (for parallel)
#   bash scripts/run_themed_experiments.sh threat reward # subset
set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Auto-detect workspace root (parent of this script's directory)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$(pwd):$PYTHONPATH"

TOGETHER_KEY=$(cat ~/.together 2>/dev/null || echo "")
SEED=42
ROUNDS=20
CANDIDATES=30
NUM_SEEDS=1000
BOOTSTRAP_ITERS=500

# If arguments given, run only those themes; otherwise run all 4
if [ $# -gt 0 ]; then
    THEMES=("$@")
else
    THEMES=("urgency" "incrementalism" "reward" "authority")
fi

echo "=============================================="
echo "  THEMED ADVERSARY EXPERIMENTS"
echo "  Themes: ${THEMES[*]}"
echo "  Bootstrap: $NUM_SEEDS seeds per theme"
echo "  Rounds: $ROUNDS, Candidates: $CANDIDATES"
echo "  Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=============================================="

run_themed() {
    local THEME=$1
    local NAME="${THEME}_dpo"

    # Skip if already complete
    local METRICS="experiments/$NAME/metrics.jsonl"
    if [ -f "$METRICS" ]; then
        local DONE=$(wc -l < "$METRICS" | tr -d ' ')
        if [ "$DONE" -ge "$ROUNDS" ]; then
            echo "  SKIP: $NAME already has $DONE rounds."
            return
        fi
    fi

    echo ""
    echo "======================================================"
    echo "  THEME: $THEME"
    echo "  Experiment: $NAME"
    echo "  Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "======================================================"

    # Phase 1: Bootstrap with theme (bootstrap.py resumes from train.jsonl if present)
    echo "=== BOOTSTRAP ($THEME) ==="
    python3 -u -m turnstile.bootstrap \
        --num-seeds $NUM_SEEDS --num-turns 3 --seed $SEED \
        --data-dir "experiments/$NAME/data" \
        --adapter-path "experiments/$NAME/adapters" \
        --adversary-model meta-llama/Llama-3.2-3B-Instruct \
        --lora-iters $BOOTSTRAP_ITERS \
        --theme "$THEME"

    # Phase 2: DPO loop
    echo "=== DPO LOOP ($THEME) ==="
    python3 -u -m turnstile.loop \
        --name "$NAME" \
        --adversary-model meta-llama/Llama-3.2-3B-Instruct \
        --rounds $ROUNDS --candidates $CANDIDATES --num-turns 5 \
        --mode dpo --lora-iters 50 \
        --seed $SEED \
        --theme "$THEME" \
        ${TOGETHER_KEY:+--together-key "$TOGETHER_KEY"}

    echo "  Finished $THEME: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

    # Print summary
    echo "  --- Metrics ($NAME) ---"
    python3 -c "
import json
with open('experiments/$NAME/metrics.jsonl') as f:
    for line in f:
        m = json.loads(line)
        breach = f'  breach={m[\"mean_turn_of_breach\"]:.1f}' if m.get('mean_turn_of_breach') else ''
        print(f'  Round {m[\"round\"]:2d}: ASR={m[\"asr\"]:.1%}{breach}')
" 2>/dev/null || echo "  (no metrics yet)"
}

for theme in "${THEMES[@]}"; do
    run_themed "$theme"
done

echo ""
echo "=============================================="
echo "  THEMED EXPERIMENTS COMPLETE: ${THEMES[*]}"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=============================================="

# Cross-theme comparison (always shows all 4)
echo ""
echo "=== COMPARISON ==="
python3 -c "
import json
themes = ['urgency_dpo', 'incrementalism_dpo', 'reward_dpo', 'authority_dpo']
for name in themes:
    try:
        asrs = []
        with open(f'experiments/{name}/metrics.jsonl') as f:
            for line in f:
                asrs.append(json.loads(line)['asr'])
        mean = sum(asrs)/len(asrs) if asrs else 0
        final = asrs[-1] if asrs else 0
        print(f'{name:20s}: mean_ASR={mean:.1%}  final_ASR={final:.1%}  ({len(asrs)} rounds)')
    except FileNotFoundError:
        print(f'{name:20s}: (not run yet)')
" 2>/dev/null
