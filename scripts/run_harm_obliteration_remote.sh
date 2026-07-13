#!/bin/bash
# Combined harm-obliteration training + ASR eval on a single A100 80GB.
# Phase 1: train_compliance_obliteration.py --label-mode harm to step 500.
# Phase 2: asr_eval_obliterated.py with the new adapter.
# Marker files for the watchdog:
#   $OUT/.training_done   on full success
#   $OUT/.training_failed on any phase failure
set -u
set -o pipefail

cd /workspace/turnstile
OUT=experiments/harm_obliteration_v1
mkdir -p "$OUT"
rm -f "$OUT/.training_done" "$OUT/.training_failed"

export HF_TOKEN=$(cat /root/.hf_token)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Phase 1: harm obliteration train ==="
date -Is
python -u -m scripts.train_compliance_obliteration \
    --output "$OUT" \
    --label-mode harm \
    --labels-path working/uplift/stage_b_scores_llama.jsonl \
    --steps 500 \
    --batch-pos 6 --batch-neg 6 \
    --lr 1e-4 \
    --lora-rank 16 --lora-alpha 32 \
    --beta 4.0 \
    --ridge-lambda 1.0 \
    --auc-tau 1.0 \
    --max-length 1024 \
    --eval-every 100 \
    --save-every 500 \
    --probe-layer 16 \
    --seed 0
RC=$?
if [ $RC -ne 0 ]; then
    echo "PHASE1_FAILED rc=$RC" >> "$OUT/.training_failed"
    exit 1
fi

if [ ! -d "$OUT/adapter" ]; then
    echo "PHASE1_NO_ADAPTER" >> "$OUT/.training_failed"
    exit 1
fi

echo "=== Phase 2: ASR eval ==="
date -Is
python -u -m scripts.asr_eval_obliterated \
    --adapter "$OUT/adapter" \
    --output "$OUT/asr_results.jsonl" \
    --n-convs 50 \
    --max-new-tokens 256 \
    --seed 0
RC=$?
if [ $RC -ne 0 ]; then
    echo "PHASE2_FAILED rc=$RC" >> "$OUT/.training_failed"
    exit 1
fi

touch "$OUT/.training_done"
echo "=== ALL DONE ==="
date -Is
