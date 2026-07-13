#!/bin/bash
# Compliance + harm obliterations restricted to corpus-compliant turns.
# Both runs + ASR evals run sequentially on one A100 80GB.
# Marker files:
#   experiments/compliant_obliterations_v1/.training_done   on full success
#   experiments/compliant_obliterations_v1/.training_failed on any phase failure
set -u
set -o pipefail

cd /workspace/turnstile
MARKER_DIR=experiments/compliant_obliterations_v1
COMP_OUT=experiments/compliance_obliteration_compliant_v1
HARM_OUT=experiments/harm_obliteration_compliant_v1
mkdir -p "$MARKER_DIR" "$COMP_OUT" "$HARM_OUT"
rm -f "$MARKER_DIR/.training_done" "$MARKER_DIR/.training_failed"

export HF_TOKEN=$(cat /root/.hf_token)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Phase 1: compliance obliteration, compliant-only SFT ────────────────────
echo "=== Phase 1: compliance obliteration (compliant-corpus SFT) ==="
date -Is
python -u -m scripts.train_compliance_obliteration \
    --output "$COMP_OUT" \
    --label-mode compliance \
    --corpus-compliant-only \
    --steps 500 \
    --batch-pos 4 --batch-neg 4 \
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
    echo "PHASE1_FAILED rc=$RC" >> "$MARKER_DIR/.training_failed"; exit 1
fi
if [ ! -d "$COMP_OUT/adapter" ]; then
    echo "PHASE1_NO_ADAPTER" >> "$MARKER_DIR/.training_failed"; exit 1
fi

# ── Phase 2: ASR eval for compliance obliteration compliant ─────────────────
echo "=== Phase 2: ASR eval (compliance-compliant adapter) ==="
date -Is
python -u -m scripts.asr_eval_obliterated \
    --adapter "$COMP_OUT/adapter" \
    --output "$COMP_OUT/asr_results.jsonl" \
    --n-convs 50 \
    --max-new-tokens 256 \
    --seed 0
RC=$?
if [ $RC -ne 0 ]; then
    echo "PHASE2_FAILED rc=$RC" >> "$MARKER_DIR/.training_failed"; exit 1
fi

# ── Phase 3: harm obliteration, compliant-pool + compliant SFT ──────────────
echo "=== Phase 3: harm obliteration (compliant-corpus pool + SFT) ==="
date -Is
python -u -m scripts.train_compliance_obliteration \
    --output "$HARM_OUT" \
    --label-mode harm \
    --labels-path working/uplift/stage_b_scores_llama.jsonl \
    --corpus-compliant-only \
    --steps 500 \
    --batch-pos 4 --batch-neg 4 \
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
    echo "PHASE3_FAILED rc=$RC" >> "$MARKER_DIR/.training_failed"; exit 1
fi
if [ ! -d "$HARM_OUT/adapter" ]; then
    echo "PHASE3_NO_ADAPTER" >> "$MARKER_DIR/.training_failed"; exit 1
fi

# ── Phase 4: ASR eval for harm obliteration compliant ───────────────────────
echo "=== Phase 4: ASR eval (harm-compliant adapter) ==="
date -Is
python -u -m scripts.asr_eval_obliterated \
    --adapter "$HARM_OUT/adapter" \
    --output "$HARM_OUT/asr_results.jsonl" \
    --n-convs 50 \
    --max-new-tokens 256 \
    --seed 0
RC=$?
if [ $RC -ne 0 ]; then
    echo "PHASE4_FAILED rc=$RC" >> "$MARKER_DIR/.training_failed"; exit 1
fi

touch "$MARKER_DIR/.training_done"
echo "=== ALL DONE ==="
date -Is
