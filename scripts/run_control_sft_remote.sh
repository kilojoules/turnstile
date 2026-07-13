#!/bin/bash
# Full pipeline for control SFT ablation:
#   Phase 1: Stage-B judge the 2 existing compliant-corpus ASR results
#   Phase 2: Train control SFT compliance (beta=0, compliant-only SFT) + ASR eval + Stage-B
#   Phase 3: Train control SFT harm     (beta=0, compliant-pool SFT)  + ASR eval + Stage-B
# Marker files:
#   experiments/control_sft_v1/.done    on full success
#   experiments/control_sft_v1/.failed  on any phase failure
set -u
set -o pipefail

cd /workspace/turnstile
MARKER_DIR=experiments/control_sft_v1
CTRL_COMP=experiments/control_sft_compliance_v1
CTRL_HARM=experiments/control_sft_harm_v1
mkdir -p "$MARKER_DIR" "$CTRL_COMP" "$CTRL_HARM"
rm -f "$MARKER_DIR/.done" "$MARKER_DIR/.failed"

export HF_TOKEN=$(cat /root/.hf_token)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

die() { echo "$1" >> "$MARKER_DIR/.failed"; exit 1; }

# ── Phase 1a: Stage-B judge compliance-compliant ASR results ────────────────
echo "=== Phase 1a: Stage-B judge (compliance-compliant) ===" && date -Is
python -u -m scripts.judge_stage_b_post_hoc \
    --input  experiments/compliance_obliteration_compliant_v1/asr_results.jsonl \
    --output experiments/compliance_obliteration_compliant_v1/asr_results_stage_b.jsonl
RC=$?; [ $RC -ne 0 ] && die "PHASE1a_FAILED rc=$RC"

# ── Phase 1b: Stage-B judge harm-compliant ASR results ─────────────────────
echo "=== Phase 1b: Stage-B judge (harm-compliant) ===" && date -Is
python -u -m scripts.judge_stage_b_post_hoc \
    --input  experiments/harm_obliteration_compliant_v1/asr_results.jsonl \
    --output experiments/harm_obliteration_compliant_v1/asr_results_stage_b.jsonl
RC=$?; [ $RC -ne 0 ] && die "PHASE1b_FAILED rc=$RC"

# ── Phase 2a: Control SFT compliance (beta=0, compliant-only SFT) ───────────
echo "=== Phase 2a: Control SFT compliance (beta=0) ===" && date -Is
python -u -m scripts.train_compliance_obliteration \
    --output "$CTRL_COMP" \
    --label-mode compliance \
    --corpus-compliant-only \
    --steps 500 \
    --batch-pos 4 --batch-neg 4 \
    --lr 1e-4 \
    --lora-rank 16 --lora-alpha 32 \
    --beta 0.0 \
    --ridge-lambda 1.0 \
    --auc-tau 1.0 \
    --max-length 1024 \
    --eval-every 100 \
    --save-every 500 \
    --probe-layer 16 \
    --seed 0
RC=$?; [ $RC -ne 0 ] && die "PHASE2a_FAILED rc=$RC"
[ ! -d "$CTRL_COMP/adapter" ] && die "PHASE2a_NO_ADAPTER"

# ── Phase 2b: ASR eval control compliance ───────────────────────────────────
echo "=== Phase 2b: ASR eval (control compliance) ===" && date -Is
python -u -m scripts.asr_eval_obliterated \
    --adapter "$CTRL_COMP/adapter" \
    --output  "$CTRL_COMP/asr_results.jsonl" \
    --n-convs 50 \
    --max-new-tokens 256 \
    --seed 0
RC=$?; [ $RC -ne 0 ] && die "PHASE2b_FAILED rc=$RC"

# ── Phase 2c: Stage-B judge control compliance ──────────────────────────────
echo "=== Phase 2c: Stage-B judge (control compliance) ===" && date -Is
python -u -m scripts.judge_stage_b_post_hoc \
    --input  "$CTRL_COMP/asr_results.jsonl" \
    --output "$CTRL_COMP/asr_results_stage_b.jsonl"
RC=$?; [ $RC -ne 0 ] && die "PHASE2c_FAILED rc=$RC"

# ── Phase 3a: Control SFT harm (beta=0, compliant-pool SFT) ────────────────
echo "=== Phase 3a: Control SFT harm (beta=0) ===" && date -Is
python -u -m scripts.train_compliance_obliteration \
    --output "$CTRL_HARM" \
    --label-mode harm \
    --labels-path working/uplift/stage_b_scores_llama.jsonl \
    --corpus-compliant-only \
    --steps 500 \
    --batch-pos 4 --batch-neg 4 \
    --lr 1e-4 \
    --lora-rank 16 --lora-alpha 32 \
    --beta 0.0 \
    --ridge-lambda 1.0 \
    --auc-tau 1.0 \
    --max-length 1024 \
    --eval-every 100 \
    --save-every 500 \
    --probe-layer 16 \
    --seed 0
RC=$?; [ $RC -ne 0 ] && die "PHASE3a_FAILED rc=$RC"
[ ! -d "$CTRL_HARM/adapter" ] && die "PHASE3a_NO_ADAPTER"

# ── Phase 3b: ASR eval control harm ─────────────────────────────────────────
echo "=== Phase 3b: ASR eval (control harm) ===" && date -Is
python -u -m scripts.asr_eval_obliterated \
    --adapter "$CTRL_HARM/adapter" \
    --output  "$CTRL_HARM/asr_results.jsonl" \
    --n-convs 50 \
    --max-new-tokens 256 \
    --seed 0
RC=$?; [ $RC -ne 0 ] && die "PHASE3b_FAILED rc=$RC"

# ── Phase 3c: Stage-B judge control harm ────────────────────────────────────
echo "=== Phase 3c: Stage-B judge (control harm) ===" && date -Is
python -u -m scripts.judge_stage_b_post_hoc \
    --input  "$CTRL_HARM/asr_results.jsonl" \
    --output "$CTRL_HARM/asr_results_stage_b.jsonl"
RC=$?; [ $RC -ne 0 ] && die "PHASE3c_FAILED rc=$RC"

touch "$MARKER_DIR/.done"
echo "=== ALL DONE ===" && date -Is
