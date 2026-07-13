#!/bin/bash
# One-time experiment: LoRA+SFT with compliance-probe obliteration penalty.
#
# Trains a LoRA on Llama-3.1-8B-Instruct over the 11-run probe corpus
# (existing victim responses), driving the L16 compliance signal out of the
# residuals via a closed-form ridge probe + soft-AUC penalty. A long-lived
# MLP audit probe runs alongside as a nonlinear leakage monitor.
#
# Designed to run on a single A100 (80GB recommended; 40GB needs --quant-4bit
# and/or smaller --max-length / batch).
#
# Status markers under $OUT_DIR/:
#   .training_done    -> trainer finished cleanly
#   .training_failed  -> trainer crashed (see train_log.jsonl tail)

set -u
set -o pipefail

cd "$(dirname "$0")/.."  # repo root

OUT_DIR=experiments/compliance_obliteration_v1
mkdir -p "$OUT_DIR"

if [ ! -f /root/.hf_token ] && [ -z "${HF_TOKEN:-}" ]; then
  echo "FATAL: no HF token (need /root/.hf_token or HF_TOKEN env)."
  echo "FATAL_NO_HF_TOKEN" >> "$OUT_DIR/.training_failed"
  exit 99
fi
if [ -f /root/.hf_token ] && [ -z "${HF_TOKEN:-}" ]; then
  export HF_TOKEN="$(cat /root/.hf_token)"
fi

# Sanity check: labels file exists.
LABELS=experiments/steering_v3/layer_sweep/qwen_per_turn_compliance.jsonl
if [ ! -f "$LABELS" ]; then
  echo "FATAL: labels file missing: $LABELS"
  echo "FATAL_NO_LABELS" >> "$OUT_DIR/.training_failed"
  exit 98
fi

LOG="$OUT_DIR/run.log"
echo "[$(date -Is)] starting compliance_obliteration trainer" | tee -a "$LOG"

python -u -m scripts.train_compliance_obliteration \
    --output "$OUT_DIR" \
    --steps 2000 \
    --batch-pos 8 --batch-neg 8 \
    --lr 1e-4 \
    --lora-rank 16 --lora-alpha 32 \
    --beta 4.0 \
    --ridge-lambda 1.0 \
    --auc-tau 1.0 \
    --max-length 1024 \
    --eval-every 100 \
    --save-every 500 \
    --probe-layer 16 \
    --seed 0 \
    2>&1 | tee -a "$LOG"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
  touch "$OUT_DIR/.training_done"
  echo "[$(date -Is)] done" | tee -a "$LOG"
else
  echo "TRAINING_CRASHED" >> "$OUT_DIR/.training_failed"
  echo "[$(date -Is)] crashed" | tee -a "$LOG"
  exit 1
fi
