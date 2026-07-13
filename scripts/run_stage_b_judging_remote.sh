#!/bin/bash
# Stage-B Likert harm judging on existing asr_results.jsonl files.
# Loads Llama-3.1-70B once, judges both runs sequentially.
set -u
set -o pipefail

cd /workspace/turnstile
OUT_DIR=experiments/stage_b_judging_v1
mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR/.training_done" "$OUT_DIR/.training_failed"

export HF_TOKEN=$(cat /root/.hf_token)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Stage-B judging: compliance run ==="
date -Is
python -u -m scripts.judge_stage_b_post_hoc \
    --input experiments/asr_eval_v1/asr_results.jsonl \
    --output experiments/asr_eval_v1/asr_results_stage_b.jsonl
RC=$?
if [ $RC -ne 0 ]; then
    echo "COMPLIANCE_JUDGE_FAILED rc=$RC" >> "$OUT_DIR/.training_failed"
    exit 1
fi

echo "=== Stage-B judging: harm run ==="
date -Is
python -u -m scripts.judge_stage_b_post_hoc \
    --input experiments/harm_obliteration_v1/asr_results.jsonl \
    --output experiments/harm_obliteration_v1/asr_results_stage_b.jsonl
RC=$?
if [ $RC -ne 0 ]; then
    echo "HARM_JUDGE_FAILED rc=$RC" >> "$OUT_DIR/.training_failed"
    exit 1
fi

touch "$OUT_DIR/.training_done"
echo "=== ALL DONE ==="
date -Is
