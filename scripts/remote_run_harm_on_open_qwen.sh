#!/bin/bash
# Box-2 runner: gen (deterministic, same as box 1) -> judge Qwen-72B only.
set -e
cd /root/c
export HF_TOKEN=$(cat /root/.hf_token 2>/dev/null)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
echo "=== installing deps ==="
pip -q install "transformers>=4.44,<4.46" accelerate bitsandbytes sentencepiece protobuf 2>&1 | tail -2
echo "=== [1/2] generation (8B victim, composed steering) ==="
python3 scripts/steer_harm_on_open.py 2>&1 | tee /root/gen.log
test -s experiments/steer_harm_on_open/sweep.jsonl || { echo "GEN FAILED"; exit 1; }
echo "=== [2/2] judge Qwen2.5-72B ==="
python3 scripts/judge_postresponse_sweep.py \
  --input experiments/steer_harm_on_open/sweep.jsonl \
  --output experiments/steer_harm_on_open/judged_qwen.jsonl \
  --model Qwen/Qwen2.5-72B-Instruct 2>&1 | tee /root/jq.log
touch /root/ALLDONE_QWEN
echo "=== QWEN DONE ==="
