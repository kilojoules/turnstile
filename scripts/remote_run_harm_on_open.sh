#!/bin/bash
# Remote runner for the harm-on-open composed-steering experiment.
# Gen (Llama-3.1-8B victim) -> judge Llama-70B -> judge Qwen-72B. Writes /root/ALLDONE on success.
set -e
cd /root/c
export HF_TOKEN=$(cat /root/.hf_token 2>/dev/null)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "=== installing deps ==="
pip -q install "transformers>=4.44,<4.46" accelerate bitsandbytes sentencepiece protobuf 2>&1 | tail -2

echo "=== [1/3] generation (8B victim, composed steering) ==="
python3 scripts/steer_harm_on_open.py 2>&1 | tee /root/gen.log
test -s experiments/steer_harm_on_open/sweep.jsonl || { echo "GEN FAILED"; exit 1; }
echo "sweep rows: $(wc -l < experiments/steer_harm_on_open/sweep.jsonl)"

echo "=== [2/3] judge Llama-3.1-70B ==="
python3 scripts/judge_postresponse_sweep.py \
  --input experiments/steer_harm_on_open/sweep.jsonl \
  --output experiments/steer_harm_on_open/judged_llama.jsonl 2>&1 | tee /root/jl.log

echo "=== [3/3] judge Qwen2.5-72B ==="
python3 scripts/judge_postresponse_sweep.py \
  --input experiments/steer_harm_on_open/sweep.jsonl \
  --output experiments/steer_harm_on_open/judged_qwen.jsonl \
  --model Qwen/Qwen2.5-72B-Instruct 2>&1 | tee /root/jq.log

touch /root/ALLDONE
echo "=== ALL DONE ==="
