#!/bin/bash
# Box-1 (self-sufficient): gen -> Llama-70B judge -> Qwen-72B judge.
set -e
cd /root/c
export HF_TOKEN=$(cat /root/.hf_token 2>/dev/null); export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
pip -q install "transformers>=4.44,<4.46" accelerate bitsandbytes sentencepiece protobuf 2>&1 | tail -1
echo "=== [1/3] gen ==="; python3 scripts/steer_cvh_matched.py 2>&1 | tee /root/gen.log
test -s experiments/steer_cvh_matched/sweep.jsonl || { echo GEN_FAILED; exit 1; }
echo "=== [2/3] Llama-70B ==="; python3 scripts/judge_postresponse_sweep.py --input experiments/steer_cvh_matched/sweep.jsonl --output experiments/steer_cvh_matched/judged_llama.jsonl 2>&1 | tee /root/jl.log
echo "=== [3/3] Qwen-72B ==="; python3 scripts/judge_postresponse_sweep.py --input experiments/steer_cvh_matched/sweep.jsonl --output experiments/steer_cvh_matched/judged_qwen.jsonl --model Qwen/Qwen2.5-72B-Instruct 2>&1 | tee /root/jq.log
touch /root/ALLDONE; echo ALL_DONE
