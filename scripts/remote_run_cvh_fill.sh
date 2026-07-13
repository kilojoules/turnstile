#!/bin/bash
# Grid-fill: harm dirs at the missing alphas, gen -> Llama-70B -> Qwen-72B.
set -e
cd /root/c
export HF_TOKEN=$(cat /root/.hf_token 2>/dev/null); export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export CVH_OUT=experiments/steer_cvh_fill
export CVH_ALPHAS="-0.25,0.25,0.75"
export CVH_DIRS="harm_dm_llama,harm_dm_resid,random_1"
export CVH_SKIP_BASELINE=1
pip -q install "transformers>=4.44,<4.46" accelerate bitsandbytes sentencepiece protobuf 2>&1 | tail -1
echo "=== [1/3] gen fill ==="; python3 scripts/steer_cvh_matched.py 2>&1 | tee /root/gen.log
test -s experiments/steer_cvh_fill/sweep.jsonl || { echo GEN_FAILED; exit 1; }
echo "rows: $(wc -l <experiments/steer_cvh_fill/sweep.jsonl)"
echo "=== [2/3] Llama-70B ==="; python3 scripts/judge_postresponse_sweep.py --input experiments/steer_cvh_fill/sweep.jsonl --output experiments/steer_cvh_fill/judged_llama.jsonl 2>&1 | tee /root/jl.log
echo "=== [3/3] Qwen-72B ==="; python3 scripts/judge_postresponse_sweep.py --input experiments/steer_cvh_fill/sweep.jsonl --output experiments/steer_cvh_fill/judged_qwen.jsonl --model Qwen/Qwen2.5-72B-Instruct 2>&1 | tee /root/jq.log
touch /root/ALLDONE; echo ALL_DONE
