#!/bin/bash
# Extend decoded steering to goals[90:100] (the missing 10 -> goals[50:]). gen -> Llama -> (free70B) -> Qwen.
set -e
cd /root/c
export HF_TOKEN=$(cat /root/.hf_token 2>/dev/null); export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN HF_HUB_DISABLE_XET=1
export CVH_OUT=experiments/steer_decoded_ext10
export CVH_GOALS="90:100"
export CVH_DIRS="comp_pre_llama,comp_pre_qwen,harm_pre_llama,comp_dm_out"
export CVH_ALPHAS="-1.5,-1.0,-0.5,0.5,1.0,1.5"
pip -q install "transformers>=4.44,<4.46" accelerate bitsandbytes sentencepiece protobuf >/root/pip.log 2>&1 || true
D=experiments/steer_decoded_ext10
echo "=== gen ==="; python3 scripts/steer_cvh_matched.py >/root/gen.log 2>&1
test -s $D/sweep.jsonl || { echo GEN_FAILED; tail -5 /root/gen.log; exit 1; }
echo "rows=$(wc -l <$D/sweep.jsonl)"
echo "=== Llama-70B ==="; python3 scripts/judge_postresponse_sweep.py --input $D/sweep.jsonl --output $D/judged_llama.jsonl >/root/jl.log 2>&1
test "$(wc -l <$D/judged_llama.jsonl)" -ge 1 || { echo LLAMA_FAILED; tail -5 /root/jl.log; exit 1; }
rm -rf /root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-70B-Instruct*
echo "=== Qwen-72B ==="; python3 scripts/judge_postresponse_sweep.py --input $D/sweep.jsonl --output $D/judged_qwen.jsonl --model Qwen/Qwen2.5-72B-Instruct >/root/jq.log 2>&1
test "$(wc -l <$D/judged_qwen.jsonl)" -ge 1 || { echo QWEN_FAILED; tail -5 /root/jq.log; exit 1; }
touch /root/ALLDONE; echo "ALL_DONE l=$(wc -l <$D/judged_llama.jsonl) q=$(wc -l <$D/judged_qwen.jsonl)"
