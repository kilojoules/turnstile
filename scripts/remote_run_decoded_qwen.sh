#!/bin/bash
# Box-2 parallel: gen (deterministic, same sweep) -> Qwen-72B judge only.
set -e
cd /root/c
export HF_TOKEN=$(cat /root/.hf_token 2>/dev/null); export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN HF_HUB_DISABLE_XET=1
export CVH_OUT=experiments/steer_decoded
export CVH_DIRS="comp_pre_llama,comp_pre_qwen,harm_pre_llama,comp_dm_out"
export CVH_ALPHAS="-1.5,-1.0,-0.5,0.5,1.0,1.5"
pip -q install "transformers>=4.44,<4.46" accelerate bitsandbytes sentencepiece protobuf >/root/pip.log 2>&1 || true
D=experiments/steer_decoded
echo "=== gen ==="; python3 scripts/steer_cvh_matched.py >/root/gen.log 2>&1
test -s $D/sweep.jsonl || { echo GEN_FAILED; tail -5 /root/gen.log; exit 1; }
echo "=== Qwen-72B ==="; python3 scripts/judge_postresponse_sweep.py --input $D/sweep.jsonl --output $D/judged_qwen.jsonl --model Qwen/Qwen2.5-72B-Instruct >/root/jq.log 2>&1
touch /root/ALLDONE_QWEN; echo "QWEN_DONE $(wc -l <$D/judged_qwen.jsonl)"
