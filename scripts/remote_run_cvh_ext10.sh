#!/bin/bash
# ext10: gen -> Llama-70B judge -> (free 70B cache) -> Qwen-72B judge. Direct redirects so set -e catches crashes.
set -e
cd /root/c
export HF_TOKEN=$(cat /root/.hf_token 2>/dev/null); export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export HF_HUB_DISABLE_XET=1
pip -q install "transformers>=4.44,<4.46" accelerate bitsandbytes sentencepiece protobuf >/root/pip.log 2>&1 || true
D=experiments/steer_cvh_ext10
echo "=== [1/3] gen ==="; python3 scripts/steer_cvh_ext10.py >/root/gen.log 2>&1
test -s $D/sweep.jsonl || { echo GEN_FAILED; tail -5 /root/gen.log; exit 1; }
echo "gen rows: $(wc -l <$D/sweep.jsonl)"
echo "=== [2/3] Llama-70B ==="; python3 scripts/judge_postresponse_sweep.py --input $D/sweep.jsonl --output $D/judged_llama.jsonl >/root/jl.log 2>&1
test "$(wc -l <$D/judged_llama.jsonl)" -ge 1 || { echo LLAMA_JUDGE_FAILED; tail -5 /root/jl.log; exit 1; }
echo "  llama rows: $(wc -l <$D/judged_llama.jsonl)"
echo "=== free 70B cache before Qwen (disk) ==="; rm -rf /root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-70B-Instruct* ; df -h /root | tail -1
echo "=== [3/3] Qwen-72B ==="; python3 scripts/judge_postresponse_sweep.py --input $D/sweep.jsonl --output $D/judged_qwen.jsonl --model Qwen/Qwen2.5-72B-Instruct >/root/jq.log 2>&1
test "$(wc -l <$D/judged_qwen.jsonl)" -ge 1 || { echo QWEN_JUDGE_FAILED; tail -5 /root/jq.log; exit 1; }
touch /root/ALLDONE; echo "ALL_DONE llama=$(wc -l <$D/judged_llama.jsonl) qwen=$(wc -l <$D/judged_qwen.jsonl)"
