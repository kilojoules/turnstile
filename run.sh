#!/bin/bash
# Turnstile MVP — run on Vast.ai GPU instance
set -e

MODEL="meta-llama/Llama-3.1-8B-Instruct"

echo "=== Installing dependencies ==="
pip install -q vllm openai

echo "=== Logging into HuggingFace ==="
huggingface-cli login --token "$(cat ~/.hf_token)" 2>/dev/null || true

echo "=== Starting vLLM server ==="
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --dtype auto &
VLLM_PID=$!

echo "Waiting for vLLM server..."
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 3
done
echo "Server ready!"

echo "=== Running Turnstile ==="
python -m turnstile.main --model "$MODEL" --num-turns 5

echo "=== Done ==="
kill $VLLM_PID 2>/dev/null || true
