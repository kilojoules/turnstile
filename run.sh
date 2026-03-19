#!/bin/bash
# Turnstile MVP — run on Vast.ai GPU instance
set -e

MODEL="meta-llama/Llama-3.1-8B-Instruct"

echo "=== Installing pixi ==="
curl -fsSL https://pixi.sh/install.sh | bash
export PATH="$HOME/.pixi/bin:$PATH"

echo "=== Logging into HuggingFace ==="
huggingface-cli login --token "$(cat ~/.hf_token)" 2>/dev/null || true

echo "=== Installing dependencies ==="
pixi install

echo "=== Starting vLLM server ==="
pixi run serve &
VLLM_PID=$!

echo "Waiting for vLLM server..."
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 3
done
echo "Server ready!"

echo "=== Running Turnstile ==="
pixi run run --model "$MODEL" --num-turns 5

echo "=== Done ==="
kill $VLLM_PID 2>/dev/null || true
