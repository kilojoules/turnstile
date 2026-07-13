#!/bin/bash
# Provision a Vast.ai L40 (or A40/A6000) instance and launch the paired-
# obliteration pipeline (rewrite -> replay -> judge -> probe).
#
# Outputs instance info to experiments/intent_obliteration_paired/instance.txt
# Then run scripts/babysit_intent_obliteration_paired.sh to monitor + tear down.
#
# Requires: vastai CLI, ~/.fastai_key, ~/.hf_token
# Usage: bash scripts/launch_intent_obliteration_paired.sh
set -e

IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK=200
LABEL="turnstile-intent-obliteration-paired"
LOCAL_TURNSTILE="/Users/julianquick/portfolio_copy/turnstile"
INSTANCE_FILE="$LOCAL_TURNSTILE/experiments/intent_obliteration_paired/instance.txt"

# Files to upload (rsync; relative to LOCAL_TURNSTILE)
UPLOAD_DIRS=(
    "turnstile/"
    "data/"
    "experiments/authority_dpo/rounds/"
    "experiments/control_hard_s456/rounds/"
    "experiments/control_s42/rounds/"
    "experiments/frozen_v1/rounds/"
    "experiments/incrementalism_dpo/rounds/"
    "experiments/reward_dpo/rounds/"
    "experiments/stealth_hard_s456/rounds/"
    "experiments/stealth_jbb_v1/rounds/"
    "experiments/stealth_s42/rounds/"
    "experiments/urgency_dpo/rounds/"
    "experiments/urgency_v1/rounds/"
)

mkdir -p "$LOCAL_TURNSTILE/experiments/intent_obliteration_paired"

echo "=== LAUNCHING PAIRED-OBLITERATION PIPELINE ==="

# --- Find cheapest L40/A40/A6000 with >=45GB VRAM and >=200GB disk ---
echo "Searching for GPU offers (>=45GB VRAM, >=200GB disk)..."
OFFER_ID=$(echo "n" | vastai search offers \
    'gpu_name in [L40,L40S,A40,A6000,A100,A100_PCIE,A100_SXM4] num_gpus=1 dph<1.0 inet_down>500 disk_space>=200 reliability>0.99 cuda_max_good>=12.0' \
    --order 'dph' --limit 1 --raw 2>&1 | grep -v "Update\|selected" | python3 -c "
import json, sys
data = json.load(sys.stdin)
if data:
    o = data[0]
    print(o['id'])
    print(f'  GPU: {o[\"gpu_name\"]}', file=sys.stderr)
    print(f'  VRAM: {o.get(\"gpu_ram\",0)/1024:.0f} GB', file=sys.stderr)
    print(f'  Disk: {o.get(\"disk_space\",0):.0f} GB', file=sys.stderr)
    print(f'  Price: \${o[\"dph_total\"]:.3f}/hr', file=sys.stderr)
    print(f'  Reliability: {o.get(\"reliability2\",0):.4f}', file=sys.stderr)
else:
    print('')
")

if [ -z "$OFFER_ID" ]; then
    echo "  No suitable offers found! Try relaxing constraints."
    exit 1
fi

# --- Create instance ---
echo ""
echo "Creating instance..."
RESULT=$(echo "n" | vastai create instance "$OFFER_ID" \
    --image "$IMAGE" --disk "$DISK" --ssh --direct \
    --label "$LABEL" --raw 2>&1 | grep -v "Update\|selected")
INSTANCE_ID=$(echo "$RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin)['new_contract'])")
echo "  Instance ID: $INSTANCE_ID"

# --- Wait for running ---
echo "  Waiting for instance to start..."
STATUS=""
for i in $(seq 1 60); do
    STATUS=$(echo "n" | vastai show instance "$INSTANCE_ID" --raw 2>&1 | \
        grep -v "Update\|selected" | \
        python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('actual_status','?'))" 2>/dev/null)
    if [ "$STATUS" = "running" ]; then
        echo "  Instance running!"
        break
    fi
    echo "    status: $STATUS ($i/60)"
    sleep 15
done

if [ "$STATUS" != "running" ]; then
    echo "  [ERROR] Instance did not start. Destroying..."
    echo "n" | vastai destroy instance "$INSTANCE_ID" 2>&1 | grep -v "Update"
    exit 1
fi

# --- Get SSH info ---
SSH_INFO=$(echo "n" | vastai show instance "$INSTANCE_ID" --raw 2>&1 | \
    grep -v "Update\|selected" | \
    python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{d[\"ssh_host\"]} {d[\"ssh_port\"]}')")
SSH_HOST=$(echo "$SSH_INFO" | cut -d' ' -f1)
SSH_PORT=$(echo "$SSH_INFO" | cut -d' ' -f2)
echo "  SSH: ssh -p $SSH_PORT root@$SSH_HOST"

echo "$INSTANCE_ID $SSH_HOST $SSH_PORT" > "$INSTANCE_FILE"
echo "  Instance info saved to $INSTANCE_FILE"

# --- Wait for SSH ---
echo "  Waiting for SSH..."
SSH_OK=0
for i in $(seq 1 30); do
    if ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
           root@"$SSH_HOST" "echo ok" 2>/dev/null; then
        SSH_OK=1
        break
    fi
    sleep 10
done
if [ "$SSH_OK" -ne 1 ]; then
    echo "  [ERROR] SSH never came up. Destroying..."
    echo "n" | vastai destroy instance "$INSTANCE_ID" 2>&1 | grep -v "Update"
    exit 1
fi

# --- Install dependencies ---
echo ""
echo "Installing dependencies..."
ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no root@"$SSH_HOST" \
    "pip install -q transformers peft bitsandbytes accelerate scikit-learn requests jailbreakbench 2>&1 | tail -3 && pip uninstall -y hf-xet 2>&1 | tail -1" || true

# --- Upload HF token ---
# Prefer ~/.hf_token_2 (full-read role, can download gated repo files); fall
# back to ~/.hf_token (which is fine-grained and only allows API metadata
# reads on gated repos -- 403s on file downloads).
HF_TOKEN_FILE="$HOME/.hf_token_2"
if [ ! -f "$HF_TOKEN_FILE" ]; then
    HF_TOKEN_FILE="$HOME/.hf_token"
fi
echo "Uploading HF token from $HF_TOKEN_FILE ..."
scp -P "$SSH_PORT" -o StrictHostKeyChecking=no "$HF_TOKEN_FILE" \
    root@"$SSH_HOST":/root/.hf_token

# --- Upload code + rounds data ---
echo ""
echo "Uploading code and conversation data..."
ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no root@"$SSH_HOST" \
    "mkdir -p /workspace/turnstile"
for d in "${UPLOAD_DIRS[@]}"; do
    echo "  uploading $d..."
    ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no root@"$SSH_HOST" \
        "mkdir -p /workspace/turnstile/$(dirname "$d")" 2>/dev/null
    rsync -az -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
        "$LOCAL_TURNSTILE/$d" "root@$SSH_HOST:/workspace/turnstile/$d"
done

# --- Upload remote runner script ---
echo ""
echo "Uploading remote runner..."
scp -P "$SSH_PORT" -o StrictHostKeyChecking=no \
    "$LOCAL_TURNSTILE/scripts/run_intent_obliteration_paired_remote.sh" \
    root@"$SSH_HOST":/workspace/run_remote.sh
ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no root@"$SSH_HOST" \
    "chmod +x /workspace/run_remote.sh"

# --- Launch the pipeline ---
echo ""
echo "Launching pipeline (nohup)..."
ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no root@"$SSH_HOST" \
    "cd /workspace && nohup bash run_remote.sh > /workspace/run.log 2>&1 < /dev/null &"

echo ""
echo "=== LAUNCH COMPLETE ==="
echo "  Instance:  $INSTANCE_ID"
echo "  SSH:       ssh -p $SSH_PORT root@$SSH_HOST"
echo "  Log:       ssh -p $SSH_PORT root@$SSH_HOST 'tail -f /workspace/run.log'"
echo "  Babysit:   bash scripts/babysit_intent_obliteration_paired.sh"
echo "  Destroy:   vastai destroy instance $INSTANCE_ID"
echo ""
echo "  Instance info: $INSTANCE_FILE"
