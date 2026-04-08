#!/bin/bash
# Provision a Vast.ai instance and launch the layer-31 clamping experiment.
#
# Requires: vastai CLI, ~/.fastai_key (Vast API key), ~/.hf_token, ~/.together
#
# Usage: bash scripts/launch_clamp.sh
set -e

IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK=80
LOCAL_TURNSTILE="/Users/julianquick/portfolio_copy/turnstile"
ADAPTER_SRC="experiments/stealth_s42/adapters"
EXPERIMENT_NAME="clamp_v1"
INSTANCE_FILE="$LOCAL_TURNSTILE/experiments/clamp_instance.txt"

# Files to upload
UPLOAD_FILES=(
    "turnstile/"
    "data/"
)

echo "=== LAUNCHING CLAMPING EXPERIMENT ==="
echo "  Adapter: $ADAPTER_SRC"
echo ""

# --- Find cheapest 4090 ---
echo "Searching for GPU offers..."
OFFER_ID=$(echo "n" | vastai search offers \
    'gpu_name=RTX_4090 num_gpus=1 dph<0.50 inet_down>600 disk_space>=80 reliability>0.998' \
    --order 'dph' --limit 1 --raw 2>&1 | grep -v "Update\|selected" | python3 -c "
import json, sys
data = json.load(sys.stdin)
if data:
    print(data[0]['id'])
    print(f'  Price: \${data[0][\"dph_total\"]:.3f}/hr', file=sys.stderr)
    print(f'  GPU: {data[0][\"gpu_name\"]}', file=sys.stderr)
else:
    print('')
")

if [ -z "$OFFER_ID" ]; then
    echo "  No suitable offers found! Try relaxing constraints."
    exit 1
fi
echo "  Offer ID: $OFFER_ID"

# --- Create instance ---
echo ""
echo "Creating instance..."
RESULT=$(echo "n" | vastai create instance $OFFER_ID \
    --image $IMAGE --disk $DISK --ssh --direct \
    --label "turnstile-clamp" --raw 2>&1 | grep -v "Update\|selected")
INSTANCE_ID=$(echo "$RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin)['new_contract'])")
echo "  Instance ID: $INSTANCE_ID"

# --- Wait for running ---
echo "  Waiting for instance to start..."
for i in $(seq 1 40); do
    STATUS=$(echo "n" | vastai show instance $INSTANCE_ID --raw 2>&1 | \
        grep -v "Update\|selected" | \
        python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('actual_status','?'))" 2>/dev/null)
    if [ "$STATUS" = "running" ]; then
        echo "  Instance running!"
        break
    fi
    echo "    status: $STATUS ($i/40)"
    sleep 15
done

if [ "$STATUS" != "running" ]; then
    echo "  [ERROR] Instance did not start. Destroying..."
    echo "n" | vastai destroy instance $INSTANCE_ID 2>&1 | grep -v "Update"
    exit 1
fi

# --- Get SSH info ---
SSH_INFO=$(echo "n" | vastai show instance $INSTANCE_ID --raw 2>&1 | \
    grep -v "Update\|selected" | \
    python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{d[\"ssh_host\"]} {d[\"ssh_port\"]}')")
SSH_HOST=$(echo $SSH_INFO | cut -d' ' -f1)
SSH_PORT=$(echo $SSH_INFO | cut -d' ' -f2)
echo "  SSH: ssh -p $SSH_PORT root@$SSH_HOST"

# Save instance info
echo "$INSTANCE_ID $SSH_HOST $SSH_PORT" > "$INSTANCE_FILE"
echo "  Instance info saved to $INSTANCE_FILE"

# --- Wait for SSH ---
echo "  Waiting for SSH..."
for i in $(seq 1 20); do
    ssh -p $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@$SSH_HOST "echo ok" 2>/dev/null && break
    sleep 10
done

# --- Install dependencies ---
echo ""
echo "Installing dependencies..."
ssh -p $SSH_PORT -o StrictHostKeyChecking=no root@$SSH_HOST \
    "pip install peft bitsandbytes accelerate scikit-learn jailbreakbench requests 2>&1 | tail -3 && pip uninstall hf-xet -y 2>&1 | tail -1"

# --- Upload HF token and Together key ---
echo ""
echo "Uploading credentials..."
scp -P $SSH_PORT -o StrictHostKeyChecking=no ~/.hf_token root@$SSH_HOST:/root/.hf_token 2>/dev/null
scp -P $SSH_PORT -o StrictHostKeyChecking=no ~/.together root@$SSH_HOST:/root/.together 2>/dev/null
ssh -p $SSH_PORT -o StrictHostKeyChecking=no root@$SSH_HOST \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\"" 2>/dev/null

# --- Upload code and data ---
echo ""
echo "Uploading code and data..."
ssh -p $SSH_PORT -o StrictHostKeyChecking=no root@$SSH_HOST "mkdir -p /workspace/turnstile" 2>/dev/null

for f in "${UPLOAD_FILES[@]}"; do
    echo "  uploading $f..."
    rsync -az -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
        "$LOCAL_TURNSTILE/$f" "root@$SSH_HOST:/workspace/turnstile/$f" 2>/dev/null
done

# Upload adapter
echo "  uploading adapter..."
ssh -p $SSH_PORT -o StrictHostKeyChecking=no root@$SSH_HOST \
    "mkdir -p /workspace/turnstile/$ADAPTER_SRC" 2>/dev/null
rsync -az -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
    "$LOCAL_TURNSTILE/$ADAPTER_SRC/" "root@$SSH_HOST:/workspace/turnstile/$ADAPTER_SRC/" 2>/dev/null

# --- Launch experiment ---
echo ""
echo "Launching experiment..."
ssh -p $SSH_PORT -o StrictHostKeyChecking=no root@$SSH_HOST \
    "cd /workspace/turnstile && nohup python -m turnstile.clamp_experiment \
        --adapter $ADAPTER_SRC \
        --n-goals 100 \
        --num-turns 5 \
        --output experiments/$EXPERIMENT_NAME \
        > experiments/clamp.log 2>&1 &"

echo ""
echo "=== EXPERIMENT LAUNCHED ==="
echo "  Instance:  $INSTANCE_ID"
echo "  SSH:       ssh -p $SSH_PORT root@$SSH_HOST"
echo "  Log:       ssh -p $SSH_PORT root@$SSH_HOST 'tail -f /workspace/turnstile/experiments/clamp.log'"
echo "  Monitor:   bash scripts/babysit_clamp.sh"
echo ""
echo "  Instance info: $INSTANCE_FILE"
