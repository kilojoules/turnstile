#!/bin/bash
# Launch parallel instances for seeds 123 and 456.
# Seed 42 continues on the existing instance.
set -e

SEEDS_NEW="123 456"
IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK=80
LOCAL_TURNSTILE="/Users/julianquick/portfolio_copy/turnstile"

# Files to upload to each instance
UPLOAD_FILES=(
    "turnstile/"
    "scripts/"
    "data/jbb_verified_wins.jsonl"
    "results/rejudge_jbb.jsonl"
    "results/rejudge_jbb_remaining.jsonl"
    "results/probe_jbb/"
    "results/hidden_states_jbb/"
    "experiments/selfplay_jbb_v1/adapters/"
)

echo "=== LAUNCHING PARALLEL INSTANCES ==="

for SEED in $SEEDS_NEW; do
    echo ""
    echo "--- Seed $SEED ---"

    # Find cheapest 4090
    OFFER_ID=$(echo "n" | vastai search offers \
        'gpu_name=RTX_4090 num_gpus=1 dph<0.45 inet_down>200 disk_space>=50 reliability>0.97' \
        --order 'dph' --limit 1 --raw 2>&1 | grep -v "Update\|selected" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data[0]['id']) if data else print('')
")

    if [ -z "$OFFER_ID" ]; then
        echo "  No offers found for seed $SEED!"
        continue
    fi
    echo "  Offer: $OFFER_ID"

    # Create instance
    RESULT=$(echo "n" | vastai create instance $OFFER_ID \
        --image $IMAGE --disk $DISK --ssh --direct \
        --label "turnstile-s${SEED}" --raw 2>&1 | grep -v "Update\|selected")
    INSTANCE_ID=$(echo "$RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin)['new_contract'])")
    echo "  Instance: $INSTANCE_ID"

    # Wait for running
    echo "  Waiting for instance..."
    for i in $(seq 1 30); do
        STATUS=$(echo "n" | vastai show instance $INSTANCE_ID --raw 2>&1 | \
            grep -v "Update\|selected" | \
            python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('actual_status','?'))" 2>/dev/null)
        if [ "$STATUS" = "running" ]; then
            break
        fi
        sleep 10
    done

    # Get SSH info
    SSH_INFO=$(echo "n" | vastai show instance $INSTANCE_ID --raw 2>&1 | \
        grep -v "Update\|selected" | \
        python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{d[\"ssh_host\"]} {d[\"ssh_port\"]}')")
    SSH_HOST=$(echo $SSH_INFO | cut -d' ' -f1)
    SSH_PORT=$(echo $SSH_INFO | cut -d' ' -f2)
    echo "  SSH: ssh -p $SSH_PORT root@$SSH_HOST"

    # Wait for SSH
    for i in $(seq 1 12); do
        ssh -p $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@$SSH_HOST "echo ok" 2>/dev/null && break
        sleep 10
    done

    # Install deps
    echo "  Installing deps..."
    ssh -p $SSH_PORT -o StrictHostKeyChecking=no root@$SSH_HOST \
        "pip install peft bitsandbytes accelerate scikit-learn jailbreakbench requests 2>&1 | tail -1" 2>/dev/null

    # HF login
    scp -P $SSH_PORT -o StrictHostKeyChecking=no ~/.hf_token root@$SSH_HOST:/root/.hf_token 2>/dev/null
    scp -P $SSH_PORT -o StrictHostKeyChecking=no ~/.together root@$SSH_HOST:/root/.together 2>/dev/null
    ssh -p $SSH_PORT -o StrictHostKeyChecking=no root@$SSH_HOST \
        "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\"" 2>/dev/null

    # Upload code and data
    echo "  Uploading..."
    ssh -p $SSH_PORT -o StrictHostKeyChecking=no root@$SSH_HOST "mkdir -p /workspace/turnstile" 2>/dev/null
    for f in "${UPLOAD_FILES[@]}"; do
        rsync -az -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
            "$LOCAL_TURNSTILE/$f" "root@$SSH_HOST:/workspace/turnstile/$f" 2>/dev/null
    done

    # Launch
    echo "  Launching seed $SEED..."
    ssh -p $SSH_PORT -o StrictHostKeyChecking=no root@$SSH_HOST \
        "nohup bash /workspace/turnstile/scripts/run_seed.sh $SEED > /workspace/turnstile/loop_s${SEED}.log 2>&1 &"

    echo "  LAUNCHED: instance=$INSTANCE_ID, seed=$SEED, ssh=$SSH_HOST:$SSH_PORT"
    echo "$INSTANCE_ID $SEED $SSH_HOST $SSH_PORT" >> "$LOCAL_TURNSTILE/experiments/parallel_instances.txt"
done

echo ""
echo "=== ALL LAUNCHED ==="
echo "Instance info saved to experiments/parallel_instances.txt"
echo "Monitor with: ssh -p PORT root@HOST 'tail -f /workspace/turnstile/loop_sSEED.log'"
