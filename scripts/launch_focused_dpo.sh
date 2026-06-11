#!/bin/bash
# Focused DPO on Network hacking goal + watchdog.
set -e
IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK=80
LOCAL="/Users/julianquick/portfolio_copy/turnstile"
INSTANCE_FILE="$LOCAL/experiments/focused_dpo_instance.txt"

echo "=== LAUNCH FOCUSED DPO (Network hacking) ==="

OFFER_ID=$(echo "n" | vastai search offers \
    'gpu_name=RTX_4090 num_gpus=1 dph<0.50 inet_down>600 disk_space>=80 reliability>0.998' \
    --order 'dph' --limit 1 --raw 2>&1 | grep -v "Update\|selected" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d[0]['id']) if d else print('')
")
[ -z "$OFFER_ID" ] && echo "No offers" && exit 1
echo "  Offer: $OFFER_ID"

RESULT=$(echo "n" | vastai create instance $OFFER_ID \
    --image $IMAGE --disk $DISK --ssh --direct \
    --label "turnstile-focdpo" --raw 2>&1 | grep -v "Update\|selected")
INST=$(echo "$RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin)['new_contract'])")
echo "  Instance: $INST"

for i in $(seq 1 40); do
    STATUS=$(echo "n" | vastai show instance $INST --raw 2>&1 | \
        grep -v "Update\|selected" | \
        python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('actual_status','?'))" 2>/dev/null)
    [ "$STATUS" = "running" ] && break
    echo "    status: $STATUS ($i/40)"
    sleep 15
done
[ "$STATUS" != "running" ] && echo "FAIL" && echo "n" | vastai destroy instance $INST 2>&1 | grep -v Update && exit 1

SSH_INFO=$(echo "n" | vastai show instance $INST --raw 2>&1 | \
    grep -v "Update\|selected" | \
    python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{d[\"ssh_host\"]} {d[\"ssh_port\"]}')")
HOST=$(echo $SSH_INFO | cut -d' ' -f1)
PORT=$(echo $SSH_INFO | cut -d' ' -f2)
echo "$INST $HOST $PORT" > "$INSTANCE_FILE"
echo "  SSH: ssh -p $PORT root@$HOST"

for i in $(seq 1 20); do
    ssh -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@$HOST "echo ok" 2>/dev/null && break
    sleep 10
done

echo "Installing deps..."
ssh -p $PORT -o StrictHostKeyChecking=no root@$HOST \
    "pip install peft bitsandbytes accelerate scikit-learn jailbreakbench requests 2>&1 | tail -1 && pip uninstall hf-xet -y 2>&1 | tail -1"

scp -P $PORT -o StrictHostKeyChecking=no ~/.hf_token root@$HOST:/root/.hf_token 2>/dev/null
scp -P $PORT -o StrictHostKeyChecking=no ~/.together root@$HOST:/root/.together 2>/dev/null
ssh -p $PORT -o StrictHostKeyChecking=no root@$HOST \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\"" 2>/dev/null

echo "Uploading..."
ssh -p $PORT -o StrictHostKeyChecking=no root@$HOST \
    "mkdir -p /workspace/turnstile/experiments/stealth_s42" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/turnstile/" "root@$HOST:/workspace/turnstile/turnstile/" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/data/" "root@$HOST:/workspace/turnstile/data/" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/experiments/stealth_s42/adapters/" \
    "root@$HOST:/workspace/turnstile/experiments/stealth_s42/adapters/" 2>/dev/null

echo "Launching focused DPO..."
ssh -p $PORT -o StrictHostKeyChecking=no root@$HOST \
    "cd /workspace/turnstile && nohup python -u -m turnstile.focused_dpo \
        --name nethack_focused \
        --behavior 'Network hacking' \
        --adapter experiments/stealth_s42/adapters \
        --rounds 15 --candidates 50 \
        > experiments/focused_dpo.log 2>&1 &"

echo ""
echo "=== LAUNCHED ==="
echo "  Instance: $INST"
echo "  SSH: ssh -p $PORT root@$HOST"
echo "  Log: ssh -p $PORT root@$HOST 'tail -f /workspace/turnstile/experiments/focused_dpo.log'"
