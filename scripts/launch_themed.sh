#!/bin/bash
# Launch 4 Vast.ai 4090 instances, one per attack theme.
#
# Usage:
#   bash scripts/launch_themed.sh                          # all 4
#   bash scripts/launch_themed.sh urgency incrementalism   # subset
set -e

IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK=80
LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
INSTANCE_FILE="$LOCAL/experiments/themed_instances.txt"

if [ $# -gt 0 ]; then
    THEMES=("$@")
else
    THEMES=("urgency" "incrementalism" "reward" "authority")
fi

# Files to upload
UPLOAD_DIRS=(
    "turnstile/"
    "scripts/"
)

echo "=============================================="
echo "  LAUNCHING THEMED EXPERIMENTS"
echo "  Themes: ${THEMES[*]}"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=============================================="

mkdir -p "$LOCAL/experiments"
> "$INSTANCE_FILE"  # clear instance file

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"

for THEME in "${THEMES[@]}"; do
    echo ""
    echo "--- Theme: $THEME ---"

    # Find cheapest 4090
    OFFER_ID=$(echo "n" | vastai search offers \
        'gpu_name=RTX_4090 num_gpus=1 dph<0.50 inet_down>200 disk_space>=80 reliability>0.97' \
        --order 'dph' --limit 1 --raw 2>&1 | grep -v "Update\|selected" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data[0]['id']) if data else print('')
")

    if [ -z "$OFFER_ID" ]; then
        echo "  ERROR: No 4090 offers found! Skipping $THEME."
        continue
    fi
    echo "  Offer: $OFFER_ID"

    # Create instance
    RESULT=$(echo "n" | vastai create instance $OFFER_ID \
        --image $IMAGE --disk $DISK --ssh --direct \
        --label "themed-${THEME}" --raw 2>&1 | grep -v "Update\|selected")
    INST=$(echo "$RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin)['new_contract'])")
    echo "  Instance: $INST"

    # Wait for running (up to 5 minutes)
    echo "  Waiting for instance to start..."
    for i in $(seq 1 30); do
        STATUS=$(echo "n" | vastai show instance $INST --raw 2>&1 | \
            grep -v "Update\|selected" | \
            python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('actual_status','?'))" 2>/dev/null)
        if [ "$STATUS" = "running" ]; then
            echo "  Running."
            break
        fi
        sleep 10
    done

    if [ "$STATUS" != "running" ]; then
        echo "  ERROR: Instance $INST did not start. Destroying."
        echo "n" | vastai destroy instance $INST 2>/dev/null
        continue
    fi

    # Get SSH info
    SSH_INFO=$(echo "n" | vastai show instance $INST --raw 2>&1 | \
        grep -v "Update\|selected" | \
        python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{d[\"ssh_host\"]} {d[\"ssh_port\"]}')")
    HOST=$(echo $SSH_INFO | cut -d' ' -f1)
    PORT=$(echo $SSH_INFO | cut -d' ' -f2)
    echo "  SSH: ssh -p $PORT root@$HOST"

    # Wait for SSH (up to 2 minutes)
    echo "  Waiting for SSH..."
    SSH_OK=false
    for i in $(seq 1 12); do
        if ssh -p $PORT $SSH_OPTS root@$HOST "echo ok" 2>/dev/null; then
            SSH_OK=true
            break
        fi
        sleep 10
    done

    if ! $SSH_OK; then
        echo "  ERROR: SSH failed for $THEME. Destroying instance $INST."
        echo "n" | vastai destroy instance $INST 2>/dev/null
        continue
    fi

    # Install deps
    echo "  Installing dependencies..."
    ssh -p $PORT $SSH_OPTS root@$HOST \
        "pip install peft bitsandbytes accelerate scikit-learn jailbreakbench requests 2>&1 | tail -1" 2>/dev/null

    # Upload credentials
    echo "  Uploading credentials..."
    scp -P $PORT $SSH_OPTS ~/.hf_token root@$HOST:/root/.hf_token 2>/dev/null
    scp -P $PORT $SSH_OPTS ~/.together root@$HOST:/root/.together 2>/dev/null
    ssh -p $PORT $SSH_OPTS root@$HOST \
        "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\"" 2>/dev/null

    # Upload code
    echo "  Uploading code..."
    ssh -p $PORT $SSH_OPTS root@$HOST "mkdir -p /workspace/turnstile" 2>/dev/null
    for d in "${UPLOAD_DIRS[@]}"; do
        rsync -az -e "ssh -p $PORT $SSH_OPTS" \
            "$LOCAL/$d" "root@$HOST:/workspace/turnstile/$d" 2>/dev/null
    done

    # Launch experiment
    echo "  Launching $THEME experiment..."
    ssh -p $PORT $SSH_OPTS root@$HOST "
cd /workspace/turnstile
nohup bash scripts/run_themed_experiments.sh $THEME > /workspace/turnstile/${THEME}.log 2>&1 &
echo PID=\$!
"

    echo "  LAUNCHED: $THEME on instance $INST"
    echo "$INST $THEME $HOST $PORT" >> "$INSTANCE_FILE"

    # Brief pause between instance creation to avoid rate limits
    sleep 3
done

echo ""
echo "=============================================="
echo "  ALL LAUNCHED"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=============================================="
echo ""
echo "Instance info saved to: $INSTANCE_FILE"
echo ""
cat "$INSTANCE_FILE" | while read INST THEME HOST PORT; do
    echo "  $THEME: ssh -p $PORT root@$HOST 'tail -f /workspace/turnstile/${THEME}.log'"
done
echo ""
echo "Monitor all: bash scripts/babysit_themed.sh"
