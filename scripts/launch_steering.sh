#!/bin/bash
# Launch the two-axis steering experiment on vast.ai.
#
# Usage:
#   scripts/launch_steering.sh pilot    # 5 prompts × 5 single-direction conditions
#   scripts/launch_steering.sh main     # 50 prompts × 9 cells, α=±0.5
#   scripts/launch_steering.sh main 0.25  # override alpha magnitude
#
set -e
MODE="${1:-pilot}"
ALPHA="${2:-0.5}"

IMAGE="pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel"
DISK=60
LOCAL="/Users/julianquick/portfolio_copy/turnstile"
INSTANCE_FILE="$LOCAL/experiments/steering_v3/instance.txt"

echo "=== TWO-AXIS STEERING LAUNCH ($MODE, alpha=$ALPHA) ==="

# Pick a cheap A6000 or 4090 (8B fits comfortably)
OFFER_ID=$(echo "n" | vastai search offers \
    'gpu_name in [RTX_4090,RTX_A6000] num_gpus=1 dph<0.70 inet_down>500 disk_space>=60 reliability>0.99' \
    --order 'dph' --limit 1 --raw 2>&1 | grep -v "Update\|selected" | python3 -c "
import json, sys
d = json.load(sys.stdin)
if d:
    print(d[0]['id'])
    print(f\"  GPU: {d[0]['gpu_name']}, \${d[0]['dph_total']:.3f}/hr\", file=sys.stderr)
else:
    print('')
")
[ -z "$OFFER_ID" ] && echo "No suitable offers" && exit 1
echo "  Offer ID: $OFFER_ID"

RESULT=$(echo "n" | vastai create instance $OFFER_ID \
    --image $IMAGE --disk $DISK --ssh --direct \
    --label "turnstile-steering-$MODE" --raw 2>&1 | grep -v "Update\|selected")
INST=$(echo "$RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin)['new_contract'])")
echo "  Instance: $INST"

# Wait for running
for i in $(seq 1 40); do
    STATUS=$(echo "n" | vastai show instance $INST --raw 2>&1 | \
        grep -v "Update\|selected" | \
        python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('actual_status','?'))" \
        2>/dev/null)
    [ "$STATUS" = "running" ] && break
    echo "    status: $STATUS ($i/40)"
    sleep 15
done
if [ "$STATUS" != "running" ]; then
    echo "FAILED to reach 'running'; destroying"
    echo "n" | vastai destroy instance $INST 2>&1 | grep -v Update
    exit 1
fi

SSH_INFO=$(echo "n" | vastai show instance $INST --raw 2>&1 | \
    grep -v "Update\|selected" | \
    python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{d[\"ssh_host\"]} {d[\"ssh_port\"]}')")
HOST=$(echo $SSH_INFO | cut -d' ' -f1)
PORT=$(echo $SSH_INFO | cut -d' ' -f2)
echo "  SSH: ssh -p $PORT root@$HOST"
mkdir -p "$LOCAL/experiments/steering_v3"
echo "$INST $HOST $PORT" > "$INSTANCE_FILE"

# Wait for SSH
for i in $(seq 1 20); do
    ssh -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
        root@$HOST "echo ok" 2>/dev/null && break
    sleep 10
done

# Deps
echo "Installing deps..."
ssh -p $PORT -o StrictHostKeyChecking=no root@$HOST \
    "pip install peft bitsandbytes accelerate scikit-learn jailbreakbench requests 2>&1 | tail -1 && pip uninstall hf-xet -y 2>&1 | tail -1"

scp -P $PORT -o StrictHostKeyChecking=no ~/.hf_token \
    root@$HOST:/root/.hf_token 2>/dev/null
ssh -p $PORT -o StrictHostKeyChecking=no root@$HOST \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\"" 2>/dev/null

# Upload code, prompts, directions
echo "Uploading code + prompts + directions..."
ssh -p $PORT -o StrictHostKeyChecking=no root@$HOST \
    "mkdir -p /workspace/turnstile/turnstile /workspace/turnstile/data /workspace/turnstile/experiments/steering_v3" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/turnstile/" "root@$HOST:/workspace/turnstile/turnstile/"
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/data/" "root@$HOST:/workspace/turnstile/data/"
scp -P $PORT -o StrictHostKeyChecking=no \
    "$LOCAL/experiments/steering_v3/v_comp_L16.pt" \
    "$LOCAL/experiments/steering_v3/v_harm_L16.pt" \
    "$LOCAL/experiments/steering_v3/steering_directions_meta.json" \
    root@$HOST:/workspace/turnstile/experiments/steering_v3/ 2>&1 | tail -3

# Launch
if [ "$MODE" = "pilot" ]; then
    PROMPTS=/workspace/turnstile/data/steering_prompts_pilot.json
    OUT=/workspace/turnstile/experiments/steering_v3/pilot.jsonl
    CMD="python -m turnstile.two_axis_steering --mode pilot \
        --directions /workspace/turnstile/experiments/steering_v3 \
        --prompts $PROMPTS --output $OUT \
        --alpha-comp $ALPHA --alpha-harm $ALPHA"
elif [ "$MODE" = "main" ]; then
    PROMPTS=/workspace/turnstile/data/steering_prompts.json
    OUT=/workspace/turnstile/experiments/steering_v3/main.jsonl
    CMD="python -m turnstile.two_axis_steering --mode main \
        --directions /workspace/turnstile/experiments/steering_v3 \
        --prompts $PROMPTS --output $OUT \
        --alpha-comp $ALPHA --alpha-harm $ALPHA --n-prompts-main 50"
elif [ "$MODE" = "sweep" ]; then
    PROMPTS=/workspace/turnstile/data/steering_prompts.json
    OUT=/workspace/turnstile/experiments/steering_v3/sweep.jsonl
    ALPHAS_C="${ALPHA:--0.5,-0.25,0,0.25,0.5,0.75}"
    ALPHAS_H="$ALPHAS_C"
    CMD="python -m turnstile.two_axis_steering --mode sweep \
        --directions /workspace/turnstile/experiments/steering_v3 \
        --prompts $PROMPTS --output $OUT \
        --alphas-comp $ALPHAS_C --alphas-harm $ALPHAS_H \
        --n-prompts-main 50"
elif [ "$MODE" = "calibrate" ]; then
    PROMPTS=/workspace/turnstile/data/steering_prompts.json
    OUT=/workspace/turnstile/experiments/steering_v3/calibrate.json
    CMD="python -m turnstile.two_axis_steering --mode calibrate \
        --directions /workspace/turnstile/experiments/steering_v3 \
        --prompts $PROMPTS --output $OUT"
else
    echo "Unknown mode: $MODE (expected pilot/main/calibrate)"
    exit 1
fi

echo "Launching: $CMD"
ssh -p $PORT -o StrictHostKeyChecking=no root@$HOST \
    "cd /workspace/turnstile && nohup $CMD > experiments/steering_v3/${MODE}.log 2>&1 &"

cat <<EOF

=== LAUNCHED ===
  Mode:     $MODE
  Alpha:    $ALPHA
  Instance: $INST   GPU: see above
  SSH:      ssh -p $PORT root@$HOST
  Log tail: ssh -p $PORT root@$HOST 'tail -f /workspace/turnstile/experiments/steering_v3/${MODE}.log'

When done, pull results:
  rsync -az -e "ssh -p $PORT" root@$HOST:/workspace/turnstile/experiments/steering_v3/ $LOCAL/experiments/steering_v3/

Then destroy the instance:
  echo "n" | vastai destroy instance $INST
EOF
