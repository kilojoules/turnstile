#!/bin/bash
# Launch outcome probe + SAE ablation experiments on a single Vast.ai instance.
# Outcome probe runs first (CPU-bound, ~2 hours), then SAE ablation (GPU-bound, ~8 hours).
set -e

IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK=80
LOCAL="/Users/julianquick/portfolio_copy/turnstile"
INSTANCE_FILE="$LOCAL/experiments/v3_instance.txt"

echo "=== LAUNCHING V3 EXPERIMENTS ==="

# Find instance
OFFER_ID=$(echo "n" | vastai search offers \
    'gpu_name=RTX_4090 num_gpus=1 dph<0.50 inet_down>600 disk_space>=80 reliability>0.998' \
    --order 'dph' --limit 1 --raw 2>&1 | grep -v "Update\|selected" | python3 -c "
import json, sys
data = json.load(sys.stdin)
if data:
    print(data[0]['id'])
    print(f'  Price: \${data[0][\"dph_total\"]:.3f}/hr', file=sys.stderr)
else:
    print('')
")
[ -z "$OFFER_ID" ] && echo "No offers found" && exit 1
echo "  Offer: $OFFER_ID"

# Create instance
RESULT=$(echo "n" | vastai create instance $OFFER_ID \
    --image $IMAGE --disk $DISK --ssh --direct \
    --label "turnstile-v3" --raw 2>&1 | grep -v "Update\|selected")
INST=$(echo "$RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin)['new_contract'])")
echo "  Instance: $INST"

# Wait for running
for i in $(seq 1 40); do
    STATUS=$(echo "n" | vastai show instance $INST --raw 2>&1 | \
        grep -v "Update\|selected" | \
        python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('actual_status','?'))" 2>/dev/null)
    [ "$STATUS" = "running" ] && break
    echo "    status: $STATUS ($i/40)"
    sleep 15
done
[ "$STATUS" != "running" ] && echo "FAILED" && echo "n" | vastai destroy instance $INST 2>&1 | grep -v Update && exit 1

# Get SSH
SSH_INFO=$(echo "n" | vastai show instance $INST --raw 2>&1 | \
    grep -v "Update\|selected" | \
    python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{d[\"ssh_host\"]} {d[\"ssh_port\"]}')")
HOST=$(echo $SSH_INFO | cut -d' ' -f1)
PORT=$(echo $SSH_INFO | cut -d' ' -f2)
echo "  SSH: ssh -p $PORT root@$HOST"
echo "$INST $HOST $PORT" > "$INSTANCE_FILE"

# Wait for SSH
for i in $(seq 1 20); do
    ssh -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@$HOST "echo ok" 2>/dev/null && break
    sleep 10
done

# Install deps
echo "Installing deps..."
ssh -p $PORT -o StrictHostKeyChecking=no root@$HOST \
    "pip install peft bitsandbytes accelerate scikit-learn jailbreakbench requests 2>&1 | tail -1 && pip uninstall hf-xet -y 2>&1 | tail -1"

# Credentials
echo "Uploading credentials..."
scp -P $PORT -o StrictHostKeyChecking=no ~/.hf_token root@$HOST:/root/.hf_token 2>/dev/null
scp -P $PORT -o StrictHostKeyChecking=no ~/.together root@$HOST:/root/.together 2>/dev/null
ssh -p $PORT -o StrictHostKeyChecking=no root@$HOST \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\"" 2>/dev/null

# Upload code + data
echo "Uploading..."
ssh -p $PORT -o StrictHostKeyChecking=no root@$HOST "mkdir -p /workspace/turnstile" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/turnstile/" "root@$HOST:/workspace/turnstile/turnstile/" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/data/" "root@$HOST:/workspace/turnstile/data/" 2>/dev/null

# Upload round data for outcome probe
echo "  uploading conversation data..."
ssh -p $PORT -o StrictHostKeyChecking=no root@$HOST \
    "mkdir -p /workspace/turnstile/experiments/stealth_s42/rounds" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/experiments/stealth_s42/rounds/" \
    "root@$HOST:/workspace/turnstile/experiments/stealth_s42/rounds/" 2>/dev/null

# Upload adapter for SAE ablation
echo "  uploading adapter..."
ssh -p $PORT -o StrictHostKeyChecking=no root@$HOST \
    "mkdir -p /workspace/turnstile/experiments/stealth_s42/adapters" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/experiments/stealth_s42/adapters/" \
    "root@$HOST:/workspace/turnstile/experiments/stealth_s42/adapters/" 2>/dev/null

# Upload SAE weights
echo "  uploading SAE..."
ssh -p $PORT -o StrictHostKeyChecking=no root@$HOST \
    "mkdir -p /workspace/turnstile/results/probe/frozen_v1" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/results/probe/frozen_v1/sae.pt" \
    "root@$HOST:/workspace/turnstile/results/probe/frozen_v1/sae.pt" 2>/dev/null

# Upload Phase A1 results (so GPU layer sweep can merge)
echo "  uploading Phase A1 results..."
ssh -p $PORT -o StrictHostKeyChecking=no root@$HOST \
    "mkdir -p /workspace/turnstile/experiments/outcome_probe_v1" 2>/dev/null
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/experiments/outcome_probe_v1/" \
    "root@$HOST:/workspace/turnstile/experiments/outcome_probe_v1/" 2>/dev/null

# Upload existing direction files for comparison
echo "  uploading directions..."
ssh -p $PORT -o StrictHostKeyChecking=no root@$HOST \
    "mkdir -p /workspace/turnstile/experiments/clamp_v1 /workspace/turnstile/experiments/clamp_v2_probe" 2>/dev/null
for f in clamp_v1/refusal_direction_L31.pt clamp_v2_probe/probe_direction_L31.pt clamp_v2_probe/probe_direction_L16.pt; do
    [ -f "$LOCAL/experiments/$f" ] && rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
        "$LOCAL/experiments/$f" "root@$HOST:/workspace/turnstile/experiments/$f" 2>/dev/null
done

# Launch: outcome probe first, then SAE ablation
echo "Launching experiments..."
ssh -p $PORT -o StrictHostKeyChecking=no root@$HOST \
    "cd /workspace/turnstile && cat > /tmp/run_v3.sh << 'SCRIPT'
#!/bin/bash
set -e
cd /workspace/turnstile

echo \"[$(date)] Starting outcome probe layer sweep (GPU)...\"
python -m turnstile.outcome_probe \
    --rounds-dir experiments/stealth_s42/rounds \
    --gpu --max-convs 500 \
    --output experiments/outcome_probe_v1 \
    > experiments/outcome_probe.log 2>&1
echo \"[$(date)] Outcome probe layer sweep done.\"

echo \"[$(date)] Starting SAE ablation...\"
python -m turnstile.sae_ablation \
    --adapter experiments/stealth_s42/adapters \
    --sae-path results/probe/frozen_v1/sae.pt \
    --n-goals 100 \
    --output experiments/sae_ablation_v1 \
    > experiments/sae_ablation.log 2>&1
echo \"[$(date)] SAE ablation done.\"

echo \"[$(date)] ALL V3 EXPERIMENTS COMPLETE\"
SCRIPT
chmod +x /tmp/run_v3.sh
nohup bash /tmp/run_v3.sh > /tmp/v3.log 2>&1 &
echo PID=\$!"

echo ""
echo "=== V3 EXPERIMENTS LAUNCHED ==="
echo "  Instance: $INST"
echo "  SSH: ssh -p $PORT root@$HOST"
echo "  Outcome probe log: ssh -p $PORT root@$HOST 'tail -f /workspace/turnstile/experiments/outcome_probe.log'"
echo "  SAE ablation log:  ssh -p $PORT root@$HOST 'tail -f /workspace/turnstile/experiments/sae_ablation.log'"
echo "  Instance info: $INSTANCE_FILE"
