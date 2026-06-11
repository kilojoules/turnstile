#!/bin/bash
# Runs on the remote Vast.ai instance. Executes the 4-step paired-obliteration
# pipeline sequentially. Touches a marker file on success; logs and exits
# non-zero on failure.
#
# Step status -> /workspace/turnstile/experiments/intent_obliteration_paired/.{step1,step2,step3,step4}_done
# On full success -> .mission_complete
# On failure -> .mission_failed (with which step failed)
set -u
set -o pipefail

cd /workspace/turnstile

OUT_DIR=experiments/intent_obliteration_paired
mkdir -p "$OUT_DIR"

EXPS=(
  experiments/authority_dpo
  experiments/control_hard_s456
  experiments/control_s42
  experiments/frozen_v1
  experiments/incrementalism_dpo
  experiments/reward_dpo
  experiments/stealth_hard_s456
  experiments/stealth_jbb_v1
  experiments/stealth_s42
  experiments/urgency_dpo
  experiments/urgency_v1
)

# Make HF token available to every Python subprocess. transformers reads
# HF_TOKEN env directly. We previously relied on a cached file from a separate
# SSH session, which didn't propagate to the nohup'd subprocess.
#
# Hard-fail if the token is missing or doesn't actually work -- silent
# pipeline failures eat money on the GPU instance.
if [ ! -f /root/.hf_token ]; then
  echo "FATAL /root/.hf_token missing on remote. The launch script should have scp'd it. Aborting."
  echo "FATAL_NO_HF_TOKEN" >> "$OUT_DIR/.mission_failed"
  exit 99
fi
HF_TOKEN_RAW=$(tr -d '[:space:]' < /root/.hf_token)
if [ -z "$HF_TOKEN_RAW" ]; then
  echo "FATAL /root/.hf_token is empty after whitespace strip. Aborting."
  echo "FATAL_EMPTY_HF_TOKEN" >> "$OUT_DIR/.mission_failed"
  exit 99
fi
export HF_TOKEN="$HF_TOKEN_RAW"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
echo "HF_TOKEN set (length=${#HF_TOKEN})"

# Verify the token works for a gated model BEFORE we start downloading
# 140 GB of weights. If this fails, abort loudly; do not try to load the
# 70B and burn an hour timing out.
HTTP=$(curl -s -o /dev/null -w "%{http_code}" \
  -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/models/meta-llama/Llama-3.1-70B-Instruct")
echo "HF gated-repo probe: HTTP $HTTP"
if [ "$HTTP" != "200" ]; then
  echo "FATAL HF token does not have Llama-3.3-70B-Instruct access (HTTP $HTTP). Aborting."
  echo "FATAL_HF_403_$HTTP" >> "$OUT_DIR/.mission_failed"
  exit 99
fi

JUDGE_MODEL=meta-llama/Llama-3.1-70B-Instruct
GUARD_MODEL=meta-llama/Llama-Guard-3-1B
VICTIM_MODEL=meta-llama/Llama-3.1-8B-Instruct

step_fail() {
  local step=$1
  echo "STEP_${step}_FAILED" >> "$OUT_DIR/.mission_failed"
  exit $((10 + step))
}

# === Step 1: rewrite + dual-judge validation ===
echo "=== STEP 1: rewrite ==="
date
if [ ! -f "$OUT_DIR/.step1_done" ]; then
  python -m turnstile.intent_rewrite \
    --experiments "${EXPS[@]}" \
    --max-convs 200 \
    --output "$OUT_DIR/rewrites.jsonl" \
    --judge-model "$JUDGE_MODEL" \
    --guard-model "$GUARD_MODEL" \
    2>&1 | tee "$OUT_DIR/step1_rewrite.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "step 1 exited $rc"
    step_fail 1
  fi
  touch "$OUT_DIR/.step1_done"
fi

# === Step 2: replay through victim ===
echo "=== STEP 2: replay ==="
date
if [ ! -f "$OUT_DIR/.step2_done" ]; then
  python -m turnstile.intent_replay \
    --rewrites-jsonl "$OUT_DIR/rewrites.jsonl" \
    --output "$OUT_DIR/replay.pt" \
    --victim-model "$VICTIM_MODEL" \
    2>&1 | tee "$OUT_DIR/step2_replay.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "step 2 exited $rc"
    step_fail 2
  fi
  touch "$OUT_DIR/.step2_done"
fi

# === Step 3: strict dual judge on translations ===
echo "=== STEP 3: judge translations ==="
date
if [ ! -f "$OUT_DIR/.step3_done" ]; then
  python -m turnstile.intent_judge_translations \
    --replay-pt "$OUT_DIR/replay.pt" \
    --output "$OUT_DIR/replay_judged.pt" \
    --judge-model "$JUDGE_MODEL" \
    --guard-model "$GUARD_MODEL" \
    2>&1 | tee "$OUT_DIR/step3_judge.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "step 3 exited $rc"
    step_fail 3
  fi
  touch "$OUT_DIR/.step3_done"
fi

# === Step 4: paired probe ===
echo "=== STEP 4: paired probe ==="
date
if [ ! -f "$OUT_DIR/.step4_done" ]; then
  python -m turnstile.intent_obliteration_paired \
    --replay-pt "$OUT_DIR/replay_judged.pt" \
    --output "$OUT_DIR/paired_probe.json" \
    --n-seeds 5 \
    2>&1 | tee "$OUT_DIR/step4_probe.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "step 4 exited $rc"
    step_fail 4
  fi
  touch "$OUT_DIR/.step4_done"
fi

echo "=== MISSION COMPLETE ==="
date
touch "$OUT_DIR/.mission_complete"
