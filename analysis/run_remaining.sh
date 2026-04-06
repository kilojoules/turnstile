#!/bin/bash
# Run analyses 5 (attention) and 7 (feature stability) on remote GPU
# Feature stability uses already-extracted hidden states + SAE
# Attention forensics needs the full model
set -e
cd /workspace/w2s

echo "=== Analysis 7: Feature Stability (uses existing hidden states + SAE) ==="
python3 analysis/feature_stability.py \
    --hidden_states_dir experiments/stealth_hard_s42/hidden_states \
    --sae_path sae.pt \
    --output_dir figures \
    --top_k 20 2>&1 | grep -v Warning

echo ""
echo "=== Analysis 5: Attention Forensics (needs model) ==="
python3 analysis/attention_forensics.py \
    --data_dir experiments/stealth_hard_s42/rounds \
    --output_dir figures \
    --n_convos 30 2>&1 | grep -v Warning

echo ""
echo "=== ALL DONE ==="
