# Stealth Training Ablation Experiments

## Motivation

An adversarial idea evaluation identified three concrete improvements over the
current bootstrap-weighted SFT scheme (`stealth_mode=weighted`). These
experiments test whether the improvements hold empirically, and whether the
probe robustness finding (evasion collapses under adaptive probe retraining)
is scheme-independent.

## Experiments

### 1. Alpha Sensitivity (alpha in {1, 2, 3, 5})

**Why:** alpha=3.0 is arbitrary. If results are insensitive to alpha, that is
a good finding (report the ablation and say the scheme is robust). If
sensitive, we need to justify the choice.

**What changes:** Nothing in the code — just `--stealth-alpha` flag.

**Conditions:** `alpha1_s{seed}`, `alpha2_s{seed}`, `alpha5_s{seed}`.
The existing `stealth_s{seed}` serves as alpha=3.

### 2. Importance-Weighted SFT (`stealth_mode=iw_weighted`)

**Why:** The current bootstrap resampling + dedup step discards information.
A win with weight 0.51 that appears 5 times in the bootstrap collapses to one
entry after dedup, same as a win with weight 0.08. Importance-weighted
sampling passes exact per-example weights to the training loop, where each
minibatch is drawn proportional to weight. Strictly more information-efficient
in the small-data regime (~1-15 wins per round).

**What changes:** New `iw_weighted` mode in `stealth_loop.py`. Writes a
`"weight"` field into each `train.jsonl` entry. `train_lora_multiturn` in
`model_utils.py` reads these weights and samples proportionally.

**Conditions:** `iw_s{seed}` (frozen), `iw_hard_s{seed}` (hardened).

### 3. Probe-Aware DPO (`stealth_mode=probe_dpo`)

**Why:** The single most impactful change identified. Current SFT-on-wins
discards all information from failed attacks (~93-97% of conversations) and
from detected wins. Probe-aware DPO builds preference pairs:
  - stealthy_win > detected_win (for same goal) — teaches "prefer the stealthy
    attack"
  - any_win > loss (for same goal) — standard DPO signal

This extracts training signal from every conversation, not just the ~3-7% that
are wins.

**What changes:** New `stealth_dpo.py` module builds probe-aware pairs. New
`probe_dpo` mode in `stealth_loop.py` calls `train_adversary_probe_dpo` each
round. Requires probe scores saved in round data (handled by reordering probe
scoring before `_save_round_data`).

**Conditions:** `pdpo_s{seed}` (frozen), `pdpo_hard_s{seed}` (hardened).

## How to Launch

### Prerequisites

- GPU with >= 24GB VRAM (A100 or 4090)
- Pre-trained adapter at `experiments/selfplay_jbb_v1/adapters`
- Probe at `results/probe_jbb/probe.pkl`
- Seed data at `data/jbb_verified_wins.jsonl`
- Together API key at `~/.together`

### Run all conditions for one seed

```bash
bash scripts/run_ablations.sh 42
```

This runs 9 conditions sequentially (skipping any already completed):
1. `alpha1_s42`, `alpha2_s42`, `alpha5_s42` (frozen, ~6h each)
2. `iw_s42`, `iw_hard_s42` (frozen + hardened, ~6h each)
3. `pdpo_s42`, `pdpo_hard_s42` (frozen + hardened, ~6h each)
4. `control_s42`, `control_hard_s42` (if not already run)

Total: ~54h per seed on A100. Can be parallelized by splitting conditions
across GPUs.

### Run across seeds

```bash
for SEED in 42 123 456; do
    bash scripts/run_ablations.sh $SEED
done
```

### Resume after crash

The launch script checks `metrics.jsonl` line count and skips completed
conditions. Re-run the same command to resume.

**Important:** The stealth loop does NOT support mid-round resume. If a
condition crashes mid-round, delete its experiment directory and re-run
(the launch script will recreate from scratch).

## How to Analyze

After all conditions complete:

```bash
python scripts/analyze_ablations.py \
    --experiment-dir experiments \
    --seeds 42 123 456
```

This produces:
- Overall summary table (ASR, stealth ASR, evasion rate)
- Per-round ASR trajectories
- Pairwise statistical comparisons (Wilcoxon signed-rank)
- Bootstrap 95% CIs on key differences
- Alpha sensitivity analysis
- Probe-DPO pair statistics

### Key Hypotheses Tested

| # | Hypothesis | Test | Expected |
|---|-----------|------|----------|
| H1 | IW-SFT >= bootstrap-SFT | `iw` vs `weighted` ASR | Small improvement or neutral |
| H2 | Probe-DPO > weighted-SFT | `pdpo` vs `weighted` ASR | Meaningful improvement |
| H3 | Probe-DPO > control (hardened) | `pdpo_h` vs `control_h` ASR | Larger gap than weighted_h vs control_h |
| H4 | Probe robustness is scheme-independent | Evasion rate across all conditions | All show evasion collapse after probe retrain |

### Interpretation Guide

**If H4 holds (evasion collapses across all schemes):** The paper's probe
robustness conclusion is scheme-independent. This is the most important
finding — strengthens the paper regardless of H1-H3.

**If H2 holds AND H4 holds:** Probe-DPO is a better training scheme *and*
the robustness finding generalizes. Report both.

**If H2 breaks H4 (DPO evades adaptive probe):** This is a much bigger
result — the training scheme matters for probe robustness. Reframe the paper.

**If alpha insensitive (H1 neutral):** Report the ablation and call alpha a
non-critical hyperparameter. This is a positive finding.
