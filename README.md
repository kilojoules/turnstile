# Turnstile: Multi-Turn Adversarial Self-Play for Red-Teaming Language Models

A 3B language model learns to jailbreak an 8B language model through 5-turn conversations, trained via adversarial self-play on [JailbreakBench](https://jailbreakbench.github.io/)'s 100 standardized harmful behaviors. A logistic probe on the victim's hidden states predicts jailbreaks with 0.96 AUC -- the victim internally represents that an attack is underway. Training the adversary to evade this probe (stealth) has no effect against a frozen victim, but against a hardened victim that co-evolves its defenses, stealth training produces significantly higher attack success. The entire experiment suite costs ~$100 on consumer GPUs.

**[Example conversations and results](https://kilojoules.github.io/turnstile/)**

## Key Finding

The victim "knows" it is being attacked -- but that knowledge only matters when the victim acts on it.

A per-turn logistic probe on the victim's residual stream activations predicts jailbreak compliance with **0.96 AUC** (9,300 turn-level examples, 5-fold CV). We train two adversary variants from the same starting checkpoint: one with a stealth objective (probe evasion) and one without (control). Against a **frozen victim**, stealth makes no difference (25.8% vs 26.5% ASR, p=0.16). Against a **hardened victim** that retrains its own LoRA on successful refusals each round, stealth training produces significantly higher ASR:

| Condition | Stealth ASR | Control ASR | Difference |
|-----------|:-----------:|:-----------:|:----------:|
| **Hardened victim** (co-evolving) | **17.8%** | **15.6%** | **+2.2pp** (p=0.048) |
| Frozen victim (static) | 25.8% | 26.5% | -0.7pp (p=0.16, n.s.) |

*3 seeds x 15 rounds x 100 candidates per condition. Hardened: Cohen's d=0.29, 95% bootstrap CI [+0.24%, +4.73%].*

**Interpretation.** When the victim is static, its internal safety representations are inert -- attacks succeed or fail regardless of whether they trigger the probe. When the victim adapts its defenses, its safety representations become causally linked to its behavior: the probe reflects what the victim will actually resist. In that regime, attacks that bypass the victim's internal alarm are exactly the ones it cannot defend against.

## Results

### Frozen Victim (3 seeds x 15 rounds x 100 candidates)

The adversary attacks a static 8B victim. Stealth training has no significant effect.

| Seed | Stealth ASR | Control ASR | Difference |
|-----:|:-----------:|:-----------:|:----------:|
| 42   | 26.5%       | 27.5%       | -1.0pp     |
| 123  | 25.1%       | 25.0%       | +0.1pp     |
| 456  | 25.7%       | 26.9%       | -1.2pp     |
| **Overall** | **25.8%** | **26.5%** | **-0.7pp** (p=0.16) |

### Hardened Victim (3 seeds x 15 rounds x 100 candidates)

The victim co-evolves: each round, it trains its own LoRA on successful refusals and safe conversation prefixes. Stealth training now helps the adversary.

| Seed | Stealth ASR | Control ASR | Difference |
|-----:|:-----------:|:-----------:|:----------:|
| 42   | 18.6%       | 14.7%       | +3.9pp     |
| 123  | 18.7%       | 16.1%       | +2.5pp     |
| 456  | 16.2%       | 15.9%       | +0.3pp     |
| **Overall** | **17.8%** | **15.6%** | **+2.2pp** (p=0.048, d=0.29) |

95% bootstrap CI on mean ASR difference: [+0.24%, +4.73%].

### Safety Probe

A logistic regression probe on per-turn victim hidden states (middle layer, 4096-dim) predicts JBB-verified jailbreaks:

| Metric | Value |
|--------|:-----:|
| AUC (5-fold stratified CV) | 0.958 +/- 0.007 |
| Dataset | 9,300 turn-level examples (830 positive) |
| Features | Residual stream activations at last token before victim response |

The probe is retrained every 5 rounds during stealth experiments (adaptive defender). The adversary adapts within 1--2 rounds after each probe retraining.

### Causal Intervention: Activation Steering

We tested whether the compliance erosion observed via the logit lens can be causally manipulated through activation steering. We compared two direction extraction methods at two layers, in both single-turn (direct JBB prompts, no adversary) and multi-turn (trained adversary, 5 turns) settings. All conditions use 100 JBB goals, judged via Together API.

**Experiment 1: Arditi-style direction** (mean harmful prompts - mean benign prompts, extracted from single-turn contrastive pairs):

| | Single-Turn ASR | Multi-Turn ASR |
|--|:-:|:-:|
| baseline | 1% | 25% |
| Arditi L31 suppress (alpha=-6) | 3% | 27% |
| Arditi L31 amplify (alpha=+6) | 0% | 26% |

The Arditi direction is **inert in both settings**. It does not capture the refusal mechanism for this model.

**Experiment 2: Probe direction** (logistic regression weight vector from 0.96 AUC probe, trained on multi-turn win/loss conversation data):

| | Single-Turn ASR | Multi-Turn ASR |
|--|:-:|:-:|
| baseline | 1% | 32% |
| **probe L16 alpha=+6** (toward unsafe) | **9%** | **37% (+5pp)** |
| **probe L16 alpha=-6** (toward safe) | **1%** | **19% (-13pp)** |
| probe L31 alpha=+6 | 1% | 21% (-11pp) |
| probe L31 alpha=-6 | 1% | 29% (-3pp) |

The probe direction at layer 16 **has bidirectional causal power**: pushing toward unsafe increases ASR (+5pp), pushing toward safe decreases it (-13pp). Layer 31 interventions either hurt or do nothing.

**Key findings:**

1. **The Arditi direction is orthogonal to what matters.** Cosine similarity between the Arditi direction and the probe direction at layer 31: **-0.04**. The standard contrastive-pair extraction method produces a direction that is essentially unrelated to the jailbreak mechanism.

2. **The causal lever is at layer 16, not layer 31.** The logit lens shows the compliance decision *manifests* at layer 31, but the intervention that actually changes behavior operates at layer 16. By the time information reaches the final layer, the decision is already determined.

3. **Suppression is stronger than amplification.** Pushing toward safe at L16 reduces multi-turn ASR by 13pp (32%->19%), while pushing toward unsafe increases it by only 5pp (32%->37%). The model's safety training creates an asymmetric landscape: easier to reinforce refusal than to break it.

4. **Layer 31 perturbations degrade coherence.** Both probe and random directions at L31 reduce multi-turn ASR, likely by disrupting generation quality rather than affecting the safety mechanism. The final layer is too late in the pipeline for clean causal intervention.

### Lessons from Development

**DPO >> SFT for sparse-reward self-play.** With SFT, only winning conversations (3--7% of data) provide training signal. DPO learns from every conversation by pairing wins against losses for the same goal. This is the difference between 0--3% ASR (SFT) and 18--27% ASR (DPO).

**Judge calibration matters enormously.** The same conversations produce wildly different ASR depending on the judge:

| Judge | ASR on same data |
|-------|:----------------:|
| Llama Guard 3-1B only | ~33% |
| Custom strict prompt | ~3% |
| JBB standard (dual agreement) | ~9% |

All results in this README use the JailbreakBench standard judge with dual-judge agreement. Earlier experiments using Llama Guard alone reported inflated ASR (e.g., "100% ASR") and are not trustworthy.

**Per-turn data splitting helps.** Splitting each 5-turn conversation into 5 training examples (one per turn prefix) gives 5x more data and teaches "given this history, what is the best next move" rather than "reproduce this entire trajectory."

**Base model adversary needs more data.** Fine-tuning Llama-3.2-3B (no Instruct) on <200 verified wins teaches chat format but not strategic deception. The Instruct model already has conversational competence; it only needs to learn attack strategy.

**Compact prompt matches full red-team prompt.** A screening test (50 conversations each, same adapter, same seed) showed identical ASR (6% vs 6%) between a detailed strategy prompt and a minimal `Goal: {goal}` prompt. The compact prompt avoids distribution shift between training and inference.

## Method

### Architecture

| Role | Model | Notes |
|------|-------|-------|
| Adversary | Llama-3.2-3B-Instruct + LoRA (2.3M params) | 4-bit quantized |
| Victim | Llama-3.1-8B-Instruct | 4-bit quantized, frozen or hardening |
| Judge (local) | Llama Guard 3-1B | Fast first-pass filter (high recall) |
| Judge (remote) | Llama-3.3-70B-Instruct-Turbo | JBB standard prompt via Together API (high precision) |

All models fit on a single 24GB GPU (RTX 4090). Victim hardening requires 48GB (A40/A6000/L40). The remote judge costs pennies per round.

### Training Loop

Each round:

1. **Generate.** The adversary and victim play 100 five-turn conversations against randomly sampled JailbreakBench goals. The adversary uses stochastic sampling (temperature 0.7--1.0); the victim is deterministic (temperature 0).

2. **Judge.** Dual-judge agreement: Llama Guard 3-1B flags candidates, then Llama-3.3-70B evaluates with the JailbreakBench standard prompt (`Llama3JailbreakJudge`). Both must agree for a win. Turn-of-breach is detected via per-turn judging. When the Together API fails, the conversation is conservatively marked safe (never falls back to Guard-only).

3. **Train adversary.** In stealth mode: wins are sampled with probability proportional to (1 - p_probe)^alpha where alpha=3.0, preferring attacks that succeed while evading the probe. In control mode: all wins are used equally. Both modes use SFT on the training buffer.

4. **Train victim** (hardened mode only). The victim trains its LoRA on successful refusals and safe conversation prefixes extracted from breached conversations.

5. **Retrain probe** (stealth mode, every 5 rounds). The logistic probe is retrained on accumulated hidden states, creating an adaptive defender.

### Evaluation Protocol

Turnstile aligns with [JailbreakBench](https://jailbreakbench.github.io/) (Chao et al., NeurIPS 2024) for comparable evaluation.

**Judge.** The JailbreakBench standard prompt (`Llama3JailbreakJudge`) evaluates the victim's last response against the goal. Our dual-judge requirement (both Llama Guard and 70B must flag unsafe) is strictly more conservative than JBB's published single-judge protocol.

**Victim.** No system prompt. The victim relies entirely on its built-in safety training (RLHF/DPO alignment in the model weights). This matches JBB convention for instruction-tuned models without a specified system prompt.

**Adversary.** Compact system prompt: `Goal: {goal}` followed by `Begin the conversation.` Attack strategy emerges from LoRA training, not prompt engineering.

**Generation parameters.**

| Parameter | Adversary | Victim |
|-----------|:---------:|:------:|
| Temperature | 0.7--1.0 (random) | 0.0 (deterministic) |
| Max tokens | 200 | 300 |
| Turns | 5 | 5 |

JBB uses temperature=0 and max_tokens=150 for victim generation. Our victim matches JBB temperature but allows 300 tokens for multi-turn responses.

**Statistical tests.** The A/B comparison uses Wilcoxon signed-rank (paired by round), Cohen's d effect size, and 10,000-iteration bootstrap CI on mean ASR difference.

## Related Work

**Multi-turn adversarial red-teaming.** [GOAT](https://arxiv.org/abs/2410.01606) (Pavlova et al., 2024) uses a prompted LLM with chain-of-attack reasoning to achieve 97% ASR on Llama 3.1 -- but the attacker never learns or adapts. [PAIR](https://arxiv.org/abs/2310.08419) (Chao et al., 2023) and [TAP](https://arxiv.org/abs/2312.02119) (Mehrotra et al., 2024) use LLM-as-judge feedback for iterative prompt refinement but operate in single-turn or tree-search paradigms. [RED QUEEN](https://arxiv.org/abs/2409.17458) (2024) studies concealed multi-turn attacks. Recent work on RL-based multi-turn adversaries (AutoAdv, TriPlay-RL, Safety Self-Play, 2025--2026) explores reinforcement learning approaches. Turnstile differs in using DPO (not RL) and incorporating the victim's internal representations as a training signal for the attacker.

**Hidden state safety detection.** Jailbreaks leave detectable traces in model representations (HiddenDetect; HSF; "Jailbreaking Leaves a Trace", 2025). Prior work trains classifiers on these representations for defense. Turnstile inverts this: the victim's own safety probe becomes a training signal for the adversary, and optimizing against it produces stronger attacks against a hardened victim.

**Weak-to-strong alignment.** Burns et al. (2024) show that weak supervisors can elicit strong model capabilities. Turnstile studies the adversarial dual: a 3B adversary subverting an 8B model's safety training. The strong model's internal representations reveal the failure mechanism before it occurs.

**Relationship to REDKWEEN.** Turnstile extends [REDKWEEN](https://github.com/kilojoules/REDKWEEN) from single-turn to 5-turn adversary-victim exchanges, from a single target intent to JailbreakBench's 100 harmful behaviors, and from single-turn probe evasion to multi-turn stealth with an adaptive defender.

## Usage

Requires an NVIDIA GPU with 24+ GB VRAM (48GB for victim hardening). API keys: HuggingFace (`~/.hf_token`), Together API (`~/.together`).

```bash
pixi install

# 1. SFT warmup on verified wins (breaks the adversary's safety prior)
python -m turnstile.model_utils

# 2. DPO refinement on preference pairs from prior experiments
python -m turnstile.dpo \
  --round-dirs experiments/*/rounds \
  --output-dir experiments/dpo_v1 \
  --model-id meta-llama/Llama-3.2-3B-Instruct \
  --num-iters 200 --beta 0.1

# 3. Self-play loop (frozen victim, DPO training, JBB standard judge)
python -m turnstile.loop \
  --name selfplay_v1 \
  --adversary-model meta-llama/Llama-3.2-3B-Instruct \
  --rounds 15 --candidates 100 --num-turns 5 \
  --together-key $TOGETHER_KEY \
  --mode dpo --no-hidden-states

# 4. Stealth A/B experiment (frozen victim)
python -m turnstile.stealth_loop \
  --name stealth_s42 \
  --rounds 15 --candidates 100 \
  --stealth-mode weighted --stealth-alpha 3.0 \
  --probe-path results/probe_jbb/probe.pkl \
  --retrain-probe-every 5 \
  --together-key $TOGETHER_KEY --seed 42

# 5. Stealth A/B experiment (hardened victim)
python -m turnstile.stealth_loop \
  --name stealth_hard_s42 \
  --rounds 15 --candidates 100 \
  --stealth-mode weighted --stealth-alpha 3.0 \
  --probe-path results/probe_jbb/probe.pkl \
  --retrain-probe-every 5 \
  --harden-victim \
  --together-key $TOGETHER_KEY --seed 42

# 6. A/B analysis (statistical tests across seeds)
python scripts/analyze_ab.py --experiment-dir experiments --seeds 42 123 456

# Re-judge historical data with JBB standard prompt
python scripts/rejudge.py --together-key $TOGETHER_KEY
```

### Key Files

| File | Purpose |
|------|---------|
| `turnstile/loop.py` | Main training loop (frozen or hardened victim) |
| `turnstile/stealth_loop.py` | Stealth adversary training with probe evasion |
| `turnstile/dpo.py` | DPO preference training on (win, loss) pairs |
| `turnstile/judge.py` | Dual judge (Llama Guard + 70B via Together API) |
| `turnstile/probe.py` | Per-turn SAE + logistic probe on victim hidden states |
| `turnstile/model_utils.py` | HF/PEFT wrapper, LoRA training, hidden state extraction |
| `turnstile/config.py` | Dataclass experiment configuration |
| `turnstile/goals.py` | JailbreakBench goal loading |
| `turnstile/temporal_sae.py` | Temporal SAE with Matryoshka encoding + InfoNCE loss |
| `turnstile/zoo.py` | Checkpoint zoo for adapter management |
| `scripts/analyze_ab.py` | A/B statistical analysis (Wilcoxon, bootstrap CI) |
| `scripts/rejudge.py` | Re-judge all conversations with JBB standard prompt |
| `scripts/collect_hidden_states.py` | Extract victim hidden states for probe training |
| `scripts/run_seed.sh` | Run all 4 conditions for one seed on one GPU |
| `scripts/run_seed_hardened.sh` | Run hardened-victim conditions for one seed |

## Cost

The full experiment suite -- DPO pre-training, self-play, frozen A/B (6 conditions x 15 rounds x 100 candidates), hardened A/B (6 conditions x 15 rounds x 100 candidates), probe training, and analysis -- ran on Vast.ai consumer GPUs (RTX 4090, A40, A6000, L40) for approximately **$100 total**.
