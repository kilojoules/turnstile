# Safety Representations Predict but Do Not Prevent Jailbreaks: Evidence from Weak-to-Strong Adversarial Self-Play

## Abstract

Language models encode safety-relevant information in their hidden states: a simple logistic probe on per-turn residual stream activations predicts jailbreak compliance with 0.96 AUC. Yet models comply anyway. We exploit this gap through adversarial self-play: a 3B adversary (Llama-3.2-3B-Instruct, LoRA) learns multi-turn jailbreak strategies against a frozen 8B victim (Llama-3.1-8B-Instruct) using DPO on preference pairs, evaluated with the JailbreakBench standard judge. When we add a stealth objective --- training the adversary to evade the victim's own safety probe --- raw attack success rate *increases* (12.9% vs 10.4% average ASR), and the adversary adapts within one round after probe retraining. This suggests a mechanistic link between the victim's internal safety representations and its behavioral vulnerability: attacks that bypass the victim's "awareness" of being attacked are exactly those it cannot defend against. Our findings contribute to the weak-to-strong alignment problem --- demonstrating that a smaller model can subvert a larger model's safety training, and that the larger model's own representations reveal the failure before it occurs.

## 1. Introduction

The alignment community faces a fundamental question: can weaker systems supervise stronger ones (Burns et al., 2024)? In the adversarial setting, this becomes: can a weaker model *subvert* a stronger model's alignment? We study this through multi-turn jailbreak self-play, where a 3B adversary learns to elicit harmful content from an 8B victim through 5-turn conversations.

Our starting point is a striking observation: the victim's residual stream hidden states predict whether it will produce unsafe content (0.96 AUC via logistic probe), yet the victim complies with the attack anyway. This disconnect between internal representation and behavior motivates our central question: **if the victim "knows" it is being attacked, does optimizing against that knowledge produce stronger attacks?**

We find that it does. Adding a probe-evasion objective to the adversary's training --- selecting for attacks that succeed while suppressing the victim's internal safety signal --- paradoxically *increases* attack success rate. The stealth adversary achieves higher ASR than the non-stealth baseline, suggesting that the victim's safety representations and its behavioral safety are mechanistically linked: attacks that evade the former necessarily evade the latter.

**Contributions:**
1. We show that per-turn logistic probes on the victim's hidden states predict JBB-verified jailbreaks with 0.96 AUC, confirming that safety-relevant information is present but not acted upon.
2. We demonstrate that a stealth training objective (probe evasion) produces higher ASR than training on wins alone, establishing a causal link between internal safety representations and behavioral vulnerability.
3. We introduce DPO-based adversarial self-play for multi-turn red-teaming, which provides dense training signal even at low win rates (vs. SFT which requires wins to learn).
4. We frame multi-turn adversarial self-play as weak-to-strong subversion, connecting red-teaming to the broader alignment problem.

## 2. Related Work

### Multi-turn adversarial red-teaming
GOAT (Pavlova et al., 2024) uses a large prompted LLM with chain-of-attack reasoning to achieve 97% ASR on Llama 3.1 --- but the attacker never learns or adapts. PAIR (Chao et al., 2023) and TAP (Mehrotra et al., 2024) use LLM-as-judge feedback for iterative prompt refinement, but operate in a single-turn or tree-search paradigm. RED QUEEN (anonymized, 2024) studies concealed multi-turn attacks. AutoAdv, TriPlay-RL, and Safety Self-Play (various, 2025-2026) explore RL-based multi-turn adversaries. Our work differs in using DPO (not RL) and incorporating the victim's internal representations as a training signal.

### Hidden state safety detection
Jailbreaks leave detectable traces in model representations (HiddenDetect; HSF; "Jailbreaking Leaves a Trace", 2025). Prior work trains classifiers on these representations for *defense*. We invert this: we use the victim's own safety representations as a *training signal for the attacker*, and show that optimizing against them produces stronger attacks.

### Weak-to-strong alignment
Burns et al. (2024) show that weak supervisors can elicit strong model capabilities. We study the adversarial dual: weak adversaries subverting strong model alignment. Our setting --- 3B attacking 8B --- is a concrete instance of weak-to-strong failure where the strong model's internal representations reveal the failure mechanism.

## 3. Method

### 3.1 Multi-turn adversarial self-play

**Setup.** A 3B adversary (Llama-3.2-3B-Instruct + LoRA, 2.3M trainable parameters) engages a frozen 8B victim (Llama-3.1-8B-Instruct, 4-bit quantized) in 5-turn conversations. Goals are sampled from JailbreakBench's 100 standardized harmful behaviors. The adversary receives a red-team system prompt at inference time; training data uses a compact goal-only format.

**Judging.** We use the JailbreakBench standard evaluation: the `Llama3JailbreakJudge` prompt applied to the victim's last response, evaluated by Llama-3.3-70B-Instruct via Together API. A dual-judge requirement (Llama Guard 3-1B agreement) provides additional precision.

**DPO training.** At each round, all conversations (wins and losses) are grouped by goal and paired: successful jailbreaks are "chosen," failed attacks are "rejected." Each multi-turn conversation is split into per-turn examples (a 5-turn conversation yields 5 training pairs at different conversation depths). DPO loss trains the adversary to prefer winning attack patterns over failing ones.

### 3.2 Safety probe

We extract per-turn hidden states from the victim's residual stream (middle layer, 4096-dim) at the last token position before each victim response. A logistic regression probe (L2-regularized, balanced class weights) is trained on these activations to predict JBB-verified unsafe labels.

**Dataset.** 1,860 conversations re-judged with the JBB standard prompt yield 166 verified jailbreaks. Per-turn splitting gives 9,300 turn-level examples (830 positive). 5-fold stratified CV achieves 0.958 +/- 0.007 AUC.

### 3.3 Stealth objective

The stealth loop scores each conversation's hidden states with the probe (max probability across turns). Adversary training uses stealth-weighted selection: wins are sampled with probability proportional to $(1 - p_{\text{probe}})^\alpha$ where $\alpha = 3.0$. This preferentially trains on attacks that succeed while producing low probe scores.

The probe is retrained every 5 rounds on accumulated data (adaptive defender), creating a competitive dynamic.

## 4. Experiments

### Experiment 1: Probe predicts jailbreaks (Section 5.1)

**Data.** 1,860 multi-turn conversations from multiple experimental conditions, all re-judged with JBB standard prompt. 166 JBB-verified unsafe (8.9%).

**Method.** Per-turn logistic regression probe on victim hidden states. 5-fold stratified CV.

**Expected result.** AUC ~0.96. *Already have this result.*

**Ablations:**
- Per-turn vs. conversation-level (mean pooling) probe
- Layer selection (early / middle / late)
- Probe at different turn depths (does prediction improve as conversation progresses?)

### Experiment 2: DPO vs SFT for sparse-reward self-play (Section 5.2)

**Setup.** Same starting adapter, same number of rounds (20), same candidates (30/round), same JBB judge.
- **Condition A:** SFT on wins only (buffered mode)
- **Condition B:** DPO on all conversations (win/loss pairs)

**Metric.** Per-round ASR, total wins. Mann-Whitney U test on per-round ASR distributions.

**Expected result.** DPO produces significantly higher ASR. *Partially have this --- dpo_jbb_v1 (7.5% avg) vs earlier SFT runs (1-3%). Need matched conditions.*

**Status:** Need to run a clean SFT baseline with the same starting adapter and JBB judge.

### Experiment 3: Stealth objective improves ASR (Section 5.3) [KEY EXPERIMENT]

**Setup.** Same DPO-trained adapter as starting point, same rounds (15), same candidates (50/round), same JBB judge.
- **Condition A:** Standard self-play (SFT on wins, no probe signal)
- **Condition B:** Stealth-weighted training (probe-evasion objective)

**Metrics.** Per-round raw ASR, stealth ASR, evasion rate, mean probe score on wins. Mann-Whitney U test on per-round ASR distributions.

**Expected result.** Condition B achieves higher raw ASR than Condition A. *Currently running Condition B (stealth_jbb_v1). Need matched Condition A.*

**This is the critical experiment.** If stealth training produces statistically significantly higher raw ASR under the JBB standard judge, we have the paper's central finding.

### Experiment 4: Adversary adapts to retrained probe (Section 5.4)

**Setup.** Within the stealth loop, the probe is retrained every 5 rounds on accumulated data.

**Metric.** ASR and evasion rate in the round immediately after probe retraining vs. 2 rounds later.

**Expected result.** ASR dips after retraining, recovers within 1-2 rounds. *Already seeing this in stealth_jbb_v1: Round 4 (retrain) drops to 4% ASR, Round 5 recovers to 18%.*

### Experiment 5: Qualitative attack analysis (Section 5.5)

**Method.** Compare attack strategies between stealth and non-stealth wins:
- Opening move classification (persona, simulation, direct, etc.)
- Turn-by-turn escalation patterns
- Victim response analysis (when does compliance begin?)
- Probe activation analysis (which hidden state dimensions differ?)

**Expected result.** Stealth attacks use qualitatively different strategies that avoid triggering the victim's safety representations.

**Status:** Need the control arm to compare against.

## 5. Results

[To be filled after experiments complete]

### 5.1 The victim knows it is being attacked
### 5.2 DPO provides dense signal for sparse-reward self-play
### 5.3 Stealth training improves attack success
### 5.4 The adversary adapts to stronger probes
### 5.5 Stealth attacks are qualitatively different

## 6. Discussion

### Implications for alignment
The disconnect between internal representation and behavior is concerning: the victim encodes safety-relevant information that, if acted upon, would prevent compliance --- yet it complies. This suggests that RLHF/DPO safety training creates behavioral guardrails that can be bypassed without disrupting the underlying safety representations, or conversely, that safety representations emerge as a byproduct of training but are not causally connected to refusal behavior.

### Weak-to-strong subversion
Our 3B adversary succeeds against an 8B victim despite a significant capability gap. The adversary doesn't need to be smarter than the victim --- it needs to find inputs that exploit the gap between the victim's safety knowledge and its safety behavior. This has implications for the scalability of alignment: if alignment fails in a predictable way (detectable in hidden states), then aligned models may be systematically vulnerable to adversaries that target this failure mode.

### Limitations
- Single victim model (Llama-3.1-8B-Instruct). Generalization to other architectures and scales unknown.
- ASR of 8-20% is far below GOAT's 97%. We study a constrained setting (small adversary, learned attacks) that prioritizes mechanistic insight over attack effectiveness.
- Probe is a linear classifier on a single layer. Nonlinear probes or multi-layer features may capture different aspects of safety representations.
- JBB judge is itself an LLM with its own biases. Human evaluation would strengthen the findings.

## 7. Conclusion

We demonstrate that language model safety representations predict but do not prevent jailbreaks: a 0.96 AUC probe detects impending compliance, yet the victim complies. A weak (3B) adversary exploits this gap through DPO self-play, and adding a stealth objective --- optimizing against the victim's own safety probe --- paradoxically increases attack success. This establishes a mechanistic link between internal safety representations and behavioral vulnerability, with implications for the weak-to-strong alignment problem.

---

## Experiment Schedule

### Already complete:
- [x] Probe training on JBB-labeled data (0.958 AUC)
- [x] DPO pre-training (dpo_jbb_v1, 20 rounds, 45/600 wins)
- [x] Self-play baseline (selfplay_jbb_v1, 10 rounds, 52/500 wins)
- [x] Stealth arm, partial (stealth_jbb_v1, ~7 rounds, in progress)
- [x] Re-judging all historical data with JBB standard (1860 convs)

### Still needed:
- [ ] **Complete stealth arm** (stealth_jbb_v1, 15 rounds total) [RUNNING NOW]
- [ ] **Control arm** (same adapter, 15 rounds, 50 candidates, NO stealth, SFT on wins) [QUEUE NEXT]
- [ ] **Clean SFT baseline** (same starting point as DPO, 20 rounds SFT-only, for Exp 2)
- [ ] **Probe ablations** (layer sweep, turn-depth analysis)
- [ ] **Qualitative analysis** (attack strategy comparison between stealth/non-stealth)
- [ ] **Statistical tests** (Mann-Whitney U on per-round ASR)

### Compute estimate:
- Stealth arm: ~8 more rounds, ~8 hours on 4090
- Control arm: 15 rounds, ~12 hours on 4090
- SFT baseline: 20 rounds, ~16 hours on 4090
- Probe ablations: ~1 hour (CPU)
- Total: ~36 hours on single 4090 (~$12)
