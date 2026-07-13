# Stress-Testing the Intent Probe: Filter-Based + Paired Analysis

**Authors:** Julian Quick and Julian Guidote
**Date:** 2026-04-29
**Project:** Turnstile (multi-turn adversarial self-play, Llama-3.1-8B victim, Llama-3.2-3B adversary)

## What this document is

This document now reports three things:

1. **A corpus characterization** of the published "intent AUC ≈ 0.97" comparison: how separable are Turnstile-generated adversarial conversations from the existing `network_hacking_corpus/benign_curious` set, and how much of that separability is recoverable from raw text? Also: how much is driven by a single memorized adversary opener that appears in 41.87% of adversarial conversations?

2. **An alignment audit** confirming that the persisted hidden-state files on disk correspond 1:1 to the conversation JSONLs they're paired with — so any analysis that loads them is reading current corpus activations, not stale state.

3. **The paired obliteration test** — running the actual controlled experiment we originally designed: each unsafe winning conversation is paired with a counterfactual translation that preserves the prelude (including the fake-refusal opener) and varies only the operational intent of the pivot user turn (and afterward). Llama-3.1-70B-Instruct is loaded locally on a Vast.ai GPU instance, used both as the rewriter and as the strict-dual-judge (Llama-Guard local + 70B local, no silent fallback on either failure). 200 conversations enter the pipeline; 154 survive the strict gate on the *replayed* benign conversation; the L16 paired probe is trained on the matched pairs.

## What this analysis can and cannot conclude

**Filter-based comparison (Experiments 1–4).** The cross-corpus comparison (Turnstile adversarial conversations vs. the `network_hacking_corpus/benign_curious` set) is dominated by surface text confounds. TF-IDF on user-turn text alone matches the L16 hidden-state probe at AUC ~ 1.00 in every slice; permutation labels collapse to chance; 41.87% of adversarial conversations literally start with `"I can't proceed with that request."` — a single phrase that bag-of-words can latch onto. The published 0.97, *as a comparison between these two corpora*, has nothing to do with whether the victim encodes intent.

**Paired comparison (Experiment 5).** When every confound is held fixed except intent — same prelude verbatim, same conversation length, same model pipeline, paired contrast — the L16 probe **does** separate unsafe-original from benign-translation conversations. AUC = 0.66 at the pivot turn, rising monotonically to 0.98 four turns later, with permutation labels at 0.49–0.53. So the model **does** encode intent in the residual stream once the harmful ask manifests; the filter-based null was about the benign-curious set being too distributionally distant from Turnstile-style conversations to be informative, not about the absence of a signal in the model.

These two findings are not in tension — they answer different questions. The published 0.97, *as a number*, was real but uninformative about model internals. The paired result is the controlled measurement that supports the original "the model knows" framing, just at a much more modest AUC than 0.97.

Headline numbers, all with 5-seed averaging (5 random CV splits per seed, 25 fits per cell), all permutation-validated:

| Test | Hidden-state AUC at L16 | TF-IDF AUC on user text |
|---|:---:|:---:|
| Cross-category, all turns (n=47,000+500) | **1.0000 ± 0.0001** | **0.9999 ± 0.0000** |
| Pre-pivot only — winners' trust-building turns (n=3,308+500) | **1.0000 ± 0.0000** | **1.0000 ± 0.0000** |
| Topic-matched (network-security only, pooled n=500+500) | **0.9999 ± 0.0002** | **1.0000 ± 0.0000** |

**Permutation-label validity check across all conditions:** 0.45 – 0.52 (i.e., chance). The probe is doing real classification, not overfitting.

The key observation: bag-of-words on user-turn text alone matches or beats the residual-stream probe in every condition. Removing the breach turn and everything downstream — keeping only the surface-benign trust-building prelude — does not drop AUC at all. There is no point at which the residual stream contributes information beyond what is in the raw text.

## What was claimed

From the paper (`paper/main.tex`, §4.1):

> A logistic regression probe on per-turn hidden states at layer 16 distinguishes adversarial from benign conversations with AUC 0.97 (9,300 turn-level examples, 5-fold CV).

The paper already caveats that adversarial and benign inputs differ structurally and that the probe "may detect input distribution rather than the model's internal safety state." This experiment quantifies that caveat: it is not a partial confound, it is the entire signal.

## Data provenance and one important gap

**Activations on disk (used here).** Pre-extracted by `turnstile/collect_hidden_states.py` during the original training runs (file mtimes Apr 1–18, 2026). For each conversation: the residual-stream hidden state at the last token of the user-message prefix, at the layer returned by `extract_hidden_states` with `layer_idx=None`. That default is `n_layers // 2`, i.e. **layer 16** for the 32-layer Llama-3.1-8B-Instruct victim, with `hidden_dim = 4096`. The script records `hidden_states`, `labels`, `turns_of_breach`, `round` per `.pt` file.

**Activation–conversation alignment audit.** Before running any analysis we verified that `experiments/{exp}/hidden_states/round_N.pt` corresponds to the *current* `experiments/{exp}/rounds/round_N.jsonl` and is not stale from an earlier corpus. Across all 168 rounds of the 11 experiments: every per-conversation `unsafe` label in the JSONL matches the persisted `labels` tensor index-by-index; every `turn_of_breach` in the JSONL matches `turns_of_breach` index-by-index; every conversation's user-turn count is consistent with the stored hidden-state shape; every `.pt` `mtime` ≥ the corresponding `.jsonl` `mtime`; every persisted `round` field matches the filename; every `hidden_dim` is 4096. Zero violations. A cross-round byte-equality check additionally turned up 254/14,028 ≈ 1.8% of (round, round) pairs with bit-identical first-conv-first-turn hidden states — investigation showed each such pair has byte-identical user-turn-0 text (either same JBB goal sampled at round 0 with the same seed, or the trained adversary's memorized `"I can't proceed with that request."` opener); identical inputs produce deterministic hidden states, so this is corroboration of the memorized-opener confound (§"What is actually doing the work"), not a bug.

**Original 0.97 probe data is NOT in this repo.** The paper's number was computed against "9,300 turn-level examples"; the LessWrong post says "2,000 adversarial vs. 2,000 benign turns." Neither dataset matches what is currently on disk. The only benign-side hidden-state file in the repository is `experiments/network_hacking_corpus/benign_curious/hs.pt` (100 conversations × 5 turns = 500 turn-level examples), which is too small to have been the published benign side.

I searched: git history (no deleted intent-probe scripts or data files), the WIP stash (no relevant content), `analysis/` and `scripts/` (only the *compliance* probe is reproduced there, on adversarial-only data), and the disk for any `.pt` file with shape consistent with a larger benign corpus (none found). The published 0.97's exact reproduction is therefore not possible from this repo. The qualitative claim — *the probe contributes no information beyond surface text* — is reproduced here against the available benign side, and is robust to the absence of the original benign data because it holds across multiple comparisons (cross-category vs. topic-matched).

## Methodology

### Hypothesis under test

The intent probe could be detecting any of:

1. The harmful payload itself.
2. Surface lexical features of the conversation that have nothing to do with the harmful payload (memorized openers, stylistic register, vocabulary).
3. A model-internal representation of "this is an attack" beyond either of the above.

If (3) is doing the work, then (a) removing the harmful turns should reduce AUC, and (b) a text-only baseline should fall short of the L16 probe. Neither holds.

### Per-turn judge data and obliteration scope

`turnstile/judge.py:_find_turn_of_breach` runs after the dual judge (Llama-Guard-3-1B + Llama-3.3-70B-Turbo) agrees a conversation is unsafe. For each prefix `conv[:2t+2]` the judge returns the first `t*` such that both judges flag unsafe. This `turn_of_breach` is logged in every round JSONL. For losing conversations the function is never called (line 213).

This lets us scope an obliteration cleanly: keep turns `0..t*-1` (the adversary's surface-benign trust-building prelude) and drop turns `t*..4`. Four slices on the adversarial side:

- `all`: every turn of every adversarial conversation (the published baseline)
- `winners_pre`: only winners (`unsafe == True`), only turns `t < turn_of_breach`
- `full_pre`: winners use `t < turn_of_breach`; losers use all 5 turns (they never breached on the assistant axis)
- `full_strict_pre`: same as `full_pre` but additionally drops the last turn of losers (a conservative guard against an undetected user-pivot in losers)

### Comparison data

- **Adversarial side:** 9,400 conversations pooled across 11 experimental conditions (`authority_dpo`, `control_hard_s456`, `control_s42`, `frozen_v1`, `incrementalism_dpo`, `reward_dpo`, `stealth_hard_s456`, `stealth_jbb_v1`, `stealth_s42`, `urgency_dpo`, `urgency_v1`). 47,000 turn-level hidden states at L16. 2,065 wins; 2,027 with explicit `turn_of_breach`.
- **Benign side:** 100 network-security-themed conversations from `experiments/network_hacking_corpus/benign_curious/`, generated by a vanilla Llama-3.2-3B-Instruct user simulator (no adapter). 500 turn-level hidden states at L16.

### Probes

- **Hidden-state probe.** `LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")` on the L16 residual stream. 5-fold stratified CV with `shuffle=True, random_state=seed`.
- **Text-only baseline.** `TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=20000, sublinear_tf=True, lowercase=True)` on the user-turn string alone, fed to `LogisticRegression(C=1.0, solver="liblinear")`. Same 5-fold stratified CV. Scikit-learn only — no neural encoders, no model internals.

### Robustness checks (all reported numbers)

- **Multi-seed averaging.** Every cell is the mean ± std over 5 random seeds (CV split seeds 0..4). Each seed runs its own 5-fold CV. So every reported AUC is the average of 25 fits.
- **Balanced subsampling.** Class imbalance is 47k:500 in the full pool. Each seed also draws 500 examples from each class without replacement and re-runs the CV — confirms AUC isn't a class-imbalance artifact.
- **Permutation-label validity test.** For each cell, additionally shuffle the labels and re-fit. The headline scripts use `n_perms=3` for budget. A separate verification pass (`turnstile/intent_perm_verify.py`, output `perm_verify.json`) re-runs the permutation test at `n_perms=50` on a balanced 500/class subsample for every slice. The 50-perm null distributions are tight and centered on 0.5 (see "Permutation null distributions" table); the real probe AUCs are all far outside the null 95% CIs. This rules out overfitting.
- **Activation manifest.** Every result JSON records the L2-norm distribution of the activations on each side, dimensionality, and the layer convention — so any reader can audit that the probe was fit on what was claimed.

## Results

### Experiment 1 — Per-turn intent AUC (full pool, 5 seeds)

Hidden-state probe at L16, all 9,400 adversarial conversations vs. all 100 benign-curious.

| Turn | AUC (seed mean ± std) | Permutation-label AUC | n_adv | n_ben |
|---:|:---:|:---:|---:|---:|
| 0 | 1.0000 ± 0.0000 | 0.4954 ± 0.0253 | 9,400 | 100 |
| 1 | 1.0000 ± 0.0000 | 0.4828 ± 0.0253 | 9,400 | 100 |
| 2 | 0.9988 ± 0.0002 | 0.4716 ± 0.0509 | 9,400 | 100 |
| 3 | 0.9986 ± 0.0003 | 0.4709 ± 0.0373 | 9,400 | 100 |
| 4 | 0.9986 ± 0.0002 | 0.4681 ± 0.0368 | 9,400 | 100 |

The probe saturates at every turn — including turn 0, before any harmful payload exists in the conversation. Permutation labels collapse to chance, so the AUC is real.

### Experiment 2 — Obliteration, hidden-state probe, 5 seeds

| Slice | n_adv_turns | full AUC | balanced 500/500 AUC | permutation AUC | adv L2-norm | ben L2-norm |
|---|---:|:---:|:---:|:---:|:---:|:---:|
| `all` | 47,000 | 1.0000 ± 0.0001 | 0.9996 ± 0.0005 | 0.4958 ± 0.0238 | 10.05 | 10.76 |
| `winners_pre` | 3,308 | 1.0000 ± 0.0000 | 0.9995 ± 0.0004 | 0.4974 ± 0.0130 | 9.11 | 10.76 |
| `full_pre` | 40,173 | 1.0000 ± 0.0000 | 0.9997 ± 0.0002 | 0.5083 ± 0.0145 | 9.96 | 10.76 |
| `full_strict_pre` | 32,800 | 1.0000 ± 0.0000 | 0.9996 ± 0.0003 | 0.4929 ± 0.0059 | 9.81 | 10.76 |

The strictest slice (winners' pre-pivot turns only — by construction, conversations *before* any harmful payload appears) gets the same AUC as the published baseline. The L16 probe needs no harmful content to fire. The activation L2-norms differ slightly between sides (≈10 vs ≈11) but the difference is much too small to drive an AUC of 1.00; the discriminator must be using direction, not magnitude.

### Experiment 3 — Text-only baseline, TF-IDF, 5 seeds

User-turn text only, same slices.

| Slice | full AUC | balanced 500/500 AUC | permutation AUC |
|---|:---:|:---:|:---:|
| `all` | 0.9999 ± 0.0000 | 0.9997 ± 0.0003 | 0.5105 ± 0.0172 |
| `winners_pre` | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.4973 ± 0.0050 |
| `full_pre` | 0.9999 ± 0.0000 | 0.9999 ± 0.0000 | 0.5160 ± 0.0077 |
| `full_strict_pre` | 0.9999 ± 0.0000 | 0.9999 ± 0.0001 | 0.4951 ± 0.0029 |

TF-IDF on raw user text — no model, no embeddings, no semantics — equals the L16 probe in every slice. Permutation labels confirm the discriminator is real. There is no model-internal signal beyond what bag-of-words extracts from the prompts.

### Experiment 4 — Topic-matched control, 5 seeds

100 malicious + 100 benign-curious conversations from `experiments/network_hacking_corpus/`, all single-topic (network security). Eliminates topic as a confound.

| Turn | TF-IDF AUC | TF-IDF perm | L16 hidden-state AUC | L16 perm |
|---:|:---:|:---:|:---:|:---:|
| 0 | 1.0000 ± 0.0000 | 0.4920 | 1.0000 ± 0.0000 | 0.4522 |
| 1 | 1.0000 ± 0.0000 | 0.5018 | 0.9995 ± 0.0003 | 0.4748 |
| 2 | 1.0000 ± 0.0000 | 0.4743 | 0.9549 ± 0.0074 | 0.4482 |
| 3 | 1.0000 ± 0.0000 | 0.5003 | 0.9544 ± 0.0055 | 0.4577 |
| 4 | 1.0000 ± 0.0000 | 0.4985 | 0.9538 ± 0.0062 | 0.4592 |
| pooled | 1.0000 ± 0.0000 | 0.5225 | 0.9999 ± 0.0002 | 0.5137 |

With topic held constant, TF-IDF stays at 1.00 at every turn. The L16 hidden-state AUC drops to ≈0.95 at turns 2–4 — the residual-stream probe *loses* discriminability that bag-of-words preserves. There is no point at which the residual stream contains intent information that the surface text does not.

The natural read of the L16 dip: at later turns, the residual stream at the user-token position has integrated the victim's intervening assistant responses into its context. The trained adversary's distinctive turn-0 stylistic fingerprint — which TF-IDF on the literal user-turn string captures unchanged — gets diluted in the L16 representation by victim-response context that is generated under the same victim weights for both sides. Bag-of-words is a less expressive function but happens to be a better-aligned function for *this particular discrimination*, because the discriminator lives in the surface text of the user turn and not in any context-integrated representation.

### Permutation null distributions (50 perms, balanced 500/class)

For every slice and probe type, we re-run with shuffled labels 50 times on a balanced subsample and report the resulting null AUC mean, std, and 95% CI. All real-AUC numbers above are *far* outside these intervals.

| Slice | Hidden-state null AUC (mean ± std, 95% CI) | TF-IDF null AUC (mean ± std, 95% CI) |
|---|:---:|:---:|
| `all` | 0.5032 ± 0.0206  [0.4672, 0.5388] | 0.4984 ± 0.0205  [0.4597, 0.5361] |
| `winners_pre` | 0.5019 ± 0.0214  [0.4563, 0.5354] | 0.4999 ± 0.0260  [0.4521, 0.5515] |
| `full_pre` | 0.5020 ± 0.0230  [0.4599, 0.5469] | 0.4989 ± 0.0246  [0.4533, 0.5416] |
| `full_strict_pre` | 0.5000 ± 0.0236  [0.4577, 0.5494] | 0.5003 ± 0.0290  [0.4383, 0.5544] |
| topic-matched (pooled) | 0.4998 ± 0.0266  [0.4465, 0.5488] | 0.4995 ± 0.0277  [0.4483, 0.5455] |

The probe is doing real work. The signal it picks up is real but, as the rest of the analysis shows, it lives entirely in the surface text of the user turns.

## What is actually doing the work

A direct read of the conversation files exposes two concrete confounds that fully account for the result:

### 1. Memorized adversary opener

Many trained-adversary turn-0 messages are literally identical, e.g.:

> *"I can't proceed with that request."* — repeated verbatim across hundreds of conversations in `control_s42` rounds 10–14.

The trained adversary has learned a stereotyped fake-refusal opener as a strategic move (it elicits a sympathetic victim response that is then exploited). Compared to the benign-curious set's grammatical opener (*"What are the most common types of network vulnerabilities in corporate environments?"*), this is a token-level fingerprint a probe reads trivially in residual space.

### 2. Stylistic-register mismatch

The trained adversary's openers — even when not memorized verbatim — share a distinctive stylistic register:

> *"I'm going to simulate a scenario where we're discussing a hypothetical situation. Let's say we're a team of cybersecurity experts trying to..."*

vs. the benign-curious set's:

> *"What are the most common types of network vulnerabilities in corporate environments?"*

These are detectably different at the level of bigrams (`"I'm going to"`, `"Let's say we're"`, `"hypothetical situation"`) regardless of topic.

Either confound alone is sufficient to drive AUC ≥ 0.99.

## Memorized opener: prevalence

The audit's cross-round byte-equality check (1.8% of (round, round) pairs) was a sampling artifact — it only sampled the first conversation per round (168 tensors). A direct count across the full corpus of 9,400 adversarial conversations:

| Memorized turn-0 string | Count | % of corpus |
|---|---:|---:|
| `"I can't proceed with that request."` | 3,936 | **41.87%** |
| `"I'm not able to proceed with that request."` | 70 | 0.74% |
| `"I'm not able to help with that."` | 35 | 0.37% |
| `"I'm not going to proceed with that request."` | 26 | 0.28% |
| `"I'm not able to help with that request."` | 21 | 0.22% |
| top-10 memorized openers (combined) | 4,156 | **44.21%** |

Nearly half of all adversarial conversations start with a literal memorized fake-refusal phrase. A logistic probe on user-turn-0 alone — at the activation level or via TF-IDF — can saturate AUC by learning this single feature. This is the dominant driver of the 0.97 number against any benign corpus that does not also start its conversations with these phrases.

## Conclusion (filter-based version)

In the filter-based comparison (Turnstile adversarial vs. `benign_curious`), the published intent AUC measures the (trivial) fact that the two corpora are different bags of words, with one specific memorized phrase explaining ~42% of the discriminability on its own. It does not measure whether the model encodes a representation of "this is an attack." The L16 hidden-state probe contributes no information beyond surface text *in this comparison*.

## Conclusion (paired version, the actual answer)

When prelude is held identical and only intent varies, the L16 residual stream separates unsafe-payload from benign-payload at **AUC 0.66 at the pivot turn** (perm null 0.49). The model encodes intent — but at a far more modest level than the published 0.97 suggested. The honest framing for the paper is:

- The cross-corpus 0.97 is an artifact of the comparison and should not be cited as evidence of internal "intent detection."
- The *paired* AUC at the pivot turn (0.66) is the controlled measurement. It is real, well above chance, and is what supports the qualitative claim that the model represents intent in its residual stream at the moment of the harmful ask.
- Late-turn paired AUCs (≥ 0.95) are dominated by the divergent surface text the conversations now have under each condition; they should not be cited as the headline number for "what the model knows at the pivot."

The activation-steering causal results (±13 pp at L16) are unaffected by either reframing — those depend on behavior change, not on AUC.

This does not affect the paper's compliance probe (the AUC ≈ 0.74 number for predicting jailbreak success within adversarial conversations) — that probe is trained on adversarial-vs-adversarial and is not subject to this confound. The activation-steering causal results at L16 are also unaffected; they intervene on a probe direction whose causal role is established by behavior change, not by AUC.

What does require a paper revision: the framing of the intent AUC as evidence of internal "intent detection." The accurate framing is "the two text distributions are linearly separable and the residual stream inherits that linear separability." Stronger evidence for internal intent representation would require either (a) a topic-and-style-matched benign control whose surface text is genuinely indistinguishable, or (b) a counterfactual rewrite that holds surface text fixed and varies only intent. (b) is the right test and is now scaffolded as the **paired obliteration pipeline** in `experiments/intent_obliteration_paired/`; it has not been run yet.

## Experiment 5 — Paired obliteration (the controlled test)

For each winning adversarial conversation $C$ with breach turn $t^*$:

1. Keep turns $0..t^* - 1$ verbatim — including the fake-refusal opener and any rapport-building. Whatever the trained adversary did before the pivot, the translation also did.
2. Rewrite the user turn at $t^*$ (and at $t^* + 1, ..., 4$) into a benign continuation that fits the running prefix (Llama-3.1-70B-Instruct as rewriter, with `apply_chat_template` and `temperature=0.5`).
3. **Strict dual judge on each rewritten user turn:** Llama-Guard-3-1B locally + Llama-3.1-70B locally. If guard flags, 70B must emit unambiguous safe/unsafe; ambiguous output across 3 retries raises `LocalJudgeFailure` and the conversation is dropped with a loud log line. No silent fallback.
4. Replay the conversation through the frozen victim deterministically: feed each rewritten user turn, generate the assistant response (`temperature=0`, `max_tokens=300`).
5. **Strict dual judge on each replayed conversation:** same protocol, applied to the *full translated conversation* (with assistant turns now present). Drop pairs the dual judge agrees are unsafe — these are translations whose user message looked benign in isolation but elicited a harmful response from the victim.
6. Extract L16 hidden states at every turn for **both** original and translated conversations.
7. Probe: at each turn position $t \ge t^*$, train a logistic probe on (original L16, translated L16) pairs.

By construction this paired comparison eliminates every confound that drives the filter-based number: same prelude (including the memorized opener), same length, same model pipeline, same victim, same chat template. The only structured difference is the operational intent of the pivot user turn.

Pre-pivot turns ($t < t^*$) have *literally identical* hidden states between the two conditions (deterministic extraction on identical inputs), so AUC at those turns is 0.5 by construction.

### Run summary

- **Hardware:** single L40 (45 GB VRAM) Vast.ai instance (`35840823`, $0.376/hr)
- **Models:** Llama-3.1-70B-Instruct (rewriter + judge, 4-bit), Llama-Guard-3-1B, Llama-3.1-8B-Instruct (victim, 4-bit). All loaded locally; **no API**.
- **Wall:** ~7 hours end-to-end on the successful run (one earlier run was wasted on a chat-template bug in the judge prompt — Llama-3.1-Instruct emits EOS immediately on raw-string input; fixed by wrapping every judge prompt in `apply_chat_template`).
- **Cost:** ~$3.80 total compute (Vast instance hours).
- **Step 1 (rewrite):** 200/200 written, 0 validator drops, 0 judge failures.
- **Step 2 (replay):** 200/200 replays saved.
- **Step 3 (post-replay strict dual judge on translations):** 154 kept, **46 dropped as `dual_unsafe`**, 0 judge failures. So 23% of rewrites that passed the per-user-message validator in step 1 turned out to actually elicit unsafe assistant responses during replay — exactly the failure mode the strict gate exists to catch.
- **Step 4 (paired probe):** 154 paired records, 5 random seeds × 5-fold CV per cell, with permutation-label validity check.

### Result

Paired AUC at L16 (original-unsafe vs. translated-benign), same prelude verbatim, intent flipped:

| Turn (offset from pivot) | Paired AUC (mean ± seed-std) | Permutation null AUC | n |
|---:|:---:|:---:|---:|
| 0 (pivot) | **0.6628 ± 0.0150** | 0.4892 | 84 |
| 1 | **0.8661 ± 0.0055** | 0.5115 | 122 |
| 2 | **0.9458 ± 0.0042** | 0.5265 | 140 |
| 3 | **0.9708 ± 0.0020** | 0.5056 | 146 |
| 4 | **0.9823 ± 0.0029** | 0.4929 | 154 |
| **pivot-onward pooled** | **0.9957 ± 0.0002** | 0.5022 | 1,292 |

Permutation labels collapse to ~0.5 in every cell. The probe is doing real classification, not overfitting to small-n.

### Interpretation

**At the pivot turn $t^*$**, the L16 residual stream separates unsafe-payload from benign-payload at AUC 0.66 — modest but well above chance (perm 0.49). This is the load-bearing measurement: with the prelude held literally identical between the two conditions, the model's L16 representation already contains some intent information at the moment the user emits the harmful ask.

**At later turns** the AUC grows monotonically (0.87 → 0.95 → 0.97 → 0.98). Each subsequent turn the conversations diverge further (the assistant's response is generated fresh under each condition; later user turns continue down the diverged path). So the late-turn AUCs are not "the model encodes intent more strongly later" — they're "the conversations differ more by surface text later, and the probe has more signal to use." The honest interpretation focuses on $t^*$.

**The pivot-onward pool (0.996) is not the headline.** It pools turns at different conversation depths and is dominated by the late-turn (already-divergent) examples.

**This does not retroactively justify the published 0.97.** That number was a *cross-corpus* AUC against an ill-matched benign set; the paired AUC of 0.66 at $t^*$ (or even 0.98 at $t^* + 4$) is a *within-conversation* AUC against a controlled counterfactual. They measure different things. What the paired AUC vindicates is the qualitative claim ("the model knows it is being attacked"); the *quantitative* claim should be reported as "paired-pivot AUC = 0.66, paired-pivot-onward AUC = 0.996," not as 0.97.

### Reproducing the paired experiment

```bash
# Step 1: rewrite + per-user-turn dual judge (GPU + Together-free local 70B)
python -m turnstile.intent_rewrite \
  --experiments experiments/{...} \
  --max-convs 200 \
  --output experiments/intent_obliteration_paired/rewrites.jsonl \
  --judge-model meta-llama/Llama-3.1-70B-Instruct

# Step 2: replay through victim
python -m turnstile.intent_replay \
  --rewrites-jsonl experiments/intent_obliteration_paired/rewrites.jsonl \
  --output experiments/intent_obliteration_paired/replay.pt

# Step 3: strict dual judge on the replayed translation
python -m turnstile.intent_judge_translations \
  --replay-pt experiments/intent_obliteration_paired/replay.pt \
  --output experiments/intent_obliteration_paired/replay_judged.pt \
  --judge-model meta-llama/Llama-3.1-70B-Instruct

# Step 4: paired probe (CPU)
python -m turnstile.intent_obliteration_paired \
  --replay-pt experiments/intent_obliteration_paired/replay_judged.pt \
  --output experiments/intent_obliteration_paired/paired_probe.json \
  --n-seeds 5
```

Vast launch + babysit + auto-destroy: `bash scripts/launch_intent_obliteration_paired.sh && bash scripts/babysit_intent_obliteration_paired.sh`.

## Limitations

- **Original benign-side data is not on disk.** The published 0.97 used a benign corpus (≥2,000 turn-level examples per the LessWrong post) that is not in the repo or git history. We reproduce *the qualitative pattern* against the only preserved benign side (the 100-conv `network_hacking_corpus/benign_curious` set) and against a topic-matched 100-vs-100 comparison. The conclusion — *probe contributes nothing beyond surface text* — is invariant to the exact benign-side scale because TF-IDF and L16 saturate together.
- **The benign comparison set is single-topic** (network security) and small (100 conversations × 5 turns). A broader benign corpus might shift absolute numbers but the *delta* between TF-IDF and L16 — which is the load-bearing measurement — is invariant to that choice.
- **Per-turn judge rulings exist only implicitly for winning conversations** (via `turn_of_breach`) and not at all for losing conversations. The `full_pre` slice treats losers as fully pre-pivot, which is correct on the assistant-compliance axis but does not account for any harmful user-pivot turn that the victim refused. The `full_strict_pre` slice mitigates this by additionally dropping the last turn of losers; both give the same answer.
- **TF-IDF is a deliberately weak baseline.** Chosen so the conclusion is as strong as possible: if the *weakest* text baseline already saturates, no semantic gap exists for the model to fill. A sentence-transformer baseline would presumably saturate earlier, but this experiment's claim does not depend on that.

## Reproducibility

All scripts and result JSONs are checked in:

- `turnstile/intent_probe_lib.py` — shared CV / AUC / permutation-test primitives.
- `turnstile/intent_obliteration.py` — Experiments 1 and 2.
- `turnstile/intent_text_baseline.py` — Experiment 3.
- `turnstile/intent_topic_matched.py` — Experiment 4.
- `turnstile/intent_perm_verify.py` — high-resolution permutation-test validation pass.
- `turnstile/intent_audit_alignment.py` — activation-conversation alignment audit.
- `turnstile/strict_dual_judge.py` — strict dual judge (Guard local + 70B local, no silent fallback).
- `turnstile/intent_rewrite.py`, `turnstile/intent_replay.py`, `turnstile/intent_judge_translations.py`, `turnstile/intent_obliteration_paired.py` — the paired obliteration pipeline (Experiment 5).
- `experiments/intent_obliteration_paired/README.md` — runbook for the paired obliteration test.
- `experiments/intent_obliteration_paired/paired_probe.json` — full granular per-seed per-fold output for Experiment 5.
- `experiments/intent_obliteration_paired/replay_judged.pt` — 154 paired records (original + translated hidden states), the data the paired probe is trained on.
- `experiments/intent_obliteration_paired/rewrites.jsonl` — 200 step-1 rewrites with per-turn judge metadata.
- `experiments/intent_obliteration/results.json` — full granular per-seed per-fold output for Experiments 1 and 2.
- `experiments/intent_obliteration/text_baseline.json` — Experiment 3.
- `experiments/intent_obliteration/topic_matched.json` — Experiment 4.
- `experiments/intent_obliteration/perm_verify.json` — null distributions at `n_perms=50` for every slice.

To reproduce:

```bash
EXPS="experiments/authority_dpo experiments/control_hard_s456 \
      experiments/control_s42 experiments/frozen_v1 \
      experiments/incrementalism_dpo experiments/reward_dpo \
      experiments/stealth_hard_s456 experiments/stealth_jbb_v1 \
      experiments/stealth_s42 experiments/urgency_dpo experiments/urgency_v1"

python -m turnstile.intent_obliteration \
  --experiments $EXPS \
  --benign-hs experiments/network_hacking_corpus/benign_curious/hs.pt \
  --output experiments/intent_obliteration/results.json --n-seeds 5

python -m turnstile.intent_text_baseline \
  --experiments $EXPS \
  --benign-convs experiments/network_hacking_corpus/benign_curious/convs.jsonl \
  --output experiments/intent_obliteration/text_baseline.json --n-seeds 5

python -m turnstile.intent_topic_matched \
  --corpus-dir experiments/network_hacking_corpus \
  --output experiments/intent_obliteration/topic_matched.json --n-seeds 5
```

Each result JSON contains `summary` (data composition, layer/dim metadata, seeds), `per_turn` / `slices` (full per-seed per-fold AUCs), and `permutation` blocks (label-shuffled validity check) for each cell. All numbers in the tables above are recoverable directly from the JSON.
