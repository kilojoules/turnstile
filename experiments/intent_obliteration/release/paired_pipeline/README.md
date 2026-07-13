# Proper Obliteration Test — Paired Original vs. Translation

This experiment is the **controlled** version of the intent-probe stress test. The earlier work in `experiments/intent_obliteration/` compared adversarial pre-pivot turns against the existing `network_hacking_corpus/benign_curious` set — a comparison confounded by structural and stylistic differences (the trained adversary's memorized opener appears on the adversarial side and not the benign side, etc.).

This experiment removes those confounds by **construction**:
- Both sides come from the same conversations.
- The benign side inherits the original prelude **verbatim**, including the fake-refusal opener (`"I can't proceed with that request."`) — whatever the trained adversary did before the pivot, the translation also did.
- Only the user pivot turn (and its successors) is rewritten to be benign.

## Pipeline

Four scripts, run in order. **All judge / rewrite work is local — no API.**

1. **`turnstile/intent_rewrite.py`** (GPU, ~36 GB VRAM). Loads Llama-3.3-70B-Instruct (rewriter + judge) and Llama-Guard-3-1B at 4-bit quantization. For each winning adversarial conversation, generate a benign rewrite for each user turn from `turn_of_breach` onward via local 70B, then validate via the **strict dual judge** (`turnstile/strict_dual_judge.py`): if guard flags the rewrite, the 70B must emit an unambiguous safe/unsafe verdict. On persistent ambiguity or CUDA error, `LocalJudgeFailure` is raised and the conversation is dropped with a loud log line. There is no silent fallback. Output: `rewrites.jsonl`.

2. **`turnstile/intent_replay.py`** (GPU, ~24 GB VRAM). Load the victim Llama-3.1-8B. For each rewritten conversation, replay deterministically: keep turns `0..t*-1` verbatim, substitute rewritten user turns at `t*..end`, generate fresh assistant responses at `temperature=0`. Extract L16 hidden states at each turn for **both** the original and translated conversations. Exception handling is narrow (only CUDA OOM and malformed-input get caught and skipped); other errors propagate. Output: `replay.pt`.

3. **`turnstile/intent_judge_translations.py`** (GPU, ~36 GB VRAM). Reload 70B + Guard, run the strict dual judge on every translated conversation produced by replay. Drop pairs the dual judge agrees are unsafe. Hard-fail with `LocalJudgeFailure` on judge errors — never silently treat an unjudgeable translation as safe. Output: `replay_judged.pt`.

4. **`turnstile/intent_obliteration_paired.py`** (CPU). For each turn position `t >= t*`, train a probe on (original L16, translated L16) pairs from `replay_judged.pt`. Multi-seed CV with permutation-label validity check. Output: `paired_probe.json`.

### Why local 70B instead of Together API

- **Removes silent-fallback risk by construction.** With an API there's a non-trivial probability of transient failure on each call; with the model loaded locally there is no API to fail. The remaining failure modes (CUDA error, ambiguous model output) are still handled with `LocalJudgeFailure`, never silently defaulted.
- **No per-call cost.** All inference is local GPU time.
- **70B is reused.** The same loaded 70B serves both the rewriter (text generation, T=0.5) and the judge (verdict generation, T=0) in step 1, so we pay the load cost once.

### VRAM budget per step

| Step | Models loaded | 4-bit VRAM |
|---|---|---:|
| 1 (rewrite) | Llama-3.3-70B + Llama-Guard-3-1B | ~36 GB |
| 2 (replay) | Llama-3.1-8B (victim) | ~6 GB |
| 3 (judge translations) | Llama-3.3-70B + Llama-Guard-3-1B | ~36 GB |
| 4 (probe) | none | 0 GB |

Each step fits comfortably on a 48 GB instance (A40 / A6000 / L40).

## Headline question

At the breach turn `t*` (and `t*+1, t*+2, ...`), can a probe on the victim's L16 hidden state distinguish the unsafe-payload original from the benign-payload translation? The two conversations share an *identical* prelude, identical length, and a paired contrast that differs only in the operational intent of the user turn at `t*` and afterwards.

- **If AUC at `t*` is high (≥ 0.7).** The model encodes intent in the residual stream — the safety machinery represents "this is asking for something harmful" beyond surface text. The published 0.97 was over-explained by structural confounds, but a real intent signal exists at meaningful AUC.
- **If AUC at `t*` is near 0.5.** The model does not encode intent at L16. Safety machinery, if it exists, lives elsewhere (later layers, attention patterns, output-side logits).

Pre-pivot turns (`t < t*`) are deterministically identical hidden states between the two conditions (same input prompt → same output of `extract_hidden_states`). AUC at those turns is 0.5 by construction; we report it anyway as a validity check.

## Run

```bash
# Step 1: rewrite + dual-judge validation (local 70B + Guard on GPU)
python -m turnstile.intent_rewrite \
  --experiments experiments/{authority_dpo,control_hard_s456,control_s42,frozen_v1,incrementalism_dpo,reward_dpo,stealth_hard_s456,stealth_jbb_v1,stealth_s42,urgency_dpo,urgency_v1} \
  --max-convs 200 \
  --output experiments/intent_obliteration_paired/rewrites.jsonl \
  --judge-model meta-llama/Llama-3.3-70B-Instruct \
  --guard-model meta-llama/Llama-Guard-3-1B

# Step 2: replay (load victim Llama-3.1-8B)
python -m turnstile.intent_replay \
  --rewrites-jsonl experiments/intent_obliteration_paired/rewrites.jsonl \
  --output experiments/intent_obliteration_paired/replay.pt \
  --victim-model meta-llama/Llama-3.1-8B-Instruct

# Step 3: strict dual judge on translations (reload 70B + Guard)
python -m turnstile.intent_judge_translations \
  --replay-pt experiments/intent_obliteration_paired/replay.pt \
  --output experiments/intent_obliteration_paired/replay_judged.pt \
  --judge-model meta-llama/Llama-3.3-70B-Instruct \
  --guard-model meta-llama/Llama-Guard-3-1B

# Step 4: paired probe (CPU, < 5 min)
python -m turnstile.intent_obliteration_paired \
  --replay-pt experiments/intent_obliteration_paired/replay_judged.pt \
  --output experiments/intent_obliteration_paired/paired_probe.json \
  --n-seeds 5
```

## Cost & time estimate (local 70B)

The 70B at 4-bit is the bottleneck. On an A40 (48 GB) you should expect ~5–10 tokens/sec for the rewriter and ~50 tokens/sec for the (very short, 12-token) judge calls.

| Step | Hardware | Time | Cost |
|---|---|---|---|
| Rewrite (200 convs × ~3 turns: ~600 rewrites @ 350 tok + ~600 judge calls @ 12 tok) | 1× A40/A6000 (48 GB) | ~3–6 hours | Vast.ai instance time |
| Replay (200 convs × ~3 victim generations + HS extractions) | 1× 4090/A40 | ~30–60 min | Vast.ai instance time |
| Translation judge (~200 conversations × 1 dual-judge each) | 1× A40/A6000 | ~10–20 min | Vast.ai instance time |
| Paired probe | local | ~3 min | $0 |

Total: ~4–7 hours wall time on a 48 GB GPU instance, **no API spend**. At $0.40/hr for an A40 on Vast.ai, ~$2–3 of compute. If you have access to an A100 80 GB the rewrite step roughly halves.

## Strict dual-judge protocol (no silent fallback)

Both step 1 (rewriter validator) and step 3 (translation gating) use `turnstile/strict_dual_judge.py`, whose contract is:

1. Run **Llama-Guard-3-1B** locally on the input (a user message in step 1, a full conversation in step 3).
2. If guard returns "safe": the input is treated as safe (single-judge-safe is "safe" by JBB-standard). 70B is **not** consulted, because there is nothing to confirm.
3. If guard returns "unsafe": **70B must emit an unambiguous verdict.** The local 70B is given the JBB-standard judge prompt (or the user-turn variant in step 1) and asked for a single-token `safe`/`unsafe` answer. If the output contains neither token unambiguously, we re-generate up to `--judge-max-retries` times. If the model still cannot be parsed, or if generation raises a CUDA error, the function raises `LocalJudgeFailure` (a `RuntimeError` subclass). The caller drops the record with a loud log line. The judge **never** silently treats an unjudgeable input as "safe" or "unsafe."

This is intentionally stricter than the existing `turnstile/judge.py` used during training-time labelling, which on Together API failure sets `together_unsafe = False` and proceeds (line 200–202 of that file). That training-time default is conservative for *gating jailbreak success* but is the wrong default for *validating that a counterfactual is benign* — a silently-mis-validated harmful translation would contaminate the benign side of the paired probe. The strict module exists to remove that risk; running 70B locally further removes the network-failure axis entirely.

## Caveats / failure modes to watch

- **Rewriter quality.** If the rewriter produces text that's too formulaic ("How do I learn about X for academic purposes?"), it may be in a distinct distributional bucket from "natural" benign user input. The result of the probe is *still* meaningful (it pairs original vs. rewriter-output, both flowing through the same pipeline), but generalization to "real benign users" is bounded. Spot-check ~50 rewrites manually before trusting the AUC.
- **Length & register match.** The rewriter is asked to stay within ±25% of original length. If rewrites drift longer/shorter on average, that's a residual confound. The paired-probe JSON includes `manifest_dense` per turn (norm distributions), which exposes any systematic drift.
- **High judge-failure rate is a signal, not noise.** If step 1 or step 3 drops more than ~5% of records due to `TogetherJudgeFailure`, do not just re-run with a higher retry count. Investigate: it likely means the Together API is in degraded service or the judge prompt is hitting an edge case the model can't handle. Either way, silent fallback would be worse — it would let arbitrary translations through.
