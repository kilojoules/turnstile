# Output-content positive control (output_content_control_v1)

Purpose: the harm-ablation null is only informative if the projection-ablation
method can move OUTPUT-CONTENT properties at all (the only prior positive control,
refusal, is an INPUT-GATE concept fit on prompts). Fit two output-content
diff-in-means directions on post-response reps at L16, then ADD (sufficiency) and
ABLATE (necessity, the arm harm failed) with random controls. Auto-measured, no
LLM judge. n=50/arm.

| property | arm | baseline | +direction / ablate | random |
|---|---|---|---|---|
| FRENCH (%french) | ADD (English-instructed) | 0% | **98%** | 0% |
| FRENCH (%french) | ABLATE (French-instructed) | 100% | **90%** | 100% |
| VERBOSITY (mean tokens) | ADD (terse-instructed) | 30 | **115** | 33 |
| VERBOSITY (mean tokens) | ABLATE (verbose-instructed) | 200 | **192** | 200 |

Ablate arms stayed coherent (191-192 tokens, not degraded).

## Finding: ADD works for output content, ABLATE is a weak lever

Consistent across a gate-ish (language) and a distributed (verbosity) property:
- **ADD is strongly causal**: French 0→98%, verbosity 30→115 tokens (~4x). The
  method is NOT blind to output content.
- **ABLATE barely moves it**: French 100→90%, verbosity 200→192 — even though the
  SAME direction, ADDED, flips the property strongly. Random controls flat.

## Reinterpretation of the harm-ablation null

The harm-ablation null (ablate harm ≈ random) is now **largely uninformative**:
projection ablation is a weak lever for output-content properties in general, so
"ablating the harm direction doesn't reduce uplift" is consistent with harm being
a normal, well-represented output-content property — not evidence that harm is
unrepresented or "not a direction." The mechanistic "harm != compliance" claim
CANNOT rest on the ablation null.

The intervention that DOES move output content is ADDITION. We proved ADD-French
(98%) and ADD-verbose (4x); we never ran a clean ADD-harm (the original additive
harm sweep was norm-confounded). So the decisive open test is **ADD-harm at
calibrated magnitude**.

## Caveat

The French/verbose properties are EXPLICITLY instructed via system prompt, so the
weak ABLATE arm is partly confounded — ablating a direction can't override an
explicit in-context instruction that keeps re-injecting the signal. Harm in the
win-replay is only CONTEXTUALLY induced (not instructed), so harm ablation failing
is somewhat more meaningful than French ablation failing. Still, the clean,
unconfounded takeaway is the ADD arm: the method moves output content via addition.

## Decisive next run

**Clean ADD-harm test.** On a set where harm can be expressed (win-replay or
refusal-bypassed harmful prompts), ADD the harm diff-in-means direction at
calibrated raw magnitude (like ADD-French), measure Stage-B uplift vs random-add.
- If ADD-harm raises uplift like ADD-French raises French -> harm IS a steerable
  output-content direction (just not ablatable, like everything else); the paper's
  "harm isn't steerable" is wrong.
- If ADD-harm fails while ADD-French/verbose succeed -> harm is special:
  plausibly because uplift = CAPABILITY/knowledge the 8B lacks (you can't inject
  content the model doesn't have), whereas French/verbosity are FORMS it can always
  produce. That is the real, capability-based form of "harm != compliance".
