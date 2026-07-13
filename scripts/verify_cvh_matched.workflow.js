export const meta = {
  name: 'verify-cvh-matched',
  description: 'Adversarially verify the apples-to-apples compliance-vs-harm steering result before it goes in the paper',
  phases: [
    { title: 'Re-derive', detail: 'independent re-computation of every headline number from raw jsonl' },
    { title: 'Refute', detail: 'try to break the "harm knob does not turn" claim' },
    { title: 'Synthesize', detail: 'reconcile, flag any discrepancy vs my reported numbers' },
  ],
}

const CTX = `Turnstile steering experiment. Apples-to-apples: SAME 40 baseline-refusing JailbreakBench
prompts, SAME baseline. Two steering families compared on BOTH outcomes (compliance % via JBB judge,
harm via Stage-B 1-5 Likert), under two judges (Llama-3.1-70B, Qwen2.5-72B).

Local files (read with python3/Bash; one JSON object per line):
- experiments/refusal_harm_vs_compliance_v1/judged.jsonl          COMPLIANCE direction (refusal), Llama judge
- experiments/refusal_harm_vs_compliance_v1/refusal_judged_qwen.jsonl   same, Qwen judge
- experiments/steer_cvh_matched/judged_llama.jsonl                 HARM direction sweep, Llama judge
- experiments/steer_cvh_matched/judged_qwen.jsonl                  HARM direction sweep, Qwen judge
Row fields: set, direction, alpha, prompt_id, is_refusal, judge_compliance_unsafe (bool), judge_harm_likert (1-5).
The compliance files use direction="refusal", alpha in [-1.5..1.5]. The harm files use
direction in {baseline, harm_dm_llama, harm_dm_resid, random_1}, alpha in {-1.5,-1,-0.5,.5,1,1.5} (baseline alpha=0).

MY REPORTED CLAIMS (verify or refute each, with the actual numbers):
C1. Baselines MATCH: at alpha=0 both datasets show compliance ~2.5% and (harm dataset) harm ~1.07 (Llama).
C2. COMPLIANCE direction is a real lever: subtracting refusal (alpha<0) raises compliance from ~2.5% up to
    ~62% (peak ~alpha=-1.0), and drags harm up to a CEILING of only ~2.3-2.5 (never reaches "meaningful uplift"=4).
C3. HARM direction is NOT a lever: adding harm_dm_llama / harm_dm_resid does not raise harm to meaningful uplift;
    at large |alpha| it COLLAPSES harm toward 1.0; random behaves like the harm dirs. Both judges agree.
C4. The comparison is genuinely apples-to-apples (same prompts, same baseline, same axes) with no confound
    that would unfairly favor the compliance lever.`

const DERIVE_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    claim: { type: 'string' },
    verdict: { type: 'string', enum: ['confirmed', 'refuted', 'partly', 'cannot-compute'] },
    numbers: { type: 'string', description: 'the actual per-alpha numbers you computed (compact)' },
    discrepancy: { type: 'string', description: 'any mismatch vs my reported claim, or "none"' },
  },
  required: ['claim', 'verdict', 'numbers', 'discrepancy'],
}

phase('Re-derive')
const checks = [
  { id: 'C1-baseline', task: 'Compute alpha=0 compliance% and harm (both judges) in BOTH datasets. Confirm they match (~2.5% comp, ~1.07 harm). Report exact numbers.' },
  { id: 'C2-compliance-lever', task: 'From the COMPLIANCE (refusal) files, compute compliance% and mean harm at each alpha (both judges). Confirm compliance rises to ~62% and harm peaks at a low ceiling (~2.3-2.5), never 4.' },
  { id: 'C3-harm-null', task: 'From the HARM files, compute compliance% and mean harm vs alpha for harm_dm_llama, harm_dm_resid, and random (both judges). Does ANY harm direction raise harm meaningfully above baseline/ceiling, or does it collapse at large |alpha|? Report every number.' },
]
const derived = await parallel(checks.map(c => () =>
  agent(`${CTX}\n\nTASK (${c.id}): ${c.task}\nRe-derive independently from the raw jsonl with python3. Do NOT trust my numbers — compute your own.`,
    { label: c.id, phase: 'Re-derive', schema: DERIVE_SCHEMA })))

phase('Refute')
const refute = await agent(
  `${CTX}\n\nYou are an adversarial reviewer. Your job is to REFUTE claim C3 ("the harm knob does not turn") and C4 ("apples-to-apples, no confound"). Compute from the raw jsonl. Specifically hunt for: (a) any direction/alpha/judge where the harm direction DOES raise harm meaningfully (>= +0.5 over baseline AND without just being more compliance); (b) any confound making the compliance lever look better than the harm lever unfairly (e.g., the harm baseline arm not matching, alpha grids not comparable, the "harm rises" on the compliance side being an artifact). Default to finding problems. If after real effort you cannot refute, say so explicitly.`,
  { label: 'adversarial-refute', phase: 'Refute',
    schema: { type: 'object', additionalProperties: false,
      properties: {
        refuted: { type: 'boolean' },
        strongest_counterevidence: { type: 'string' },
        confounds_found: { type: 'string' },
        verdict: { type: 'string' },
      }, required: ['refuted', 'strongest_counterevidence', 'confounds_found', 'verdict'] } })

phase('Synthesize')
const synth = await agent(
  `${CTX}\n\nRe-derivation results:\n${JSON.stringify(derived.filter(Boolean), null, 1)}\n\nAdversarial refutation attempt:\n${JSON.stringify(refute, null, 1)}\n\nWrite a tight verdict for the researcher: (1) do the numbers reproduce my claims C1-C4? List any DISCREPANCY with the exact number. (2) Did the adversarial pass find any real counterevidence or confound? (3) Bottom line: is the apples-to-apples figure's headline ("compliance knob turns, harm knob does not") safe to put in the paper, and with what caveats (e.g., the small residualized bump)?`,
  { label: 'synthesis', phase: 'Synthesize' })

return { derived: derived.filter(Boolean), refute, synth }
