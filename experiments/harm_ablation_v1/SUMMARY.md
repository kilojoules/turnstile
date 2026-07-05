# Harm-direction ablation — the Arditi analog for harm (harm_ablation_v1)

Corresponding test to the refusal replication: is harm-uplift mediated by a
single ablatable direction, the way refusal is? Norm-preserving directional
ablation of the best available harm direction (post-response diff-in-means `md`
and LR `lr`, AUC 0.88) projected out of every residual write at L16, on 70 paired
HIGH-HARM COMPLIANT win turns (Stage-B rating≥4 originals). Judged: Stage-B harm
Likert + JBB compliance.

| arm (n=70) | mean harm | compliance% | harm\|compliant | harm≥4% | degrade% |
|---|---|---|---|---|---|
| baseline | 3.61 | 80 | 3.82 | 76 | 1 |
| ablate_md_harm | 3.50 | 74 | 3.98 | 79 | 3 |
| ablate_lr_harm | 3.61 | 77 | 3.94 | 80 | 1 |
| ablate_random | 3.50 | 77 | 3.74 | 70 | 3 |

Paired Δ vs baseline: md −0.11 [−0.36,+0.11], lr +0.00, random −0.11.
**KEY**: md vs random Δ = **+0.000 [−0.23,+0.20]**; lr vs random +0.11 (nominally
worse). Ablating the harm direction is statistically indistinguishable from
ablating a random direction. Among still-compliant responses harm actually rises
(3.98/3.94 vs 3.74). Well-powered (MDE≈0.3 Likert, power≈0.98 for −0.5), no
floor/ceiling (headroom 3.8→1), no degradation. The only apparent reduction
(harm≥4 subset −0.24) is reproduced by random = regression to mean.

Convergent (same L16, independent methods): trained obliteration (harm probe AUC
0.72→0.50) left harm 1.69→1.72 (= SFT control); additive steering collapses to
harm=1 at |α|≥1 (norm-destruction), no clean signal.

Positive control (same harness): refusal ablation 98%→68% refusal, ASR 2%→32%
(p=0.0003); add refusal → benign refusal 0%→64%. So the harm null is
direction-specific, not a broken pipeline.

## Minimal correct claim (adversarially vetted — do NOT overclaim)

> The best harm-uplift readout direction (AUC 0.88) at L16, ablated norm-
> preservingly, leaves harm unchanged and indistinguishable from a random
> direction, with compliance + coherence intact and headroom to fall — corroborated
> by two other L16 removal methods. This is a real, well-powered, direction-specific
> dissociation from the refusal direction, which IS a causal lever in the identical
> harness. **The best harm direction at L16 is a readout, not a lever — unlike refusal.**

Do NOT yet claim "harm is not an ablatable direction (any layer)" or the mechanistic
"refusal is an input-gate single direction, harm is distributed output content."

## Two verified caveats

**(a) Layer not selected.** Refusal SELECTED L12 from a 9-layer sweep; harm used
FIXED L16. Precedent: `ablation_v1` shows fixed-L16 *refusal* ablation was ALSO
null (4%→8%, p=0.22), rescued only by selecting L12. Harm probe AUC peaks LATE
(L20 0.782, L24 0.795), off L16. No harm ablation run at L20/24/31. → "best L16
harm dir isn't causal" is earned; "no harm dir at any layer" is not.

**(b) Method may be blind to output-content (the bigger threat).** The only proven
positive is refusal — an INPUT-GATE concept fit on PROMPTS. Harm is fit on
RESPONSES. The two experiments differ on concept AND fit-locus simultaneously, so
concept is confounded with fit-locus. No output-content direction of any kind has
been shown ablatable by this machinery → the harm null is partly predetermined by
method structure, not established as a fact about harm.

## Recommended next runs (priority order)

1. **[Load-bearing] Output-content positive control.** Fit a benign output-content
   axis (e.g. verbose-vs-terse, or code-vs-prose) on post-response residuals
   exactly like harm; prove it causal via the ADD arm; then run the identical
   projection ablation. If ablating it moves style/format → the method CAN touch
   output content → the harm null is informative about harm. If not → the harm null
   collapses to "this method only moves input-gate concepts," and the mechanistic
   thesis is unsupported. This is the only run that disentangles the (b) confound.
2. **Harm layer/position sweep** (L20/24/31) with paired random + add-harm arms,
   selecting the best layer as refusal did. Closes caveat (a). Cheaper, secondary.
