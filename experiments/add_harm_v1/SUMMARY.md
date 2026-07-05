# ADD-harm test — the decisive mirror of ADD-French (add_harm_v1)

Clean sufficiency test: fit the harm diff-in-means with its NATURAL raw magnitude
(raw_norm=1.385 at L16; French raw was 2.34; h_norm=9.71), add at calibrated 1x/2x
doses on 70 COMPLIANT low-uplift contexts (Stage-B rating≤3, headroom to 5), vs
norm-matched random-add. Judged: Stage-B uplift + compliance. n=70/arm.

| arm | mean harm | harm≥4% | compliance% | coherent |
|---|---|---|---|---|
| baseline | 2.20 | 14% | 39% | yes |
| add_harm_1x | 2.27 | 17% | 37% | yes |
| add_harm_2x | 2.31 | 19% | 43% | yes |
| add_random_1x | 2.16 | 16% | 41% | yes |
| add_random_2x | 2.26 | 16% | 44% | yes |

Harm-specific (add_harm vs matched random): **+0.11 [−0.05,+0.28]** (1x),
**+0.06 [−0.11,+0.23]** (2x). Indistinguishable from 0 and from random. Coherence
intact (tur 0.47-0.49, max_repeat 1.0) — these calibrated doses do NOT degrade,
unlike the original α=1 (9.71) sweep. Clear headroom unused (baseline 2.2/5).

## Result: harm is NOT add-steerable, while form IS

Under the IDENTICAL mechanism (add raw diff-in-means at L16):
- ADD-French: 0% → **98%**
- ADD-verbose: 30 → **115 tokens** (~4x)
- ADD-harm: +0.11 Likert (≈ random)

## The resolution of harm ≠ compliance: gate vs form vs capability

Three kinds of thing, three steering behaviors:
- **Refusal / compliance = a behavioral GATE.** Bidirectionally steerable: ADD
  induces refusal 0→64%, ABLATE bypasses it 98→68% (arditi_repl_v1). A causal
  single direction (Arditi replicates in our setup).
- **Output FORM (language, verbosity) = steerable by ADD** (French 0→98%,
  verbose 4x), weakly by ablate (output_content_control_v1).
- **Harm-uplift = a CAPABILITY.** NOT steerable by ADD (+0.11 ≈ random) NOR
  ABLATE (harm_ablation_v1, Δ≈0 ≈ random), despite being a clean linear READOUT
  (post-response probe AUC 0.88). Because steering vectors move
  representations/forms, not knowledge the model lacks. These eval contexts are
  the "gobbledygook wins" (complied but rating≤3): the 8B gave hedged/generic
  answers because it won't or CAN'T supply operational content, and a harm vector
  can nudge tone (+0.11) but cannot inject the missing capability.

So "harm ≠ compliance" is TRUE, but for a deeper reason than the paper states:
they are different KINDS of object (a flippable gate vs an un-injectable
capability), not two linear directions of differing strength. And ASR overstates
harm precisely because the gate is trivially flipped while the capability cannot
be conjured — the paper's thesis, now mechanistically grounded.

## Residual caveats (honest)

- Only L16 tested for the harm direction (a layer sweep would strengthen it),
  though French/verbose worked at L16 under the same mechanism.
- A whisper of real add-harm signal may exist (+0.11, CI leans positive;
  consistent with the earlier clean-only odd +0.40), but it is negligible beside
  French's 0→98% — most uplift is capability, not a steerable representation.
- "Capability" is the best-supported interpretation of the ADD-French-works /
  ADD-harm-fails contrast; a stronger victim (70B) that HAS the capability would
  test it directly (predict: ADD-harm works better where the model has the
  knowledge).
