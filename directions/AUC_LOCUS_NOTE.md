# AUC / locus note (advisor correction, 2026-07)

The Phase-0 "harm AUC 0.94 (maximally readable)" was an **overclaim from a locus (and
corpus) mismatch**. Corrected understanding:

## What 0.94 actually is
- **OUTPUT locus** (mean-pooled RESPONSE tokens; verified `build_directions.py:96`,
  the same locus `harm_dm` is built) — this is the paper's own aside: "read the last
  token of the response and both labels jump to ~95%."
- **ALL-TURNS** (wins + losses). Including the 311 losses lets the probe use the
  comply/refuse signal, inflating it further.
- CV is clean (GroupKFold, out-of-fold; harm grouped by goal, compliance by
  conversation) — so 0.94 is a genuine out-of-fold number, just at the wrong locus/corpus.

## Numbers from our pipeline (HS=17 = layer-16 output, goal-grouped)
| probe | locus | corpus | Llama | Qwen |
|---|---|---|---|---|
| harm | OUTPUT | all-turns (Phase-0) | 0.942 | 0.939 |
| harm | OUTPUT | wins-only (n=289) | 0.887 | 0.892 |
| harm | INPUT (pre-response) | wins-only (n=289) | 0.856 | 0.833 |
| compliance | INPUT | all-turns | 0.813 | — |
| compliance | OUTPUT | all-turns | 0.924 | — |

## Why we do NOT report the input-locus harm number (0.85)
Our quick input-locus probe (0.856/0.833) does **not** reproduce Fig 5's published
pre-response harm (~0.70 Llama / high-0.70s Qwen), and it does not even reproduce
Fig 5's judge-instability *direction* (we get Llama ≈ Qwen; Fig 5 has Qwen > Llama by
0.05–0.10). So our inventory's "input" probe differs from Fig 5's careful probe (token
position, layer-index convention, preprocessing, or corpus rows). **Fig 5 is
authoritative for pre-response readability; we defer to it and do NOT substitute our
number.**

## What to say in the section (two distinct, never-substituted quantities)
- **Pre-response readability (Fig-5 locus, the paper-carrying claim):** harm mid-0.7s
  (Llama ~0.70, Qwen high-0.70s), judge-unstable. Cite Fig 5; do not recompute.
- **Output-locus separability (the steering locus):** harmful vs benign responses are
  trivially separable (~0.89 wins-only / ~0.94 all-turns) yet the separating direction
  injects nothing — **SEPARABLE-YET-NOT-INJECTABLE** (separability ≠ steerability).

Fig 5 is untouched.
