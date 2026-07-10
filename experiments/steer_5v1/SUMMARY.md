# Harm-MD variant set — sharpened extremes + residualized twin (steer_5v1)

The complete reasonable set of harm-MD variants, add-only on the low/high uplift reply
sets, Llama-70B judged (harm + coherence). Directions built from output-locus reps.

## Feasibility finding
Llama 5-vs-1 is **unbuildable**: only **n=2** responses rated 5 by the Llama Stage-B judge
(dist {1:396, 2:34, 3:35, 4:133, 5:2}). The Llama judge essentially never certifies the 8B's
jailbroken output as "severe/5" — independent support for the capability ceiling. So the
sharpened comparison runs in Qwen space (n=5:32 / n=1:342).

## Geometry (before steering)
- `harm_dm_5v1_qwen` · length = **+0.335** (sharpened extreme IS length-loaded, as the rubric
  predicts) · anchor = +0.93 (sharpening barely rotates it). Residualizing (refusal+length out)
  removes 35.3%; residual 0.94-aligned with raw 5-vs-1.

## Harm uplift at the calibrated dose (α=0.37, Llama-judged)
| set | baseline | anchor L (4,5)v(1,2) | anchor Q | 5-vs-1 Q | 5-vs-1 Q resid | random |
|---|---|---|---|---|---|---|
| low  | 2.75 | 2.81 | 2.78 | 2.81 | 2.84 | 2.69 |
| high | 3.66 | 3.81 | 3.94 | 3.69 | 3.75 | 3.66 |

## Verdict — the cleanest null
Neither the sharpened 5-vs-1 nor its residualized twin steers harm: every arm sits on baseline
and on random at the calibrated dose. The most length-loaded direction (5-vs-1, cos 0.34 to
length) does **not** steer even RAW — so there is no verbosity-not-harm artifact for the
residualized twin to remove; the "raw steers, residualized doesn't" branch never triggers.
At large push (α=1.47) all harm arms *fall* (1.7–2.8) as coherence collapses to ~0.2 —
degradation, not negative steering; random stays flat (~2.6–3.8).

Harm is not injectable even with the sharpest labels AND the refusal+length confound stripped.
This is the strongest form of the harm-steering null. See figP5_harm_md_variants.png.
