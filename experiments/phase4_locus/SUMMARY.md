# Phase 4 — locus control (does OUTPUT-locus compliance steer?)

Deepest caveat: compliance is fit at an INPUT token (last user token), harm on OUTPUTS
(mean-pooled response), so "harm ≠ compliance" is partly confounded with "output-content
directions behave differently from input-gate directions." Control for it directly: build an
OUTPUT-locus compliance direction and see if it steers compliance.

`comp_dm_out` = μ(complied turns) − μ(refused turns), mean-pooled response reps (replay corpus).
Phase 0 showed it is 86% length, so also built `comp_dm_out_resid` (length projected out, 51%
of the raw direction remains). Steered both + random on 30 harmful single-turn prompts,
two-sided α, Llama-70B judged. Overlaid the INPUT-locus refusal diff-in-means as the reference.

Peak COHERENT compliance (coherence > 0.75):
| direction | peak coherent compliance |
|---|---|
| refusal_dm (INPUT gate, reference) | ~60% (steers) |
| comp_dm_out (output) | 17% |
| comp_dm_out_resid (output, length-free) | 7% |
| random | 27% |

**Verdict (honest boundary):** output-locus compliance does NOT steer compliance — neither
direction beats random in the coherent regime; their 37–90% spikes are all at coherence
0.20–0.56 (degradation false-positives). So the steerable/unsteerable split is **INPUT-GATE vs
OUTPUT-CONTENT, not harm vs compliance.** The comply/refuse decision is made upstream at the
input (the refusal gate is the only real lever); directions read off the OUTPUT — whether harm
OR complied-vs-refused content — describe the consequence and don't causally control it.

Consequence for the paper: do NOT claim "harm is uniquely unsteerable vs compliance." The
precise, defensible claim is SEPARABLE-YET-NOT-INJECTABLE: at the OUTPUT locus we steer at,
harmful vs benign responses are trivially separable (~0.94 harm, ~0.92 output-compliance — this
is OUTPUT-locus separability, the paper's "~95% at last response token" aside, NOT the
pre-response mid-0.7s readability of Fig 5), yet adding the separating direction injects nothing.
Output-content directions in general (incl. compliance) don't steer; only the input refusal gate
does. Note also: the below-random output-compliance numbers (7% vs random 27%) and the ~27%
"random compliance" are the degradation/false-positive band — nothing at the output locus is a
clean lever — NOT negative steering. ASR still overstates harm because ASR = the input gate
(trivially flippable) while harm = output capability (not injectable). See figP4. Caveat:
comp_dm_out was fit on multi-turn
replay turns and steered on single-turn prompts (a distribution shift), which may further weaken
it — but the qualitative locus conclusion matches output-locus harm exactly.
