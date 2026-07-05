# Causal follow-ups to the steering review

Two necessity tests of the claim that the L16 linear **compliance direction** is
a causal handle on compliance behavior (as opposed to a correlational readout).
Addition-steering could not establish this — a norm-matched random direction
moved ASR as much as the compliance direction (symmetric U-shape = norm effect).

## Experiment 2 — trained obliteration (STRONG, from existing `asr_eval_v1`)

The `compliance_obliteration_v1` LoRA drives the L16 linear compliance probe
AUC 0.75 -> 0.50 (and the nonlinear audit MLP to ~0.56) while preserving
behavior. Evaluated on the trainer's held-out (source, round) split, local
dual judge, n=250 paired per-turn examples:

| metric | base | obliterated |
|---|---|---|
| ASR | 28.8% | 30.8% |
| CE/token of corpus response | 0.308 | 0.184 |

- ASR change +2.0pp, McNemar p=0.46 → **statistically unchanged**.
- On originally-refused turns (label 0): ASR 3.4% -> 11.2% (complies *more*).
- Behavior preserved (CE improved).

**Removing the linear compliance signal does not remove the behavior.** The
decodable direction is a correlational readout, not a causal control knob.

## Experiment 1 — inference-time projection-ablation (SUGGESTIVE, weak +control)

True directional ablation `h -= (h·d̂)d̂` at every layer/position (not the
additive intervention the `clamp_v1` "suppress" arms secretly used). Single-turn,
100 JBB direct requests, greedy Llama-3.1-8B-Instruct, compliance judged by
local 70B.

Geometry (L16, unit directions):
- cos(refusal, lr_comp) = **-0.002**  (paper's compliance dir ⟂ Arditi refusal dir)
- cos(refusal, md_comp) = +0.017
- cos(lr_comp, md_comp) = **+0.145**  (the two "compliance" dirs barely agree)

| condition | ASR | Δ | McNemar vs baseline |
|---|---|---|---|
| baseline | 4% | — | — |
| ablate_random | 4% | +0 | ns (control OK) |
| ablate_refusal (Arditi) | 8% | +4 | p=0.22 ns |
| ablate_lr_comp | 4% | +0 | p=1.0 ns |
| ablate_md_comp | 11% | +7 | p=0.023 sig |

**Caveat: the positive control is weak.** Ablating the Arditi refusal direction
only moved ASR 4%→8% (ns), where Arditi et al. report near-0 → 60-90% on
Llama-family models. So this single-direction ablation is too weak to firmly
establish necessity on its own — likely because (a) direction fit on 20 pairs at
one layer, ablated as a single vector (vs Arditi's per-component ablation), and
(b) single-turn is off-distribution for a direction fit on multi-turn breaches.
Read Exp 1 as corroborating (lr_comp non-causal + orthogonal to refusal), with
Exp 2 as the dispositive result.

## Combined takeaway

Neither adding the compliance direction (steering ≈ random) nor removing it
(obliteration: ASR unchanged; ablation: lr_comp no effect) controls compliance
behavior direction-specifically. The paper's "compliance is a steerable/causal
direction" framing is not supported; "refusal is a shallow, perturbation-fragile
gate and the linear compliance probe is a correlational readout" is.

Stronger follow-up if a clean inference-time necessity test is wanted: Arditi's
full per-component ablation with a working positive control, in the multi-turn
regime.
