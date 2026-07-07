# Does the harm direction steer compliance? (steer_compare_v1)

The headroom test the matrix couldn't give: steer {refusal, harm, random} on 30
JailbreakBench **harmful single-turn prompts** (baseline **0%** compliance — full
headroom, the regime where the refusal direction shows its 2→62% jump), α grid at
α×7.56=‖added vector‖, judged by Llama-70B for compliance + harm + coherence.

## Compliance % (70B judge) on harmful prompts

| α | refusal | harm | random |
|---|---|---|---|
| −1.5 | 43% | 0% | 33% |
| −0.75 | **60%** | 0% | 23% |
| 0 | 0% | 0% | 0% |
| +0.75 | 0% | **30%** | 0% |
| +1.5 | 0% | 7% | 3% |

At face value the harm direction looks like it bumps compliance to 30% at α=+0.75.
**But the coherence control kills it:**

| dir | α | comp% | harm | coherence |
|---|---|---|---|---|
| harm | −0.75 | 0% | 1.03 | **0.94** (clean, still refusing) |
| harm | +0.75 | 30% | 1.63 | **0.32** (degraded) |
| harm | +1.5 | 7% | 1.00 | **0.09** (gibberish) |
| random | −1.5 | 33% | 1.47 | 0.45 (degraded) |
| refusal | −0.75 | 60% | 1.93 | 0.55 |

## Conclusion

**The harm direction steers neither harm nor compliance.** At every coherence-preserving
magnitude it leaves compliance at 0% (the model keeps refusing); its only "compliance"
appears at α≥+0.75 where the output has collapsed to gibberish (coherence 0.32→0.09), and
that is the known **degradation false-positive** — the JBB judge miscounts broken
non-refusals as unsafe, the exact artifact the *random* direction also produces on its
degraded side (33% at coherence 0.45). The **refusal** direction is the only genuine
compliance knob (opens the gate at lower magnitude; Arditi-confirmed causal elsewhere).

So harm-uplift is not add-steerable, and the harm direction is not a back-door compliance
knob either — pushing it just breaks the model. See fig9_harm_vs_compliance.png.

Caveat: the benign arm is uninformative here — "compliance" is JBB-unsafe, which is 0% for
benign prompts by construction, so it can't show refusal *induction* on benign.
