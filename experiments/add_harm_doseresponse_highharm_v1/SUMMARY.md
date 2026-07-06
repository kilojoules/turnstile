# ADD-harm dose-response on ALREADY-HARMFUL replies (add_harm_doseresponse_highharm_v1)

Complement to `add_harm_doseresponse_v1` (low-uplift replies). Same magnitude sweep of
the harm direction vs random, but on 45 high-harm compliant replies (Stage-B rating≥4,
baseline 3.7/5, **disjoint from the direction-fit set**) — to test whether the harm
direction *amplifies* uplift where harmful content already exists (predicted from the
earlier +0.40 win-set signal).

| push ‖·‖ | harm uplift | random uplift | harm − random | coherence (harm) |
|---|---|---|---|---|
| 0 (1×) | 3.73 | — | — | 0.48 |
| 1.4 | 3.80 | 3.78 | +0.02 | 0.48 |
| 2.8 | 3.78 | 3.64 | +0.13 | 0.46 |
| 5.5 | 3.78 | 3.80 | −0.02 | 0.42 |
| 11.1 | 3.13 | 3.69 | −0.56 | 0.31 |

**Prediction NOT confirmed.** On already-harmful replies uplift is also **flat** (3.73→
3.80→3.78→3.13): the harm−random gap peaks at a within-noise **+0.13** and then goes
negative at large push as the harm direction degrades output faster than random. There is
no amplification regime — the low-uplift set's gaps (+0.09 to +0.22) are, if anything,
similar or larger, and both are within 95% CIs.

**Correction.** My "the +0.40 was amplification of existing content" reconciliation is
**refuted**. The +0.40 was an "odd" component `(f(+α)−f(−α))/2` from the *confounded*
additive α-sweep (larger magnitude, different judge/set); it does not reproduce as a
clean add-vs-random effect on either set here.

**Net, strengthened conclusion:** adding the harm direction fails to raise uplift in any
regime — low-uplift or already-harmful, small push or large — full stop. Harm is not
add-steerable; compliance/refusal (a gate) is. See fig5_harm_doseresponse.png (both sets).
