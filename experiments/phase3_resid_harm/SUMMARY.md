# Phase 3 — confound kill (residualized harm direction)

Objection: "the harm direction is really harm + refusal + length, so of course it's a bad
lever." Killed directly. Built `harm_dm_resid` = harm_dm_llama with its refusal_dm and
length_dm components projected out (Gram–Schmidt), renormalized. **39.3%** of the raw harm
direction was refusal+length; the residual stays 0.92-aligned with raw harm and is now
exactly orthogonal to refusal and length.

Steered harm_dm_resid, harm_dm_llama (ref), random on the low/high uplift reply sets,
add-only α grid, Llama-70B judged.

| set | baseline | harm_dm_resid (α0.37) | harm_dm_llama | random |
|---|---|---|---|---|
| low-uplift | 2.75 | 2.75 | 2.81 | 2.69 |
| already-harmful | 3.66 | 3.88 | 3.81 | 3.66 |

**Verdict:** the pure-harm residual is just as inert as the raw harm direction and as random —
flat at the calibrated dose, falling only at large push as coherence collapses (identically
across all arms). Removing the 39% confound does NOT rescue steering. So harm's unsteerability
is not an artifact of the direction being contaminated by refusal/length. See figP3.
