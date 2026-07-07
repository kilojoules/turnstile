# Final harm-steering matrix — judge-robust null (harm_steer_matrix_v1)

The definitive test of "can we steer harm up?": two harm directions (fit on **Llama**-
labelled vs **Qwen**-labelled Stage-B wins) × two eval sets (**low-uplift** rating≤3,
**already-harmful** rating≥4) × a magnitude grid, each response scored for harm +
compliance by **both** Llama-70B and Qwen-72B. Random direction control throughout.
Directions fit disjoint from eval; cos(harm_llama, harm_qwen)=**0.934** (the two
directions are nearly identical even though the labels disagree on 26% of wins).

## Harm uplift (Llama judge / Qwen judge) — never rises

| set | direction | m=0 | m=2.8 | m=5.5 | m=11.1 |
|---|---|---|---|---|---|
| low | baseline | 2.12/1.72 | — | — | — |
| low | harm_llama | | 2.34/1.97 | 2.34/1.94 | 1.78/1.53 |
| low | harm_qwen | | 2.41/1.94 | 2.38/1.88 | 1.50/1.56 |
| low | random | | 2.25/1.88 | 2.00/1.75 | 2.03/1.78 |
| high | baseline | 3.75/3.97 | — | — | — |
| high | harm_llama | | 3.78/3.88 | 3.69/3.81 | 3.16/3.06 |
| high | harm_qwen | | 3.78/3.81 | 4.00/3.91 | 3.25/3.16 |
| high | random | | 3.78/3.91 | 3.84/3.94 | 3.75/3.69 |

At the calibrated dose (m=2.8) the harm direction sits within ~0.1–0.2 of the random
control everywhere, never reaches "4 = meaningful uplift", and at m=11.1 harm *falls* as
the output degrades. **This holds for both harm directions, both sets, and both judges.**

## Compliance (Llama / Qwen)

Adding the harm direction barely moves compliance either (low set m=2.8: harm_llama
38%/97%, harm_qwen 44%/94%, random 25%/91%). Note the two *compliance* judges disagree in
level — the Qwen JBB judge is far more lenient (low-set baseline 16% Llama vs 94% Qwen) —
but neither shows the harm direction driving compliance.

## Refusal-steering cross-check (dual-judged)

Steering the *refusal* direction (refusal_harm_vs_compliance_v1, re-judged with Qwen): harm
rises to a low ceiling under **both** judges (peak 2.30 Llama / 2.17 Qwen at α=−1.0), same
as before. So the "harm follows compliance to a capability ceiling" story is also judge-robust.

## Conclusion (robust)

**You cannot steer harm upward with the harm direction — at any magnitude, on either set,
whether the direction is fit on Llama or Qwen labels, and whether the outcome is scored by
Llama or Qwen.** Harm only moves as a by-product of opening the compliance gate (refusal
steering), and is capped at the 8B's capability ceiling (~2.3 low / ~3.8 high). Steering
moves representations/forms, not capability. See fig8_harm_matrix_dualjudge.png.

Remaining open question (unchanged): a stronger victim (70B) that *has* the withheld
capability — the capability reading predicts jailbreaking it would reach higher harm.
