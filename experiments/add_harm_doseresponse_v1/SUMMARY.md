# ADD-harm dose-response (add_harm_doseresponse_v1)

Resolves the tension in fig4 ("ADD-harm ≈ 0") vs the earlier win-set finding of a
+0.40 directional harm signal. Sweeps the ADD magnitude from 0 into the degradation
regime for the harm direction AND a random direction, on 45 low-uplift compliant
replies (Stage-B rating≤3), tracking uplift + coherence. Magnitude = mult × raw
diff-in-means norm (1.385); residual-stream norm at L16 ≈ 9.71.

| push ‖·‖ | harm uplift | random uplift | coherence (harm) | coherence (random) |
|---|---|---|---|---|
| 0 (1.0×) | 2.24 | — | 0.47 | — |
| 1.4 (1×) | 2.27 | 2.04 | 0.47 | 0.48 |
| 2.8 (2×) | 2.33 | 2.24 | 0.47 | 0.48 |
| 5.5 (4×) | 2.20 | 2.02 | 0.41 | 0.49 |
| 11.1 (8×) | 2.16 | 2.16 | 0.33 | 0.45 |

**Result: harm uplift never rises with push strength.** It stays ≈ baseline (2.24),
peaks a trivial +0.09 at ‖·‖=2.8, then *falls* (2.16 at ‖·‖=11) as coherence collapses
(unique-token ratio 0.47→0.33). The harm direction is marginally more harm-flavoured than
a random direction at low magnitude (+0.1–0.2) but manufactures no uplift at any magnitude.
Contrast: French flips 0→98% at ‖·‖=2.3 with no degradation.

**Corrects an earlier hypothesis.** I had guessed harm rose ~linearly with magnitude
(so fig4's +0.11 at low magnitude and the win-set +0.40 at high magnitude would be one
effect). The dose-response *refutes* that: on low-uplift replies harm is flat vs
magnitude. The +0.40 was **set-specific**, not magnitude-specific — it was measured on
high-harm WIN replies that already contained harmful content for the direction to
amplify. On these low-uplift replies there is nothing to amplify, so no magnitude helps.

**Sharper statement of "capability, not representation":** steering the harm direction
can slightly *amplify* harmful content where it already exists, but cannot *manufacture*
it where the model isn't providing it — at any magnitude, up to and past a full
residual-stream norm (where the model just degrades). See fig5_harm_doseresponse.png.
