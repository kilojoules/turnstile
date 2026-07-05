# Arditi replication with a live positive control (arditi_repl_v1)

Tests whether our setup CAN reproduce Arditi et al. 2024 refusal-direction
steering when we match his method — and whether our compliance-probe direction
(orthogonal to refusal) behaves differently. Llama-3.1-8B-Instruct, bf16,
single-turn, greedy. Direction = harmful(JBB)-vs-harmless(Alpaca) difference-in-
means at the last user token, layer selected by refusal-bypass (l*=L12).
Intervention = norm-preserving directional ablation `h -= (h·r̂)r̂` projected out
of every residual write (embed + every attn + every mlp), all layers/positions.
Metrics: Arditi refusal-substring rate + 70B compliance ASR (unsafe).

Geometry: cos(refusal, lr_comp) = **−0.011**, cos(refusal, random) = +0.001.

| arm (harmful, n=50) | refusal% | ASR (unsafe)% |
|---|---|---|
| baseline | 98% | 2% |
| **ablate refusal dir** | **68%** | **32%** |
| ablate random | 98% | 2% |
| ablate lr_comp (our probe) | 98% | 2% |

| arm (harmless, n=100) | refusal% |
|---|---|
| baseline | 0% |
| **add refusal dir** | **64%** |

- **Positive control FIRES, both halves:** ablating the refusal direction drops
  refusal 98%→68% and raises ASR 2%→32% (McNemar 15 safe→unsafe / 0 unsafe→safe,
  **p=0.0003**); adding it induces refusal on benign prompts 0%→64% (coherent,
  tur=0.99). Bypassed responses are coherent, on-topic harmful compliance
  (gerontocracy-elimination article, sexist email, cryptomining hijack script,
  antivirus-disabling malware) — **surgical, 0/50 degraded** (max_repeat>8 or
  tur<0.35).
- **Our compliance-probe direction is causally inert for refusal:** ablating
  lr_comp = ablating random = baseline (98% refusal, 2% ASR). Consistent with
  cos(refusal, lr_comp) ≈ 0.

## Why this matters

1. **The harness CAN replicate Arditi.** Our earlier failure to steer was a
   method problem (wrong direction, additive not ablation, uncalibrated
   magnitude, no selection), not evidence against Arditi. This retires the dead
   positive control from ablation_v1 (there: refusal ablation 4%→8%, p=0.22 —
   that used a 20-pair L16-only direction and layer-output-only ablation).
2. **Confirms we are silent on Arditi, not contradictory.** Arditi's refusal
   direction is causal here; our compliance probe (orthogonal to it) is not.
   Different axes. This reconciles Exp 2 (obliterating the compliance probe left
   ASR unchanged) with Arditi's causal refusal direction.
3. **The ASR bump (2%→32%) is real and significant but modest vs Arditi's
   headline (~0→~80%)** — attributable to a single 50-pair diff-in-means at one
   selected layer (L12), no post-instruction position selection, greedy decoding,
   and Llama-3.1-8B robustness. The induce-refusal arm (0%→64%) is the cleaner,
   stronger half of the control.

## For the paper

The steering section should state: (a) additive steering of a probe direction at
up-to-full-residual-norm magnitude produces norm-energy artifacts (U / inverted-U,
random-matched); (b) when we run Arditi's actual method (harmful-vs-harmless
diff-in-means, norm-preserving ablation, calibrated magnitude, selected layer),
the refusal direction IS causal and the positive control fires; (c) our
compliance-probe direction is orthogonal to and causally distinct from the
refusal direction. Do NOT claim we refute Arditi.
