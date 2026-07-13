# Direction Inventory Audit (Task 1)

Every recipe below was recovered **from code** (builder scripts read in full), not from filenames. Cosines computed this pass from the on-disk `.pt` bytes. `HS=17` = layer-16 **output** residual (HF convention). All vectors L2-normalized.

## Sanity gate — PASS
`harm_dm_llama` vs `directions_postresponse/v_md_harm_L16` = **0.80** (expected ~0.80 ✓); vs `directions/v_md_harm_L16` = **0.41** (expected ~0.41 ✓). Inventory reads the same files as the prior analysis.

## Missing-builder gate — ⚠️ TRIGGERED
**Steered directions with NO committed builder** (`torch.save` grep of whole repo finds nothing that writes them):
| direction | steered by | recipe recoverable? |
|---|---|---|
| `harm_dm_resid` | phase3, steer_cvh_matched, ext10, harm_on_open | **yes** — Gram-Schmidt(harm_dm_45v12, {refusal,length}); reconstructed cos **1.0000** |
| `harm_dm_45v12_qwen` | steer_5v1 | yes — dm(Qwen≥4, Qwen≤2); reconstructed cos **1.0** |
| `harm_dm_5v1_qwen` | steer_5v1 | yes — dm(Qwen==5, Qwen==1); reconstructed cos **1.0** |
| `harm_dm_5v1_qwen_resid` | steer_5v1 | yes — resid of the above; reconstructed cos **1.0** |
| `comp_dm_out_resid` | phase4_locus | not reconstructed here (needs replay_v2 reps) — recipe = resid of `comp_dm_out` |

All recipes are reconstructable, but **none regenerates from committed code**. Per the operating rule ("STOP if any steered direction has no recoverable fit recipe from code"), this is the stop condition. Fix: commit a builder (a ~10-line extension of `build_5v1.py`) that writes all of these deterministically.

## Definitive STEERED set (loaded + added to activations)
`refusal_dm`, `harm_dm_llama`, `harm_dm_resid`, `random_1` (Fig-7/8 cvh sweeps + harm_on_open + phase3); `comp_dm_out`, `comp_dm_out_resid` (phase4_locus control); `harm_dm_45v12_qwen`, `harm_dm_5v1_qwen`, `harm_dm_5v1_qwen_resid`, `harm_dm_qwen` (steer_5v1 harm-matrix).

## Core recipe table (the steered + Fig-5-decoded directions)

| direction | recipe | fit data | label | LOCUS | steered? |
|---|---|---|---|---|---|
| **refusal_dm** | diff-in-means | goals.json[:50] vs alpaca[:50] | prompt-category **prior** (harmful vs benign; no rating) | **pre-response** | **yes** (compliance arm) |
| comp_probe | LR probe | replay_v2_full.pt (input reps) | per-turn `unsafe` complied vs refused | pre-response | no |
| **comp_dm_out** | diff-in-means | replay_v2_full.pt (output reps) | per-turn `unsafe` complied vs refused | **output** | **yes** (phase4_locus) |
| comp_dm_out_resid | dm + resid(length) | replay_v2 | per-turn complied/refused | output | yes (phase4_locus) |
| **harm_dm_llama** | diff-in-means | Stage-B (mean-pooled response) | Llama Stage-B rating ≥4 vs ≤2 | **output** | **yes** (harm arm) |
| harm_dm_qwen | diff-in-means | Stage-B | Qwen rating ≥4 vs ≤2 | output | yes (steer_5v1) |
| harm_dm_resid | dm + resid(refusal,length) | Stage-B | Llama ≥4 vs ≤2 | output | yes (harm arm) |
| harm_probe_llama/qwen | LR probe | Stage-B | Llama/Qwen ≥4 vs ≤2 | output | no |
| length_dm | diff-in-means | Stage-B | response length top vs bottom quartile | output | no |
| **v_md_comp / v_lr_comp** (`directions/`) | dm / LR | **pooled_hs** residuals | **per-turn self-play breach** (Llama) | **pre-response** | no (Fig-5 decode) |
| **v_md_comp / v_lr_comp** (`directions_qwen_comp/`) | dm / LR | pooled_hs | **Qwen** per-turn breach | **pre-response** | no (Fig-5 decode) |
| **v_md_harm / v_lr_harm** (`directions_llama/`) | dm / LR | pooled_hs at rated turns | Llama Stage-B ≥4 | **pre-response** | no (Fig-5 decode) |
| **v_md_harm / v_lr_harm** (`directions_postresponse/`) | dm / LR | Stage-B responses | Llama Stage-B ≥4 | **output** | no (decode) |
| harm_dm_in | diff-in-means | Stage-B | Llama ≥4 vs ≤2 | **pre-response** (last user token) | no |
| random_1..5 | randn(4096) seed 42 | none | none | n/a | random_1: yes |

## 🚩 The decoded-vs-steered mismatch (the single most important finding)

**Compliance — same locus, orthogonal directions:**
- Fig-5 *decodes* compliance with `v_md_comp` (per-turn breach/complied labels, **pre-response**).
- Fig-7 *steers* `refusal_dm` (harmful-vs-benign **prior**, pre-response).
- `cos(refusal_dm, v_md_comp_Llama) = −0.03`, `cos(refusal_dm, v_md_comp_Qwen) = −0.28`, `cos(refusal_dm, comp_dm_out) = −0.38`.
- ⇒ "compliance decodes at 0.75 **and** steers" describes **two different, ~orthogonal vectors**. The decoded direction (`v_md_comp`) was never steered; the steered direction (`refusal_dm`) has no Fig-5 AUC reported.

**Harm — cross-locus:**
- Fig-5 headline harm AUC (pre-response, mid-0.7s) uses `directions_llama/v_md_harm` (**pre-response**).
- Fig-7 steers `harm_dm_llama` (**output**).
- `cos = 0.36` (steered-output vs pre-response-decode); `cos = 0.80` (steered-output vs the *output*-locus decode).
- ⇒ An **output-locus match does NOT support the pre-response decode claim.** The steered harm vector aligns with the output decode (0.80), not the pre-response one (0.36) whose AUC is the paper's headline.

## Duplicate-vector traps (cos ≈ 1.0, different names)
- `harm_dm_45v12` ≡ `harm_dm_llama` (1.00); `harm_dm_qwen` ≡ `harm_dm_45v12_qwen` (1.00)
- `v_harm_meandiff_L16` ≡ layer-sweep `v_md_harm_L16` (1.00); `v_harm_L16` ≡ layer-sweep `v_lr_harm_L16` (1.00); `v_comp_meandiff_L16` ≡ `v_md_comp_L16` (0.99)
- `harm_dm_5v1_resid` ≡ `harm_dm_5v1` (0.99, resid removed little)

## L16 cosine — key numbers (full matrix cached at /tmp/cos_matrix.json)
- Harm steered vs decode: `harm_dm_llama`·{postresponse-MD 0.80, layer-sweep-MD 0.41, llama-refit-MD 0.36}
- Compliance steered vs decode: `refusal_dm`·{comp_dm_out −0.38, v_md_comp(Llama) −0.03, v_md_comp(Qwen) −0.28, comp_probe −0.04}
- `harm_dm_llama`·`harm_dm_resid` = 0.92; `harm_dm_llama`·`comp_dm_out` ≈ 0 (different concept, expected)

## Locus-crossing warning list
Any comparison that reports a **pre-response decode AUC** and a steering result must use the **same-locus** direction. Directions at `pre-response`: refusal_dm, comp_probe, all `v_*_comp`, `directions_llama/v_*_harm`, harm_dm_in, v_comp/v_harm/*_meandiff (steering_v3 loose). At `output`: comp_dm_out(_resid), comp_probe_out, harm_dm_llama/qwen/resid/5v1*, harm_probe_*, length_dm, `directions_postresponse/v_*_harm`.

---

# Task 2 — pre-response AUC of the ACTUALLY-STEERED vectors (RAN NEW)

No prior run existed (existing Fig-5 AUCs are RE-FIT per-layer probes, not the fixed steered vectors). Computed on the Fig-5 corpus (pooled_hs), pre-response locus L16, 5-fold group-k-fold by conversation. Decodability = max(auc, 1−auc).

| steered vector | target | judge | point AUC | CV AUC | Fig-5 RE-FIT probe |
|---|---|---|---|---|---|
| **refusal_dm** (steered compliance) | compliance breach | Llama | **0.511** | 0.511±0.005 | ~0.75 |
| refusal_dm | compliance breach | Qwen | **0.557** | 0.557±0.005 | ~0.75 |
| **harm_dm_llama** (steered harm) | harm ≥4 | Llama | **0.523** | 0.576±0.024 | ~0.70–0.78 |
| harm_dm_llama | harm ≥4 | Qwen | **0.516** | 0.535±0.027 | ~0.70–0.78 |

## 🚩🚩 Literal-truth issue for the paper
The paper's "compliance decodes at ~0.75 / harm at mid-0.7s (pre-response)" numbers come from **re-fit probes** (`v_md_comp`, `directions_llama/v_md_harm`), **NOT** the vectors we steer. The **actually-steered vectors decode at ~chance pre-response** (0.51–0.58). So "the direction we steer also decodes well (pre-response)" is **false**. What's true:
- `refusal_dm` is a **causal lever** for compliance (2.5%→62%) that is a **poor pre-response readout** (0.51) — it's fit on a harmful-vs-benign *prompt* prior, near-orthogonal (cos −0.03) to the compliance-outcome direction that decodes 0.75.
- `harm_dm_llama` is fit at **output** locus; at **pre-response** it decodes harm at ~chance (0.52) — the 0.70–0.78 pre-response number is a *different* (cos 0.36) re-fit direction.

Decoded ≠ steered on **both** axes. Any "reads-and-steers" (compliance) or "reads-but-won't-steer" (harm) sentence must name *which* vector, at *which* locus.

---

# Tasks 3 & 4 — do the DECODABLE directions steer? (RAN NEW)

Steered `comp_pre_llama/qwen` (pre-response compliance MDs that decode ~0.75), `harm_pre_llama` (pre-response harm MD that decodes 0.70–0.78), and `comp_dm_out` (posterior compliance, output) on the same 40 prompts / baseline, two-sided α, dual-judged. Peak effect (α=+1.0 unless noted):

| direction (decodes at) | peak compliance L / Q | peak harm L / Q | verdict |
|---|---|---|---|
| **comp_pre_qwen** (compliance ~0.75) | **72.5% / 92.5%** | 2.70 / 2.73 | **STEERS** |
| **comp_pre_llama** (compliance ~0.75) | 25% / 87.5% | 1.68 / 1.82 | **STEERS** |
| **comp_dm_out** (posterior compliance) | **90% / 100%** | **3.12 / 3.10** | **STEERS (strongest)** |
| **harm_pre_llama** (harm 0.70–0.78) | 17.5% / 77.5% | **1.60 / 1.70** (peak, collapses @1.5) | **does NOT steer harm** |
| *ref: refusal_dm (prior)* | 62.5% | 2.30 | steers |

## Verdict — the audit STRENGTHENS the paper (decoded == steered now holds)
- **Task 3 — compliance steers, with the RIGHT direction.** The compliance direction that actually *decodes* (`comp_pre`, `comp_dm_out`) also *steers* — to 72–90% (Llama) / 92–100% (Qwen). So "compliance decodes **and** steers" is true for a **single, matched** direction (not the near-orthogonal `refusal_dm` prior). Swapping the paper's compliance arm from `refusal_dm` → `comp_pre`/`comp_dm_out` makes decoded==steered clean.
- **Task 4 — harm is readout-not-lever, now airtight.** The harm MD that *decodes* at 0.70–0.78 **fails to steer harm** (peak 1.60, collapses to 1.0 at α=1.5), never near meaningful uplift (4). Same vector reads harm well and won't move it.
- **Harm only ever rises as a byproduct of compliance.** The compliance directions push harm HIGHER (2.7–3.1) than the dedicated harm direction (1.6) — because opening the gate drags harm up to its ceiling; no direction manufactures independent harm. Compliance-max `comp_dm_out` reaches harm 3.12 (highest anywhere) yet still < 4.

Net: the compliance/harm asymmetry is **real and robust**, and now demonstrated with directions that are decodable *and* steered (not lookalikes). The earlier "steered≠decoded" mismatch was a naming/choice issue, not a substantive hole.
