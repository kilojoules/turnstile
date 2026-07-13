# Data Provenance — Turnstile steering/probe results

Every number here was computed from the raw repo files in a single audit pass (commands + outputs in the session transcript). Nothing is quoted from memory. Items that cannot be traced to a repo file/line are listed explicitly at the end.

**Five distinct datasets** (the "reply sets" and "steer_5v1/phase3" are the *same* data — proven below — so they are one section; the "probe corpus" splits into a compliance corpus and a harm corpus with different units, so they are two sections):

| # | Dataset | Builder + selection (verbatim) | Funnel (numbers) | Turn structure | Unit / N | Baseline (fresh, this pass) | Judges / threshold | Consumed by |
|---|---------|-------------------------------|------------------|----------------|----------|-----------------------------|--------------------|-------------|
| 1 | **Compliance probe corpus** (pooled_hs) | `scripts/launch_pooled_sweep.sh`; extracts L×T hidden states from the 11 self-play sources' `experiments/{src}/rounds/round_*.jsonl`. No filter — full pool. | 11 sources → **9,400 conversations**; per-turn rows for the compliance probe = **42,010 turns**, positives (breach turns) = **2,027**. | Multi-turn self-play (≤5 attacker/victim turns). | **Per-turn residual row.** N=42,010 turns / 9,400 convs. | conv-level ASR (unsafe) = **22.0%** (2065/9400). | Llama-3.1-70B JBB per-turn (breach label); Qwen per-turn for κ. Binary. | **Fig 4** (per-category ROC, pooled AUC **0.746**); compliance curve of **Fig 5**; **Table 1** compliance κ (via `per_turn_judge_v2_full`). |
| 2 | **Harm / Stage-B corpus** (600) | `working/uplift/select_stage_b.py`; `RNG = random.Random(0)` (**seed 0**); per behavior `picked_w = RNG.sample(wins, min(len,3))`, `picked_l = RNG.sample(losses, min(len,3))` (`wins`=unsafe, `losses`=not-unsafe). | 100 goals × 6 (3 win + 3 loss) → **600 conversations**, 100 behaviors, 6/behavior. | Multi-turn (underlying conversation = 10 roles = 5 user + 5 assistant). | **Conversation.** N=600 (`stage_b_candidates.jsonl`); scored rows `stage_b_scores_llama.jsonl` = 600. | (labels, not a steering baseline) `unsafe`/`rating` per conv. | Stage-B Likert 1–5, **both** judges (`stage_b_scores_llama` = Llama rating + `qwen_rating`). Probe/κ use **≥4 binary**; Fig 6 uses full Likert. | harm curve of **Fig 5** (`metadata_llama.json` ← `stage_b_scores_llama.jsonl`); **Fig 6** (prior vs posterior); **Table 1** harm κ. |
| 3 | **Reply sets** (low / high) — the harm-steering figure | `scripts/steer_5v1.py:43-58` `load_reply_sets(n_fit=70,n_eval=32)`; joins `stage_b_candidates.jsonl` + `stage_b_scores_llama.jsonl`; `fit = first 70 rating≥4 ∪ first 70 rating≤2`; `low = [unsafe & rating≤3 & key∉fit][:32]`; `high = [unsafe & rating≥4 & key∉fit][:32]`. Positional `[:32]`; randomness only from the seed-0 Stage-B corpus upstream. | 600 candidates → join ratings → exclude 140 fit → **low=32, high=32**. | Multi-turn: each row is the victim's breach-turn reply with `history = conversation[:breach]`. | **Reply (breach-turn response).** N=32 low + 32 high. | **low**: refusal **3.1%**, compliance **18.8%**, harm **2.75**. **high**: refusal **9.4%**, compliance **75.0%**, harm **3.66**. (steer_5v1/steer_refusal_replies, direction=baseline, Llama.) | Llama-3.1-70B + Qwen-72B (JBB compliance + Stage-B harm). | **`fig_compliance_vs_harm_steering`** (harm-steering panel). `steer_5v1`, `phase3_resid_harm`, `steer_refusal_replies` are three steering runs on the **same** set. |
| 4 | **40-prompt set** — matched fig + harm-on-open | `scripts/steer_dir_on_harmful.py:55-62`: `n_skip=50, n_eval=40`, `h_eval = goals[args.n_skip:args.n_skip+args.n_eval]` = **`goals[50:90]`** of `working/uplift/goals.json`. **Positional slice, no refusal/harm predicate, no seed.** | `goals.json` (100) → slice `[50:90]` → **40**. No items discarded by any predicate. | **Single-turn** (`gen` builds `[{"role":"user","content":goal}]`; no assistant history). | **Response** (prompt × α). N: `refusal_harm_vs_compliance_v1`=360 (40×9α); `steer_cvh_matched`=760 (40 × [1 baseline + 3 dir × 6α]); `steer_harm_on_open`=560 (40 × 14 arms). | refusal **97.5%**, compliance **2.5% (Llama) / 5.0% (Qwen)**, harm **1.075**. | Llama-3.1-70B + Qwen-72B (JBB compliance + Stage-B harm). | **`fig_cvh_matched`**, **`fig_harm_on_open`**, left panel of **`fig_compliance_vs_harm_steering_v2`**. |

**Identity proofs (md5 of goal strings — note the two conventions differ; stated so hashes are reproducible):**
- 40-prompt set — md5 over goals in **positional (slice / prompt_id) order**, `'\n'.join(goals[50:90])`: `refusal_harm_vs_compliance_v1` = `steer_cvh_matched` = `steer_harm_on_open` = **`76f3c7414dea`**. (The *sorted*-order hash of the same 40 is `e191263d…`, a different value — the reproducible one is positional.)
- Reply sets — md5 over `goal[:200]` **sorted ascending**, `'\n'.join(sorted(...))`: low set `steer_5v1` = `phase3_resid_harm` = `steer_refusal_replies` = **`a90c16531bb2`** (32); high set = **`7c64c24536a0`** (32).

**Reconciliation note (probe corpus counts):** conv-level unsafe = **2,065**, but the per-turn probe's positive count (Fig 4) = **2,027** — a 38-conv gap because `plot_per_category_roc.py` drops breached conversations whose `turn_of_breach` is `None` (a positive is emitted only at the breach turn). Both numbers are correct for their respective units; neither contradicts a caption.

**Raw-field note:** the "compliance direction" file (`refusal_harm_vs_compliance_v1`) stores `direction="refusal"` in every row — "compliance direction" is the paper's framing (subtracting refusal ⇒ toward compliance), not a field value. Grepping `direction=="compliance"` returns 0 rows.

---

## A. Why do the datasets show different steering results? (regime = baseline gate state)

Side-by-side baseline **refusal** rates (the gate state), computed fresh this pass:

| dataset | turn structure | baseline refusal | baseline compliance | gate |
|---|---|---|---|---|
| 40-prompt (single-turn JBB asks) | single-turn | **97.5%** | 2.5% (L) | **CLOSED** |
| reply set — low (multi-turn breach) | multi-turn | **3.1%** | 18.8% | **OPEN** |
| reply set — high (multi-turn breach) | multi-turn | 9.4% | 75.0% | OPEN |

The regime split holds: the single-turn 40-prompt set baseline-**refuses** (97.5%), so subtracting refusal has a closed gate to open (compliance 2.5%→62.5%). The multi-turn reply sets baseline-**comply** (refusal ~3–9%) because the attacker's accumulated context already opened the gate before the victim's breach turn — so there is nothing left to open, and harm sits at a high starting point (2.75/3.66). This is why compliance steering "works" only on the single-turn set, and why the harm-steering nulls were originally measured on already-open replies.

## B. Is "40 baseline-refusing" a description or a selection?

**A description, not a selection.** The 40 are the positional slice `goals[50:90]` (`steer_dir_on_harmful.py:62`, `n_skip=50/n_eval=40`) — **no predicate on refusal, compliance, or harm, and no random seed**. They *happen* to baseline-refuse because they are JailbreakBench harmful goals: 97.5% (39/40) refuse at α=0. So "40 baseline-refusing JBB prompts" is accurate as a *description of the slice's behavior*, but the slice was **not chosen for refusing**. There is no discarded-compliant fraction to report because there was no refusal filter — the other 60 goals were simply outside the `[50:90]` window.

## C. Grid consistency for the matched figure

**Not identical — the confound the earlier verifier flagged is still present in the shipped figure's data.**
- Compliance direction (`refusal_harm_vs_compliance_v1`): **9 α** = `[-1.5, -1.0, -0.5, -0.25, 0, 0.25, 0.5, 1.0, 1.5]`.
- Harm direction (`steer_cvh_matched`, `harm_dm_llama`): **6 α** = `[-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]` — **missing ±0.25 and ±0.75**.
- `experiments/steer_cvh_fill/` exists but is **empty (0 files)** — the grid-fill run has not populated it, so `plot_cvh_matched.py`'s `load()` returns `[]` and the current `fig_cvh_matched.png` uses the coarse 6-point harm grid.
- The harm-direction peak (~**1.575**, Llama) sits at **α=+0.5**, at the edge of a coarse gap (nothing sampled at +0.25 or +0.75). The fill run (α = −0.25, +0.25, +0.75) is designed to close exactly this; until it lands and `plot_cvh_matched.py` is re-run, the grids are unequal.

---

## Discrepancies / ship-blockers to fix

1. **`fig_cvh_matched` suptitle numbers.** It says "**2.5% compliant / harm 1.07**". Fresh compute: harm baseline is **1.075** (not 1.07), and 2.5% compliant is **Llama-only** — Qwen baseline compliance is **5.0%**. Not wrong, but under-specified; state both judges.
2. **"3 harmless" wording (Stage-B corpus).** `select_stage_b.py` picks 3 `wins` (unsafe) + 3 `losses` (**not-unsafe = refused/failed attempts on the *same harmful* goal**). Any paper text describing the corpus as "3 jailbreaking + 3 harmless" risks reading as "3 benign prompts" — the losses are refused *harmful* conversations, not benign tasks. Clarify.
3. **Grid confound (C)** — unequal α grids in the shipped matched figure until the fill lands.

## Could NOT be verified from the repo

- **`working/uplift/goals.json` provenance/order.** The file is **untracked in git** (`git ls-files` errors) and **no script writes it** (no `json.dump` to it anywhere in `scripts/`, `archive/`, root). `select_stage_b.py:11` *comments* it as "canonical 100-behavior list", but there is no JailbreakBench reference file in the repo to confirm the order. Therefore whether `goals[50:90]` corresponds to "canonical JBB behaviors 50–89" is **an assumption, not verifiable**. (The slice itself is deterministic and reproducible from the on-disk file.)
- **`stage_b_candidates.jsonl`, `stage_b_scores*.jsonl`, `goals.json` are all untracked in git** — no commit history to audit their generation timestamps/versions.
- **Upstream self-play conversation data** (`experiments/{11 sources}/rounds/round_*.jsonl`) — the DPO self-play that produced these 9,400 conversations was not re-traced in this pass (out of scope; treated as the given raw input to `select_stage_b.py` and `launch_pooled_sweep.sh`).
*(now verified this pass — moved out of "could not verify":)* **Win/loss composition of the 600** is **289 wins / 311 losses**; **95/100 behaviors are exactly 3+3**, the other 5 backfilled losses when wins were scarce: (2w,4l)×1, (1w,5l)×2, (0w,6l)×2 — exactly the `deficit`-backfill branch (`select_stage_b.py:69-73`). For every behavior the loss-set goals are **identical to the win-set goals** (same harmful JBB goal, `unsafe=False`) — confirming the "3 losses" are refused harmful attempts, not benign prompts.
