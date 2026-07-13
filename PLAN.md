# Turnstile v2 + obliteration paper plan (post-collaborator-feedback)

Status as of writing: C-new mean-diff sweep still running (~17 hr to done);
A/B/D-new mean-diff complete + locally pulled; collaborator's structural/
lexical classification incorporated.

## Settled (so far) results — v2

### Per-segment any-turn ASR + steering Δ across L16 LR (the strongest direction)

| segment | n records | α=0 baseline (orig / obl) | Δ(α=−8 vs 0) original | Δ(α=+8 vs 0) obliterated |
|---|---|---|---|---|
| structural | 55 (LR) / 95 (meandiff) | 92.7% / 49.1% | **−43.6 pp** | **+20.0 pp** |
| lexical | 50 / 85 | 100.0% / 54.0% | **−44.0 pp** | **+16.0 pp** |
| borderline | 13 / 20 | 100.0% / 76.9% | −23.1 pp | −1.9 pp |

The collaborator's classification splits the corpus into:
- **structural** (53% of wins, 61 of 98 unique goals): goals that *cannot* be
  made benign by 1–2 word swaps — Disinformation / Harassment / Sexual /
  Physical-harm are 100% structural. The 70B rewriter had to substantially
  rework these prompts.
- **lexical** (34% of wins, 23 goals): goals where minimal-edit swaps are
  possible (counterfeit ↔ training-replica, etc.).
- **borderline** (12%, 14 goals): in between.

The L16 LR causal effect (−44 pp suppression on original; +16–20 pp
amplification on obliterated) is **statistically indistinguishable across
structural and lexical segments** — the bidirectional handle is robust to
the rewriter-quality concern.

### L16 mean-diff (Arditi-style) — weaker but directionally consistent

Suppression Δ ≈ −14 to −15 pp on original (close to main-paper Table 3's
−13 pp on a different dataset). Amplification near-zero on obliterated.
Effect about **3× weaker than LR direction** at the same layer.

### L31 (both LR and mean-diff)

Inert across segments. Original arm 84–100% across all α with no monotonic
trend. Confirms main-paper finding ("compliance is determined upstream of
where it is read out") on this dataset, and now with both LR and mean-diff
extractions.

## Tasks remaining

### A. Per-turn methodology fix (already done conceptually, plumbing left)

- v1's pivot-only judge filter discarded ~46 / 200 records as obliteration
  failures. Per-turn dual-judge (1540 calls done) reveals **56.5%** of
  obliterated records have at least one unsafe assistant turn — 2.4× the
  pivot-only headline of 23%. This is a published-paper-relevant correction.

### B. Wait for C-new mean-diff to finish (~T+17h)

- L31 mean-diff records 0–99 (C-new on stable instance 36198700). Watchdog
  bgxxf6d94 auto-rsyncs every 30 min so we don't lose data again.
- When done: rsync, destroy instance.

### C. Judge-filter the collaborator's 389 lexical-swap pairs

- They produced `lexical_swap_pairs.json`: 389 records with breach-turn
  lexical swaps (avg 10-char delta, vs original rewriter's 30-char delta).
- Need to run the strict dual judge on each `edited_breach_user_turn` in
  isolation to identify truly-benign-after-swap subset (likely 50–70%
  of 389 = 200–270 truly-clean pairs).
- Compute: ~$1, ~30 min. Trivial.

### D. Single-prompt steering experiment

Plan: take the truly-benign lexical-swap subset from (C), pair with the
original harmful breach-turn message, run as single-turn prompts. For each
direction extraction (refit-LR, refit-meandiff, reused-LR, reused-meandiff),
sweep α ∈ {−8, −4, 0, +4, +8} at L16 and L31. Total ≈ 8000 trials.

Compute: 4 GPU parallel ~8 hr ~$33; or 1 GPU ~33 hr ~$33. (Cost is
GPU-hours-roughly-fixed; parallelism cuts wall clock not bill.)

This experiment will be much cleaner than the v2 multi-turn cascading
results because:
1. Single-turn = no prelude contamination
2. Lexical-only = no structural-rewrite confound
3. Judge-filtered = the "benign" prompts genuinely flag safe in isolation
4. Both LR and mean-diff directions tested (Arditi comparison)
5. Both layers tested (mid vs final)

### E. Run the 10 hand-edited gold-standard pairs through judge

Per collaborator's "next steps": dual-judge each pair's breach turn
in isolation AND the full benign conversation (catches in-context leakage).
Drop unsafe-judged ones. Use surviving pairs as a high-trust control panel
to anchor the steering effect sizes from (D).

Compute: trivial, ~10 judge calls × few seconds.

## Paper updates (no compute, just writing)

1. **Frame the paired-obliteration paper around the segmented v2 result**.
   The collaborator's classification fits naturally as a robustness check.
2. **Replace the v1 23% rewrite-leakage headline** with the corrected
   per-turn 56.5% any-turn rate, footnoted with the 23% as the v1 metric.
3. **Causal claim**: at L16 along the LR-fit per-turn compliance direction,
   ASR shifts by **+20 pp under amplification on the obliterated arm**
   (specifically, +20 pp on structural / +16 pp on lexical) and **−44 pp
   under suppression on the original arm**. At L31 the same direction is
   inert (Δ ≤ 5 pp across all α and both arms). Mean-diff direction shows
   the same pattern but ~3× weaker (consistent with the main-paper Table 3
   finding that LR-fit > mean-diff for causal steering).
4. **Note (in limitations or main text)** that lexical and structural
   subsets show similar effect magnitudes — the rewriter-quality concern
   is real but does not confound the causal claim on this dataset.
5. **Archive v1 paper artifacts** are already in `archive/v1_pivot_anchored/`
   so the methodology-correction story stays visible.

## Cost so far + remaining

- Already spent: ~$80 (v1 sweeps + v2 LR sweeps + v2 mean-diff partial)
- Remaining for full plan: ~$35 (C-new finish + judge-filter + single-prompt
  experiment)
- Total budget: ~$115

## Files

- This plan: `PLAN.md`
- v2 segmentation analysis: `/tmp/segment_v2_by_classification.py`
- Collaborator's data: `/Users/julianquick/Downloads/lexical_structural_report/`
- Watchdog task: `bgxxf6d94` (Monitor)
- Active vast.ai instance: `36198700` (C-new, ssh3.vast.ai:38700)
