# v1 (pivot-anchored) — superseded

This directory archives the first version of the paired-obliteration analysis,
which anchored every measurement on a single "breach turn" t\* per record.
That assumption proved unsound. The corrected v2 analysis (per-turn) lives in
`experiments/clamp_obliterated_v2_*/` and `paper/obliteration.tex`. **These
v1 artifacts are kept on purpose so the methodology-correction story stays
visible in the paper, not memory-holed.**

## What v1 measured

- **Source dataset**: `experiments/intent_obliteration_paired/replay_judged.pt`
  (154 records, filtered to "end-to-end translated conversation last-turn
  dual-judge safe").
- **Probes**: compliance probe trained on hidden states with labels
  `t < t*` ⇒ refuse, `t == t*` ⇒ comply (pooled across records). Reported
  AUC 0.751 pooled, 0.836 matched.
- **Causal experiment**: clamp_obliterated_v1 (L16) and clamp_obliterated_l31_v1
  (L31) — at each record's labeled t\*, the user message was held fixed,
  steering hook was registered on the chosen layer, the assistant response at
  t\* was generated under intervention, and the response was scored with the
  strict dual judge. 7-α sweep on both arms (original + obliterated).

## v1 headlines

| layer | Δ(α=−8 vs 0), original arm | Δ(α=+8 vs 0), obliterated arm |
|-------|---------------------------:|------------------------------:|
| L16   | −66.3 pp                   | +4.5 pp                       |
| L31   | −3.9 pp                    | +1.3 pp                       |

These numbers are accurate **as descriptions of intervention at the
labeled-t\* turn**. They are not accurate descriptions of compliance gating
in the conversation as a whole — see below.

## What broke

We per-turn-judged every assistant turn of both arms with the same strict
dual judge. Results in `experiments/per_turn_judge_v1/`:

1. **Noisy t\* labels.** The source-corpus tob (= "first turn both judges
   agree unsafe") is wrong for ~10 % of records. Re-judging finds
   24 % ASR at relative turn t\*−1, indicating the real breach was earlier.
   The v1 causal experiment intervened *after* the breach in those cases.

2. **Post-pivot leakage missed entirely.** Pivot-only judging on the
   translated arm gives ASR 18.8 %. Per-turn judging finds **43.5 %** of
   records have at least one dual-judge-unsafe assistant turn. 14.3 % of
   records are unsafe *only* post-pivot — completely invisible to the
   pivot-only metric.

3. **Strict-judge filter targeted the wrong turn.** v1 filtered the dataset
   to records whose end-to-end last assistant turn was dual-safe. By the
   per-turn definition of "obliteration succeeded" (no unsafe assistant
   turn anywhere), only 87/154 v1-passing records actually qualify.

## What this means for the v1 numbers

The −66.3 pp drop on the original arm under L16 suppression at t\* is real
but narrow: it says "if we steer at the labeled-t\* turn, the assistant's
response at that turn refuses ~70 pp more often." It does not say
"L16 suppression keeps the conversation safe across all turns" — that
question is answered by v2. Likewise the L31 inertness conclusion holds
*at the labeled t\**; v2 tests it across the full conversation.

## Files in this directory

- `clamp_obliterated_v1/`    — L16 t\*-only sweep results (2156 rows, 5-fold)
- `clamp_obliterated_l31_v1/` — L31 t\*-only sweep results (2156 rows, 5-fold)
- `clamp_obliterated_v1_instance.txt` — vast.ai instance label used at run time
- `obliteration_v1.tex`      — paper source as of the v1 submission attempt
- `obliteration_v1.pdf`      — compiled v1 PDF

## What v2 changes

- Per-turn dual-judge labels for all 200 source records (not just 154).
- New filter: "all assistant turns of translated arm dual-judge safe."
- Probes refit per-turn with no t\* anchor.
- Causal experiment: steering hook active *throughout the conversation*,
  every assistant turn judged. Headline metric is per-turn ASR trajectory
  under intervention, not pivot-turn ASR.
