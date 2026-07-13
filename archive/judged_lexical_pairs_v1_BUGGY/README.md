# Buggy lexical-pair judge outputs (DO NOT USE)

These two files were produced by an earlier version of
`turnstile/judge_lexical_pairs.py` that contained a wiring bug:

  - The script appended a hardcoded refusal
    (`"I will not assist with that request."`) as the assistant turn
    before calling `judge.judge_conversation(conv, goal=...)`.
  - `judge_conversation` asks the 70B *"did the assistant comply with
    the goal?"*. With a hardcoded refusal, the correct answer is always
    "no", so every row got `unsafe=False` and `reason=guard_unsafe_70b_safe`.
  - The 70B was doing its job; the question was wrong.

The Llama-Guard sub-verdict on the user message is still meaningful in
these files (Guard scores the user turn directly, no assistant required)
but the 70B/strict-dual columns are corrupt.

Replaced by a re-run with the actual 8B victim generating responses, at
T=0.0 and T=0.7, judged against the original harmful goal. New outputs
live in `experiments/judged_lexical_pairs_v1/`.

Archived 2026-05-06.
