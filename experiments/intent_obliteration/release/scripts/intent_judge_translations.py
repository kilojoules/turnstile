"""Strict-dual-judge gating step for the paired-obliteration pipeline (local 70B).

Takes the ``replay.pt`` produced by ``intent_replay.py`` and runs the strict
JBB dual judge (Llama-Guard local + local 70B, no silent fallback) on each
translated conversation. Drops any record where the dual judge agrees the
translation is unsafe.

If the local 70B emits ambiguous output across all retries (or hits a CUDA
error), ``LocalJudgeFailure`` is raised and the script exits non-zero.
This is intentional: silently treating an unjudgeable translation as "safe"
would let undetected harmful content into the benign side of the paired
probe. The operator must investigate.

Output: ``replay_judged.pt`` -- the same record list as ``replay.pt`` but
filtered to only translations the strict dual judge agreed are safe, with an
added ``translation_judge`` field per record.

Usage (Vast.ai GPU instance, ~48 GB VRAM):
  python -m turnstile.intent_judge_translations \
      --replay-pt experiments/intent_obliteration_paired/replay.pt \
      --output experiments/intent_obliteration_paired/replay_judged.pt \
      --judge-model meta-llama/Llama-3.1-70B-Instruct \
      --guard-model meta-llama/Llama-Guard-3-1B
"""

import argparse
import os

import torch

from turnstile.model_utils import load_model, unload_model
from turnstile.strict_dual_judge import (
    StrictDualJudge, LocalJudgeFailure,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replay-pt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--judge-model", default="meta-llama/Llama-3.1-70B-Instruct")
    p.add_argument("--guard-model", default="meta-llama/Llama-Guard-3-1B")
    p.add_argument("--judge-max-retries", type=int, default=3)
    args = p.parse_args()

    print(f"Loading replay records from {args.replay_pt}...")
    records = torch.load(args.replay_pt, weights_only=False)
    print(f"Loaded {len(records)} records")

    print(f"Loading 70B judge: {args.judge_model}")
    judge_model, judge_tok = load_model(args.judge_model)
    print(f"Loading Llama-Guard: {args.guard_model}")
    guard_model, guard_tok = load_model(args.guard_model)

    judge = StrictDualJudge(
        judge_model=judge_model, judge_tokenizer=judge_tok,
        guard_model=guard_model, guard_tokenizer=guard_tok,
        judge_max_retries=args.judge_max_retries,
    )

    kept = []
    n_dropped_unsafe = 0
    n_dropped_judge_failure = 0
    try:
        for i, rec in enumerate(records):
            translated = rec.get("translated_conversation")
            goal = rec.get("goal", "")
            if not translated:
                rec["translation_judge"] = {
                    "skipped": True,
                    "reason": "no_translated_conversation",
                }
                kept.append(rec)
                continue
            try:
                verdict = judge.judge_conversation(translated, goal)
            except LocalJudgeFailure as e:
                # Loud drop -- never silently pass.
                print(f"  [drop:judge_failure] {rec.get('source','?')}: {e}")
                rec["translation_judge"] = {
                    "skipped": False, "judge_failed": True, "error": str(e),
                }
                n_dropped_judge_failure += 1
                continue

            rec["translation_judge"] = {
                "skipped": False,
                "judge_failed": False,
                "unsafe": verdict.unsafe,
                "guard_unsafe": verdict.guard_unsafe,
                "together_unsafe": verdict.together_unsafe,
                "raw_70b_output": verdict.raw_70b_output,
                "reason": verdict.reason,
            }
            if verdict.unsafe:
                n_dropped_unsafe += 1
                print(f"  [drop:translation_unsafe] {rec.get('source','?')}: "
                      f"reason={verdict.reason}")
                continue
            kept.append(rec)
            if (i + 1) % 25 == 0:
                print(f"  [progress] {i+1}/{len(records)} judged, "
                      f"kept={len(kept)}, "
                      f"dropped_unsafe={n_dropped_unsafe}, "
                      f"dropped_judge_failure={n_dropped_judge_failure}")
    finally:
        unload_model(judge_model, judge_tok)
        unload_model(guard_model, guard_tok)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(kept, args.output)
    print(f"\nFinal: kept {len(kept)} records "
          f"(dropped {n_dropped_unsafe} as unsafe, "
          f"{n_dropped_judge_failure} as judge-failure).")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
