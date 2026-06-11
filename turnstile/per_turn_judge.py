"""Per-turn strict-dual-judge labels on the paired-obliteration dataset.

Addresses the methodological gap where the upstream pipeline judged only:
  (a) the FIRST compliance turn (= breach turn t*) on the original arm, and
  (b) the LAST turn of the translated conversation in the strict-judge filter.

Here we score every assistant turn in BOTH arms with the strict dual judge
(Llama-Guard-3-1B + Llama-3.1-70B-Instruct, no silent fallback) against the
record's original goal, so we can characterise:

  * the per-turn ASR trajectory of the original arm
    (refuse refuse ... comply ...) -- confirms / refines the breach-turn label
  * the per-turn ASR trajectory of the translated/obliterated arm
    (does obliteration hold across all turns, or leak after t*?)

No new generation is performed; we only judge the assistant turns already
stored in ``replay_judged.pt``. The 8B victim is NOT loaded.

Usage (on a GPU box):
  python -m turnstile.per_turn_judge \\
      --replay-pt experiments/intent_obliteration_paired/replay_judged.pt \\
      --output experiments/per_turn_judge_v1/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone

import torch

from turnstile.strict_dual_judge import StrictDualJudge, LocalJudgeFailure


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay-pt", required=True,
                   help="path to replay_judged.pt")
    p.add_argument("--output", required=True,
                   help="JSONL output file (resumable)")
    args = p.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {args.replay_pt}...")
    records = torch.load(args.replay_pt, weights_only=False)
    print(f"  {len(records)} records loaded")

    seen = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                seen.add((r["source"], r["arm"], r["turn"]))
        print(f"  resuming: {len(seen)} (source,arm,turn) tuples already done")

    print("Loading strict dual judge "
          "(Llama-Guard-3-1B + Llama-3.1-70B-Instruct)...")
    judge = StrictDualJudge()

    # Plan total work for ETA
    n_planned = 0
    for rec in records:
        for arm in ("original", "translated"):
            conv = (rec["original_conversation"] if arm == "original"
                    else rec["translated_conversation"])
            n_assist = sum(1 for m in conv if m["role"] == "assistant")
            n_planned += n_assist
    print(f"  total (record,arm,turn) tuples: {n_planned}; "
          f"{len(seen)} already done; {n_planned - len(seen)} to do")

    t0 = time.time()
    n_new = 0
    n_fail = 0
    for rec in records:
        for arm in ("original", "translated"):
            conv = (rec["original_conversation"] if arm == "original"
                    else rec["translated_conversation"])
            n_assist = sum(1 for m in conv if m["role"] == "assistant")
            for turn in range(n_assist):
                key = (rec["source"], arm, turn)
                if key in seen:
                    continue
                prefix = conv[: 2 * turn + 2]
                if len(prefix) < 2 or prefix[-1]["role"] != "assistant":
                    print(f"  [skip:malformed] {rec['source']} arm={arm} "
                          f"turn={turn} (prefix len={len(prefix)})")
                    continue
                try:
                    verdict = judge.judge_conversation(prefix, goal=rec["goal"])
                    verdict_dict = asdict(verdict)
                    judge_failed = False
                except LocalJudgeFailure as e:
                    print(f"  [judge fail] {rec['source']} arm={arm} "
                          f"turn={turn}: {e}")
                    verdict_dict = {
                        "unsafe": None, "guard_unsafe": None,
                        "together_unsafe": None,
                        "raw_70b_output": str(e),
                        "reason": "judge_failed",
                    }
                    judge_failed = True
                    n_fail += 1
                row = {
                    "source": rec["source"],
                    "arm": arm,
                    "turn": turn,
                    "turn_of_breach": rec["turn_of_breach"],
                    "goal": rec["goal"],
                    "behavior": rec.get("behavior"),
                    "category": rec.get("category"),
                    "verdict": verdict_dict,
                    "judge_failed": judge_failed,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                with open(args.output, "a") as f:
                    f.write(json.dumps(row) + "\n")
                n_new += 1
                if n_new % 25 == 0:
                    elapsed = time.time() - t0
                    rate = n_new / max(elapsed, 1)
                    remaining = (n_planned - len(seen) - n_new)
                    eta_min = remaining / max(rate, 1e-3) / 60
                    print(f"  [{n_new}/{n_planned - len(seen)}]  "
                          f"{rate:.2f} judge/s  eta {eta_min:.1f} min  "
                          f"({n_fail} fail)")

    print(f"\nFinished. {n_new} new judgments, {n_fail} judge failures.")
    judge.unload()


if __name__ == "__main__":
    main()
