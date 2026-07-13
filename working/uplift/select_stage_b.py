"""Stage B candidate selection: pick ~600 conversations to ship for response-
quality scoring on the same 80B judge.

Stratified by behavior (per-behavior win/loss split if possible) so that the
Stage-B "did the response actually deliver the uplift the goal could have
permitted?" question has per-behavior power matching Stage A.

Inputs:
  ../experiments/{source}/rounds/round_*.jsonl  for the 11 pooled sources
  ../experiments/outcome_probe_v1/per_behavior_L16_T1.json  (optional, for prioritisation)
  goals.json   (canonical 100-behavior list)

Outputs:
  stage_b_candidates.jsonl   one record per chosen conversation:
    {behavior, category, goal, source, round, idx, unsafe, conversation}
"""

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
SOURCES = [
    "control_s42", "stealth_s42", "control_hard_s456", "stealth_hard_s456",
    "authority_dpo", "incrementalism_dpo", "reward_dpo", "urgency_dpo",
    "frozen_v1", "urgency_v1", "stealth_jbb_v1",
]
EXPER = ROOT.parent.parent / "experiments"
GOALS = json.loads((ROOT / "goals.json").read_text())
TARGETS_PER_BEHAVIOR = 6  # split 3 wins + 3 losses; relax if shortfall
RNG = random.Random(0)


def load_all_convs():
    by_behavior = defaultdict(list)
    for src in SOURCES:
        for path in sorted((EXPER / src / "rounds").glob("round_*.jsonl")):
            round_idx = int(path.stem.split("_")[1])
            with path.open() as f:
                for i, line in enumerate(f):
                    d = json.loads(line)
                    by_behavior[d["behavior"]].append({
                        "behavior": d["behavior"],
                        "category": d["category"],
                        "goal": d["goal"],
                        "source": src,
                        "round": round_idx,
                        "idx": i,
                        "unsafe": bool(d.get("unsafe", False)),
                        "conversation": d["conversation"],
                    })
    return by_behavior


def main():
    by_behavior = load_all_convs()
    chosen = []
    shortfalls = []
    for g in GOALS:
        b = g["behavior"]
        pool = by_behavior.get(b, [])
        wins = [c for c in pool if c["unsafe"]]
        losses = [c for c in pool if not c["unsafe"]]
        n_each = TARGETS_PER_BEHAVIOR // 2
        picked_w = RNG.sample(wins, min(len(wins), n_each))
        picked_l = RNG.sample(losses, min(len(losses), n_each))
        # Backfill if one side is short
        deficit = TARGETS_PER_BEHAVIOR - len(picked_w) - len(picked_l)
        if deficit > 0 and len(losses) > len(picked_l):
            extra = RNG.sample([c for c in losses if c not in picked_l],
                               min(deficit, len(losses) - len(picked_l)))
            picked_l += extra
        chosen.extend(picked_w + picked_l)
        if len(picked_w) + len(picked_l) < TARGETS_PER_BEHAVIOR:
            shortfalls.append((b, len(picked_w), len(picked_l)))

    out = ROOT / "stage_b_candidates.jsonl"
    with out.open("w") as f:
        for c in chosen:
            f.write(json.dumps(c) + "\n")
    print(f"wrote {len(chosen)} candidates to {out}")
    print(f"behaviors with shortfalls: {len(shortfalls)}")
    for b, nw, nl in shortfalls[:10]:
        print(f"  {b[:40]}  wins={nw} losses={nl}")


if __name__ == "__main__":
    main()
