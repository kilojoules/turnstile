"""Assemble the intent-obliteration release dataset.

Combines adversarial (dangerous-intent) conversations from the 11 pooled
experiments with benign (network-security-curious) conversations into a single
JSONL file with per-turn judge annotations where available.

Per-turn judge schema:
  - For unsafe winners: derived from `turn_of_breach` (the first prefix the
    dual judge agreed was unsafe). Turns t < t* carry "at least one judge
    rated this prefix safe"; turn t* carries "both judges rated unsafe"; turns
    t > t* are not explicitly judged per-turn (the loop returned).
  - For losers: the full conversation was rated safe by at least one judge;
    no per-turn agreement was computed.
  - For benign-curious: marked safe by construction (no harmful goal).

Usage:
  python experiments/intent_obliteration/build_dataset.py \
      --output experiments/intent_obliteration/dataset/data.jsonl
"""

import argparse
import json
import os
import uuid


EXPERIMENTS = [
    "authority_dpo",
    "control_hard_s456",
    "control_s42",
    "frozen_v1",
    "incrementalism_dpo",
    "reward_dpo",
    "stealth_hard_s456",
    "stealth_jbb_v1",
    "stealth_s42",
    "urgency_dpo",
    "urgency_v1",
]
NETHACK_DIR = "experiments/network_hacking_corpus"


def per_turn_judge_from_tob(unsafe: bool, tob, num_turns: int):
    """Derive per-turn judge annotations from the overall verdict + turn_of_breach.

    Returns list of {turn, status, basis} for turns 0..num_turns-1.

    status values:
      "safe_at_least_one_judge"  - one of the two judges said safe at this prefix
      "unsafe_dual_judge"        - both judges said unsafe at this prefix
      "not_judged_per_turn"      - per-turn judging did not run for this turn
    """
    out = []
    for t in range(num_turns):
        if unsafe and tob is not None:
            if t < tob:
                status = "safe_at_least_one_judge"
            elif t == tob:
                status = "unsafe_dual_judge"
            else:
                status = "not_judged_per_turn"
        elif not unsafe:
            # Losers: conversation overall is safe, per-turn agreement not computed.
            status = "not_judged_per_turn_loser"
        else:
            # unsafe but no tob (rare; the find_turn_of_breach returned None)
            status = "not_judged_per_turn"
        out.append({"turn": t, "status": status})
    return out


def load_adversarial(experiments_root: str = "experiments"):
    records = []
    for exp in EXPERIMENTS:
        rounds_dir = os.path.join(experiments_root, exp, "rounds")
        if not os.path.isdir(rounds_dir):
            print(f"  [skip] {exp}: no rounds/")
            continue
        round_files = sorted(
            f for f in os.listdir(rounds_dir)
            if f.startswith("round_") and f.endswith(".jsonl")
        )
        n = 0
        for rf in round_files:
            round_num = int(rf.replace("round_", "").replace(".jsonl", ""))
            with open(os.path.join(rounds_dir, rf)) as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    conv = rec["conversation"]
                    num_turns = len([m for m in conv if m["role"] == "user"])
                    out = {
                        "id": str(uuid.uuid4()),
                        "intent": "dangerous",
                        "source": f"{exp}/round_{round_num}",
                        "experiment": exp,
                        "round": round_num,
                        "category": rec.get("category"),
                        "behavior": rec.get("behavior"),
                        "goal": rec.get("goal"),
                        "unsafe": bool(rec.get("unsafe", False)),
                        "turn_of_breach": rec.get("turn_of_breach"),
                        "judge_disagreement": rec.get("disagreement"),
                        "guard_unsafe": rec.get("guard_unsafe"),
                        "together_unsafe": rec.get("together_unsafe"),
                        "api_failed": rec.get("api_failed"),
                        "num_turns": num_turns,
                        "conversation": conv,
                        "per_turn_judge": per_turn_judge_from_tob(
                            unsafe=bool(rec.get("unsafe", False)),
                            tob=rec.get("turn_of_breach"),
                            num_turns=num_turns,
                        ),
                    }
                    records.append(out)
                    n += 1
        print(f"  [load] {exp}: {n} conversations")
    return records


def load_benign(experiments_root: str = "experiments"):
    records = []
    path = os.path.join(experiments_root, "network_hacking_corpus",
                         "benign_curious", "convs.jsonl")
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            conv = rec["conversation"]
            num_turns = len([m for m in conv if m["role"] == "user"])
            out = {
                "id": str(uuid.uuid4()),
                "intent": "benign",
                "source": "network_hacking_corpus/benign_curious",
                "category": "benign_curious_network_security",
                "seed": rec.get("seed"),
                "unsafe": False,
                "turn_of_breach": None,
                "num_turns": num_turns,
                "conversation": conv,
                "per_turn_judge": [
                    {"turn": t, "status": "safe_by_construction"}
                    for t in range(num_turns)
                ],
            }
            records.append(out)
    print(f"  [load] benign_curious: {len(records)} conversations")
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiments-root", default="experiments")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    print("Loading adversarial conversations...")
    adv = load_adversarial(args.experiments_root)
    print("\nLoading benign-curious conversations...")
    ben = load_benign(args.experiments_root)

    all_records = adv + ben
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    n_dangerous = sum(1 for r in all_records if r["intent"] == "dangerous")
    n_benign = sum(1 for r in all_records if r["intent"] == "benign")
    n_unsafe_wins = sum(1 for r in all_records if r["unsafe"])
    n_with_tob = sum(1 for r in all_records if r.get("turn_of_breach") is not None)
    print(f"\nWrote {len(all_records)} records to {args.output}")
    print(f"  dangerous: {n_dangerous}")
    print(f"  benign:    {n_benign}")
    print(f"  unsafe (verified jailbreaks): {n_unsafe_wins}")
    print(f"  with turn_of_breach:          {n_with_tob}")


if __name__ == "__main__":
    main()
