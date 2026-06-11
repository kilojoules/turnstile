"""Sample 200 Network hacking conversations, stratified for labeling.

Balances across: source (corpus/focused/pooled), judge label (unsafe/safe),
turn_of_breach (early/late/none).

Writes: experiments/judge_calibration/label_tasks.jsonl
"""
import json
import os
import random
from collections import defaultdict

random.seed(42)
N = 200
TARGET_BEHAVIOR = "Network hacking"
OUT = "experiments/judge_calibration/label_tasks.jsonl"


def load_all():
    rows = []
    # corpus
    with open("experiments/network_hacking_corpus/malicious/convs.jsonl") as f:
        for line in f:
            r = json.loads(line)
            r["_src"] = "corpus"
            rows.append(r)
    # focused
    fd = "experiments/nethack_focused/rounds"
    if os.path.isdir(fd):
        for fn in sorted(os.listdir(fd)):
            if not fn.endswith(".jsonl"):
                continue
            for line in open(f"{fd}/{fn}"):
                r = json.loads(line)
                r["_src"] = f"focused_{fn.replace('.jsonl','')}"
                rows.append(r)
    # pooled
    sources = ["authority_dpo", "control_hard_s456", "control_s42", "frozen_v1",
               "incrementalism_dpo", "reward_dpo", "stealth_hard_s456",
               "stealth_jbb_v1", "stealth_s42", "urgency_dpo", "urgency_v1"]
    for src in sources:
        rd = f"experiments/{src}/rounds"
        if not os.path.isdir(rd):
            continue
        for fn in os.listdir(rd):
            if not fn.endswith(".jsonl"):
                continue
            for line in open(f"{rd}/{fn}"):
                r = json.loads(line)
                if r.get("behavior") != TARGET_BEHAVIOR:
                    continue
                r["_src"] = f"pooled_{src}"
                rows.append(r)
    return rows


def main():
    rows = load_all()
    print(f"Total Network hacking convs: {len(rows)}")

    unsafe = [r for r in rows if r.get("unsafe")]
    safe = [r for r in rows if not r.get("unsafe")]
    print(f"  unsafe: {len(unsafe)}, safe: {len(safe)}")

    # Stratify: half unsafe, half safe
    n_unsafe = min(N // 2, len(unsafe))
    n_safe = N - n_unsafe

    random.shuffle(unsafe)
    random.shuffle(safe)

    picked = unsafe[:n_unsafe] + safe[:n_safe]
    random.shuffle(picked)

    # Build tasks: strip only what labeler needs
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tasks = []
    for i, r in enumerate(picked):
        task = {
            "id": i,
            "src": r.get("_src", "?"),
            "goal": r.get("goal", ""),
            "behavior": r.get("behavior", "?"),
            "category": r.get("category", "?"),
            "conversation": r.get("conversation", []),
            "judge_unsafe": bool(r.get("unsafe")),
            "guard_unsafe": r.get("guard_unsafe"),
            "together_unsafe": r.get("together_unsafe"),
            "turn_of_breach": r.get("turn_of_breach"),
        }
        tasks.append(task)

    with open(OUT, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")

    # Breakdown
    breakdown = defaultdict(lambda: defaultdict(int))
    for t in tasks:
        src_group = t["src"].split("_")[0]
        breakdown[src_group]["unsafe" if t["judge_unsafe"] else "safe"] += 1
    print(f"\nWrote {len(tasks)} tasks to {OUT}")
    print("\nBreakdown:")
    for src, d in sorted(breakdown.items()):
        print(f"  {src:<12s}: unsafe={d['unsafe']}  safe={d['safe']}")


if __name__ == "__main__":
    main()
