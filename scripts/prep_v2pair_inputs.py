"""Build inputs.json for the redesigned single-prompt steering experiment.

Paired contrast = within-corpus. For each of 200 multi-turn jailbreak
records in replay_v2_full.pt:
  - harmful breach turn = original_conversation[2*t*]['content']
  - benign-rewritten breach turn = translated_conversation[2*t*]['content']
The two share buildup; only the breach user-turn differs (length, register,
audit-style preamble matched by construction).

Stratified train/test split by category, 50/50 record-paired.

Steering target: held-out test-side harmful breach turns.
Direction-fitting: train-side paired (harmful, benign-rewritten) breach turns.
"""
import argparse
import json
import os
import random
from collections import defaultdict

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replay-pt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args()

    recs = torch.load(args.replay_pt, weights_only=False)
    rng = random.Random(args.seed)

    pairs = []
    for i, r in enumerate(recs):
        oc = r.get("original_conversation") or []
        tc = r.get("translated_conversation") or []
        t_star = r.get("turn_of_breach")
        if t_star is None or 2 * t_star >= len(oc) or 2 * t_star >= len(tc):
            continue
        oh = oc[2 * t_star]; tb = tc[2 * t_star]
        if oh.get("role") != "user" or tb.get("role") != "user":
            continue
        pairs.append({
            "record_idx": i,
            "category": r.get("category"),
            "goal": r.get("goal"),
            "turn_of_breach": int(t_star),
            "harmful_breach": oh["content"],
            "benign_breach": tb["content"],
        })

    # Stratified 50/50 split by category, within-pair
    by_cat = defaultdict(list)
    for p in pairs:
        by_cat[p["category"]].append(p)
    train, test = [], []
    for cat, items in by_cat.items():
        rng.shuffle(items)
        mid = len(items) // 2
        train.extend(items[:mid])
        test.extend(items[mid:])

    print(f"Total pairs: {len(pairs)}")
    print(f"Train: {len(train)}, Test: {len(test)}")
    print("Per-category counts (train / test):")
    for cat, items in sorted(by_cat.items()):
        n_tr = sum(1 for x in train if x["category"] == cat)
        n_te = sum(1 for x in test if x["category"] == cat)
        print(f"  {cat:>30s}: {n_tr:>3d} / {n_te:>3d}")

    # Length stats — needed for the length-median nuisance LR control
    train_breach_lens = [len(p["harmful_breach"]) for p in train] + [len(p["benign_breach"]) for p in train]
    median_len = sorted(train_breach_lens)[len(train_breach_lens)//2]
    print(f"\nMedian breach-turn length (train, both arms pooled): {median_len} chars")

    out = {
        "train": train,
        "test": test,
        "config": {
            "replay_pt_basename": os.path.basename(args.replay_pt),
            "n_train_pairs": len(train),
            "n_test_pairs": len(test),
            "seed": args.seed,
            "median_breach_length_chars": median_len,
        },
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
