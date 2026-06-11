"""Build inputs.json for the single-prompt steering experiment.

Two sources, paired into a probe-fitting + α-sweep dataset:

  - 100 *harmful* prompts: the breach user turns from 100 records of
    ``replay_v2_full.pt`` (deterministic stratified sample by goal class
    + turn_of_breach), used as standalone single-turn requests.
  - 100 *benign* prompts: deterministic random sample from AlpacaEval-v1
    (805 instructions). Seeded so re-running the prep produces the same
    inputs.

The resulting JSON is consumed by ``turnstile.single_prompt_steering``.
"""

import argparse
import hashlib
import json
import os
import random

import torch


def goal_hash(goal):
    return hashlib.sha1(goal.encode("utf-8")).hexdigest()[:10]


def extract_breach_turns(replay_path, n_target):
    records = torch.load(replay_path, weights_only=False)
    out = []
    for i, r in enumerate(records):
        oc = r.get("original_conversation") or []
        t_star = r.get("turn_of_breach")
        if t_star is None or 2 * t_star >= len(oc):
            continue
        msg = oc[2 * t_star]
        if msg.get("role") != "user":
            continue
        out.append({
            "prompt_id": f"harmful_{i:03d}",
            "label": "harmful",
            "source": "v2_breach_turn",
            "goal": r["goal"],
            "category": r.get("category"),
            "turn_of_breach": int(t_star),
            "prompt": msg["content"],
            "v2_record_index": i,
        })
    if len(out) > n_target:
        # deterministic: stratify by category, take round-robin
        by_cat = {}
        for x in out:
            by_cat.setdefault(x["category"], []).append(x)
        rng = random.Random(13)
        for v in by_cat.values():
            rng.shuffle(v)
        sampled = []
        cats = list(by_cat.keys())
        i = 0
        while len(sampled) < n_target:
            c = cats[i % len(cats)]
            if by_cat[c]:
                sampled.append(by_cat[c].pop())
            i += 1
            if all(not v for v in by_cat.values()):
                break
        out = sampled[:n_target]
    return out


def sample_alpaca(alpaca_path, n_target):
    d = json.load(open(alpaca_path))
    rng = random.Random(13)
    rng.shuffle(d)
    out = []
    for i, x in enumerate(d[:n_target]):
        out.append({
            "prompt_id": f"benign_{i:03d}",
            "label": "benign",
            "source": f"alpaca_eval/{x['dataset']}",
            "goal": None,  # benign — judged-against goal is N/A
            "prompt": x["instruction"],
            "alpaca_dataset": x["dataset"],
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replay-pt", required=True)
    p.add_argument("--alpaca-json", required=True)
    p.add_argument("--n-harmful", type=int, default=100)
    p.add_argument("--n-benign", type=int, default=100)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    harmful = extract_breach_turns(args.replay_pt, args.n_harmful)
    benign = sample_alpaca(args.alpaca_json, args.n_benign)

    print(f"harmful: {len(harmful)} (target {args.n_harmful})")
    print(f"benign:  {len(benign)} (target {args.n_benign})")
    print(f"  category distribution (harmful): "
          + ", ".join(f"{c}={sum(1 for h in harmful if h['category']==c)}"
                      for c in sorted({h['category'] for h in harmful})))

    out = {
        "harmful": harmful,
        "benign": benign,
        "config": {
            "replay_pt_basename": os.path.basename(args.replay_pt),
            "alpaca_basename": os.path.basename(args.alpaca_json),
            "n_harmful": len(harmful),
            "n_benign": len(benign),
            "seed": 13,
        },
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
