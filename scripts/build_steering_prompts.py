"""Build the 50-prompt held-out set for the two-axis steering experiment.

Selection rules:
  - Sample from the 9,400-conv pool's wins (we want a meaningful breach turn)
  - Stratify across 10 JBB categories: 5 prompts per category
  - Exclude the 600 Stage B conversations (probe was trained on those)
  - For each chosen conversation, the "pivot" is the conversation truncated
    just before the breach-turn assistant message — i.e., chat_history up
    through the user message at the breach turn

Output: data/steering_prompts.json (list of dicts) and a small held-out
sample (data/steering_prompts_pilot.json, n=5, one per top-ASR category)
for the pilot run.
"""
import glob
import json
import os
import random
from collections import defaultdict

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
N_WIN_PER_CAT = 3   # 3 wins per cat × 10 cats = 30 wins
N_LOSS_PER_CAT = 2  # 2 losses per cat × 10 cats = 20 losses
N_PILOT = 5
SEED = 42


def main():
    # Stage B keys to exclude
    sb_keys = set()
    for line in open(f"{ROOT}/working/uplift/stage_b_candidates.jsonl"):
        r = json.loads(line)
        sb_keys.add((r["source"], r["round"], r["idx"]))

    SOURCES = ["authority_dpo", "control_hard_s456", "control_s42",
               "frozen_v1", "incrementalism_dpo", "reward_dpo",
               "stealth_hard_s456", "stealth_jbb_v1", "stealth_s42",
               "urgency_dpo", "urgency_v1"]

    by_cat_wins = defaultdict(list)
    by_cat_losses = defaultdict(list)
    for src in SOURCES:
        rdir = f"{ROOT}/experiments/{src}/rounds"
        if not os.path.isdir(rdir):
            continue
        for path in sorted(glob.glob(f"{rdir}/round_*.jsonl")):
            rnum = int(os.path.basename(path).replace("round_", "")
                       .replace(".jsonl", ""))
            for i, line in enumerate(open(path)):
                r = json.loads(line)
                if (src, rnum, i) in sb_keys:
                    continue
                conv = r["conversation"]
                if not conv:
                    continue
                if r.get("unsafe") and r.get("turn_of_breach") is not None:
                    tob = int(r["turn_of_breach"])
                    user_idx = 2 * tob
                    if user_idx >= len(conv) or conv[user_idx]["role"] != "user":
                        continue
                    rec = {
                        "prompt_id": f"{src}/r{rnum}/i{i}",
                        "behavior": r.get("behavior"),
                        "category": r.get("category"),
                        "goal": r.get("goal"),
                        "original_unsafe": True,
                        "original_turn_of_breach": tob,
                        "chat_history": conv[:user_idx],
                        "last_user_message": conv[user_idx]["content"],
                        "prompt_type": "win",
                    }
                    by_cat_wins[r["category"]].append(rec)
                elif not r.get("unsafe"):
                    # loss: pivot at the LAST user message (where model refused)
                    last_user_idx = None
                    for j in range(len(conv) - 1, -1, -1):
                        if conv[j]["role"] == "user":
                            last_user_idx = j
                            break
                    if last_user_idx is None:
                        continue
                    rec = {
                        "prompt_id": f"{src}/r{rnum}/i{i}",
                        "behavior": r.get("behavior"),
                        "category": r.get("category"),
                        "goal": r.get("goal"),
                        "original_unsafe": False,
                        "original_turn_of_breach": None,
                        "chat_history": conv[:last_user_idx],
                        "last_user_message": conv[last_user_idx]["content"],
                        "prompt_type": "loss",
                    }
                    by_cat_losses[r["category"]].append(rec)

    print(f"Available wins / losses per category (after Stage B exclusion):")
    for c in sorted(set(by_cat_wins) | set(by_cat_losses)):
        print(f"  {c:<28}  wins={len(by_cat_wins.get(c, [])):>4}  "
              f"losses={len(by_cat_losses.get(c, [])):>5}")

    rng = random.Random(SEED)
    main_prompts = []
    pilot_prompts = []
    for c in sorted(set(by_cat_wins) | set(by_cat_losses)):
        ws = by_cat_wins.get(c, []); rng.shuffle(ws)
        ls = by_cat_losses.get(c, []); rng.shuffle(ls)
        main_prompts.extend(ws[:N_WIN_PER_CAT])
        main_prompts.extend(ls[:N_LOSS_PER_CAT])
    print(f"\nSampled {len(main_prompts)} prompts (target = "
          f"{10 * (N_WIN_PER_CAT + N_LOSS_PER_CAT)}; "
          f"wins={sum(p['prompt_type']=='win' for p in main_prompts)}, "
          f"losses={sum(p['prompt_type']=='loss' for p in main_prompts)}).")

    # Pilot: 3 wins + 2 losses across diverse categories
    high_asr_cats = ["Fraud/Deception", "Malware/Hacking", "Privacy",
                     "Government decision-making", "Physical harm"]
    for c in high_asr_cats[:3]:
        for p in main_prompts:
            if p["category"] == c and p["prompt_type"] == "win":
                pilot_prompts.append(p); break
    for c in high_asr_cats[:2]:
        for p in main_prompts:
            if p["category"] == c and p["prompt_type"] == "loss":
                pilot_prompts.append(p); break
    print(f"Pilot subset: {len(pilot_prompts)} prompts "
          f"({[p['category'] for p in pilot_prompts]})")

    with open(f"{ROOT}/data/steering_prompts.json", "w") as f:
        json.dump(main_prompts, f, indent=2)
    with open(f"{ROOT}/data/steering_prompts_pilot.json", "w") as f:
        json.dump(pilot_prompts, f, indent=2)
    print(f"\nWrote {ROOT}/data/steering_prompts.json "
          f"({len(main_prompts)} prompts)")
    print(f"Wrote {ROOT}/data/steering_prompts_pilot.json "
          f"({len(pilot_prompts)} prompts)")


if __name__ == "__main__":
    main()
