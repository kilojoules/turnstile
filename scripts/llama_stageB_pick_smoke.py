"""Phase 1.1 sampler: pick 20 Stage-B candidates for the Llama-70B smoke test.

Composition (deterministic, seed=42):
  6 with Qwen Likert = 5 (most-extreme positives — the load-bearing class)
  6 with Qwen Likert = 4 (boundary-positives)
  3 with Qwen Likert = 3
  3 with Qwen Likert = 2
  2 with Qwen Likert = 1

The 12 high-harm rows let us detect refusal *concentration* on the Likert-5 vs
Likert-4 sub-distribution (per Phase 1.1 instructions): if Llama refuses 5/6
Likert-5 items but 1/6 Likert-4, the threshold-≥4 high-harm class shrinks
non-randomly and the Phase 2 re-fit will be biased.

Writes the joined (candidate row + Qwen rating) records to:
  working/uplift/stage_b_smoke_input.jsonl

Each record carries `conversation`, `goal`, `unsafe`, `turn_of_breach` (used to
rebuild the exact Stage-B prompt) plus `qwen_rating` for the comparison.
"""
import json
import random
import os

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
CANDS = f"{ROOT}/working/uplift/stage_b_candidates.jsonl"
SCORES = f"{ROOT}/working/uplift/stage_b_scores.jsonl"
OUT = f"{ROOT}/working/uplift/stage_b_smoke_input.jsonl"

TARGET_PER_RATING = {5: 6, 4: 6, 3: 3, 2: 3, 1: 2}
SEED = 42


def conv_id(rec):
    return f"{rec['source']}/r{rec['round']}/i{rec['idx']}"


def main():
    cands_by_id = {}
    for line in open(CANDS):
        r = json.loads(line)
        cands_by_id[conv_id(r)] = r
    print(f"loaded {len(cands_by_id)} candidates")

    # Index Qwen scores by conv_id (only parse_ok rows)
    scores_by_id = {}
    for line in open(SCORES):
        r = json.loads(line)
        if not r.get("parse_ok"):
            continue
        scores_by_id[r["conv_id"]] = r["rating"]
    print(f"loaded {len(scores_by_id)} parse_ok Qwen scores")

    # Bucket joined rows by Qwen rating
    buckets = {1: [], 2: [], 3: [], 4: [], 5: []}
    for cid, qwen_rating in scores_by_id.items():
        if cid not in cands_by_id:
            continue
        if qwen_rating not in buckets:
            continue
        buckets[qwen_rating].append(cid)

    print("\nbucket sizes (parse_ok candidates with Qwen rating):")
    for k in sorted(buckets):
        print(f"  L={k}: {len(buckets[k])}")

    rng = random.Random(SEED)
    picked = []
    for rating, n_want in TARGET_PER_RATING.items():
        pool = buckets[rating]
        if len(pool) < n_want:
            print(f"WARN: only {len(pool)} candidates at L={rating}, want {n_want}")
            n_want = len(pool)
        chosen = rng.sample(pool, n_want)
        for cid in chosen:
            rec = dict(cands_by_id[cid])
            rec["qwen_rating"] = rating
            rec["conv_id"] = cid
            picked.append(rec)

    print(f"\npicked {len(picked)} rows; composition:")
    from collections import Counter
    c = Counter(r["qwen_rating"] for r in picked)
    for k in sorted(c):
        print(f"  L={k}: {c[k]}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for r in picked:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
